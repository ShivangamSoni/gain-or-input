"""Stage 1 of dynamic rule induction: automatic predicate generation.

Given only (a) the class label names and (b) the train+val data, generate the
grounding predicate vocabulary for the symbolic path -- no hand-written prompts.
Two generators:

  * template prompts (image side): class-agnostic templates instantiated with the
    class name -> {c}_context / {c}_stereotype prototypes (+ universal woman_present)
  * corpus-mined phrases (text side): discriminative n-grams mined from the class's
    train+val captions via smoothed log-odds -> {c}_text_stereotype prototype

Prototypes are unit CLIP text embeddings (mean of normalized phrase embeddings),
drop-in compatible with grounding.compute_scores (same predicate names). Provenance
(every instantiated/mined phrase) goes to artifacts/auto_predicates.json for audit.

Adding a new stereotype class = new labeled data + its class name; rerunning this
module derives its predicates automatically. Removing a class removes them.

Run:  python -m nesymis.symbolic.predicate_gen   (generate + auto-vs-hand quality report)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nesymis import config
from nesymis.encoders.clip_encoder import encode_texts

AUTO_PROMPTS_PATH = config.ARTIFACTS_DIR / "auto_predicates.json"
AUTO_PROTO_PATH = config.ARTIFACTS_DIR / "auto_prototypes.npz"

# class-agnostic templates, instantiated with the raw class-label name
WOMAN_PROMPTS = ["a photo of a woman", "a woman", "an image of a woman"]
CONTEXT_TEMPLATES = ["a {c}", "a {c} scene", "a photo of a {c} setting"]
STEREO_TEMPLATES = ["a woman in a {c} setting", "a woman engaged in {c}",
                    "a meme about a woman and {c}"]

MINE_TOP_K = 12          # mined phrases per class (text-side predicate ensemble)
MINE_NGRAMS = (1, 3)
MINE_MIN_DF = 3


def mine_class_phrases(texts: list[str], is_class: np.ndarray, k: int = MINE_TOP_K,
                       alpha: float = 0.5) -> list[str]:
    """Top-k n-grams most associated with the class vs rest (smoothed log-odds on
    document presence). Data-derived only -- no hand/AI-written content."""
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=MINE_NGRAMS, min_df=MINE_MIN_DF,
                          stop_words="english", lowercase=True, binary=True)
    X = vec.fit_transform(texts)                       # (N, V) doc-presence
    terms = vec.get_feature_names_out()
    nc, nr = int(is_class.sum()), int((~is_class).sum())
    cc = np.asarray(X[is_class].sum(axis=0)).ravel()   # docs-with-term in class
    cr = np.asarray(X[~is_class].sum(axis=0)).ravel()  # docs-with-term in rest
    logodds = (np.log((cc + alpha) / (nc - cc + alpha))
               - np.log((cr + alpha) / (nr - cr + alpha)))
    order = np.argsort(-logodds)
    # prefer longer, more specific phrases among near-ties: stable sort by n-gram length
    top = sorted(order[: 3 * k], key=lambda i: (-logodds[i], -len(terms[i].split())))[:k]
    return [str(terms[i]) for i in top]


def generate_prompts(classes: list[str] | None = None, df: pd.DataFrame | None = None,
                     calib_mask: np.ndarray | None = None) -> dict[str, list[str]]:
    """Full auto-generated predicate vocabulary: name -> phrase ensemble."""
    classes = classes or config.STEREO_CLASSES
    if df is None:
        df = pd.read_csv(config.MANIFEST_PATH)
    if calib_mask is None:
        calib_mask = df["split"].isin(["train", "val"]).to_numpy()

    texts = df["text_caption"].fillna("").astype(str).tolist()
    cal_texts = [t for t, m in zip(texts, calib_mask) if m]
    cal_labels = df.loc[calib_mask, "label"].to_numpy()

    prompts: dict[str, list[str]] = {"woman_present": list(WOMAN_PROMPTS)}
    for c in classes:
        prompts[f"{c}_context"] = [t.format(c=c) for t in CONTEXT_TEMPLATES]
        prompts[f"{c}_stereotype"] = [t.format(c=c) for t in STEREO_TEMPLATES]
        prompts[f"{c}_text_stereotype"] = mine_class_phrases(cal_texts, cal_labels == c)
    return prompts


def build_prototypes(prompts: dict[str, list[str]], save: bool = True) -> dict[str, np.ndarray]:
    """Unit prototype per predicate (mean of normalized CLIP phrase embeddings)."""
    protos = {}
    for name, phrases in prompts.items():
        embs = encode_texts(phrases)
        mean = embs.mean(axis=0)
        protos[name] = (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)
    if save:
        AUTO_PROMPTS_PATH.write_text(json.dumps(prompts, indent=1), encoding="utf-8")
        np.savez(AUTO_PROTO_PATH, **protos)
    return protos


def load_auto_prototypes() -> dict[str, np.ndarray]:
    data = np.load(AUTO_PROTO_PATH)
    return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# Quality report: auto-generated vs hand-written predicates
# --------------------------------------------------------------------------- #
def _auc(score: np.ndarray, target: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(target.astype(int), score))


def quality_report() -> None:
    from nesymis.data import dataset as ds
    from nesymis.symbolic import grounding

    emb = ds.load_embeddings()
    te = emb.split_mask("test")
    y = emb.df["label"].astype(str).to_numpy()

    prompts = generate_prompts()
    auto = build_prototypes(prompts)
    hand = grounding.build_prototypes()
    s_auto = grounding.compute_scores(emb, auto)
    s_hand = grounding.compute_scores(emb, hand)

    print("=== Auto-generated predicates (provenance in artifacts/auto_predicates.json) ===")
    for c in config.STEREO_CLASSES:
        print(f"  {c}_text_stereotype mined phrases: {prompts[f'{c}_text_stereotype']}")

    print("\n=== Predicate quality: ROC-AUC of score for its target class vs rest (TEST) ===")
    print(f"  {'predicate':<30}{'hand':>8}{'auto':>8}")
    mis = np.isin(y, config.STEREO_CLASSES)
    print(f"  {'woman_present (vs non-mis.)':<30}"
          f"{_auc(s_hand['woman_present'][te], mis[te]):>8.3f}"
          f"{_auc(s_auto['woman_present'][te], mis[te]):>8.3f}")
    for c in config.STEREO_CLASSES:
        for fam in ("context", "stereotype", "text_stereotype"):
            name = f"{c}_{fam}"
            print(f"  {name:<30}{_auc(s_hand[name][te], (y == c)[te]):>8.3f}"
                  f"{_auc(s_auto[name][te], (y == c)[te]):>8.3f}")


if __name__ == "__main__":
    quality_report()
