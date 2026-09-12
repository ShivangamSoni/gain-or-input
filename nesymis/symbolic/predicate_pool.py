"""Role-free predicate pool (T1 + T2): predicates with no hand-designed role slots.

T1  lexical predicates    says:"phrase"  -- exact (normalized) substring presence of
                          a corpus-mined discriminative n-gram in the meme text; plus
    semantic twins        sem:"phrase"   -- CLIP cosine of the caption embedding to
                          the same phrase (robust to the corpus's typos/paraphrase)
T2  exemplar predicates   ex:"caption.." -- cosine to an unsupervised k-means centroid
                          of train+val caption embeddings, named by its medoid caption
                          (captures full stereotype PROPOSITIONS, not just topics)

Contract (anti-degeneracy): predicate CONTENT comes from unsupervised structure or
mined surface forms only; class labels are never used to train predicate internals --
selection happens later, in rule induction. Every predicate carries a human-readable
name. Provenance -> artifacts/predicate_pool.json.

Run:  python -m nesymis.symbolic.predicate_pool   (build + summary)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from nesymis import config
from nesymis.encoders.clip_encoder import encode_texts
from nesymis.symbolic.predicate_gen import mine_class_phrases

POOL_PATH = config.ARTIFACTS_DIR / "predicate_pool.json"

LEX_PER_CLASS = 15     # mined phrases per class (T1)
N_EXEMPLARS = 40       # caption clusters (T2)
MIN_CLUSTER = 15       # drop tiny clusters
NAME_CHARS = 45        # exemplar display-name length


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(s).lower())).strip()


def build_pool(emb, classes: list[str] | None = None, calib_mask: np.ndarray | None = None,
               save: bool = True, text_source: str = "caption", return_spec: bool = False):
    """Returns (continuous score frame [sem:/ex: columns], boolean lexical matrix,
    lexical names [says:]); with return_spec=True additionally (spec_vecs, phrases)
    -- the persistable spec {sem:/ex: name -> unit CLIP vector} + phrase list, so
    deployed inference never re-mines or re-clusters. text_source selects
    which text field is mined and grounded ('caption' or 'ocr')."""
    classes = classes or config.STEREO_CLASSES
    df = emb.df
    if calib_mask is None:
        calib_mask = df["split"].isin(["train", "val"]).to_numpy()

    text_col = "text_caption" if text_source == "caption" else "text_ocr"
    texts = df[text_col].fillna("").astype(str).tolist()
    cal_texts = [t for t, m in zip(texts, calib_mask) if m]
    cal_labels = df.loc[calib_mask, "label"].to_numpy()

    # ---- T1: mined phrases -> lexical (exact) + semantic (CLIP) predicates ----
    phrases: list[str] = []
    for c in classes:
        for p in mine_class_phrases(cal_texts, cal_labels == c, k=LEX_PER_CLASS):
            if p not in phrases:
                phrases.append(p)

    norm_texts = [f" {_norm(t)} " for t in texts]
    lex_names = [f'says:"{p}"' for p in phrases]
    lex = np.stack([np.array([f" {p} " in t for t in norm_texts]) for p in phrases], 1)

    U = emb.text_caption if text_source == "caption" else emb.text_ocr

    cont, spec_vecs = {}, {}
    ph_emb = encode_texts(phrases)                     # (P, 512) unit
    sem = U @ ph_emb.T                                 # (N, P) cosines
    for j, p in enumerate(phrases):
        cont[f'sem:"{p}"'] = sem[:, j]
        spec_vecs[f'sem:"{p}"'] = ph_emb[j].astype(np.float32)

    # ---- T2: unsupervised caption-exemplar predicates ----
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_EXEMPLARS, random_state=config.SEED, n_init=4).fit(U[calib_mask])
    sizes = np.bincount(km.labels_, minlength=N_EXEMPLARS)
    cal_idx = np.where(calib_mask)[0]
    exemplars = {}
    for k in range(N_EXEMPLARS):
        if sizes[k] < MIN_CLUSTER:
            continue
        cen = km.cluster_centers_[k]
        cen = cen / (np.linalg.norm(cen) + 1e-8)
        scores = U @ cen                               # (N,)
        medoid_i = cal_idx[np.argmax(scores[calib_mask] * (km.labels_ == k))]
        medoid = (str(df[text_col].iloc[medoid_i])[:NAME_CHARS]
                  .encode("ascii", "replace").decode())   # console/log-safe name
        name = f'ex:"{medoid}"'
        cont[name] = scores
        spec_vecs[name] = cen.astype(np.float32)
        exemplars[name] = str(df[text_col].iloc[medoid_i])

    if save:
        POOL_PATH.write_text(json.dumps(
            {"lexical": phrases, "exemplars": exemplars}, indent=1), encoding="utf-8")
    frame = pd.DataFrame(cont, index=df.index)
    if return_spec:
        return frame, lex, lex_names, spec_vecs, phrases
    return frame, lex, lex_names


def pool_features_from_text(texts: list[str], text_emb: np.ndarray,
                            spec_vecs: dict, phrases: list[str],
                            index=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pool features for arbitrary rows (deployed inference):
    says: substring bools from raw text; sem:/ex: cosines from the text embedding."""
    boolf = {f'says:"{p}"': np.array([f" {p} " in f" {_norm(t)} " for t in texts])
             for p in phrases}
    cont = {name: text_emb @ v for name, v in spec_vecs.items()}
    idx = range(len(texts)) if index is None else index
    return pd.DataFrame(cont, index=idx), pd.DataFrame(boolf, index=idx)


def main():
    from nesymis.data import dataset as ds

    emb = ds.load_embeddings()
    cont, lex, lex_names = build_pool(emb)
    calib = emb.df["split"].isin(["train", "val"]).to_numpy()
    print(f"[pool] {len(lex_names)} lexical + {sum(c.startswith('sem:') for c in cont.columns)} "
          f"semantic + {sum(c.startswith('ex:') for c in cont.columns)} exemplar predicates")
    print(f"[pool] lexical support (calib): mean="
          f"{lex[calib].mean():.3f}  max={lex[calib].mean(0).max():.3f}")
    ex_cols = [c for c in cont.columns if c.startswith("ex:")]
    print("[pool] sample exemplars:")
    for c in ex_cols[:8]:
        print(f"   {c}")


if __name__ == "__main__":
    main()
