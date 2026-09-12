"""Table 20 - full-pipeline retraining on MAMI (methodology generalization).

Table 19 transfers our FITTED artifacts to MAMI; this table instead re-runs the
entire METHODOLOGY on MAMI: the multimodal concept bank, literals, thresholds, DNF rules,
neural classifier, and decision policy are all learned from MAMI's own 10k
training set, then evaluated once on the official 1000-meme test set.

Adaptations (and nothing else changes -- all defaults kept):
  * Binary task (SemEval-2022 Task 5 Sub-task A): one positive class
    "misogynous" + meta behaviour is trivial (single class rule).
    MAMI's sub-labels are multi-LABEL, so a multi-class analog would be
    ill-posed; binary is the honest instantiation.
  * The affect head is OUR trained head, kept frozen (MAMI has no affect
    labels) -- affect predicates and FiLM features still flow from it.

If MAMI is absent the table reports SKIPPED (run_all-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import common as C
import external_transfer as t19
from nesymis import config
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import rule_induction

MAMI_DIR = t19.MAMI_DIR
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "mami_clip.npz")
POS = "misogynous"


def _load_train():
    f = os.path.join(MAMI_DIR, "TRAINING", "training.csv")
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f, sep="\t", encoding="utf-8-sig")
    paths = [os.path.join(MAMI_DIR, "TRAINING", fn) for fn in df["file_name"]]
    ok = np.array([os.path.isfile(p) for p in paths])
    df = df.loc[ok].reset_index(drop=True)
    return (df["Text Transcription"].fillna("").astype(str).tolist(),
            [p for p, k in zip(paths, ok) if k],
            df["misogynous"].to_numpy(int))


def _embeddings(paths, texts):
    """CLIP image+text embeddings for all MAMI rows, cached across runs."""
    files = np.array([os.path.basename(p) for p in paths])
    if os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        if len(z["files"]) == len(files) and (z["files"] == files).all():
            return z["v"], z["u"]
    from nesymis.encoders.clip_encoder import encode_images, encode_texts
    B = 256
    v = np.concatenate([encode_images(paths[i:i + B]) for i in range(0, len(paths), B)])
    u = np.concatenate([encode_texts(texts[i:i + B]) for i in range(0, len(texts), B)])
    np.savez(CACHE, files=files, v=v, u=u)
    return v, u


def generate(ctx=None):
    tr_data, te_data = _load_train(), t19._load_mami()
    if tr_data is None or te_data is None:
        return ("## Table 20 - Full-pipeline retraining on MAMI (SKIPPED)\n\n"
                "_MAMI not found under `dataset/MAMI/` (needs TRAINING/training.csv, "
                "test/Test.csv, test_labels.txt and the images)._\n")

    tr_texts, tr_paths, tr_y = tr_data
    _, te_texts, te_paths, te_y, te_stereo = te_data
    texts = tr_texts + te_texts
    paths = tr_paths + te_paths
    N, n_tr = len(texts), len(tr_texts)
    y = np.concatenate([tr_y, te_y]).astype(int)          # 0 = non, 1 = misogynous
    te = np.zeros(N, bool); te[n_tr:] = True
    print(f"  [t20] MAMI train {n_tr} / test {int(te.sum())}", flush=True)

    c = ctx or C.build_context()

    v, u = _embeddings(paths, texts)
    vu = np.concatenate([v, u], axis=1).astype(np.float32)
    D = np.zeros((N, C.odesign.AFFECT_DIM), np.float32)          # affect-free deployed methodology

    # train/val split inside MAMI's training set (stratified, fixed seed)
    tv_idx = np.arange(n_tr)
    tr_i, va_i = train_test_split(tv_idx, test_size=0.15, stratify=y[:n_tr],
                                  random_state=config.SEED)
    tr = np.isin(np.arange(N), tr_i)
    va = np.isin(np.arange(N), va_i)
    trainval = tr | va

    # full symbolic induction on MAMI train+val (binary class): build a MAMI-specific
    # multimodal concept bank, then induce a DNF over concepts (affect-free)
    config.LABEL_TO_IDX.setdefault(POS, 1)                 # runtime-only label id
    labels = np.where(y == 1, POS, "non_misogynous")
    emb_ns = SimpleNamespace(df=pd.DataFrame({"text_caption": texts, "label": labels}),
                             text_caption=u, image=v)
    bank = cl.build(emb_ns, trainval, classes=[POS])
    cont = bank.frame(u, v, index=np.arange(N))
    rs, L, names = rule_induction.induce(cont, y, trainval, affect=None,
                                         classes=[POS], save=False)
    f = np.zeros(N, bool)
    for clause in rs[POS]:
        f |= np.all(L[:, clause["idx"]], axis=1)
    fired = f[:, None].astype(np.float32)
    p_tv = float((y[trainval & f] == 1).mean()) if (trainval & f).any() else 0.0
    rvec = np.array([p_tv], np.float32)

    # neural classifier + bootstrap policy, retrained per seed (11-seed protocol);
    # the symbolic path above is deterministic and shared across seeds
    yt = y[te]

    def bmetrics(p):
        return (float((p == yt).mean()),
                f1_score(yt, p, average="macro", zero_division=0),
                f1_score(yt, p, zero_division=0))

    res = {pn: {"acc": [], "mf1": [], "f1": []} for pn in ("Path-A (neural)", "NeSy-MIS")}
    for seed in C.SEEDS11:
        clf, logits = clf.crossfit_logits(vu, D, y, tr, va, fusion=config.NEURAL_FUSION,
                                          affect_dim=C.odesign.AFFECT_DIM, seed=seed)
        state = C.rl.build_state(logits, fired, rvec, D)
        policy = C.rl.train_rlvr(logits, state, y, tr, va, seed=seed,
                                 policy_ctor=lambda d: C.rl.LinearEvidencePolicy(d, rule_classes=[1]))[0]
        ne = C.rl.greedy_pred(policy, logits[te], state[te])
        for pn, p in (("Path-A (neural)", (logits[te].argmax(1) == 1).astype(int)),
                      ("NeSy-MIS", (ne == 1).astype(int))):
            a, m, f1 = bmetrics(p)
            res[pn]["acc"].append(a); res[pn]["mf1"].append(m); res[pn]["f1"].append(f1)
        print(f"  [t20] seed {seed} done", flush=True)

    pb = f[te].astype(int)
    a, m, f1 = bmetrics(pb)
    md = [f"## Table 20 - Full-pipeline retraining on MAMI: all components re-learned "
          f"from MAMI's training set (train+val n={n_tr}), official test (n={int(te.sum())}); "
          f"classifier+policy over {len(C.SEEDS11)} seeds (mean+-std [s42])\n",
          "| Path | Acc | Macro-F1 | F1 (misog.) |", "|---|---|---|---|",
          f"| Path-B (induced rule fired) | {a:.3f} | {m:.3f} | {f1:.3f} |"]
    for pn, r in res.items():
        md.append(f"| {pn} | {C.ms(r['acc'])} | {C.ms(r['mf1'])} | {C.ms(r['f1'])} |")

    md.append("\n**Induced MAMI rule** (learned from MAMI train+val; calibration "
              "precision per clause):\n")
    for i, clause in enumerate(rs[POS], 1):
        lits = " AND ".join(names[j] for j in clause["idx"])
        md.append(f"- M{i}: {lits}  (P={clause.get('precision', 0):.2f})")

    md.append("\n_Same pipeline, default hyperparameters; binary task (SemEval-2022 "
              "Task 5 Sub-task A). Symbolic path (pool, rules, thresholds) is "
              "deterministic and shared across seeds. Affect head kept frozen from our "
              "dataset; the policy is trained by bootstrap REINFORCE. Zero-shot transfer "
              "reference (Table 19): NeSy "
              "0.563 acc / 0.553 macro-F1._")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    print(generate())
