"""Duplicate-aware evaluation: whole near-duplicate groups kept on one side of the split
(fixed before any model was run).

Groups: connected components of links between memes -- caption CLIP cosine >= 0.95, in-image (uniform
OCR) text CLIP cosine >= 0.95 between texts of >= 3 normalised characters, image CLIP cosine >= 0.95,
or identical normalised OCR text (>= 3 characters). Split: StratifiedGroupKFold (20 folds, shuffled,
seed 42), 3 folds test, 3 validation, 14 training. Under it:

  s2       the original design, symbolic layer on the post caption vs on the uniform OCR text
           (neural classifier on OCR in both), 5 seeds
  repair   the deployed concept-routed decision on the five-class case-study corpus, 11 seeds
  wbms4    the same on WBMS memes only (the typing setting), 11 seeds

Intervals are cluster-bootstrap intervals (groups resampled); row-bootstrap intervals are kept too.

  python experiments/duplicate_free_split.py [--only s2,repair,wbms4]
"""
import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from nesymis import config

OUT = config.ARTIFACTS_DIR / "duplicate_free_split.json"
COS = 0.95


def _unit(p):
    E = np.load(p).astype(np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def groups_and_split():
    """Duplicate groups and the group split, for the rows of the corrected manifest."""
    df = pd.read_csv(config.CORRECTED_MANIFEST_PATH).fillna("")
    n = len(df)
    cap, ocr, img = (_unit(p) for p in (config.TEXT_CAPTION_EMB_PATH, config.TEXT_OCR_CORRECTED_EMB_PATH,
                                        config.IMAGE_EMB_PATH))
    txt = [re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip() for t in df["text_ocr"]]
    ok = np.array([len(t) >= 3 for t in txt])
    A = np.zeros((n, n), bool)
    for E, mask in ((cap, None), (ocr, ok), (img, None)):
        S = E @ E.T
        L = S >= COS
        if mask is not None:
            L &= mask[:, None] & mask[None, :]
        A |= L
    same = {}
    for i, t in enumerate(txt):
        if ok[i]:
            same.setdefault(t, []).append(i)
    for members in same.values():
        for i in members:
            A[i, members] = True
    np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A), directed=False)
    y = df["label_idx"].to_numpy()
    folds = [te for _, te in StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42).split(
        np.zeros(n), y, lab)]
    split = np.full(n, "train", dtype=object)
    for i in range(3):
        split[folds[i]] = "test"
    for i in range(3, 6):
        split[folds[i]] = "val"
    i, j = np.nonzero(A)
    assert (split[i] == split[j]).all(), "a duplicate link crosses the split"
    sizes = np.bincount(lab)
    info = {"n_groups": int(lab.max() + 1), "largest_group": int(sizes.max()),
            "share_in_multi_groups": float((sizes[lab] > 1).mean()),
            "split_sizes": {s: int((split == s).sum()) for s in ("train", "val", "test")},
            "test_class_counts": np.bincount(y[split == "test"], minlength=5).tolist()}
    return df, lab, split.astype(str), info


# --------------------------------------------------------------------------- #
def run_s2(df, lab, split, seeds=5):
    """The original design's same-input test under the group split (cf. s2_text_regimes)."""
    import common as C
    import uniform_ocr_protocol as cb
    import neural_scaling as t22
    from inputaudit.probes import paired_bootstrap
    c = C.build_context()
    c.tr, c.va, c.te = split == "train", split == "val", split == "test"
    c.trainval = c.tr | c.va
    dfc, _y, _v, u_ocr, u_cap, *_ = cb.load_corrected()
    y, v = c.y, c.emb.image
    vu = np.concatenate([v, u_ocr], 1).astype(np.float32)                 # neural: image + uniform OCR
    D = np.zeros((len(y), config.AFFECT_DIM), np.float32)
    yt, g = y[c.te], lab[c.te]
    out = {}
    for regime, sym_u, col in (("equal text (OCR)", u_ocr, "text_ocr"), ("post caption", u_cap, "text_caption")):
        df2 = c.df.copy()
        df2["text_caption"] = dfc[col].fillna("").astype(str).to_numpy()
        fired, rvec, pb = t22._induce(df2, y, c.trainval, sym_u, v)
        PA, NE = [], []
        for seed in C.SEEDS11[:seeds]:
            _, logits = t22.clf.crossfit_logits(vu, D, y, c.tr, c.va, fusion=config.NEURAL_FUSION,
                                               affect_dim=C.odesign.AFFECT_DIM, cfg=t22._clf_cfg(None), seed=int(seed))
            state = C.rl.build_state(logits, fired, rvec, D)
            pol = C.rl.train_rlvr(logits, state, y, c.tr, c.va, seed=int(seed),
                                  policy_ctor=lambda dd: t22.LinearEvidencePolicy(dd))[0]
            NE.append(C.rl.greedy_pred(pol, logits[c.te], state[c.te]))
            PA.append(logits[c.te].argmax(1))
        f1 = lambda p: C.compute_metrics(yt, p)["macro_f1"]
        out[regime] = {"rules_alone_f1": f1(pb[c.te]), "neural_f1": [f1(p) for p in PA],
                       "full_f1": [f1(p) for p in NE],
                       "margin_cluster": paired_bootstrap(yt, np.array(NE), np.array(PA), 5, groups=g),
                       "margin_rows": paired_bootstrap(yt, np.array(NE), np.array(PA), 5)}
        r = out[regime]
        print(f"[grp] S2 {regime}: rules {r['rules_alone_f1']:.3f} neural {np.mean(r['neural_f1']):.3f} "
              f"full {np.mean(r['full_f1']):.3f} margin {r['margin_cluster']['d_macro_f1']['mean']:+.3f} "
              f"{[round(x, 3) for x in r['margin_cluster']['d_macro_f1']['ci']]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
def run_repair(df, lab, split, which):
    import repair_source_controlled as source_controlled
    from nesymis import concept_routed as evp
    from inputaudit.data_tier import d6_duplicates
    d0 = evp.load_data()
    assert d0.df["id"].astype(str).tolist() == df["id"].astype(str).tolist()
    tr, va, te = split == "train", split == "val", split == "test"
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)
    if which == "repair":
        d = SimpleNamespace(**vars(d0))
        d.tr, d.va, d.te = tr, va, te
        d.classes = list(config.LABELS)                  # concepts mined for the stereotype classes, as deployed
        m = np.ones(len(df), bool)
        y_all = d.y
    else:
        m = d0.df["source"].str.startswith("wbms").to_numpy()
        classes = list(config.STEREO_CLASSES)
        sub = d0.df[m].reset_index(drop=True)
        y_all = sub["label"].map({c_: i for i, c_ in enumerate(classes)}).to_numpy()
        d = SimpleNamespace(df=sub, v=d0.v[m], u=d0.u[m], y=y_all, tr=tr[m], va=va[m], te=te[m],
                            text=d0.text[m].reset_index(drop=True), classes=classes, mine_classes=classes,
                            text_channel="OCR of the image (uniform-OCR protocol)")
    sp = np.where(d.tr, "train", np.where(d.va, "val", "test"))
    loose = d6_duplicates(d.df["text_caption"].tolist(), sp, u_cap[m], y_all)      # cos 0.90, what remains
    res = source_controlled.evaluate(f"group split: {which}", d, loose["novel_test_mask"], np.random.default_rng(config.SEED),
                       groups=lab[m][d.te])
    res["loose_duplicates_at_0.90"] = {k: loose[k] for k in ("dup_share", "leaky_share", "leaky_null", "leaky_excess")}
    b = res["bootstrap"]
    print(f"[grp] {which}: dF1 {b['cluster']['d_macro_f1']['mean']:+.3f} {[round(x, 3) for x in b['cluster']['d_macro_f1']['ci']]} "
          f"dacc {b['cluster']['d_acc']['mean']:+.3f} {[round(x, 3) for x in b['cluster']['d_acc']['ci']]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="s2,repair,wbms4")
    which = ap.parse_args().only.split(",")
    df, lab, split, info = groups_and_split()
    print(f"[grp] groups {info['n_groups']}, largest {info['largest_group']}, in multi-member groups "
          f"{info['share_in_multi_groups']:.3f}; split {info['split_sizes']}; test classes {info['test_class_counts']}",
          flush=True)
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    res["groups"] = info
    for key in which:
        t0 = time.time()
        res[key] = run_s2(df, lab, split) if key == "s2" else run_repair(df, lab, split, key)
        res[key]["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print(f"_saved {OUT}_")


if __name__ == "__main__":
    main()
