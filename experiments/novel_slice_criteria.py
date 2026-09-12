"""Does the novel slice depend on being defined by caption similarity?

The novel slice -- test memes whose nearest train+val neighbour is below 0.90 cosine -- has always
been defined on the post caption, which is awkward: the caption is the privileged input the audit
objects to, and it is not an input of the repaired system at all. This re-runs both experiments that
report a novel slice and recomputes their novel-slice numbers under three criteria:

  caption  CLIP text embedding of the post caption   (as reported)
  image    CLIP image embedding
  joint    the two concatenated and L2-normalised    (a mean of the two cosines)

Nothing about the systems changes: same seeds, same splits, same predictions. Only which test rows
count as novel changes, so any difference is the criterion's.

  python experiments/novel_slice_criteria.py [--seeds 11]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp
from inputaudit.probes import paired_bootstrap

OUT = config.ARTIFACTS_DIR / "novel_slice_criteria.json"
COS = 0.90


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def masks(te_i, trainval):
    """novel masks over the test rows, one per criterion."""
    cap = unit(np.load(config.TEXT_CAPTION_EMB_PATH))
    img = unit(np.load(config.IMAGE_EMB_PATH))
    joint = unit(np.concatenate([cap, img], 1))
    out = {}
    for name, E in (("caption", cap), ("image", img), ("joint", joint)):
        sim = E[te_i] @ E[trainval].T
        out[name] = sim.max(1) < COS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=11)
    n_seeds = ap.parse_args().seeds
    seeds = [int(s) for s in C.SEEDS11[:n_seeds]]
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    tv = np.flatnonzero(d.tr | d.va)
    y = d.y[te_i]
    k = len(config.LABELS)
    M = masks(te_i, tv)
    res = {"cos": COS, "seeds": seeds,
           "slice_sizes": {n: int(m.sum()) for n, m in M.items()}, "n_test": int(len(y)),
           "agreement": {f"{a}|{b}": float((M[a] == M[b]).mean()) for a, b in
                         (("caption", "image"), ("caption", "joint"), ("image", "joint"))}}
    print(f"[nov] novel slice sizes of {len(y)} test memes: {res['slice_sizes']}; "
          f"agreement {json.dumps({a: round(v, 3) for a, v in res['agreement'].items()})}", flush=True)

    # --- the repair (repair_benchmark_split's system, 11 seeds) ---
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    sym = evp.build_symbolic(d)
    Pp, Pn = [], []
    for s in seeds:
        t0 = time.time()
        sysm, logits = evp.fit(seed=s, data=d, sym=sym)
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i], neural_logits=logits[te_i])
        Pp.append(out["pred"])
        Pn.append(logits[te_i].argmax(1))
        print(f"[nov] repair seed {s}: {time.time() - t0:.0f}s (w={sysm.w})", flush=True)
    Pp, Pn = np.array(Pp), np.array(Pn)
    res["repair"] = {n: {"n": int(m.sum()),
                         "f1_neural": float(np.mean([cp.macro_f1(y[m], p[m], k) for p in Pn])),
                         "f1_pooled": float(np.mean([cp.macro_f1(y[m], p[m], k) for p in Pp])),
                         **paired_bootstrap(y[m], Pp[:, m], Pn[:, m], k)}
                     for n, m in M.items()}

    # --- S2 under the uniform-OCR protocol (s2_text_regimes, 5 seeds) ---
    import neural_scaling as t22
    import s2_text_regimes as trc
    saved = json.load(open(config.ARTIFACTS_DIR / "s2_text_regimes.json", encoding="utf-8"))
    c = C.build_context()
    slices = trc._slices(c)
    S2 = {}
    for regime in ("corrected", "corrected-split"):
        t22.PRED_SINK = {}
        trc.run(regime, c, slices, saved[regime]["seeds"])
        pa, ne = (np.array([p[j] for p in t22.PRED_SINK[regime]]) for j in (0, 1))
        assert abs(np.mean([cp.macro_f1(y, p, k) for p in ne])
                   - np.mean(saved[regime]["nesymis"]["full"]["macro_f1"])) < 1e-9, "S2 did not reproduce"
        S2[regime] = {n: {"n": int(m.sum()),
                          "margin": paired_bootstrap(y[m], ne[:, m], pa[:, m], k)["d_macro_f1"]}
                      for n, m in M.items()}
    t22.PRED_SINK = None
    res["s2"] = S2
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)

    print("\n[nov] repair, novel-slice macro-F1 difference (concept-routed - neural):")
    for n, r in res["repair"].items():
        e = r["d_macro_f1"]
        print(f"  {n:8s} n={r['n']:3d}  {r['f1_neural']:.3f} -> {r['f1_pooled']:.3f}  "
              f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    print("[nov] S2 margin (full system - neural) on the novel slice:")
    for regime, blk in S2.items():
        for n, r in blk.items():
            e = r["margin"]
            print(f"  {regime:16s} {n:8s} n={r['n']:3d}  {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
