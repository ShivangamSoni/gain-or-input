"""Per-class paired bootstrap intervals for the appendix's per-class tables.

The per-class tables report each class's F1 with its seed-to-seed spread. This script adds the other
source of uncertainty -- which test memes happen to be in the split -- as a 95% paired bootstrap
interval (2,000 resamples of the test memes, every seed scored on each resample, averaged over seeds)
for each class's F1 difference. It re-runs the two experiments behind the tables to recover their
test predictions, and checks that every seed reproduces the saved per-class F1 before using them:

  S2      s2_text_regimes.run(): uniform-OCR protocol, symbolic layer on OCR (equal inputs) and
          on the post caption, 5 seeds; difference = full system - neural classifier
  repair  concept_routed.fit(): the concept-routed decision on the benchmark split, 11 seeds;
          difference = concept-routed - neural classifier

  python experiments/per_class_intervals.py [--only s2,repair]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
from nesymis import config

OUT = config.ARTIFACTS_DIR / "per_class_intervals.json"
LAB = list(config.LABELS)
K = len(LAB)
B = 2000


def f1_per_class(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    den = cm.sum(0) + cm.sum(1)
    return np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0.0)


def perclass_boot(y, Ps, Pb, seed=config.SEED + 1):
    """Ps, Pb: (seeds, N) predictions of the system and the baseline. Per-class F1 difference,
    seed-averaged, with a 95% interval over resampled test rows."""
    rng = np.random.default_rng(seed)
    n, S = len(y), len(Ps)
    d = lambda i: np.mean([f1_per_class(y[i], Ps[s, i]) - f1_per_class(y[i], Pb[s, i]) for s in range(S)], 0)
    stats = np.array([d(rng.integers(0, n, n)) for _ in range(B)])
    point = d(np.arange(n))
    lo, hi = np.percentile(stats, 2.5, 0), np.percentile(stats, 97.5, 0)
    return {LAB[c]: {"mean": float(point[c]), "ci": [float(lo[c]), float(hi[c])],
                     "n_test": int((y == c).sum())} for c in range(K)}


def s2():
    """Per-class intervals for both S2 configurations, and an interval for the swing between them:
    the full system's macro-F1 with the post caption minus with the OCR text both components read."""
    import neural_scaling as t22
    import s2_text_regimes as trc
    from inputaudit.probes import paired_bootstrap
    saved = json.load(open(config.ARTIFACTS_DIR / "s2_text_regimes.json", encoding="utf-8"))
    c = C.build_context()
    slices = trc._slices(c)
    y = c.y[c.te]
    out, P = {}, {}
    for regime in ("corrected", "corrected-split"):
        t22.PRED_SINK = {}
        trc.run(regime, c, slices, saved[regime]["seeds"])
        pa, ne = (np.array([p[j] for p in t22.PRED_SINK[regime]]) for j in (0, 1))
        for row, Q in (("path_a", pa), ("nesymis", ne)):
            want = [[d[l] for l in LAB] for d in saved[regime][row]["full"]["per_class"]]
            got = [list(f1_per_class(y, q)) for q in Q]
            err = float(np.abs(np.array(want) - np.array(got)).max())
            print(f"[pc] S2 {regime} {row}: max per-class F1 deviation from the saved run {err:.2e}", flush=True)
            assert err < 1e-9, f"S2 {regime} {row} does not reproduce the saved per-class F1"
        out[regime] = perclass_boot(y, ne, pa)
        P[regime] = (ne, pa)
    t22.PRED_SINK = None
    # S2's swing, with the same paired bootstrap the tool uses for a margin
    sw = paired_bootstrap(y, P["corrected-split"][0], P["corrected"][0], K)
    out["swing"] = {"d_macro_f1": sw["d_macro_f1"], "d_acc": sw["d_acc"],
                    "note": "full system, post caption minus equal OCR text (2,000 test-row resamples)"}
    e = sw["d_macro_f1"]
    print(f"[pc] S2 swing (caption - equal text): {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]", flush=True)
    return out


def repair():
    from nesymis import concept_routed as evp
    saved = json.load(open(config.ARTIFACTS_DIR / "repair_benchmark_split.json", encoding="utf-8"))["per_seed"]
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    y = d.y[te_i]
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    sym = evp.build_symbolic(d)
    Pp, Pn = [], []
    for rec in saved:
        sysm, logits = evp.fit(seed=rec["seed"], data=d, sym=sym, persist=False)
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i], neural_logits=logits[te_i])
        pred, pn = out["pred"], logits[te_i].argmax(1)
        err = max(abs(f1_per_class(y, p)[c] - rec["per_class"][LAB[c]][k])
                  for p, k in ((pred, "f1_pooled"), (pn, "f1_neural")) for c in range(K))
        print(f"[pc] repair seed {rec['seed']}: w={sysm.w} (saved {rec['w']}); max per-class F1 deviation "
              f"{err:.2e}", flush=True)
        assert err < 1e-9 and sysm.w == rec["w"], f"repair seed {rec['seed']} does not reproduce the saved run"
        Pp.append(pred)
        Pn.append(pn)
    return perclass_boot(y, np.array(Pp), np.array(Pn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="s2,repair")
    parts = ap.parse_args().only.split(",")
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    res["bootstrap_B"] = B
    for part in parts:
        res[part] = {"s2": s2, "repair": repair}[part]()
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for part in ("s2", "repair"):
        if part not in res:
            continue
        blocks = res[part].items() if part == "s2" else [("concept-routed", res[part])]
        for name, r in blocks:
            if name == "swing":                               # not a per-class block
                e = r["d_macro_f1"]
                print(f"\ns2 | swing: {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] ({r['note']})")
                continue
            print(f"\n{part} | {name}")
            for c, e in r.items():
                print(f"  {c:15s} n={e['n_test']:3d}  d={e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
