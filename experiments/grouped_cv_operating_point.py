"""Does the operating point depend on near-duplicates inside the training pool?

The deployed operating point w is chosen by 5-fold cross-validation over train+val
(concept_routed.select_weight_cv). Those folds are stratified but not grouped, so a fold can be
scored on near-copies of the memes its concept bank and scorecard were fitted on, which would make
the concepts look better than they are and push w toward them.

This runs the same 11-seed protocol twice, changing only the fold construction:

  ungrouped  StratifiedKFold      -- the deployed choice, reproduces repair_benchmark_split.json
  grouped    StratifiedGroupKFold -- near-duplicate clusters (the group-split linkage: caption, OCR
                                     or image CLIP cosine >= 0.95, or identical normalised OCR text)
                                     kept inside one fold

and reports, for each, the chosen w per seed and the test-set result of the system it selects, with
paired bootstrap intervals over the test rows. Everything else -- concept bank, rules, scorecard,
neural classifier, splits -- is identical, so any difference is the fold construction alone.

  python experiments/grouped_cv_operating_point.py [--seeds 11]
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

OUT = config.ARTIFACTS_DIR / "grouped_cv_operating_point.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=11)
    seeds = [int(s) for s in C.SEEDS11[:ap.parse_args().seeds]]

    import duplicate_free_split as gs
    df, lab, _, info = gs.groups_and_split()          # clusters for every row of the corpus
    d = evp.load_data()
    assert df["id"].astype(str).tolist() == d.df["id"].astype(str).tolist(), "group rows are not the corpus rows"
    tv = d.tr | d.va
    sizes = np.bincount(lab[tv])
    print(f"[gcv] {info['n_groups']} duplicate groups over {len(lab)} memes; in the training pool "
          f"{100 * (sizes[lab[tv]] > 1).mean():.1f}% of memes share a group with another", flush=True)

    te_i = np.flatnonzero(d.te)
    y = d.y[te_i]
    k = len(config.LABELS)
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    sym = evp.build_symbolic(d)
    res = {"seeds": seeds, "groups": {k2: info[k2] for k2 in ("n_groups", "largest_group",
                                                              "share_in_multi_groups")},
           "share_of_pool_in_multi_groups": float((sizes[lab[tv]] > 1).mean()), "per_seed": {}}
    preds = {"ungrouped": [], "grouped": []}
    neural = []

    for s in seeds:
        t0 = time.time()
        row = {}
        for name, groups in (("ungrouped", None), ("grouped", lab)):
            sysm, logits = evp.fit(seed=s, data=d, sym=sym, groups=groups)
            out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i],
                              neural_logits=logits[te_i])
            pred, pn = out["pred"], logits[te_i].argmax(1)
            preds[name].append(pred)
            if name == "ungrouped":
                neural.append(pn)
            row[name] = {"w": sysm.w, "f1": cp.macro_f1(y, pred, k), "acc": float((pred == y).mean()),
                         "f1_neural": cp.macro_f1(y, pn, k), "acc_neural": float((pn == y).mean()),
                         "concept_share": float(out["concept_share"].mean()),
                         "changed_vs_neural": float((pred != pn).mean())}
        res["per_seed"][str(s)] = row
        print(f"[gcv] seed {s}: {time.time() - t0:.0f}s | w {row['ungrouped']['w']} -> "
              f"{row['grouped']['w']} | F1 {row['ungrouped']['f1']:.3f} -> {row['grouped']['f1']:.3f} | "
              f"acc {row['ungrouped']['acc']:.3f} -> {row['grouped']['acc']:.3f}", flush=True)

    N = np.array(neural)
    for name in ("ungrouped", "grouped"):
        P = np.array(preds[name])
        res[name] = {"w": [res["per_seed"][str(s)][name]["w"] for s in seeds],
                     "vs_neural": paired_bootstrap(y, P, N, k)}
    res["grouped_vs_ungrouped"] = paired_bootstrap(y, np.array(preds["grouped"]),
                                                   np.array(preds["ungrouped"]), k)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)

    for name in ("ungrouped", "grouped"):
        b = res[name]["vs_neural"]
        ws = res[name]["w"]
        print(f"\n[gcv] {name}: w {sorted(set(ws))} (mode {max(set(ws), key=ws.count)}); "
              f"dF1 {b['d_macro_f1']['mean']:+.3f} {np.round(b['d_macro_f1']['ci'], 3)}; "
              f"dacc {b['d_acc']['mean']:+.3f} {np.round(b['d_acc']['ci'], 3)}")
    g = res["grouped_vs_ungrouped"]
    print(f"[gcv] grouped minus ungrouped: dF1 {g['d_macro_f1']['mean']:+.3f} "
          f"{np.round(g['d_macro_f1']['ci'], 3)}, dacc {g['d_acc']['mean']:+.3f} "
          f"{np.round(g['d_acc']['ci'], 3)}")
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
