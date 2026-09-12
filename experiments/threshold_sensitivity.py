"""How the audit's verdicts depend on its default thresholds.

The thresholds are default heuristics, not validated constants. For every check that has
thresholds, all of them are scaled together by m in {0.75, 0.875, 1, 1.125, 1.25} (probability-
like thresholds capped at 1), every verdict is recomputed from the saved statistics, and each
case's break-even multiplier -- where its verdict would flip -- is found on a fine grid in
[0.5, 1.5]. S2 (interval-based) and S6 (a criterion fixed in advance) have no heuristic
threshold and are not varied.

Statistics come from the saved artefacts of the runs reported in the paper; nothing is re-fitted
except D2 on our corpus (seconds).

  python experiments/threshold_sensitivity.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nesymis import config
from inputaudit.data_tier import D3_NOTABLE, D3_SEVERE, D6_FLAG, d2_field_relations

A = config.ARTIFACTS_DIR
OUT = A / "threshold_sensitivity.json"
MULTS = (0.75, 0.875, 1.0, 1.125, 1.25)
GRID = np.round(np.arange(0.50, 1.5001, 0.005), 3)
cap = lambda x: min(x, 1.0)


def load(name):
    return json.load(open(A / name, encoding="utf-8"))


def cases():
    """{check: [(case name, stats dict)]} and {check: rule(stats, m) -> flagged?}."""
    C, R = {}, {}
    # D2 -- our corpus (both versions, all relations) and MMHS150K
    # (the enrichment test's significance level is a statistical choice, held fixed, not scaled)
    c2 = []
    for ver, path in (("ours, first", config.MANIFEST_PATH), ("ours, uniform", config.CORRECTED_MANIFEST_PATH)):
        df = pd.read_csv(path).fillna("")
        d2 = d2_field_relations(df, ["text_ocr", "text_caption"], "label")
        for n, r in d2["relations"].items():
            c2.append((f"{ver}: {n}", {"support": r["support"], "precision": r["precision"],
                                       "enriched": r["p_value"] <= d2["thresholds"]["max_p"] / d2["thresholds"]["relations_tested"]}))
    ext = load("d2_d4_external_tasks.json")["mmhs150k"]["D2"]
    for n, r in ext["relations"].items():
        c2.append((f"MMHS150K: {n}", {"support": r["support"], "precision": r["precision"],
                                      "enriched": r["p_value"] <= ext["thresholds"]["max_p"] / ext["thresholds"]["relations_tested"]}))
    C["D2"], R["D2"] = c2, lambda s, m: (s["support"] >= 20 * m and s["precision"] >= cap(0.95 * m)
                                         and s["enriched"])

    # D3 -- the 14 screened rows
    ss, ts = load("d3_surface_screen.json"), load("d5_modality_probes.json")
    uni_ref = float(np.mean(load("uniform_ocr_protocol.json")["baselines"]["rows"]
                            ["official, text only: [u_ocr]"]["macro_f1"]))
    c3 = []
    for k, r in ss.items():
        ref = uni_ref if k == "ours-ocr-corrected" else float(np.mean(ts[k]["views"]["text"]["macro_f1"]))
        shp = r["surface"]["macro_f1"]
        c3.append((k, {"lift": shp - r["majority_f1"], "share": shp / ref}))

    def d3(s, m):
        sev = s["lift"] >= D3_SEVERE[0] * m and s["share"] >= cap(D3_SEVERE[1] * m)
        nota = s["lift"] >= D3_NOTABLE[0] * m and s["share"] >= cap(D3_NOTABLE[1] * m)
        return sev or nota
    C["D3"], R["D3"] = c3, d3

    # D4 -- our corpus and HarMeme
    av, hm = load("tool_validation.json")["ours"], load("d2_d4_external_tasks.json")["harmeme"]
    c4 = [(f"ours, {v}: {f}", {"auc": av[v]["D4 source AUC"][f][0], "v": 1.0})
          for v in ("first", "uniform") for f in ("text_caption", "text_ocr")]
    c4 += [(f"HarMeme {t}", {"auc": hm[t]["D4"]["source_auc_from_shape"], "v": hm[t]["D4"]["source_label_cramers_v"]})
           for t in ("harmful (binary)", "3-class intensity")]
    C["D4"], R["D4"] = c4, lambda s, m: s["auc"] >= cap(0.80 * m) and s["v"] >= 0.50 * m

    # D5 -- best single modality per task, both encoders where available
    sg = load("d5_two_encoders.json")
    c5 = []
    for k in ts:
        best = max(np.mean(ts[k]["views"][v]["macro_f1"]) for v in ("image", "text"))
        if k in sg:
            best = max(best, *(np.mean(sg[k]["views"][v]["macro_f1"]) for v in ("image", "text")))
        c5.append((k, {"best": float(best)}))
    C["D5"], R["D5"] = c5, lambda s, m: s["best"] >= cap(0.90 * m)

    # D6 -- leaky share above its label-permutation null, per screened row
    C["D6"] = [(k, {"excess": r["leaky_excess"]}) for k, r in load("d6_duplicates.json").items()]
    R["D6"] = lambda s, m: s["excess"] >= D6_FLAG * m

    # S3, S4, S5 -- the case study
    hb = load("headroom_analysis.json")["buckets"]
    C["S3"] = [("original design: residual after text alternatives", {"residual": hb["no-text-fix"] / 490})]
    R["S3"] = lambda s, m: s["residual"] < 0.02 * m
    st1, r3 = load("evidence_feasibility.json"), load("repair_benchmark_split.json")
    rules = float(np.mean([p["infl"]["flips_by_rules"] for p in st1["per_seed"]])) / 490
    conc = float(np.mean([p["concepts_decide"] for p in r3["per_seed"]]))
    C["S4"] = [("original design: rules", {"changed": rules}), ("repair: concepts", {"changed": conc})]
    R["S4"] = lambda s, m: s["changed"] < 0.02 * m
    ratio = (np.mean([p["faith"]["3"]["delete_top_changes"] for p in r3["per_seed"]])
             / np.mean([p["faith"]["3"]["delete_random_changes"] for p in r3["per_seed"]]))
    C["S5"] = [("repair: concepts (top-3 vs random-3)", {"ratio": float(ratio)})]
    R["S5"] = lambda s, m: s["ratio"] < 2.0 * m
    return C, R


def main():
    C, R = cases()
    out = {}
    print("| check | cases | flagged at 0.75x | 0.875x | **default** | 1.125x | 1.25x | cases that ever flip in [0.75, 1.25] |")
    print("|---|---|---|---|---|---|---|---|")
    for chk, cs in C.items():
        rule = R[chk]
        base = {n: rule(s, 1.0) for n, s in cs}
        counts = {m: sum(rule(s, m) for _, s in cs) for m in MULTS}
        flips = {}
        for n, s in cs:
            changes = [m for m in GRID if rule(s, m) != base[n]]
            lo = max([m for m in changes if m < 1.0], default=None)
            hi = min([m for m in changes if m > 1.0], default=None)
            flips[n] = {"default_flag": base[n], "flips_below": lo, "flips_above": hi}
        within = [n for n, f in flips.items()
                  if (f["flips_below"] is not None and f["flips_below"] >= 0.75)
                  or (f["flips_above"] is not None and f["flips_above"] <= 1.25)]
        out[chk] = {"n_cases": len(cs), "flagged": {str(m): counts[m] for m in MULTS},
                    "cases": {n: {**s, **flips[n]} for n, s in cs}, "flip_within_25pct": within}
        print(f"| {chk} | {len(cs)} | " + " | ".join(
            (f"**{counts[m]}**" if m == 1.0 else str(counts[m])) for m in MULTS)
            + f" | {', '.join(within) or 'none'} |")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
