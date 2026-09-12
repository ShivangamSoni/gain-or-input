"""Calibrating the audit's screening defaults with planted leaks (fixed before the run).

Real datasets do not say whether a leak is present, so each threshold is calibrated where the answer
is known: leaks of controlled strength are planted into tasks the data tier calls clean, and each
check's statistic is compared with the planted truth.

  D3  a content-free shape cue (upper-case + " !!!") on a fraction rho of one class in every split
      (label-dependent plant) or on as many random items (control). Truth: the oracle lift of the
      cue alone (the same shape model on the cue indicator only).
  D6  near-copies (lower-cased, punctuation removed, one word dropped) of a fraction f of test items
      added to training with the same label (leaky plant) or a random other label (control).
  D2  exact binomial probability that a relation independent of the label passes the rule.
  D4  null distribution of the shape-to-source AUC with sources permuted.
  S5  the saved concept-routed system with each decision's concept ranking replaced by a random one
      (a planted unfaithful explanation): the deletion ratio against the threshold of 2.

  python experiments/threshold_calibration.py [--only d3,d6,d2,d4,s5]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import binom

import d3_surface_screen as ss
from nesymis import config
from inputaudit.data_tier import D3_NOTABLE, D6_FLAG, d3_surface, d6_duplicates
from inputaudit.probes import macro_f1, shape_auc, shape_model
from inputaudit.shape import shape_features

OUT = config.ARTIFACTS_DIR / "threshold_calibration.json"
BASES = ["exist-bin", "exist-6", "hateful", "mmhs-bin", "mmhs-6"]
RHOS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
FS = [0.02, 0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2]


def split_of(s):
    return np.where(s["tr"], "train", np.where(s["va"], "val", np.where(s["te"], "test", "")))


def cue(t):
    return str(t).upper() + " !!!"


def oracle_lift(ind, y, sp, k):
    """Lift of the shape model given ONLY the planted-cue indicator: the leak's true size."""
    tr, va, te = sp == "train", sp == "val", sp == "test"
    maj = int(np.bincount(y[tr | va], minlength=k).argmax())
    p = shape_model(ind, y, tr, va).predict(ind[te])
    return macro_f1(y[te], p, k) - macro_f1(y[te], np.full(int(te.sum()), maj), k)


# --------------------------------------------------------------------------- #
def run_d3():
    res = {}
    for key in BASES:
        t0 = time.time()
        s = ss.TASKS[key]()
        y, sp, k = np.asarray(s["y"]).astype(int), split_of(s), s["k"]
        texts = [str(t) for t in s["texts"]]
        cnt = np.bincount(y[(sp == "train") | (sp == "val")], minlength=k)
        target = int(np.argsort(-cnt)[1])                # the largest class after the majority
        rows = [{"kind": "base", "rho": 0.0, "d3_lift": d3_surface(texts, y, sp, k, boot=0)["lift"],
                 "oracle_lift": 0.0}]
        tgt = np.flatnonzero(y == target)
        for rho in RHOS:
            for seed in SEEDS:
                rng = np.random.default_rng(10_000 * seed + int(rho * 1000))
                leak = rng.choice(tgt, int(round(rho * len(tgt))), replace=False)
                ctrl = rng.choice(len(y), len(leak), replace=False)
                for kind, idx in (("leak", leak), ("control", ctrl)):
                    tt = list(texts)
                    for i in idx:
                        tt[i] = cue(tt[i])
                    ind = np.zeros((len(y), 1))
                    ind[idx, 0] = 1.0
                    rows.append({"kind": kind, "rho": rho, "seed": seed,
                                 "d3_lift": d3_surface(tt, y, sp, k, boot=0)["lift"],
                                 "oracle_lift": oracle_lift(ind, y, sp, k)})
        res[key] = {"target_class": target, "rows": rows}
        print(f"[cal] D3 {key}: {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
    return res


# --------------------------------------------------------------------------- #
def perturb(t, rng):
    """A near-copy: lower-cased, punctuation removed, one word dropped when there are 3 or more."""
    w = re.sub(r"[^\w\s]", " ", str(t).lower()).split()
    if len(w) >= 3:
        w.pop(int(rng.integers(len(w))))
    return " ".join(w)


def run_d6():
    import d6_duplicates as ad
    from nesymis.encoders.clip_encoder import encode_texts
    res = {}
    for key in BASES:
        t0 = time.time()
        s = ss.TASKS[key]()
        y, sp, k = np.asarray(s["y"]).astype(int), split_of(s), s["k"]
        texts = [str(t) for t in s["texts"]]
        E = np.asarray(ad.text_embeddings(key, s), np.float32)
        te_i = np.flatnonzero((sp == "test") & np.array([bool(t.strip()) for t in texts]))
        base = d6_duplicates(texts, sp, E, y)
        rows = [{"kind": "base", "f": 0.0, "planted": 0.0, "excess": base["leaky_excess"],
                 "excess_bal": base["leaky_excess_balanced"], "dup_share": base["dup_share"]}]
        n_test = int((sp == "test").sum())
        for f in FS:
            for seed in SEEDS:
                rng = np.random.default_rng(20_000 * seed + int(f * 1000))
                pick = rng.choice(te_i, int(round(f * n_test)), replace=False)
                new_t = [perturb(texts[i], rng) for i in pick]
                new_E = np.concatenate([encode_texts(new_t[i:i + 256]) for i in range(0, len(new_t), 256)])
                for kind in ("leak", "control"):
                    new_y = y[pick] if kind == "leak" else (y[pick] + rng.integers(1, k, len(pick))) % k
                    r = d6_duplicates(texts + new_t, np.concatenate([sp, np.full(len(pick), "train")]),
                                      np.vstack([E, new_E]), np.concatenate([y, new_y]))
                    rows.append({"kind": kind, "f": f, "seed": seed,
                                 "planted": f if kind == "leak" else 0.0,
                                 "excess": r["leaky_excess"], "excess_bal": r["leaky_excess_balanced"],
                                 "dup_share": r["dup_share"]})
        res[key] = {"rows": rows}
        print(f"[cal] D6 {key}: {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
    return res


# --------------------------------------------------------------------------- #
def run_d2():
    """P(a label-independent relation on n rows has top-class precision >= 0.95) when the top class
    has base rate pi -- the rule's false-flag probability without and with an enrichment test."""
    out = {}
    for pi in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97):
        for n in (20, 50, 100, 500):
            need = int(np.ceil(0.95 * n))
            out[f"pi={pi},n={n}"] = {"pi": pi, "n": n, "p_flag": float(binom.sf(need - 1, n, pi))}
    return out


# --------------------------------------------------------------------------- #
def run_d4(perms=50):
    import pandas as pd
    res = {}
    df = pd.read_csv(config.CORRECTED_MANIFEST_PATH).fillna("")
    src = df["source"].str.startswith("wbms").to_numpy().astype(int)
    sp = df["split"].to_numpy()
    cases = {"ours (uniform OCR)": (df["text_ocr"].tolist(), src, sp)}
    try:
        import d2_d4_external_tasks as ext
        hm = ext.harmeme_all()
        cases["HarMeme"] = (hm.text.tolist(), pd.factorize(hm["source"])[0], hm.split.to_numpy())
    except Exception as e:                              # HarMeme loader optional
        print(f"[cal] D4: HarMeme skipped ({e})", flush=True)
    for name, (texts, s_, split) in cases.items():
        X = shape_features(texts)
        fit, te = np.isin(split, ["train", "val"]), split == "test"
        real = shape_auc(X, s_, fit, te)
        rng = np.random.default_rng(7)
        null = [shape_auc(X, rng.permutation(s_), fit, te) for _ in range(perms)]
        res[name] = {"auc": real, "null_mean": float(np.mean(null)), "null_p99": float(np.percentile(null, 99)),
                     "null_max": float(np.max(null)), "perms": perms}
        print(f"[cal] D4 {name}: AUC {real:.3f}; null mean {np.mean(null):.3f}, 99th pct "
              f"{np.percentile(null, 99):.3f}", flush=True)
    return res


# --------------------------------------------------------------------------- #
def run_s5(draws=20):
    from repair_core import faithfulness
    from nesymis import concept_routed as evp
    d, s = evp.load_data(), evp.load()
    te = np.flatnonzero(d.te)
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    out = s.decide(vu[te], d.u[te], d.v[te], index=d.df.index[te])
    rng = np.random.default_rng(11)
    real = faithfulness(s.card, out["Z"], out["ln"], s.w, out, rng)[3]
    ratios = []
    for _ in range(draws):
        C = out["contrib"]
        perm = np.argsort(rng.random(C.shape), 1)         # a random concept ranking per decision
        fake = dict(out, contrib=np.take_along_axis(C, perm, 1))
        f = faithfulness(s.card, out["Z"], out["ln"], s.w, fake, rng)[3]
        ratios.append(f["delete_top_changes"] / max(f["delete_random_changes"], 1e-12))
    res = {"real_ratio": real["delete_top_changes"] / real["delete_random_changes"],
           "planted_ratio_mean": float(np.mean(ratios)), "planted_ratio_p99": float(np.percentile(ratios, 99)),
           "planted_ratio_max": float(np.max(ratios)), "draws": draws}
    print(f"[cal] S5: real ratio {res['real_ratio']:.2f}; random-ranking ratio mean "
          f"{res['planted_ratio_mean']:.2f}, max {res['planted_ratio_max']:.2f}", flush=True)
    return res


# --------------------------------------------------------------------------- #
def summarise(res):
    """Detection and false-flag rates over threshold grids."""
    summ = {}
    if "d3" in res:
        leak = [r for t in res["d3"].values() for r in t["rows"] if r["kind"] == "leak"]
        neg = [r for t in res["d3"].values() for r in t["rows"] if r["kind"] != "leak"]
        grid = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
        summ["d3"] = {str(t): {
            "false_flag": float(np.mean([r["d3_lift"] >= t for r in neg])),
            "detect_oracle_ge_0.10": float(np.mean([r["d3_lift"] >= t for r in leak if r["oracle_lift"] >= 0.10])),
            "detect_oracle_ge_0.20": float(np.mean([r["d3_lift"] >= t for r in leak if r["oracle_lift"] >= 0.20]))}
            for t in grid}
        ol = np.array([r["oracle_lift"] for r in leak])
        dl = np.array([r["d3_lift"] for r in leak])
        summ["d3_recovery"] = {"slope": float(np.polyfit(ol, dl, 1)[0]), "n": int(len(ol)),
                               "n_material_0.10": int((ol >= 0.10).sum()), "n_material_0.20": int((ol >= 0.20).sum())}
    if "d6" in res:
        leak = [r for t in res["d6"].values() for r in t["rows"] if r["kind"] == "leak"]
        neg = [r for t in res["d6"].values() for r in t["rows"] if r["kind"] != "leak"]
        grid = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
        for stat, name in (("excess", "d6"), ("excess_bal", "d6_balanced")):
            if not all(stat in r for r in leak + neg):
                continue
            summ[name] = {str(t): {
                "false_flag": float(np.mean([r[stat] >= t for r in neg])),
                **{f"detect_planted_{f}": float(np.mean([r[stat] >= t for r in leak if r["planted"] == f]))
                   for f in FS}} for t in grid}
    return summ


def choose(rows, stat, grid, fp_max=0.05):
    """The rule the paper states: the smallest grid threshold whose false-flag rate on rows that
    carry no label-dependent leak (unplanted bases and label-independent controls) is at most 5%."""
    for t in grid:
        if float(np.mean([r[stat] >= t for r in rows if r["kind"] != "leak"])) <= fp_max:
            return t
    return grid[-1]


def leave_one_out(res, fp_max=0.05):
    """Are the thresholds an artefact of the tasks they were chosen on? For each task in turn, the
    threshold is chosen on the other four and applied to the held-out task, which had no part in
    choosing it. Pure re-analysis of the rows already in the artefact."""
    out = {}
    for key, stat, grid, det in (("d3", "d3_lift", [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30],
                                  lambda r: r["oracle_lift"] >= 0.10),
                                 ("d6", "excess", [0.025, 0.05, 0.075, 0.10, 0.15, 0.20],
                                  lambda r: r["planted"] > 0)):
        if key not in res:
            continue
        tasks = sorted(res[key])
        block = {}
        for held in tasks:
            train = [r for t in tasks if t != held for r in res[key][t]["rows"]]
            thr = choose(train, stat, grid, fp_max)
            rows = res[key][held]["rows"]
            neg = [r for r in rows if r["kind"] != "leak"]
            leak = [r for r in rows if r["kind"] == "leak" and det(r)]
            block[held] = {"threshold_from_other_tasks": thr,
                           "false_flag": float(np.mean([r[stat] >= thr for r in neg])), "n_negative": len(neg),
                           "detect": float(np.mean([r[stat] >= thr for r in leak])) if leak else None,
                           "n_leak": len(leak)}
        thrs = [b["threshold_from_other_tasks"] for b in block.values()]
        block["summary"] = {"thresholds": thrs, "adopted": D3_NOTABLE[0] if key == "d3" else D6_FLAG,
                            "held_out_false_flag_mean": float(np.mean([b["false_flag"] for t, b in block.items()
                                                                      if t != "summary"])),
                            "held_out_false_flag_max": float(np.max([b["false_flag"] for t, b in block.items()
                                                                    if t != "summary"])),
                            "held_out_detect_mean": float(np.mean([b["detect"] for t, b in block.items()
                                                                   if t != "summary" and b["detect"] is not None]))}
        out[key] = block
        s = block["summary"]
        print(f"[cal] {key.upper()} leave-one-task-out: thresholds {thrs} (adopted {s['adopted']}); "
              f"held-out false flags mean {s['held_out_false_flag_mean']:.3f}, max "
              f"{s['held_out_false_flag_max']:.3f}; held-out detection {s['held_out_detect_mean']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="d2,d4,s5,d3,d6")
    which = ap.parse_args().only.split(",")
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    for name, fn in (("d2", run_d2), ("d4", run_d4), ("s5", run_s5), ("d3", run_d3), ("d6", run_d6)):
        if name in which:
            res[name] = fn()
            res["summary"] = summarise(res)
            json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    if "loo" in which or which == ["loo"]:
        res["loo"] = leave_one_out(res)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(res.get("summary", {}), indent=1))
    print(f"_saved {OUT}_")


if __name__ == "__main__":
    main()
