"""Full protocol for the paper's system at the deployed operating point.

System: nesymis/concept_routed.py with its defaults -- balanced concept scorecard, pooling weight
chosen by 5-fold cross-validation over train+val with the concept bank rebuilt per fold, rule B
(the most concept-weighted w within 0.01 of the neural classifier on both macro-F1 and accuracy).
The choice was fixed by the fresh-partition study elsewhere in this work, whose fresh partitions give
its out-of-sample price. This script produces the benchmark-table numbers on the corrected
benchmark's own split; earlier runs already saw these test rows, so nothing here chooses anything.

Per seed (11 seeds):
  HEADLINE    test macro-F1, accuracy, novel-slice macro-F1 -- pooled decision vs neural alone
  SELECTION   the chosen w, and the cross-validated estimate of its cost next to the test cost
  INFLUENCE   concept share of the winning margin; decisions the concepts decide (changed when
              every concept is removed); decisions that differ from the neural classifier, split
              into fixes and breaks
  FAITHFUL    deletion / sufficiency for k = 1, 3, 5 against random concepts
  PER CLASS   F1, fixes/breaks by true class; concept share and deletion by predicted class
  EVIDENCE    rule coverage; evidence-label shares and error rates
  CURVE       the whole weight grid: accuracy, macro-F1, novel, influence, deletion
Across seeds: 95% paired bootstrap CIs over test rows for the seed-averaged differences
(pooled - neural). Seed 42's system is saved as the deployed system, reloaded, and must reproduce
its test predictions.

  python experiments/repair_benchmark_split.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
from repair_core import KS, faithfulness, pooled_pred
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp

OUT = str(config.ARTIFACTS_DIR / "repair_benchmark_split.json")
LAB = list(config.LABELS)
K = cp.K
B = 2000            # bootstrap resamples of the test rows
REPS = 20           # random-deletion repeats at the operating point
CURVE_REPS = 10     # ... and along the weight curve


def f1_per_class(y, p) -> np.ndarray:
    """Per-class F1 from the confusion counts; same definition as cp.macro_f1 (zero_division=0)."""
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1)                                  # 2tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def f1c(y, p) -> float:
    return float(f1_per_class(y, p).mean())


def paired_bootstrap(y, Pp, Pn, rng) -> dict:
    """Seed-averaged metrics and differences with 95% CIs over resampled test rows.
    Pp, Pn: (seeds, N) predictions of the pooled decision and the neural classifier."""
    n, S = len(y), len(Pp)
    stats = np.empty((B, 6))
    for b in range(B):
        i = rng.integers(0, n, n)
        yb = y[i]
        fp = np.mean([f1c(yb, Pp[s, i]) for s in range(S)])
        fn = np.mean([f1c(yb, Pn[s, i]) for s in range(S)])
        ap = float((Pp[:, i] == yb).mean())
        an = float((Pn[:, i] == yb).mean())
        stats[b] = fp, fn, fp - fn, ap, an, ap - an
    fp = np.mean([f1c(y, Pp[s]) for s in range(S)])
    fn = np.mean([f1c(y, Pn[s]) for s in range(S)])
    ap, an = float((Pp == y).mean()), float((Pn == y).mean())
    point = (fp, fn, fp - fn, ap, an, ap - an)
    keys = ("f1_pooled", "f1_neural", "d_f1", "acc_pooled", "acc_neural", "d_acc")
    lo, hi = np.percentile(stats, 2.5, 0), np.percentile(stats, 97.5, 0)
    return {k: {"mean": float(point[j]), "ci": [float(lo[j]), float(hi[j])]} for j, k in enumerate(keys)}


def per_row_deletion(card, Z, ln, w, pred, contrib, rng, k=3, reps=REPS):
    """Per-row: does deleting the decision's top-k concepts change it; how often does deleting k
    random concepts change it; does removing every concept change it."""
    rows = np.arange(len(Z))[:, None]
    top = np.argsort(-contrib, 1)[:, :k]
    Zd = Z.copy()
    Zd[rows, top] = 0.0
    top_ch = pooled_pred(card, Zd, ln, w) != pred
    rnd = np.zeros(len(Z))
    for _ in range(reps):
        rk = np.argsort(rng.random(Z.shape), 1)[:, :k]
        Zr = Z.copy()
        Zr[rows, rk] = 0.0
        rnd += pooled_pred(card, Zr, ln, w) != pred
    none_ch = pooled_pred(card, np.zeros_like(Z), ln, w) != pred
    return top_ch, rnd / reps, none_ch


def main():
    seeds = [int(s) for s in C.SEEDS11]
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    y = d.y[te_i]
    trainval = d.tr | d.va
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)                 # canonical novel slice only
    novel = (u_cap[te_i] @ u_cap[trainval].T).max(1) < 0.90
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    print(f"[r3] test {len(te_i)} rows, novel slice {int(novel.sum())}; building concept bank and "
          f"5-class rules ...", flush=True)
    sym = evp.build_symbolic(d)
    rng = np.random.default_rng(config.SEED)
    per, Pp, Pn, example = [], [], [], None

    for s in seeds:
        t0 = time.time()
        sysm, logits = evp.fit(seed=s, data=d, sym=sym, persist=(s == config.SEED))
        assert sysm.meta["select"] == "cv" and sysm.meta["acc_tol"] == 0.01, sysm.meta
        card, w = sysm.card, sysm.w
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i],
                          neural_logits=logits[te_i])
        pred, pn = out["pred"], logits[te_i].argmax(1)
        Z, ln = out["Z"], out["ln"]
        assert abs(f1c(y, pred) - cp.macro_f1(y, pred)) < 1e-12, "fast F1 disagrees with sklearn"
        ident = np.abs(out["contrib"].sum(1) + out["bias"] - out["margin_concept"]).max()
        assert ident < 1e-6, f"attribution does not sum to the concept margin (max err {ident})"

        # the selection's own estimate of the cost, beside the test cost
        tv = np.flatnonzero(trainval)
        w2, lc_oof = evp.select_weight_cv(d, logits, card.C, s, acc_tol=0.01)
        assert w2 == w, f"CV selection not reproducible ({w2} vs {w})"
        ln_tv = cp.log_softmax(logits[tv])
        p_cv, pn_cv = cp.pooled_scores(lc_oof, ln_tv, w).argmax(1), ln_tv.argmax(1)

        top_ch, rnd_ch, none_ch = per_row_deletion(card, Z, ln, w, pred, out["contrib"], rng)
        fix = (pred == y) & (pn != y)
        brk = (pred != y) & (pn == y)
        rec = {"seed": s, "w": w, "scorecard_C": card.C,
               "f1_neural": f1c(y, pn), "f1_pooled": f1c(y, pred),
               "acc_neural": float((pn == y).mean()), "acc_pooled": float((pred == y).mean()),
               "novel_f1_neural": f1c(y[novel], pn[novel]), "novel_f1_pooled": f1c(y[novel], pred[novel]),
               "cv_est_d_f1": f1c(d.y[tv], p_cv) - f1c(d.y[tv], pn_cv),
               "cv_est_d_acc": float((p_cv == d.y[tv]).mean() - (pn_cv == d.y[tv]).mean()),
               "concept_share": float(out["concept_share"].mean()),
               "concepts_decide": float(none_ch.mean()),
               "changed_vs_neural": float((pred != pn).mean()),
               "fixes": int(fix.sum()), "breaks": int(brk.sum()),
               "attribution_max_err": float(ident),
               "faith": faithfulness(card, Z, ln, w, out, rng, reps=REPS)}
        rec["per_class"] = {LAB[c]: {
            "n_true": int((y == c).sum()), "n_pred": int((pred == c).sum()),
            "f1_neural": float(f1_per_class(y, pn)[c]), "f1_pooled": float(f1_per_class(y, pred)[c]),
            "fixes": int(fix[y == c].sum()), "breaks": int(brk[y == c].sum()),
            "share_pred": float(out["concept_share"][pred == c].mean()) if (pred == c).any() else float("nan"),
            "del_top3_pred": float(top_ch[pred == c].mean()) if (pred == c).any() else float("nan"),
            "del_rand3_pred": float(rnd_ch[pred == c].mean()) if (pred == c).any() else float("nan"),
        } for c in range(K)}
        err = pred != y
        rec["coverage"] = {"any_rule": float(out["fired"].any(1).mean()),
                           "supports_label": float(out["fired"][np.arange(len(pred)), pred].mean())}
        rec["states"] = {st: {"share": float((out["states"] == st).mean()),
                              "err_rate": float(err[out["states"] == st].mean())
                              if (out["states"] == st).any() else float("nan")} for st in cp.STATES}
        rec["curve"] = {}
        for wg in cp.W_GRID:
            dec = cp.decompose(card, Z, ln, wg)
            f = faithfulness(card, Z, ln, wg, dec, rng, reps=CURVE_REPS)
            pg = dec["pred"]
            rec["curve"][str(wg)] = {"f1": f1c(y, pg), "acc": float((pg == y).mean()),
                                     "novel_f1": f1c(y[novel], pg[novel]),
                                     "share": float(dec["concept_share"].mean()),
                                     "decide": 1.0 - f["keep_none_same"],
                                     "del_top3": f[3]["delete_top_changes"],
                                     "del_rand3": f[3]["delete_random_changes"]}

        if s == config.SEED:
            re = evp.load()
            out2 = re.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i])
            rec["reload"] = {"identical_predictions": float((out2["pred"] == pred).mean()),
                             "w": re.w, "select": re.meta.get("select"), "acc_tol": re.meta.get("acc_tol")}
            recs = sysm.records(out)
            pick = [i for i, r_ in enumerate(recs) if r_["evidence"] == "SUPPORTED"
                    and r_["concept_share_of_decision"] > 0.5][:1]
            pick += [i for i, r_ in enumerate(recs) if r_["evidence"] == "CONTESTED"][:1]
            example = [recs[i] | {"true_label": LAB[int(y[i])]} for i in pick]
        per.append(rec)
        Pp.append(pred)
        Pn.append(pn)
        print(f"[r3] seed {s}: {time.time() - t0:.0f}s | w={w} | F1 {rec['f1_neural']:.3f} -> "
              f"{rec['f1_pooled']:.3f} | acc {rec['acc_neural']:.3f} -> {rec['acc_pooled']:.3f} | "
              f"share {rec['concept_share']:.3f} decide {rec['concepts_decide']:.3f} | del-top3 "
              f"{rec['faith'][3]['delete_top_changes']:.3f} vs {rec['faith'][3]['delete_random_changes']:.3f}",
              flush=True)

    Pp, Pn = np.array(Pp), np.array(Pn)
    brng = np.random.default_rng(config.SEED + 1)
    boot = {"all": paired_bootstrap(y, Pp, Pn, brng),
            "novel": paired_bootstrap(y[novel], Pp[:, novel], Pn[:, novel], brng)}
    json.dump({"seeds": seeds, "n_test": int(len(y)), "n_novel": int(novel.sum()), "bootstrap_B": B,
               "per_seed": per, "bootstrap": boot, "example_records": example},
              open(OUT, "w"), indent=1, default=str)
    report(per, boot, example, int(len(y)), int(novel.sum()))
    print(f"\n_saved {OUT}_")


def report(ps, boot, example, n, n_novel):
    g = lambda f: np.array([f(r) for r in ps], float)
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    ci = lambda e: f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    cia = lambda e: f"{e['mean']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
    print(f"\n## Full protocol, deployed operating point ({len(ps)} seeds; test n = {n}, "
          f"novel n = {n_novel}; 95% paired bootstrap CIs over test rows, B = {B})\n")
    print("| metric | neural alone | pooled (deployed) | Δ pooled − neural [95% CI] | seed sd of Δ |")
    print("|---|---|---|---|---|")
    for lab, sl, kf, ka in [("test macro-F1", "all", "f1_neural", "f1_pooled"),
                            ("test accuracy", "all", "acc_neural", "acc_pooled")]:
        b = boot[sl]
        dk = "d_f1" if "f1" in kf else "d_acc"
        print(f"| {lab} | {cia(b[kf])} | {cia(b[ka])} | **{ci(b[dk])}** | "
              f"{np.std(g(lambda r: r[ka] - r[kf])):.3f} |")
    b = boot["novel"]
    print(f"| novel-slice macro-F1 | {cia(b['f1_neural'])} | {cia(b['f1_pooled'])} | **{ci(b['d_f1'])}** | "
          f"{np.std(g(lambda r: r['novel_f1_pooled'] - r['novel_f1_neural'])):.3f} |")

    ws = [r["w"] for r in ps]
    vals, cnt = np.unique(ws, return_counts=True)
    print(f"\n### Operating point\n\nw per seed: {ws} -> " +
          ", ".join(f"w={v}: {c}" for v, c in zip(vals, cnt)) +
          f" | scorecard C: {sorted(set(r['scorecard_C'] for r in ps))}")
    print(f"\nCross-validated estimate of the cost at the chosen w (train+val, out-of-fold) vs test: "
          f"Δ F1 {ms(g(lambda r: r['cv_est_d_f1']))} vs {ms(g(lambda r: r['f1_pooled'] - r['f1_neural']))}; "
          f"Δ acc {ms(g(lambda r: r['cv_est_d_acc']))} vs {ms(g(lambda r: r['acc_pooled'] - r['acc_neural']))}")

    print("\n### Influence of the concepts on the decision\n")
    print("| quantity | value |")
    print("|---|---|")
    print(f"| concept share of the winning margin | {ms(g(lambda r: r['concept_share']))} |")
    print(f"| decisions the concepts decide (change when every concept is removed) | "
          f"{ms(g(lambda r: r['concepts_decide']))} |")
    print(f"| decisions that differ from the neural classifier | {ms(g(lambda r: r['changed_vs_neural']))} |")
    print(f"| of those: fixes / breaks (test rows, mean per seed) | {np.mean(g(lambda r: r['fixes'])):.1f} / "
          f"{np.mean(g(lambda r: r['breaks'])):.1f} |")
    print(f"| attribution identity, max abs error | {np.max(g(lambda r: r['attribution_max_err'])):.1e} |")

    print("\n### Faithfulness at the operating point\n")
    print("| k | delete top-k: changed | delete k random: changed | ratio | keep only top-k: survives | keep k random: survives |")
    print("|---|---|---|---|---|---|")
    for k in KS:
        f = lambda key: g(lambda r: r["faith"][k][key])
        print(f"| {k} | **{ms(f('delete_top_changes'))}** | {ms(f('delete_random_changes'))} | "
              f"{np.mean(f('delete_top_changes')) / max(np.mean(f('delete_random_changes')), 1e-9):.1f}× | "
              f"**{ms(f('keep_top_same'))}** | {ms(f('keep_random_same'))} |")

    print("\n### Per class (mean over seeds)\n")
    print("| class | n test | F1 neural | F1 pooled | Δ | fixes / breaks | predicted as class: concept share | delete top-3 / random-3 |")
    print("|---|---|---|---|---|---|---|---|")
    for c in LAB:
        q = lambda key: np.nanmean(g(lambda r: r["per_class"][c][key]))
        print(f"| {c} | {ps[0]['per_class'][c]['n_true']} | {q('f1_neural'):.3f} | {q('f1_pooled'):.3f} | "
              f"{q('f1_pooled') - q('f1_neural'):+.3f} | {q('fixes'):.1f} / {q('breaks'):.1f} | "
              f"{q('share_pred'):.3f} | {q('del_top3_pred'):.3f} / {q('del_rand3_pred'):.3f} |")

    print("\n### Evidence relative to the pooled decision (5-class rules)\n")
    print(f"Any rule fires: {ms(g(lambda r: r['coverage']['any_rule']))} | a rule supports the label: "
          f"{ms(g(lambda r: r['coverage']['supports_label']))}\n")
    print("| evidence label | share | error rate of the pooled decision |")
    print("|---|---|---|")
    for st in cp.STATES:
        er = [x for x in g(lambda r: r["states"][st]["err_rate"]) if x == x]
        print(f"| {st} | {ms(g(lambda r: r['states'][st]['share']))} | {ms(er) if er else '-'} |")

    print("\n### Influence-vs-accuracy curve (mean over seeds; w = 1 is the neural classifier)\n")
    print("| w | test macro-F1 | test acc | novel F1 | concept share | concepts decide | delete top-3 / random-3 |")
    print("|---|---|---|---|---|---|---|")
    for wg in cp.W_GRID:
        q = lambda key: np.mean(g(lambda r: r["curve"][str(wg)][key]))
        print(f"| {wg} | {q('f1'):.3f} | {q('acc'):.3f} | {q('novel_f1'):.3f} | {q('share'):.3f} | "
              f"{q('decide'):.3f} | {q('del_top3'):.3f} / {q('del_rand3'):.3f} |")

    r42 = next(r for r in ps if "reload" in r)
    print(f"\nPersistence (seed {r42['seed']}): reloaded system reproduces test predictions "
          f"{r42['reload']['identical_predictions']:.3f}; saved w = {r42['reload']['w']}, select = "
          f"{r42['reload']['select']}, acc_tol = {r42['reload']['acc_tol']}")
    if example:
        print("\n### Example evidence records (seed 42; concept and rule names only)\n")
        for e in example:
            print(json.dumps(e, indent=1, default=str))


if __name__ == "__main__":
    main()
