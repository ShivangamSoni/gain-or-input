"""Parity on accuracy as well as macro-F1.

The first pooled run built the concept-routed decision and showed it faithful, but at the validation-selected
operating point accuracy fell 0.780 -> 0.753 while macro-F1 held (0.633 vs 0.640). Two
candidate causes, each with its own fix:

  the RULE      the operating point was constrained on macro-F1 only, so it could trade
                majority-class accuracy for minority recall
                -> rule B: the blend must stay within 0.01 of the neural classifier on
                   validation for BOTH macro-F1 and accuracy
  the PRIOR     the scorecard is trained class-balanced, i.e. under a uniform class prior,
                which pulls it toward the small classes
                -> prior correction: shift its log-probabilities back to the training class
                   frequencies (weights and attributions unchanged; only the offset moves)

Both variants (balanced / prior-corrected) x both rules (A: macro-F1 only / B: macro-F1 and
accuracy), with the full weight sweep reported as a trade-off curve and the faithfulness test
repeated at every selected operating point.

PRE-REGISTERED DECISIONS -- made from validation only, before test is read:
  * Rule B is adopted: the paper claims parity on both metrics.
  * Prior correction is adopted iff, under rule B, its mean validation-selected w across seeds
    is smaller than the balanced scorecard's (the concepts carry more of the decision at
    validated parity). Ties keep the balanced scorecard (no extra step).

  python experiments/repair_operating_point_study.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
from repair_core import faithfulness
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp

OUT = str(config.ARTIFACTS_DIR / "repair_operating_point_study.json")
ACC_TOL = 0.01
RULES = {"A: macro-F1 only": None, "B: macro-F1 and accuracy": ACC_TOL}
VARIANTS = ("balanced", "prior-corrected")


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}+-{v.std():.3f}"


def point(card, S, ln, y, va, te_i, novel, w):
    """All metrics for one operating point."""
    lc = cp.log_softmax(card.logits(S))
    pv = cp.pooled_scores(lc[va], ln[va], w).argmax(1)
    Z = card.standardise(S[te_i])
    dec = cp.decompose(card, Z, ln[te_i], w)
    pt, yt = dec["pred"], y[te_i]
    return {"val_f1": cp.macro_f1(y[va], pv), "val_acc": float((pv == y[va]).mean()),
            "test_f1": cp.macro_f1(yt, pt), "test_acc": float((pt == yt).mean()),
            "novel_f1": cp.macro_f1(yt[novel], pt[novel]),
            "agree": float((dec["lc"].argmax(1) == pt).mean()),
            "share": float(dec["concept_share"].mean())}, dec, Z


def main():
    seeds = [int(s) for s in C.SEEDS11[:5]]
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    trainval = d.tr | d.va
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)                 # canonical novel slice only
    novel = (u_cap[te_i] @ u_cap[trainval].T).max(1) < 0.90
    print("[operating-point] building concept bank and 5-class rules ...", flush=True)
    sym = evp.build_symbolic(d)
    S = sym[1].to_numpy(np.float64)
    rng = np.random.default_rng(config.SEED)
    per_seed = []

    for s in seeds:
        sysm, logits = evp.fit(seed=s, data=d, sym=sym, select="val", acc_tol=None)  # w chosen below
        ln = cp.log_softmax(logits)
        pn = logits[te_i].argmax(1)
        rec = {"seed": s, "neural": {"test_f1": cp.macro_f1(d.y[te_i], pn),
                                     "test_acc": float((pn == d.y[te_i]).mean()),
                                     "novel_f1": cp.macro_f1(d.y[te_i][novel], pn[novel]),
                                     "val_f1": cp.macro_f1(d.y[d.va], logits[d.va].argmax(1)),
                                     "val_acc": float((logits[d.va].argmax(1) == d.y[d.va]).mean())}}
        cards = {"balanced": sysm.card, "prior-corrected": sysm.card.with_prior_correction(d.y, d.tr)}
        for vname, card in cards.items():
            lc = cp.log_softmax(card.logits(S))
            rec[vname] = {"sweep": {}, "selected": {}}
            for w in cp.W_GRID:
                rec[vname]["sweep"][str(w)] = point(card, S, ln, d.y, d.va, te_i, novel, w)[0]
            for rname, tol in RULES.items():
                w = cp.select_weight(lc[d.va], ln[d.va], d.y[d.va], acc_tol=tol)
                m, dec, Z = point(card, S, ln, d.y, d.va, te_i, novel, w)
                f = faithfulness(card, Z, ln[te_i], w, dec, rng)
                m.update(w=w, del_top3=f[3]["delete_top_changes"], del_rand3=f[3]["delete_random_changes"],
                         keep_top3=f[3]["keep_top_same"], keep_rand3=f[3]["keep_random_same"],
                         keep_none=f["keep_none_same"])
                rec[vname]["selected"][rname] = m
        per_seed.append(rec)
        b = rec["balanced"]["selected"]["B: macro-F1 and accuracy"]
        p = rec["prior-corrected"]["selected"]["B: macro-F1 and accuracy"]
        print(f"[operating-point] seed {s}: neural F1 {rec['neural']['test_f1']:.3f} acc {rec['neural']['test_acc']:.3f} | "
              f"rule B balanced w={b['w']} F1 {b['test_f1']:.3f} acc {b['test_acc']:.3f} | "
              f"prior w={p['w']} F1 {p['test_f1']:.3f} acc {p['test_acc']:.3f}", flush=True)

    # ---- pre-registered decision, validation only ----
    rb = "B: macro-F1 and accuracy"
    w_bal = float(np.mean([r["balanced"]["selected"][rb]["w"] for r in per_seed]))
    w_pri = float(np.mean([r["prior-corrected"]["selected"][rb]["w"] for r in per_seed]))
    choice = "prior-corrected" if w_pri < w_bal else "balanced"
    decision = {"rule": rb, "acc_tol": ACC_TOL, "mean_val_selected_w": {"balanced": w_bal,
                "prior-corrected": w_pri}, "variant": choice}
    json.dump({"seeds": seeds, "per_seed": per_seed, "decision": decision}, open(OUT, "w"), indent=1)
    report(per_seed, decision)
    print(f"\n_saved {OUT}_")


def report(ps, decision):
    g = lambda f: [f(r) for r in ps]
    nf = lambda k: ms(g(lambda r: r["neural"][k]))
    print(f"\n## Parity on accuracy as well as macro-F1 ({len(ps)} seeds)\n")
    print(f"Neural classifier alone: test F1 {nf('test_f1')} | test acc {nf('test_acc')} | "
          f"novel F1 {nf('novel_f1')}\n")
    for v in VARIANTS:
        print(f"### Trade-off curve, {v} scorecard (mean over seeds)\n")
        print("| w | val F1 | val acc | test F1 | test acc | novel F1 | concept part alone agrees | concept share |")
        print("|---|---|---|---|---|---|---|---|")
        for w in cp.W_GRID:
            q = lambda k: np.mean(g(lambda r: r[v]["sweep"][str(w)][k]))
            print(f"| {w} | {q('val_f1'):.3f} | {q('val_acc'):.3f} | {q('test_f1'):.3f} | "
                  f"{q('test_acc'):.3f} | {q('novel_f1'):.3f} | {q('agree'):.3f} | {q('share'):.3f} |")
        print()
    print("### Selected operating points (selection on validation; metrics on test)\n")
    print("| scorecard | rule | w per seed | test F1 | test acc | novel F1 | agrees | share | delete top-3 / random-3 | keep top-3 / random-3 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        for rn in RULES:
            q = lambda k: ms(g(lambda r: r[v]["selected"][rn][k]))
            ws = [r[v]["selected"][rn]["w"] for r in ps]
            print(f"| {v} | {rn} | {ws} | {q('test_f1')} | {q('test_acc')} | {q('novel_f1')} | "
                  f"{q('agree')} | {q('share')} | {q('del_top3')} / {q('del_rand3')} | "
                  f"{q('keep_top3')} / {q('keep_rand3')} |")
    print(f"\n### Pre-registered decision (validation only)\n")
    print(f"Rule B adopted. Mean validation-selected w under rule B: balanced "
          f"{decision['mean_val_selected_w']['balanced']:.2f}, prior-corrected "
          f"{decision['mean_val_selected_w']['prior-corrected']:.2f} -> **{decision['variant']}** scorecard.")


if __name__ == "__main__":
    main()
