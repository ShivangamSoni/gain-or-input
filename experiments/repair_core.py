"""Core of the concept-routed decision, built into the package and tested for
faithfulness.

Evaluates nesymis/concept_routed.py (the combined paper's system) on the corrected benchmark,
same OCR text to both paths. Checks, per seed:

  REPRODUCE     pooled test macro-F1, concept agreement and concept share should match the
                step-1 measurement (experiments/evidence_feasibility.py), since the package reimplements
                the same pooling -- this is the build's correctness check.
  ATTRIBUTION   the per-concept contributions plus the bias term must sum exactly to the
                concept part of every winning margin.
  DELETION      remove the top-k concepts behind each decision (set to their training mean) and
                count decisions that change; compare with removing k random concepts. If the
                named concepts matter, top-k removal changes far more decisions.
  SUFFICIENCY   keep ONLY the top-k concepts: how often does the decision survive? vs k random.
  EVIDENCE      rule coverage and the evidence label relative to the pooled decision.
  PERSISTENCE   moved to experiments/repair_benchmark_split.py, which saves the deployed (CV) system; this
                script pins the earlier operating-point rule and no longer writes the saved system.

  python experiments/repair_core.py            # 5 seeds, same as step 1
  python experiments/repair_core.py --seeds 11
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
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp

OUT = str(config.ARTIFACTS_DIR / "repair_core.json")
KS = (1, 3, 5)
LAB = list(config.LABELS)


def pooled_pred(card, Z, ln, w):
    return cp.pooled_scores(cp.log_softmax(card.logits(Z=Z)), ln, w).argmax(1)


def faithfulness(card, Z, ln, w, out, rng, reps=20):
    pred, contrib = out["pred"], out["contrib"]
    N, J = Z.shape
    rows = np.arange(N)[:, None]
    order = np.argsort(-contrib, 1)
    res = {"keep_none_same": float((pooled_pred(card, np.zeros_like(Z), ln, w) == pred).mean())}
    for k in KS:
        top = order[:, :k]
        Zd = Z.copy()
        Zd[rows, top] = 0.0
        Zk = np.zeros_like(Z)
        Zk[rows, top] = Z[rows, top]
        d_rand, k_rand = [], []
        for _ in range(reps):
            rk = np.argsort(rng.random((N, J)), 1)[:, :k]
            Zr = Z.copy()
            Zr[rows, rk] = 0.0
            d_rand.append(float((pooled_pred(card, Zr, ln, w) != pred).mean()))
            Zs = np.zeros_like(Z)
            Zs[rows, rk] = Z[rows, rk]
            k_rand.append(float((pooled_pred(card, Zs, ln, w) == pred).mean()))
        res[k] = {"delete_top_changes": float((pooled_pred(card, Zd, ln, w) != pred).mean()),
                  "delete_random_changes": float(np.mean(d_rand)),
                  "keep_top_same": float((pooled_pred(card, Zk, ln, w) == pred).mean()),
                  "keep_random_same": float(np.mean(k_rand))}
    return res


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}+-{v.std():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    seeds = [int(s) for s in C.SEEDS11[:a.seeds]]

    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    y_te = d.y[te_i]
    trainval = d.tr | d.va
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)                 # canonical novel slice only
    novel = (u_cap[te_i] @ u_cap[trainval].T).max(1) < 0.90
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)

    print("[r1] building concept bank and 5-class rules on the corrected OCR ...", flush=True)
    sym = evp.build_symbolic(d)
    rng = np.random.default_rng(config.SEED)
    per_seed, example = [], None

    for s in seeds:
        # the earlier configuration, pinned: the package default is now the cross-validated operating point,
        # and the deployed system is saved by experiments/repair_benchmark_split.py, not here
        sysm, logits = evp.fit(seed=s, data=d, sym=sym, select="val", acc_tol=None)
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i],
                          neural_logits=logits[te_i])
        pred, pn = out["pred"], logits[te_i].argmax(1)
        ident = np.abs(out["contrib"].sum(1) + out["bias"] - out["margin_concept"]).max()
        assert ident < 1e-6, f"attribution does not sum to the concept margin (max err {ident})"

        rec = {"seed": s, "w": sysm.w, "scorecard_C": sysm.card.C,
               "f1_neural": cp.macro_f1(y_te, pn), "f1_pooled": cp.macro_f1(y_te, pred),
               "acc_neural": float((pn == y_te).mean()), "acc_pooled": float((pred == y_te).mean()),
               "novel_f1_neural": cp.macro_f1(y_te[novel], pn[novel]),
               "novel_f1_pooled": cp.macro_f1(y_te[novel], pred[novel]),
               "concept_agree": float((out["lc"].argmax(1) == pred).mean()),
               "concept_share": float(out["concept_share"].mean()),
               "changed_vs_neural": float((pred != pn).mean()),
               "attribution_max_err": float(ident)}
        rec["faith"] = faithfulness(sysm.card, out["Z"], out["ln"], sysm.w, out, rng)
        err = (pred != y_te)
        rec["coverage"] = {"any_rule": float(out["fired"].any(1).mean()),
                           "supports_label": float(out["fired"][np.arange(len(pred)), pred].mean())}
        rec["states"] = {st: {"share": float((out["states"] == st).mean()),
                              "err_rate": float(err[out["states"] == st].mean())
                              if (out["states"] == st).any() else float("nan")}
                         for st in cp.STATES}

        if s == config.SEED:
            live = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i])
            rec["live_vs_crossfit_logits_identical"] = float((live["pred"] == pred).mean())
            recs = sysm.records(out)
            pick = [i for i, r_ in enumerate(recs) if r_["evidence"] == "SUPPORTED"
                    and r_["concept_share_of_decision"] > 0.5][:1]
            pick += [i for i, r_ in enumerate(recs) if r_["evidence"] == "CONTESTED"][:1]
            example = [recs[i] | {"true_label": LAB[int(y_te[i])]} for i in pick]
        per_seed.append(rec)
        print(f"[r1] seed {s}: w={sysm.w} neural {rec['f1_neural']:.3f} pooled {rec['f1_pooled']:.3f} | "
              f"agree {rec['concept_agree']:.3f} share {rec['concept_share']:.3f} | "
              f"delete-top3 changes {rec['faith'][3]['delete_top_changes']:.3f} "
              f"vs random {rec['faith'][3]['delete_random_changes']:.3f}", flush=True)

    json.dump({"seeds": seeds, "per_seed": per_seed, "example_records": example},
              open(OUT, "w"), indent=1, default=str)
    report(per_seed, example)
    print(f"\n_saved {OUT}_")


def report(ps, example):
    g = lambda f: [f(r) for r in ps]
    print(f"\n## The concept-routed decision (nesymis/concept_routed.py), {len(ps)} seeds\n")
    print("| quantity | value |")
    print("|---|---|")
    for lab, f in [("pooling weight w (validation-selected)", lambda r: r["w"]),
                   ("test macro-F1, neural alone", lambda r: r["f1_neural"]),
                   ("**test macro-F1, pooled decision**", lambda r: r["f1_pooled"]),
                   ("test accuracy, neural / pooled", None),
                   ("novel-slice macro-F1, neural", lambda r: r["novel_f1_neural"]),
                   ("novel-slice macro-F1, pooled", lambda r: r["novel_f1_pooled"]),
                   ("concept part alone makes the same call", lambda r: r["concept_agree"]),
                   ("concept share of the winning margin", lambda r: r["concept_share"]),
                   ("decisions that differ from the neural classifier", lambda r: r["changed_vs_neural"]),
                   ("attribution identity, max abs error", lambda r: r["attribution_max_err"])]:
        if f is None:
            print(f"| {lab} | {ms(g(lambda r: r['acc_neural']))} / {ms(g(lambda r: r['acc_pooled']))} |")
        else:
            print(f"| {lab} | {ms(g(f))} |")

    print("\n### Faithfulness: does removing the concepts behind a decision change it?\n")
    print("| k | delete top-k: decisions changed | delete k random: changed | keep only top-k: decision survives | keep k random: survives |")
    print("|---|---|---|---|---|")
    for k in KS:
        f = lambda key: ms(g(lambda r: r["faith"][k][key]))
        print(f"| {k} | **{f('delete_top_changes')}** | {f('delete_random_changes')} | "
              f"**{f('keep_top_same')}** | {f('keep_random_same')} |")
    print(f"\nKeep no concepts at all (concept part = its class prior): decision survives "
          f"{ms(g(lambda r: r['faith']['keep_none_same']))}.")

    print("\n### Evidence relative to the pooled decision (5-class rules)\n")
    print(f"Any rule fires: {ms(g(lambda r: r['coverage']['any_rule']))} | a rule supports the label: "
          f"{ms(g(lambda r: r['coverage']['supports_label']))}\n")
    print("| evidence label | share | error rate of the pooled decision |")
    print("|---|---|---|")
    for st in cp.STATES:
        er = [x for x in g(lambda r: r["states"][st]["err_rate"]) if x == x]
        print(f"| {st} | {ms(g(lambda r: r['states'][st]['share']))} | {ms(er) if er else '-'} |")

    r42 = ps[0]
    print(f"\nSeed {r42['seed']}: live neural logits vs cross-fit test logits give identical "
          f"decisions on {r42.get('live_vs_crossfit_logits_identical', float('nan')):.3f} of test rows "
          f"(save/reload is checked in experiments/repair_benchmark_split.py).")
    if example:
        print("\n### Example evidence records (concept and rule names only)\n")
        for e in example:
            print(json.dumps(e, indent=1, default=str))


if __name__ == "__main__":
    main()
