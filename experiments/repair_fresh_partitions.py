"""Robust operating-point selection, confirmed on fresh partitions.

Single-split selection failed its pre-registered confirmation: choosing the pooling weight w on one ~480-row
validation split (rule B: within 0.01 of the neural classifier on both macro-F1 and accuracy)
kept selecting concept-heavy weights that cost ~2 points out of sample (-0.017 F1, -0.024 acc).

Two candidate causes, both addressed by the pre-registered selection below:

  NOISE   ~480 validation rows move by more than the 0.01 tolerance between splits.
          -> choose w on ALL train+val rows (~2,357 per fresh partition) by 5-fold
             cross-validation; both experts' predictions are out-of-fold.
  LEAK    concept mining ranks phrases by per-class log-odds over train+val, i.e. it uses the
          validation LABELS; a scorecard scored on rows that mined its own concepts looks better
          than it is, which pushes w toward the concepts.
          -> rebuild the concept bank inside each fold, from that fold's training rows only.

Three selections are compared on identical test sets:
  single-split   the single-split rule on the validation split (reference; known to fail)
  CV, clean      PRE-REGISTERED: 5-fold CV over train+val, concept bank rebuilt per fold
  CV, leaky      diagnostic: same CV, but the bank mined on all train+val (measures the leak)

PRE-REGISTERED CRITERION (unchanged): CONFIRMED if, across the same 5 fresh
partitions (original test rows excluded), the mean test delta (pooled - neural) for "CV, clean"
is >= -0.01 on BOTH macro-F1 and accuracy. The original split is reported for continuity only;
its test set has been seen and cannot confirm anything.

  python experiments/repair_fresh_partitions.py
"""
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import train_test_split

import common as C
from repair_core import faithfulness
from repair_operating_point_study import point
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp

OUT = str(config.ARTIFACTS_DIR / "repair_fresh_partitions.json")
PARTITION_SEEDS = (1, 2, 3, 4, 5)
ACC_TOL = 0.01
CRITERION = -0.01
SELECTIONS = ("single-split", "CV, clean", "CV, leaky")


def partitions():
    """('original', data) then the 5 fresh partitions exactly as the confirmation run built them."""
    d0 = evp.load_data()
    yield "original", 42, d0
    pool = np.flatnonzero(d0.tr | d0.va)
    for i, ps in enumerate(PARTITION_SEEDS):
        yv = d0.y[pool]
        idx = np.arange(len(pool))
        tri, tmp = train_test_split(idx, test_size=0.30, stratify=yv, random_state=ps)
        vai, tei = train_test_split(tmp, test_size=0.50, stratify=yv[tmp], random_state=ps)
        m = lambda a: np.isin(idx, a)
        d = SimpleNamespace(df=d0.df.iloc[pool].reset_index(drop=True), v=d0.v[pool],
                            u=d0.u[pool], y=yv, tr=m(tri), va=m(vai), te=m(tei),
                            text=d0.text.iloc[pool].reset_index(drop=True), rows=pool)
        yield ps, int(C.SEEDS11[i]), d


def main():
    u_cap_all = np.load(config.TEXT_CAPTION_EMB_PATH)           # novel-slice definition only
    rng = np.random.default_rng(config.SEED)
    per = []
    for name, tseed, d in partitions():
        t0 = time.time()
        te_i = np.flatnonzero(d.te)
        uc = u_cap_all[getattr(d, "rows", np.arange(len(d.y)))]
        novel = (uc[te_i] @ uc[d.tr | d.va].T).max(1) < 0.90
        sym = evp.build_symbolic(d)
        S = sym[1].to_numpy(np.float64)
        sysm, logits = evp.fit(seed=tseed, data=d, sym=sym, select="val", acc_tol=None)  # w chosen below
        card = sysm.card
        ln = cp.log_softmax(logits)
        lc = cp.log_softmax(card.logits(S))
        pn = logits[te_i].argmax(1)
        neural = {"test_f1": cp.macro_f1(d.y[te_i], pn), "test_acc": float((pn == d.y[te_i]).mean()),
                  "novel_f1": cp.macro_f1(d.y[te_i][novel], pn[novel])}
        ws = {"single-split": cp.select_weight(lc[d.va], ln[d.va], d.y[d.va], acc_tol=ACC_TOL)}
        ws["CV, clean"], _ = evp.select_weight_cv(d, logits, card.C, tseed, acc_tol=ACC_TOL,
                                                  rebuild_bank=True)
        ws["CV, leaky"], _ = evp.select_weight_cv(d, logits, card.C, tseed, acc_tol=ACC_TOL,
                                                  rebuild_bank=False, S_full=S)
        rec = {"partition": name, "train_seed": tseed, "n_test": int(d.te.sum()), "neural": neural}
        for sel, w in ws.items():
            mt, dec, Z = point(card, S, ln, d.y, d.va, te_i, novel, w)
            mt.update(w=w, d_f1=mt["test_f1"] - neural["test_f1"],
                      d_acc=mt["test_acc"] - neural["test_acc"],
                      d_novel=mt["novel_f1"] - neural["novel_f1"])
            if sel == "CV, clean":
                f = faithfulness(card, Z, ln[te_i], w, dec, rng)
                mt.update(del_top3=f[3]["delete_top_changes"], del_rand3=f[3]["delete_random_changes"],
                          keep_none=f["keep_none_same"])
            rec[sel] = mt
        per.append(rec)
        c = rec["CV, clean"]
        print(f"[fresh-partitions] partition {name}: {time.time() - t0:.0f}s | neural F1 {neural['test_f1']:.3f} "
              f"acc {neural['test_acc']:.3f} | w single={ws['single-split']} cv-clean={ws['CV, clean']} "
              f"cv-leaky={ws['CV, leaky']} | cv-clean dF1 {c['d_f1']:+.3f} dAcc {c['d_acc']:+.3f}",
              flush=True)

    fresh = [r for r in per if r["partition"] != "original"]
    mean = lambda sel, k: float(np.mean([r[sel][k] for r in fresh]))
    confirmed = mean("CV, clean", "d_f1") >= CRITERION and mean("CV, clean", "d_acc") >= CRITERION
    verdict = {"criterion": f"mean test delta >= {CRITERION} on both metrics, 5 fresh partitions",
               "cv_clean_mean_d_f1": mean("CV, clean", "d_f1"),
               "cv_clean_mean_d_acc": mean("CV, clean", "d_acc"), "confirmed": bool(confirmed)}
    json.dump({"per_partition": per, "verdict": verdict}, open(OUT, "w"), indent=1, default=str)
    report(per, verdict)
    print(f"\n_saved {OUT}_")


def report(per, verdict):
    fmt = lambda v: f"{np.mean(v):+.3f} ± {np.std(v):.3f}"
    print("\n## Cross-validated operating-point selection (the same 5 fresh partitions)\n")
    print("| partition | neural F1 / acc | w: single / CV-clean / CV-leaky | CV-clean Δ F1 | Δ acc | Δ novel | single Δ F1 / Δ acc | leaky Δ F1 / Δ acc |")
    print("|---|---|---|---|---|---|---|---|")
    for r in per:
        n, c, s, l = r["neural"], r["CV, clean"], r["single-split"], r["CV, leaky"]
        tag = f"{r['partition']}" + (" *(seen; not counted)*" if r["partition"] == "original" else "")
        print(f"| {tag} | {n['test_f1']:.3f} / {n['test_acc']:.3f} | {s['w']} / {c['w']} / {l['w']} | "
              f"{c['d_f1']:+.3f} | {c['d_acc']:+.3f} | {c['d_novel']:+.3f} | "
              f"{s['d_f1']:+.3f} / {s['d_acc']:+.3f} | {l['d_f1']:+.3f} / {l['d_acc']:+.3f} |")
    fresh = [r for r in per if r["partition"] != "original"]
    print("\nOver the 5 fresh partitions:\n")
    for sel in SELECTIONS:
        g = lambda k: [r[sel][k] for r in fresh]
        print(f"* **{sel}**: mean w {np.mean(g('w')):.2f} | Δ macro-F1 {fmt(g('d_f1'))} | Δ accuracy "
              f"{fmt(g('d_acc'))} | Δ novel {fmt(g('d_novel'))} | concept share {np.mean(g('share')):.3f}")
    g = lambda k: [r["CV, clean"][k] for r in fresh]
    print(f"\nFaithfulness at the CV-clean point: delete top-3 changes {np.mean(g('del_top3')):.3f} vs "
          f"random {np.mean(g('del_rand3')):.3f}; unchanged with no concepts {np.mean(g('keep_none')):.3f}")
    print(f"\n### Verdict (pre-registered): {verdict['criterion']}\n")
    print(f"CV, clean: Δ F1 {verdict['cv_clean_mean_d_f1']:+.3f}, Δ acc {verdict['cv_clean_mean_d_acc']:+.3f}"
          f" -> **{'CONFIRMED' if verdict['confirmed'] else 'NOT CONFIRMED'}**")


if __name__ == "__main__":
    main()
