"""Feasibility check: "symbolic evidence on every output, at neural parity".

Everything runs on the CORRECTED benchmark under the official protocol: both paths get the
same uniform-pipeline OCR text; the clean caption is never used. Four questions, each of which
the EAAI design depends on:

  1. COVERAGE    What fraction of outputs carry rule evidence at all, and does it agree with
                 the decision? Measured with the deployed 4-class rules (non_stereotype has no
                 rule, by design) and with a 5-class extension that induces rules for it too.
  2. INFLUENCE   Do the rules actually change decisions? The learned rule weights, and the
                 number of test decisions that flip when rule evidence is removed -- separated
                 from flips caused by the decision layer's unconstrained term over the neural
                 probabilities, which can also move a decision.
  3. DEFERRAL    The proposed measurable advantage. At a fixed human-review budget, does
                 symbolic evidence catch neural errors that the classifier's own confidence
                 misses? Error-detection AUROC, errors caught at 10% / 20% budgets, and the
                 accuracy of what is left. Combinations use rank-averaging with equal weights,
                 fixed in advance, so nothing is tuned on the evaluation rows.
  4. HYBRID      Log-linear pooling of a concept scorecard with the neural classifier,
                 final = (1-w) log p_concept + w log p_neural. How concept-driven can the
                 decision be while staying at neural accuracy? The operating point is chosen on
                 validation (the scorecard is trained on train only and the neural logits for
                 validation rows are out-of-fold, so the choice uses held-out predictions).

The decision under test is the neural classifier's: the design keeps it as the decision-maker,
so accuracy is neural by construction and the evidence is judged on what it adds to it.
5 seeds (common.SEEDS11[:5]); rules and scorecards do not depend on the seed, the neural
classifier and the decision layer do.

  python experiments/evidence_feasibility.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import uniform_ocr_protocol as cb
from nesymis import config
from nesymis.fusion.decision_layer import LinearEvidencePolicy
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import rule_induction as ri

OUT = str(config.ARTIFACTS_DIR / "evidence_feasibility.json")
SEEDS = [int(s) for s in C.SEEDS11[:5]]
LAB = list(config.LABELS)
ST = list(config.STEREO_CLASSES)
ST_IDX = [config.LABEL_TO_IDX[c] for c in ST]
K = config.NUM_CLASSES
BUDGETS = (0.10, 0.20)
CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
WS = [round(w, 1) for w in np.linspace(0.0, 1.0, 11)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STATES = ["SUPPORTED", "MIXED", "CONTESTED", "NONE"]


def mf1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0))


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fire(rs, Lm, classes):
    """(N, len(classes)) fired flags and best-fired-clause precision, per class."""
    N = Lm.shape[0]
    fired = np.zeros((N, len(classes)), bool)
    trig = np.zeros((N, len(classes)), np.float32)
    for j, c in enumerate(classes):
        for clause in rs.get(c, []):
            f = np.all(Lm[:, clause["idx"]], axis=1)
            fired[:, j] |= f
            trig[:, j] = np.where(f, np.maximum(trig[:, j], clause["precision"]), trig[:, j])
    return fired, trig


def to5(fired, trig, classes):
    """Scatter per-rule-class columns into the 5-class label space (absent classes = False)."""
    f5 = np.zeros((fired.shape[0], K), bool)
    t5 = np.zeros((fired.shape[0], K), np.float32)
    for j, c in enumerate(classes):
        f5[:, config.LABEL_TO_IDX[c]] = fired[:, j]
        t5[:, config.LABEL_TO_IDX[c]] = trig[:, j]
    return f5, t5


def evidence(pred, f5, t5):
    """Rule evidence relative to a decision: state per row, and a suspicion score
    (best contesting precision minus best supporting precision; 0 when nothing fires)."""
    n = len(pred)
    r = np.arange(n)
    sup = f5[r, pred]
    other = f5.copy()
    other[r, pred] = False
    con = other.any(1)
    sup_p = np.where(sup, t5[r, pred], 0.0)
    t_o = np.where(other, t5, 0.0)
    con_p = t_o.max(1)
    state = np.where(sup & con, "MIXED",
                     np.where(sup, "SUPPORTED", np.where(con, "CONTESTED", "NONE")))
    return state, (con_p - sup_p).astype(np.float64)


def scorecard(S, y, tr, va, penalty):
    """Class-weighted logistic regression on the concept scores, C chosen on validation.
    Trained on TRAIN only, so its validation predictions are held-out."""
    sc = StandardScaler().fit(S[tr])
    Xs = sc.transform(S)
    solver, it = ("saga", 8000) if penalty == "l1" else ("lbfgs", 3000)
    best = None
    for c in CS:
        m = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=it,
                               class_weight="balanced", random_state=config.SEED)
        m.fit(Xs[tr], y[tr])
        f = mf1(y[va], m.predict(Xs[va]))
        if best is None or f > best[0]:
            best = (f, c, m)
    _, c, m = best
    assert list(m.classes_) == list(range(K))
    return m.predict_proba(Xs), c, int((np.abs(m.coef_) > 1e-6).sum())


def rnorm(x):
    r = rankdata(x)
    return (r - 1.0) / max(len(x) - 1, 1)


def caught(score, err, b, rng, reps=50):
    """Fraction of errors inside the top-b most suspicious items, and accuracy of the rest.
    Ties (rule evidence has many) are broken at random and averaged over `reps` draws."""
    n = len(score)
    m = int(np.ceil(b * n))
    tot = max(int(err.sum()), 1)
    c, a = [], []
    for _ in range(reps):
        order = np.lexsort((rng.random(n), -score))
        top = np.zeros(n, bool)
        top[order[:m]] = True
        c.append(err[top].sum() / tot)
        a.append(1.0 - err[~top].mean())
    return float(np.mean(c)), float(np.mean(a))


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}+-{v.std():.3f}"


def main():
    df, y, v, u_ocr, u_cap, tr, va, te = cb.load_corrected()
    trainval = tr | va
    te_i = np.flatnonzero(te)
    y_te = y[te_i]
    n_te = len(te_i)
    txt = df["text_ocr"].fillna("").astype(str)

    # ---- symbolic side: concept bank + rules on the corrected OCR channel -----------------
    df2 = df.copy()
    df2["text_caption"] = txt.values
    bank = cl.build(SimpleNamespace(df=df2, text_caption=u_ocr, image=v), trainval)
    cont = bank.frame(u_ocr, v, index=df.index)
    rs4, Lm, _ = ri.induce(cont, y, trainval, affect=None, save=False)           # deployed
    rs5, Lm5, _ = ri.induce(cont, y, trainval, affect=None, classes=LAB, save=False)
    assert Lm5.shape == Lm.shape
    fired4, trig4 = fire(rs4, Lm, ST)
    fired5, trig5 = fire(rs5, Lm5, LAB)
    f45, t45 = to5(fired4, trig4, ST)
    rvec = np.array([((y[trainval] == config.LABEL_TO_IDX[c]) & fired4[trainval, j]).sum()
                     / max(int(fired4[trainval, j].sum()), 1) for j, c in enumerate(ST)],
                    np.float32)
    n_clauses4 = {c: len(rs4.get(c, [])) for c in ST}
    n_clauses5 = {c: len(rs5.get(c, [])) for c in LAB}
    print(f"[step1] concepts={cont.shape[1]} literals={Lm.shape[1]} "
          f"clauses4={n_clauses4} clauses5={n_clauses5} rvec={np.round(rvec, 3)}", flush=True)

    # ---- concept scorecards (seed-independent) ---------------------------------------------
    S = cont.to_numpy().astype(np.float32)
    P2, c2, nnz2 = scorecard(S, y, tr, va, "l2")
    P1, c1, nnz1 = scorecard(S, y, tr, va, "l1")
    print(f"[step1] scorecards: L2 C={c2} test F1 {mf1(y_te, P2[te_i].argmax(1)):.3f} | "
          f"L1 C={c1} nnz={nnz1} test F1 {mf1(y_te, P1[te_i].argmax(1)):.3f}", flush=True)

    vu = np.concatenate([v, u_ocr], 1).astype(np.float32)
    D = np.zeros((len(y), config.AFFECT_DIM), np.float32)
    cfg = dict(config.MLP)
    rng = np.random.default_rng(config.SEED)
    per_seed = []

    for s in SEEDS:
        clf, logits = clf.crossfit_logits(vu, D, y, tr, va, fusion=config.NEURAL_FUSION,
                                         affect_dim=C.odesign.AFFECT_DIM, cfg=cfg, seed=s)
        state = C.rl.build_state(logits, fired4.astype(np.float32), rvec, D)
        pol = C.rl.train_rlvr(logits, state, y, tr, va, seed=s,
                              policy_ctor=lambda d: LinearEvidencePolicy(d))[0]
        pol.eval()
        NL = logits[te_i]
        st_te = state[te_i]
        with torch.no_grad():
            t = torch.from_numpy(st_te).float().to(DEV)
            free = pol.free(t[:, pol.idx_free]).cpu().numpy()
        w_f, w_p = pol.rule_weights()
        R = len(ST)
        contrib = st_te[:, K + 1:K + 1 + R] * w_f + st_te[:, K + 1 + R:K + 1 + 2 * R] * w_p
        rule_add = np.zeros((n_te, K), np.float32)
        rule_add[:, ST_IDX] = contrib
        p_neu = NL.argmax(1)
        p_free = (NL + free).argmax(1)
        p_full = (NL + free + rule_add).argmax(1)
        p_gp = C.rl.greedy_pred(pol, NL, st_te)
        assert (p_gp == p_full).mean() > 0.999, "decomposition does not reproduce greedy_pred"

        rec = {"seed": s, "f1_neural": mf1(y_te, p_neu), "f1_nesy": mf1(y_te, p_full),
               "acc_neural": float((p_neu == y_te).mean()),
               "acc_nesy": float((p_full == y_te).mean())}

        # 2. influence -------------------------------------------------------------------
        fired_any4 = fired4[te_i].any(1)
        top2 = np.sort(NL, 1)
        rec["infl"] = {
            "w_fired": w_f.tolist(), "w_fprec": w_p.tolist(),
            "rows_with_rule_fired": int(fired_any4.sum()),
            "flips_by_rules": int((p_full != p_free).sum()),
            "flips_by_free_term": int((p_free != p_neu).sum()),
            "flips_total_vs_neural": int((p_full != p_neu).sum()),
            "rule_flips_fixed": int(((p_full != p_free) & (p_full == y_te) & (p_free != y_te)).sum()),
            "rule_flips_broken": int(((p_full != p_free) & (p_free == y_te) & (p_full != y_te)).sum()),
            "mean_rule_contrib_when_fired": float(contrib.sum(1)[fired_any4].mean())
            if fired_any4.any() else 0.0,
            "mean_neural_margin": float((top2[:, -1] - top2[:, -2]).mean()),
        }

        # 1. coverage & agreement, relative to the neural decision -----------------------
        st4, sus4 = evidence(p_neu, f45[te_i], t45[te_i])
        st5, sus5 = evidence(p_neu, fired5[te_i], trig5[te_i])
        f5t, t5t = fired5[te_i], trig5[te_i]
        any5 = f5t.any(1)
        best5 = np.where(f5t, t5t, -1.0).argmax(1)
        rec["cov"] = {
            "any_rule_4": float(fired_any4.mean()), "any_rule_5": float(any5.mean()),
            "supports_decision_4": float(f45[te_i][np.arange(n_te), p_neu].mean()),
            "supports_decision_5": float(f5t[np.arange(n_te), p_neu].mean()),
            "agree_when_fired_5": float((best5[any5] == p_neu[any5]).mean()) if any5.any() else 0.0,
            "concept_agree_L2": float((P2[te_i].argmax(1) == p_neu).mean()),
            "concept_agree_L1": float((P1[te_i].argmax(1) == p_neu).mean()),
            "support_by_pred_class_5": {LAB[k]: float(f5t[p_neu == k, k].mean())
                                        if (p_neu == k).any() else float("nan")
                                        for k in range(K)},
        }
        err = (p_neu != y_te).astype(int)
        rec["states5"] = {sname: {"share": float((st5 == sname).mean()),
                                  "err_rate": float(err[st5 == sname].mean())
                                  if (st5 == sname).any() else float("nan")}
                          for sname in STATES}
        rec["states4"] = {sname: {"share": float((st4 == sname).mean()),
                                  "err_rate": float(err[st4 == sname].mean())
                                  if (st4 == sname).any() else float("nan")}
                          for sname in STATES}

        # 3. deferral --------------------------------------------------------------------
        prob = np.exp(log_softmax(NL))
        ps = np.sort(prob, 1)
        conc = 1.0 - P2[te_i][np.arange(n_te), p_neu]
        scores = {
            "confidence (1-maxprob)": 1.0 - ps[:, -1],
            "margin (top1-top2)": -(ps[:, -1] - ps[:, -2]),
            "rules, deployed 4-class": sus4,
            "rules, 5-class": sus5,
            "concept scorecard": conc,
        }
        scores["evidence: rules5+concept"] = (rnorm(sus5) + rnorm(conc)) / 2
        scores["margin + concept"] = (rnorm(scores["margin (top1-top2)"]) + rnorm(conc)) / 2
        scores["margin + rules5"] = (rnorm(scores["margin (top1-top2)"]) + rnorm(sus5)) / 2
        scores["margin + rules5 + concept"] = (rnorm(scores["margin (top1-top2)"]) + rnorm(sus5)
                                               + rnorm(conc)) / 3
        rec["n_err"] = int(err.sum())
        rec["defer"] = {}
        for nm, sc in scores.items():
            d = {"auroc": float(roc_auc_score(err, sc)) if 0 < err.sum() < n_te else float("nan")}
            for b in BUDGETS:
                cfrac, racc = caught(sc, err, b, rng)
                d[f"caught@{int(b * 100)}"] = cfrac
                d[f"retained_acc@{int(b * 100)}"] = racc
            rec["defer"][nm] = d

        # 4. hybrid ----------------------------------------------------------------------
        zc = np.log(P2 + 1e-9)
        zn = log_softmax(logits)
        hyb = {}
        for w in WS:
            f = (1.0 - w) * zc + w * zn
            pv, pt = f[va].argmax(1), f[te_i].argmax(1)
            srt = np.argsort(f[te_i], 1)
            k1, k2 = srt[:, -1], srt[:, -2]
            r = np.arange(n_te)
            mc = np.abs((1.0 - w) * (zc[te_i][r, k1] - zc[te_i][r, k2]))
            mn = np.abs(w * (zn[te_i][r, k1] - zn[te_i][r, k2]))
            hyb[w] = {"val_f1": mf1(y[va], pv), "test_f1": mf1(y_te, pt),
                      "test_acc": float((pt == y_te).mean()),
                      "concept_agree": float((zc[te_i].argmax(1) == pt).mean()),
                      "concept_margin_share": float(np.mean(mc / (mc + mn + 1e-12)))}
        ref = hyb[1.0]["val_f1"]
        w_star = min(w for w in WS if hyb[w]["val_f1"] >= ref - 0.01)
        rec["hybrid"] = {str(w): hyb[w] for w in WS}
        rec["w_star"] = w_star
        per_seed.append(rec)
        print(f"[step1] seed {s}: neural F1 {rec['f1_neural']:.3f} NeSy {rec['f1_nesy']:.3f} | "
              f"flips rules={rec['infl']['flips_by_rules']} free={rec['infl']['flips_by_free_term']} "
              f"| w*={w_star} test F1 {hyb[w_star]['test_f1']:.3f}", flush=True)

    json.dump({"clauses4": n_clauses4, "clauses5": n_clauses5, "rvec": rvec.tolist(),
               "scorecard": {"l2_C": c2, "l1_C": c1, "l1_nnz": nnz1},
               "per_seed": per_seed}, open(OUT, "w"), indent=1)
    report(per_seed, n_clauses4, n_clauses5, rvec, n_te)
    print(f"\n_saved {OUT}_")


def report(ps, n4, n5, rvec, n_te):
    g = lambda f: [f(r) for r in ps]
    print("\n## EAAI step 1 -- corrected benchmark, official protocol, 5 seeds\n")
    print(f"Neural macro-F1 {ms(g(lambda r: r['f1_neural']))} | NeSy {ms(g(lambda r: r['f1_nesy']))} "
          f"| neural acc {ms(g(lambda r: r['acc_neural']))} | test n={n_te}\n")
    print(f"Clauses, deployed 4-class rules: {n4} | 5-class extension: {n5} | "
          f"rule precision (rvec): {np.round(rvec, 3).tolist()}\n")

    print("### 1. Coverage and agreement (relative to the neural decision)\n")
    print("| quantity | value |")
    print("|---|---|")
    for key, lab in [("any_rule_4", "outputs with any rule fired, deployed 4-class rules"),
                     ("any_rule_5", "outputs with any rule fired, 5-class rules"),
                     ("supports_decision_4", "a rule for the PREDICTED class fired, 4-class"),
                     ("supports_decision_5", "a rule for the PREDICTED class fired, 5-class"),
                     ("agree_when_fired_5", "best fired rule's class == decision, when any fires (5-class)"),
                     ("concept_agree_L2", "concept scorecard (L2) agrees with decision -- 100% coverage"),
                     ("concept_agree_L1", "readable scorecard (L1) agrees with decision -- 100% coverage")]:
        print(f"| {lab} | {ms(g(lambda r: r['cov'][key]))} |")
    print("\nSupport rate by predicted class (5-class rules): " + ", ".join(
        f"{c} {np.nanmean(g(lambda r: r['cov']['support_by_pred_class_5'][c])):.2f}" for c in LAB))

    for tag in ("states5", "states4"):
        print(f"\n### Evidence state vs neural error rate ({'5-class' if tag == 'states5' else 'deployed 4-class'} rules)\n")
        print("| state | share of outputs | neural error rate |")
        print("|---|---|---|")
        for sname in STATES:
            sh = g(lambda r: r[tag][sname]["share"])
            er = [x for x in g(lambda r: r[tag][sname]["err_rate"]) if x == x]
            print(f"| {sname} | {ms(sh)} | {ms(er) if er else '-'} |")

    print("\n### 2. Influence: do the rules change decisions?\n")
    wf = np.array(g(lambda r: r["infl"]["w_fired"]))
    wp = np.array(g(lambda r: r["infl"]["w_fprec"]))
    print("| rule | w_fired (mean over seeds) | w_fprec |")
    print("|---|---|---|")
    for j, c in enumerate(ST):
        print(f"| {c} | {wf[:, j].mean():.4f} | {wp[:, j].mean():.4f} |")
    print()
    print("| quantity | per seed (mean+-std) |")
    print("|---|---|")
    for key, lab in [("rows_with_rule_fired", "test rows with any rule fired"),
                     ("flips_by_rules", "decisions changed by rule evidence"),
                     ("rule_flips_fixed", "  of which: wrong -> right"),
                     ("rule_flips_broken", "  of which: right -> wrong"),
                     ("flips_by_free_term", "decisions changed by the unconstrained term"),
                     ("flips_total_vs_neural", "total decisions differing from the neural"),
                     ("mean_rule_contrib_when_fired", "mean rule logit contribution, when fired"),
                     ("mean_neural_margin", "mean neural logit margin (top1-top2)")]:
        print(f"| {lab} | {ms(g(lambda r: r['infl'][key]))} |")

    print("\n### 3. Deferral: which signal catches the neural classifier's errors?\n")
    print(f"Neural errors on test: {ms(g(lambda r: r['n_err']))}. A random review policy catches "
          f"10% / 20% of errors in expectation.\n")
    print("| deferral score | error AUROC | caught @10% | caught @20% | acc of rest @10% | acc of rest @20% |")
    print("|---|---|---|---|---|---|")
    for nm in ps[0]["defer"]:
        dd = lambda k: ms(g(lambda r: r["defer"][nm][k]))
        print(f"| {nm} | {dd('auroc')} | {dd('caught@10')} | {dd('caught@20')} | "
              f"{dd('retained_acc@10')} | {dd('retained_acc@20')} |")
    base = "margin (top1-top2)"
    print(f"\nPaired against `{base}` (per-seed differences):\n")
    print("| deferral score | d AUROC | d caught @10% | d caught @20% | seeds better @20% |")
    print("|---|---|---|---|---|")
    for nm in ps[0]["defer"]:
        if nm == base:
            continue
        da = [r["defer"][nm]["auroc"] - r["defer"][base]["auroc"] for r in ps]
        d1 = [r["defer"][nm]["caught@10"] - r["defer"][base]["caught@10"] for r in ps]
        d2 = [r["defer"][nm]["caught@20"] - r["defer"][base]["caught@20"] for r in ps]
        print(f"| {nm} | {ms(da)} | {ms(d1)} | {ms(d2)} | {sum(x > 0 for x in d2)}/{len(ps)} |")

    print("\n### 4. Hybrid: final = (1-w) log p_concept + w log p_neural\n")
    print("| w | test F1 | test acc | concept part alone agrees with decision | concept share of winning margin |")
    print("|---|---|---|---|---|")
    for w in WS:
        h = lambda k: ms(g(lambda r: r["hybrid"][str(w)][k]))
        print(f"| {w} | {h('test_f1')} | {h('test_acc')} | {h('concept_agree')} | {h('concept_margin_share')} |")
    wst = g(lambda r: r["w_star"])
    print(f"\nw* (most concept-weighted point within 0.01 of neural on VALIDATION), per seed: {wst}")
    print("At w*: test F1 " + ms([r["hybrid"][str(r["w_star"])]["test_f1"] for r in ps])
          + " | concept agreement " + ms([r["hybrid"][str(r["w_star"])]["concept_agree"] for r in ps])
          + " | concept margin share " + ms([r["hybrid"][str(r["w_star"])]["concept_margin_share"] for r in ps]))


if __name__ == "__main__":
    main()
