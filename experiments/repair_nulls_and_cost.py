"""The error-detection null and the cost, re-measured on the DEPLOYED decision.

Step 1 (experiments/evidence_feasibility.py) measured both before the concept-routed decision existed:
error detection against the NEURAL classifier's errors, and cost for the original design's
symbolic path. The paper reports both for the system it presents, so both are re-measured here.

  NULLS   11 seeds, deployed system (concept_routed defaults), benchmark split. Errors are the
          concept-routed decision's own. Review signals as in step 1 (higher = more suspicious):
          the decision's confidence and margin, the neural classifier's confidence, the 5-class
          rule suspicion, the concept scorecard's doubt about the label, and equal-weight rank
          averages fixed in advance. Error AUROC, errors caught at 10% / 20% review budgets.
  COST    The saved seed-42 system. Batched (490 test memes) and single-meme latency for every
          inference component, with the frozen encoder and OCR reported separately because the
          neural path needs them too. Run with the GPU otherwise idle.

  python experiments/repair_nulls_and_cost.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as C
from evidence_feasibility import caught, rnorm
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp

OUT = str(config.ARTIFACTS_DIR / "repair_nulls_and_cost.json")
BUDGETS = (0.10, 0.20)
REPS = 20


# --------------------------------------------------------------------------- #
# Nulls: error detection for the deployed decision
# --------------------------------------------------------------------------- #
def review_scores(out):
    sc = cp.softmax(out["scores"])
    s_srt = np.sort(sc, 1)
    pn = cp.softmax(out["neural_logits"])
    pc = np.exp(out["lc"])
    i = np.arange(len(sc))
    conf, margin = 1.0 - s_srt[:, -1], -(s_srt[:, -1] - s_srt[:, -2])
    rules, concept = out["suspicion"], 1.0 - pc[i, out["pred"]]
    return {
        "decision confidence (1 - max prob)": conf,
        "decision margin (top1 - top2)": margin,
        "neural classifier confidence": 1.0 - pn.max(1),
        "rules, 5-class": rules,
        "concept scorecard": concept,
        "evidence: rules + concepts": (rnorm(rules) + rnorm(concept)) / 2,
        "margin + rules": (rnorm(margin) + rnorm(rules)) / 2,
        "margin + concepts": (rnorm(margin) + rnorm(concept)) / 2,
        "margin + rules + concepts": (rnorm(margin) + rnorm(rules) + rnorm(concept)) / 3,
    }


def nulls():
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    y = d.y[te_i]
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    sym = evp.build_symbolic(d)
    rng = np.random.default_rng(config.SEED)
    per = []
    for s in (int(x) for x in C.SEEDS11):
        sysm, logits = evp.fit(seed=s, data=d, sym=sym)
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i],
                          neural_logits=logits[te_i])
        err = (out["pred"] != y).astype(int)
        rec = {"seed": s, "w": sysm.w, "n_err": int(err.sum()), "acc": float(1 - err.mean()),
               "defer": {}}
        for nm, sc in review_scores(out).items():
            r = {"auroc": float(roc_auc_score(err, sc))}
            for b in BUDGETS:
                c, a = caught(sc, err, b, rng)
                r[f"caught@{int(b * 100)}"], r[f"retained_acc@{int(b * 100)}"] = c, a
            rec["defer"][nm] = r
        per.append(rec)
        dd = rec["defer"]
        print(f"[r11] seed {s}: w={sysm.w} errors {rec['n_err']} | AUROC confidence "
              f"{dd['decision confidence (1 - max prob)']['auroc']:.3f} rules+concepts "
              f"{dd['evidence: rules + concepts']['auroc']:.3f}", flush=True)
    return per


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def timed(fn, reps=REPS):
    """Median wall-clock over `reps`, CUDA synchronised so GPU work is counted."""
    ts, out = [], None
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def cost():
    d = evp.load_data()
    s = evp.load()
    te_i = np.flatnonzero(d.te)
    n = len(te_i)
    u, v = d.u[te_i], d.v[te_i]
    vu = np.concatenate([v, u], 1).astype(np.float32)
    idx = d.df.index[te_i]
    s.decide(vu, u, v, index=idx)                                     # warm-up

    ms = {}
    t, cont = timed(lambda: s.bank.frame(u, v, index=idx))
    ms["concept scoring (30 concepts)"] = t
    cont = cont[s.card.names]
    t, lg = timed(lambda: s.neural_logits(vu))
    ms["neural classifier forward"] = t
    Z = s.card.standardise(cont.to_numpy(np.float64))
    ln = cp.log_softmax(lg)
    t, dec = timed(lambda: cp.decompose(s.card, Z, ln, s.w))
    ms["scorecard + pooling + exact attribution"] = t
    t, _ = timed(lambda: cp.evidence_state(dec["pred"], *cp.fire_rules(s.rule_sets, cont)[:2]))
    ms["rules (5 classes) + evidence label"] = t
    t, out = timed(lambda: s.decide(vu, u, v, index=idx))
    ms["decide(), all of the above"] = t
    t, _ = timed(lambda: s.records(out), reps=5)
    ms["evidence records (formatting)"] = t
    batched = {k: v_ / n * 1000 for k, v_ in ms.items()}
    batched["symbolic side (concepts + scorecard/pooling + rules)"] = (
        batched["concept scoring (30 concepts)"] + batched["scorecard + pooling + exact attribution"]
        + batched["rules (5 classes) + evidence label"])

    single = []
    for i in range(50):
        sl = slice(i, i + 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = s.decide(vu[sl], u[sl], v[sl], index=idx[sl])
        s.records(o)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single.append((time.perf_counter() - t0) * 1000)
    single = {"median_ms": float(np.median(single)), "p95_ms": float(np.percentile(single, 95))}

    from nesymis.original_design import _abs_path, _live_ocr
    from nesymis.encoders.clip_encoder import encode_images, encode_texts
    paths = [_abs_path(p) for p in d.df["path"].astype(str).iloc[te_i[:32]]]
    texts = d.text.iloc[te_i[:32]].tolist()
    encode_images(paths[:2]), encode_texts(texts[:2])                 # warm-up
    t_img, _ = timed(lambda: encode_images(paths), reps=3)
    t_txt, _ = timed(lambda: encode_texts(texts), reps=3)
    _live_ocr(paths[0])                                               # warm-up (loads the reader)
    t_ocr, _ = timed(lambda: [_live_ocr(p) for p in paths[:16]], reps=1)
    shared = {"CLIP image encoder": t_img / 32 * 1000, "CLIP text encoder": t_txt / 32 * 1000,
              "OCR (EasyOCR)": t_ocr / 16 * 1000}
    return {"batched_ms_per_meme": batched, "single_meme": single, "shared_ms_per_meme": shared,
            "n_test": n, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def main():
    per = nulls()
    c = cost()
    json.dump({"nulls": per, "cost": c}, open(OUT, "w"), indent=1, default=str)
    report(per, c)
    print(f"\n_saved {OUT}_")


def report(per, c):
    g = lambda nm, k: np.array([r["defer"][nm][k] for r in per])
    ms = lambda v: f"{v.mean():.3f} ± {v.std():.3f}"
    names = list(per[0]["defer"])
    print(f"\n## Error detection for the deployed decision ({len(per)} seeds; its own errors, "
          f"mean {np.mean([r['n_err'] for r in per]):.1f} of 490)\n")
    print("| review signal | error AUROC | caught @10% | caught @20% |")
    print("|---|---|---|---|")
    for nm in names:
        print(f"| {nm} | {ms(g(nm, 'auroc'))} | {ms(g(nm, 'caught@10'))} | {ms(g(nm, 'caught@20'))} |")
    base = "decision confidence (1 - max prob)"
    print(f"\nAgainst `{base}` (seeds better at the 20% budget / AUROC higher):\n")
    for nm in names:
        if nm == base:
            continue
        b20 = int((g(nm, "caught@20") > g(base, "caught@20")).sum())
        bau = int((g(nm, "auroc") > g(base, "auroc")).sum())
        print(f"* {nm}: d AUROC {np.mean(g(nm, 'auroc') - g(base, 'auroc')):+.3f} "
              f"(higher on {bau}/{len(per)} seeds); caught@20 better on {b20}/{len(per)}")
    print(f"\n## Cost -- saved seed-42 system, {c['device']}\n")
    print("| component | ms per meme (batched, 490 test memes) |")
    print("|---|---|")
    for k, v in c["batched_ms_per_meme"].items():
        print(f"| {k} | {v:.4f} |")
    print(f"\nOne meme at a time (decide + evidence record): median {c['single_meme']['median_ms']:.2f} ms,"
          f" p95 {c['single_meme']['p95_ms']:.2f} ms")
    print("\nShared with the neural path (paid by any system on this input):\n")
    for k, v in c["shared_ms_per_meme"].items():
        print(f"* {k}: {v:.1f} ms per meme")


if __name__ == "__main__":
    main()
