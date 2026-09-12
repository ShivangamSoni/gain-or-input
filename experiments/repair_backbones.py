"""Backbone generality: the concept-routed decision on stronger neural classifiers.

The deployed neural classifier reads frozen CLIP ViT-B/32 features and scores 0.64 macro-F1. A
natural question is whether the concept layer still matters, and what it costs, when the neural
classifier is much stronger. Here the symbolic side is held fixed -- the deployed B/32 concept
bank and scorecard on the uniform OCR text -- and only the neural expert changes: the package's
head (config.MLP, cross-fitted) on each backbone's [image; OCR text] features, all under the
uniform-OCR protocol.

  b32          frozen CLIP ViT-B/32 (the deployed system; reference)            [default]
  siglip       frozen SigLIP SO400M-384, artifacts/enc_siglip-so400m_uniform.npz  [default]
  ft-l14-xfit  CLIP ViT-L/14 with its top 6 blocks fine-tuned, as its own classifier. Its
               logits are CROSS-FITTED (strong_baselines.py --crossfit l14 --uniform): every
               train+val row is predicted by a model fine-tuned without it, test rows by the
               model fine-tuned on the training split. No head is trained on top; seeds vary only
               the concept side's cross-validation folds. One encoder seed.

Why not a head on features exported from one fine-tuned encoder: that encoder saw the training
labels, so its train+val logits look nearly perfect and the operating point would be pushed to
the neural classifier alone -- an artefact, not a finding.

Operating point: the deployed rule (cross-validation over train+val, concept bank rebuilt per
fold, within 0.01 of the neural classifier on macro-F1 and accuracy). The concept side does not
depend on the backbone, so its out-of-fold predictions are computed once per seed and shared.
11 seeds (the head's seed; frozen encoders). Paired bootstrap CIs as in repair_benchmark_split.py.

  python experiments/repair_backbones.py                              # b32, siglip
  python experiments/repair_backbones.py --backbones ft-l14-xfit      # after --crossfit
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
from repair_core import faithfulness, pooled_pred
from repair_benchmark_split import f1_per_class, f1c, paired_bootstrap
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp
from nesymis.neural import classifier as clf

ART = config.ARTIFACTS_DIR
LAB = list(config.LABELS)
ACC_TOL = 0.01


def backbones(d, names):
    """name -> ("features", X) for a cross-fitted head, or ("logits", L) for fixed logits."""
    def feats(fname):
        z = np.load(ART / fname)
        assert z["v"].shape[0] == len(d.y), f"{fname}: row count differs"
        return "features", np.concatenate([z["v"], z["u_ocr"]], 1).astype(np.float32)

    def logits(fname):
        z = np.load(ART / fname)
        assert z["logits"].shape == (len(d.y), config.NUM_CLASSES), f"{fname}: shape differs"
        return "logits", z["logits"].astype(np.float64)

    make = {"b32": lambda: ("features", np.concatenate([d.v, d.u], 1).astype(np.float32)),
            "siglip": lambda: feats("enc_siglip-so400m_uniform.npz"),
            "ft-l14-xfit": lambda: logits("ft-l14_uniform_crossfit_logits.npz")}
    return {n: make[n]() for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="b32,siglip")
    names = ap.parse_args().backbones.split(",")
    out_path = str(ART / ("repair_backbones.json" if names == ["b32", "siglip"]
                          else f"repair_backbones_{'_'.join(names)}.json"))
    d = evp.load_data()
    te_i = np.flatnonzero(d.te)
    tv = np.flatnonzero(d.tr | d.va)
    y = d.y[te_i]
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)                   # novel-slice definition only
    novel = (u_cap[te_i] @ u_cap[d.tr | d.va].T).max(1) < 0.90
    sym = evp.build_symbolic(d)
    S = sym[1].to_numpy(np.float64)
    card = cp.ConceptScorecard.fit(S, list(sym[1].columns), d.y, d.tr, d.va)
    Z = card.standardise(S[te_i])
    B = backbones(d, names)
    D0 = np.zeros((len(d.y), evp.AFFECT_DIM), np.float32)
    rng = np.random.default_rng(config.SEED)
    res = {b: [] for b in B}
    preds = {b: ([], []) for b in B}

    for s in (int(x) for x in C.SEEDS11):
        t0 = time.time()
        lc_oof = evp.concept_oof(d, card.C, s)                      # shared by every backbone
        for b, (kind, X) in B.items():
            if kind == "features":
                _, logits = clf.crossfit_logits(X, D0, d.y, d.tr, d.va, fusion=config.NEURAL_FUSION,
                                               affect_dim=evp.AFFECT_DIM, cfg=dict(config.MLP), seed=s)
            else:
                logits = X
            w = cp.select_weight(lc_oof, cp.log_softmax(logits[tv]), d.y[tv], acc_tol=ACC_TOL)
            ln = cp.log_softmax(logits[te_i])
            dec = cp.decompose(card, Z, ln, w)
            pred, pn = dec["pred"], ln.argmax(1)
            f = faithfulness(card, Z, ln, w, dec, rng)
            rec = {"seed": s, "w": w,
                   "f1_neural": f1c(y, pn), "f1_pooled": f1c(y, pred),
                   "acc_neural": float((pn == y).mean()), "acc_pooled": float((pred == y).mean()),
                   "novel_f1_neural": f1c(y[novel], pn[novel]),
                   "novel_f1_pooled": f1c(y[novel], pred[novel]),
                   "concept_share": float(dec["concept_share"].mean()),
                   "concepts_decide": 1.0 - f["keep_none_same"],
                   "changed_vs_neural": float((pred != pn).mean()),
                   "del_top3": f[3]["delete_top_changes"], "del_rand3": f[3]["delete_random_changes"],
                   "f1_class_neural": f1_per_class(y, pn).tolist(),
                   "f1_class_pooled": f1_per_class(y, pred).tolist(),
                   "curve": {}}
            for wg in cp.W_GRID:
                dg = cp.decompose(card, Z, ln, wg)
                keep_none = float((pooled_pred(card, np.zeros_like(Z), ln, wg) == dg["pred"]).mean())
                rec["curve"][str(wg)] = {"f1": f1c(y, dg["pred"]), "acc": float((dg["pred"] == y).mean()),
                                         "share": float(dg["concept_share"].mean()),
                                         "decide": 1.0 - keep_none}
            res[b].append(rec)
            preds[b][0].append(pred)
            preds[b][1].append(pn)
        line = " | ".join(f"{b} w={res[b][-1]['w']} F1 {res[b][-1]['f1_neural']:.3f}->"
                          f"{res[b][-1]['f1_pooled']:.3f} acc {res[b][-1]['acc_neural']:.3f}->"
                          f"{res[b][-1]['acc_pooled']:.3f}" for b in B)
        print(f"[r4] seed {s}: {time.time() - t0:.0f}s | {line}", flush=True)

    brng = np.random.default_rng(config.SEED + 1)
    boot = {b: paired_bootstrap(y, np.array(preds[b][0]), np.array(preds[b][1]), brng) for b in B}
    json.dump({"seeds": [int(x) for x in C.SEEDS11], "per_backbone": res, "bootstrap": boot},
              open(out_path, "w"), indent=1, default=str)
    report(res, boot)
    print(f"\n_saved {out_path}_")


def report(res, boot):
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    ci = lambda e: f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    print("\n## The concept-routed decision on three backbones (uniform-OCR protocol, 11 seeds)\n")
    print("| backbone | neural F1 / acc | pooled F1 / acc | Δ F1 [95% CI] | Δ acc [95% CI] | "
          "Δ novel F1 | w chosen | concept share | concepts decide | delete top-3 / random-3 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, rs in res.items():
        g = lambda k: [r[k] for r in rs]
        ws = g("w")
        wtxt = ", ".join(f"{w:.1f}×{ws.count(w)}" for w in sorted(set(ws)))
        dn = np.mean(np.array(g("novel_f1_pooled")) - np.array(g("novel_f1_neural")))
        print(f"| {b} | {np.mean(g('f1_neural')):.3f} / {np.mean(g('acc_neural')):.3f} | "
              f"{np.mean(g('f1_pooled')):.3f} / {np.mean(g('acc_pooled')):.3f} | "
              f"{ci(boot[b]['d_f1'])} | {ci(boot[b]['d_acc'])} | {dn:+.3f} | {wtxt} | "
              f"{ms(g('concept_share'))} | {ms(g('concepts_decide'))} | "
              f"{np.mean(g('del_top3')):.3f} / {np.mean(g('del_rand3')):.3f} |")
    print("\n### Per-class F1, neural -> pooled (mean over seeds)\n")
    print("| backbone | " + " | ".join(LAB) + " |")
    print("|---|" + "---|" * len(LAB))
    for b, rs in res.items():
        fn = np.mean([r["f1_class_neural"] for r in rs], 0)
        fp = np.mean([r["f1_class_pooled"] for r in rs], 0)
        print(f"| {b} | " + " | ".join(f"{a:.3f}->{p:.3f}" for a, p in zip(fn, fp)) + " |")
    print("\n### Curves (mean over seeds): macro-F1 / accuracy / concepts decide\n")
    print("| w | " + " | ".join(res) + " |")
    print("|---|" + "---|" * len(res))
    for wg in cp.W_GRID:
        cells = []
        for b, rs in res.items():
            q = lambda k: np.mean([r["curve"][str(wg)][k] for r in rs])
            cells.append(f"{q('f1'):.3f} / {q('acc'):.3f} / {q('decide'):.3f}")
        print(f"| {wg} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
