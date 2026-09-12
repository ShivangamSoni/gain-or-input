"""Do the known-answer verdicts depend on the baseline and the pooling rule?

Every system in experiments/known_answer_architectures.py shares that test's baseline component N (a probe on CLIP
ViT-B/32 image + in-image text) and that test's decision rule (equal-weight log-linear pooling), so the
five architectures vary P and nothing else. This varies the other two parts, with P held at that test's
probe on the tweet text:

  baseline    N is a probe on SigLIP SO400M image + in-image text, and P reads the tweet through
              SigLIP too; the negative control gives P N's own inputs through CLIP (the other
              encoder, weaker here -- a genuine gain if it is one at all)
  pooling     N and P as in the first known-answer test (CLIP), but the decision is a logistic regression over both
              components' log-probabilities, fitted on the validation split, instead of the
              equal-weight pooling fixed in advance

Expectations are that test's, fixed before the run: positive control S1 FLAG and S2 FLAG, negative control
S1 ok and S2 ok. 5 seeds, the official test split.

  python experiments/known_answer_baseline_and_pooling.py [--only baseline,pooling]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression

import known_answer_mmhs as a1
from nesymis import config
from inputaudit import system_tier as st
from inputaudit.probes import macro_f1, paired_bootstrap
from d5_modality_probes import fit_probe

OUT = config.ARTIFACTS_DIR / "known_answer_baseline_and_pooling.json"
SEEDS = a1.SEEDS
AS, EQ = "as built (P reads the tweet)", "equal inputs (P reads the image text)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="baseline,pooling")
    which = ap.parse_args().only.split(",")
    ids, post, img, y6, split, all_ids = a1.load_pool()
    tr, va, te = split == "train", split == "val", split == "test"
    E = a1.embeddings(ids, post, img, all_ids, need=("su_post",))
    print(f"[var] pool {len(ids)}: train {tr.sum()} val {va.sum()} test {te.sum()}", flush=True)
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}

    VIEWS = {
        "baseline": {"N": np.concatenate([E["sv"], E["su_img"]], 1),          # SigLIP baseline
                     "tweet": E["su_post"], "same": E["su_img"],
                     "neg": np.concatenate([E["v"], E["u_img"]], 1)},         # the other encoder
        "pooling": {"N": np.concatenate([E["v"], E["u_img"]], 1),             # that test's baseline
                    "tweet": E["u_post"], "same": E["u_img"],
                    "neg": np.concatenate([E["sv"], E["su_img"]], 1)},
    }
    NEG_NAME = {"baseline": "P reads image + image text via CLIP",
                "pooling": "P reads image + image text via SigLIP"}

    for variant in which:
        views = VIEWS[variant]
        for task, y, k in (("6-class hate type", y6, 6), ("sexist vs rest", (y6 == 2).astype(int), 2)):
            t0 = time.time()
            cache = {}

            def lg(view, seed):
                if (view, seed) not in cache:
                    out, _ = fit_probe(np.ascontiguousarray(views[view].astype(np.float32)), y, tr, va, k, seed)
                    cache[(view, seed)] = a1.log_softmax(out)                 # every row, not just test
                return cache[(view, seed)]

            def run(name, cfg, seed):
                ln, lp = lg("N", seed), lg(cfg["P"], seed)
                if variant == "pooling":                                       # fitted decision rule
                    stack = LogisticRegression(max_iter=2000, C=1.0)
                    stack.fit(np.hstack([ln[va], lp[va]]), y[va])
                    pred = stack.predict(np.hstack([ln[te], lp[te]]))
                else:
                    pred = (0.5 * ln[te] + 0.5 * lp[te]).argmax(1)
                return pred, ln[te].argmax(1)

            yt = y[te]
            pos = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": ["tweet text"]},
                                         ["N", "P"], ["N"], ["image", "image text", "tweet text"]),
                   "s2": st.s2_same_input(run, {AS: {"P": "tweet"}, EQ: {"P": "same"}}, SEEDS, yt, k, equal=EQ)}
            neg = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": ["image", "image text"]},
                                         ["N", "P"], ["N"], ["image", "image text"]),
                   "s2": st.s2_same_input(run, {NEG_NAME[variant]: {"P": "neg"}}, SEEDS, yt, k,
                                          equal=NEG_NAME[variant])}
            alone = {v: float(np.mean([macro_f1(yt, lg(v, s)[te].argmax(1), k) for s in SEEDS]))
                     for v in views}
            res.setdefault(variant, {})[task] = {"positive_control": pos, "negative_control": neg,
                                                 "components_alone": alone, "n_test": int(te.sum()),
                                                 "minutes": round((time.time() - t0) / 60, 1)}
            json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
            c, p2, n2 = pos["s2"]["configs"], pos["s2"], neg["s2"]
            sw = c[AS]["swing"]
            print(f"[var] {variant} | {task} | alone {json.dumps({a: round(b, 3) for a, b in alone.items()})}\n"
                  f"  positive: as built {c[AS]['margin']['mean']:+.3f} {np.round(c[AS]['margin']['ci'], 3)}, "
                  f"equal {c[EQ]['margin']['mean']:+.3f} {np.round(c[EQ]['margin']['ci'], 3)}, "
                  f"swing {sw['mean']:+.3f} {np.round(sw['ci'], 3)}\n"
                  f"  positive: S1 {pos['s1']['verdict']} | S2 {p2['verdict']} "
                  f"(first-stated rule: {p2['verdict_no_gain_survives']})\n"
                  f"  negative: margin {list(n2['configs'].values())[0]['margin']['mean']:+.3f} "
                  f"{np.round(list(n2['configs'].values())[0]['margin']['ci'], 3)} | S1 {neg['s1']['verdict']} "
                  f"| S2 {n2['verdict']}", flush=True)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
