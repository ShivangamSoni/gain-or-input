"""Table 11 - multi-seed robustness of the complete system (11 seeds).

Protocol: seed 42 (baseline anchor) + 10 seeds drawn once from a seeded
meta-RNG (arbitrary, chosen blind to results, reproducible). Per seed the
affect-aware classifier and the linear evidence decision layer are retrained; the
learned symbolic path (predicates, thresholds, rules) is deterministic.
Reported on the full test set and the novel slice (no near-duplicate caption
in train+val, cosine < 0.90)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
from nesymis import config
from nesymis.neural import classifier as clf

META_SEED = 42
SEEDS = [42] + list(np.random.default_rng(META_SEED).integers(0, 2**31 - 1, 10))
NOVEL_COS = 0.90
OUT = config.ARTIFACTS_DIR / "original_design_multiseed.json"     # per-seed values behind the table


def generate(ctx=None):
    c = ctx or C.build_context()
    y, tr, va, te, trainval, df, vu, D = (
        c.y,
        c.tr,
        c.va,
        c.te,
        c.trainval,
        c.df,
        c.vu,
        c.D,
    )
    y_te = y[te]

    U = c.emb.text_caption
    novel = (U[te] @ U[trainval].T).max(1) < NOVEL_COS
    slices = {"full": np.ones(len(y_te), bool), "novel": novel}

    pb = C.rules.rule_only_prediction(c.rule_df).to_numpy()[te]
    pb_s = {k: C.compute_metrics(y_te[m], pb[m]) for k, m in slices.items()}

    # res[row][slice][metric] -> list over seeds
    res = {r: {k: {"acc": [], "macro_f1": []} for k in slices} for r in ("Path-A", "NeSy")}
    for seed in SEEDS:
        clf, logits = clf.crossfit_logits(
            vu, D, y, tr, va, fusion=config.NEURAL_FUSION, affect_dim=C.odesign.AFFECT_DIM, seed=int(seed)
        )
        state = C.rl.build_state(logits, c.fired, c.rprec_vec, D)
        policy = C.full_policy(logits, state, y, tr, va, seed=int(seed))
        ne = C.rl.greedy_pred(policy, logits[te], state[te])
        pa = logits[te].argmax(1)
        for k, m in slices.items():
            for row, pred in (("Path-A", pa), ("NeSy", ne)):
                mm = C.compute_metrics(y_te[m], pred[m])
                res[row][k]["acc"].append(mm["acc"])
                res[row][k]["macro_f1"].append(mm["macro_f1"])
        print(f"  [multiseed] seed {seed} done", flush=True)
    json.dump({"design": {"neural_text": config.NEURAL_TEXT_SOURCE, "symbolic_text": config.SYMBOLIC_TEXT_SOURCE},
               "seeds": [int(s) for s in SEEDS], "n_test": int(len(y_te)), "n_novel": int(novel.sum()),
               "Path-B": pb_s, "per_seed": res}, open(OUT, "w", encoding="utf-8"), indent=1, default=float)

    def cell(vals):
        v = np.array(vals)
        return f"{v.mean():.3f}+-{v.std():.3f} [{v[0]:.3f}]"

    def seed_row(name, row):
        return (f"| {name} | {cell(res[row]['full']['acc'])} | {cell(res[row]['full']['macro_f1'])} "
                f"| {cell(res[row]['novel']['acc'])} | {cell(res[row]['novel']['macro_f1'])} |")

    md = [
        f"## Table 11 - Multi-seed robustness ({len(SEEDS)} seeds = 42 + 10 meta-seeded; "
        f"TEST n={len(y_te)}, novel={int(novel.sum())})\n",
        "| Path | Acc full (mean+-std [s42]) | Macro-F1 full (mean+-std [s42]) "
        "| Acc novel (mean+-std [s42]) | Macro-F1 novel (mean+-std [s42]) |",
        "|---|---|---|---|---|",
        f"| Path-B (learned rules, deterministic) | {pb_s['full']['acc']:.3f} | {pb_s['full']['macro_f1']:.3f} "
        f"| {pb_s['novel']['acc']:.3f} | {pb_s['novel']['macro_f1']:.3f} |",
        seed_row("Path-A (Neural)", "Path-A"),
        seed_row("NeSy-MIS", "NeSy"),
        f"\n_Seeds: {[int(s) for s in SEEDS]}_",
    ]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    print(generate())
