"""Shared context + helpers for the Results table scripts.

build_context() fits/loads the complete model once and precomputes everything the
table scripts need (neural logits, symbolic firings, policy state, splits), so
run_all.py builds it a single time and passes it to every table.
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nesymis import config
from nesymis import original_design as odesign
from nesymis.data import dataset as ds
from nesymis.fusion import decision_layer as rl
from nesymis.fusion.numeric import softmax
from nesymis.metrics import compute_metrics
from nesymis.symbolic import grounding, rules

L = config.IDX_TO_LABEL
CONTENT_GRID = {"tau": [.6, .7, .8, .9, 1.01], "lam_pos": [0, .5, 1, 2], "lam_neg": [0, .05, .1, .2, .3]}
HDR = ("| Model | Acc | Macro-F1 | Non-Mis | Kitchen | Leadership | Working | Shopping |\n"
       "|---|---|---|---|---|---|---|---|")


class Ctx:
    pass


def build_context() -> Ctx:
    c = Ctx()
    c.system = odesign.load()
    c.emb = ds.load_embeddings()
    c.df, c.y = c.emb.df, c.emb.labels
    c.tr, c.va, c.te = (c.emb.split_mask(s) for s in ("train", "val", "test"))
    c.trainval = c.df["split"].isin(["train", "val"]).to_numpy()
    c.vu = c.emb.features("both").astype(np.float32)
    c.D = np.zeros((len(c.df), config.AFFECT_DIM), np.float32)   # AFFECT-FREE: the slot is fed zeros
    c.logits = c.system.neural_logits(c.vu)                 # deployed (inference) logits
    # Logits the deployed decision layer was actually fitted on: out-of-fold on
    # train+val, full-model on test. Controls that retrain a policy must use
    # these, or they compete against a neural branch that has memorized its
    # training rows (identical on test rows, so evaluation is unaffected).
    c.oof_logits = c.logits
    if config.OOF_LOGITS_PATH.exists():
        z = np.load(config.OOF_LOGITS_PATH)
        if z.shape == c.logits.shape:
            c.oof_logits = z.astype(np.float32)
    c.probs = softmax(c.logits)
    c.nconf = c.probs.max(1)
    c.rule_df = c.system.symbolic_eval(c.emb)
    c.fired = np.stack([c.rule_df[f"{cl}_fired"].to_numpy() for cl in config.STEREO_CLASSES], 1)
    c.rprec_vec = c.system.rprec_vec
    c.state = rl.build_state(c.logits, c.fired, c.rprec_vec, c.D)
    c.pol = c.system.policy_type
    return c


def grid_pred(probs, nconf, fired, rprec_vec, p):
    s = probs.copy()
    for j, cl in enumerate(config.STEREO_CLASSES):
        i = config.LABEL_TO_IDX[cl]
        s[:, i] += np.where(fired[:, j], p["lam_pos"] * rprec_vec[j], -p["lam_neg"])
    return np.where(nconf < p["tau"], s.argmax(1), probs.argmax(1))


def tune_grid(probs, nconf, fired, rprec_vec, y):
    best = None
    for combo in itertools.product(*CONTENT_GRID.values()):
        p = dict(zip(CONTENT_GRID, combo))
        m = compute_metrics(y, grid_pred(probs, nconf, fired, rprec_vec, p))
        key = (m["macro_f1"], m["acc"])
        if best is None or key > best[0]:
            best = (key, p)
    return best[1]


def row(name, m):
    p = m["per_class_f1"]
    return (f"| {name} | {m['acc']:.3f} | {m['macro_f1']:.3f} | {p['non_stereotype']:.3f} | "
            f"{p['kitchen']:.3f} | {p['leadership']:.3f} | {p['working']:.3f} | {p['shopping']:.3f} |")


# 11-seed protocol: canonical baseline seed 42 + 10 meta-seeded draws (reproducible,
# chosen blind to results). Shared by every multi-seed table.
SEEDS11 = [42] + [int(s) for s in np.random.default_rng(42).integers(0, 2**31 - 1, 10)]


def full_policy(logits, state, y, tr, va, trainval=None, df=None, seed=config.SEED):
    """Deployed decision-layer training: LinearEvidencePolicy, correctness bootstrap
    only.
    trainval/df kept for signature compatibility; unused."""
    return rl.train_rlvr(logits, state, y, tr, va, seed=seed,
                         policy_ctor=rl.LinearEvidencePolicy)[0]


def ms(vals):
    """mean+-std [first seed] cell for multi-seed tables."""
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():.3f}+-{v.std():.3f} [{v[0]:.3f}]"
