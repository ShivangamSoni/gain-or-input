"""System tier: checks that need the system under test.

The tier is a Python API: the auditor supplies the few functions only they can write (train and
score the system under a given input configuration; predict from a feature matrix). Every
function returns {"check", "verdict", ...statistics}, as in the data tier.
"""
from __future__ import annotations

import numpy as np

from inputaudit.probes import SEED, paired_bootstrap

FLAG, OK, NOTE = "FLAG", "ok", "note"


# --------------------------------------------------------------------------- #
# S1 -- input inventory
# --------------------------------------------------------------------------- #
def s1_inventory(components: dict, system: list, baseline: list, available_at_inference: list):
    """`components`: {name: [inputs it reads]}. `system` / `baseline`: the components each side of
    the comparison uses. FLAGGED if the system reads an input the baseline does not (a gain may
    belong to the input, not the architecture) or reads an input unavailable at inference."""
    sys_in = set().union(*(components[c] for c in system))
    base_in = set().union(*(components[c] for c in baseline))
    only_sys = sorted(sys_in - base_in)
    unavailable = sorted(sys_in - set(available_at_inference))
    return {"check": "S1 input inventory", "verdict": FLAG if (only_sys or unavailable) else OK,
            "system_inputs": sorted(sys_in), "baseline_inputs": sorted(base_in),
            "only_in_system": only_sys, "unavailable_at_inference": unavailable,
            "next": "run S2 with those inputs equalised" if only_sys else ""}


# --------------------------------------------------------------------------- #
# S2 -- same-input test
# --------------------------------------------------------------------------- #
def s2_same_input(run, configs: dict, seeds, y_test, k, equal: str, B=2000):
    """`run(config_name, config, seed) -> (pred_system, pred_baseline)` on the test rows.
    `configs`: {name: input assignment}; `equal` names the configuration in which the system and
    the baseline read the same inputs. Margin = macro-F1(system) - macro-F1(baseline), with a
    paired bootstrap interval over test rows for the seed-averaged margin. The swing of a
    configuration is its system's macro-F1 minus the equal-input system's (paired bootstrap).
    FLAGGED when some configuration with unequal inputs shows a gain (margin interval above zero)
    of which a significant part belongs to the inputs (swing interval above zero); the
    equal-input margin is the architecture's own share. The rule first stated -- flag only when no
    gain survives equal inputs -- is kept as `verdict_no_gain_survives`: it misses input-driven
    gains that coexist with a small genuine one (experiments/known_answer_architectures.py)."""
    res, preds = {}, {}
    for name, cfg in configs.items():
        P = [run(name, cfg, s) for s in seeds]
        Ps, Pb = np.array([p[0] for p in P]), np.array([p[1] for p in P])
        preds[name] = Ps
        bt = paired_bootstrap(y_test, Ps, Pb, k, B)
        res[name] = {"config": cfg, "margin": bt["d_macro_f1"], "d_acc": bt["d_acc"]}
    for name in res:
        if name != equal:
            res[name]["swing"] = paired_bootstrap(y_test, preds[name], preds[equal], k, B)["d_macro_f1"]
    eq_lo = res[equal]["margin"]["ci"][0]
    gains = [n for n, r in res.items() if n != equal and r["margin"]["ci"][0] > 0]
    input_driven = [n for n in gains if res[n]["swing"]["ci"][0] > 0]
    verdict = FLAG if input_driven else OK
    return {"check": "S2 same-input test", "verdict": verdict, "configs": res, "equal": equal,
            "gain_partly_from_inputs": input_driven,
            "verdict_no_gain_survives": FLAG if (gains and eq_lo <= 0) else OK,
            "swing": (max(r["margin"]["mean"] for r in res.values()) - res[equal]["margin"]["mean"])}


# --------------------------------------------------------------------------- #
# S3 -- headroom by information type
# --------------------------------------------------------------------------- #
def s3_headroom(y, base_preds, alternatives: dict, stable=None, min_residual_share=0.02):
    """`base_preds`: (seeds, N) predictions of the system (or its neural part) on the test rows.
    `alternatives`: {name: (seeds, N)} predictions of the same learner given one more kind of
    information (e.g. richer text, the image alone). A row is a stable error when wrong in
    >= `stable` seeds (default: 60%); an alternative fixes it when right in >= `stable` seeds.
    The residual -- stable errors no alternative fixes -- bounds what a component adding a NEW
    kind of information can achieve. FLAGGED when the residual is below `min_residual_share` of
    the test set."""
    y = np.asarray(y)
    base = np.atleast_2d(base_preds)
    s = stable or int(np.ceil(0.6 * len(base)))
    err = (base != y).sum(0) >= s
    fixed_by = {}
    any_fix = np.zeros(len(y), bool)
    for name, P in alternatives.items():
        f = err & ((np.atleast_2d(P) == y).sum(0) >= s)
        fixed_by[name] = int(f.sum())
        any_fix |= f
    residual = int((err & ~any_fix).sum())
    share = residual / len(y)
    return {"check": "S3 headroom by information type",
            "verdict": FLAG if share < min_residual_share else NOTE,
            "stable_errors": int(err.sum()), "fixed_by": fixed_by, "residual": residual,
            "residual_share_of_test": share, "stable_threshold": s}


# --------------------------------------------------------------------------- #
# S4 -- influence of the explanatory component
# --------------------------------------------------------------------------- #
def s4_influence(y, pred_full, pred_without, margin_share=None, min_changed=0.02):
    """How many decisions the component changes: `pred_full` with it, `pred_without` without it
    (e.g. its contribution zeroed). `margin_share`: optional per-row share of the winning margin
    that the component supplies. FLAGGED as decorative when it changes < `min_changed` of
    decisions: evidence it reports then describes decisions it did not make."""
    y, a, b = np.asarray(y), np.asarray(pred_full), np.asarray(pred_without)
    ch = a != b
    out = {"check": "S4 influence", "verdict": FLAG if ch.mean() < min_changed else OK,
           "changed": float(ch.mean()), "n_changed": int(ch.sum()),
           "fixes": int((ch & (a == y)).sum()), "breaks": int((ch & (b == y)).sum())}
    if margin_share is not None:
        out["mean_margin_share"] = float(np.mean(margin_share))
    return out


# --------------------------------------------------------------------------- #
# S5 -- faithfulness of named reasons
# --------------------------------------------------------------------------- #
def s5_faithfulness(predict, Z, contrib, ks=(1, 3, 5), reps=20, neutral=0.0, min_ratio=2.0,
                    seed=SEED):
    """`predict(Z) -> labels` for a feature matrix Z (N, J); `contrib` (N, J): each feature's
    contribution to each row's decision, larger = more for the label. Deleting a row's top-k
    features (setting them to `neutral`) should change far more decisions than deleting k random
    ones, and keeping only the top-k should preserve more than keeping k random. FLAGGED when
    deleting the top 3 changes fewer than `min_ratio` times as many decisions as random."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, float)
    N, J = Z.shape
    base = predict(Z)
    rows = np.arange(N)[:, None]
    order = np.argsort(-np.asarray(contrib), 1)
    res = {"unchanged_with_all_removed": float((predict(np.full_like(Z, neutral)) == base).mean())}
    for k in ks:
        top = order[:, :k]
        Zd = Z.copy()
        Zd[rows, top] = neutral
        Zk = np.full_like(Z, neutral)
        Zk[rows, top] = Z[rows, top]
        dr, kr = [], []
        for _ in range(reps):
            rk = np.argsort(rng.random((N, J)), 1)[:, :k]
            Zr = Z.copy()
            Zr[rows, rk] = neutral
            dr.append(float((predict(Zr) != base).mean()))
            Zs = np.full_like(Z, neutral)
            Zs[rows, rk] = Z[rows, rk]
            kr.append(float((predict(Zs) == base).mean()))
        res[k] = {"delete_top": float((predict(Zd) != base).mean()), "delete_random": float(np.mean(dr)),
                  "keep_top": float((predict(Zk) == base).mean()), "keep_random": float(np.mean(kr))}
    k3 = 3 if 3 in ks else ks[len(ks) // 2]
    ratio = res[k3]["delete_top"] / max(res[k3]["delete_random"], 1e-9)
    return {"check": "S5 faithfulness", "verdict": FLAG if ratio < min_ratio else OK,
            "deletion_ratio_top_vs_random": ratio, "k_for_ratio": k3, "by_k": res}


# --------------------------------------------------------------------------- #
# S6 -- fresh partitions for test-informed choices
# --------------------------------------------------------------------------- #
def s6_partitions(y, pool_idx, seeds=(1, 2, 3, 4, 5), test_size=0.15, val_size=0.15):
    """Stratified train/val/test re-partitions of `pool_idx` (the rows that were NOT the test set
    the choice was made on). Returns [{"seed", "train", "val", "test"}] as row indices."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    pool = np.asarray(pool_idx)
    out = []
    for s in seeds:
        tr, tmp = train_test_split(pool, test_size=test_size + val_size, stratify=y[pool],
                                   random_state=s)
        va, te = train_test_split(tmp, test_size=test_size / (test_size + val_size),
                                  stratify=y[tmp], random_state=s)
        out.append({"seed": s, "train": tr, "val": va, "test": te})
    return out


def s6_verdict(deltas: list, criterion: dict):
    """`deltas`: one {metric: value} per fresh partition (e.g. system - baseline); `criterion`:
    {metric: minimum mean}, fixed BEFORE running. CONFIRMED only if every metric's mean meets it;
    otherwise FLAGGED (the choice does not hold out of sample)."""
    means = {m: float(np.mean([d[m] for d in deltas])) for m in criterion}
    sds = {m: float(np.std([d[m] for d in deltas])) for m in criterion}
    ok = all(means[m] >= criterion[m] for m in criterion)
    return {"check": "S6 fresh partitions", "verdict": OK if ok else FLAG, "mean": means, "sd": sds,
            "criterion": criterion, "n_partitions": len(deltas)}
