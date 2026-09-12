"""Stage-3b of dynamic rule induction: learning the rule STRUCTURE.

Removes the last hand-designed element of the symbolic path -- the fixed schema
`woman AND context AND (stereotype OR text)` -- by inducing each class's rule as
a small DNF (up to MAX_CLAUSES conjunctions of up to MAX_LEN literals) from
train+val data alone:

  * literals   : (predicate, quantile-cut) pairs over the auto-generated predicate
                 scores + the 6-d affect-head outputs, plus negations -- so
                 thresholds are learned as part of the structure
  * enumeration: per class, guided exhaustive search (top single literals by
                 F_beta, extended to pairs/triples with support pruning)
  * selection  : greedy DNF -- repeatedly add the clause that most improves the
                 class's F_beta (precision-leaning, beta=0.5) on train+val

Clause confidence = calibration precision, used as the fired trigger for the
policy / Path-B tie-breaks. Learned rules are saved verbatim (human-readable) to
artifacts/learned_rules.json. Nothing here uses class-specific domain knowledge;
the remaining fixed pieces are learner hyperparameters only.

Run:  python -m nesymis.symbolic.rule_induction
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nesymis import config
from nesymis.symbolic.rules import META_RULE, RULE_OF_CLASS

LEARNED_RULES_PATH = config.ARTIFACTS_DIR / "learned_rules.json"

QUANTILES = (0.50, 0.75, 0.90)   # fixed fallback cut-points (pre-E2.1 behaviour)
N_THRESH_CAND = 19               # candidate cuts per concept (interior quantiles)
N_THRESH_KEEP = 3                # cuts retained per concept, matching |QUANTILES|
MIN_CUT_SEP = 0.10               # min separation between retained cuts (quantile units);
                                 # selected on validation macro-F1, see phase2_e21_thresholds.py
BETA = 0.5                       # precision-leaning clause/DNF objective
MIN_SUPPORT = 15                 # min calib rows a clause must cover
MAX_LEN = 3                      # max literals per clause
MAX_CLAUSES = 3                  # REVERTED from 4: width 4 gains nothing on the full split
                                 # and costs the novel slice (phase2_e25_fullprotocol.py).
                                 # Clause length still saturates at 3, which remains an
                                 # empirical result rather than an assumption.
POOL = 40                        # per-class literal pool (top singles by F_beta)
MAX_NEG = 1                      # at most one negated literal per clause
MAX_SUPPORT_FRAC = 0.90          # drop near-tautological literals (calib support >90%)
# clause-quality guardrails (anti noise-dredging): a clause enters the candidate
# set only if its calib precision clears max(MIN_PREC_ABS, min(MIN_LIFT*base, PREC_CAP))
MIN_PREC_ABS = 0.35
MIN_LIFT = 3.0
PREC_CAP = 0.80

# --- E2.2: learned rule complexity + beam search over DNFs ---
BEAM = 8                         # partial DNFs kept alive during selection
BEAM_CANDS = 300                 # candidate clauses considered per beam expansion
CAND_CAP = 2000                  # cap on enumerated clauses per class
CV_FOLDS = 3                     # inner folds for complexity selection
COMPLEXITY_GRID = [(l, c) for l in (1, 2, 3, 4) for c in (1, 2, 3, 4)]
FORCE_COMPLEXITY = None          # (len, width) to bypass CV selection (diagnostic)


def _fbeta(tp, fp, n_pos, beta=BETA):
    p = tp / max(tp + fp, 1)
    r = tp / max(n_pos, 1)
    b2 = beta * beta
    return (1 + b2) * p * r / max(b2 * p + r, 1e-9), p, r


def learned_cuts(v: np.ndarray, calib_mask: np.ndarray, y_labels: np.ndarray,
                 n_cand: int = N_THRESH_CAND, keep: int = N_THRESH_KEEP) -> np.ndarray:
    """Per-concept thresholds chosen by optimising the literal, not fixed a priori.

    The fixed 50/75/90 grid assumes every concept separates its class at the same
    place in its own score distribution, which is only true by coincidence: a
    concept firing on 5% of memes and one firing on 60% need different cuts. Here
    each candidate cut is scored by the best F_beta it achieves against *any*
    class (one-vs-rest) on the calibration rows, and the top `keep` cuts are
    retained. Thresholds therefore adapt per concept while the downstream rule
    search is unchanged, so the comparison against the fixed grid is controlled.

    Candidates are interior quantiles of the calibration scores, which keeps the
    search bounded and avoids proposing cuts outside the observed range.
    """
    x = v[calib_mask]
    cand = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_cand)))
    y = y_labels[calib_mask]
    best = []
    for cut in cand:
        fired = x > cut
        n_fired = int(fired.sum())
        if n_fired < MIN_SUPPORT or n_fired > MAX_SUPPORT_FRAC * len(x):
            continue
        s = 0.0
        for c in np.unique(y):
            is_c = y == c
            tp = int((fired & is_c).sum())
            s = max(s, _fbeta(tp, n_fired - tp, int(is_c.sum()))[0])
        best.append((s, cut))
    if not best:                                   # degenerate concept -> fall back
        return np.unique(np.quantile(x, QUANTILES))
    # Greedy selection with a diversity constraint. Picking the top-`keep` cuts by
    # score alone returns near-duplicate thresholds, which starves the downstream
    # DNF search of distinct literals -- the fixed 50/75/90 grid is weak per cut
    # but well spread. We therefore keep the best cut, then require each further
    # cut to sit at least MIN_CUT_SEP apart in quantile space.
    q = {cut: i / (len(cand) - 1) for i, cut in enumerate(cand)} if len(cand) > 1 else {cand[0]: 0.0}
    best.sort(key=lambda t: -t[0])
    picked = []
    for _, cut in best:
        if all(abs(q[cut] - q[c]) >= MIN_CUT_SEP for c in picked):
            picked.append(cut)
        if len(picked) == keep:
            break
    for _, cut in best:                            # top up if diversity was too strict
        if len(picked) == keep:
            break
        if cut not in picked:
            picked.append(cut)
    return np.unique(picked)


def build_literals(scores: pd.DataFrame, affect: np.ndarray | None,
                   calib_mask: np.ndarray,
                   extra_bool: tuple[list[str], np.ndarray] | None = None,
                   y_labels: np.ndarray | None = None,
                   ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(N, n_lit) boolean literal matrix + names + is_negated flags.

    Continuous columns (predicate scores, affect dims) become quantile-cut
    literals; `extra_bool` adds ready-made boolean predicates (e.g. lexical
    says:"..."). Negations of everything are appended; near-tautological
    literals (calib support > MAX_SUPPORT_FRAC) are dropped."""
    cols, names = [], []
    feats = {name: scores[name].to_numpy() for name in scores.columns}
    if affect is not None:
        for j, nm in enumerate(config.AFFECT_FEATURES):
            feats[f"emo:{nm}"] = affect[:, j]
    learn = config.LEARNED_THRESHOLDS and y_labels is not None
    for name, v in feats.items():
        cuts = (learned_cuts(v, calib_mask, y_labels) if learn
                else np.unique(np.quantile(v[calib_mask], QUANTILES)))
        for cut in cuts:
            cols.append(v > cut)
            names.append(f"{name}>{cut:.3f}")
    if extra_bool is not None:
        xb_names, xb = extra_bool
        for j, name in enumerate(xb_names):
            cols.append(xb[:, j].astype(bool))
            names.append(name)
    L = np.stack(cols, 1)
    L = np.concatenate([L, ~L], 1)
    neg = np.array([False] * len(names) + [True] * len(names))
    names = names + [f"NOT {n}" for n in names]
    frac = L[calib_mask].mean(0)
    keep = frac <= MAX_SUPPORT_FRAC
    return L[:, keep], [n for n, k in zip(names, keep) if k], neg[keep]


def _enumerate_clauses(Lc, names, neg, is_c, max_len):
    """Candidate conjunctions up to `max_len` literals that clear the precision floor."""
    n_pos = int(is_c.sum())
    floor = max(MIN_PREC_ABS, min(MIN_LIFT * n_pos / len(is_c), PREC_CAP))
    feat = [(n[4:] if n.startswith("NOT ") else n).split(">")[0] for n in names]

    def score(fired):
        tp = int((fired & is_c).sum())
        return _fbeta(tp, int(fired.sum()) - tp, n_pos)

    singles = [(score(Lc[:, i])[0], i) for i in range(Lc.shape[1])
               if Lc[:, i].sum() >= MIN_SUPPORT]
    pool = [i for _, i in sorted(singles, reverse=True)[:POOL]]
    cands = []

    def consider(idx, fired):
        if fired.sum() < MIN_SUPPORT:
            return None
        if len({feat[i] for i in idx}) < len(idx):
            return None
        if sum(neg[i] for i in idx) <= MAX_NEG and not all(neg[i] for i in idx):
            fb, p, r = score(fired)
            if p >= floor:
                cands.append((fb, p, r, idx, fired))
        return fired

    def extend(idx, fired, depth, start):
        if depth >= max_len:
            return
        for j in range(start, len(pool)):
            f2 = consider(idx + (pool[j],), fired & Lc[:, pool[j]])
            if f2 is not None:
                extend(idx + (pool[j],), f2, depth + 1, j + 1)

    for a in range(len(pool)):
        fa = consider((pool[a],), Lc[:, pool[a]])
        if fa is not None:
            extend((pool[a],), fa, 1, a + 1)
    cands.sort(key=lambda t: -t[0])
    return cands[:CAND_CAP]


def _select_dnf(cands, is_c, max_clauses, beam=BEAM):
    """Beam search over DNFs.

    Greedy selection commits to the single best clause each round, which is
    optimal only if clause contributions are independent -- they are not, since
    clauses overlap heavily in coverage. A beam keeps `beam` partial DNFs alive so
    a clause that looks worse alone but combines better can survive. Returns the
    best DNF found at ANY width up to max_clauses, so a narrower rule wins when
    the extra clause does not pay for itself.
    """
    n_pos = int(is_c.sum())

    def sc(fired):
        tp = int((fired & is_c).sum())
        return _fbeta(tp, int(fired.sum()) - tp, n_pos)[0]

    states = [(0.0, np.zeros(len(is_c), bool), ())]     # (score, coverage, clause ids)
    best = (0.0, ())
    for _ in range(max_clauses):
        nxt = {}
        for s_score, fired_dnf, chosen in states:
            for ci, (fb, p, r, idx, fired) in enumerate(cands[:BEAM_CANDS]):
                if ci in chosen:
                    continue
                u = fired_dnf | fired
                key = u.tobytes()
                v = sc(u)
                if key not in nxt or v > nxt[key][0]:
                    nxt[key] = (v, u, chosen + (ci,))
        if not nxt:
            break
        states = sorted(nxt.values(), key=lambda t: -t[0])[:beam]
        if states[0][0] > best[0]:
            best = (states[0][0], states[0][2])
    return list(best[1])


def _cv_complexity(Lc, names, neg, is_c, grid, folds=CV_FOLDS, seed=config.SEED):
    """Pick (max_len, max_clauses) per class by cross-validation inside calibration.

    The submitted pipeline fixed both at 3. Selecting them on the outer validation
    split would be unreliable at this sample size (see E2.1), so complexity is
    chosen by stratified K-fold *within* the calibration rows: enumeration and DNF
    selection happen on the fold-train rows and F_beta is measured on held-out
    fold rows. Enumeration depends only on max_len, so it is done once per
    (fold, max_len) and reused across widths.
    """
    from sklearn.model_selection import StratifiedKFold
    n_pos = int(is_c.sum())
    if n_pos < folds * 2:
        return max(g[0] for g in grid), max(g[1] for g in grid)
    lens = sorted({g[0] for g in grid})
    widths = sorted({g[1] for g in grid})
    tally = {g: [] for g in grid}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(np.zeros(len(is_c)), is_c):
        for ml in lens:
            cands = _enumerate_clauses(Lc[tr_i], names, neg, is_c[tr_i], ml)
            if not cands:
                continue
            for mc in widths:
                if (ml, mc) not in tally:
                    continue
                pick = _select_dnf(cands, is_c[tr_i], mc)
                if not pick:
                    continue
                cov = np.zeros(len(te_i), bool)
                for ci in pick:                     # re-evaluate clause on held-out rows
                    idx = cands[ci][3]
                    f = np.ones(len(te_i), bool)
                    for i in idx:
                        f &= Lc[te_i][:, i]
                    cov |= f
                tp = int((cov & is_c[te_i]).sum())
                tally[(ml, mc)].append(_fbeta(tp, int(cov.sum()) - tp, int(is_c[te_i].sum()))[0])
    scored = {g: float(np.mean(v)) for g, v in tally.items() if v}
    if not scored:
        return MAX_LEN, MAX_CLAUSES
    # tie-break toward the simpler rule: fewest literals x clauses
    best = max(scored.values())
    return min((g for g, v in scored.items() if v >= best - 1e-9), key=lambda g: g[0] * g[1])


def induce_class_rule(Lc: np.ndarray, names: list[str], neg: np.ndarray,
                      is_c: np.ndarray) -> list[dict]:
    """DNF induction for one class.

    With config.LEARNED_COMPLEXITY the clause length and DNF width are selected by
    cross-validation inside the calibration split and the DNF is chosen by beam
    search; otherwise the original greedy search under the fixed 3x3 caps runs.
    """
    if config.LEARNED_COMPLEXITY:
        # FORCE_COMPLEXITY isolates the search from the complexity selection:
        # set it to (max_len, max_clauses) to run beam search at a fixed shape.
        ml, mc = (FORCE_COMPLEXITY if FORCE_COMPLEXITY
                  else _cv_complexity(Lc, names, neg, is_c, COMPLEXITY_GRID))
        cands = _enumerate_clauses(Lc, names, neg, is_c, ml)
        out = []
        for ci in _select_dnf(cands, is_c, mc):
            fb, p, r, idx, fired = cands[ci]
            out.append({"literals": [names[i] for i in idx], "idx": [int(i) for i in idx],
                        "precision": round(p, 4), "recall": round(r, 4),
                        "support": int(fired.sum()), "max_len": ml, "max_clauses": mc})
        return out

    n_pos = int(is_c.sum())
    floor = max(MIN_PREC_ABS, min(MIN_LIFT * n_pos / len(is_c), PREC_CAP))
    # base feature of each literal (same-feature literals are redundant in a clause)
    feat = [(n[4:] if n.startswith("NOT ") else n).split(">")[0] for n in names]

    def score(fired):
        tp = int((fired & is_c).sum())
        return _fbeta(tp, int(fired.sum()) - tp, n_pos)

    # literal pool: top singles by F_beta with support pruning
    singles = [(score(Lc[:, i])[0], i) for i in range(Lc.shape[1])
               if Lc[:, i].sum() >= MIN_SUPPORT]
    pool = [i for _, i in sorted(singles, reverse=True)[:POOL]]

    # enumerate conjunctions up to MAX_LEN (>=1 positive, <=MAX_NEG negated);
    # low-precision conjunctions stay EXTENDABLE (a further literal can rescue
    # precision) but only floor-clearing clauses become candidates
    cands: list[tuple[float, float, float, tuple[int, ...], np.ndarray]] = []

    def consider(idx: tuple[int, ...], fired: np.ndarray):
        if fired.sum() < MIN_SUPPORT:
            return None                                  # too small to keep or extend
        if len({feat[i] for i in idx}) < len(idx):
            return None                                  # same-feature redundancy
        if sum(neg[i] for i in idx) <= MAX_NEG and not all(neg[i] for i in idx):
            fb, p, r = score(fired)
            if p >= floor:
                cands.append((fb, p, r, idx, fired))
        return fired

    for a in range(len(pool)):
        fa = consider((pool[a],), Lc[:, pool[a]])
        if fa is None:
            continue
        if MAX_LEN < 2:
            continue
        for b in range(a + 1, len(pool)):
            fab = consider((pool[a], pool[b]), fa & Lc[:, pool[b]])
            if fab is None:
                continue
            if MAX_LEN >= 3:
                for d in range(b + 1, len(pool)):
                    consider((pool[a], pool[b], pool[d]), fab & Lc[:, pool[d]])

    # greedy DNF selection
    cands.sort(key=lambda t: -t[0])
    cands = cands[:2000]
    chosen, fired_dnf, best_fb = [], np.zeros(len(is_c), bool), -1.0
    for _ in range(MAX_CLAUSES):
        gain = None
        for fb, p, r, idx, fired in cands:
            u_fb = score(fired_dnf | fired)[0]
            if u_fb > best_fb and (gain is None or u_fb > gain[0]):
                gain = (u_fb, p, r, idx, fired)
        if gain is None:
            break
        best_fb, fired_dnf = gain[0], fired_dnf | gain[4]
        chosen.append({"literals": [names[i] for i in gain[3]],
                       "idx": [int(i) for i in gain[3]],
                       "precision": round(gain[1], 4), "recall": round(gain[2], 4),
                       "support": int(gain[4].sum())})
    return chosen


def induce(scores: pd.DataFrame, y_idx: np.ndarray, calib_mask: np.ndarray,
           affect: np.ndarray | None = None, classes: list[str] | None = None,
           extra_bool: tuple[list[str], np.ndarray] | None = None,
           save: bool = True, save_path=None, label_to_idx: dict | None = None
           ) -> tuple[dict, np.ndarray, list[str]]:
    """Learn a DNF rule set per class. Returns (rule_sets, literal matrix, names).
    `label_to_idx` maps class names to label indices (default: the NeSy-MIS task's)."""
    classes = classes or config.STEREO_CLASSES
    label_to_idx = label_to_idx or config.LABEL_TO_IDX
    L, names, neg = build_literals(scores, affect, calib_mask, extra_bool=extra_bool,
                                   y_labels=y_idx)
    Lc = L[calib_mask]
    rule_sets = {}
    for c in classes:
        is_c = (y_idx[calib_mask] == label_to_idx[c])
        rule_sets[c] = induce_class_rule(Lc, names, neg, is_c)
        print(f"  [induce] {c}: {len(rule_sets[c])} clause(s)", flush=True)
    if save:
        (save_path or LEARNED_RULES_PATH).write_text(json.dumps(
            {c: [{k: v for k, v in cl.items() if k != "idx"} for cl in rs]
             for c, rs in rule_sets.items()}, indent=1), encoding="utf-8")
    return rule_sets, L, names


def evaluate(rule_sets: dict, L: np.ndarray, index,
             classes: list[str] | None = None) -> pd.DataFrame:
    """rules.evaluate()-compatible frame; trigger = best fired clause's precision."""
    classes = classes or config.STEREO_CLASSES
    N = len(index)
    fired_all = np.zeros((N, len(classes)), bool)
    trig_all = np.zeros((N, len(classes)), np.float32)
    out = {}
    for j, c in enumerate(classes):
        for cl in rule_sets[c]:
            f = np.all(L[:, cl["idx"]], axis=1)
            fired_all[:, j] |= f
            trig_all[:, j] = np.where(f, np.maximum(trig_all[:, j], cl["precision"]), trig_all[:, j])
        out[f"{c}_fired"], out[f"{c}_trigger"] = fired_all[:, j], trig_all[:, j]
    any_f = fired_all.any(1)
    masked = np.where(fired_all, trig_all, -np.inf)
    best_j = masked.argmax(1)
    idx_of = np.array([config.LABEL_TO_IDX[c] for c in classes])
    out["sym_pred"] = np.where(any_f, idx_of[best_j], -1)
    out["sym_conf"] = np.where(any_f, trig_all[np.arange(N), best_j], 0.0)
    out["fired_rules"] = [
        [RULE_OF_CLASS[c] for j, c in enumerate(classes) if fired_all[i, j]]
        + ([META_RULE] if any_f[i] else []) for i in range(N)]
    return pd.DataFrame(out, index=index)


def parse_literal(name: str) -> tuple[str, float | None, bool]:
    """Literal name -> (feature, cut or None for boolean predicates, negated).
    Names: 'sem:"kitchen">0.764', 'says:"kitchen"', 'NOT emo:humor>0.659'."""
    negated = name.startswith("NOT ")
    if negated:
        name = name[4:]
    feat, sep, cut = name.rpartition(">")
    if sep and _is_float(cut):
        return feat, float(cut), negated
    return name, None, negated


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def evaluate_features(rule_sets: dict, cont: pd.DataFrame, boolf: pd.DataFrame | None,
                      classes: list[str] | None = None) -> pd.DataFrame:
    """rules.evaluate()-compatible frame computed from FEATURE frames (no literal
    matrix needed) -- the deployed-inference path. `cont` holds continuous
    features (sem:/ex:/emo: columns), `boolf` boolean ones (says: columns)."""
    classes = classes or config.STEREO_CLASSES
    N = len(cont)

    def lit_value(name):
        feat, cut, negated = parse_literal(name)
        if cut is None:
            v = boolf[feat].to_numpy().astype(bool)
        else:
            v = cont[feat].to_numpy() > cut
        return ~v if negated else v

    fired_all = np.zeros((N, len(classes)), bool)
    trig_all = np.zeros((N, len(classes)), np.float32)
    out = {}
    for j, c in enumerate(classes):
        for cl in rule_sets[c]:
            f = np.ones(N, bool)
            for lname in cl["literals"]:
                f &= lit_value(lname)
            fired_all[:, j] |= f
            trig_all[:, j] = np.where(f, np.maximum(trig_all[:, j], cl["precision"]), trig_all[:, j])
        out[f"{c}_fired"], out[f"{c}_trigger"] = fired_all[:, j], trig_all[:, j]
    any_f = fired_all.any(1)
    masked = np.where(fired_all, trig_all, -np.inf)
    best_j = masked.argmax(1)
    idx_of = np.array([config.LABEL_TO_IDX[c] for c in classes])
    out["sym_pred"] = np.where(any_f, idx_of[best_j], -1)
    out["sym_conf"] = np.where(any_f, trig_all[np.arange(N), best_j], 0.0)
    out["fired_rules"] = [
        [RULE_OF_CLASS[c] for j, c in enumerate(classes) if fired_all[i, j]]
        + ([META_RULE] if any_f[i] else []) for i in range(N)]
    return pd.DataFrame(out, index=cont.index)


def main():
    from nesymis.data import dataset as ds
    from nesymis.affect import head as affect_head
    from nesymis.symbolic import grounding, predicate_gen

    emb = ds.load_embeddings()
    y = emb.labels
    calib = emb.df["split"].isin(["train", "val"]).to_numpy()
    protos = predicate_gen.build_prototypes(predicate_gen.generate_prompts(), save=False)
    scores = grounding.compute_scores(emb, protos)
    D, _ = affect_head.load_outputs()

    rule_sets, L, names = induce(scores, y, calib, affect=D)
    print("\n=== Learned rules (train+val; saved to artifacts/learned_rules.json) ===")
    for c, rs in rule_sets.items():
        print(f"  {c}:")
        for cl in rs:
            print(f"    IF {' AND '.join(cl['literals'])}"
                  f"   (prec={cl['precision']:.2f}, rec={cl['recall']:.2f}, n={cl['support']})")


if __name__ == "__main__":
    main()
