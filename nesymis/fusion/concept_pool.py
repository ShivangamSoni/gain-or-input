"""Concept-routed decision -- the combined paper's decision layer (C4).

The label comes from log-linear pooling of two experts that read the SAME inputs (the image
and its OCR text, both embedded by the frozen CLIP encoder):

    score_k = (1 - w) * log p_concept(k) + w * log p_neural(k)

p_concept is a linear scorecard over the named concept scores; p_neural is the neural
classifier. Because both terms enter the decision, the concept evidence reported for an output
is part of what produced it. That is the difference from the original design's rule bonuses,
which were added to neural logits and changed 5 of 490 decisions on the fair protocol
(experiments/evidence_feasibility.py).

Exact attribution. Within one row a log-softmax difference equals the logit difference, so the
concept part of the winning margin (label k vs runner-up r) splits exactly over concepts:

    (1 - w) * [ (b_k - b_r) + sum_j (W_kj - W_rj) * z_j ]

z_j is concept j's standardised score. Each summand is concept j's contribution to this
decision, and setting z_j = 0 (the concept at its training mean) removes exactly that amount --
which is what the deletion test measures.

Rules are evaluated here too, for all five classes, but they are evidence that supports or
contests the label, not inputs to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from nesymis import config
from nesymis.symbolic.rule_induction import parse_literal

K = config.NUM_CLASSES
CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
W_GRID = tuple(round(float(x), 1) for x in np.linspace(0.0, 1.0, 11))
W_TOL = 0.01        # the operating point may cost at most this much validation macro-F1
STATES = ("SUPPORTED", "MIXED", "CONTESTED", "NONE")


def macro_f1(y, p, k: int | None = None) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(k or K)), zero_division=0))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


# --------------------------------------------------------------------------- #
# The concept expert
# --------------------------------------------------------------------------- #
@dataclass
class ConceptScorecard:
    """Multinomial logistic regression over standardised concept scores. Trained on the TRAIN
    split only, so its validation predictions are held out for choosing C and the pooling
    weight."""
    names: list
    mu: np.ndarray          # (J,) train mean of each concept score
    sd: np.ndarray          # (J,) train standard deviation
    coef: np.ndarray        # (K, J)
    intercept: np.ndarray   # (K,)
    C: float
    penalty: str = "l2"
    # Class-balanced training fits the scorecard under a uniform class prior, which pulls it
    # toward the small classes. log_prior (train class log-frequencies) shifts it back to the
    # data prior; None = no shift. Only the offset changes -- weights and attributions do not.
    log_prior: np.ndarray | None = None

    @property
    def offset(self) -> np.ndarray:
        return self.intercept if self.log_prior is None else self.intercept + self.log_prior

    def with_prior_correction(self, y, tr) -> "ConceptScorecard":
        cnt = np.bincount(np.asarray(y)[tr], minlength=len(self.intercept)).astype(np.float64)
        return ConceptScorecard(self.names, self.mu, self.sd, self.coef, self.intercept,
                                self.C, self.penalty, np.log(cnt / cnt.sum()))

    def standardise(self, S) -> np.ndarray:
        return (np.asarray(S, np.float64) - self.mu) / self.sd

    def logits(self, S=None, Z=None) -> np.ndarray:
        Z = self.standardise(S) if Z is None else Z
        return Z @ self.coef.T + self.offset

    @classmethod
    def fit(cls, S, names, y, tr, va, penalty="l2", Cs=CS, seed=config.SEED,
            k: int | None = None) -> "ConceptScorecard":
        k = k or K
        sc = StandardScaler().fit(S[tr])
        Z = sc.transform(S)
        solver, iters = ("saga", 8000) if penalty == "l1" else ("lbfgs", 3000)
        best = None
        for c in Cs:
            m = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=iters,
                                   class_weight="balanced", random_state=seed).fit(Z[tr], y[tr])
            f = macro_f1(y[va], m.predict(Z[va]), k)
            if best is None or f > best[0]:
                best = (f, c, m)
        _, c, m = best
        assert list(m.classes_) == list(range(k)), "a class is missing from the train split"
        coef, icpt = m.coef_.astype(np.float64), m.intercept_.astype(np.float64)
        if coef.shape[0] == 1:          # binary: sklearn fits one logit z; the two-class softmax with
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])      # logits (-z/2, +z/2) gives the same
            icpt = np.array([-icpt[0] / 2, icpt[0] / 2])       # probabilities, and W_1 - W_0 = w
        return cls(list(names), sc.mean_.astype(np.float64), sc.scale_.astype(np.float64),
                   coef, icpt, float(c), penalty)

    def save(self, path) -> None:
        np.savez(path, names=np.array(self.names, dtype=object), mu=self.mu, sd=self.sd,
                 coef=self.coef, intercept=self.intercept, C=self.C, penalty=self.penalty,
                 log_prior=np.zeros(K) if self.log_prior is None else self.log_prior,
                 prior_corrected=self.log_prior is not None)

    @classmethod
    def load(cls, path) -> "ConceptScorecard":
        z = np.load(path, allow_pickle=True)
        lp = z["log_prior"] if "prior_corrected" in z.files and bool(z["prior_corrected"]) else None
        return cls([str(n) for n in z["names"]], z["mu"], z["sd"], z["coef"], z["intercept"],
                   float(z["C"]), str(z["penalty"]), lp)


# --------------------------------------------------------------------------- #
# Pooling and exact attribution
# --------------------------------------------------------------------------- #
def pooled_scores(lc: np.ndarray, ln: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * lc + w * ln


def select_weight(lc_val, ln_val, y_val, grid=W_GRID, tol=W_TOL, acc_tol=None) -> float:
    """The most concept-weighted w whose validation macro-F1 is within `tol` of the neural
    classifier alone (w = 1) -- and, when `acc_tol` is set, whose validation accuracy is also
    within `acc_tol`. Macro-F1 alone let the class-balanced scorecard trade majority-class
    accuracy for minority recall (R1: accuracy 0.753 vs 0.780). Validation rows must carry
    held-out predictions from both experts (scorecard on train; neural logits out-of-fold)."""
    k = ln_val.shape[1]
    pn = ln_val.argmax(1)
    ref_f1, ref_acc = macro_f1(y_val, pn, k), float((pn == y_val).mean())
    for w in sorted(grid):
        p = pooled_scores(lc_val, ln_val, w).argmax(1)
        if macro_f1(y_val, p, k) >= ref_f1 - tol and (
                acc_tol is None or float((p == y_val).mean()) >= ref_acc - acc_tol):
            return float(w)
    return 1.0


def decompose(card: ConceptScorecard, Z: np.ndarray, ln: np.ndarray, w: float) -> dict:
    """Pooled decision plus its exact split into a concept part, a neural part, and
    per-concept contributions to the winning margin."""
    lc = log_softmax(card.logits(Z=Z))
    s = pooled_scores(lc, ln, w)
    order = np.argsort(s, 1)
    k, r = order[:, -1], order[:, -2]
    i = np.arange(len(s))
    mc = (1.0 - w) * (lc[i, k] - lc[i, r])                       # concept part of the margin
    mn = w * (ln[i, k] - ln[i, r])                               # neural part of the margin
    contrib = (1.0 - w) * (card.coef[k] - card.coef[r]) * Z      # (N, J) per-concept parts of mc
    bias = (1.0 - w) * (card.offset[k] - card.offset[r])
    return {"pred": k, "runner_up": r, "lc": lc, "scores": s, "margin_concept": mc,
            "margin_neural": mn, "contrib": contrib, "bias": bias,
            "concept_share": np.abs(mc) / (np.abs(mc) + np.abs(mn) + 1e-12)}


# --------------------------------------------------------------------------- #
# Rules as evidence (all five classes)
# --------------------------------------------------------------------------- #
def fire_rules(rule_sets: dict, cont, classes=None, label_to_idx: dict | None = None):
    """Name-based rule evaluation for any class list, including non_stereotype (the legacy
    evaluator only names the four stereotype rules). Returns fired (N,K), best fired-clause
    precision (N,K), and the fired clauses per row. `label_to_idx` (class name -> column)
    defaults to the NeSy-MIS task's."""
    classes = list(classes or config.LABELS)
    label_to_idx = label_to_idx or config.LABEL_TO_IDX
    K = len(label_to_idx)
    N = len(cont)
    cache = {}

    def lit(name):
        if name not in cache:
            feat, cut, neg = parse_literal(name)
            if cut is None:
                raise ValueError(f"boolean literal {name!r} has no column in the concept frame")
            v = cont[feat].to_numpy() > cut
            cache[name] = ~v if neg else v
        return cache[name]

    fired = np.zeros((N, K), bool)
    trig = np.zeros((N, K), np.float32)
    clauses = [[] for _ in range(N)]
    for c in classes:
        j = label_to_idx[c]
        for cl in rule_sets.get(c, []):
            f = np.ones(N, bool)
            for ln_ in cl["literals"]:
                f &= lit(ln_)
            fired[:, j] |= f
            trig[:, j] = np.where(f, np.maximum(trig[:, j], cl["precision"]), trig[:, j])
            for i in np.flatnonzero(f):
                clauses[i].append({"class": c, "literals": list(cl["literals"]),
                                   "precision": float(cl["precision"])})
    return fired, trig, clauses


def evidence_state(pred, fired, trig):
    """SUPPORTED: a rule for the label fired, none against. MIXED: both. CONTESTED: only rules
    for other classes. NONE: no rule fired. Also a suspicion score (best contesting precision
    minus best supporting precision)."""
    n = len(pred)
    i = np.arange(n)
    sup = fired[i, pred]
    other = fired.copy()
    other[i, pred] = False
    con = other.any(1)
    sup_p = np.where(sup, trig[i, pred], 0.0)
    con_p = np.where(other, trig, 0.0).max(1)
    state = np.where(sup & con, "MIXED",
                     np.where(sup, "SUPPORTED", np.where(con, "CONTESTED", "NONE")))
    return state, (con_p - sup_p).astype(np.float64)


def concept_label(col: str) -> str:
    """'concept:"kitchen-place-cook"' -> 'kitchen-place-cook'."""
    return col.split('"')[1] if '"' in col else col
