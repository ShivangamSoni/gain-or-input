"""Learners and metrics shared by the checks."""
import os

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# cuBLAS reductions are only deterministic with a fixed workspace, which must be set before the
# CUDA context exists; without it the MLP probe's scores vary by up to ~0.02 between runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 42


def seed_list(n: int, base: int = SEED) -> list:
    """[base] + (n - 1) draws from default_rng(base): the 11-seed protocol's construction."""
    return [base] + [int(x) for x in np.random.default_rng(base).integers(0, 2**31 - 1, n - 1)]


def macro_f1(y, p, k: int) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(k)), zero_division=0))


def majority_f1(y, fit, te, k: int) -> float:
    maj = np.bincount(y[fit], minlength=k).argmax()
    return macro_f1(y[te], np.full(int(te.sum()), maj), k)


def shape_model(X, y, tr, va, seed: int = SEED):
    """Gradient-boosted trees on shape features: fitted on the training rows and early-stopped on
    the validation rows (validation loss, 10 iterations without improvement). A strong learner on
    purpose -- a clean verdict should mean something -- whose complexity is chosen out of sample."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, va = np.asarray(tr), np.asarray(va)
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, early_stopping=True,
                                       n_iter_no_change=10, scoring="loss", random_state=seed)
    if va.any():
        m.fit(X[tr], y[tr], X_val=X[va], y_val=y[va])
    else:                                          # no validation rows: hold out 15% of training
        m.set_params(validation_fraction=0.15).fit(X[tr], y[tr])
    return m


def shape_model_f1(X, y, tr, va, te, k: int, seed: int = SEED) -> float:
    """Test macro-F1 of `shape_model`."""
    return macro_f1(y[te], shape_model(X, y, tr, va, seed).predict(X[te]), k)


def bootstrap_lift(y, pred, maj_class, k: int, text_ref=None, B: int = 2000, seed: int = SEED + 1):
    """95% intervals for lift (macro-F1 of `pred` minus that of always predicting `maj_class`, on
    the same resample) and share (macro-F1 of `pred` / `text_ref`) over resampled test rows."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    lifts, shps = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        s = macro_f1(y[i], pred[i], k)
        lifts.append(s - macro_f1(y[i], np.full(len(i), maj_class), k))
        shps.append(s)
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    out = {"lift_ci": [float(lo), float(hi)]}
    if text_ref:
        slo, shi = np.percentile(np.asarray(shps) / text_ref, [2.5, 97.5])
        out["share_ci"] = [float(slo), float(shi)]
    return out


def shape_auc(X, target, fit, te, seed: int = SEED) -> float:
    """AUC of gradient-boosted trees predicting `target` (binary, or one-vs-rest macro) from
    shape features."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    m.fit(X[fit], target[fit])
    pr = m.predict_proba(X[te])
    if pr.shape[1] == 2:
        return float(roc_auc_score(target[te], pr[:, 1]))
    return float(roc_auc_score(target[te], pr, multi_class="ovr", average="macro",
                               labels=m.classes_))


def cramers_v(a, b) -> float:
    """Association between two categorical arrays (0 = independent, 1 = one determines the other)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(tab, (ai, bi), 1)
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    r = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * r))) if r > 0 else 0.0


def probe_f1(X, y, tr, va, te, k: int, seeds, backend: str = "auto") -> list:
    """Test macro-F1 of a small probe per seed, checkpointed on validation macro-F1.
    backend "mlp": LayerNorm -> 512 -> ReLU -> dropout 0.1 -> k, class-weighted, AdamW 1e-3,
    60 epochs (needs PyTorch); "logreg": class-balanced logistic regression; "auto": mlp if
    PyTorch imports, else logreg."""
    if backend == "auto":
        try:
            import torch  # noqa: F401
            backend = "mlp"
        except ImportError:
            backend = "logreg"
    X = np.ascontiguousarray(np.asarray(X, np.float32))
    return [(_mlp if backend == "mlp" else _logreg)(X, y, tr, va, te, k, s) for s in seeds]


def _logreg(X, y, tr, va, te, k, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr])
    Z = sc.transform(X)
    best = None
    for c in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced",
                               random_state=seed).fit(Z[tr], y[tr])
        f = macro_f1(y[va], m.predict(Z[va]), k)
        if best is None or f > best[0]:
            best = (f, m)
    return macro_f1(y[te], best[1].predict(Z[te]), k)


def _mlp(X, y, tr, va, te, k, seed, epochs=60, bs=64, lr=1e-3, wd=1e-4):
    import random

    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    net = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], 512), nn.ReLU(),
                        nn.Dropout(0.1), nn.Linear(512, k)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    cnt = np.bincount(y[tr], minlength=k).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(cnt.sum() / (k * np.clip(cnt, 1, None))).to(dev))
    T = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(np.asarray(y)).long().to(dev)
    tr_i = torch.from_numpy(np.flatnonzero(tr)).to(dev)
    va_i, te_i = np.flatnonzero(va), np.flatnonzero(te)
    rng = np.random.default_rng(seed)
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(tr_i))).to(dev)
        for i in range(0, len(perm), bs):
            j = tr_i[perm[i:i + bs]]
            opt.zero_grad(set_to_none=True)
            crit(net(T[j]), yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            f = macro_f1(y[va_i], net(T[va_i]).argmax(1).cpu().numpy(), k)
        if f > best:
            best, state = f, {n: t.detach().clone() for n, t in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        return macro_f1(y[te_i], net(T[te_i]).argmax(1).cpu().numpy(), k)


def paired_bootstrap(y, Pa, Pb, k: int, B: int = 2000, seed: int = SEED + 1, groups=None) -> dict:
    """Seed-averaged macro-F1 and accuracy differences (a - b) with 95% intervals over resampled
    test rows. Pa, Pb: (seeds, N) predictions of the two systems on the same rows. With `groups`
    (a group id per row, e.g. near-duplicate clusters) whole groups are resampled -- a cluster
    bootstrap, so that near-copies do not count as independent evidence."""
    y, Pa, Pb = np.asarray(y), np.atleast_2d(Pa), np.atleast_2d(Pb)
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None:
        g = np.asarray(groups)
        ids = np.unique(g)
        members = {c: np.flatnonzero(g == c) for c in ids}

        def draw():
            return np.concatenate([members[c] for c in rng.choice(ids, len(ids))])
    else:
        def draw():
            return rng.integers(0, n, n)

    def stats(i):
        yb = y[i]
        df = np.mean([macro_f1(yb, Pa[s, i], k) - macro_f1(yb, Pb[s, i], k) for s in range(len(Pa))])
        da = float((Pa[:, i] == yb).mean() - (Pb[:, i] == yb).mean())
        return df, da

    point = stats(np.arange(n))
    boot = np.array([stats(draw()) for _ in range(B)])
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    return {"d_macro_f1": {"mean": float(point[0]), "ci": [float(lo[0]), float(hi[0])]},
            "d_acc": {"mean": float(point[1]), "ci": [float(lo[1]), float(hi[1])]}}
