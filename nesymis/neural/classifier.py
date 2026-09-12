"""
P4 -- affect-aware neural classifier (Path-A').

The 6-d affect head output is projected to `affect_dim`
(the ablation knob), fused with the frozen CLIP [v;u] via one of the fusion blocks,
then classified. Training mirrors the original Path-A (cross-entropy, best-val-
macro-F1 checkpoint, same seed) so affect results are directly comparable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nesymis import config
from nesymis.metrics import compute_metrics
from nesymis.fusion.fusion_blocks import FUSIONS
from nesymis.base import set_seed

_device = "cuda" if torch.cuda.is_available() else "cpu"


class Classifier(nn.Module):
    """`vu_mu`/`vu_sd` are non-trainable buffers holding the train-split feature
    statistics; they are the identity (0/1) unless the trainer fills them, so
    standardization travels with the checkpoint and inference needs no change."""

    def __init__(self, affect_raw_dim, fusion="concat", affect_dim=64, num_classes=config.NUM_CLASSES,
                 vu_dim=2 * config.EMB_DIM, hidden=None):
        super().__init__()
        self.proj_e = nn.Linear(affect_raw_dim, affect_dim)
        self.fusion = FUSIONS[fusion](vu_dim=vu_dim, affect_dim=affect_dim, **config.AFFECT_FUSION)
        self.register_buffer("vu_mu", torch.zeros(vu_dim))
        self.register_buffer("vu_sd", torch.ones(vu_dim))
        din = self.fusion.d_out
        if hidden is None:
            self.head = nn.Linear(din, num_classes)
        else:
            self.head = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(hidden, num_classes))

    def forward(self, vu, e_raw):
        return self.head(self.fusion((vu - self.vu_mu) / self.vu_sd, self.proj_e(e_raw)))


@torch.no_grad()
def predict_logits(model, vu, e_raw) -> np.ndarray:
    model.eval()
    out = model(torch.from_numpy(vu).float().to(_device), torch.from_numpy(e_raw).float().to(_device))
    return out.cpu().numpy()


def train_classifier(vu, e_raw, y, train_mask, val_mask, fusion="concat",
                             affect_dim=64, cfg=None, seed=config.SEED, verbose=False, num_classes=None):
    """Returns (model, best_val_macro_f1). vu, e_raw, y are full-length arrays; masks select rows.
    `num_classes` defaults to the NeSy-MIS task's (config.NUM_CLASSES)."""
    cfg = cfg or config.MLP
    k = num_classes or config.NUM_CLASSES
    set_seed(seed)
    model = Classifier(e_raw.shape[1], fusion=fusion, affect_dim=affect_dim, num_classes=k,
                              vu_dim=vu.shape[1], hidden=cfg["hidden"]).to(_device)
    if cfg.get("standardize", False):
        tr_vu = vu[train_mask]
        model.vu_mu.copy_(torch.from_numpy(tr_vu.mean(0)).to(_device))
        model.vu_sd.copy_(torch.from_numpy(np.clip(tr_vu.std(0), 1e-6, None)).to(_device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weight = None
    if cfg.get("class_weighted", False):
        counts = np.bincount(y[train_mask], minlength=k).astype(np.float32)
        weight = torch.from_numpy(counts.sum() / (k * np.clip(counts, 1, None))).to(_device)
    crit = nn.CrossEntropyLoss(weight=weight)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(vu[train_mask]).float(),
                      torch.from_numpy(e_raw[train_mask]).float(),
                      torch.from_numpy(y[train_mask]).long()),
        batch_size=cfg["batch_size"], shuffle=True,
    )
    best_f1, best_state = -1.0, None
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, eb, yb in loader:
            xb, eb, yb = xb.to(_device), eb.to(_device), yb.to(_device)
            opt.zero_grad()
            loss = crit(model(xb, eb), yb)
            loss.backward()
            opt.step()
        vp = predict_logits(model, vu[val_mask], e_raw[val_mask]).argmax(1)
        m = compute_metrics(y[val_mask], vp)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == cfg["epochs"] - 1):
            print(f"  ep {ep:>3} val macroF1={m['macro_f1']:.3f}")
    model.load_state_dict(best_state)
    return model, best_f1


def crossfit_logits(vu, e_raw, y, train_mask, val_mask, fusion="concat", affect_dim=64,
                    cfg=None, seed=config.SEED, nfold=None, verbose=False, fit_fn=None,
                    num_classes=None):
    """Neural logits for the decision layer, out-of-fold on train+val.

    The decision layer learns *when the neural branch is wrong*. Fitted on the
    same rows the branch was trained on, it sees an almost error-free branch
    (train macro-F1 ~1.0 for a properly-trained head) and therefore under-uses
    the symbolic evidence, then meets real errors at test time. Standard stacking
    practice is to feed it out-of-fold predictions instead.

    Every train+val row gets a logit from a model that never saw it -- including
    for checkpoint selection, which uses a held-out slice of each fold's OWN
    training rows, so no selection leakage enters the returned logits. Rows
    outside train+val (i.e. test) keep the full-data model, which is what is
    actually deployed at inference.

    Returns (full_data_model, logits). nfold=None reads config.POLICY_CROSSFIT;
    nfold falsy => plain in-sample logits (the pre-Phase-0 behaviour). `fit_fn`
    overrides how a model is fitted -- fit_fn(train_mask, val_mask) -> model --
    so callers with their own trainer (e.g. the 6-class external head) get the
    same treatment instead of silently falling back to in-sample logits.
    """
    from sklearn.model_selection import StratifiedKFold

    nfold = config.POLICY_CROSSFIT if nfold is None else nfold
    if fit_fn is None:
        def fit_fn(m_tr, m_va):
            return train_classifier(vu, e_raw, y, m_tr, m_va, fusion=fusion,
                                            affect_dim=affect_dim, cfg=cfg, seed=seed,
                                            num_classes=num_classes)[0]
    model = fit_fn(train_mask, val_mask)
    logits = predict_logits(model, vu, e_raw)
    if not nfold:
        return model, logits

    idx = np.flatnonzero(train_mask | val_mask)
    rng = np.random.default_rng(seed)
    for k, (fit_i, out_i) in enumerate(
            StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(idx, y[idx])):
        fit_rows, out_rows = idx[fit_i], idx[out_i]
        # checkpoint-selection slice taken from this fold's own training rows
        sel = rng.permutation(len(fit_rows))[: max(1, int(0.1 * len(fit_rows)))]
        m_va = np.zeros(len(y), bool); m_va[fit_rows[sel]] = True
        m_tr = np.zeros(len(y), bool); m_tr[fit_rows] = True; m_tr[fit_rows[sel]] = False
        fm = fit_fn(m_tr, m_va)
        logits[out_rows] = predict_logits(fm, vu[out_rows], e_raw[out_rows])
        if verbose:
            print(f"  [crossfit] fold {k + 1}/{nfold}", flush=True)
    return model, logits
