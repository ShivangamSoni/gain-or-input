"""Affect head (inference-only).

Trains a small head on the frozen CLIP fused embedding [v; u_ocr] (1024) to
reproduce a 6-d affect vector: 5 tone intensities (regressed to [0,1]
via sigmoid, MSE) + gender_directed (BCE), trained on the training split.

To avoid leakage when the affect-head outputs are later fed into the neural
classifier (a stacked model), TRAIN-split predictions are produced out-of-fold
(K-fold); val/test rows are predicted by a head trained on all train rows.
Caches:
  * affect_outputs.npy (N,6) in [0,1]  -> neural 6-d source + policy state
  * affect_hidden.npy  (N,512)         -> hidden representation

Only the cache reader is kept in the release: the deployed systems are affect-free
and feed this slot zeros.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from nesymis import config
from nesymis.base import set_seed
from nesymis.data import dataset as ds

_device = "cuda" if torch.cuda.is_available() else "cpu"


class AffectHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_tones: int = 5, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.tone = nn.Linear(hidden, n_tones)
        self.gd = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return h, torch.sigmoid(self.tone(h)), torch.sigmoid(self.gd(h))


def _train_head(X, tone_t, gd_t, idx, cfg) -> AffectHead:
    set_seed(config.SEED)
    model = AffectHead(X.shape[1], cfg["hidden"], dropout=cfg["dropout"]).to(_device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    mse, bce = nn.MSELoss(), nn.BCELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(tone_t[idx]),
                      torch.from_numpy(gd_t[idx])),
        batch_size=cfg["batch_size"], shuffle=True,
    )
    model.train()
    for _ in range(cfg["epochs"]):
        for xb, tb, gb in loader:
            xb, tb, gb = xb.to(_device), tb.to(_device), gb.to(_device)
            opt.zero_grad()
            _, tone, gd = model(xb)
            loss = mse(tone, tb) + bce(gd.squeeze(1), gb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _predict(model, X) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    h, tone, gd = model(torch.from_numpy(X).to(_device))
    out6 = torch.cat([tone, gd], dim=1)
    return h.cpu().numpy().astype(np.float32), out6.cpu().numpy().astype(np.float32)


def load_outputs() -> tuple[np.ndarray, np.ndarray]:
    return np.load(config.AFFECT_OUTPUTS_PATH), np.load(config.AFFECT_HIDDEN_PATH)


def predict_affect(model: AffectHead, vu: np.ndarray) -> np.ndarray:
    """frozen CLIP [v;u] (N,1024) -> 6-d affect outputs in [0,1]."""
    _, out6 = _predict(model, vu.astype(np.float32))
    return out6


if __name__ == "__main__":
    build_cache()
