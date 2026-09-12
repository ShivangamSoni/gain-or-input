"""Base neural classifier (Path-A) + shared seeding.

Used by the pathway ablation (image/text/both modalities); also provides set_seed,
imported by the affect / policy modules.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nesymis import config
from nesymis.data import dataset as ds
from nesymis.metrics import compute_metrics
from nesymis.neural.mlp import NeuralClassifier

_device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = config.SEED) -> None:
    """Seed every RNG and constrain GPU nondeterminism.

    Seeding alone does not make this pipeline bit-reproducible: cuBLAS reductions
    vary run to run, and because training keeps the best-val-macro-F1 checkpoint,
    a float-level difference can select a different epoch entirely and move a
    minority-class F1 by a visible amount. cudnn.benchmark is disabled so kernel
    selection cannot vary with it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # CUBLAS_WORKSPACE_CONFIG is set in config.py,
                                             # which must happen before the CUDA context


def _weight_path(modality: str) -> str:
    return str(config.WEIGHTS_DIR / f"mlp_{modality}.pt")


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X).float().to(_device)).cpu().numpy()


def train_classifier(emb: ds.Embeddings, modality: str = "both", cfg: dict | None = None,
                     class_weighted: bool | None = None, seed: int = config.SEED, verbose: bool = False):
    """Train a frozen-CLIP classifier for one modality; keep best-val-macro-F1.

    Honours the same config.MLP switches as the deployed Path-A trainer
    (class_weighted, standardize, hidden) so the modality ablation in Table 3 and
    the deployed branch are trained by one recipe. `class_weighted` may still be
    passed explicitly to override the config for a targeted ablation.
    """
    cfg = cfg or config.MLP
    if class_weighted is None:
        class_weighted = cfg.get("class_weighted", False)
    set_seed(seed)
    X_tr, y_tr, _ = emb.get_split("train", modality)
    X_va, y_va, _ = emb.get_split("val", modality)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
                        batch_size=cfg["batch_size"], shuffle=True)
    model = NeuralClassifier(X_tr.shape[1], config.NUM_CLASSES, hidden=cfg["hidden"]).to(_device)
    if cfg.get("standardize", False):
        model.mu.copy_(torch.from_numpy(X_tr.mean(0)).to(_device))
        model.sd.copy_(torch.from_numpy(np.clip(X_tr.std(0), 1e-6, None)).to(_device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weight = None
    if class_weighted:
        counts = np.bincount(y_tr, minlength=config.NUM_CLASSES).astype(np.float32)
        weight = torch.from_numpy(counts.sum() / (config.NUM_CLASSES * np.clip(counts, 1, None))).to(_device)
    crit = nn.CrossEntropyLoss(weight=weight)

    best_f1, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(_device), yb.to(_device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        m = compute_metrics(y_va, predict_logits(model, X_va).argmax(1))
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "modality": modality, "input_dim": X_tr.shape[1]}, _weight_path(modality))
    return model, best_f1


def load_classifier(modality: str = "both") -> NeuralClassifier:
    ckpt = torch.load(_weight_path(modality), map_location=_device)
    model = NeuralClassifier(ckpt["input_dim"], config.NUM_CLASSES, hidden=config.MLP["hidden"]).to(_device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
