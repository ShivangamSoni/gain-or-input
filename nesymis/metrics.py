"""Shared evaluation metrics and pretty-printing for the paper's tables."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from nesymis import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, and per-class F1 keyed by label name."""
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(config.NUM_CLASSES)), zero_division=0)
    per = f1_score(y_true, y_pred, average=None, labels=list(range(config.NUM_CLASSES)), zero_division=0)
    per_class = {config.IDX_TO_LABEL[i]: float(per[i]) for i in range(config.NUM_CLASSES)}
    return {"acc": float(acc), "macro_f1": float(macro), "per_class_f1": per_class}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))


def format_row(name: str, m: dict) -> str:
    p = m["per_class_f1"]
    return (
        f"{name:<16} acc={m['acc']:.3f}  macroF1={m['macro_f1']:.3f}  | "
        f"NonMis={p['non_stereotype']:.3f}  Kit={p['kitchen']:.3f}  "
        f"Lead={p['leadership']:.3f}  Work={p['working']:.3f}  Shop={p['shopping']:.3f}"
    )


def print_table3_header() -> None:
    print(
        f"{'Model':<16} {'Acc':>6} {'MacF1':>6} | "
        f"{'NonMis':>7} {'Kitchen':>7} {'Leader':>7} {'Working':>7} {'Shop':>7}"
    )
    print("-" * 78)


def print_table3_row(name: str, m: dict) -> None:
    p = m["per_class_f1"]
    print(
        f"{name:<16} {m['acc']:>6.3f} {m['macro_f1']:>6.3f} | "
        f"{p['non_stereotype']:>7.3f} {p['kitchen']:>7.3f} {p['leadership']:>7.3f} "
        f"{p['working']:>7.3f} {p['shopping']:>7.3f}"
    )
