"""Gender-directed predicate from the 6-d affect head outputs.

Column order = [sarcasm, contempt, anger, humor, neutral, gender_directed], all
in [0,1]. The calibrated gender-directed threshold is used by the inference
pipeline to surface a human-readable explanation flag.
"""
from __future__ import annotations

import numpy as np

from nesymis import config

GD = 5
NON = config.LABEL_TO_IDX["non_stereotype"]


def calibrate_gd(affect: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Pick the gender_directed threshold maximizing misogyny F1 on a calibration mask."""
    from sklearn.metrics import f1_score

    p = affect[mask, GD]
    mis = (y[mask] != NON).astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(mis, (p > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t
