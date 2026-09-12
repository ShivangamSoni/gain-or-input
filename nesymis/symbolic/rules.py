"""
Symbolic rule reasoning (Path-B, step 2) -- Table II.

R1..R4 (one per stereotype class c):
    woman_present AND {c}_context AND ({c}_stereotype OR {c}_text_stereotype)
        => {c} Misogyny
R5 (meta):  any subtype  => misogyny

Each fired subtype rule carries a confidence = max({c}_stereotype score,
{c}_text_stereotype score), used by the inference-time decision policy. The
rule-only symbolic prediction picks the highest-confidence fired subtype, and
abstains (-> non_stereotype) when no rule fires.
"""
from __future__ import annotations

import pandas as pd

from nesymis import config
from nesymis.symbolic import grounding

RULE_OF_CLASS = {"kitchen": "R1", "leadership": "R2", "working": "R3", "shopping": "R4"}
CLASS_OF_RULE = {v: k for k, v in RULE_OF_CLASS.items()}
META_RULE = "R5"


def evaluate(scores: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame (row-aligned with `scores`) with columns:
      fired_rules : list[str]      e.g. ["R1", "R5"]
      sym_pred    : int            class idx of highest-confidence fired subtype, else -1
      sym_conf    : float          trigger score of that subtype, else 0.0
      <c>_fired   : bool           per-class subtype rule fired
      <c>_trigger : float          per-class trigger score (max of the two stereotype cues)
    """
    thresholds = thresholds or config.THRESHOLDS
    act = grounding.activations(scores, thresholds)

    records = []
    for idx in scores.index:
        woman = bool(act.at[idx, "woman_present"])
        fired, triggers = [], {}
        per_class = {}
        for c in config.STEREO_CLASSES:
            ctx = woman and bool(act.at[idx, f"{c}_context"])
            trig = bool(act.at[idx, f"{c}_stereotype"]) or bool(act.at[idx, f"{c}_text_stereotype"])
            trigger_score = float(max(scores.at[idx, f"{c}_stereotype"], scores.at[idx, f"{c}_text_stereotype"]))
            per_class[f"{c}_fired"] = ctx and trig
            per_class[f"{c}_trigger"] = trigger_score
            if ctx and trig:
                fired.append(RULE_OF_CLASS[c])
                triggers[c] = trigger_score
        if fired:
            fired.append(META_RULE)
            best_c = max(triggers, key=triggers.get)
            sym_pred = config.LABEL_TO_IDX[best_c]
            sym_conf = triggers[best_c]
        else:
            sym_pred, sym_conf = -1, 0.0
        records.append({"fired_rules": fired, "sym_pred": sym_pred, "sym_conf": sym_conf, **per_class})

    return pd.DataFrame(records, index=scores.index)


def rule_only_prediction(rule_df: pd.DataFrame) -> "pd.Series":
    """Path-B baseline: fired subtype -> its class, else non_stereotype (abstain)."""
    return rule_df["sym_pred"].apply(lambda p: p if p >= 0 else config.LABEL_TO_IDX["non_stereotype"])


def rule_precision(rule_df: pd.DataFrame, y) -> dict:
    """P(true class == c | subtype rule R_c fired), per stereotype class."""
    import numpy as np

    y = np.asarray(y)
    out = {}
    for c in config.STEREO_CLASSES:
        f = rule_df[f"{c}_fired"].to_numpy().astype(bool)
        ci = config.LABEL_TO_IDX[c]
        out[c] = float(((y == ci) & f).sum() / max(int(f.sum()), 1))
    return out
