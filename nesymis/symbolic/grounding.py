"""
Symbol grounding (Path-B, step 1).

Each high-level predicate (woman_present, {class}_context, {class}_stereotype,
{class}_text_stereotype) is grounded by cosine similarity between a frozen CLIP
embedding and a small ensemble of hand-written prompts. Image-side predicates
score against the image embedding v; *_text_stereotype scores against the text
embedding u. Prompt ensembles are averaged into a unit prototype, so the score
is a single cosine per (sample, predicate). Fixed thresholds then yield boolean
predicate activations (Table II).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nesymis import config
from nesymis.encoders.clip_encoder import encode_texts

# predicate -> which embedding it scores against
PREDICATES = list(config.PROMPTS.keys())


def predicate_modality(name: str) -> str:
    if name.endswith("_text_stereotype"):
        return "text"
    return "image"  # woman_present, *_context, *_stereotype


def threshold_for(name: str, thresholds: dict) -> float:
    if name == "woman_present":
        return thresholds["woman_present"]
    if name.endswith("_text_stereotype"):
        return thresholds["text_stereotype"]
    if name.endswith("_context"):
        return thresholds["context"]
    if name.endswith("_stereotype"):
        return thresholds["stereotype"]
    raise KeyError(name)


_PROTO_CACHE = config.ARTIFACTS_DIR / "prototypes.npz"


def build_prototypes(use_cache: bool = True) -> dict[str, np.ndarray]:
    """Unit prototype vector per predicate (mean of normalized prompt embeddings)."""
    if use_cache and _PROTO_CACHE.exists():
        data = np.load(_PROTO_CACHE)
        if set(data.files) == set(config.PROMPTS):
            return {k: data[k] for k in data.files}
    protos = {}
    for name, prompts in config.PROMPTS.items():
        embs = encode_texts(prompts)               # (k, 512), already L2-normalized
        mean = embs.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        protos[name] = mean.astype(np.float32)
    np.savez(_PROTO_CACHE, **protos)
    return protos


def compute_scores(emb, prototypes: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """(N, num_predicates) cosine scores, row-aligned with emb.df."""
    protos = prototypes or build_prototypes()
    cols = {}
    for name in PREDICATES:
        # image predicates -> image emb; *_text_stereotype -> caption emb
        mat = emb.image if predicate_modality(name) == "image" else emb.symbolic_text
        cols[name] = mat @ protos[name]            # cosine (both unit-norm)
    return pd.DataFrame(cols, index=emb.df.index)


def activations(scores: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Boolean predicate activations: score > threshold."""
    out = {name: scores[name] > threshold_for(name, thresholds) for name in PREDICATES}
    return pd.DataFrame(out, index=scores.index)
