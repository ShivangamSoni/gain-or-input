"""External #3 - EXIST 2024 Memes: shared data loaders (English subset).

Best-matched external benchmark: memes (same modality as our data), English,
~balanced. Test gold is withheld, so we use the labeled TRAINING memes.

This module is now a LOADER ONLY. load_all_en() + embed() + task4_hard() are
imported by data_exist2024_6class.py and by the unified concept-layer transfer
harness (experiments/concept_scorecard_eval.py), which is where every EXIST evaluation
(zero-shot / binary / multiclass) now lives under the deployed concept grounding.
The former standalone pool-based eval was retired when the concept-abstraction
layer became the deployed grounding.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset", "EXIST", "2024 EXIST", "EXIST 2024 Memes Dataset", "training")
GT = os.path.join(BASE, "EXIST2024_training.json")
CACHE = os.path.join(ROOT, "artifacts", "exist_clip.npz")   # shared with the task-6 script
POS = "sexist"
LANG = "en"


def clean(t):
    return re.sub(r"\s+", " ", (t or "").replace("�", " ")).strip()


def load_all_en():
    """ALL English memes (no label filtering) so the cache is shared across tasks."""
    d = json.load(open(GT, encoding="utf-8"))
    ids, texts, paths, raw = [], [], [], []
    for k, v in d.items():
        if v["lang"] != LANG:
            continue
        p = os.path.join(BASE, v["path_memes"])
        if not os.path.isfile(p):
            continue
        ids.append(k); texts.append(clean(v["text"])); paths.append(p); raw.append(v)
    return ids, texts, paths, raw


def embed(ids, paths, texts):
    if os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == ids:
            return z["vu"], z["u"]
    from nesymis.encoders.clip_encoder import encode_images, encode_texts
    B = 256
    v = np.concatenate([encode_images(paths[i:i + B]) for i in range(0, len(paths), B)])
    u = np.concatenate([encode_texts(texts[i:i + B]) for i in range(0, len(texts), B)])
    vu = np.concatenate([v, u], axis=1).astype(np.float32)
    np.savez(CACHE, ids=np.array(ids, object), vu=vu, u=u)
    return vu, u


def task4_hard(v):
    c = Counter(x.upper() for x in v["labels_task4"])
    if c.get("YES", 0) > c.get("NO", 0):
        return 1
    if c.get("NO", 0) > c.get("YES", 0):
        return 0
    return -1                                          # annotator tie -> drop
