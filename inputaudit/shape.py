"""Nine content-free text-shape features. They read how a text is shaped, never its words."""
import re

import numpy as np

FEATURES = ("n_chars", "n_words", "mean_word_len", "max_word_len", "upper_ratio",
            "digit_ratio", "punct_ratio", "non_alnum_ratio", "type_token_ratio")
_PUNCT = re.compile(r"[^\w\s]")


def shape_features(texts) -> np.ndarray:
    """(N, 9) float array, columns as in FEATURES. Missing texts count as empty."""
    rows = []
    for t in texts:
        s = "" if t is None or (isinstance(t, float) and t != t) else str(t)
        w = s.split()
        n, nw = len(s), len(w)
        lens = [len(x) for x in w] or [0]
        alnum = sum(ch.isalnum() for ch in s)
        rows.append([
            n,
            nw,
            float(np.mean(lens)),
            float(np.max(lens)),
            (sum(ch.isupper() for ch in s) / n) if n else 0.0,
            (sum(ch.isdigit() for ch in s) / n) if n else 0.0,
            (len(_PUNCT.findall(s)) / n) if n else 0.0,
            ((n - alnum) / n) if n else 0.0,
            (len(set(x.lower() for x in w)) / nw) if nw else 0.0,
        ])
    return np.asarray(rows, dtype=np.float64)
