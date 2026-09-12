"""D5's two-encoder finding with 10 seeds instead of 3.

The paper's D5 rests on two tasks that look image-independent under CLIP ViT-B/32 but not under
SigLIP SO400M (HarMeme 3-class, EXIST 2024 6-class). Repeated 3-seed runs of a probe can differ by
up to ~0.02 macro-F1, so the value of the image (both - text) is re-estimated here over 10 seeds
per view and encoder, with the same probes, splits and cached features as the 3-seed runs (whose
first three seeds these reproduce exactly).

  python experiments/d5_seed_stability.py        # GPU optional; a few minutes
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import d5_two_encoders as sg
import d5_modality_probes as ts
from nesymis import config

TASKS = ["harmeme-3", "exist-6"]
SEEDS = 10
OUT = config.ARTIFACTS_DIR / "d5_seed_stability.json"


def main():
    tmp = {"clip-b32": str(config.ARTIFACTS_DIR / "_d5_seeds_clip.json"),
           "siglip-so400m": str(config.ARTIFACTS_DIR / "_d5_seeds_siglip.json")}
    ts.OUT, sg.OUT = tmp["clip-b32"], tmp["siglip-so400m"]    # never overwrite the 3-seed artefacts
    ts.cmd_run(",".join(TASKS), SEEDS)
    sg.cmd_probe(",".join(TASKS), SEEDS)
    res = {}
    for enc, path in tmp.items():
        r = json.load(open(path, encoding="utf-8"))
        for t in TASKS:
            v = {x: np.array(r[t]["views"][x]["macro_f1"]) for x in ("image", "text", "both")}
            d = v["both"] - v["text"]
            res.setdefault(t, {})[enc] = {**{x: float(a.mean()) for x, a in v.items()},
                                          "value_of_image": float(d.mean()), "value_of_image_sd": float(d.std()),
                                          "value_of_image_first3": float(d[:3].mean()), "seeds": SEEDS}
        os.remove(path)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for t, e in res.items():
        print(f"[d5] {t}: value of image " + ", ".join(
            f"{k} {x['value_of_image']:+.3f} (sd {x['value_of_image_sd']:.3f}; first 3 seeds "
            f"{x['value_of_image_first3']:+.3f})" for k, x in e.items()), flush=True)
    print(f"_saved {OUT}_")


if __name__ == "__main__":
    main()
