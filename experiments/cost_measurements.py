"""measured cost of each data-tier check on our corpus (3,263 memes, CPU), and the
system-tier costs from the runs already made.

  CUDA_VISIBLE_DEVICES= python experiments/cost_measurements.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nesymis import config
from inputaudit.data_tier import (d1_provenance, d2_field_relations, d3_surface, d4_source,
                                  d5_modality, d6_duplicates)

A = config.ARTIFACTS_DIR
OUT = A / "cost_measurements.json"


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    df = pd.read_csv(config.CORRECTED_MANIFEST_PATH).fillna("")
    df["source_group"] = np.where(df["source"].str.startswith("wbms"), "wbms", "goat")
    classes = sorted(df.label.unique())
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    sp, k = df.split.to_numpy(), len(classes)
    v, u = np.load(config.IMAGE_EMB_PATH), np.load(config.TEXT_OCR_CORRECTED_EMB_PATH)
    prov = {"text_ocr": {"wbms": "EasyOCR", "goat": "EasyOCR"}}
    t = {
        "D1 provenance": timed(lambda: d1_provenance(df, ["text_ocr"], "source_group", prov)),
        "D2 field relations": timed(lambda: d2_field_relations(df, ["text_ocr", "text_caption"], "label")),
        "D3 surface screen (shape model)": timed(lambda: d3_surface(df.text_ocr.tolist(), y, sp, k)),
        "D4 source leak": timed(lambda: d4_source(df.text_ocr.tolist(), df.source_group, y, sp)),
        "D6 near-duplicates (text + embedding)": timed(lambda: d6_duplicates(df.text_ocr.tolist(), sp, u)),
        "D5 modality check, one encoder, 3 seeds, logistic probe":
            timed(lambda: d5_modality({"clip": {"image": v, "text": u}}, y, sp, k, 3, backend="logreg")),
        "D5 modality check, one encoder, 3 seeds, MLP probe (CPU)":
            timed(lambda: d5_modality({"clip": {"image": v, "text": u}}, y, sp, k, 3, backend="mlp")),
    }
    cb = json.load(open(A / "cost_benchmark.json", encoding="utf-8"))
    ts = cb["train_seconds"]
    system = {"S1 input inventory": "reading the system's code and configuration",
              "S2 same-input test": f"one training per input configuration "
                                    f"({sum(ts.values()):.0f} s per seed for the original design)",
              "S3 headroom by type": "one probe training per alternative input per seed",
              "S4 influence": "one prediction pass without the component",
              "S5 faithfulness": "2k x 20 prediction passes of a linear scorecard (under a second here)",
              "S6 fresh partitions": "the full pipeline once per partition"}
    json.dump({"data_tier_seconds_cpu": t, "system_tier": system}, open(OUT, "w"), indent=1)
    for n, s in t.items():
        print(f"| {n} | {s:.1f} s |")
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
