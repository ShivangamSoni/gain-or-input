"""Command line: the data tier on any manifest.

  python -m inputaudit data --manifest memes.csv --label label --text caption,ocr \
      [--split split] [--source source] [--provenance provenance.json] \
      [--emb clip:image=img.npy --emb clip:text:ocr=ocr.npy \
       --emb siglip:image=img_s.npy --emb siglip:text:ocr=ocr_s.npy] \
      [--seeds 3] [--probe auto|mlp|logreg] [--out audit]

Embeddings are row-aligned .npy arrays; list encoders weakest first. provenance.json is
{field: {source: "how the field was produced"}}. Writes <out>.md and <out>.json.
"""
import argparse
import json

import numpy as np
import pandas as pd

from inputaudit.data_tier import run_data_tier
from inputaudit.report import data_report


def _embeddings(specs):
    out = {}
    for s in specs or []:
        head, path = s.split("=", 1)
        enc, view = head.split(":", 1)
        out.setdefault(enc, {})[view] = np.load(path)
    return out or None


def main():
    ap = argparse.ArgumentParser(prog="python -m inputaudit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="data-tier checks D1-D6 on a manifest")
    d.add_argument("--manifest", required=True)
    d.add_argument("--label", required=True)
    d.add_argument("--text", required=True, help="comma-separated text columns; the first is primary")
    d.add_argument("--split", help="column with train/val/test (default: stratified 70/15/15)")
    d.add_argument("--source", help="column naming each row's data source")
    d.add_argument("--provenance", help="JSON {field: {source: production method}}")
    d.add_argument("--emb", action="append", help="ENCODER:VIEW=path.npy, VIEW = image | text:<field>")
    d.add_argument("--seeds", type=int, default=3)
    d.add_argument("--probe", default="auto", choices=["auto", "mlp", "logreg"])
    d.add_argument("--out", default="audit")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest).fillna("")
    prov = json.load(open(a.provenance, encoding="utf-8")) if a.provenance else None
    r = run_data_tier(df, a.label, a.text.split(","), a.split, a.source, prov,
                      _embeddings(a.emb), a.seeds, a.probe)
    json.dump(r, open(f"{a.out}.json", "w", encoding="utf-8"), indent=1, default=str)
    md = data_report(r)
    open(f"{a.out}.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
