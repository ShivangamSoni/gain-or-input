"""
Extract OCR text from stereotype meme images with EasyOCR (GPU), and write it
back into the manifest's `text` column.

Images are loaded via PIL -> numpy (not by passing the path) so that non-ASCII
filenames -- the captions contain unicode like the curly apostrophe -- don't trip
up the reader on Windows. Results are cached in artifacts/ocr_cache.json keyed by
the project-relative path, so re-runs are incremental and resumable.

Run:  python -m nesymis.data.ocr_text            # OCR all stereotype rows
      python -m nesymis.data.ocr_text --limit 5  # quick smoke test
      python -m nesymis.data.ocr_text --force     # ignore cache, re-OCR
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from nesymis import config
from nesymis.data.imageio import open_rgb

_WS = re.compile(r"\s+")


def _clean(t: str) -> str:
    return _WS.sub(" ", t.replace("\n", " ")).strip()


def _prep(img: Image.Image, min_side: int = 720) -> np.ndarray:
    """Upscale small memes so EasyOCR can read low-res burned-in captions."""
    w, h = img.size
    m = min(w, h)
    if m < min_side:
        s = min_side / m
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.array(img)


def _read(reader, arr) -> str:
    # paragraph=True merges word boxes into fuller lines; lowered thresholds raise
    # recall (we want more caption content, noise is acceptable / even desirable).
    chunks = reader.readtext(arr, detail=0, paragraph=True,
                             text_threshold=0.6, low_text=0.3, mag_ratio=1.5)
    return _clean(" ".join(chunks))


def _load_cache() -> dict:
    if config.OCR_CACHE_PATH.exists():
        return json.loads(config.OCR_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    config.OCR_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8"
    )


def run(limit: int | None = None, force: bool = False) -> None:
    import easyocr  # heavy import; keep local

    df = pd.read_csv(config.MANIFEST_PATH, dtype=str).fillna("")
    todo = df[df["source"].str.startswith("wbms")].copy()
    if limit:
        todo = todo.head(limit)
    print(f"[ocr] {len(todo)} stereotype rows to OCR (limit={limit}, force={force})")

    cache = {} if force else _load_cache()
    reader = easyocr.Reader(["en"], gpu=True)

    new = 0
    for i, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc="OCR")):
        key = row["path"]
        if key in cache and not force:
            continue
        try:
            cache[key] = _read(reader, _prep(open_rgb(key)))
        except Exception as e:  # corrupt/unreadable -> empty text
            cache[key] = ""
            tqdm.write(f"  [warn] {key}: {e}")
        new += 1
        if new % 200 == 0:
            _save_cache(cache)
    _save_cache(cache)
    print(f"[ocr] OCR'd {new} new images; cache size = {len(cache)}")

    # write OCR text back into the manifest for stereotype rows
    mask = df["source"].str.startswith("wbms")
    df.loc[mask, "text_ocr"] = df.loc[mask, "path"].map(lambda p: cache.get(p, ""))
    df.to_csv(config.MANIFEST_PATH, index=False)

    nonempty = int((df.loc[mask, "text_ocr"].str.len() > 0).sum())
    avg_len = float(df.loc[mask, "text_ocr"].str.len().mean())
    print(
        f"[ocr] wrote text to manifest: {nonempty}/{int(mask.sum())} stereotype rows "
        f"have non-empty OCR text (avg {avg_len:.1f} chars)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, force=args.force)
