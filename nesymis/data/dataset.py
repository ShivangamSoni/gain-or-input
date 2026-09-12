"""
Access layer over the manifest + cached CLIP embeddings.

Loads the manifest and the (image, text) embedding caches, asserts they are
row-aligned, and exposes convenience getters for split/modality slices used by
the neural classifier, the symbolic grounder, and the evaluation scripts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nesymis import config


@dataclass
class Embeddings:
    df: pd.DataFrame          # manifest, row-aligned with the arrays below
    image: np.ndarray         # (N, 512) L2-normalized
    text_ocr: np.ndarray      # (N, 512) OCR text -> NEURAL path
    text_caption: np.ndarray  # (N, 512) caption text -> SYMBOLIC path

    @property
    def text(self) -> np.ndarray:
        """The text embedding feeding the NEURAL path (config.NEURAL_TEXT_SOURCE)."""
        return self.text_caption if config.NEURAL_TEXT_SOURCE == "caption" else self.text_ocr

    @property
    def symbolic_text(self) -> np.ndarray:
        """The text embedding feeding the SYMBOLIC path (config.SYMBOLIC_TEXT_SOURCE)."""
        return self.text_caption if config.SYMBOLIC_TEXT_SOURCE == "caption" else self.text_ocr

    def features(self, modality: str) -> np.ndarray:
        if modality == "image":
            return self.image
        if modality == "text":
            return self.text
        if modality == "both":
            return np.concatenate([self.image, self.text], axis=1)  # h = [v; u]
        raise ValueError(f"unknown modality: {modality}")

    def split_mask(self, split: str) -> np.ndarray:
        return (self.df["split"] == split).to_numpy()

    def get_split(self, split: str, modality: str = "both"):
        """Return (X, y, sub_df) for a split. sub_df is row-aligned with X/y."""
        mask = self.split_mask(split)
        X = self.features(modality)[mask]
        y = self.df.loc[mask, "label_idx"].to_numpy()
        sub = self.df.loc[mask].reset_index(drop=True)
        return X, y, sub

    @property
    def labels(self) -> np.ndarray:
        return self.df["label_idx"].to_numpy()


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(config.MANIFEST_PATH)
    for col in ("text_ocr", "text_caption"):
        df[col] = df[col].fillna("").astype(str)
    return df


def compose_text(df: pd.DataFrame, mode: str | None = None) -> list[str]:
    """Build the per-sample CLIP text input from caption/OCR pieces.

    See config.TEXT_SOURCE for the modes. Non-stereotype rows (GOAT) have
    caption == OCR == jsonl text, so every mode except "empty" returns that text.
    """
    mode = mode or config.TEXT_SOURCE
    is_stereo = df["source"].str.startswith("wbms").to_numpy()
    cap = df["text_caption"].fillna("").astype(str).tolist()
    ocr = df["text_ocr"].fillna("").astype(str).tolist()
    out = []
    for st, c, o in zip(is_stereo, cap, ocr):
        if not st:
            out.append(o)                       # GOAT jsonl text
        elif mode == "caption":
            out.append(c)
        elif mode == "ocr":
            out.append(o)
        elif mode == "empty":
            out.append("")
        else:  # "fused"
            out.append((c + " " + o).strip() if o else c)
    return out


def load_embeddings() -> Embeddings:
    df = load_manifest()
    image = np.load(config.IMAGE_EMB_PATH)
    text_ocr = np.load(config.TEXT_OCR_EMB_PATH)
    text_caption = np.load(config.TEXT_CAPTION_EMB_PATH)
    ids = json.loads(config.EMB_IDS_PATH.read_text(encoding="utf-8"))
    assert df["id"].tolist() == ids, "manifest/embedding row order mismatch — re-run clip_encoder"
    assert len(df) == len(image) == len(text_ocr) == len(text_caption), "length mismatch"
    return Embeddings(df=df, image=image, text_ocr=text_ocr, text_caption=text_caption)
