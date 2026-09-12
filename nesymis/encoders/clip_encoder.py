"""
Frozen CLIP-ViT-B/32 encoder + embedding cache.

Encodes every meme image (v) and its text (u) into 512-d, L2-normalized CLIP
embeddings, aligned to the manifest row order, and caches them as .npy. Because
the encoder is frozen, this runs once and all downstream training / inference /
ablations read from the cache (seconds instead of minutes).

transformers>=5 note: get_image_features(...)/get_text_features(...) return a
ModelOutput whose `.pooler_output` is the (un-normalized) projected 512-d
embedding -- parallel to the canonical image_embeds/text_embeds. We normalize.

Run:  python -m nesymis.encoders.clip_encoder
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from nesymis import config
from nesymis.data.imageio import open_rgb

_model = None
_proc = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def get_clip():
    """Lazily load the frozen CLIP model + processor (cached singletons)."""
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(_device).eval()
        for p in _model.parameters():
            p.requires_grad_(False)
        _proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    return _model, _proc


@torch.no_grad()
def encode_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """L2-normalized CLIP text embeddings, (N, 512)."""
    model, proc = get_clip()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = [t if isinstance(t, str) and t else "" for t in texts[i : i + batch_size]]
        inp = proc(text=chunk, return_tensors="pt", padding=True, truncation=True).to(_device)
        emb = model.get_text_features(**inp).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def encode_images(paths: list[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalized CLIP image embeddings, (N, 512). `paths` are project-relative."""
    model, proc = get_clip()
    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc="image-emb"):
        imgs = []
        for p in paths[i : i + batch_size]:
            try:
                imgs.append(open_rgb(p))
            except Exception as e:
                print(f"  [warn] cannot open {p}: {e}; using gray placeholder")
                imgs.append(Image.new("RGB", (224, 224), (127, 127, 127)))
        inp = proc(images=imgs, return_tensors="pt").to(_device)
        emb = model.get_image_features(**inp).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def build_cache() -> None:
    from nesymis.data.dataset import compose_text, load_manifest

    df = load_manifest()
    print(f"[clip] encoding {len(df)} samples on {_device} ...")

    img_emb = encode_images(df["path"].tolist())
    ocr_emb = encode_texts(compose_text(df, "ocr"))
    cap_emb = encode_texts(compose_text(df, "caption"))

    np.save(config.IMAGE_EMB_PATH, img_emb)
    np.save(config.TEXT_OCR_EMB_PATH, ocr_emb)
    np.save(config.TEXT_CAPTION_EMB_PATH, cap_emb)
    config.EMB_IDS_PATH.write_text(json.dumps(df["id"].tolist()), encoding="utf-8")
    print(
        f"[clip] saved image_emb {img_emb.shape}, text_emb_ocr {ocr_emb.shape}, "
        f"text_emb_caption {cap_emb.shape}"
    )


if __name__ == "__main__":
    build_cache()
