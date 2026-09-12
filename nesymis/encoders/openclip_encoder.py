"""open_clip encoders for the larger CLIP and SigLIP backbones used in the
encoder-scaling study. Same interface as ``nesymis.encoders.clip_encoder``
(``encode_images`` / ``encode_texts`` -> L2-normalized float32 arrays), but the
active backbone is selected with ``set_model(key)``.

Registry keys:
  clip-g14        ViT-g-14            / laion2b_s34b_b88k   (1024-d)
  clip-bigg14     ViT-bigG-14         / laion2b_s39b_b160k  (1280-d)
  siglip-l16-384  ViT-L-16-SigLIP-384 / webli               (1024-d)
  siglip-so400m   ViT-SO400M-14-SigLIP-384 / webli          (1152-d)

Weights download to the open_clip / HF cache on first use. On CUDA the model is
loaded in fp16 (so ViT-bigG-14 fits an 8 GB card); embeddings are returned as
fp32 either way, so everything downstream (concept bank, classifier, policy) is
unchanged and simply sees a differently sized [v; u] vector.

Download only (no GPU needed):  python -m nesymis.encoders.openclip_encoder download [key|all]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import open_clip

from nesymis.data.imageio import open_rgb

_device = "cuda" if torch.cuda.is_available() else "cpu"

# key -> (open_clip name, pretrained tag, embed_dim, img_batch, txt_batch)
ENCODERS = {
    "clip-g14":       ("ViT-g-14",                "laion2b_s34b_b88k",  1024, 32, 256),
    "clip-bigg14":    ("ViT-bigG-14",             "laion2b_s39b_b160k", 1280,  8, 128),
    "siglip-l16-384": ("ViT-L-16-SigLIP-384",     "webli",              1024, 16, 128),
    "siglip-so400m":  ("ViT-SO400M-14-SigLIP-384", "webli",             1152, 16, 128),
}

_state: dict = {"key": None, "model": None, "preprocess": None, "tokenizer": None, "prec": None}


def embed_dim(key: str) -> int:
    return ENCODERS[key][2]


def set_model(key: str, precision: str | None = None, device: str | None = None):
    """Load (and cache) the backbone for ``key``; downloads weights on first use."""
    if key not in ENCODERS:
        raise KeyError(f"unknown encoder '{key}'; options: {list(ENCODERS)}")
    dev = device or _device
    prec = precision or ("fp16" if dev == "cuda" else "fp32")
    if _state["key"] == key and _state["model"] is not None and _state["prec"] == prec:
        return
    name, tag = ENCODERS[key][0], ENCODERS[key][1]
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=tag, precision=prec, device=dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _state.update(key=key, model=model, preprocess=preprocess,
                  tokenizer=open_clip.get_tokenizer(name), prec=prec, device=dev)


def _need() -> dict:
    if _state["model"] is None:
        raise RuntimeError("call set_model(key) before encoding")
    return _state


@torch.no_grad()
def encode_images(paths: list[str], batch_size: int | None = None) -> np.ndarray:
    """L2-normalized image embeddings, (N, embed_dim), for the active backbone."""
    s = _need()
    bs = batch_size or ENCODERS[s["key"]][3]
    pre, model = s["preprocess"], s["model"]
    half = s["prec"] == "fp16"
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([pre(open_rgb(p)) for p in paths[i:i + bs]]).to(s["device"])
        if half:
            batch = batch.half()
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


@torch.no_grad()
def encode_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """L2-normalized text embeddings, (N, embed_dim), for the active backbone."""
    s = _need()
    bs = batch_size or ENCODERS[s["key"]][4]
    tok, model = s["tokenizer"], s["model"]
    out = []
    for i in range(0, len(texts), bs):
        chunk = [t if isinstance(t, str) and t else "" for t in texts[i:i + bs]]
        toks = tok(chunk).to(s["device"])
        emb = model.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def download(key: str):
    """Fetch weights into the cache (CPU, fp32) without touching the GPU."""
    name, tag = ENCODERS[key][0], ENCODERS[key][1]
    print(f"  [openclip] downloading {key}: {name} / {tag} ...", flush=True)
    open_clip.create_model_and_transforms(name, pretrained=tag, precision="fp32", device="cpu")
    print(f"  [openclip] {key} ready.", flush=True)


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "download" else "all"
    keys = list(ENCODERS) if which == "all" else [which]
    for k in keys:
        download(k)
    print("DOWNLOADS_DONE", flush=True)
