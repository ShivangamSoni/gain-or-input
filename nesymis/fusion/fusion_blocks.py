"""
P4 -- three neural fusion blocks behind one shared interface.

Each block maps (vu: B×1024 fused CLIP [v;u], e: B×affect_dim projected affect) -> z,
exposing `.d_out`. Only this block changes across the fusion ablation; the affect
projection and the classifier head stay fixed, so the comparison is controlled.

  concat    : z = [vu; e]                       (affect adds dims)
  gate(FiLM): z = (1+γ(e)) ⊙ vu + β(e)          (affect modulates content; zero-init => starts identity)
  crossattn : self-attention over {v, u, e} tokens -> mean-pool  (models pairwise interactions / irony)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PlainFusion(nn.Module):
    """No affect, no gating: z = [v; u] passed straight to the classifier, so
    the neural branch is a plain MLP over the frozen CLIP embedding. The affect
    argument is ignored (kept in the signature for a shared interface)."""

    def __init__(self, vu_dim: int, affect_dim: int, **_):
        super().__init__()
        self.d_out = vu_dim

    def forward(self, vu, e):
        return vu


class ConcatFusion(nn.Module):
    def __init__(self, vu_dim: int, affect_dim: int, **_):
        super().__init__()
        self.d_out = vu_dim + affect_dim

    def forward(self, vu, e):
        return torch.cat([vu, e], dim=1)


class GateFusion(nn.Module):
    """FiLM: affect produces a per-feature scale+shift on [v;u]. Zero-init the
    film layer so it starts as the identity (z=vu) and learns to use affect."""

    def __init__(self, vu_dim: int, affect_dim: int, **_):
        super().__init__()
        self.film = nn.Linear(affect_dim, 2 * vu_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.d_out = vu_dim

    def forward(self, vu, e):
        gamma, beta = self.film(e).chunk(2, dim=1)
        return (1.0 + gamma) * vu + beta


class CrossAttnFusion(nn.Module):
    """Self-attention over three tokens {v, u, e} (each projected to d_model)."""

    def __init__(self, vu_dim: int, affect_dim: int, d_model: int = 256, nhead: int = 4, dropout: float = 0.1, **_):
        super().__init__()
        self.half = vu_dim // 2
        self.pv = nn.Linear(self.half, d_model)
        self.pu = nn.Linear(self.half, d_model)
        self.pe = nn.Linear(affect_dim, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True)
        self.d_out = d_model

    def forward(self, vu, e):
        v, u = vu[:, : self.half], vu[:, self.half:]
        toks = torch.stack([self.pv(v), self.pu(u), self.pe(e)], dim=1)   # (B,3,d_model)
        return self.layer(toks).mean(dim=1)


FUSIONS = {"plain": PlainFusion, "concat": ConcatFusion, "gate": GateFusion,
           "crossattn": CrossAttnFusion}
