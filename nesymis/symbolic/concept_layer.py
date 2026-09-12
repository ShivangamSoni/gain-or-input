"""Multimodal concept-abstraction layer (deployed symbolic grounding).

Replaces raw says:/sem:/ex: predicate phrases with a bank of grounded CONCEPTS:
mined text-phrase anchors AND CLIP image-scene anchors are clustered (after
per-modality gap correction) into named concept prototypes. A meme fires a
concept via  max(cos(text, centroid), cos(image, centroid))  in the centered
space, so concepts fire on the caption OR the picture and generalise beyond
surface words. Concepts are named from member phrases (text clusters) or from
the captions of the memes they fire on (pure-image clusters).

This is the deployed grounding as of the concept-layer work; it is shared by the
production pipeline and by every external-transfer comparison so the exact same
concept machinery is used everywhere.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from nesymis import config
from nesymis.symbolic import predicate_pool

# ---- defaults (the mm2b configuration that gave 0.910/0.847, novel 0.793) ----
K_CONCEPTS = 30         # REVERTED from 50. K=50 wins on the full test split but costs
                        # -0.018 macro-F1 on the novel slice under the 11-seed protocol
                        # (experiments/phase2_e25_fullprotocol.py). The E2.3 sweep that chose 50
                        # scored only the full split on 3 seeds and could not see this.
N_IMG = 80              # CLIP image-scene anchor prototypes
TOPM = 40              # memes used to name a pure-image concept
MIN_MEMBERS = 2
LEX_RICH, EX_RICH = 40, 120
_STOP = set("a an the of to in on for and or but with without is are be as at by so you "
            "your my me we they them their he she her his it its this that these those do "
            "does did done not no yes if then when where what who how i am re ve ll nothing "
            "get got go going im dont cant just really means expect when she he aint one "
            "will now here there like want".split())


_VOCAB = None
COMMON_RANK = 25000          # CLIP merges are frequency-ordered: ids below this are common words
CORPUS_MIN_DF = 10           # a word in this many training texts is kept even if CLIP lacks it
_SHORT = {"a", "i", "in", "on", "at", "to", "of", "it", "is", "my", "me", "we", "he", "be", "do",
          "go", "no", "so", "up", "us", "an", "am", "or", "if", "by"}
_KNOWN: set = set()          # corpus words for the current build (see build())


def _vocab() -> set:
    """Common whole words of CLIP's BPE vocabulary (entries ending in </w>, early merges only),
    read from the local model cache: an offline dictionary used only to NAME concepts."""
    global _VOCAB
    if _VOCAB is None:
        try:
            from transformers import CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained(config.CLIP_MODEL)
            _VOCAB = {w[:-4] for w, i in tok.get_vocab().items()
                      if w.endswith("</w>") and w[:-4].isalpha() and i < COMMON_RANK}
        except Exception:
            _VOCAB = set()
    return _VOCAB


_SPLIT_VOCAB = None
SPLIT_RANK = 8000            # pieces of a split merge must be very common words


def _split_vocab() -> set:
    global _SPLIT_VOCAB
    if _SPLIT_VOCAB is None:
        try:
            from transformers import CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained(config.CLIP_MODEL)
            _SPLIT_VOCAB = {w[:-4] for w, i in tok.get_vocab().items()
                            if w.endswith("</w>") and w[:-4].isalpha() and i < SPLIT_RANK} | _SHORT
        except Exception:
            _SPLIT_VOCAB = set()
    return _SPLIT_VOCAB


def _split(w: str, vocab: set) -> list | None:
    """Split an OCR merge ('thekitchen', 'iwant') into the fewest very common words, or None.
    Pieces of one or two letters are allowed only when they are real short words."""
    best = [None] * (len(w) + 1)
    best[0] = []
    for i in range(1, len(w) + 1):
        for j in range(max(0, i - 15), i):
            piece = w[j:i]
            if best[j] is not None and piece in vocab and (len(piece) > 2 or piece in _SHORT):
                if best[i] is None or len(best[j]) + 1 < len(best[i]):
                    best[i] = best[j] + [piece]
    return best[-1]


def _toks(t: str) -> list[str]:
    """Name words from a phrase or text (E9 name hygiene). OCR merges are split into very common
    words ('thekitchen' -> 'the kitchen'); otherwise a token is kept only if it is a common word or
    frequent in the training texts. Names never enter a decision, so this changes how evidence
    reads, not what it does."""
    vocab, sv = _vocab(), _split_vocab()
    out = []
    for w in re.findall(r"[a-z]+", t.lower()):
        if not vocab or w in vocab:                         # a common word: keep as is
            ws = [w]
        else:
            parts = _split(w, sv) if len(w) > 4 else None   # an OCR merge of common words
            if parts and len(parts) > 1:
                ws = parts
            elif w in _KNOWN:                               # frequent in this corpus ('covid')
                ws = [w]
            else:                                           # OCR noise
                ws = []
        out += [x for x in ws if x not in _STOP and len(x) > 2]
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


@dataclass
class ConceptBank:
    names: list                 # concept names (data-derived)
    centroids: np.ndarray       # (K, d) unit vectors in the gap-centered space
    mu_t: np.ndarray            # (d,) text modality center
    mu_i: np.ndarray            # (d,) image modality center
    cwords: list                # per-concept name-word set (for disjointness analysis)
    cmix: list                  # per-concept (n_text_members, n_image_members)

    def score(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(N,K) concept scores for meme text emb u and image emb v (both (N,d))."""
        Uc = _unit(_unit(u) - self.mu_t)
        Vc = _unit(_unit(v) - self.mu_i)
        return np.maximum(Uc @ self.centroids.T, Vc @ self.centroids.T).astype(np.float32)

    def frame(self, u, v, index=None) -> pd.DataFrame:
        S = self.score(u, v)
        return pd.DataFrame({f'concept:"{nm}"': S[:, j] for j, nm in enumerate(self.names)},
                            index=index)

    def to_files(self, npz_path, json_path):
        np.savez(npz_path, centroids=self.centroids, mu_t=self.mu_t, mu_i=self.mu_i,
                 names=np.array(self.names, dtype=object))
        json.dump({"names": self.names, "cmix": self.cmix,
                   "cwords": [sorted(w) for w in self.cwords]},
                  open(json_path, "w", encoding="utf-8"), indent=1)

    @staticmethod
    def from_files(npz_path, json_path) -> "ConceptBank":
        z = np.load(npz_path, allow_pickle=True)
        meta = json.load(open(json_path, encoding="utf-8"))
        return ConceptBank(names=[str(n) for n in z["names"]], centroids=z["centroids"],
                           mu_t=z["mu_t"], mu_i=z["mu_i"],
                           cwords=[set(w) for w in meta["cwords"]], cmix=meta["cmix"])


def build(emb, trainval, classes=None, k=K_CONCEPTS, n_img=N_IMG,
          lex=LEX_RICH, ex=EX_RICH, seed=config.SEED) -> ConceptBank:
    """Build a concept bank from an embeddings-like object (needs .image,
    .text_caption, .df[text_caption,label]) using train+val rows only.

    `classes` selects which class labels drive the per-class phrase mining
    (defaults to config.STEREO_CLASSES); image anchors and clustering are
    unsupervised.
    """
    predicate_pool.LEX_PER_CLASS, predicate_pool.N_EXEMPLARS = lex, ex
    _, _, _, spec, _ = predicate_pool.build_pool(emb, classes=classes, calib_mask=trainval,
                                                 save=False, return_spec=True)
    tnames = list(spec.keys())
    Tvec = _unit(np.stack([spec[n] for n in tnames]).astype(np.float32))
    tphr = [re.sub(r'^(sem|ex):"(.*)"$', r"\2", n) for n in tnames]

    V = _unit(emb.image.astype(np.float32))
    ik = KMeans(n_clusters=min(n_img, int(trainval.sum())), random_state=seed, n_init=6).fit(V[trainval])
    Ivec = _unit(ik.cluster_centers_.astype(np.float32))

    mu_t, mu_i = Tvec.mean(0), Ivec.mean(0)                 # per-modality centers (gap correction)
    Tc = _unit(Tvec - mu_t); Ic = _unit(Ivec - mu_i); Vc = _unit(V - mu_i)
    caps = emb.df["text_caption"].fillna("").astype(str).tolist()
    tv = np.where(trainval)[0]
    global _KNOWN                                           # frequent training-text words, for naming
    dfc = Counter(w for i in tv for w in set(re.findall(r"[a-z]+", caps[i].lower())))
    _KNOWN = {w for w, n in dfc.items() if n >= CORPUS_MIN_DF and len(w) > 2}

    A = np.concatenate([Tc, Ic], 0)
    is_img = np.array([False] * len(Tc) + [True] * len(Ic))
    phr = tphr + [""] * len(Ic)
    km = KMeans(n_clusters=min(k, len(A)), random_state=seed, n_init=10).fit(A)

    names, cens, cwords, cmix, used = [], [], [], [], set()
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) < MIN_MEMBERS:
            continue
        cen = _unit(km.cluster_centers_[c])
        words = Counter()
        for i in idx:
            words.update(set(_toks(phr[i])))
        if not words:                                       # pure-image: name from firing memes' captions
            sc = Vc[tv] @ cen
            for i in tv[np.argsort(-sc)[:TOPM]]:
                words.update(set(_toks(caps[i])))
        # Counter.most_common breaks ties by insertion order, and insertion order
        # here comes from iterating `set(...)` of strings -- Python randomises
        # string hashes per process, so equal-count words were ordered differently
        # on every run and the same cluster got a different display name. Names are
        # quoted in the paper, so tie-break explicitly: count desc, then word asc.
        nw = [w for w, _ in sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        base = "-".join(nw) or ("unreadable-text" if _vocab() else "scene")   # E9: nothing readable
        nm, t = base, 2
        while nm in used:
            nm = f"{base}-{t}"; t += 1
        used.add(nm)
        names.append(nm); cens.append(cen.astype(np.float32)); cwords.append(set(nw))
        cmix.append((int((~is_img[idx]).sum()), int(is_img[idx].sum())))
    return ConceptBank(names, np.stack(cens), mu_t.astype(np.float32),
                       mu_i.astype(np.float32), cwords, cmix)
