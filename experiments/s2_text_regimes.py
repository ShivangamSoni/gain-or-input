"""Validity control: is the task solved by the clean caption rather than by the model?

The original design routes two different text channels to the two paths -- noisy
OCR to the neural branch, the post caption ("caption": the title/caption scraped with
each WBMS image from its source page, stored as the filename) to the symbolic
branch -- and calls the asymmetry an information split. This script tests whether
that asymmetry, rather than the architecture, produces the reported margin.

What prompted it: a 2-layer MLP over frozen CLIP [image; caption] reaches 0.973
test macro-F1, against the original design's 0.847. Restricted to the four
stereotype classes, so no GOAT/WBMS source confound can contribute, the caption
channel lifts a plain MLP from 0.591 to 0.975 macro-F1. The post caption very
nearly determines the topic label -- and it is mostly NOT the text in the image.

Three regimes, everything else held at the deployed configuration:

  caption   symbolic path grounds on the caption (DEPLOYED)
  ocr       symbolic path grounds on OCR -- the same channel the neural path gets,
            so no path holds privileged text. OCR is the only text field produced
            the same way for every row of the combined corpus.
  parity    neural path ALSO gets the caption, symbolic path keeps it -- the
            "give the baseline everything" comparison.

No source change is needed for the ocr regime: concept_layer.build() reads
`emb.text_caption` and `emb.df["text_caption"]`, so substituting the OCR strings
and OCR embeddings there re-mines, re-cuts and re-induces the whole symbolic path
on OCR exactly as the deployed pipeline would.

  python experiments/s2_text_regimes.py --seeds 5
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import common as C
import neural_scaling as t22
from nesymis import config

OUT = str(config.ARTIFACTS_DIR / "s2_text_regimes.json")


def _slices(c):
    U, te, trainval = c.emb.text_caption, c.te, c.trainval
    novel = (U[te] @ U[trainval].T).max(1) < t22.NOVEL_COS
    return {"full": np.ones(int(te.sum()), bool), "novel": novel}


_FUSED = {}


def fused_text(df):
    """(strings, embeddings) for the original design's `fused` mode: caption + OCR
    as one string. Never deployed -- config.TEXT_SOURCE is "ocr" -- so there is no
    cached embedding for it and it is encoded here (3263 strings, a few seconds)."""
    if not _FUSED:
        from nesymis.data.dataset import compose_text
        from nesymis.encoders import clip_encoder as ce
        txt = compose_text(df, "fused")
        _FUSED["txt"] = txt
        _FUSED["emb"] = ce.encode_texts(txt)
        print(f"  [regime] encoded {len(txt)} fused caption+OCR strings", flush=True)
    return _FUSED["txt"], _FUSED["emb"]


def run(regime, c, slices, seeds):
    """One regime: re-mine concepts, re-induce rules, retrain head + decision layer."""
    df, y, trainval = c.df, c.y, c.trainval
    v = c.emb.image
    if regime in ("corrected", "corrected-split"):
        # The corrected benchmark: our EasyOCR pipeline run over ALL rows, including the
        # 1133 GOAT images the original build skipped. Row order and splits are identical
        # to manifest.csv (asserted in load_corrected), so c.y/tr/va/te stay valid.
        import uniform_ocr_protocol as cb
        dfc, _y, _v, u_ocr_c, u_cap_c, *_ = cb.load_corrected()
        neu_u = u_ocr_c
        if regime == "corrected":                 # official: neither path holds better text
            sym_u = u_ocr_c
            sym_txt = dfc["text_ocr"].fillna("").astype(str)
        else:                                     # deployed-style split, corrected corpus
            sym_u = u_cap_c
            sym_txt = dfc["text_caption"].fillna("").astype(str)
        sym_txt.index = df.index
    elif regime == "fused":
        ftxt, femb = fused_text(df)
        sym_u, sym_txt = femb, pd.Series(ftxt, index=df.index)
        neu_u = femb
    else:
        sym_u = c.emb.text_caption if regime in ("caption", "parity") else c.emb.text_ocr
        sym_txt = df["text_caption"] if regime in ("caption", "parity") else df["text_ocr"]
        # Neural features: [v; u_ocr] deployed, [v; u_caption] under parity.
        neu_u = c.emb.text_caption if regime == "parity" else c.emb.text_ocr
    vu = np.concatenate([v, neu_u], 1).astype(np.float32)
    D = np.zeros((len(df), config.AFFECT_DIM), np.float32)          # affect-free

    df2 = df.copy()
    df2["text_caption"] = sym_txt.fillna("").astype(str)
    fired, rvec, pb = t22._induce(df2, y, trainval, sym_u, v)

    old = C.SEEDS11
    try:
        C.SEEDS11 = old[:seeds]
        res = t22.run_cell(regime, vu, D, fired, rvec, config.MLP["hidden"], c, slices)
    finally:
        C.SEEDS11 = old
    pbm = {k: C.compute_metrics(y[c.te][m], pb[c.te][m]) for k, m in slices.items()}
    return res, pbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--regimes", default="caption,ocr,parity")
    a = ap.parse_args()

    c = C.build_context()
    slices = _slices(c)
    store = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    for regime in a.regimes.split(","):
        print(f"\n########## text regime: {regime} ##########", flush=True)
        res, pbm = run(regime, c, slices, a.seeds)
        keys = ("acc", "macro_f1", "per_class")
        store[regime] = {
            "seeds": a.seeds,
            "path_a": {k: {m: res["Path-A"][k][m] for m in keys} for k in slices},
            "nesymis": {k: {m: res["NeSy"][k][m] for m in keys} for k in slices},
            "path_b": {k: {"acc": pbm[k]["acc"], "macro_f1": pbm[k]["macro_f1"],
                           "per_class": pbm[k]["per_class_f1"]} for k in slices},
        }
        json.dump(store, open(OUT, "w"), indent=1)

    print("\n## Text-regime validity control "
          f"({a.seeds}-seed, everything else at the deployed configuration)\n")
    print("| symbolic text | neural text | Path-B rule-only F1 | Path-A full Acc / F1 | "
          "NeSy full Acc / F1 | NeSy novel Acc / F1 | margin F1 |")
    print("|---|---|---|---|---|---|---|")
    NT = {"caption": "OCR", "ocr": "OCR", "parity": "**caption**", "fused": "caption+OCR",
          "corrected": "OCR*", "corrected-split": "OCR*"}
    ST = {"caption": "caption *(deployed)*", "ocr": "**OCR**", "parity": "caption",
          "fused": "caption+OCR", "corrected": "**OCR***", "corrected-split": "caption"}
    for regime in a.regimes.split(","):
        r = store.get(regime)
        if not r:
            continue
        pa, ne = r["path_a"], r["nesymis"]
        margin = np.mean(ne["full"]["macro_f1"]) - np.mean(pa["full"]["macro_f1"])
        print(f"| {ST[regime]} | {NT[regime]} | {r['path_b']['full']['macro_f1']:.3f} | "
              f"{C.ms(pa['full']['acc'])} / {C.ms(pa['full']['macro_f1'])} | "
              f"{C.ms(ne['full']['acc'])} / {C.ms(ne['full']['macro_f1'])} | "
              f"{C.ms(ne['novel']['acc'])} / {C.ms(ne['novel']['macro_f1'])} | "
              f"{margin:+.3f} |")

    # Per class, because a flat accuracy alongside a positive macro-F1 delta can
    # only mean the rare classes moved -- and whether they did is the difference
    # between a modest positive result and a purely diagnostic paper.
    for regime in a.regimes.split(","):
        r = store.get(regime)
        if not r or "per_class" not in r["path_a"]["full"]:
            continue
        print(f"\n### Per-class F1, {regime} regime (full test split, "
              f"{r['seeds']} seeds)\n")
        cls = list(config.LABELS)
        # ASCII only: redirected stdout on Windows is cp1252 and a bare "Delta"
        # glyph raises UnicodeEncodeError, which would lose the whole table.
        print("| class | Path-B rule-only | Path-A | NeSy | delta NeSy-PathA |")
        print("|---|---|---|---|---|")
        for c in cls:
            pb = r["path_b"]["full"]["per_class"][c]
            pa_v = np.mean([d[c] for d in r["path_a"]["full"]["per_class"]])
            ne_v = np.mean([d[c] for d in r["nesymis"]["full"]["per_class"]])
            print(f"| {config.LABEL_DISPLAY[c]} | {pb:.3f} | {pa_v:.3f} | {ne_v:.3f} | "
                  f"{ne_v - pa_v:+.3f} |")


if __name__ == "__main__":
    main()
