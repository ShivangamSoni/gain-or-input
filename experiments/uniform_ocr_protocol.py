"""Corrected benchmark: one OCR pipeline for every row, and a protocol that says so.

## The defect

`build_manifest._load_non_stereo_pool()` sets `text_ocr = text_caption = <jsonl text>`
for the 1133 GOAT rows ("GOAT jsonl text serves both roles"), and
`ocr_text.run()` filters to `source.startswith("wbms")`, so our EasyOCR pipeline never
touched a GOAT image. Two consequences:

1. `text_caption == text_ocr` holds for 1133/1133 non-stereotype rows and 0/2130
   stereotype rows -- an exact, label-revealing indicator of the source group.
2. The field means different things per source: for stereotype (WBMS) rows it is the post
   title/caption scraped with the image (stored as the filename) -- post-level text, mostly
   not the text in the image -- while for GOAT rows it is GOAT-Bench's own meme text.

Because the original design routes the caption to the symbolic path and OCR to the neural
path, the reported +0.189 macro-F1 "neuro-symbolic margin" measured an input asymmetry. Equalise the channels and it is +0.012; see
experiments/s2_text_regimes.py.

## What this script does

Runs the *identical* EasyOCR pipeline (`ocr_text._prep` / `_read`, same thresholds, same
upscaling) over the 1133 GOAT images, so every row's OCR channel comes from one
procedure. Writes a NEW manifest and a NEW OCR embedding cache; the originals are left
alone so every published number stays reproducible and the two can be compared.

Row order, ids, labels and splits are preserved exactly, so `image_emb.npy` and
`text_emb_caption.npy` remain valid and only the OCR text embedding is recomputed.

## The corrected protocol

* **Official input = image + OCR**, from one pipeline for all rows. This is what is
  actually available at inference on an unseen meme, and it is symmetric across classes.
* **Caption text = a separate, caveated setting**, never a shared input on this combined
  corpus: it is a scraped post title for WBMS rows but meme text for GOAT rows, and its
  style alone identifies the source (AUC 0.978 from nine content-free shape features,
  experiments/d3_surface_screen.py), which decides stereotype vs not. (Result
  keys still say "oracle text" so saved artifacts stay comparable; the word is retired.)

  python experiments/uniform_ocr_protocol.py --ocr        # OCR the GOAT images (~10 min)
  python experiments/uniform_ocr_protocol.py --build      # corrected manifest + OCR embeddings
  python experiments/uniform_ocr_protocol.py --verify     # is the artifact gone?
  python experiments/uniform_ocr_protocol.py --baselines  # reference numbers, corrected protocol
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from nesymis import config

ART = config.ARTIFACTS_DIR
MANIFEST_C = ART / "manifest_corrected.csv"
OCR_EMB_C = ART / "text_emb_ocr_corrected.npy"
OUT = str(ART / "uniform_ocr_protocol.json")


def _save(d):
    cur = json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}
    cur.update(d)
    json.dump(cur, open(OUT, "w", encoding="utf-8"), indent=1)


# --------------------------------------------------------------------------- #
# 1. OCR the rows the original pipeline skipped
# --------------------------------------------------------------------------- #
def cmd_ocr(force=False, limit=None):
    """Run the SAME EasyOCR pipeline over the GOAT images. Additive: the cache is
    keyed by project-relative path, so existing WBMS entries are untouched."""
    import easyocr
    from tqdm import tqdm

    from nesymis.data.imageio import open_rgb
    from nesymis.data.ocr_text import _prep, _read, _load_cache, _save_cache

    df = pd.read_csv(config.MANIFEST_PATH, dtype=str).fillna("")
    todo = df[df["source"].str.startswith("goat")]
    if limit:
        todo = todo.head(limit)
    cache = _load_cache()
    have = sum(1 for p in todo["path"] if p in cache)
    print(f"[cb] {len(todo)} GOAT rows; {have} already cached; force={force}", flush=True)

    reader = easyocr.Reader(["en"], gpu=True)
    new = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="OCR-goat"):
        key = row["path"]
        if key in cache and not force:
            continue
        try:
            cache[key] = _read(reader, _prep(open_rgb(key)))
        except Exception as e:  # noqa: BLE001
            cache[key] = ""
            tqdm.write(f"  [warn] {key}: {e}")
        new += 1
        if new % 200 == 0:
            _save_cache(cache)
    _save_cache(cache)
    got = [cache.get(p, "") for p in todo["path"]]
    ne = sum(1 for t in got if str(t).strip())
    print(f"[cb] OCR'd {new} new GOAT images; cache now {len(cache)} entries", flush=True)
    print(f"[cb] {ne}/{len(got)} GOAT rows have non-empty OCR "
          f"(avg {np.mean([len(str(t)) for t in got]):.1f} chars)", flush=True)


# --------------------------------------------------------------------------- #
# 2. Corrected manifest + OCR embeddings (row order preserved)
# --------------------------------------------------------------------------- #
def cmd_build():
    from nesymis.data.ocr_text import _load_cache

    df = pd.read_csv(config.MANIFEST_PATH, dtype=str).fillna("")
    cache = _load_cache()
    missing = [p for p in df["path"] if p not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no cached OCR (e.g. {missing[0]}). "
                         f"Run --ocr first.")
    out = df.copy()
    out["text_ocr_orig"] = df["text_ocr"]
    out["text_ocr"] = [cache.get(p, "") for p in df["path"]]
    out.to_csv(MANIFEST_C, index=False)
    print(f"[cb] wrote {MANIFEST_C} ({len(out)} rows, order preserved)", flush=True)

    # Re-encode only the OCR channel; image and caption caches stay valid.
    from nesymis.data.dataset import compose_text
    from nesymis.encoders import clip_encoder as ce
    texts = compose_text(out, "ocr")
    emb = ce.encode_texts(texts)
    np.save(OCR_EMB_C, emb)
    print(f"[cb] wrote {OCR_EMB_C} {emb.shape}", flush=True)


def load_corrected():
    """(df, y, image, u_ocr_corrected, u_caption, tr, va, te) for the corrected benchmark."""
    if not MANIFEST_C.exists() or not OCR_EMB_C.exists():
        raise SystemExit("build the corrected benchmark first: --ocr then --build")
    df = pd.read_csv(MANIFEST_C).fillna("")
    ids = json.loads(config.EMB_IDS_PATH.read_text(encoding="utf-8"))
    assert df["id"].astype(str).tolist() == [str(i) for i in ids], "row order drifted"
    v = np.load(config.IMAGE_EMB_PATH)
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)
    u_ocr = np.load(OCR_EMB_C)
    y = df["label_idx"].to_numpy().astype(int)
    tr, va, te = (df["split"].to_numpy() == s for s in ("train", "val", "test"))
    return df, y, v, u_ocr, u_cap, tr, va, te


# --------------------------------------------------------------------------- #
# 3. Did the correction work?
# --------------------------------------------------------------------------- #
def cmd_verify():
    df, y, v, u_ocr, u_cap, tr, va, te = load_corrected()
    ster = df["source"].str.startswith("wbms").to_numpy()
    same = (df["text_caption"].str.strip() == df["text_ocr"].str.strip()).to_numpy()
    orig_same = (df["text_caption"].str.strip() == df["text_ocr_orig"].str.strip()).to_numpy()

    print("## Correction check\n")
    print("| quantity | original | corrected |")
    print("|---|---|---|")
    print(f"| `caption == ocr` on GOAT rows | {int(orig_same[~ster].sum())}/"
          f"{int((~ster).sum())} | {int(same[~ster].sum())}/{int((~ster).sum())} |")
    print(f"| `caption == ocr` on WBMS rows | {int(orig_same[ster].sum())}/"
          f"{int(ster.sum())} | {int(same[ster].sum())}/{int(ster.sum())} |")
    for nm, m in (("GOAT", ~ster), ("WBMS", ster)):
        o = df.loc[m, "text_ocr_orig"].str.len()
        c = df.loc[m, "text_ocr"].str.len()
        print(f"| mean OCR length, {nm} | {o.mean():.1f} | {c.mean():.1f} |")
        print(f"| empty OCR, {nm} | {float((o == 0).mean()):.3f} | "
              f"{float((c == 0).mean()):.3f} |")

    # Residual asymmetry: can the OCR/caption pair still identify the source group?
    print("\n### Residual: is the source group still recoverable from text length alone?\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    for nm, cols in (("original", ["text_ocr_orig"]), ("corrected", ["text_ocr"])):
        X = np.stack([df[c].str.len().to_numpy() for c in cols] +
                     [df["text_caption"].str.len().to_numpy()], 1).astype(float)
        lr = LogisticRegression(max_iter=500).fit(X[tr | va], ster[tr | va])
        auc = roc_auc_score(ster[te], lr.predict_proba(X[te])[:, 1])
        print(f"* {nm}: source-group AUC from (ocr_len, caption_len) = **{auc:.3f}**")
    _save({"verify": {"goat_equal_corrected": int(same[~ster].sum()),
                      "goat_equal_original": int(orig_same[~ster].sum())}})


# --------------------------------------------------------------------------- #
# 4. Reference baselines under the corrected protocol
# --------------------------------------------------------------------------- #
def cmd_baselines(seeds=5):
    """Official (image+OCR) and oracle-text (image+caption) reference numbers."""
    import d5_modality_probes as ts

    df, y, v, u_ocr, u_cap, tr, va, te = load_corrected()
    ster = df["source"].str.startswith("wbms").to_numpy()
    rows = {}
    views = {
        "official: [v; u_ocr]": np.concatenate([v, u_ocr], 1),
        "official, image only: [v]": v,
        "official, text only: [u_ocr]": u_ocr,
        "oracle text: [v; u_cap]": np.concatenate([v, u_cap], 1),
    }
    for name, X in views.items():
        X = np.ascontiguousarray(X.astype(np.float32))
        runs = []
        for s in [config.SEED] + [int(x) for x in
                                  np.random.default_rng(config.SEED).integers(0, 2**31 - 1,
                                                                             seeds - 1)]:
            lg, _ = ts.fit_probe(X, y, tr, va, 5, s)
            runs.append(ts.score(y[te], lg[te].argmax(1), 5))
        rows[name] = {m: [r[m] for r in runs] for m in ("acc", "macro_f1")}
        print(f"  [cb] {name:<28} acc {np.mean(rows[name]['acc']):.3f} "
              f"macro-F1 {np.mean(rows[name]['macro_f1']):.3f}"
              f"+-{np.std(rows[name]['macro_f1']):.3f}", flush=True)

    # Four stereotype classes only: removes the source group entirely, so nothing
    # about GOAT-vs-WBMS can contribute to the number.
    remap = np.full(len(y), -1)
    for new, lab in enumerate(["kitchen", "leadership", "working", "shopping"]):
        remap[(df["label"] == lab).to_numpy()] = new
    for name, X in (("4-class: [v; u_ocr]", np.concatenate([v, u_ocr], 1)),
                    ("4-class: [v; u_cap]", np.concatenate([v, u_cap], 1))):
        Xs = np.ascontiguousarray(X[ster].astype(np.float32))
        ys = remap[ster]
        lg, _ = ts.fit_probe(Xs, ys, tr[ster], va[ster], 4, config.SEED)
        m = ts.score(ys[te[ster]], lg[te[ster]].argmax(1), 4)
        rows[name] = {"acc": [m["acc"]], "macro_f1": [m["macro_f1"]]}
        print(f"  [cb] {name:<28} acc {m['acc']:.3f} macro-F1 {m['macro_f1']:.3f}",
              flush=True)

    _save({"baselines": {"seeds": seeds, "rows": rows}})
    print("\n## Corrected-benchmark reference baselines "
          f"(frozen CLIP B/32, plain probe, {seeds} seeds)\n")
    print("| input | Acc | Macro-F1 |")
    print("|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {np.mean(r['acc']):.3f} | {np.mean(r['macro_f1']):.3f} |")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ocr", action="store_true")
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--baselines", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.ocr:
        cmd_ocr(a.force, a.limit)
    elif a.build:
        cmd_build()
    elif a.verify:
        cmd_verify()
    else:
        cmd_baselines(a.seeds)


if __name__ == "__main__":
    main()
