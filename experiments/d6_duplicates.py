"""D6 on every screened task, and D3 re-run on the duplicate-free test rows.

Found while screening the public tasks: 76% of HarMeme's US-politics test memes have their exact text in
train/val, and within that sub-corpus a shape-only model reaches 0.86-0.88 macro-F1 -- lookup of
seen memes, not text shape predicting harm. So for every task in the surface screen (D3): what
share of the test split duplicates a train/val item (D6), and does D3's lift survive on the
duplicate-free rows?

One criterion for every task: exact match of normalised text (lower-case alphanumerics), or CLIP
ViT-B/32 text-embedding cosine >= 0.90 (the novel-slice rule) -- the cached embeddings the D5
probes use, checked row by row against the screen's labels and splits. The statistic is the leaky
share (lookup answers the test item) in excess of its label-permutation null. A D3 flag "survives"
the duplicate-free slice when its lift there stays >= the notable lift threshold (D3_NOTABLE); the share
is not recomputed on the slice (it would need the text probe re-fitted per slice).

  python experiments/d6_duplicates.py                     # CPU only
  python experiments/d6_duplicates.py --cos 0.85,0.9,0.95 # the criterion's sensitivity
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import d3_surface_screen as ss
from nesymis import config
from inputaudit.data_tier import D3_NOTABLE, d3_surface, d6_duplicates

OUT = config.ARTIFACTS_DIR / "d6_duplicates.json"
PAPER = {"harmeme-3": "severe", "harmeme": "severe", "pridemm": "notable", "mami": "notable",
         "exist-bin": "clean", "hateful": "clean", "exist-6": "clean", "mmhs-6": "clean",
         "mmhs-bin": "clean", "mmsd": "", "pridemm-tgt": "",
         "ours-caption": "", "ours-ocr": "", "ours-ocr-corrected": ""}
EMB = {"ours-caption": config.TEXT_CAPTION_EMB_PATH, "ours-ocr": config.TEXT_OCR_EMB_PATH,
       "ours-ocr-corrected": config.TEXT_OCR_CORRECTED_EMB_PATH}


def text_embeddings(key, s):
    """CLIP B/32 text rows for task `key`, aligned with the screen's rows `s`: the D5 cache when
    its labels and splits match row by row, else the screen's own texts encoded (and cached)."""
    if key in EMB:
        return np.load(EMB[key])
    import d5_modality_probes as tsv
    t = tsv.TASKS[key]()
    same = (len(t["y"]) == len(s["y"]) and (np.asarray(t["y"]) == np.asarray(s["y"])).all()
            and (t["te"] == s["te"]).all() and ((t["tr"] | t["va"]) == (s["tr"] | s["va"])).all())
    if same:
        return t["u"]
    import hashlib
    h = hashlib.sha1("\n".join(map(str, s["texts"])).encode("utf-8")).hexdigest()[:12]
    cache = config.ARTIFACTS_DIR / f"d6_text_{key}.npz"
    if cache.exists() and str(np.load(cache)["h"]) == h:
        return np.load(cache)["u"]
    from nesymis.encoders.clip_encoder import encode_texts
    texts = [str(x) for x in s["texts"]]
    u = np.concatenate([encode_texts(texts[i:i + 256]) for i in range(0, len(texts), 256)])
    np.savez(cache, h=h, u=u)
    print(f"[dup] {key}: D5 cache misaligned; encoded {len(texts)} texts -> {cache.name}", flush=True)
    return u


def cos_sensitivity(cosines):
    """Is the 0.90 cosine criterion doing the work? D6 and the duplicate-free D3 lift at each
    cosine, for every screened task. Exact-text matches are found at any cosine, so the lowest
    row of each column is the text-only part of the criterion."""
    ts = json.load(open(config.ARTIFACTS_DIR / "d5_modality_probes.json", encoding="utf-8"))
    out = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    block = {}
    for key in PAPER:
        s = ss.TASKS[key]()
        split = np.where(s["tr"], "train", np.where(s["va"], "val", np.where(s["te"], "test", "")))
        E = text_embeddings(key, s)
        block[key] = {}
        for cos in cosines:
            d6 = d6_duplicates(s["texts"], split, E, s["y"], cos_min=cos)
            novel = d3_surface(s["texts"], s["y"], split, s["k"], None,
                               test_keep=d6["novel_test_mask"], boot=0)
            block[key][f"{cos:.2f}"] = {
                "dup_share": d6["dup_share"], "leaky_excess": d6["leaky_excess"], "D6": d6["verdict"],
                "n_novel": int(np.sum(d6["novel_test_mask"])), "lift_novel": novel["lift"],
                "lift_survives_on_novel": bool(novel["lift"] >= D3_NOTABLE[0])}
        r = block[key]
        print(f"[cos] {key:<20} " + " | ".join(
            f"{c}: dup {r[c]['dup_share']:.2f} excess {r[c]['leaky_excess']:+.3f} {r[c]['D6']:<4} "
            f"novel lift {r[c]['lift_novel']:+.3f}" for c in sorted(r)), flush=True)
    out["cos_sensitivity"] = block
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    flips = [k for k, r in block.items()
             if len({r[c]["D6"] for c in r}) > 1 or len({r[c]["lift_survives_on_novel"] for c in r}) > 1]
    print(f"[cos] tasks whose D6 verdict or duplicate-free D3 survival changes over "
          f"{sorted(block[list(block)[0]])}: {flips or 'none'}")
    print(f"_saved {OUT}_")


def main():
    ts = json.load(open(config.ARTIFACTS_DIR / "d5_modality_probes.json", encoding="utf-8"))
    rows = {}
    for key, paper_level in PAPER.items():
        s = ss.TASKS[key]()
        split = np.where(s["tr"], "train", np.where(s["va"], "val", np.where(s["te"], "test", "")))
        d6 = d6_duplicates(s["texts"], split, text_embeddings(key, s), s["y"])
        ref_key = ss.TEXT_REF.get(key, key)
        ref = float(np.mean(ts[ref_key]["views"]["text"]["macro_f1"])) if ref_key in ts else None
        full = d3_surface(s["texts"], s["y"], split, s["k"], ref)
        novel = d3_surface(s["texts"], s["y"], split, s["k"], None, test_keep=d6["novel_test_mask"])
        survives = novel["lift"] >= D3_NOTABLE[0]
        rows[key] = {"paper_level": paper_level, **{x: d6[x] for x in (
                         "dup_share", "exact_text_share", "embedding_near_share", "leaky_share",
                         "leaky_null", "leaky_excess", "n_test")},
                     "D6": d6["verdict"], "n_novel": int(np.sum(d6["novel_test_mask"])),
                     "lift_full": full["lift"], "level_full": full["level"],
                     "shape_f1_novel": novel["shape_f1"], "majority_novel": novel["majority_f1"],
                     "lift_novel": novel["lift"], "lift_survives_on_novel": bool(survives)}
        r = rows[key]
        print(f"[dup] {key:<20} D6 {r['D6']:<4} dup {r['dup_share']:.3f} (exact {r['exact_text_share']:.3f}) "
              f"leaky {r['leaky_share']:.3f} null {r['leaky_null']:.3f} excess {r['leaky_excess']:+.3f} "
              f"| lift full {r['lift_full']:+.3f} [{r['level_full'] or '-'}] -> novel {r['lift_novel']:+.3f} "
              f"(n={r['n_novel']}) {'SURVIVES' if survives else 'falls below 0.20'}", flush=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos", default="", help="comma-separated cosine thresholds; re-analysis only")
    a = ap.parse_args()
    if a.cos:
        cos_sensitivity([float(c) for c in a.cos.split(",")])
    else:
        main()
