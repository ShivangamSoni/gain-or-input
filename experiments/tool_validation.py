"""Validate the inputaudit tool against the paper's numbers.

Part 1 -- our corpus, as first assembled and under the uniform-OCR protocol, every data-tier
check (D1-D5) through the tool's public API. Expected from the paper:
  D1  the caption FLAGGED (produced differently per source); the OCR field FLAGGED as first
      assembled, ok under uniform OCR
  D2  caption == OCR FLAGGED: 1133 rows as first assembled, 43 under uniform OCR
  D3  shape-only macro-F1 0.631 (caption) / 0.369 (OCR, first) / 0.359 (OCR, uniform) -- exact
  D4  source recoverable from shape: AUC 0.978 (caption) / 0.866 (OCR, first) / 0.837 (uniform)
  D5  the caption is not D5's text view (OCR is the official input); OCR adds ~0 over the image
Part 2 -- the 11 external tasks: D3 through the tool (must reproduce d3_surface_screen.json
exactly, and the paper's flags), and D5's decision rules applied to the saved CLIP and SigLIP
probe scores (d5_modality_probes.json, d5_two_encoders.json).

  CUDA_VISIBLE_DEVICES= python experiments/tool_validation.py      # CPU only; the GPU may be busy
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from nesymis import config
from inputaudit.data_tier import d3_surface, d5_decide, run_data_tier
from inputaudit.report import data_report

ART = config.ARTIFACTS_DIR
OUT = ART / "tool_validation.json"
CAPTION = {"wbms": "title/caption that accompanied the image at its source",
           "goat": "GOAT-Bench's released meme text"}
PROVENANCE = {
    "first": {"text_caption": CAPTION,
              "text_ocr": {"wbms": "EasyOCR on the image", "goat": "GOAT-Bench's released meme text"}},
    "uniform": {"text_caption": CAPTION,
                "text_ocr": {"wbms": "EasyOCR on the image", "goat": "EasyOCR on the image"}},
}
EXPECT = {"first": {"D2": 1133, "D3": {"text_caption": 0.631, "text_ocr": 0.369},
                    "D4": {"text_caption": 0.978, "text_ocr": 0.866}},
          "uniform": {"D2": 43, "D3": {"text_caption": 0.631, "text_ocr": 0.359},
                      "D4": {"text_caption": 0.978, "text_ocr": 0.837}}}
PAPER_FLAGS = {"harmeme-3": "severe", "harmeme": "severe", "pridemm": "notable", "mami": "notable",
               "exist-bin": "clean", "hateful": "clean", "exist-6": "clean", "mmhs-6": "clean",
               "mmhs-bin": "clean", "mmsd": "", "pridemm-tgt": ""}


def corpus(version):
    if version == "first":
        df = pd.read_csv(config.MANIFEST_PATH).fillna("")
        u_ocr, sig = np.load(config.TEXT_OCR_EMB_PATH), np.load(ART / "enc_siglip-so400m.npz")
    else:
        df = pd.read_csv(config.CORRECTED_MANIFEST_PATH).fillna("")
        u_ocr, sig = np.load(config.TEXT_OCR_CORRECTED_EMB_PATH), np.load(ART / "enc_siglip-so400m_uniform.npz")
    df["source_group"] = np.where(df["source"].str.startswith("wbms"), "wbms", "goat")
    emb = {"clip-b32": {"image": np.load(config.IMAGE_EMB_PATH), "text:text_ocr": u_ocr,
                        "text:text_caption": np.load(config.TEXT_CAPTION_EMB_PATH)},
           "siglip-so400m": {"image": sig["v"], "text:text_ocr": sig["u_ocr"]}}
    return df, emb


def part1():
    out = {}
    for ver in ("first", "uniform"):
        df, emb = corpus(ver)
        r = run_data_tier(df, "label", ["text_ocr", "text_caption"], "split", "source_group",
                          PROVENANCE[ver], emb, seeds=3)
        (ART / f"tool_validation_{ver}.md").write_text(data_report(r), encoding="utf-8")
        eq = r["D2"]["relations"].get("text_ocr == text_caption", {})
        chk = {"D1": {f: v["verdict"] for f, v in r["D1"]["fields"].items()},
               "D2 caption==OCR rows": (eq.get("support"), EXPECT[ver]["D2"], eq.get("verdict")),
               "D3 shape F1": {f: (round(r["D3"][f]["shape_f1"], 3), EXPECT[ver]["D3"][f])
                               for f in EXPECT[ver]["D3"]},
               "D4 source AUC": {f: (round(r["D4"][f]["source_auc_from_shape"], 3), EXPECT[ver]["D4"][f])
                                 for f in EXPECT[ver]["D4"]},
               "D5": {"verdict": r["D5"]["verdict"], "claims": r["D5"]["claims"],
                      "probes": {n: e["macro_f1"] for n, e in r["D5"]["encoders"].items()}}}
        out[ver] = {"checks": chk, "full": r}
        print(f"\n### ours, {ver}\n{json.dumps(chk, indent=1)}", flush=True)
    return out


def part2():
    import d3_surface_screen as ss
    saved = json.load(open(ART / "d3_surface_screen.json", encoding="utf-8"))
    ts = json.load(open(ART / "d5_modality_probes.json", encoding="utf-8"))
    sg = json.load(open(ART / "d5_two_encoders.json", encoding="utf-8"))
    rows = {}
    for key, paper_flag in PAPER_FLAGS.items():
        s = ss.TASKS[key]()
        split = np.where(s["tr"], "train", np.where(s["va"], "val", np.where(s["te"], "test", "")))
        ref = float(np.mean(ts[key]["views"]["text"]["macro_f1"]))
        d3 = d3_surface(s["texts"], s["y"], split, s["k"], ref)
        d5 = d5_decide({"clip-b32": {v: ts[key]["views"][v]["macro_f1"] for v in ("image", "text", "both")},
                        "siglip-so400m": {v: sg[key]["views"][v]["macro_f1"] for v in ("image", "text", "both")}})
        rows[key] = {"shape_f1_tool": d3["shape_f1"], "shape_f1_paper": saved[key]["surface"]["macro_f1"],
                     "level_tool": d3["level"], "level_paper": paper_flag, "share": d3["share"],
                     "lift": d3["lift"], "d5_verdict": d5["verdict"], "d5_claims": d5["claims"]}
        r = rows[key]
        print(f"[validate] {key}: shape F1 tool {r['shape_f1_tool']:.3f} vs paper {r['shape_f1_paper']:.3f} | "
              f"level {r['level_tool'] or '-'} vs {r['level_paper'] or '-'} | D5 {r['d5_verdict']}; "
              f"image: {r['d5_claims']['image'][:60]}", flush=True)
    return rows


def main():
    p1 = part1()
    p2 = part2()
    json.dump({"ours": {v: p1[v]["checks"] for v in p1}, "external": p2}, open(OUT, "w", encoding="utf-8"),
              indent=1, default=str)
    exact = all(abs(r["shape_f1_tool"] - r["shape_f1_paper"]) < 1e-9 for r in p2.values())
    same_flags = all(r["level_tool"] == r["level_paper"] for r in p2.values())
    print(f"\nexternal D3 exact reproduction: {exact}; flags match the paper: {same_flags}")
    print(f"_saved {OUT}; reports: {ART / 'tool_validation_first.md'}, {ART / 'tool_validation_uniform.md'}_")


if __name__ == "__main__":
    main()
