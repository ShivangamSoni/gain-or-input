"""the audit's D2 and D4 on public data, and D4 as an explanation of a D3 flag.

D2 and D4 need what most meme benchmarks lack: two text fields (D2) or two sources (D4). Two of
the public corpora have them:

  MMHS150K  every tweet's own text AND the dataset creators' OCR of the text in the image ->
            D2 (relations between the two fields) on all 138k labelled tweets
  HarMeme   two sub-corpora, COVID-19 memes and US-politics memes, released together -> D4
            (can text shape identify the sub-corpus, and does the sub-corpus predict the label?)

HarMeme is the task D3 flagged as SEVERE (shape-only reaches 92% of a text probe). If D4 flags it
too, D3 is re-run WITHIN each sub-corpus: a shortcut that falls away within source is a
composition effect, which is what the audit's D3 -> D4 chain is for.

  python experiments/d2_d4_external_tasks.py        # CPU only
"""
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import data_mmhs150k as m6
from data_benchmarks import DS, HARM3
from nesymis import config
from inputaudit.data_tier import d1_provenance, d2_field_relations, d3_surface, d4_source

OUT = config.ARTIFACTS_DIR / "d2_d4_external_tasks.json"


def mmhs_all():
    gt = json.load(open(m6.GT, encoding="utf-8"))
    z = zipfile.ZipFile(m6.ZIP)
    names = set(z.namelist())
    rows = []
    for s in ("train", "val", "test"):
        for tid in z.read(f"splits/{s}_ids.txt").decode().split():
            e = gt.get(tid)
            if e is None or f"img_resized/{tid}.jpg" not in names:
                continue
            maj = m6.majority(e["labels"])
            if maj < 0:
                continue
            it = ""
            if f"img_txt/{tid}.json" in names:
                try:
                    it = json.loads(z.read(f"img_txt/{tid}.json")).get("img_text", "")
                except Exception:
                    it = ""
            rows.append({"tweet_text": m6.clean(e["tweet_text"]), "image_text": it.strip(),
                         "label": m6.NAMES[maj], "split": s})
    return pd.DataFrame(rows)


def harmeme_all():
    base = os.path.join(DS, "HarMeme")
    subs = [("HarMeme_V0_DataFiles_Covid19", "harmeme_images_covid_19", "covid-19"),
            ("HarMeme_V0_DataFiles_USPolitics", "harmeme_images_us_pol", "us-politics")]
    imgroot = os.path.join(base, "_images", "HarMeme_Images")
    annroot = os.path.join(base, "_V0", "HarMeme_V0")
    rows = []
    for datafile, imgdir, src in subs:
        for sp, fn in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test.jsonl")):
            ap = os.path.join(annroot, datafile, "datasets", "memes", "defaults", "annotations", fn)
            if not os.path.isfile(ap):
                continue
            for ln in open(ap, encoding="utf-8"):
                r = json.loads(ln)
                harm = next((HARM3[l] for l in r["labels"] if l in HARM3), None)
                if harm is None or not os.path.isfile(os.path.join(imgroot, imgdir, r["image"])):
                    continue
                rows.append({"text": (r["text"] or "").replace("\n", " "), "source": src,
                             "harm3": harm, "harmful": int(harm > 0), "split": sp})
    return pd.DataFrame(rows)


def main():
    out = {}
    # ---- MMHS150K: D2 between the tweet text and the in-image text ---------------------------
    mm = mmhs_all()
    print(f"[a2] MMHS150K: {len(mm)} tweets; in-image text present {(mm.image_text != '').mean():.2f}",
          flush=True)
    out["mmhs150k"] = {"n": len(mm), "D2": d2_field_relations(mm, ["tweet_text", "image_text"], "label"),
                       "D1_profile": d1_provenance(mm, ["tweet_text", "image_text"])["profile"]}
    fl = {k: v for k, v in out["mmhs150k"]["D2"]["relations"].items() if v["verdict"] == "FLAG"}
    print(f"[a2] MMHS150K D2 verdict {out['mmhs150k']['D2']['verdict']}; flagged: {list(fl) or 'none'}")
    for k, v in out["mmhs150k"]["D2"]["relations"].items():
        print(f"      {k}: {v['support']} rows, top {v['top_class']} precision {v['precision']:.3f}")

    # ---- HarMeme: D4 on the two sub-corpora, then D3 within each ---------------------------
    hm = harmeme_all()
    print(f"\n[a2] HarMeme: {len(hm)} memes; sources {hm.source.value_counts().to_dict()}", flush=True)
    res = {"n": len(hm), "sources": hm.source.value_counts().to_dict(),
           "harmful_rate_by_source": hm.groupby("source").harmful.mean().round(3).to_dict(),
           "D1_profile": d1_provenance(hm, ["text"], "source")["profile"]}
    for task, col, k in (("harmful (binary)", "harmful", 2), ("3-class intensity", "harm3", 3)):
        y = hm[col].to_numpy()
        d4 = d4_source(hm.text.tolist(), hm.source.to_numpy(), y, hm.split.to_numpy())
        d3_all = d3_surface(hm.text.tolist(), y, hm.split.to_numpy(), k)
        within = {}
        for src in sorted(hm.source.unique()):
            m = (hm.source == src).to_numpy()
            within[src] = d3_surface(hm.text[m].tolist(), y[m], hm.split.to_numpy()[m], k)
        res[task] = {"D4": d4, "D3_pooled": d3_all, "D3_within_source": within}
        print(f"[a2] HarMeme {task}: D4 {d4['verdict']} (source AUC from shape "
              f"{d4['source_auc_from_shape']:.3f}, source-label V {d4['source_label_cramers_v']:.2f}) | "
              f"D3 lift pooled {d3_all['lift']:+.3f} | within: " +
              ", ".join(f"{s} {w['lift']:+.3f}" for s, w in within.items()), flush=True)
    out["harmeme"] = res
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
