"""Loaders + cached CLIP encoding for the new external benchmarks:
  * MMSD2.0        -- multimodal sarcasm (binary)
  * PrideMM        -- LGBTQ+ Pride memes: hate (binary) + hate target (4-class)
  * HatefulMemeMeta-- Meta Hateful Memes (binary; dev = labelled eval split)
  * HarMeme        -- Harm-C (COVID) + Harm-P (US politics): 3-way harmfulness

Each loader returns (ids, texts, paths, y, split) with split in {train,val,test}.
Nothing runs on import.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # Full_Code
DS = os.path.join(ROOT, "dataset")
CACHE = os.path.join(ROOT, "artifacts")


def encode(ids, paths, texts, name):
    """CLIP image+text embeddings, cached to artifacts/<name>_clip.npz keyed by ids."""
    cp = os.path.join(CACHE, f"{name}_clip.npz")
    if os.path.isfile(cp):
        z = np.load(cp, allow_pickle=True)
        if list(z["ids"]) == [str(i) for i in ids]:
            return z["vu"], z["u"]
    from nesymis.encoders.clip_encoder import encode_images, encode_texts
    B = 256
    v = np.concatenate([encode_images(paths[i:i + B]) for i in range(0, len(paths), B)])
    u = np.concatenate([encode_texts(texts[i:i + B]) for i in range(0, len(texts), B)])
    vu = np.concatenate([v, u], 1).astype(np.float32)
    np.savez(cp, ids=np.array([str(i) for i in ids], object), vu=vu, u=u)
    return vu, u


def load_mmsd():
    base = os.path.join(DS, "MMSD2.0")
    imgd = os.path.join(base, "dataset_image")
    ids, texts, paths, y, split = [], [], [], [], []
    for sp, fn in (("train", "train"), ("val", "valid"), ("test", "test")):
        data = json.load(open(os.path.join(base, "text_json_final", f"{fn}.json"), encoding="utf-8"))
        for r in data:
            p = os.path.join(imgd, f"{r['image_id']}.jpg")
            if not os.path.isfile(p):
                continue
            ids.append(str(r["image_id"])); texts.append(r["text"] or ""); paths.append(p)
            y.append(int(r["label"])); split.append(sp)
    return ids, texts, paths, np.array(y, int), np.array(split)


def load_pridemm():
    base = os.path.join(DS, "PrideMM")
    rows = list(csv.DictReader(open(os.path.join(base, "PrideMM.csv"), encoding="utf-8")))
    ids, texts, paths, hate, target, split = [], [], [], [], [], []
    for r in rows:
        p = os.path.join(base, "Images", r["name"])
        if not os.path.isfile(p):
            continue
        ids.append(r["name"]); texts.append(r["text"] or ""); paths.append(p)
        hate.append(int(r["hate"]))
        target.append(int(r["target"]) if r["target"] not in ("NA", "") else -1)
        split.append(r["split"])
    return ids, texts, paths, np.array(hate, int), np.array(target, int), np.array(split)


def load_hateful():
    base = os.path.join(DS, "HatefulMemeMeta")
    ids, texts, paths, y, split = [], [], [], [], []
    for sp, fn in (("train", "train.jsonl"), ("test", "dev.jsonl")):     # dev = labelled eval set
        for ln in open(os.path.join(base, fn), encoding="utf-8"):
            r = json.loads(ln)
            if "label" not in r:
                continue
            p = os.path.join(base, r["img"])
            if not os.path.isfile(p):
                continue
            ids.append(str(r["id"])); texts.append(r["text"] or ""); paths.append(p)
            y.append(int(r["label"])); split.append(sp)
    return ids, texts, paths, np.array(y, int), np.array(split)


HARM3 = {"not harmful": 0, "somewhat harmful": 1, "very harmful": 2}


def load_harmeme():
    base = os.path.join(DS, "HarMeme")
    subs = [("HarMeme_V0_DataFiles_Covid19", "harmeme_images_covid_19"),
            ("HarMeme_V0_DataFiles_USPolitics", "harmeme_images_us_pol")]
    imgroot = os.path.join(base, "_images", "HarMeme_Images")
    annroot = os.path.join(base, "_V0", "HarMeme_V0")
    ids, texts, paths, y3, split = [], [], [], [], []
    for datafile, imgdir in subs:
        for sp, fn in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test.jsonl")):
            ap = os.path.join(annroot, datafile, "datasets", "memes", "defaults", "annotations", fn)
            if not os.path.isfile(ap):
                continue
            for ln in open(ap, encoding="utf-8"):
                r = json.loads(ln)
                harm = next((HARM3[l] for l in r["labels"] if l in HARM3), None)
                if harm is None:
                    continue
                p = os.path.join(imgroot, imgdir, r["image"])
                if not os.path.isfile(p):
                    continue
                ids.append(r["id"]); texts.append((r["text"] or "").replace("\n", " ")); paths.append(p)
                y3.append(harm); split.append(sp)
    return ids, texts, paths, np.array(y3, int), np.array(split)
