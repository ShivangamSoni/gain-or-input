"""Known-answer test of the audit's system tier on a public benchmark (MMHS150K).

MMHS150K pairs every tweet's own text (post-level text, like our WBMS captions) with the dataset
creators' OCR of the text inside the image. That reproduces the shape of the confound in our
case study on data we did not build, so the audit's system tier can be tested where the answer
is known in advance.

System under test: log-linear pooling (equal weights, fixed a priori) of two probes --
  N  the baseline component: image + in-image text      (CLIP ViT-B/32 [v; u_img_text])
  P  the added component, one of:
       P_post    reads the TWEET text                    (the asymmetric design)
       P_same    reads the in-image text N also reads    (S2's equal-input configuration)
       P_siglip  reads image + in-image text through SigLIP SO400M (same raw inputs, a better
                 encoder -- a gain that is real, not an input asymmetry)

PRE-REGISTERED EXPECTATIONS (fixed before running):
  positive control (P_post):  S1 FLAG (P reads an input N never sees); S2 FLAG (the gain appears
                              with the tweet text and not with equal inputs)
  negative control (P_siglip): S1 ok (same raw inputs); S2 ok (its gain, if any, is measured
                              under equal inputs)
Whatever the outcomes are, they are reported.

Rows: the 28,410-tweet pool of experiments/d5_two_encoders.py (whole official test split + a
stratified 20,000-tweet training pool). Tasks: 6-class hate type and sexist-vs-rest. 5 seeds.

  python experiments/known_answer_mmhs.py
"""
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import data_mmhs150k as m6
from nesymis import config
from inputaudit import system_tier as st
from inputaudit.probes import macro_f1
from d5_modality_probes import fit_probe

ART = config.ARTIFACTS_DIR
OUT = ART / "known_answer_mmhs.json"
CACHE = ART / "mmhs_pool_split_text.npz"
SEEDS = [int(s) for s in C.SEEDS11[:5]]
CAP = 20000


def load_pool():
    """The d5_two_encoders pool, with tweet text and in-image text kept apart."""
    gt = json.load(open(m6.GT, encoding="utf-8"))
    z = zipfile.ZipFile(m6.ZIP)
    names = set(z.namelist())
    split_of = {}
    for s in ("train", "val", "test"):
        for tid in z.read(f"splits/{s}_ids.txt").decode().split():
            split_of[tid] = s
    ids, post, img, y, split = [], [], [], [], []
    for tid, s in split_of.items():
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
        ids.append(tid)
        post.append(m6.clean(e["tweet_text"]))
        img.append(it.strip())
        y.append(maj)
        split.append(s)
    y, split = np.array(y, int), np.array(split)
    rng = np.random.default_rng(config.SEED)                 # the same draw as d5_two_encoders
    te_i = np.flatnonzero(split == "test")
    pool = np.flatnonzero(split != "test")
    keep = []
    for c in np.unique(y[pool]):
        ci = pool[y[pool] == c]
        keep.append(rng.permutation(ci)[:max(1, int(round(CAP * len(ci) / len(pool))))])
    sel = np.sort(np.concatenate([np.sort(np.concatenate(keep)), te_i]))
    return ([ids[i] for i in sel], [post[i] for i in sel], [img[i] for i in sel], y[sel],
            split[sel], ids)


def embeddings(ids, post, img, all_ids, need=()):
    """CLIP v (from the full cache), CLIP u_post and u_img, SigLIP v and u_img. `need` names extra
    views to top the cache up with: "su_post" is SigLIP's text embedding of the tweet, which the
    multi-system runs use when SigLIP is the baseline's encoder."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == ids:
            out = {k: z[k] for k in z.files if k != "ids"}
            missing = [k for k in need if k not in out]
            if not missing:
                return out
            if "su_post" in missing:
                from nesymis.encoders import openclip_encoder as oce
                oce.set_model("siglip-so400m")
                out["su_post"] = oce.encode_texts(post)
                print(f"[a1] cache topped up with su_post ({len(post)} tweets)", flush=True)
            np.savez(CACHE, ids=np.array(ids, object), **out)
            return out
    full = np.load(ART / "mmhs_full_clip.npz", allow_pickle=True)
    pos = {t: i for i, t in enumerate(full["ids"])}
    assert list(full["ids"]) == all_ids, "full CLIP cache out of order"
    v = full["vu"][[pos[t] for t in ids], :512]
    sig = np.load(ART / "mmhs_siglip-so400m.npz", allow_pickle=True)
    assert list(sig["ids"]) == ids, "SigLIP pool differs from the reconstructed pool"
    from nesymis.encoders.clip_encoder import encode_texts
    u_post = np.concatenate([encode_texts(post[i:i + 512]) for i in range(0, len(post), 512)])
    u_img = np.concatenate([encode_texts(img[i:i + 512]) for i in range(0, len(img), 512)])
    from nesymis.encoders import openclip_encoder as oce
    oce.set_model("siglip-so400m")
    s_img = oce.encode_texts(img)
    out = {"v": v, "u_post": u_post, "u_img": u_img, "sv": sig["v"], "su_img": s_img}
    np.savez(CACHE, ids=np.array(ids, object), **out)
    return out


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    ids, post, img, y6, split, all_ids = load_pool()
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"[a1] pool {len(ids)}: train {tr.sum()} val {va.sum()} test {te.sum()}; "
          f"in-image text present {np.mean([bool(t) for t in img]):.2f}", flush=True)
    E = embeddings(ids, post, img, all_ids)
    views = {"N": np.concatenate([E["v"], E["u_img"]], 1),
             "P_post": E["u_post"], "P_same": E["u_img"],
             "P_siglip": np.concatenate([E["sv"], E["su_img"]], 1)}
    res = {}
    for task, y, k in (("6-class hate type", y6, 6), ("sexist vs rest", (y6 == 2).astype(int), 2)):
        cache = {}

        def logits(view, seed):
            key = (view, seed)
            if key not in cache:
                lg, _ = fit_probe(np.ascontiguousarray(views[view].astype(np.float32)), y, tr, va,
                                  k, seed)
                cache[key] = log_softmax(lg[te])
            return cache[key]

        def run(name, cfg, seed):
            ln = logits("N", seed)
            lp = logits(cfg["P"], seed)
            return (0.5 * ln + 0.5 * lp).argmax(1), ln.argmax(1)

        yt = y[te]
        pos = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": ["tweet text"]},
                                     ["N", "P"], ["N"], ["image", "image text", "tweet text"]),
               "s2": st.s2_same_input(run, {"as built (P reads the tweet)": {"P": "P_post"},
                                            "equal inputs (P reads the image text)": {"P": "P_same"}},
                                      SEEDS, yt, k, equal="equal inputs (P reads the image text)")}
        neg = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": ["image", "image text"]},
                                     ["N", "P"], ["N"], ["image", "image text"]),
               "s2": st.s2_same_input(run, {"P reads image + image text via SigLIP": {"P": "P_siglip"}},
                                      SEEDS, yt, k, equal="P reads image + image text via SigLIP")}
        alone = {v: float(np.mean([macro_f1(yt, logits(v, s).argmax(1), k) for s in SEEDS]))
                 for v in views}
        res[task] = {"positive_control": pos, "negative_control": neg, "components_alone": alone}
        p2, n2 = pos["s2"], neg["s2"]
        print(f"\n[a1] {task}: components alone {json.dumps({a: round(b, 3) for a, b in alone.items()})}")
        for nm, r in p2["configs"].items():
            print(f"  positive | {nm}: margin {r['margin']['mean']:+.3f} {np.round(r['margin']['ci'], 3)}")
        sw = p2["configs"]["as built (P reads the tweet)"]["swing"]
        print(f"  positive | S1 {pos['s1']['verdict']} (only in system: {pos['s1']['only_in_system']}) | "
              f"S2 {p2['verdict']} (first-stated rule: {p2['verdict_no_gain_survives']}) | "
              f"swing {sw['mean']:+.3f} {np.round(sw['ci'], 3)}")
        for nm, r in n2["configs"].items():
            print(f"  negative | {nm}: margin {r['margin']['mean']:+.3f} {np.round(r['margin']['ci'], 3)}")
        print(f"  negative | S1 {neg['s1']['verdict']} | S2 {n2['verdict']} "
              f"(first-stated rule: {n2['verdict_no_gain_survives']})", flush=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
