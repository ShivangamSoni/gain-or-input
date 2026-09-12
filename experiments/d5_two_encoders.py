"""Removes the encoder caveat from the text-solvability sweep.

`experiments/d5_modality_probes.py` probes frozen CLIP B/32 features and finds the image
channel contributing <=0.01 macro-F1 on three external tasks. That is confounded with
feature quality: a weak vision encoder and an uninformative image channel look identical
to a linear probe. So the finding is re-measured with **SigLIP SO400M-384**, the
strongest encoder in this repo, which is also the one our own corpus was probed with.

If the conclusion holds with SO400M it is a property of those datasets. If it reverses,
the B/32 numbers were an artefact and the claim must be dropped -- which is the point of
running it.

Two mechanical differences from the CLIP sweep, both forced:

* SO400M is encoded at 384px and is ~880M params, so images are the cost. Corpora are
  encoded once and cached in `artifacts/<name>_siglip-so400m.npz`, resumable per corpus.
* MMHS150K has 138,109 rows, which is hours of encoding for one control row. Its
  **test split is kept whole** (8,411 rows) and only the training pool is subsampled,
  stratified by label. Documented in the output rather than hidden.

  python experiments/d5_two_encoders.py --list
  python experiments/d5_two_encoders.py --encode mami,exist
  python experiments/d5_two_encoders.py --encode all
  python experiments/d5_two_encoders.py --probe all --seeds 3
  python experiments/d5_two_encoders.py --report
"""
import argparse
import io
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from nesymis import config
from nesymis.encoders import openclip_encoder as oce

KEY = "siglip-so400m"
ART = config.ARTIFACTS_DIR
OUT = str(ART / "d5_two_encoders.json")
MMHS_TRAIN_CAP = 20000          # training-pool cap for MMHS; test split untouched
SEED = config.SEED


def _cache(name):
    return ART / f"{name}_{KEY}.npz"


# --------------------------------------------------------------------------- #
# Encoding: accepts PIL images, so zip-backed corpora work like path-backed ones
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_images(imgs_fn, n, batch=32):
    """imgs_fn(i, j) -> list of PIL images for rows [i, j)."""
    oce.set_model(KEY)
    s = oce._need()
    pre, model, dev = s["preprocess"], s["model"], s["device"]
    half = s["prec"] == "fp16"
    out = []
    for i in range(0, n, batch):
        imgs = imgs_fn(i, min(i + batch, n))
        x = torch.stack([pre(im) for im in imgs]).to(dev)
        if half:
            x = x.half()
        e = model.encode_image(x).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32))
        if (i // batch) % 50 == 0:
            print(f"    images {i}/{n}", flush=True)
    return np.concatenate(out)


def _paths_fn(paths):
    from nesymis.data.imageio import open_rgb

    def fn(i, j):
        out = []
        for p in paths[i:j]:
            try:
                out.append(open_rgb(p))
            except Exception:  # noqa: BLE001
                out.append(Image.new("RGB", (384, 384), (127, 127, 127)))
        return out
    return fn


def _zip_fn(z, ids):
    def fn(i, j):
        out = []
        for tid in ids[i:j]:
            try:
                out.append(Image.open(io.BytesIO(z.read(f"img_resized/{tid}.jpg"))).convert("RGB"))
            except Exception:  # noqa: BLE001
                out.append(Image.new("RGB", (384, 384), (127, 127, 127)))
        return out
    return fn


# --------------------------------------------------------------------------- #
# Corpus providers -> (ids, texts, y, split, images_fn, n, k, classes, name)
# --------------------------------------------------------------------------- #
def p_mami():
    import external_transfer as t19
    import mami_full_train as t20
    tr_t, tr_p, tr_y = t20._load_train()
    _, te_t, te_p, te_y, _ = t19._load_mami()
    texts, paths = tr_t + te_t, tr_p + te_p
    y = np.concatenate([tr_y, te_y]).astype(int)
    split = np.array(["train"] * len(tr_t) + ["test"] * len(te_t))
    return dict(name="mami", ids=list(range(len(y))), texts=texts, y=y, split=split,
                images_fn=_paths_fn(paths), n=len(y), k=2,
                label="MAMI (misogynous, binary)")


def p_exist(task6=False):
    import data_exist2024 as ex
    import data_exist2024_6class as ex6
    ids, texts, paths, raw = ex.load_all_en()
    lab = np.array([(ex6.label6(r) if task6 else ex.task4_hard(r)) for r in raw])
    return dict(name="exist", ids=ids, texts=texts, y=lab, split=None,
                images_fn=_paths_fn(paths), n=len(ids), k=6 if task6 else 2,
                label=f"EXIST-2024 ({'Task 6, 6-class' if task6 else 'sexist, binary'})")


def p_nd(loader, name, label, k, ycol=None, keepfn=None):
    import data_benchmarks as nd
    got = getattr(nd, loader)()
    if len(got) == 6:
        ids, texts, paths, y1, y2, split = got
        y = y2 if ycol == "target" else y1
    else:
        ids, texts, paths, y, split = got
    y = np.asarray(y)
    if keepfn is not None:
        y = keepfn(y)
    return dict(name=name, ids=[str(i) for i in ids], texts=texts, y=y,
                split=np.asarray(split), images_fn=_paths_fn(paths), n=len(y), k=k,
                label=label)


def p_mmhs(six=False):
    import data_mmhs150k as m6
    z, ids, texts, y6, split = m6.load_all()
    y6 = np.asarray(y6)
    split = np.asarray(split)
    # keep the whole test split; subsample the training pool, stratified
    rng = np.random.default_rng(SEED)
    te_i = np.flatnonzero(split == "test")
    pool = np.flatnonzero(split != "test")
    if len(pool) > MMHS_TRAIN_CAP:
        keep = []
        for c in np.unique(y6[pool]):
            ci = pool[y6[pool] == c]
            take = max(1, int(round(MMHS_TRAIN_CAP * len(ci) / len(pool))))
            keep.append(rng.permutation(ci)[:take])
        pool = np.sort(np.concatenate(keep))
    sel = np.sort(np.concatenate([pool, te_i]))
    return dict(name="mmhs", ids=[ids[i] for i in sel], texts=[texts[i] for i in sel],
                y=(y6[sel] if six else (y6[sel] == 2).astype(int)), split=split[sel],
                images_fn=_zip_fn(z, [ids[i] for i in sel]), n=len(sel),
                k=6 if six else 2, subsampled=len(sel) != len(ids),
                label=f"MMHS150K ({'6-class hate type' if six else 'sexist, binary'})")


# name -> provider used for ENCODING (one cache per corpus, shared by its tasks)
ENCODE = {
    "mami": p_mami,
    "exist": p_exist,
    "mmsd": lambda: p_nd("load_mmsd", "mmsd", "MMSD2.0 (sarcasm, binary)", 2),
    "pridemm": lambda: p_nd("load_pridemm", "pridemm", "PrideMM (hate, binary)", 2),
    "hateful": lambda: p_nd("load_hateful", "hateful", "Hateful Memes (binary)", 2),
    "harmeme": lambda: p_nd("load_harmeme", "harmeme", "HarMeme (harmful, binary)", 2,
                            keepfn=lambda y: (y > 0).astype(int)),
    "mmhs": p_mmhs,
}

# task key -> (corpus cache name, provider giving the LABELS for that task)
TASKS = {
    "mami": ("mami", p_mami),
    "exist-bin": ("exist", p_exist),
    "exist-6": ("exist", lambda: p_exist(task6=True)),
    "mmsd": ("mmsd", ENCODE["mmsd"]),
    "pridemm": ("pridemm", ENCODE["pridemm"]),
    "pridemm-tgt": ("pridemm", lambda: p_nd("load_pridemm", "pridemm",
                                            "PrideMM (4-class target)", 4, ycol="target")),
    "hateful": ("hateful", ENCODE["hateful"]),
    "harmeme": ("harmeme", ENCODE["harmeme"]),
    "harmeme-3": ("harmeme", lambda: p_nd("load_harmeme", "harmeme",
                                          "HarMeme (3-class intensity)", 3)),
    "mmhs-bin": ("mmhs", p_mmhs),
    "mmhs-6": ("mmhs", lambda: p_mmhs(six=True)),
}


def cmd_encode(which):
    names = list(ENCODE) if which == "all" else [x.strip() for x in which.split(",")]
    for nm in names:
        cp = _cache(nm)
        if cp.exists():
            print(f"[sig] {nm}: cached", flush=True)
            continue
        print(f"\n[sig] encoding {nm} with {KEY} ...", flush=True)
        try:
            spec = ENCODE[nm]()
            v = _encode_images(spec["images_fn"], spec["n"])
            oce.set_model(KEY)
            u = oce.encode_texts(list(spec["texts"]))
            np.savez(cp, ids=np.array([str(i) for i in spec["ids"]], object), v=v, u=u)
            print(f"[sig] {nm}: v{v.shape} u{u.shape} -> {cp}", flush=True)
        except Exception:
            print(f"[sig] !! FAILED {nm}:\n{traceback.format_exc()}", flush=True)
    print("\nSIG_ENCODE_DONE", flush=True)


def _splits(spec, n):
    import concept_scorecard_eval as ce
    from sklearn.model_selection import train_test_split
    if spec["split"] is not None:
        return ce._split_masks(np.asarray(spec["split"]))
    idx = np.arange(n)
    y = spec["y"]
    tri, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    vai, tei = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    return np.isin(idx, tri), np.isin(idx, vai), np.isin(idx, tei)


def cmd_probe(which, seeds):
    import d5_modality_probes as ts
    keys = list(TASKS) if which == "all" else [x.strip() for x in which.split(",")]
    store = json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}
    for tk in keys:
        cname, prov = TASKS[tk]
        cp = _cache(cname)
        if not cp.exists():
            print(f"[sig] {tk}: no {cp.name}; run --encode {cname}", flush=True)
            continue
        print(f"\n########## siglip solvability: {tk} ##########", flush=True)
        try:
            z = np.load(cp, allow_pickle=True)
            spec = prov()
            v, u = z["v"], z["u"]
            y = np.asarray(spec["y"])
            keep = y >= 0
            if len(y) != len(v):
                raise RuntimeError(f"{tk}: {len(y)} labels vs {len(v)} cached rows")
            tr, va, te = _splits(spec, len(y))
            if not keep.all():
                v, u, y = v[keep], u[keep], y[keep]
                tr, va, te = tr[keep], va[keep], te[keep]
            out = {}
            for vn, X in (("image", v), ("text", u), ("both", np.concatenate([v, u], 1))):
                X = np.ascontiguousarray(X.astype(np.float32))
                runs = []
                for s in [SEED] + [int(x) for x in
                                   np.random.default_rng(SEED).integers(0, 2**31 - 1,
                                                                        seeds - 1)]:
                    lg, _ = ts.fit_probe(X, y, tr, va, spec["k"], s)
                    runs.append(ts.score(y[te], lg[te].argmax(1), spec["k"]))
                out[vn] = {m: [r[m] for r in runs] for m in runs[0]
                           if runs[0][m] is not None}
                print(f"  [{tk}] {vn:<6} macro-F1 {np.mean(out[vn]['macro_f1']):.3f}"
                      f"+-{np.std(out[vn]['macro_f1']):.3f}", flush=True)
            store[tk] = {"label": spec["label"], "k": spec["k"], "n": int(len(y)),
                         "n_test": int(te.sum()), "seeds": seeds,
                         "subsampled": bool(spec.get("subsampled", False)), "views": out}
            json.dump(store, open(OUT, "w", encoding="utf-8"), indent=1)
        except Exception:
            print(f"[sig] !! FAILED {tk}:\n{traceback.format_exc()}", flush=True)
    print("\nSIG_PROBE_DONE", flush=True)


def cmd_report():
    d = json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}
    if not d:
        raise SystemExit("nothing yet -- run --encode all then --probe all")
    cl = json.load(open(ART / "d5_modality_probes.json", encoding="utf-8")) \
        if (ART / "d5_modality_probes.json").exists() else {}
    print(f"## Text-solvability under {KEY} vs CLIP B/32 (macro-F1)\n")
    print("The B/32 sweep found the image channel contributing <=0.01 on three tasks. "
          "A stronger encoder tests whether that was the data or the features.\n")
    print("| Corpus / task | n test | B/32 d both-text | SO400M image | SO400M text | "
          "SO400M both | SO400M d both-text | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for tk, r in d.items():
        g = {vn: float(np.mean(r["views"][vn]["macro_f1"]))
             for vn in ("image", "text", "both")}
        dbt = g["both"] - g["text"]
        old = cl.get(tk)
        odbt = (float(np.mean(old["views"]["both"]["macro_f1"]))
                - float(np.mean(old["views"]["text"]["macro_f1"]))) if old else float("nan")
        verdict = "image adds <=0.01" if dbt <= 0.01 else "needs both"
        star = " *(train pool subsampled)*" if r.get("subsampled") else ""
        print(f"| {r['label']}{star} | {r['n_test']} | {odbt:+.3f} | {g['image']:.3f} | "
              f"{g['text']:.3f} | {g['both']:.3f} | {dbt:+.3f} | {verdict} |")


def cmd_list():
    print(f"{'corpus':<12}{'encoded':<10}{'tasks'}")
    print("-" * 56)
    for nm in ENCODE:
        tks = [t for t, (c, _) in TASKS.items() if c == nm]
        print(f"{nm:<12}{'yes' if _cache(nm).exists() else '-':<10}{', '.join(tks)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--encode", metavar="NAMES")
    g.add_argument("--probe", metavar="KEYS")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.encode:
        cmd_encode(a.encode)
    elif a.probe:
        cmd_probe(a.probe, a.seeds)
    elif a.report:
        cmd_report()
    else:
        cmd_list()


if __name__ == "__main__":
    main()
