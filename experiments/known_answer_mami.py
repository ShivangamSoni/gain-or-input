"""A second known-answer benchmark for the system tier: MAMI.

MMHS150K plants a *content* asymmetry: the tweet text says things the in-image text does not. MAMI
plants the other kind. Its organisers ship a transcription of the text inside each meme; running our
own EasyOCR over the same images recovers the same words less well. A component that reads the
official transcription therefore has no more information than one that reads our OCR -- only a
cleaner rendering of it -- which is the *provenance and quality* asymmetry D1 is about.

System under test (as in that test): equal-weight log-linear pooling of
  N  baseline: a probe on CLIP image + OUR EasyOCR text
  P  added component, one of
       P_trans   the organisers' transcription      (positive control: cleaner text N never gets)
       P_same    our EasyOCR text, which N reads    (S2's equal-input configuration)
       P_siglip  image + our OCR text via SigLIP    (negative control: same raw inputs, better encoder)

Expected, fixed before the run: positive control S1 FLAG, S2 FLAG; negative control S1 ok, S2 ok.
Tasks: MAMI misogynous and MAMI stereotype. Splits: official test; official train split with a
stratified 15% validation carve (seed 42), as in the source-controlled evaluation. 5 seeds.

  python experiments/known_answer_mami.py [--ocr]      # --ocr runs EasyOCR first (about an hour)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import train_test_split

import known_answer_mmhs as a1
import common as C
from nesymis import config
from inputaudit import system_tier as st
from inputaudit.probes import macro_f1
from d5_modality_probes import fit_probe

OUT = config.ARTIFACTS_DIR / "known_answer_mami.json"
OCR_OUT = config.ARTIFACTS_DIR / "mami_our_ocr.json"
EMB = config.ARTIFACTS_DIR / "mami_our_ocr_emb.npz"
SEEDS = a1.SEEDS
AS, EQ = "as built (P reads the official transcription)", "equal inputs (P reads our OCR text)"
NEG = "P reads image + our OCR text via SigLIP"


def rows():
    """MAMI paths, transcriptions and labels, in the order of the cached CLIP embeddings."""
    import repair_source_controlled as source_controlled
    texts, v, u, mis, ster, n_tr = source_controlled._mami_rows()
    import external_transfer as t19
    import mami_full_train as t20
    import pandas as pd
    tr = pd.read_csv(os.path.join(t20.MAMI_DIR, "TRAINING", "training.csv"), sep="\t", encoding="utf-8-sig")
    paths = [os.path.join(t20.MAMI_DIR, "TRAINING", fn) for fn in tr["file_name"]]
    paths = [p for p in paths if os.path.isfile(p)]
    _, _, te_p, _, _ = t19._load_mami()
    return texts, paths + list(te_p), v, u, mis, ster, n_tr


def run_ocr(paths):
    """Our EasyOCR pipeline, the one the uniform-OCR protocol uses, over the MAMI images."""
    import easyocr
    from tqdm import tqdm

    from nesymis.data.imageio import open_rgb
    from nesymis.data.ocr_text import _prep, _read
    cache = json.load(open(OCR_OUT, encoding="utf-8")) if OCR_OUT.exists() else {}
    todo = [p for p in paths if p not in cache]
    print(f"[mami] OCR: {len(cache)} cached, {len(todo)} to read", flush=True)
    if todo:
        reader = easyocr.Reader(["en"], gpu=True)
        for n, p in enumerate(tqdm(todo, desc="OCR-mami"), 1):
            try:
                cache[p] = _read(reader, _prep(open_rgb(p)))
            except Exception as e:                                  # noqa: BLE001
                cache[p] = ""
                tqdm.write(f"  [warn] {os.path.basename(p)}: {e}")
            if n % 200 == 0:
                json.dump(cache, open(OCR_OUT, "w", encoding="utf-8"))
        json.dump(cache, open(OCR_OUT, "w", encoding="utf-8"))
    return [cache.get(p, "") for p in paths]


def embeddings(ocr_texts):
    """CLIP and SigLIP text embeddings of our OCR text, cached."""
    if EMB.exists():
        z = np.load(EMB, allow_pickle=True)
        if len(z["u"]) == len(ocr_texts):
            return z["u"], z["su"]
    from nesymis.encoders.clip_encoder import encode_texts
    u = np.concatenate([encode_texts(ocr_texts[i:i + 512]) for i in range(0, len(ocr_texts), 512)])
    from nesymis.encoders import openclip_encoder as oce
    oce.set_model("siglip-so400m")
    su = oce.encode_texts(ocr_texts)
    np.savez(EMB, u=u, su=su)
    print(f"[mami] encoded {len(ocr_texts)} OCR texts (CLIP {u.shape}, SigLIP {su.shape})", flush=True)
    return u, su


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true", help="read the images with EasyOCR first")
    a = ap.parse_args()
    texts, paths, v, u_trans, mis, ster, n_tr = rows()
    assert len(paths) == len(texts), f"{len(paths)} images for {len(texts)} transcriptions"
    ocr = run_ocr(paths) if a.ocr else json.load(open(OCR_OUT, encoding="utf-8"))
    if not a.ocr:
        ocr = [ocr.get(p, "") for p in paths]
    nonempty = float(np.mean([bool(t.strip()) for t in ocr]))
    chars = (float(np.mean([len(t) for t in ocr])), float(np.mean([len(t) for t in texts])))
    print(f"[mami] {len(paths)} memes; our OCR non-empty on {100 * nonempty:.0f}%, "
          f"mean length {chars[0]:.0f} chars against {chars[1]:.0f} for the transcription", flush=True)
    u_ocr, su_ocr = embeddings(ocr)
    sig = np.load(config.ARTIFACTS_DIR / "mami_siglip-so400m.npz", allow_pickle=True)

    N = len(texts)
    te = np.arange(N) >= n_tr
    res = {"n": N, "n_train_pool": int(n_tr), "our_ocr_nonempty": nonempty,
           "mean_chars": {"our_ocr": chars[0], "transcription": chars[1]}, "seeds": SEEDS}
    views = {"N": np.concatenate([v, u_ocr], 1), "trans": u_trans, "same": u_ocr,
             "siglip": np.concatenate([sig["v"], su_ocr], 1)}

    for task, ylab in (("MAMI misogynous", mis), ("MAMI stereotype", ster)):
        t0 = time.time()
        y = np.asarray(ylab).astype(int)
        tri, vai = train_test_split(np.arange(n_tr), test_size=0.15, stratify=y[:n_tr],
                                    random_state=config.SEED)
        tr = np.isin(np.arange(N), tri)
        va = np.isin(np.arange(N), vai)
        cache = {}

        def lg(view, seed):
            if (view, seed) not in cache:
                out, _ = fit_probe(np.ascontiguousarray(views[view].astype(np.float32)), y, tr, va, 2, seed)
                cache[(view, seed)] = a1.log_softmax(out[te])
            return cache[(view, seed)]

        def run(name, cfg, seed):
            ln, lp = lg("N", seed), lg(cfg["P"], seed)
            return (0.5 * ln + 0.5 * lp).argmax(1), ln.argmax(1)

        yt = y[te]
        pos = {"s1": st.s1_inventory({"N": ["image", "our OCR text"], "P": ["official transcription"]},
                                     ["N", "P"], ["N"],
                                     ["image", "our OCR text", "official transcription"]),
               "s2": st.s2_same_input(run, {AS: {"P": "trans"}, EQ: {"P": "same"}}, SEEDS, yt, 2, equal=EQ)}
        neg = {"s1": st.s1_inventory({"N": ["image", "our OCR text"], "P": ["image", "our OCR text"]},
                                     ["N", "P"], ["N"], ["image", "our OCR text"]),
               "s2": st.s2_same_input(run, {NEG: {"P": "siglip"}}, SEEDS, yt, 2, equal=NEG)}
        alone = {k: float(np.mean([macro_f1(yt, lg(k, s).argmax(1), 2) for s in SEEDS])) for k in views}
        res[task] = {"positive_control": pos, "negative_control": neg, "components_alone": alone,
                     "n_test": int(te.sum()), "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
        c, sw = pos["s2"]["configs"], pos["s2"]["configs"][AS]["swing"]
        print(f"[mami] {task} | alone {json.dumps({k: round(x, 3) for k, x in alone.items()})}\n"
              f"  positive: as built {c[AS]['margin']['mean']:+.3f} {np.round(c[AS]['margin']['ci'], 3)}, "
              f"equal {c[EQ]['margin']['mean']:+.3f} {np.round(c[EQ]['margin']['ci'], 3)}, "
              f"swing {sw['mean']:+.3f} {np.round(sw['ci'], 3)}\n"
              f"  positive: S1 {pos['s1']['verdict']} | S2 {pos['s2']['verdict']} "
              f"(first-stated rule: {pos['s2']['verdict_no_gain_survives']})\n"
              f"  negative: margin {list(neg['s2']['configs'].values())[0]['margin']['mean']:+.3f} "
              f"| S1 {neg['s1']['verdict']} | S2 {neg['s2']['verdict']}", flush=True)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
