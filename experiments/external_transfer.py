"""Table 19 - external-dataset transfer (leakage control #5).

Applies the DEPLOYED system -- learned rules, classifier, and policy, all fixed,
with NO re-induction and NO fine-tuning -- to an external misogyny dataset, and
evaluates BINARY misogyny detection (our four stereotype classes collapse to
"misogynous" via the meta-rule R5; the external taxonomy is binary-only).

Primary target is the MAMI benchmark (SemEval-2022 Task 5, Fersini et al. 2022)
in its official layout under  dataset/MAMI/  (test/Test.csv transcription file +
test_labels.txt gold labels + test/ images).  MAMI's Sub-task B includes a
"stereotype" sub-label, so we additionally report recall split by whether the
misogynous meme is stereotype-based (our taxonomy) or not.

Generic fallback: place any dataset under  dataset/external/<name>/  with a
labels .csv/.tsv (image file name, meme text, binary misogyny label; column
aliases below) and the images alongside it (root or a subfolder).
If no external dataset is present the table reports SKIPPED (run_all-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob

import numpy as np
import pandas as pd

import common as C
from nesymis import config
from nesymis.symbolic import rule_induction

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
EXTERNAL_DIR = os.path.join(DATASET_DIR, "external")
MAMI_DIR = os.path.join(DATASET_DIR, "MAMI")
FILE_COLS = ("file_name", "file", "image", "img", "filename")
TEXT_COLS = ("text transcription", "text", "caption", "meme text", "transcription")
LABEL_COLS = ("misogynous", "misogyny", "label", "sexist", "misogynistic")


def _load_mami():
    """Official MAMI test set: Test.csv (TSV transcriptions) + test_labels.txt
    (TSV: file_name, misogynous, shaming, stereotype, objectification, violence)."""
    tf = os.path.join(MAMI_DIR, "test", "Test.csv")
    lf = os.path.join(MAMI_DIR, "test_labels.txt")
    if not (os.path.isfile(tf) and os.path.isfile(lf)):
        return None
    tx = pd.read_csv(tf, sep="\t")
    lb = pd.read_csv(lf, sep="\t", header=None,
                     names=["file_name", "misogynous", "shaming", "stereotype",
                            "objectification", "violence"])
    m = tx.merge(lb, on="file_name", how="inner")
    paths = [os.path.join(MAMI_DIR, "test", fn) for fn in m["file_name"]]
    ok = np.array([os.path.isfile(p) for p in paths])
    m = m.loc[ok].reset_index(drop=True)
    return ("MAMI (SemEval-2022 T5 test)",
            m["Text Transcription"].fillna("").astype(str).tolist(),
            [p for p, k in zip(paths, ok) if k],
            m["misogynous"].to_numpy(int),
            m["stereotype"].to_numpy(int))


def _find_dataset():
    for d in sorted(glob.glob(os.path.join(EXTERNAL_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        for lf in glob.glob(os.path.join(d, "*.csv")) + glob.glob(os.path.join(d, "*.tsv")):
            sep = "\t" if lf.endswith(".tsv") else None
            try:
                df = pd.read_csv(lf, sep=sep, engine="python")
            except Exception:
                continue
            cols = {c.lower().strip(): c for c in df.columns}
            fcol = next((cols[k] for k in FILE_COLS if k in cols), None)
            tcol = next((cols[k] for k in TEXT_COLS if k in cols), None)
            lcol = next((cols[k] for k in LABEL_COLS if k in cols), None)
            if fcol and tcol and lcol:
                return d, lf, df, fcol, tcol, lcol
    return None


def _resolve_images(root, names):
    subdirs = [root] + [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    paths, keep = [], []
    for i, nm in enumerate(names):
        for sd in subdirs:
            p = os.path.join(sd, str(nm))
            if os.path.isfile(p):
                paths.append(p)
                keep.append(i)
                break
    return paths, np.array(keep, dtype=int)


def _load_generic():
    found = _find_dataset()
    if found is None:
        return None
    root, lf, xdf, fcol, tcol, lcol = found
    texts_all = xdf[tcol].fillna("").astype(str).tolist()
    labels_all = xdf[lcol].astype(int).to_numpy()
    paths, keep = _resolve_images(root, xdf[fcol].tolist())
    return (os.path.basename(root), [texts_all[i] for i in keep], paths,
            labels_all[keep], None)


def generate(ctx=None):
    loaded = _load_mami() or _load_generic()
    if loaded is None:
        return ("## Table 19 - External-dataset transfer (SKIPPED)\n\n"
                "_No external dataset found. Provide MAMI under `dataset/MAMI/` "
                "(official layout) or any dataset under `dataset/external/<name>/`\n"
                "with a labels .csv/.tsv (image file, meme text, binary misogyny label)\n"
                "and its images; then re-run `python experiments/external_transfer.py`._\n")

    name, texts, paths, y_bin, stereo = loaded
    c = ctx or C.build_context()
    sysm = c.system
    print(f"  [t19] {name}: {len(paths)} memes with images "
          f"({int(y_bin.sum())} misogynous)", flush=True)

    from nesymis.encoders.clip_encoder import encode_images, encode_texts

    B = 256
    v = np.concatenate([encode_images(paths[i:i + B]) for i in range(0, len(paths), B)])
    u = np.concatenate([encode_texts(texts[i:i + B]) for i in range(0, len(texts), B)])
    vu = np.concatenate([v, u], axis=1).astype(np.float32)

    cont = sysm._concept_features(u, v)             # affect-free; concepts fire on caption OR image
    rd = rule_induction.evaluate_features(sysm.rule_sets, cont, None)
    fired = np.stack([rd[f"{cl}_fired"].to_numpy() for cl in config.STEREO_CLASSES], 1)

    logits = sysm.neural_logits(vu)
    nesymis = sysm.policy_pred(logits, fired)
    NON = config.LABEL_TO_IDX["non_stereotype"]

    preds = {"Path-B (rules fired)": fired.any(1).astype(int),
             "Path-A (binary)": (logits.argmax(1) != NON).astype(int),
             "NeSy-MIS (binary)": (nesymis != NON).astype(int)}

    from sklearn.metrics import f1_score, precision_score, recall_score
    md = [f"## Table 19 - External transfer: {name} "
          f"(n={len(y_bin)}, {int(y_bin.sum())} misogynous; deployed system, no re-induction)\n",
          "| Path | Acc | Macro-F1 | F1 (misog.) | Precision | Recall |",
          "|---|---|---|---|---|---|"]
    for pn, p in preds.items():
        md.append(f"| {pn} | {float((p == y_bin).mean()):.3f} | "
                  f"{f1_score(y_bin, p, average='macro', zero_division=0):.3f} | "
                  f"{f1_score(y_bin, p, zero_division=0):.3f} | "
                  f"{precision_score(y_bin, p, zero_division=0):.3f} | "
                  f"{recall_score(y_bin, p, zero_division=0):.3f} |")

    if stereo is not None:
        st = (y_bin == 1) & (stereo == 1)
        ot = (y_bin == 1) & (stereo == 0)
        md += ["\n**Recall by misogyny type** (our taxonomy covers role stereotypes only):\n",
               f"| Path | Stereotype misogyny (n={int(st.sum())}) | "
               f"Other misogyny (n={int(ot.sum())}) |", "|---|---|---|"]
        for pn, p in preds.items():
            md.append(f"| {pn} | {float(p[st].mean()):.3f} | {float(p[ot].mean()):.3f} |")

    md.append("\n_Binary task: any stereotype class => misogynous. The external taxonomy "
              "differs from ours (general misogyny vs four role stereotypes), so recall "
              "against non-stereotype misogyny forms is expected to be partial._")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    print(generate())
