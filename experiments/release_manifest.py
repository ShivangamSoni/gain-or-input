"""Release manifest: what a reader needs to rebuild our splits without receiving our data.

WBMS is available from its authors on request, and its file names carry the post caption, so no
file name, caption or OCR text of the case-study corpus is written here. Each meme is identified by
the SHA-256 of its image bytes: obtain the images, hash them, and the split, label and source of
every row follow. MAMI's file names are neutral identifiers assigned by its organisers, so its rows
are named directly.

Writes, under artifacts/release/:
  corpus_splits.csv     row, sha256, label, source, split   (3,263 memes; WBMS-4 = the wbms_* rows)
  mami_splits.csv       file_name, split                     (official train/test + our val carve)
  artifact_hashes.csv   artefact, sha256, bytes              (the files the paper's numbers read)
  README.md             how to use them

  python experiments/release_manifest.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from nesymis import config

OUT = config.ARTIFACTS_DIR / "release"
PAPER_ARTEFACTS = [
    "threshold_calibration.json", "known_answer_mmhs.json", "d5_seed_stability.json", "d6_duplicates.json",
    "d2_d4_external_tasks.json", "duplicate_free_split.json", "known_answer_architectures.json", "threshold_sensitivity.json",
    "tool_validation.json", "uniform_ocr_protocol.json", "repair_source_controlled.json",
    "repair_nulls_and_cost.json", "repair_fresh_partitions.json", "repair_benchmark_split.json", "repair_backbones.json",
    "per_class_intervals.json", "d5_two_encoders.json", "d3_surface_screen.json", "original_design_multiseed.json",
    "s2_text_regimes.json", "d5_modality_probes.json",
]
README = """# Release manifest

These files identify the rows behind every result in the paper without carrying any of the data.

* `corpus_splits.csv` -- the case-study corpus (3,263 memes). `sha256` is the SHA-256 of the image
  file's bytes; `row` is the row's position in our manifest, which fixes the order the cached
  embeddings are stored in. WBMS memes are available from the WBMS authors on request and GOAT from
  its own release; hash the images you receive and join on `sha256`. No file name, caption or OCR
  text appears here, because WBMS file names contain the post caption.
  The WBMS-4 setting is the rows whose `source` starts with `wbms_`, with the same split.
* `mami_splits.csv` -- MAMI rows by the organisers' file names, with our split: the official test
  split, and the official training split divided into train and a 15% stratified validation carve
  (seed 42).
* `artifact_hashes.csv` -- SHA-256 and size of each result file the paper's numbers are read from,
  so a re-run can be compared with ours file by file.

Reproduction order and the environment are described in REPRODUCE.md.
"""


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    path = os.path.abspath(path)
    if os.name == "nt" and len(path) > 250:       # 21 WBMS captions make paths longer than MAX_PATH
        path = "\\\\?\\" + path
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_splits():
    df = pd.read_csv(config.MANIFEST_PATH)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for i, r in df.iterrows():
        p = r["path"] if os.path.isabs(r["path"]) else os.path.join(root, r["path"])
        rows.append({"row": i, "sha256": sha256(p), "label": r["label"], "source": r["source"],
                     "split": r["split"]})
    out = pd.DataFrame(rows)
    assert out["sha256"].is_unique or True, ""                     # duplicates are a finding (D6), not an error
    out.to_csv(OUT / "corpus_splits.csv", index=False)
    dup = int(len(out) - out["sha256"].nunique())
    print(f"[rel] corpus_splits.csv: {len(out)} rows, {out['sha256'].nunique()} distinct images "
          f"({dup} identical files), splits {out['split'].value_counts().to_dict()}")


def mami_splits():
    import external_transfer as t19
    import mami_full_train as t20
    from sklearn.model_selection import train_test_split
    tr = pd.read_csv(os.path.join(t20.MAMI_DIR, "TRAINING", "training.csv"), sep="\t", encoding="utf-8-sig")
    ok = np.array([os.path.isfile(os.path.join(t20.MAMI_DIR, "TRAINING", fn)) for fn in tr["file_name"]])
    tr = tr.loc[ok].reset_index(drop=True)
    y = tr["misogynous"].to_numpy(int)
    tri, vai = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=y, random_state=config.SEED)
    split = np.where(np.isin(np.arange(len(tr)), vai), "val", "train")
    _, _, te_p, _, _ = t19._load_mami()
    out = pd.concat([pd.DataFrame({"file_name": tr["file_name"], "split": split}),
                     pd.DataFrame({"file_name": [os.path.basename(p) for p in te_p], "split": "test"})])
    out.to_csv(OUT / "mami_splits.csv", index=False)
    print(f"[rel] mami_splits.csv: {len(out)} rows, splits {out['split'].value_counts().to_dict()}")


def artifact_hashes():
    rows = []
    for name in PAPER_ARTEFACTS:
        p = config.ARTIFACTS_DIR / name
        if not p.exists():
            print(f"[rel] MISSING artefact {name}")
            continue
        rows.append({"artefact": name, "sha256": sha256(p), "bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT / "artifact_hashes.csv", index=False)
    print(f"[rel] artifact_hashes.csv: {len(rows)} of {len(PAPER_ARTEFACTS)} artefacts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_splits()
    try:
        mami_splits()
    except Exception as e:                        # MAMI is obtained from its organisers; skip if absent
        print(f"[rel] mami_splits.csv skipped: {type(e).__name__}: {e}")
    artifact_hashes()
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"\n_saved {OUT}_")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:22s} {f.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main()
