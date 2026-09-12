"""Do meme benchmarks leak their labels through CONTENT-FREE surface statistics?

Our own corpus turned out to carry a source-group signal in text *length* alone
(AUC 0.746, unchanged by fixing the OCR pipeline -- see experiments/uniform_ocr_protocol.py),
because it combines text-heavy GOAT political memes with picture-driven WBMS stereotype
memes. That is a dataset-composition defect, not an OCR defect, and it generalises: any
benchmark assembled from heterogeneous sources risks surface statistics that identify the
source, and therefore the label.

This script runs that check across every corpus in the suite. The features read **no
words** -- only how the text is shaped:

    n_chars, n_words, mean/max word length, uppercase ratio, digit ratio,
    punctuation ratio, non-alphanumeric ratio, type-token ratio, is-empty

A gradient-boosted tree on those nine numbers is then compared against the CLIP
text-embedding probe from experiments/d5_modality_probes.py, which does read the words. The
statistic that matters is the **ratio**: if content-free shape recovers most of what the
text embedding recovers, the benchmark has a shortcut that is not the intended task.

A strong learner is used deliberately: the check should fail loudly when a shortcut exists,
so a clean verdict means something.

Note what this does *not* claim. Length can be legitimately informative -- longer text
carries more content. The finding is only interesting when surface shape approaches the
full-text ceiling, which is why the ratio, not the absolute score, is the headline.

  python experiments/d3_surface_screen.py --run all
  python experiments/d3_surface_screen.py --report
"""
import argparse
import json
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from nesymis import config

OUT = str(config.ARTIFACTS_DIR / "d3_surface_screen.json")
SEED = config.SEED
_PUNCT = re.compile(r"[^\w\s]")


def features(texts):
    """(N, 9) content-free surface features. Reads shape, never vocabulary."""
    rows = []
    for t in texts:
        s = "" if t is None else str(t)
        w = s.split()
        n = len(s)
        nw = len(w)
        lens = [len(x) for x in w] or [0]
        alnum = sum(ch.isalnum() for ch in s)
        rows.append([
            n,                                            # chars
            nw,                                           # words
            float(np.mean(lens)),                         # mean word length
            float(np.max(lens)),                          # max word length
            (sum(ch.isupper() for ch in s) / n) if n else 0.0,
            (sum(ch.isdigit() for ch in s) / n) if n else 0.0,
            (len(_PUNCT.findall(s)) / n) if n else 0.0,
            ((n - alnum) / n) if n else 0.0,
            (len(set(x.lower() for x in w)) / nw) if nw else 0.0,   # type-token ratio
        ])
    return np.asarray(rows, dtype=np.float64)


# The shape model and its statistics live in the released tool (inputaudit.data_tier.d3_surface):
# trained on the training split, early-stopped on the validation split, with bootstrap intervals.


# --------------------------------------------------------------------------- #
# Task providers: (texts, y, tr, va, te, k, label)
# --------------------------------------------------------------------------- #
def _ours(text_col, corrected, label):
    import pandas as pd
    if corrected:
        import uniform_ocr_protocol as cb
        df = pd.read_csv(cb.MANIFEST_C).fillna("")
    else:
        from nesymis.data import dataset as ds
        df = ds.load_manifest()
    y = df["label_idx"].to_numpy().astype(int)
    sp = df["split"].to_numpy()
    return dict(texts=df[text_col].astype(str).tolist(), y=y, tr=sp == "train",
                va=sp == "val", te=sp == "test", k=5, label=label,
                extra=df)


def _from_sig(key):
    """Reuse d5_two_encoders's providers -- they already return texts + labels."""
    import d5_two_encoders as ss
    _, prov = ss.TASKS[key]
    spec = prov()
    y = np.asarray(spec["y"])
    tr, va, te = ss._splits(spec, len(y))
    keep = y >= 0
    if not keep.all():
        return dict(texts=[t for t, m in zip(spec["texts"], keep) if m], y=y[keep],
                    tr=tr[keep], va=va[keep], te=te[keep], k=spec["k"],
                    label=spec["label"])
    return dict(texts=list(spec["texts"]), y=y, tr=tr, va=va, te=te, k=spec["k"],
                label=spec["label"])


TASKS = {
    "ours-ocr": lambda: _ours("text_ocr", False, "ours-5class (OCR, original)"),
    "ours-ocr-corrected": lambda: _ours("text_ocr", True, "ours-5class (OCR, corrected)"),
    "ours-caption": lambda: _ours("text_caption", False, "ours-5class (clean caption)"),
    "mami": lambda: _from_sig("mami"),
    "mmhs-bin": lambda: _from_sig("mmhs-bin"),
    "mmhs-6": lambda: _from_sig("mmhs-6"),
    "exist-bin": lambda: _from_sig("exist-bin"),
    "exist-6": lambda: _from_sig("exist-6"),
    "mmsd": lambda: _from_sig("mmsd"),
    "pridemm": lambda: _from_sig("pridemm"),
    "pridemm-tgt": lambda: _from_sig("pridemm-tgt"),
    "hateful": lambda: _from_sig("hateful"),
    "harmeme": lambda: _from_sig("harmeme"),
    "harmeme-3": lambda: _from_sig("harmeme-3"),
}

# text-embedding reference: which d5_modality_probes.json key each task compares to
TEXT_REF = {k: k for k in TASKS}
TEXT_REF["ours-ocr-corrected"] = "ours-ocr"


def _load():
    return json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}


def _save(d):
    cur = _load()
    cur.update(d)
    json.dump(cur, open(OUT, "w", encoding="utf-8"), indent=1)


def _text_ref(k):
    """The text-embedding probe D3's share is measured against (as in the paper's tables)."""
    if k == "ours-ocr-corrected":
        cb = json.load(open(config.ARTIFACTS_DIR / "uniform_ocr_protocol.json", encoding="utf-8"))
        return float(np.mean(cb["baselines"]["rows"]["official, text only: [u_ocr]"]["macro_f1"]))
    p = config.ARTIFACTS_DIR / "d5_modality_probes.json"
    ts = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    ref = ts.get(TEXT_REF.get(k, k))
    return float(np.mean(ref["views"]["text"]["macro_f1"])) if ref else None


def cmd_run(which):
    """D3 through the released tool (inputaudit.data_tier.d3_surface): shape model fitted on the
    training split and early-stopped on the validation split, with bootstrap intervals."""
    from inputaudit.data_tier import d3_surface
    keys = list(TASKS) if which == "all" else [x.strip() for x in which.split(",")]
    for k in keys:
        print(f"\n########## surface shortcut: {k} ##########", flush=True)
        try:
            s = TASKS[k]()
            X = features(s["texts"])
            y = np.asarray(s["y"]).astype(int)
            split = np.where(s["tr"], "train", np.where(s["va"], "val", np.where(s["te"], "test", "")))
            d3 = d3_surface(s["texts"], y, split, s["k"], _text_ref(k))
            mj = d3["majority_f1"]
            r = {"macro_f1": d3["shape_f1"]}
            rec = {"label": s["label"], "k": s["k"], "n": int(len(y)),
                   "n_test": int(s["te"].sum()), "surface": r, "majority_f1": mj,
                   "lift_ci": d3["lift_ci"], "share": d3["share"], "share_ci": d3["share_ci"],
                   "level": d3["level"], "n_iter": d3["n_iter"]}
            # our corpus additionally leaks the SOURCE group; quantify it here too
            if "extra" in s:
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.metrics import roc_auc_score
                df = s["extra"]
                ster = df["source"].str.startswith("wbms").to_numpy()
                m = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
                fit = s["tr"] | s["va"]
                m.fit(X[fit], ster[fit])
                rec["source_auc"] = float(roc_auc_score(
                    ster[s["te"]], m.predict_proba(X[s["te"]])[:, 1]))
            _save({k: rec})
            extra = f" | source AUC {rec['source_auc']:.3f}" if "source_auc" in rec else ""
            print(f"  [surf] {k}: macro-F1 {r['macro_f1']:.3f} "
                  f"(majority {mj:.3f}){extra}", flush=True)
        except Exception:
            print(f"  [surf] !! FAILED {k}:\n{traceback.format_exc()}", flush=True)
    print("\nSURFACE_DONE", flush=True)


def cmd_report():
    d = _load()
    if not d:
        raise SystemExit("nothing yet -- run --run all")
    ts = json.load(open(config.ARTIFACTS_DIR / "d5_modality_probes.json", encoding="utf-8")) \
        if (config.ARTIFACTS_DIR / "d5_modality_probes.json").exists() else {}
    print("## Content-free surface-statistic shortcut (9 shape features, no vocabulary)\n")
    print("`surface` reads only text shape -- length, casing, punctuation, type-token "
          "ratio. `text emb` is the CLIP text probe, which reads the words. **share** is "
          "surface / text emb: how much of the readable-text signal is reachable without "
          "reading anything.\n")
    print("Two statistics, because either alone misleads. **lift** = surface - majority: "
          "how much a content-free model beats guessing. **share** = surface / text emb: "
          "how much of the readable-text signal is shape. A high share with near-zero lift "
          "means the text embedding was weak, not that shape is doing work.\n")
    print("| Corpus / task | cls | n test | majority | surface | lift | text emb | share | "
          "flag |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for k, r in d.items():
        surf = r["surface"]["macro_f1"]
        ref = ts.get(TEXT_REF.get(k, k))
        temb = float(np.mean(ref["views"]["text"]["macro_f1"])) if ref else float("nan")
        share = surf / temb if temb and temb == temb else float("nan")
        lift = surf - r["majority_f1"]
        flag = {"severe": "**severe**"}.get(r.get("level", ""), r.get("level", ""))   # the tool's verdict
        rows.append((lift, r, surf, temb, share, flag))
    for lift, r, surf, temb, share, flag in sorted(rows, key=lambda t: -t[0]):
        print(f"| {r['label']} | {r['k']} | {r['n_test']} | {r['majority_f1']:.3f} | "
              f"**{surf:.3f}** | **{lift:+.3f}** | {temb:.3f} | {share:.2f} | {flag} |")
    src = {k: r["source_auc"] for k, r in d.items() if "source_auc" in r}
    if src:
        print("\n### Source-group recoverability from surface shape (our corpus only)\n")
        for k, v in src.items():
            print(f"* `{k}`: AUC **{v:.3f}**")
        print("\nOur corpus is the only multi-source one here, so this row has no "
              "counterpart elsewhere -- which is itself the point.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run", metavar="KEYS")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        cmd_run(a.run)
    elif a.report:
        cmd_report()
    else:
        print("\n".join(TASKS))


if __name__ == "__main__":
    main()
