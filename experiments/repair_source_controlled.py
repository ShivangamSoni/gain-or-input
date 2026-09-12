"""The repaired system on SOURCE-CONTROLLED data.

In the case-study corpus the source decides stereotype vs non-stereotype (Cramer's V = 1), so
its 5-class numbers cannot show classification quality. Pre-registered (fixed before
any run): the deployed system -- concept_routed defaults (balanced scorecard, CV operating
point, rule B, acc_tol 0.01), re-induced on each setting's own training data, no setting-specific
tuning -- on

  WBMS-4          the four stereotype classes, all from WBMS (one source, V = 0 by construction);
                  the case-study splits restricted to WBMS rows; uniform OCR text
  MAMI misogyny   MAMI (one collection), binary; official train with a stratified 15% validation
                  split carved from it (seed 42), official test; MAMI's own transcriptions
  MAMI stereotype the same memes, stereotype vs not (MAMI's category label)

Reported per setting: neural vs concept-routed macro-F1 and accuracy with 95% paired bootstrap
intervals (11 seeds), the duplicate-free slice (D6: text or embedding cosine >= 0.90 to a
training item), concept share, decisions the concepts decide (S4), deletion / sufficiency (S5),
and the setting's data-tier verdicts (D2 where two text fields exist, D3 shape lift, D6). No
parity criterion; the price is whatever it is.

  python experiments/repair_source_controlled.py [--settings wbms4,mami,mami-stereo]
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import common as C
from repair_core import faithfulness, pooled_pred
from nesymis import config
from nesymis import concept_routed as evp
from nesymis.fusion import concept_pool as cp
from inputaudit.data_tier import d2_field_relations, d3_surface, d6_duplicates
from inputaudit.probes import paired_bootstrap, probe_f1, seed_list

OUT = config.ARTIFACTS_DIR / "repair_source_controlled.json"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _d3(texts, u, y, sp, k):
    """D3 as the tool runs it: shape-only lift, and its share of a text-embedding probe (3 seeds)."""
    tr, va, te = (np.asarray(sp) == s for s in ("train", "val", "test"))
    ref = float(np.mean(probe_f1(u, y, tr, va, te, k, seed_list(3))))
    return d3_surface(list(texts), y, sp, k, ref)


def wbms4():
    d0 = evp.load_data()
    m = d0.df["source"].str.startswith("wbms").to_numpy()
    classes = list(config.STEREO_CLASSES)
    df = d0.df[m].reset_index(drop=True)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    u_cap = np.load(config.TEXT_CAPTION_EMB_PATH)[m]
    d = SimpleNamespace(df=df, v=d0.v[m], u=d0.u[m], y=y, tr=d0.tr[m], va=d0.va[m], te=d0.te[m],
                        text=d0.text[m].reset_index(drop=True), classes=classes,
                        mine_classes=classes, text_channel="OCR of the image (uniform-OCR protocol)")
    sp = np.where(d.tr, "train", np.where(d.va, "val", "test"))
    dup = d6_duplicates(df["text_caption"].tolist(), sp, u_cap, y)          # the paper's novel-slice rule
    data_tier = {"D2": d2_field_relations(df, ["text_ocr", "text_caption"], "label"),
                 "D3": _d3(d.text.tolist(), d.u, y, sp, len(classes)),
                 "D6": {k: v for k, v in dup.items() if k != "novel_test_mask"}}
    return d, dup["novel_test_mask"], data_tier


def _mami_rows():
    import external_transfer as t19
    import mami_full_train as t20
    tr = pd.read_csv(os.path.join(t20.MAMI_DIR, "TRAINING", "training.csv"), sep="\t", encoding="utf-8-sig")
    paths = [os.path.join(t20.MAMI_DIR, "TRAINING", fn) for fn in tr["file_name"]]
    ok = np.array([os.path.isfile(p) for p in paths])
    tr = tr.loc[ok].reset_index(drop=True)
    tr_t, tr_p = tr["Text Transcription"].fillna("").astype(str).tolist(), [p for p, k in zip(paths, ok) if k]
    _, te_t, te_p, te_mis, te_ster = t19._load_mami()
    texts, pths = tr_t + te_t, tr_p + te_p
    v, u = t20._embeddings(pths, texts)                                    # the cached CLIP B/32 rows
    mis = np.concatenate([tr["misogynous"].to_numpy(int), te_mis])
    ster = np.concatenate([tr["stereotype"].to_numpy(int), te_ster])
    n_tr = len(tr_t)
    return texts, v, u, mis, ster, n_tr


def mami(target="misogynous"):
    texts, v, u, mis, ster, n_tr = _mami_rows()
    ylab = mis if target == "misogynous" else ster
    classes = ["not_misogynous", "misogynous"] if target == "misogynous" else ["not_stereotype", "stereotype"]
    N = len(texts)
    tri, vai = train_test_split(np.arange(n_tr), test_size=0.15, stratify=ylab[:n_tr],
                                random_state=config.SEED)
    tr, va = np.isin(np.arange(N), tri), np.isin(np.arange(N), vai)
    te = np.arange(N) >= n_tr
    df = pd.DataFrame({"label": [classes[i] for i in ylab], "text": texts})
    d = SimpleNamespace(df=df, v=v.astype(np.float32), u=u.astype(np.float32), y=ylab.astype(int),
                        tr=tr, va=va, te=te, text=df["text"], classes=classes, mine_classes=classes,
                        text_channel="MAMI transcription (both paths)")
    sp = np.where(tr, "train", np.where(va, "val", "test"))
    dup = d6_duplicates(texts, sp, u, ylab)
    data_tier = {"D3": _d3(texts, u, ylab, sp, 2), "D6": {k: x for k, x in dup.items() if k != "novel_test_mask"}}
    return d, dup["novel_test_mask"], data_tier


SETTINGS = {"wbms4": ("WBMS-4 (one source)", wbms4),
            "mami": ("MAMI misogyny", lambda: mami("misogynous")),
            "mami-stereo": ("MAMI stereotype", lambda: mami("stereotype"))}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def f1k(y, p, k):
    return float(f1_score(y, p, average="macro", labels=list(range(k)), zero_division=0))


def evaluate(name, d, novel, rng, groups=None):
    """11-seed comparison. `groups` (a duplicate-group id per test row) adds a cluster bootstrap."""
    k = len(d.classes)
    te_i = np.flatnonzero(d.te)
    y = d.y[te_i]
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    print(f"[source-controlled] {name}: {len(d.y)} rows, train {int(d.tr.sum())} val {int(d.va.sum())} test "
          f"{len(te_i)} (duplicate-free {int(novel.sum())}); classes {d.classes}", flush=True)
    sym = evp.build_symbolic(d)
    per, Pp, Pn = [], [], []
    for s in (int(x) for x in C.SEEDS11):
        t0 = time.time()
        sysm, logits = evp.fit(seed=s, data=d, sym=sym)
        out = sysm.decide(vu[te_i], d.u[te_i], d.v[te_i], index=d.df.index[te_i],
                          neural_logits=logits[te_i])
        pred, pn = out["pred"], logits[te_i].argmax(1)
        f = faithfulness(sysm.card, out["Z"], out["ln"], sysm.w, out, rng)
        per.append({"seed": s, "w": sysm.w, "f1_neural": f1k(y, pn, k), "f1_pooled": f1k(y, pred, k),
                    "acc_neural": float((pn == y).mean()), "acc_pooled": float((pred == y).mean()),
                    "novel_f1_neural": f1k(y[novel], pn[novel], k),
                    "novel_f1_pooled": f1k(y[novel], pred[novel], k),
                    "concept_share": float(out["concept_share"].mean()),
                    "concepts_decide": 1.0 - f["keep_none_same"],
                    "del_top3": f[3]["delete_top_changes"], "del_rand3": f[3]["delete_random_changes"],
                    "keep_top3": f[3]["keep_top_same"], "keep_rand3": f[3]["keep_random_same"],
                    "f1_class_neural": f1_score(y, pn, average=None, labels=list(range(k)), zero_division=0).tolist(),
                    "f1_class_pooled": f1_score(y, pred, average=None, labels=list(range(k)), zero_division=0).tolist()})
        Pp.append(pred)
        Pn.append(pn)
        r = per[-1]
        print(f"[source-controlled] {name} seed {s}: {time.time() - t0:.0f}s w={r['w']} F1 {r['f1_neural']:.3f}->"
              f"{r['f1_pooled']:.3f} acc {r['acc_neural']:.3f}->{r['acc_pooled']:.3f} | decide "
              f"{r['concepts_decide']:.3f} del {r['del_top3']:.3f}/{r['del_rand3']:.3f}", flush=True)
    Pp, Pn = np.array(Pp), np.array(Pn)
    boot = {"all": paired_bootstrap(y, Pp, Pn, k), "duplicate_free": paired_bootstrap(y[novel], Pp[:, novel], Pn[:, novel], k)}
    if groups is not None:
        boot["cluster"] = paired_bootstrap(y, Pp, Pn, k, groups=groups)
    return {"classes": d.classes, "n_test": int(len(te_i)), "n_duplicate_free": int(novel.sum()),
            "per_seed": per, "bootstrap": boot}


def report(res):
    print("\n## The repaired system on source-controlled data (11 seeds; 95% paired bootstrap CIs)\n")
    print("| setting | neural F1 / acc | concept-routed F1 / acc | d F1 [CI] | d acc [CI] | d F1 dup-free [CI] | w | concepts decide | delete top-3 / random-3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    ci = lambda e: f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    for name, r in res.items():
        g = lambda k: np.array([p[k] for p in r["per_seed"]])
        ws = list(g("w"))
        wtxt = ", ".join(f"{w:.1f}x{ws.count(w)}" for w in sorted(set(ws)))
        b = r["bootstrap"]
        print(f"| {name} | {g('f1_neural').mean():.3f} / {g('acc_neural').mean():.3f} | "
              f"{g('f1_pooled').mean():.3f} / {g('acc_pooled').mean():.3f} | {ci(b['all']['d_macro_f1'])} | "
              f"{ci(b['all']['d_acc'])} | {ci(b['duplicate_free']['d_macro_f1'])} | {wtxt} | "
              f"{g('concepts_decide').mean():.3f} | {g('del_top3').mean():.3f} / {g('del_rand3').mean():.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="wbms4,mami,mami-stereo")
    ap.add_argument("--data-tier-only", action="store_true",
                    help="recompute each saved setting's data-tier verdicts; models are not re-run")
    a = ap.parse_args()
    keys = a.settings.split(",")
    rng = np.random.default_rng(config.SEED)
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    for key in keys:
        name, make = SETTINGS[key]
        d, novel, data_tier = make()
        if a.data_tier_only:
            if name in res:
                assert int(novel.sum()) == res[name]["n_duplicate_free"], "duplicate-free slice changed"
                res[name]["data_tier"] = data_tier
                json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
                print(f"[source-controlled] {name}: data tier updated", flush=True)
            continue
        print(f"[source-controlled] {name} data tier: D3 lift {data_tier['D3']['lift']:+.3f} | D6 leaky "
              f"{data_tier['D6']['leaky_share']:.3f}, excess {data_tier['D6']['leaky_excess']:+.3f} "
              f"({data_tier['D6']['verdict']})", flush=True)
        r = evaluate(name, d, novel, rng)
        r["data_tier"] = data_tier
        res[name] = r
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    report(res)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
