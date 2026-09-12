"""Data tier: checks that need only the dataset (and, for D5, embeddings).

Every function returns {"check", "verdict", ...statistics}. Verdicts: "FLAG" (the problem the
check targets is present), "ok", "note" (reported, no decision rule applies). Thresholds are
defaults, exposed as arguments; the statistics are always returned so readers can apply their own.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from inputaudit.probes import (SEED, bootstrap_lift, cramers_v, macro_f1, probe_f1, seed_list,
                               shape_auc, shape_model)
from inputaudit.shape import shape_features

FLAG, OK, NOTE = "FLAG", "ok", "note"

# Screening defaults. Detection thresholds (D2's enrichment test, D3's lift, D6's excess) are the
# smallest values whose false-flag rate on planted controls and clean tasks is at most 5%
# (experiments/threshold_calibration.py); materiality thresholds (D2's support and precision, D3's share,
# D4's AUC and association) are stated choices. Every statistic is returned beside its verdict.
D3_NOTABLE = (0.15, 0.75)          # lift, share   (calibrated lift; was 0.20 before calibration)
D3_SEVERE = (0.30, 0.85)
D3_CLEAN_LIFT = 0.10
D6_FLAG = 0.05                     # excess leaky share (calibrated; was 0.10)


def _masks(split):
    sp = np.asarray(split).astype(str)
    return sp == "train", sp == "val", sp == "test"


def _worst(verdicts):
    return FLAG if FLAG in verdicts else (OK if OK in verdicts else NOTE)


# --------------------------------------------------------------------------- #
# D1 -- provenance: is each text field produced the same way for every source?
# --------------------------------------------------------------------------- #
def d1_provenance(df: pd.DataFrame, text_fields, source_col=None, declaration: dict | None = None):
    """`declaration`: {field: {source: "how the field was produced"}}. A field whose declared
    production differs across sources is FLAGGED: a model can learn the difference instead of the
    task, and a comparison across sources compares different inputs. The per-source profile
    (empty rate, length, upper-case ratio) is reported for every field either way."""
    groups = df[source_col].astype(str) if source_col else pd.Series("all", index=df.index)
    profile = {}
    for f in text_fields:
        X = shape_features(df[f].tolist())
        profile[f] = {g: {"n": int(m.sum()), "empty": float((X[m, 0] == 0).mean()),
                          "mean_chars": float(X[m, 0].mean()), "upper_ratio": float(X[m, 4].mean())}
                      for g in sorted(groups.unique()) for m in [(groups == g).to_numpy()]}
    fields = {}
    for f in text_fields:
        decl = (declaration or {}).get(f)
        if not decl:
            fields[f] = {"verdict": NOTE, "why": "provenance not declared"}
            continue
        methods = sorted(set(decl.values()))
        fields[f] = {"declared": decl, "verdict": FLAG if len(methods) > 1 else OK,
                     "why": ("produced differently across sources: " + " | ".join(methods))
                     if len(methods) > 1 else "one production method for every source"}
    return {"check": "D1 provenance", "verdict": _worst([v["verdict"] for v in fields.values()]),
            "fields": fields, "profile": profile}


# --------------------------------------------------------------------------- #
# D2 -- field relations: does a relation between text fields reveal the label?
# --------------------------------------------------------------------------- #
def d2_field_relations(df: pd.DataFrame, text_fields, label_col, min_support=20, min_precision=0.95,
                       max_p=1e-3):
    """Relations between every pair of text fields (equal, one contains the other, both empty),
    and each field being empty. A relation that holds on >= `min_support` rows, picks out one
    class with precision >= `min_precision`, AND is enriched for that class beyond chance
    (binomial tail probability under the class's base rate <= `max_p`, Bonferroni-corrected over
    the relations tested) is FLAGGED: it is a label shortcut that no model should be given. The
    enrichment test matters on imbalanced labels: with a 97% majority class, a relation unrelated
    to the label reaches precision 0.95 by chance most of the time."""
    from scipy.stats import binom
    y = df[label_col].astype(str).to_numpy()
    cls, cnt = np.unique(y, return_counts=True)
    size = dict(zip(cls, cnt))
    T = {f: df[f].fillna("").astype(str).str.strip() for f in text_fields}
    rel = {}
    for f in text_fields:
        rel[f"{f} is empty"] = (T[f] == "").to_numpy()
    for a, b in itertools.combinations(text_fields, 2):
        ea, eb = T[a] == "", T[b] == ""
        rel[f"{a} == {b}"] = ((T[a] == T[b]) & ~ea).to_numpy()
        rel[f"{a} inside {b}"] = np.array([(x != "" and x != z and x in z)
                                           for x, z in zip(T[a], T[b])])
        rel[f"{b} inside {a}"] = np.array([(z != "" and x != z and z in x)
                                           for x, z in zip(T[a], T[b])])
        rel[f"{a} and {b} both empty"] = (ea & eb).to_numpy()
    out = {}
    n_rel = max(1, sum(int(m.sum()) > 0 for m in rel.values()))
    for name, m in rel.items():
        n = int(m.sum())
        if n == 0:
            continue
        c, k = np.unique(y[m], return_counts=True)
        top = c[k.argmax()]
        prec, rec = float(k.max() / n), float(k.max() / size[top])
        base = float(size[top] / len(y))
        p = float(binom.sf(int(k.max()) - 1, n, base))    # P(>= this many of `top` by chance)
        flag = n >= min_support and prec >= min_precision and p <= max_p / n_rel
        out[name] = {"support": n, "top_class": str(top), "precision": prec, "recall": rec,
                     "base_rate": base, "p_value": p, "class_counts": {str(a): int(b) for a, b in zip(c, k)},
                     "verdict": FLAG if flag else OK}
    return {"check": "D2 field relations", "verdict": _worst([r["verdict"] for r in out.values()]),
            "thresholds": {"min_support": min_support, "min_precision": min_precision,
                           "max_p": max_p, "relations_tested": n_rel},
            "relations": out}


# --------------------------------------------------------------------------- #
# D3 -- content-free surface screen
# --------------------------------------------------------------------------- #
def d6_duplicates(texts, split, emb=None, y=None, cos_min=0.90, flag_share=D6_FLAG, n_null=20,
                  seed=SEED):
    """Near-duplicate leakage. A test item duplicates train+val when its normalised text
    (lower-case, alphanumerics only) occurs there, or -- if `emb` is given -- its embedding has
    cosine >= `cos_min` to a train+val item (non-empty texts only). With labels `y`, a duplicate is
    LEAKY when lookup answers it: the majority label of its exact matches (or, with none, of its
    nearest train+val neighbour) is its own label. Duplicates with other labels are contrastive
    (some benchmarks build them on purpose) and do not leak. Lookup also agrees by chance, the more
    so the fewer the classes, so the statistic is the leaky share in EXCESS of its null: the same
    lookup with the train+val labels permuted (`n_null` times). FLAGGED when the excess (or,
    without labels, the duplicate share) is >= `flag_share`; `novel_test_mask` (aligned with the
    test rows) marks the duplicate-free slice."""
    import re
    from collections import defaultdict
    tr, va, te = _masks(split)
    fit = tr | va
    norm = [re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip() for t in texts]
    te_i, fit_i = np.flatnonzero(te), np.flatnonzero(fit)
    groups = defaultdict(list)
    for i in fit_i:
        if norm[i]:
            groups[norm[i]].append(i)
    exact = np.array([bool(norm[i]) and norm[i] in groups for i in te_i])
    dup = exact.copy()
    near = nn = None
    if emb is not None:
        E = np.asarray(emb, np.float32)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        ref = E[fit_i]
        best, arg = np.empty(len(te_i), np.float32), np.empty(len(te_i), np.int64)
        for i in range(0, len(te_i), 1024):          # one block of similarities in memory at a time
            s = E[te_i[i:i + 1024]] @ ref.T
            best[i:i + 1024], arg[i:i + 1024] = s.max(1), s.argmax(1)
        nn = fit_i[arg]
        near = (best >= cos_min) & np.array([bool(norm[i]) for i in te_i])   # empty texts are not duplicates
        dup |= near
    out = {"check": "D6 near-duplicate leakage", "dup_share": float(dup.mean()),
           "exact_text_share": float(exact.mean()),
           "embedding_near_share": None if near is None else float(near.mean()),
           "leaky_share": None, "leaky_null": None, "leaky_excess": None,
           "n_test": int(len(te_i)), "novel_test_mask": ~dup,
           "thresholds": {"cos_min": cos_min, "flag_share": flag_share, "n_null": n_null}}
    stat = out["dup_share"]
    if y is not None:
        y = np.asarray(y).astype(int)
        k = int(y.max()) + 1
        members = [np.asarray(groups[norm[i]]) if exact[j] else None for j, i in enumerate(te_i)]

        def lookup(yf):                            # lookup's answer per test row, -1 = no duplicate
            ans = np.full(len(te_i), -1)
            for j in np.flatnonzero(dup):
                ans[j] = (np.bincount(yf[members[j]], minlength=k).argmax() if exact[j]
                          else yf[nn[j]])
            return ans

        yt = y[te_i]
        present = np.unique(yt)

        def rates(ans):                            # (overall, class-balanced) share lookup answers
            ok = ans == yt
            return float(ok.mean()), float(np.mean([ok[yt == c].mean() for c in present]))

        leaky, leaky_bal = rates(lookup(y))
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_null):
            yp = y.copy()
            yp[fit_i] = rng.permutation(y[fit_i])
            null.append(rates(lookup(yp)))
        null = np.asarray(null)
        out.update(leaky_share=leaky, leaky_null=float(null[:, 0].mean()),
                   leaky_excess=leaky - float(null[:, 0].mean()),
                   leaky_share_balanced=leaky_bal, leaky_null_balanced=float(null[:, 1].mean()),
                   leaky_excess_balanced=leaky_bal - float(null[:, 1].mean()))
        stat = out["leaky_excess"]
    out["verdict"] = FLAG if stat >= flag_share else OK
    return out


def d3_surface(texts, y, split, k, text_ref: float | None = None, seed=SEED,
               severe=D3_SEVERE, notable=D3_NOTABLE, clean_lift=D3_CLEAN_LIFT, test_keep=None,
               boot=2000):
    """Macro-F1 of a model that sees only text shape (no vocabulary), against the majority class
    (lift) and against a text-embedding probe (share = shape / text probe). Two statistics because
    either alone misleads: a high share on a tiny lift only means the text probe was weak. The shape
    model is fitted on the training rows and early-stopped on the validation rows; `boot` test-row
    resamples give 95% intervals for lift and share (0 skips them).
    severe: lift and share at or above `severe`; notable: at or above `notable`; clean: lift below
    `clean_lift`. Severe and notable are FLAGGED."""
    y = np.asarray(y).astype(int)
    tr, va, te = _masks(split)
    fit = tr | va
    if test_keep is not None:                    # e.g. D6's novel_test_mask: score a slice only
        te = te.copy()
        te[np.flatnonzero(te)[~np.asarray(test_keep)]] = False
    X = shape_features(texts)
    maj_class = int(np.bincount(y[fit], minlength=k).argmax())
    maj = macro_f1(y[te], np.full(int(te.sum()), maj_class), k)
    model = shape_model(X, y, tr, va, seed)
    pred = model.predict(X[te])
    shp = macro_f1(y[te], pred, k)
    lift = shp - maj
    share = shp / text_ref if text_ref else None
    ci = bootstrap_lift(y[te], pred, maj_class, k, text_ref, B=boot) if boot else {}
    if share is not None and lift >= severe[0] and share >= severe[1]:
        level, verdict = "severe", FLAG
    elif share is not None and lift >= notable[0] and share >= notable[1]:
        level, verdict = "notable", FLAG
    elif lift < clean_lift:
        level, verdict = "clean", OK
    else:
        level, verdict = "", NOTE
    return {"check": "D3 surface screen", "verdict": verdict, "level": level, "majority_f1": maj,
            "shape_f1": shp, "lift": lift, "lift_ci": ci.get("lift_ci"), "text_ref_f1": text_ref,
            "share": share, "share_ci": ci.get("share_ci"), "n_iter": int(model.n_iter_),
            "n_test": int(te.sum())}


# --------------------------------------------------------------------------- #
# D4 -- source leak
# --------------------------------------------------------------------------- #
def d4_source(texts, source, y, split, seed=SEED, auc_min=0.80, assoc_min=0.50):
    """Can text shape identify the data source, and does the source predict the label? FLAGGED
    when shape recovers the source with AUC >= `auc_min` AND the source-label association
    (Cramer's V) is >= `assoc_min`: then a model can learn the source instead of the task."""
    src = np.asarray(source).astype(str)
    if len(np.unique(src)) < 2:
        return {"check": "D4 source leak", "verdict": NOTE, "why": "single source"}
    tr, va, te = _masks(split)
    fit = tr | va
    codes = np.unique(src, return_inverse=True)[1]
    auc = shape_auc(shape_features(texts), codes, fit, te, seed)
    v = cramers_v(src, np.asarray(y))
    return {"check": "D4 source leak", "verdict": FLAG if (auc >= auc_min and v >= assoc_min) else OK,
            "source_auc_from_shape": auc, "source_label_cramers_v": v,
            "thresholds": {"auc_min": auc_min, "assoc_min": assoc_min},
            "sources": {s: int((src == s).sum()) for s in np.unique(src)}}


# --------------------------------------------------------------------------- #
# D5 -- modality check under two encoders
# --------------------------------------------------------------------------- #
def d5_modality(encoders: dict, y, split, k, seeds=3, tol=0.01, determines=0.90, backend="auto"):
    """`encoders`: {name: {"image": X_img, "text": X_txt}}, weakest first, strongest last. The
    same probe on image, text and both, per encoder. Two rules:
      * a modality is reported as adding little only if adding it gains < `tol` macro-F1 under the
        STRONGEST encoder; a weak encoder's verdict alone is 'encoder-limited', not a finding;
      * FLAG when one modality alone reaches `determines` macro-F1 under any encoder: the label is
        nearly determined by that input, so check where it comes from (D1).
    """
    y = np.asarray(y).astype(int)
    tr, va, te = _masks(split)
    sl = seed_list(seeds)
    scores = {}
    for name, v in encoders.items():
        views = {"image": v["image"], "text": v["text"],
                 "both": np.concatenate([v["image"], v["text"]], 1)}
        scores[name] = {vn: probe_f1(X, y, tr, va, te, k, sl, backend) for vn, X in views.items()}
    out = d5_decide(scores, tol, determines)
    out["seeds"] = sl
    return out


def d5_decide(scores: dict, tol=0.01, determines=0.90):
    """D5's decision rules on probe scores {encoder: {"image"|"text"|"both": [macro-F1 per seed]}},
    encoders weakest first -- so saved probe results can be audited without re-running them."""
    res = {}
    for name, sc in scores.items():
        m = {vn: float(np.mean(s)) for vn, s in sc.items()}
        res[name] = {"macro_f1": m, "sd": {vn: float(np.std(s)) for vn, s in sc.items()},
                     "value_of_image": m["both"] - m["text"], "value_of_text": m["both"] - m["image"]}
    names = list(scores)
    strong, weak = res[names[-1]], res[names[0]]
    claims = {}
    for mod, key in (("image", "value_of_image"), ("text", "value_of_text")):
        if strong[key] < tol:
            claims[mod] = f"adds < {tol} under the strongest encoder ({names[-1]})"
        elif weak[key] < tol:
            claims[mod] = (f"adds < {tol} under {names[0]} but {strong[key]:+.3f} under {names[-1]}: "
                           "encoder-limited, do not claim it is uninformative")
        else:
            claims[mod] = "contributes under every encoder"
    top = max((r["macro_f1"][vn], n, vn) for n, r in res.items() for vn in ("image", "text"))
    verdict = FLAG if top[0] >= determines else OK
    return {"check": "D5 modality check", "verdict": verdict, "encoders": res, "claims": claims,
            "max_single_modality": {"macro_f1": top[0], "encoder": top[1], "modality": top[2]},
            "thresholds": {"tol": tol, "determines": determines}}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ensure_split(df, split_col=None, label_col=None, seed=SEED):
    """The manifest's split column, or a stratified 70/15/15 split if there is none."""
    if split_col and split_col in df:
        return df[split_col].astype(str).to_numpy()
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    y = df[label_col].astype(str).to_numpy()
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    sp = np.empty(len(df), object)
    sp[tr], sp[va], sp[te] = "train", "val", "test"
    return sp


def run_data_tier(df, label_col, text_fields, split_col=None, source_col=None, provenance=None,
                  embeddings=None, seeds=3, backend="auto"):
    """All data-tier checks. `embeddings`: {encoder: {"image": X, "text:<field>": X}}, weakest
    first; text embeddings of a field serve as D3's text reference and, for the first field,
    as D5's text view."""
    split = ensure_split(df, split_col, label_col)
    classes = sorted(df[label_col].astype(str).unique())
    y = df[label_col].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    k = len(classes)
    tr, va, te = _masks(split)
    out = {"meta": {"n": len(df), "classes": classes, "n_test": int(te.sum()),
                    "text_fields": list(text_fields), "source": source_col}}
    out["D1"] = d1_provenance(df, text_fields, source_col, provenance)
    out["D2"] = d2_field_relations(df, text_fields, label_col) if len(text_fields) >= 1 else None
    first_enc = next(iter(embeddings)) if embeddings else None
    out["D3"] = {}
    for f in text_fields:
        ref = None
        if first_enc and f"text:{f}" in embeddings[first_enc]:
            ref = float(np.mean(probe_f1(embeddings[first_enc][f"text:{f}"], y, tr, va, te, k,
                                         seed_list(seeds), backend)))
        out["D3"][f] = d3_surface(df[f].tolist(), y, split, k, ref)
    out["D4"] = ({f: d4_source(df[f].tolist(), df[source_col], y, split) for f in text_fields}
                 if source_col else None)
    e0 = embeddings[first_enc].get(f"text:{text_fields[0]}") if first_enc else None
    out["D6"] = d6_duplicates(df[text_fields[0]].tolist(), split, e0, y)
    if out["D6"]["verdict"] == FLAG:             # re-screen the duplicate-free slice
        for f in text_fields:
            out["D3"][f]["novel_slice"] = d3_surface(df[f].tolist(), y, split, k,
                                                     test_keep=out["D6"]["novel_test_mask"])
    f0 = text_fields[0]
    if embeddings and all("image" in e and f"text:{f0}" in e for e in embeddings.values()):
        out["D5"] = d5_modality({n: {"image": e["image"], "text": e[f"text:{f0}"]}
                                 for n, e in embeddings.items()}, y, split, k, seeds, backend=backend)
    return out
