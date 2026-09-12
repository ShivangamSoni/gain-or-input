"""Unified SEED-42 baseline harness for the deployed multimodal concept layer,
across our dataset and the three external benchmarks in every regime.

    python experiments/concept_scorecard_eval.py ours
    python experiments/concept_scorecard_eval.py mami   {zeroshot|binary}
    python experiments/concept_scorecard_eval.py mmhs   {zeroshot|binary|multiclass}
    python experiments/concept_scorecard_eval.py exist  {zeroshot|binary|multiclass}

Every setting uses nesymis.symbolic.concept_layer (build concepts -> induce DNF
rules over concepts+affect -> linear evidence decision layer). Zero-shot applies
OUR fitted concept bank + rules + classifier + policy unchanged; binary/multiclass
re-induce everything on the target's own training split (the affect head stays
our frozen head, since external sets have no affect labels). Seed 42 only.
Nothing runs on import; dispatch is via argv, so this file is compile-safe.

MAMI multiclass is intentionally omitted (Sub-task B is multi-LABEL -> ill-posed
single-label multiclass; MMHS/EXIST carry the 6-class tests).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import common as C
from nesymis import config
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import rule_induction

SEED = config.SEED


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
class RandomBank:
    """Null control for E1.4: identical to ConceptBank in every respect -- same K,
    the same gap correction (mu_t/mu_i), the same max-over-modalities cosine
    scoring, feeding the same DNF induction -- except the K directions are random
    unit vectors instead of mined semantic prototypes. Anything this bank achieves
    is attributable to the *machinery*, not to concept meaning.

    It is a drop-in for ConceptBank, so it also transfers zero-shot, which is what
    makes the E1.4 comparison possible.
    """

    def __init__(self, dim, mu_t, mu_i, k=cl.K_CONCEPTS, seed=SEED):
        rng = np.random.default_rng(seed)
        V = rng.standard_normal((k, dim)).astype(np.float32)
        self.centroids = V / np.linalg.norm(V, axis=1, keepdims=True)
        self.mu_t, self.mu_i = mu_t, mu_i
        self.names = [f"rnd{i}" for i in range(k)]

    score = cl.ConceptBank.score
    frame = cl.ConceptBank.frame


class UnsupervisedBank:
    """Middle control: concepts from data structure alone, with NO labels.

    ConceptBank mines its text anchors per class by smoothed log-odds, i.e. using
    labels; its image anchors are already unsupervised k-means. This bank replaces
    *only* the supervised text mining with k-means over caption embeddings and
    keeps everything else identical (image anchors, gap correction, K=30
    clustering, scoring). It therefore separates two explanations of the real
    bank's advantage over RandomBank:

      unsupervised ~ real   -> the gain comes from genuine structure in the data
      unsupervised ~ random -> the gain comes from label supervision, and the
                               "concepts" are discriminative directions found with
                               labels rather than natural semantic clusters
    """

    def __init__(self, u, v, trainval, k=cl.K_CONCEPTS, n_anchor=cl.N_IMG, seed=SEED):
        from sklearn.cluster import KMeans
        U, V = cl._unit(u.astype(np.float32)), cl._unit(v.astype(np.float32))
        n = min(n_anchor, int(trainval.sum()))
        tk = KMeans(n_clusters=n, random_state=seed, n_init=6).fit(U[trainval])
        ik = KMeans(n_clusters=n, random_state=seed, n_init=6).fit(V[trainval])
        Tvec = cl._unit(tk.cluster_centers_.astype(np.float32))
        Ivec = cl._unit(ik.cluster_centers_.astype(np.float32))
        self.mu_t, self.mu_i = Tvec.mean(0), Ivec.mean(0)
        A = np.concatenate([cl._unit(Tvec - self.mu_t), cl._unit(Ivec - self.mu_i)], 0)
        km = KMeans(n_clusters=min(k, len(A)), random_state=seed, n_init=10).fit(A)
        self.centroids = cl._unit(km.cluster_centers_.astype(np.float32))
        self.names = [f"unsup{i}" for i in range(len(self.centroids))]

    score = cl.ConceptBank.score
    frame = cl.ConceptBank.frame


def fit_system(v, u, vu, D, texts, labels, y, tr, va, trainval,
               classes, class_idx, n_classes=None, train_clf=None, bank=None):
    """Concept bank + rules over concepts + classifier + linear decision layer (seed 42).
    `classes` = rule-class NAME strings (keys of rule_sets); `class_idx` = their int labels.
    `bank` overrides concept construction (used by E1.4 to inject RandomBank)."""
    emb = SimpleNamespace(image=v, text_caption=u,
                          df=pd.DataFrame({"text_caption": texts, "label": labels}))
    if bank is None:
        bank = cl.build(emb, trainval, classes=classes)
    elif callable(bank):
        bank = bank(emb, trainval, classes)
    cont = bank.frame(u, v, index=np.arange(len(y)))
    rs, L, litnames = rule_induction.induce(cont, y, trainval, affect=None,   # affect-free
                                            classes=classes, save=False)
    R = len(classes)
    fired = np.zeros((len(y), R), np.float32)
    trig = np.zeros((len(y), R), np.float32)
    rvec = np.zeros(R, np.float32)
    for j, cls in enumerate(classes):
        f = np.zeros(len(y), bool)
        for clause in rs.get(cls, []):
            fj = np.all(L[:, clause["idx"]], axis=1)
            f |= fj
            trig[:, j] = np.where(fj, np.maximum(trig[:, j], clause["precision"]), trig[:, j])
        fired[:, j] = f.astype(np.float32)
        m = trainval & f
        rvec[j] = float((y[m] == class_idx[j]).mean()) if m.any() else 0.0
    pb = np.where(trig.max(1) > 0, np.array(class_idx)[trig.argmax(1)], 0)

    fit_fn = (lambda m_tr, m_va: train_clf(vu, D, y, m_tr, m_va)) if train_clf else None
    clf, logits = clf.crossfit_logits(vu, D, y, tr, va, fusion=config.NEURAL_FUSION,
                                     affect_dim=C.odesign.AFFECT_DIM, seed=SEED, fit_fn=fit_fn)
    state = C.rl.build_state(logits, fired, rvec, D)
    nc = n_classes or config.NUM_CLASSES
    ctor = lambda d: C.rl.LinearEvidencePolicy(d, n_classes=nc, rule_classes=class_idx)
    pol = C.rl.train_rlvr(logits, state, y, tr, va, seed=SEED, policy_ctor=ctor)[0]
    return dict(bank=bank, rs=rs, classes=classes, class_idx=class_idx, rvec=rvec,
                clf=clf, pol=pol, fired=fired, pb=pb, litnames=litnames)


def zeroshot(sysd, u, v, vu, D, non_idx=0):
    """Apply a fitted concept system to external embeddings; binary preds (!= default)."""
    cont = sysd["bank"].frame(u, v, index=np.arange(len(u)))   # affect-free: concept rules only
    rd = rule_induction.evaluate_features(sysd["rs"], cont, None, classes=sysd["classes"])
    fired = np.stack([rd[f"{cl}_fired"].to_numpy() for cl in sysd["classes"]], 1).astype(np.float32)
    logits = clf.predict_logits(sysd["clf"], vu, D)
    state = C.rl.build_state(logits, fired, sysd["rvec"], D)
    ne = C.rl.greedy_pred(sysd["pol"], logits, state)
    return {"Path-B (rules)": fired.any(1).astype(int),
            "Path-A": (logits.argmax(1) != non_idx).astype(int),
            "NeSy": (ne != non_idx).astype(int)}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _brow(nm, y, p):
    return (f"| {nm} | {float((p==y).mean()):.3f} | {f1_score(y,p,average='macro',zero_division=0):.3f} "
            f"| {f1_score(y,p,zero_division=0):.3f} | {precision_score(y,p,zero_division=0):.3f} "
            f"| {recall_score(y,p,zero_division=0):.3f} |")


def report_binary(title, preds, y):
    print(f"\n## {title}\n\n| Path | Acc | Macro-F1 | F1(pos) | Prec | Rec |\n|---|---|---|---|---|---|")
    for nm, p in preds.items():
        print(_brow(nm, y, p))


def report_multiclass(title, y_te, pb, pa, ne, names):
    labs = list(range(len(names)))
    print(f"\n## {title}\n\n| Path | Acc | Macro-F1 | " + " | ".join(f"F1-{n}" for n in names) + " |")
    print("|" + "---|" * (len(names) + 3))
    for nm, p in (("Path-B", pb), ("Path-A", pa), ("NeSy", ne)):
        per = f1_score(y_te, p, labels=labs, average=None, zero_division=0)
        print(f"| {nm} | {float((p==y_te).mean()):.3f} | "
              f"{f1_score(y_te,p,labels=labs,average='macro',zero_division=0):.3f} | "
              + " | ".join(f"{x:.3f}" for x in per) + " |")


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
_OURS = {}
def fit_ours():
    if _OURS:
        return _OURS["sysd"], _OURS["ctx"]
    c = C.build_context()
    texts = c.df["text_caption"].fillna("").astype(str).tolist()
    sysd = fit_system(c.emb.image, c.emb.text_caption, c.vu, c.D, texts, c.df["label"].tolist(),
                      c.y, c.tr, c.va, c.trainval, list(config.STEREO_CLASSES),
                      [config.LABEL_TO_IDX[cl] for cl in config.STEREO_CLASSES])
    _OURS.update(sysd=sysd, ctx=c)
    return sysd, c


def ours():
    sysd, c = fit_ours()
    logits = clf.predict_logits(sysd["clf"], c.vu, c.D)
    state = C.rl.build_state(logits, sysd["fired"], sysd["rvec"], c.D)
    ne = C.rl.greedy_pred(sysd["pol"], logits[c.te], state[c.te])
    names = [config.IDX_TO_LABEL[i] for i in range(config.NUM_CLASSES)]
    report_multiclass("Ours 5-class — deployed concept NeSy (seed 42)",
                      c.y[c.te], sysd["pb"][c.te], logits[c.te].argmax(1), ne, names)


def _ext_affect(vu):
    return np.zeros((len(vu), C.odesign.AFFECT_DIM), np.float32)   # affect-free deployed model


def mami(regime):
    import external_transfer as t19
    import mami_full_train as t20
    if regime == "zeroshot":
        loaded = t19._load_mami()
        _, texts, paths, y_bin, stereo = loaded
        v, u = t20._embeddings(paths, texts)
        vu = np.concatenate([v, u], 1).astype(np.float32); D = _ext_affect(vu)
        sysd, _ = fit_ours()
        report_binary("MAMI zero-shot (concept layer, no re-induction)",
                      zeroshot(sysd, u, v, vu, D), y_bin)
    elif regime == "binary":
        trd, ted = t20._load_train(), t19._load_mami()
        tr_t, tr_p, tr_y = trd; _, te_t, te_p, te_y, _ = ted
        texts, paths = tr_t + te_t, tr_p + te_p
        n_tr, N = len(tr_t), len(tr_t) + len(te_t)
        y = np.concatenate([tr_y, te_y]).astype(int)
        te = np.zeros(N, bool); te[n_tr:] = True
        v, u = t20._embeddings(paths, texts)
        vu = np.concatenate([v, u], 1).astype(np.float32); D = _ext_affect(vu)
        tr_i, va_i = train_test_split(np.arange(n_tr), test_size=0.15,
                                      stratify=y[:n_tr], random_state=SEED)
        tr = np.isin(np.arange(N), tr_i); va = np.isin(np.arange(N), va_i); trainval = tr | va
        config.LABEL_TO_IDX.setdefault("misogynous", 1)
        labels = np.where(y == 1, "misogynous", "non_misogynous")
        sysd = fit_system(v, u, vu, D, texts, labels, y, tr, va, trainval,
                          ["misogynous"], [1])
        _report_ext_binary("MAMI binary (full re-induction)", sysd, vu, D, y, te)
    else:
        print("MAMI multiclass omitted (Sub-task B is multi-label; use MMHS/EXIST 6-class).")


def mmhs(regime):
    import data_mmhs150k as m6
    z, ids, texts, y6, split = m6.load_all()
    vu, u = m6.encode_all(z, ids, texts); v = vu[:, :512]
    D = _ext_affect(vu)
    tr = split == "train"; va = split == "val"; te = split == "test"; trainval = tr | va
    if regime == "zeroshot":
        yb = (y6 == 2).astype(int)                                   # sexist vs rest
        sysd, _ = fit_ours()
        report_binary("MMHS zero-shot (concept layer, no re-induction)",
                      zeroshot(sysd, u[te], v[te], vu[te], D[te]), yb[te])
    elif regime == "binary":
        yb = (y6 == 2).astype(int)
        config.LABEL_TO_IDX.setdefault("sexist", 1)
        labels = np.where(yb == 1, "sexist", "non_sexist")
        sysd = fit_system(v, u, vu, D, texts, labels, yb, tr, va, trainval, ["sexist"], [1])
        _report_ext_binary("MMHS binary (full re-induction)", sysd, vu, D, yb, te)
    else:
        _multiclass_ext("MMHS 6-class (full re-induction)", v, u, vu, D, texts, y6,
                        tr, va, te, trainval, m6.NAMES, m6.HATE, m6.HATE_IDX, m6.train_clf6)


def exist(regime):
    import data_exist2024 as ex
    import data_exist2024_6class as ex6
    ids, texts, paths, raw = ex.load_all_en()
    vu, u = ex.embed(ids, paths, texts); v = vu[:, :512]
    D = _ext_affect(vu)
    if regime == "zeroshot":
        lab = np.array([ex.task4_hard(r) for r in raw]); keep = lab >= 0
        sysd, _ = fit_ours()
        report_binary("EXIST zero-shot (concept layer, no re-induction)",
                      zeroshot(sysd, u[keep], v[keep], vu[keep], D[keep]), lab[keep])
    elif regime == "binary":
        lab = np.array([ex.task4_hard(r) for r in raw]); keep = lab >= 0
        idx = np.arange(int(keep.sum()))
        vk, uk, vuk, Dk = v[keep], u[keep], vu[keep], D[keep]
        tk = [t for t, m in zip(texts, keep) if m]; yk = lab[keep]
        tr_i, tmp = train_test_split(idx, test_size=0.30, stratify=yk, random_state=SEED)
        va_i, te_i = train_test_split(tmp, test_size=0.50, stratify=yk[tmp], random_state=SEED)
        tr = np.isin(idx, tr_i); va = np.isin(idx, va_i); te = np.isin(idx, te_i); trainval = tr | va
        config.LABEL_TO_IDX.setdefault("sexist", 1)
        labels = np.where(yk == 1, "sexist", "non_sexist")
        sysd = fit_system(vk, uk, vuk, Dk, tk, labels, yk, tr, va, trainval, ["sexist"], [1])
        _report_ext_binary("EXIST binary (full re-induction)", sysd, vuk, Dk, yk, te)
    else:
        lab = np.array([ex6.label6(r) for r in raw]); keep = lab >= 0
        idx = np.arange(int(keep.sum()))
        vk, uk, vuk, Dk = v[keep], u[keep], vu[keep], D[keep]
        tk = [t for t, m in zip(texts, keep) if m]; yk = lab[keep]
        tr_i, tmp = train_test_split(idx, test_size=0.30, stratify=yk, random_state=SEED)
        va_i, te_i = train_test_split(tmp, test_size=0.50, stratify=yk[tmp], random_state=SEED)
        tr = np.isin(idx, tr_i); va = np.isin(idx, va_i); te = np.isin(idx, te_i); trainval = tr | va
        _multiclass_ext("EXIST Task-6 6-class (full re-induction)", vk, uk, vuk, Dk, tk, yk,
                        tr, va, te, trainval, ex6.NAMES, ex6.HATE, ex6.HATE_IDX, None)


def _report_ext_binary(title, sysd, vu, D, y, te):
    logits = clf.predict_logits(sysd["clf"], vu, D)
    state = C.rl.build_state(logits, sysd["fired"], sysd["rvec"], D)
    ne = C.rl.greedy_pred(sysd["pol"], logits[te], state[te])
    preds = {"Path-B (rule)": sysd["pb"][te], "Path-A": (logits[te].argmax(1) == 1).astype(int),
             "NeSy": (ne == 1).astype(int)}
    report_binary(title, preds, y[te])


def _multiclass_ext(title, v, u, vu, D, texts, y, tr, va, te, trainval,
                    NAMES, HATE, HATE_IDX, train_clf6=None):
    import data_mmhs150k as m6
    nc = len(NAMES)
    for k, nm in NAMES.items():
        config.LABEL_TO_IDX.setdefault(nm, k)
    labels = np.array([NAMES[int(t)] for t in y])
    trainer = train_clf6 or m6.train_clf6
    tclf = lambda vu_, D_, y_, tr_, va_: trainer(vu_, D_, y_, tr_, va_, C.odesign.AFFECT_DIM, SEED, num_classes=nc)
    sysd = fit_system(v, u, vu, D, texts, labels, y, tr, va, trainval,
                      HATE, HATE_IDX, n_classes=nc, train_clf=tclf)
    logits = clf.predict_logits(sysd["clf"], vu, D)
    state = C.rl.build_state(logits, sysd["fired"], sysd["rvec"], D)
    ne = C.rl.greedy_pred(sysd["pol"], logits[te], state[te])
    report_multiclass(title, y[te], sysd["pb"][te], logits[te].argmax(1), ne,
                      [NAMES[i] for i in range(nc)])


# --------------------------------------------------------------------------- #
# New external benchmarks: MMSD2.0, PrideMM, Hateful Memes, HarMeme
# --------------------------------------------------------------------------- #
def _split_masks(split):
    tr = split == "train"; va = split == "val"; te = split == "test"
    if not va.any():                                          # carve a val split from train
        tri = np.where(tr)[0]
        _, vb = train_test_split(tri, test_size=0.15, random_state=SEED)
        va = np.zeros(len(split), bool); va[vb] = True
        tr = tr & ~va
    return tr, va, te


def _ext_prep(ids, texts, paths, cache):
    import data_benchmarks as nd
    vu, u = nd.encode(ids, paths, texts, cache)
    return vu, u, vu[:, :512], _ext_affect(vu)


def ext_zeroshot(title, ids, texts, paths, ypos, split, cache):
    vu, u, v, D = _ext_prep(ids, texts, paths, cache)
    te = split == "test"
    sysd, _ = fit_ours()
    report_binary(title, zeroshot(sysd, u[te], v[te], vu[te], D[te]), ypos[te])


def ext_binary(title, ids, texts, paths, ypos, split, posname, cache):
    vu, u, v, D = _ext_prep(ids, texts, paths, cache)
    tr, va, te = _split_masks(split); trainval = tr | va
    config.LABEL_TO_IDX.setdefault(posname, 1)
    labels = np.where(ypos == 1, posname, f"non_{posname}")
    sysd = fit_system(v, u, vu, D, texts, labels, ypos, tr, va, trainval, [posname], [1])
    _report_ext_binary(title, sysd, vu, D, ypos, te)


def ext_multiclass(title, ids, texts, paths, y, split, names, cache):
    vu, u, v, D = _ext_prep(ids, texts, paths, cache)
    tr, va, te = _split_masks(split); trainval = tr | va
    HATE = [names[i] for i in range(1, len(names))]; HATE_IDX = list(range(1, len(names)))
    _multiclass_ext(title, v, u, vu, D, texts, y, tr, va, te, trainval, names, HATE, HATE_IDX)


def _sub(ids, texts, paths, keep):
    idx = np.where(keep)[0]
    return ([ids[i] for i in idx], [texts[i] for i in idx], [paths[i] for i in idx])


def mmsd(regime):
    import data_benchmarks as nd
    ids, texts, paths, y, split = nd.load_mmsd()
    if regime == "zeroshot":
        ext_zeroshot("MMSD2.0 zero-shot (sarcasm)", ids, texts, paths, y, split, "mmsd")
    elif regime == "binary":
        ext_binary("MMSD2.0 binary (sarcasm, full re-induction)", ids, texts, paths, y, split, "sarcastic", "mmsd")
    else:
        print("MMSD2.0 multiclass N/A (sarcasm is binary).")


def pridemm(regime):
    import data_benchmarks as nd
    ids, texts, paths, hate, target, split = nd.load_pridemm()
    if regime == "zeroshot":
        ext_zeroshot("PrideMM zero-shot (hate)", ids, texts, paths, hate, split, "pridemm")
    elif regime == "binary":
        ext_binary("PrideMM binary (hate, full re-induction)", ids, texts, paths, hate, split, "hate", "pridemm")
    else:
        keep = target >= 0                                   # target defined only for hate memes
        sids, stx, spa = _sub(ids, texts, paths, keep)
        names = {0: "undirected", 1: "individual", 2: "community", 3: "organization"}
        ext_multiclass("PrideMM target 4-class (full re-induction)",
                       sids, stx, spa, target[keep], split[keep], names, "pridemm_tgt")


def hateful(regime):
    import data_benchmarks as nd
    ids, texts, paths, y, split = nd.load_hateful()
    if regime == "zeroshot":
        ext_zeroshot("Hateful Memes zero-shot", ids, texts, paths, y, split, "hateful")
    elif regime == "binary":
        ext_binary("Hateful Memes binary (full re-induction)", ids, texts, paths, y, split, "hateful", "hateful")
    else:
        print("Hateful Memes multiclass N/A (binary).")


def harmeme(regime):
    import data_benchmarks as nd
    ids, texts, paths, y3, split = nd.load_harmeme()
    ybin = (y3 > 0).astype(int)                              # harmful (somewhat|very) vs not
    if regime == "zeroshot":
        ext_zeroshot("HarMeme zero-shot (harmful)", ids, texts, paths, ybin, split, "harmeme")
    elif regime == "binary":
        ext_binary("HarMeme binary (harmful, full re-induction)", ids, texts, paths, ybin, split, "harmful", "harmeme")
    else:
        names = {0: "not_harmful", 1: "somewhat_harmful", 2: "very_harmful"}
        ext_multiclass("HarMeme 3-class harm intensity (full re-induction)",
                       ids, texts, paths, y3, split, names, "harmeme")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "ours"
    reg = sys.argv[2] if len(sys.argv) > 2 else "zeroshot"
    {"ours": lambda: ours(), "mami": lambda: mami(reg), "mmhs": lambda: mmhs(reg),
     "exist": lambda: exist(reg), "mmsd": lambda: mmsd(reg), "pridemm": lambda: pridemm(reg),
     "hateful": lambda: hateful(reg), "harmeme": lambda: harmeme(reg)}[ds]()
