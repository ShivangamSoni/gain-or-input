"""How much of each meme benchmark is solved by TEXT ALONE?

Companion to s2_text_regimes.py. That script showed our own corpus is very
nearly solved by the post caption scraped with each meme, which is what made a
+0.189 macro-F1 "neuro-symbolic margin" collapse to +0.012 once both paths were
given the same channel. The obvious next question is whether that is our dataset's
private problem or a property of the benchmark family, because the answer decides
whether this is a postmortem or a field-level measurement result.

So: for every corpus and task in the external suite, fit the SAME probe on three
inputs -- image only, text only, and both -- and compare. The column that matters
is Delta(both - text). Near zero means the image contributes nothing the text did
not already carry, and any multimodal claim on that benchmark is measuring text.

Everything runs off the cached frozen CLIP embeddings the suite already built
(artifacts/*_clip.npz), so no encoder runs and no dataset is re-read. The probe is
deliberately plain -- LayerNorm, one hidden layer, class-weighted CE, best-val
checkpoint -- because the point is what the FEATURES contain, not how well a head
can be tuned.

  python experiments/d5_modality_probes.py --list
  python experiments/d5_modality_probes.py --run ours,mmsd
  python experiments/d5_modality_probes.py --run all --seeds 5
  python experiments/d5_modality_probes.py --report
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from nesymis import config
from nesymis.base import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = str(config.ARTIFACTS_DIR / "d5_modality_probes.json")
SEED = config.SEED


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
class Probe(nn.Module):
    def __init__(self, d, k, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def fit_probe(X, y, tr, va, k, seed, epochs=60, bs=64, lr=1e-3, wd=1e-4):
    set_seed(seed)
    model = Probe(X.shape[1], k).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cnt = np.bincount(y[tr], minlength=k).astype(np.float32)
    w = torch.from_numpy(cnt.sum() / (k * np.clip(cnt, 1, None))).to(DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    T = torch.from_numpy(X).float().to(DEV)
    yt = torch.from_numpy(y).long().to(DEV)
    tr_i = torch.from_numpy(np.flatnonzero(tr)).to(DEV)
    va_i = torch.from_numpy(np.flatnonzero(va)).to(DEV)
    va_np = np.flatnonzero(va)
    rng = np.random.default_rng(seed)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(tr_i))).to(DEV)
        for i in range(0, len(perm), bs):
            j = tr_i[perm[i : i + bs]]
            opt.zero_grad(set_to_none=True)
            crit(model(T[j]), yt[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = model(T[va_i]).argmax(1).cpu().numpy()
        f1 = f1_score(y[va_np], p, average="macro", labels=list(range(k)), zero_division=0)
        if f1 > best:
            best = f1
            best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(T).cpu().numpy(), best


def score(y, pred, k):
    return {"acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(k)),
                                       zero_division=0)),
            "pos_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)) if k == 2 else None}


def probe_all(task, seeds):
    """Run the image/text/both probes for one task spec."""
    v, u, y, tr, va, te, k, name = (task[x] for x in
                                    ("v", "u", "y", "tr", "va", "te", "k", "name"))
    views = {"image": v, "text": u, "both": np.concatenate([v, u], 1)}
    out = {}
    for vn, X in views.items():
        X = np.ascontiguousarray(X.astype(np.float32))
        runs = []
        for s in [SEED] + [int(x) for x in np.random.default_rng(SEED).integers(0, 2**31 - 1,
                                                                               seeds - 1)]:
            lg, _ = fit_probe(X, y, tr, va, k, s)
            runs.append(score(y[te], lg[te].argmax(1), k))
        out[vn] = {m: [r[m] for r in runs] for m in runs[0] if runs[0][m] is not None}
        print(f"  [{name}] {vn:<6} macro-F1 {np.mean(out[vn]['macro_f1']):.3f}"
              f"+-{np.std(out[vn]['macro_f1']):.3f}", flush=True)
    return {"name": name, "k": k, "n": int(len(y)), "n_test": int(te.sum()),
            "classes": task.get("classes"), "seeds": seeds, "views": out}


# --------------------------------------------------------------------------- #
# Task specs -- each returns v, u, y, masks, k. All from cached embeddings.
# --------------------------------------------------------------------------- #
def _spec(name, v, u, y, tr, va, te, k, classes=None):
    return dict(name=name, v=v, u=u, y=np.asarray(y).astype(int), tr=tr, va=va, te=te,
                k=k, classes=classes)


def t_ours_ocr():
    from nesymis.data import dataset as ds
    e = ds.load_embeddings()
    tr, va, te = (e.split_mask(s) for s in ("train", "val", "test"))
    return _spec("ours-5class (OCR text)", e.image, e.text_ocr, e.labels, tr, va, te, 5,
                 list(config.LABELS))


def t_ours_caption():
    from nesymis.data import dataset as ds
    e = ds.load_embeddings()
    tr, va, te = (e.split_mask(s) for s in ("train", "val", "test"))
    return _spec("ours-5class (clean caption)", e.image, e.text_caption, e.labels, tr, va, te,
                 5, list(config.LABELS))


def t_mami():
    import external_transfer as t19
    import mami_full_train as t20
    tr_t, tr_p, tr_y = t20._load_train()
    _, te_t, te_p, te_y, _ = t19._load_mami()
    texts, paths = tr_t + te_t, tr_p + te_p
    n_tr, N = len(tr_t), len(tr_t) + len(te_t)
    y = np.concatenate([tr_y, te_y]).astype(int)
    v, u = t20._embeddings(paths, texts)
    te = np.zeros(N, bool)
    te[n_tr:] = True
    tri, vai = train_test_split(np.arange(n_tr), test_size=0.15, stratify=y[:n_tr],
                                random_state=SEED)
    tr = np.isin(np.arange(N), tri)
    va = np.isin(np.arange(N), vai)
    return _spec("MAMI (misogynous, binary)", v, u, y, tr, va, te, 2)


def _mmhs():
    import data_mmhs150k as m6
    z, ids, texts, y6, split = m6.load_all()
    vu, u = m6.encode_all(z, ids, texts)
    return vu[:, :512], u, y6, split == "train", split == "val", split == "test", m6.NAMES


def t_mmhs_bin():
    v, u, y6, tr, va, te, _ = _mmhs()
    return _spec("MMHS150K (sexist, binary)", v, u, (np.asarray(y6) == 2).astype(int),
                 tr, va, te, 2)


def t_mmhs_6():
    v, u, y6, tr, va, te, names = _mmhs()
    return _spec("MMHS150K (6-class hate type)", v, u, y6, tr, va, te, 6,
                 [names[i] for i in range(6)])


def _exist(labfn, k, name, classes=None):
    import data_exist2024 as ex
    ids, texts, paths, raw = ex.load_all_en()
    vu, u = ex.embed(ids, paths, texts)
    lab = np.array([labfn(r) for r in raw])
    keep = lab >= 0
    idx = np.arange(int(keep.sum()))
    yk = lab[keep]
    tri, tmp = train_test_split(idx, test_size=0.30, stratify=yk, random_state=SEED)
    vai, tei = train_test_split(tmp, test_size=0.50, stratify=yk[tmp], random_state=SEED)
    return _spec(name, vu[keep][:, :512], u[keep], yk, np.isin(idx, tri), np.isin(idx, vai),
                 np.isin(idx, tei), k, classes)


def t_exist_bin():
    import data_exist2024 as ex
    return _exist(ex.task4_hard, 2, "EXIST-2024 (sexist, binary)")


def t_exist_6():
    import data_exist2024_6class as ex6
    return _exist(ex6.label6, 6, "EXIST-2024 Task 6 (6-class)",
                  [ex6.NAMES[i] for i in range(6)])


def _nd(loader, cache, name, k, ycol=None, classes=None, keepfn=None):
    import concept_scorecard_eval as ce
    import data_benchmarks as nd
    got = getattr(nd, loader)()
    if len(got) == 6:                                   # pridemm: ids, texts, paths, hate, tgt, split
        ids, texts, paths, y1, y2, split = got
        y = y2 if ycol == "target" else y1
    else:
        ids, texts, paths, y, split = got
    y = np.asarray(y)
    if keepfn is not None:
        y = keepfn(y)
    keep = y >= 0
    if not keep.all():
        idx = np.flatnonzero(keep)
        ids = [ids[i] for i in idx]
        texts = [texts[i] for i in idx]
        paths = [paths[i] for i in idx]
        y, split = y[idx], np.asarray(split)[idx]
    vu, u, v, _ = ce._ext_prep(ids, texts, paths, cache)
    tr, va, te = ce._split_masks(np.asarray(split))
    return _spec(name, v, u, y.astype(int), tr, va, te, k, classes)


TASKS = {
    "ours-ocr": t_ours_ocr,
    "ours-caption": t_ours_caption,
    "mami": t_mami,
    "mmhs-bin": t_mmhs_bin,
    "mmhs-6": t_mmhs_6,
    "exist-bin": t_exist_bin,
    "exist-6": t_exist_6,
    "mmsd": lambda: _nd("load_mmsd", "mmsd", "MMSD2.0 (sarcasm, binary)", 2),
    "pridemm": lambda: _nd("load_pridemm", "pridemm", "PrideMM (hate, binary)", 2),
    "pridemm-tgt": lambda: _nd("load_pridemm", "pridemm_tgt", "PrideMM (4-class target)", 4,
                               ycol="target",
                               classes=["undirected", "individual", "community", "organization"]),
    "hateful": lambda: _nd("load_hateful", "hateful", "Hateful Memes (binary)", 2),
    "harmeme": lambda: _nd("load_harmeme", "harmeme", "HarMeme (harmful, binary)", 2,
                           keepfn=lambda y: (y > 0).astype(int)),
    "harmeme-3": lambda: _nd("load_harmeme", "harmeme", "HarMeme (3-class intensity)", 3,
                             classes=["not_harmful", "somewhat_harmful", "very_harmful"]),
}


def _load():
    return json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}


def _save(d):
    cur = _load()
    cur.update(d)
    json.dump(cur, open(OUT, "w", encoding="utf-8"), indent=1)


def cmd_run(which, seeds):
    keys = list(TASKS) if which == "all" else [k.strip() for k in which.split(",")]
    for k in keys:
        if k not in TASKS:
            print(f"[ts] unknown task '{k}'; see --list", flush=True)
            continue
        print(f"\n########## text-solvability: {k} ##########", flush=True)
        try:
            _save({k: probe_all(TASKS[k](), seeds)})
        except Exception:
            print(f"[ts] !! FAILED {k}:\n{traceback.format_exc()}", flush=True)
    print("\nTS_DONE", flush=True)


def cmd_list():
    d = _load()
    print(f"{'key':<16}{'done':<7}{'task'}")
    print("-" * 62)
    for k in TASKS:
        print(f"{k:<16}{'yes' if k in d else '-':<7}{k}")


def cmd_report():
    d = _load()
    if not d:
        raise SystemExit("nothing yet -- run --run all")
    print("## Text-solvability of meme benchmarks "
          "(frozen CLIP B/32, identical probe, macro-F1)\n")
    print("The column to read is **Δ both−text**. At or near zero, the image adds "
          "nothing the text did not already carry.\n")
    print("| Corpus / task | classes | n test | image | text | both | Δ both−text | "
          "Δ both−image |")
    print("|---|---|---|---|---|---|---|---|")
    for k in TASKS:
        r = d.get(k)
        if not r:
            continue
        g = {vn: float(np.mean(r["views"][vn]["macro_f1"])) for vn in ("image", "text", "both")}
        flag = " ⚠️" if g["both"] - g["text"] <= 0.01 else ""
        print(f"| {r['name']} | {r['k']} | {r['n_test']} | {g['image']:.3f} | "
              f"**{g['text']:.3f}** | {g['both']:.3f} | {g['both'] - g['text']:+.3f}{flag} | "
              f"{g['both'] - g['image']:+.3f} |")
    print("\n⚠️ = the image channel contributes ≤0.01 macro-F1 over text alone.")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--run", metavar="KEYS", help="comma-separated task keys, or 'all'")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.run:
        cmd_run(a.run, a.seeds)
    elif a.report:
        cmd_report()
    else:
        cmd_list()


if __name__ == "__main__":
    main()
