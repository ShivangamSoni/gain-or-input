"""Known-answer tests of the audit's system tier across architectures (fixed before the run).

The MMHS150K test (known_answer_mmhs.py) tested S1 and S2 on one system built on MMHS150K: a baseline N
(CLIP image + in-image text) pooled with an added component P that reads the tweet text (a gain that
belongs to the input) or reads N's inputs through SigLIP (a genuine gain). Here P is built in four
further architectures, with that test's data, baseline, pooling and seeds:

  transformer  a fusion transformer over an image token and a text token
  retrieval    a cosine nearest-neighbour classifier over image + text embeddings
  concepts     the paper's concept bank and concept scorecard (neuro-symbolic)
  vlm          a zero-shot vision-language model (minicpm-v4.5, local Ollama), on a 1,500-tweet
               stratified test subsample

Expected, fixed before running (as in that test): positive control (P reads the tweet) S1 FLAG, S2 FLAG;
negative control (SigLIP view; for concepts and the VLM the equal-input view) S1 ok, S2 ok. The swing
(system F1 with the tweet minus with equal inputs) is reported with a paired bootstrap interval.

  python experiments/known_answer_architectures.py [--only transformer,retrieval,concepts,vlm]
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import zipfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import known_answer_mmhs as a1
import data_mmhs150k as m6
from nesymis import config
from nesymis.base import set_seed
from inputaudit import system_tier as st
from inputaudit.probes import macro_f1, paired_bootstrap
from d5_modality_probes import fit_probe

OUT = config.ARTIFACTS_DIR / "known_answer_architectures.json"
VLM_CACHE = config.ARTIFACTS_DIR / "mmhs_vlm_minicpm.json"
VLM_MODEL = "minicpm-v4.5:latest"
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SEEDS = a1.SEEDS
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VLM_LABELS = ["not_hate", "racist", "sexist", "homophobe", "religion", "other_hate"]   # = m6.NAMES order
PROMPT = ('Classify this tweet (image attached) for hate-speech research. Text: "{t}"\n'
          'Choose ONE label: "not_hate", "racist", "sexist", "homophobe", "religion", "other_hate".\n'
          'Answer with ONLY a JSON object: {{"label": "<label>"}}')


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- #
# Architectures for the added component P. Each returns test-row log-probabilities.
# --------------------------------------------------------------------------- #
class TokenFusion(nn.Module):
    def __init__(self, d_img, d_txt, k, d=128):
        super().__init__()
        self.pi = nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, d))
        self.pt = nn.Sequential(nn.LayerNorm(d_txt), nn.Linear(d_txt, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mod = nn.Parameter(torch.randn(1, 3, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, 0.1, batch_first=True), 2)
        self.head = nn.Linear(d, k)

    def forward(self, xi, xt):
        tok = torch.cat([self.cls.expand(len(xi), -1, -1), self.pi(xi)[:, None], self.pt(xt)[:, None]], 1)
        return self.head(self.enc(tok + self.mod)[:, 0])


def transformer_logprobs(Xi, Xt, y, tr, va, te, k, seed, epochs=40, bs=256):
    set_seed(seed)
    model = TokenFusion(Xi.shape[1], Xt.shape[1], k).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(y[tr], minlength=k).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(cnt.sum() / (k * np.clip(cnt, 1, None))).to(DEV))
    I, T = (torch.from_numpy(np.ascontiguousarray(X, np.float32)).to(DEV) for X in (Xi, Xt))
    Y = torch.from_numpy(np.asarray(y)).long().to(DEV)
    tr_i, va_i, te_i = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
    rng = np.random.default_rng(seed)

    def logits(idx):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(I[j], T[j]).cpu().numpy() for j in
                                   (torch.from_numpy(idx[s:s + 4096]).to(DEV) for s in range(0, len(idx), 4096))])

    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        for j in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // bs)):
            jj = torch.from_numpy(j).to(DEV)
            opt.zero_grad(set_to_none=True)
            crit(model(I[jj], T[jj]), Y[jj]).backward()
            opt.step()
        f = macro_f1(y[va_i], logits(va_i).argmax(1), k)
        if f > best:
            best, state = f, {n: t.detach().clone() for n, t in model.state_dict().items()}
    model.load_state_dict(state)
    return log_softmax(logits(te_i))


def retrieval_logprobs(Xi, Xt, y, fit, te, k, kk=25, balanced=False):
    """Cosine kNN. `balanced` (post hoc): each class's votes divided by its training
    frequency, since unweighted votes on MMHS150K's imbalanced labels return the majority class."""
    X = unit(np.concatenate([unit(Xi), unit(Xt)], 1))
    A, B, ya = X[fit], X[te], np.asarray(y)[fit]
    prior = np.bincount(ya, minlength=k) / len(ya)
    votes = np.zeros((len(B), k))
    for s in range(0, len(B), 2048):
        S = B[s:s + 2048] @ A.T
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        w = np.take_along_axis(S, idx, 1).clip(min=0)
        lab = ya[idx]
        for c in range(k):
            votes[s:s + 2048, c] = (w * (lab == c)).sum(1)
    if balanced:                                            # reweight the votes, then a tiny floor
        p = votes / prior + 1e-3
    else:
        p = votes + 1.0                                     # add-one smoothing (as pre-registered)
    return np.log(p / p.sum(1, keepdims=True))


def concept_logprobs(text, u, v, y, tr, va, te, classes):
    from nesymis.fusion import concept_pool as cp
    from nesymis.symbolic import concept_layer as cl
    df = pd.DataFrame({"text_caption": list(text), "label": [classes[i] for i in y]})
    trainval = tr | va
    maj = classes[int(np.bincount(y[trainval]).argmax())]
    bank = cl.build(SimpleNamespace(df=df, text_caption=u, image=v), trainval,
                    classes=[c for c in classes if c != maj])
    S = bank.frame(u, v, index=df.index).to_numpy(np.float64)
    card = cp.ConceptScorecard.fit(S, [str(j) for j in range(S.shape[1])], y, tr, va, k=len(classes))
    return cp.log_softmax(card.logits(S[te]))


def vlm_answers(ids, texts, rows, view):
    """minicpm-v4.5 labels for `rows` (indices into ids) given `texts`; cached per (tweet, view)."""
    import requests
    cache = json.load(open(VLM_CACHE, encoding="utf-8")) if VLM_CACHE.exists() else {}
    z = zipfile.ZipFile(m6.ZIP)
    t0, n_new = time.time(), 0
    for n, i in enumerate(rows):
        key = f"{ids[i]}|{view}"
        if key in cache:
            continue
        b = base64.b64encode(z.read(f"img_resized/{ids[i]}.jpg")).decode()
        lab = None
        for attempt in range(6):
            try:
                r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
                    "model": VLM_MODEL, "stream": False, "think": False,
                    "options": {"temperature": 0, "num_predict": 40, "num_ctx": 4096},
                    "messages": [{"role": "user", "content": PROMPT.format(t=str(texts[i])[:600]), "images": [b]}]})
                if r.status_code != 200 or "error" in r.json():     # server-side failure: retry, never
                    raise RuntimeError(r.text[:200])                 # record it as an unparseable answer
                m = re.search(r'"label"\s*:\s*"([a-z_ ]+)"', r.json().get("message", {}).get("content", "").lower())
                if m and m.group(1).strip().replace(" ", "_") in VLM_LABELS:
                    lab = VLM_LABELS.index(m.group(1).strip().replace(" ", "_"))
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            json.dump(cache, open(VLM_CACHE, "w", encoding="utf-8"))
            raise RuntimeError(f"VLM server failed 6 times on {key}; answers so far are cached")
        cache[key] = lab
        n_new += 1
        if n_new % 50 == 0:
            json.dump(cache, open(VLM_CACHE, "w", encoding="utf-8"))
            print(f"  [vlm] {view}: {n + 1}/{len(rows)} ({(time.time() - t0) / n_new:.1f}s each)", flush=True)
    json.dump(cache, open(VLM_CACHE, "w", encoding="utf-8"))
    return [cache[f"{ids[i]}|{view}"] for i in rows]


def vlm_logprobs(answers, k, binary):
    P = np.full((len(answers), k), 1.0 / k)
    for j, a in enumerate(answers):
        if a is None:
            continue
        c = int(a == 2) if binary else a
        P[j] = 0.2 / k
        P[j, c] += 0.8
    return np.log(P)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="transformer,retrieval,concepts,vlm")
    ap.add_argument("--tasks", default="6,2", help="6 = 6-class hate type, 2 = sexist vs rest")
    a = ap.parse_args()
    archs, tasks = a.only.split(","), a.tasks.split(",")
    ids, post, img, y6, split, all_ids = a1.load_pool()
    tr, va, te = split == "train", split == "val", split == "test"
    E = a1.embeddings(ids, post, img, all_ids)
    te_i = np.flatnonzero(te)
    sub = np.sort(train_test_split(te_i, train_size=1500, stratify=y6[te_i], random_state=config.SEED)[0])
    sub_pos = np.searchsorted(te_i, sub)                    # subsample positions within the test rows
    print(f"[ms] pool {len(ids)}: train {tr.sum()} val {va.sum()} test {te.sum()}; VLM subsample {len(sub)}", flush=True)
    res = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    Nview = np.concatenate([E["v"], E["u_img"]], 1)

    for task, y, k in (("6-class hate type", y6, 6), ("sexist vs rest", (y6 == 2).astype(int), 2)):
        if str(k) not in tasks:
            continue
        classes =[m6.NAMES[i] for i in range(6)] if k == 6 else ["not_sexist", "sexist"]
        n_cache = {}

        def n_logits(seed):
            if seed not in n_cache:
                lg, _ = fit_probe(np.ascontiguousarray(Nview.astype(np.float32)), y, tr, va, k, seed)
                n_cache[seed] = log_softmax(lg[te])
            return n_cache[seed]

        for arch in archs:
            t0 = time.time()
            views = {"tweet": ("image", "tweet text"), "same": ("image", "image text")}
            if arch in ("transformer", "retrieval", "retrieval-balanced"):
                views["siglip"] = ("image", "image text")
            p_cache = {}
            rows = sub_pos if arch == "vlm" else np.arange(len(te_i))

            def p_logits(view, seed):
                key = (view, seed if arch == "transformer" else 0)
                if key in p_cache:
                    return p_cache[key]
                Xi = E["sv"] if view == "siglip" else E["v"]
                Xt = {"tweet": E["u_post"], "same": E["u_img"], "siglip": E["su_img"]}[view]
                if arch == "transformer":
                    out = transformer_logprobs(Xi, Xt, y, tr, va, te, k, seed)
                elif arch in ("retrieval", "retrieval-balanced"):
                    out = retrieval_logprobs(Xi, Xt, y, tr | va, te, k, balanced=(arch == "retrieval-balanced"))
                elif arch == "concepts":
                    out = concept_logprobs(post if view == "tweet" else img, Xt, Xi, y, tr, va, te, classes)
                else:
                    ans = vlm_answers(ids, post if view == "tweet" else img, sub, view)
                    out = vlm_logprobs(ans, k, binary=(k == 2))
                p_cache[key] = out
                return out

            def run(name, cfg, seed):
                ln = n_logits(seed)[rows]
                lp = p_logits(cfg["P"], seed)
                return (0.5 * ln + 0.5 * lp).argmax(1), ln.argmax(1)

            yt = y[te][rows]
            eq = "equal inputs (P reads the image text)"
            pos = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": list(views["tweet"])}, ["N", "P"], ["N"],
                                         ["image", "image text", "tweet text"]),
                   "s2": st.s2_same_input(run, {"as built (P reads the tweet)": {"P": "tweet"}, eq: {"P": "same"}},
                                          SEEDS, yt, k, equal=eq)}
            Pt = np.array([run("", {"P": "tweet"}, s)[0] for s in SEEDS])
            Pe = np.array([run("", {"P": "same"}, s)[0] for s in SEEDS])
            pos["swing"] = paired_bootstrap(yt, Pt, Pe, k)["d_macro_f1"]
            neg_view = "siglip" if "siglip" in views else "same"
            neg_name = "P reads image + image text via SigLIP" if neg_view == "siglip" else "P reads image + image text"
            neg = {"s1": st.s1_inventory({"N": ["image", "image text"], "P": ["image", "image text"]}, ["N", "P"], ["N"],
                                         ["image", "image text"]),
                   "s2": st.s2_same_input(run, {neg_name: {"P": neg_view}}, SEEDS, yt, k, equal=neg_name)}
            alone = {"N": float(np.mean([macro_f1(yt, n_logits(s)[rows].argmax(1), k) for s in SEEDS]))}
            for v_ in views:
                alone[f"P_{v_}"] = float(np.mean([macro_f1(yt, p_logits(v_, s).argmax(1), k) for s in SEEDS]))
            res.setdefault(arch, {})[task] = {"positive_control": pos, "negative_control": neg,
                                              "components_alone": alone, "n_test": int(len(rows)),
                                              "minutes": round((time.time() - t0) / 60, 1)}
            json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
            p2, n2 = pos["s2"], neg["s2"]
            print(f"[ms] {arch} | {task} | alone {json.dumps({a: round(b, 3) for a, b in alone.items()})}", flush=True)
            for nm, r_ in p2["configs"].items():
                print(f"  positive | {nm}: margin {r_['margin']['mean']:+.3f} {np.round(r_['margin']['ci'], 3)}")
            print(f"  positive | S1 {pos['s1']['verdict']} | S2 {p2['verdict']} | swing {pos['swing']['mean']:+.3f} "
                  f"{np.round(pos['swing']['ci'], 3)}")
            for nm, r_ in n2["configs"].items():
                print(f"  negative | {nm}: margin {r_['margin']['mean']:+.3f} {np.round(r_['margin']['ci'], 3)}")
            print(f"  negative | S1 {neg['s1']['verdict']} | S2 {n2['verdict']}", flush=True)
    print(f"\n_saved {OUT}_")


if __name__ == "__main__":
    main()
