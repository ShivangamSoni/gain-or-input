"""External #2c - full 6-CLASS retraining on MMHS150K (framework-generalization
test). Unlike the binary analogue, this exercises the entire framework on the
dataset's NATIVE single-label 6-way taxonomy:

  0 NotHate (default, no rules -- analogous to our non_stereotype)
  1 Racist   2 Sexist   3 Homophobe   4 Religion   5 OtherHate  (rule classes)

Every component is re-learned from MMHS's own train split: per-class predicate
mining, per-class DNF induction (+ meta any-hate rule), a task-specific 6-class
neural classifier (a local copy of the trainer; the deployed 5-class classifier
is untouched), and the linear evidence decision layer with n_classes=6 and one
rule group per hate class. The affect module is OUR frozen head. Classifier +
policy over the 11-seed protocol; symbolic path deterministic. Shares the CLIP
embedding cache (artifacts/mmhs_full_clip.npz) with the binary full-train run.

Note: severe imbalance (Religion has ~24 test samples); macro-F1 over 6 classes
is low/noisy by construction, and classes below MIN_SUPPORT induce no rules.
"""
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import common as C
from nesymis import config
from nesymis.base import set_seed
from nesymis.metrics import compute_metrics
from nesymis.neural import classifier as clf
from nesymis.symbolic import predicate_pool, rule_induction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMHS = os.path.join(ROOT, "dataset", "MMHS150K")
ZIP = os.path.join(MMHS, "MMHS150K.zip")
GT = os.path.join(MMHS, "MMHS150K_GT.json")
CACHE = os.path.join(ROOT, "artifacts", "mmhs_full_clip.npz")   # shared with binary run

NAMES = {0: "nothate", 1: "racist", 2: "sexist", 3: "homophobe", 4: "religion", 5: "otherhate"}
HATE = ["racist", "sexist", "homophobe", "religion", "otherhate"]        # rule classes (1..5)
HATE_IDX = [1, 2, 3, 4, 5]
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")


def clean(text, img_text=""):
    t = re.sub(r"\s+", " ", _MENTION.sub(" ", _URL.sub(" ", text or ""))).strip()
    return (t + " " + img_text.strip()).strip() if img_text else t


def majority(labels):
    top, n = Counter(labels).most_common(1)[0]
    return top if n >= 2 else -1


def load_all():
    gt = json.load(open(GT, encoding="utf-8"))
    z = zipfile.ZipFile(ZIP)
    names = set(z.namelist())
    split_of = {}
    for s in ("train", "val", "test"):
        for tid in z.read(f"splits/{s}_ids.txt").decode().split():
            split_of[tid] = s
    ids, texts, y, split = [], [], [], []
    for tid, s in split_of.items():
        e = gt.get(tid)
        if e is None or f"img_resized/{tid}.jpg" not in names:
            continue
        maj = majority(e["labels"])
        if maj < 0:
            continue
        it = ""
        if f"img_txt/{tid}.json" in names:
            try:
                it = json.loads(z.read(f"img_txt/{tid}.json")).get("img_text", "")
            except Exception:
                it = ""
        ids.append(tid); texts.append(clean(e["tweet_text"], it))
        y.append(maj); split.append(s)
    return z, ids, texts, np.array(y, int), np.array(split)


def encode_all(z, ids, texts):
    if os.path.isfile(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if list(d["ids"]) == ids:
            print("  [6cls] embeddings from shared cache", flush=True)
            return d["vu"], d["u"]
    from nesymis.encoders.clip_encoder import get_clip, encode_texts
    model, proc = get_clip()
    dev = next(model.parameters()).device
    B, V = 128, []
    for i in range(0, len(ids), B):
        imgs = []
        for tid in ids[i:i + B]:
            try:
                imgs.append(Image.open(io.BytesIO(z.read(f"img_resized/{tid}.jpg"))).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (127, 127, 127)))
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            emb = model.get_image_features(**inp).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
        V.append(emb.cpu().numpy().astype(np.float32))
        if (i // B) % 100 == 0:
            print(f"  [6cls] images {i}/{len(ids)}", flush=True)
    v = np.concatenate(V)
    u = np.concatenate([encode_texts(texts[i:i + 256]) for i in range(0, len(texts), 256)])
    vu = np.concatenate([v, u], axis=1).astype(np.float32)
    np.savez(CACHE, ids=np.array(ids, object), vu=vu, u=u)
    return vu, u


def train_clf6(vu, e_raw, y, tr, va, affect_dim, seed, num_classes=6):
    """Local 6-class copy of clf.train_classifier (deployed 5-class one untouched)."""
    cfg = config.MLP
    set_seed(seed)
    model = clf.Classifier(e_raw.shape[1], fusion=config.NEURAL_FUSION, affect_dim=affect_dim,
                                 vu_dim=vu.shape[1], num_classes=num_classes,
                                 hidden=cfg["hidden"]).to(clf._device)
    if cfg.get("standardize", False):
        model.vu_mu.copy_(torch.from_numpy(vu[tr].mean(0)).to(clf._device))
        model.vu_sd.copy_(torch.from_numpy(np.clip(vu[tr].std(0), 1e-6, None)).to(clf._device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weight = None
    if cfg.get("class_weighted", False):
        cnt = np.bincount(y[tr], minlength=num_classes).astype(np.float32)
        weight = torch.from_numpy(cnt.sum() / (num_classes * np.clip(cnt, 1, None))).to(clf._device)
    crit = nn.CrossEntropyLoss(weight=weight)
    loader = DataLoader(TensorDataset(torch.from_numpy(vu[tr]).float(),
                                      torch.from_numpy(e_raw[tr]).float(),
                                      torch.from_numpy(y[tr]).long()),
                        batch_size=cfg["batch_size"], shuffle=True)
    best_f1, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for xb, eb, yb in loader:
            xb, eb, yb = xb.to(clf._device), eb.to(clf._device), yb.to(clf._device)
            opt.zero_grad(); crit(model(xb, eb), yb).backward(); opt.step()
        vp = clf.predict_logits(model, vu[va], e_raw[va]).argmax(1)
        f1 = compute_metrics(y[va], vp)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    if not (os.path.isfile(ZIP) and os.path.isfile(GT)):
        print("MMHS150K not found under dataset/MMHS150K/"); return
    z, ids, texts, y, split = load_all()
    tr, va, te = split == "train", split == "val", split == "test"
    trainval = tr | va
    dist = {NAMES[k]: int((y[te] == k).sum()) for k in range(6)}
    print(f"  [6cls] train {tr.sum()} / val {va.sum()} / test {te.sum()}; test dist {dist}", flush=True)

    c = C.build_context()
    from nesymis.affect import head as affect_head
    vu, u = encode_all(z, ids, texts)
    D = affect_head.predict_affect(c.system.head, vu)

    for k, nm in NAMES.items():                                   # register label ids for induction
        config.LABEL_TO_IDX.setdefault(nm, k)
    labels = np.array([NAMES[int(v)] for v in y])
    emb_ns = SimpleNamespace(df=pd.DataFrame({"text_caption": texts, "label": labels}),
                             text_caption=u)
    pool_df, lex, lexn = predicate_pool.build_pool(emb_ns, classes=HATE, calib_mask=trainval, save=False)
    rs, L, names = rule_induction.induce(pool_df, y, trainval, affect=D, classes=HATE,
                                         extra_bool=(lexn, lex), save=False)

    # per-class fired + trigger precision; Path-B pred = argmax fired precision else NotHate(0)
    R = len(HATE)
    fired = np.zeros((len(ids), R), np.float32)
    trig = np.zeros((len(ids), R), np.float32)
    rvec = np.zeros(R, np.float32)
    for j, hc in enumerate(HATE):
        prec_j = 0.0
        for cl in rs[hc]:
            fj = np.all(L[:, cl["idx"]], axis=1)
            fired[:, j] = np.maximum(fired[:, j], fj.astype(np.float32))
            trig[:, j] = np.where(fj, np.maximum(trig[:, j], cl["precision"]), trig[:, j])
            prec_j = max(prec_j, cl["precision"])
        m = trainval & (fired[:, j] > 0)
        rvec[j] = float((y[m] == HATE_IDX[j]).mean()) if m.any() else 0.0
    any_f = fired.any(1)
    pb_all = np.where(any_f, np.array(HATE_IDX)[trig.argmax(1)], 0)

    yte = y[te]
    labels6 = list(range(6))

    def mc(p):
        return (float((p == yte).mean()), f1_score(yte, p, labels=labels6, average="macro", zero_division=0))

    res = {r: {"acc": [], "mf1": [], "per": []} for r in ("Path-A", "NeSy")}
    ctor = lambda d: C.rl.LinearEvidencePolicy(d, n_classes=6, rule_classes=HATE_IDX)
    nseed = len(C.SEEDS11)
    for si, seed in enumerate(C.SEEDS11, 1):
        print(f"  [6cls] seed {si}/{nseed} (seed={seed}) starting ...", flush=True)
        clf = train_clf6(vu, D, y, tr, va, C.odesign.AFFECT_DIM, int(seed))
        logits = clf.predict_logits(clf, vu, D)
        state = C.rl.build_state(logits, fired, rvec, D)
        pol = C.rl.train_rlvr(logits, state, y, tr, va, seed=int(seed), policy_ctor=ctor)[0]
        ne = C.rl.greedy_pred(pol, logits[te], state[te])
        for r, p in (("Path-A", logits[te].argmax(1)), ("NeSy", ne)):
            a, mf = mc(p)
            res[r]["acc"].append(a); res[r]["mf1"].append(mf)
            res[r]["per"].append(f1_score(yte, p, labels=labels6, average=None, zero_division=0))
        print(f"  [6cls] seed {si}/{nseed} done -> "
              f"Path-A {res['Path-A']['acc'][-1]:.3f}/{res['Path-A']['mf1'][-1]:.3f}  "
              f"NeSy {res['NeSy']['acc'][-1]:.3f}/{res['NeSy']['mf1'][-1]:.3f}", flush=True)

    pb_a, pb_m = mc(pb_all[te])
    pb_per = f1_score(yte, pb_all[te], labels=labels6, average=None, zero_division=0)
    hdr = " | ".join(f"F1-{NAMES[k]}" for k in range(6))
    print("\n## Full 6-class retraining on MMHS150K: native taxonomy, all components "
          f"re-learned from MMHS train (n={int(tr.sum())}), official test (n={int(te.sum())})\n")
    print(f"| Path | Acc | Macro-F1 | {hdr} |")
    print("|" + "---|" * (9))
    print(f"| Path-B (rules) | {pb_a:.3f} | {pb_m:.3f} | " + " | ".join(f"{x:.3f}" for x in pb_per) + " |")
    for r in ("Path-A", "NeSy"):
        per = np.mean(res[r]["per"], 0)
        print(f"| {r} | {C.ms(res[r]['acc'])} | {C.ms(res[r]['mf1'])} | " +
              " | ".join(f"{x:.3f}" for x in per) + " |")
    print("\n**Induced MMHS hate rules per class** (learned from train+val):")
    for hc in HATE:
        if rs[hc]:
            for i, cl in enumerate(rs[hc], 1):
                print(f"- {hc}-{i}: " + " AND ".join(names[j] for j in cl["idx"]) +
                      f"  (P={cl.get('precision', 0):.2f})")
        else:
            print(f"- {hc}: (no rule met support/precision guardrails)")
    print("\n_Native single-label 6-way task; NotHate is the rule-free default. Macro-F1 is "
          "over all 6 classes and is depressed by rare classes (e.g. Religion ~24 test). "
          "Binary sexist-vs-rest reference: zero-shot NeSy 0.645/0.460._")


if __name__ == "__main__":
    main()
