"""External #3b - EXIST 2024 Memes, Task 6 (sexism categorization) as a native
6-class framework-generalization test, parallel to the MMHS 6-class run.

EXIST Task 6 is multi-LABEL; our framework is single-label, so we collapse each
meme to one class:
  * Task 4 majority NO  -> class 0 (not_sexist; rule-free default),
  * Task 4 majority YES -> the most-voted Task-6 category among the 5 sexism
    types (deterministic tie-break by fixed order),
  * annotator ties / sexist-without-category -> dropped.
This discards secondary sexism types (a meme can carry several), which
under-represents multi-type memes -- stated openly.

Classes (single-label 6-way):
  0 not_sexist  1 ideological  2 stereotyping  3 objectification
  4 sexual_violence  5 misogyny_nonsexual        (1..5 = rule classes)

Everything is re-learned from a fresh stratified 70/15/15 split of the English
memes: per-class predicate mining, per-class DNF induction, a 6-class classifier
(reused local trainer; deployed 5-class classifier untouched), and the linear
decision layer at n_classes=6. Affect head frozen. 11-seed protocol; symbolic
path deterministic. Shares the CLIP cache with data_exist2024.py.
"""
import os
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import common as C
from nesymis import config
from nesymis.neural import classifier as clf
from nesymis.symbolic import predicate_pool, rule_induction
from data_exist2024 import load_all_en, embed
from data_mmhs150k import train_clf6

CAT6 = ["IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE", "OBJECTIFICATION",
        "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"]                      # -> classes 1..5
NAMES = {0: "not_sexist", 1: "ideological", 2: "stereotyping", 3: "objectification",
         4: "sexual_violence", 5: "misogyny_nonsexual"}
HATE = [NAMES[i] for i in range(1, 6)]
HATE_IDX = [1, 2, 3, 4, 5]


def label6(v):
    t4 = Counter(x.upper() for x in v["labels_task4"])
    yes, no = t4.get("YES", 0), t4.get("NO", 0)
    if no > yes:
        return 0                                      # not sexist -> default class
    if yes <= no:
        return -1                                     # sexist/not tie -> drop
    cnt = Counter()
    for ann in v["labels_task6"]:
        for cc in ann:
            if cc in CAT6:
                cnt[cc] += 1
    if not cnt:
        return -1                                     # sexist but no category -> drop
    top = max(CAT6, key=lambda cc: (cnt.get(cc, 0), -CAT6.index(cc)))
    return CAT6.index(top) + 1


def main():
    ids, texts, paths, raw = load_all_en()
    lab = np.array([label6(v) for v in raw])
    keep = lab >= 0
    c = C.build_context()
    from nesymis.affect import head as affect_head
    vu, u = embed(ids, paths, texts)                  # shared cache with the binary script
    D = affect_head.predict_affect(c.system.head, vu)

    vu, u, D = vu[keep], u[keep], D[keep]
    texts = [t for t, m in zip(texts, keep) if m]
    y = lab[keep]
    dist = {NAMES[k]: int((y == k).sum()) for k in range(6)}
    print("## EXIST 2024 Memes - Task 6 (native 6-class sexism categorization, English)\n")
    print(f"n={len(y)} (of {len(ids)} EN memes; {int((~keep).sum())} dropped as tie/"
          f"uncategorized); class distribution {dist}\n")

    # fresh stratified 70/15/15 (EXIST test gold withheld)
    idx = np.arange(len(y))
    tr_i, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=config.SEED)
    va_i, te_i = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=config.SEED)
    tr = np.isin(idx, tr_i); va = np.isin(idx, va_i); te = np.isin(idx, te_i)
    trainval = tr | va

    for k, nm in NAMES.items():
        config.LABEL_TO_IDX.setdefault(nm, k)
    labels = np.array([NAMES[int(v)] for v in y])
    emb_ns = SimpleNamespace(df=pd.DataFrame({"text_caption": texts, "label": labels}),
                             text_caption=u)
    pool_df, lex, lexn = predicate_pool.build_pool(emb_ns, classes=HATE, calib_mask=trainval, save=False)
    rs, L, names = rule_induction.induce(pool_df, y, trainval, affect=D, classes=HATE,
                                         extra_bool=(lexn, lex), save=False)

    R = len(HATE)
    fired = np.zeros((len(y), R), np.float32)
    trig = np.zeros((len(y), R), np.float32)
    rvec = np.zeros(R, np.float32)
    for j, hc in enumerate(HATE):
        for cl in rs[hc]:
            fj = np.all(L[:, cl["idx"]], axis=1)
            fired[:, j] = np.maximum(fired[:, j], fj.astype(np.float32))
            trig[:, j] = np.where(fj, np.maximum(trig[:, j], cl["precision"]), trig[:, j])
        m = trainval & (fired[:, j] > 0)
        rvec[j] = float((y[m] == HATE_IDX[j]).mean()) if m.any() else 0.0
    any_f = fired.any(1)
    pb_all = np.where(any_f, np.array(HATE_IDX)[trig.argmax(1)], 0)

    yte = y[te]
    lab6 = list(range(6))
    mc = lambda p: (float((p == yte).mean()),
                    f1_score(yte, p, labels=lab6, average="macro", zero_division=0))
    ctor = lambda d: C.rl.LinearEvidencePolicy(d, n_classes=6, rule_classes=HATE_IDX)

    res = {r: {"acc": [], "mf1": [], "per": []} for r in ("Path-A", "NeSy")}
    nseed = len(C.SEEDS11)
    for si, seed in enumerate(C.SEEDS11, 1):
        print(f"  [exist-t6] seed {si}/{nseed} starting ...", flush=True)
        clf = train_clf6(vu, D, y, tr, va, C.odesign.AFFECT_DIM, int(seed))
        logits = clf.predict_logits(clf, vu, D)
        state = C.rl.build_state(logits, fired, rvec, D)
        pol = C.rl.train_rlvr(logits, state, y, tr, va, seed=int(seed), policy_ctor=ctor)[0]
        ne = C.rl.greedy_pred(pol, logits[te], state[te])
        for r, p in (("Path-A", logits[te].argmax(1)), ("NeSy", ne)):
            a, mf = mc(p)
            res[r]["acc"].append(a); res[r]["mf1"].append(mf)
            res[r]["per"].append(f1_score(yte, p, labels=lab6, average=None, zero_division=0))
        print(f"  [exist-t6] seed {si}/{nseed} done -> "
              f"Path-A {res['Path-A']['acc'][-1]:.3f}/{res['Path-A']['mf1'][-1]:.3f}  "
              f"NeSy {res['NeSy']['acc'][-1]:.3f}/{res['NeSy']['mf1'][-1]:.3f}", flush=True)

    pb_a, pb_m = mc(pb_all[te])
    pb_per = f1_score(yte, pb_all[te], labels=lab6, average=None, zero_division=0)
    hdr = " | ".join(f"F1-{NAMES[k]}" for k in range(6))
    print(f"\n### Full 6-class re-induction + retraining (70/15/15; test n={int(te.sum())})\n")
    print(f"| Path | Acc | Macro-F1 | {hdr} |")
    print("|" + "---|" * 9)
    print(f"| Path-B (rules) | {pb_a:.3f} | {pb_m:.3f} | " + " | ".join(f"{x:.3f}" for x in pb_per) + " |")
    for r in ("Path-A", "NeSy"):
        per = np.mean(res[r]["per"], 0)
        print(f"| {r} | {C.ms(res[r]['acc'])} | {C.ms(res[r]['mf1'])} | " +
              " | ".join(f"{x:.3f}" for x in per) + " |")
    print("\n**Induced EXIST sexism-type rules per class** (train+val):")
    for hc in HATE:
        if rs[hc]:
            for i, cl in enumerate(rs[hc], 1):
                print(f"- {hc}-{i}: " + " AND ".join(names[j] for j in cl["idx"]) +
                      f"  (P={cl.get('precision',0):.2f})")
        else:
            print(f"- {hc}: (no rule met support/precision guardrails)")
    print("\n_Native single-label 6-way (multi-label collapsed to top-voted type). Macro-F1 over "
          "all 6 classes; rare sexism types may induce no rule. Task-4 binary reference in "
          "data_exist2024.py._")


if __name__ == "__main__":
    main()
