"""Table 22 - neural-branch scaling grid (strawman-baseline control).

Does the neuro-symbolic margin survive a stronger neural branch?  2x2 grid:

  encoder    in { CLIP ViT-B/32 (deployed), CLIP ViT-L/14 }
  classifier in { linear probe (deployed), 1-hidden-layer MLP (512) }

For ViT-L/14 the ENTIRE pipeline is re-learned in the new embedding space:
fresh image/OCR/caption embeddings, affect head retrained (5-fold OOF for
train rows, replicating the deployed protocol), the multimodal concept bank
re-built, thresholds and rules re-induced.  The decision layer is the deployed
Classifier + policy retrain under the 11-seed protocol per cell; the novel
slice is the canonical one (B/32 caption cosine < 0.90) in all cells so the
column stays comparable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from types import SimpleNamespace

import numpy as np

import common as C
from nesymis import config
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import rule_induction
from nesymis.fusion.decision_layer import LinearEvidencePolicy

NOVEL_COS = 0.90
L14 = "openai/clip-vit-large-patch14"
DEEP_HIDDEN = 512
PRED_SINK = None    # set to a dict to collect {tag: [(Path-A pred, NeSy pred) per seed]} on the test rows


def _clf_cfg(hidden):
    cfg = dict(config.MLP)
    cfg["hidden"] = hidden
    return cfg


def _l14_embeddings(df):
    """Fresh ViT-L/14 embeddings for all rows: (v, u_ocr, u_caption)."""
    from nesymis.encoders import clip_encoder as ce
    config.CLIP_MODEL = L14
    ce._model = ce._proc = None                                # force reload
    paths = df["path"].astype(str).tolist()
    ocr = df["text_ocr"].fillna("").astype(str).tolist()
    cap = df["text_caption"].fillna("").astype(str).tolist()
    v = ce.encode_images(paths)
    u_ocr = ce.encode_texts(ocr)
    u_cap = ce.encode_texts(cap)
    print(f"  [t22] L/14 embeddings done v{v.shape} u{u_ocr.shape}", flush=True)
    return v, u_ocr, u_cap


def _induce(df, y, trainval, u_cap, v):
    emb_ns = SimpleNamespace(df=df, text_caption=u_cap, image=v)
    bank = cl.build(emb_ns, trainval)                        # concept bank in the L/14 space
    cont = bank.frame(u_cap, v, index=df.index)
    rs, L, _ = rule_induction.induce(cont, y, trainval, affect=None, save=False)
    rd = rule_induction.evaluate(rs, L, df.index)
    fired = np.stack([rd[f"{cn}_fired"].to_numpy() for cn in config.STEREO_CLASSES], 1)
    rprec = C.rules.rule_precision(rd.loc[trainval], y[trainval])
    rvec = np.array([rprec[cn] for cn in config.STEREO_CLASSES], np.float32)
    pb = C.rules.rule_only_prediction(rd).to_numpy()
    print("  [t22] L/14 concept bank + rules induced", flush=True)
    return fired, rvec, pb


def run_cell(tag, vu, D, fired, rvec, hidden, c, slices):
    y, tr, va, te, trainval, df = c.y, c.tr, c.va, c.te, c.trainval, c.df
    y_te = y[te]
    ctor = lambda d: LinearEvidencePolicy(d)
    cfg = _clf_cfg(hidden)
    # per_class is additive -- existing consumers read only acc/macro_f1. It exists
    # because a flat accuracy with a rising macro-F1 can only mean the rare classes
    # moved, and that has to be visible per class to be a claim.
    res = {r: {k: {"acc": [], "macro_f1": [], "per_class": []} for k in slices}
           for r in ("Path-A", "NeSy")}
    for seed in C.SEEDS11:
        clf, logits = clf.crossfit_logits(vu, D, y, tr, va, fusion=config.NEURAL_FUSION,
                                          affect_dim=C.odesign.AFFECT_DIM, cfg=cfg, seed=int(seed))
        state = C.rl.build_state(logits, fired, rvec, D)
        pol = C.rl.train_rlvr(logits, state, y, tr, va, seed=int(seed), policy_ctor=ctor)[0]
        ne = C.rl.greedy_pred(pol, logits[te], state[te])
        pa = logits[te].argmax(1)
        if PRED_SINK is not None:
            PRED_SINK.setdefault(tag, []).append((pa, ne))
        for k, m in slices.items():
            for row, pred in (("Path-A", pa), ("NeSy", ne)):
                mm = C.compute_metrics(y_te[m], pred[m])
                res[row][k]["acc"].append(mm["acc"])
                res[row][k]["macro_f1"].append(mm["macro_f1"])
                res[row][k]["per_class"].append(mm["per_class_f1"])
        print(f"  [t22] {tag} seed {seed} done", flush=True)
    # crash-safe: emit the cell's numbers as soon as they exist
    print(f"  [t22] CELL {tag}: "
          f"Path-A full {C.ms(res['Path-A']['full']['acc'])} / {C.ms(res['Path-A']['full']['macro_f1'])} | "
          f"NeSy full {C.ms(res['NeSy']['full']['acc'])} / {C.ms(res['NeSy']['full']['macro_f1'])} | "
          f"NeSy novel {C.ms(res['NeSy']['novel']['acc'])} / {C.ms(res['NeSy']['novel']['macro_f1'])}",
          flush=True)
    return res


def generate(ctx=None):
    c = ctx or C.build_context()
    y, te, trainval, df = c.y, c.te, c.trainval, c.df
    U = c.emb.text_caption
    novel = (U[te] @ U[trainval].T).max(1) < NOVEL_COS
    slices = {"full": np.ones(int(te.sum()), bool), "novel": novel}

    cells = {}
    # --- B/32 cells: cached embeddings, deployed rules ---
    cells[("B/32", "linear")] = run_cell("B32-linear", c.vu, c.D, c.fired, c.rprec_vec,
                                         None, c, slices)
    cells[("B/32", "MLP-512")] = run_cell("B32-deep", c.vu, c.D, c.fired, c.rprec_vec,
                                          DEEP_HIDDEN, c, slices)

    # --- L/14 cells: full re-learn in the new embedding space ---
    v, u_ocr, u_cap = _l14_embeddings(df)
    vu14 = np.concatenate([v, u_ocr], axis=1).astype(np.float32)
    D14 = np.zeros((len(df), config.AFFECT_DIM), np.float32)   # affect-free
    fired14, rvec14, pb14 = _induce(df, y, trainval, u_cap, v)
    pb_m = {k: C.compute_metrics(y[te][m], pb14[te][m]) for k, m in slices.items()}
    cells[("L/14", "linear")] = run_cell("L14-linear", vu14, D14, fired14, rvec14,
                                         None, c, slices)
    cells[("L/14", "MLP-512")] = run_cell("L14-deep", vu14, D14, fired14, rvec14,
                                          DEEP_HIDDEN, c, slices)

    md = ["## Table 22 - Neural-branch scaling: does the margin survive a stronger "
          "neural path? (11-seed protocol, linear evidence decision layer)\n",
          "| Encoder | Classifier | Path-A full Acc / F1 | NeSy full Acc / F1 | "
          "NeSy novel Acc / F1 |", "|---|---|---|---|---|"]
    for (enc, clf), res in cells.items():
        pa, ne = res["Path-A"], res["NeSy"]
        md.append(f"| {enc} | {clf} | {C.ms(pa['full']['acc'])} / {C.ms(pa['full']['macro_f1'])} | "
                  f"{C.ms(ne['full']['acc'])} / {C.ms(ne['full']['macro_f1'])} | "
                  f"{C.ms(ne['novel']['acc'])} / {C.ms(ne['novel']['macro_f1'])} |")
    md.append(f"\n_ViT-L/14 symbolic path fully re-induced; its rule-only Path-B scores "
              f"{pb_m['full']['acc']:.3f} / {pb_m['full']['macro_f1']:.3f} full and "
              f"{pb_m['novel']['acc']:.3f} / {pb_m['novel']['macro_f1']:.3f} novel "
              f"(B/32 deployed: 0.761 / 0.695 and 0.682 / 0.436). Novel slice fixed to the "
              f"canonical B/32 definition in all cells._")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    print(generate())
