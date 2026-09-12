"""The combined paper's system: a neural classifier and a concept-routed decision, emitting
symbolic evidence for every output, under the fair protocol.

    meme image --> OCR --> meme text            (the same text feeds both paths)
         |                  |
      CLIP image         CLIP text              frozen ViT-B/32
         +------ v, u ------+
         |                  |
    neural path        symbolic path
    MLP over [v;u]     30 concept scores from (u, v)
         |               |- concept scorecard   --> part of the decision
         |               '- rules, 5 classes    --> supporting evidence
         '---- decision: pooled(concept scorecard, neural) ----'
                              |
       label + neural confidence + the concepts that produced it
             + fired rules + evidence label (supported / mixed / contested / none)

It differs from nesymis/original_design.py -- the ORIGINAL design -- in three ways, each forced by
the audit of the original design:

  1. One text channel for both paths: the OCR of the image, from one pipeline for every row
     (the corrected benchmark). The original routed the scraped post caption to the symbolic
     path only, and that asymmetry, not the architecture, produced its reported margin.
  2. The decision is a pooled model of a concept scorecard and the neural classifier
     (nesymis/fusion/concept_pool.py). The original added rule bonuses to the neural logits.
  3. Rules exist for all five classes and are reported as evidence, not decision inputs.

original_design.py is kept unchanged: the paper reports the original design's own result
before the controlled test that explains it.

    s, oof_logits = fit(seed=42, persist=True)
    s = load()
    rec = s.predict_image("path/to/meme.jpg")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nesymis import config
from nesymis.fusion import concept_pool as cp
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import rule_induction as ri

_dev = "cuda" if torch.cuda.is_available() else "cpu"
AFFECT_DIM = config.AFFECT_DIM      # the inert affect slot of the classifier; fed zeros throughout


# --------------------------------------------------------------------------- #
# Data: the corrected benchmark
# --------------------------------------------------------------------------- #
def load_data() -> SimpleNamespace:
    """Corrected benchmark: manifest + uniform-pipeline OCR embeddings, row-aligned with the
    cached image embeddings (asserted)."""
    df = pd.read_csv(config.CORRECTED_MANIFEST_PATH).fillna("")
    ids = json.loads(config.EMB_IDS_PATH.read_text(encoding="utf-8"))
    assert df["id"].astype(str).tolist() == [str(i) for i in ids], "row order drifted"
    v = np.load(config.IMAGE_EMB_PATH).astype(np.float32)
    u = np.load(config.TEXT_OCR_CORRECTED_EMB_PATH).astype(np.float32)
    sp = df["split"].to_numpy()
    return SimpleNamespace(df=df, v=v, u=u, y=df["label_idx"].to_numpy().astype(int),
                           tr=sp == "train", va=sp == "val", te=sp == "test",
                           text=df["text_ocr"].astype(str))


def _classes(d) -> list:
    """Class names in label-index order: the data's own (`d.classes`), else the NeSy-MIS task's."""
    return list(getattr(d, "classes", None) or config.LABELS)


def _l2i(classes) -> dict:
    return {c: i for i, c in enumerate(classes)}


def build_symbolic(d: SimpleNamespace):
    """Concept bank and rules for every class, on the OCR channel, from train+val only.
    Seed-independent, so evaluations build it once and pass it to every fit(). Any task works:
    `d.classes` names the labels (default: the NeSy-MIS five), `d.df["label"]` holds the names,
    and `d.mine_classes` picks the classes whose phrases are mined (default: the four
    stereotype classes, as in the original design)."""
    trainval = d.tr | d.va
    classes = _classes(d)
    df2 = d.df.copy()
    df2["text_caption"] = d.text.values         # concept_layer mines phrases from this column
    bank = cl.build(SimpleNamespace(df=df2, text_caption=d.u, image=d.v), trainval,
                    classes=getattr(d, "mine_classes", None))
    cont = bank.frame(d.u, d.v, index=d.df.index)
    rs, _, _ = ri.induce(cont, d.y, trainval, affect=None, classes=classes, save=False,
                         label_to_idx=_l2i(classes))
    rule_sets = {c: [{k: val for k, val in clause.items() if k != "idx"} for clause in r]
                 for c, r in rs.items()}
    return bank, cont, rule_sets


# --------------------------------------------------------------------------- #
# The system
# --------------------------------------------------------------------------- #
@dataclass
class EvidenceNeSy:
    bank: cl.ConceptBank
    rule_sets: dict
    clf: torch.nn.Module
    card: cp.ConceptScorecard
    w: float
    meta: dict = field(default_factory=dict)
    classes: list = field(default_factory=lambda: list(config.LABELS))

    def neural_logits(self, vu: np.ndarray) -> np.ndarray:
        D = np.zeros((len(vu), AFFECT_DIM), np.float32)
        return clf.predict_logits(self.clf, vu.astype(np.float32), D)

    def decide(self, vu, u, v, index=None, neural_logits=None) -> dict:
        """Pooled decision for a batch, with everything the evidence record needs."""
        cont = self.bank.frame(u, v, index=index)
        cont = cont[self.card.names]                      # scorecard column order
        Z = self.card.standardise(cont.to_numpy(np.float64))
        logits = self.neural_logits(vu) if neural_logits is None else neural_logits
        ln = cp.log_softmax(logits)
        out = cp.decompose(self.card, Z, ln, self.w)
        fired, trig, clauses = cp.fire_rules(self.rule_sets, cont, self.classes, _l2i(self.classes))
        states, suspicion = cp.evidence_state(out["pred"], fired, trig)
        out.update(Z=Z, ln=ln, neural_logits=np.asarray(logits), fired=fired, trig=trig,
                   clauses=clauses, states=states, suspicion=suspicion)
        return out

    def records(self, out: dict, rows=None, topn: int = 3) -> list:
        """One evidence record per row: the label, both confidences, the concepts that pushed
        the decision toward the label and away from the runner-up (exact contributions), the
        fired rules, and the evidence label."""
        rows = range(len(out["pred"])) if rows is None else rows
        names = [cp.concept_label(n) for n in self.card.names]
        pooled = cp.softmax(out["scores"])
        neural = cp.softmax(out["neural_logits"])
        recs = []
        for i in rows:
            k, r = int(out["pred"][i]), int(out["runner_up"][i])
            c = out["contrib"][i]
            order = np.argsort(-c)
            recs.append({
                "label": self.classes[k],
                "runner_up": self.classes[r],
                "confidence": float(pooled[i, k]),
                "neural_label": self.classes[int(neural[i].argmax())],
                "neural_confidence": float(neural[i].max()),
                "concept_share_of_decision": float(out["concept_share"][i]),
                "concepts_for": [(names[j], round(float(c[j]), 3)) for j in order[:topn] if c[j] > 0],
                "concepts_against": [(names[j], round(float(c[j]), 3))
                                     for j in order[::-1][:topn] if c[j] < 0],
                "rules_fired": out["clauses"][i],
                "evidence": str(out["states"][i]),
            })
        return recs

    def predict_image(self, image_path: str, ocr_text: str | None = None) -> dict:
        """Live path: OCR -> CLIP -> both experts -> pooled decision -> evidence record."""
        from nesymis.original_design import _abs_path, _live_ocr
        from nesymis.encoders.clip_encoder import encode_images, encode_texts
        ap = _abs_path(image_path)
        v = encode_images([ap])
        if ocr_text is None:
            ocr_text = _live_ocr(ap)
        u = encode_texts([ocr_text or ""])
        vu = np.concatenate([v, u], 1).astype(np.float32)
        rec = self.records(self.decide(vu, u, v), [0])[0]
        rec["ocr_text"] = ocr_text or ""
        return rec


# --------------------------------------------------------------------------- #
# Cross-validated operating-point selection
# --------------------------------------------------------------------------- #
def concept_oof(d: SimpleNamespace, C: float, seed: int, folds: int = 5,
                rebuild_bank: bool = True, S_full=None, groups=None) -> np.ndarray:
    """Out-of-fold concept log-probabilities for every train+val row, in row order.

    Why the bank is rebuilt inside each fold: concept mining ranks phrases by per-class
    log-odds over the rows it is given, i.e. it uses their LABELS. A scorecard evaluated on
    rows that helped mine its own concepts looks better than it is, which would push the
    operating point toward concept-heavy weights. rebuild_bank=False (reuse the train+val
    bank, S_full) is kept only to measure that leak.

    `groups` (one id per row of the corpus, e.g. near-duplicate clusters) keeps every group
    inside one fold, so that a fold is not scored on near-copies of the memes it was fitted on.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    tv = np.flatnonzero(d.tr | d.va)
    n = len(d.y)
    k = len(_classes(d))
    out = np.zeros((len(tv), k))
    if groups is None:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        split_iter = skf.split(tv, d.y[tv])
    else:
        sgk = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        split_iter = sgk.split(tv, d.y[tv], np.asarray(groups)[tv])
    for fit_pos, out_pos in split_iter:
        fit_rows, out_rows = tv[fit_pos], tv[out_pos]
        fm = np.zeros(n, bool)
        fm[fit_rows] = True
        om = np.zeros(n, bool)
        om[out_rows] = True
        if rebuild_bank:
            df2 = d.df.copy()
            df2["text_caption"] = d.text.values
            bank = cl.build(SimpleNamespace(df=df2, text_caption=d.u, image=d.v), fm,
                            classes=getattr(d, "mine_classes", None))
            S = bank.frame(d.u, d.v, index=d.df.index).to_numpy(np.float64)
        else:
            S = S_full
        card = cp.ConceptScorecard.fit(S, [str(j) for j in range(S.shape[1])], d.y, fm, om, Cs=(C,),
                                       k=k)
        out[out_pos] = cp.log_softmax(card.logits(S[out_rows]))
    return out


def select_weight_cv(d: SimpleNamespace, logits: np.ndarray, C: float, seed: int, folds: int = 5,
                     acc_tol=0.01, rebuild_bank: bool = True, S_full=None, groups=None) -> tuple:
    """Operating point chosen on ALL train+val rows by cross-validation instead of one ~480-row
    validation split, whose noise exceeded the 0.01 tolerance. Neural logits for these
    rows are already out-of-fold (crossfit_logits). `groups` keeps near-duplicate clusters inside
    one fold. Returns (w, concept OOF log-probs)."""
    tv = np.flatnonzero(d.tr | d.va)
    lc = concept_oof(d, C, seed, folds, rebuild_bank, S_full, groups=groups)
    w = cp.select_weight(lc, cp.log_softmax(logits[tv]), d.y[tv], acc_tol=acc_tol)
    return w, lc


# --------------------------------------------------------------------------- #
# Fit / persist / load
# --------------------------------------------------------------------------- #
def fit(seed: int = config.SEED, data=None, sym=None, persist: bool = False,
        prior_correct: bool = False, acc_tol=0.01, select: str = "cv", folds: int = 5, groups=None):
    """Fit the system. Returns (system, neural logits for every row: out-of-fold on train+val,
    full model on test) -- the out-of-fold train+val logits are what select the pooling weight.

    Operating point (the deployed choice): select="cv" picks w by 5-fold cross-validation over
    train+val with the concept bank rebuilt per fold (select_weight_cv), and acc_tol=0.01 makes
    it the most concept-weighted w within 0.01 of the neural classifier on BOTH macro-F1 and
    accuracy. Out of sample this retains macro-F1 and costs about 1.8 points of accuracy.
    select="val", acc_tol=None reproduces the earlier rule (one validation split, macro-F1 only).
    `prior_correct` shifts the scorecard to the training class prior (unstable; not deployed)."""
    d = data if data is not None else load_data()
    classes = _classes(d)
    k = len(classes)
    bank, cont, rule_sets = sym if sym is not None else build_symbolic(d)
    S = cont.to_numpy(np.float64)
    card = cp.ConceptScorecard.fit(S, list(cont.columns), d.y, d.tr, d.va, k=k)
    if prior_correct:
        card = card.with_prior_correction(d.y, d.tr)
    vu = np.concatenate([d.v, d.u], 1).astype(np.float32)
    D = np.zeros((len(d.y), AFFECT_DIM), np.float32)
    clf, logits = clf.crossfit_logits(vu, D, d.y, d.tr, d.va, fusion=config.NEURAL_FUSION,
                                     affect_dim=AFFECT_DIM, cfg=dict(config.MLP), seed=int(seed),
                                     num_classes=k)
    lc = cp.log_softmax(card.logits(S))
    ln = cp.log_softmax(logits)
    if select == "cv":
        w, _ = select_weight_cv(d, logits, card.C, seed, folds=folds, acc_tol=acc_tol, groups=groups)
    elif select == "val":
        w = cp.select_weight(lc[d.va], ln[d.va], d.y[d.va], acc_tol=acc_tol)
    else:
        raise ValueError(f"select must be 'cv' or 'val', not {select!r}")
    meta = {"seed": int(seed), "w": w, "scorecard_C": card.C, "n_concepts": len(card.names),
            "cv_groups": "near-duplicate clusters" if groups is not None else None,
            "prior_corrected": bool(prior_correct), "acc_tol": acc_tol, "select": select,
            "text_channel": getattr(d, "text_channel",
                                    "OCR of the image, uniform-OCR protocol (both paths)"),
            "val_macro_f1_neural": cp.macro_f1(d.y[d.va], ln[d.va].argmax(1), k),
            "val_macro_f1_pooled": cp.macro_f1(d.y[d.va],
                                               cp.pooled_scores(lc[d.va], ln[d.va], w).argmax(1), k)}
    if classes != list(config.LABELS):
        meta["classes"] = classes
    s = EvidenceNeSy(bank, rule_sets, clf, card, w, meta=meta, classes=classes)
    if persist:
        save(s)
    return s, logits


def save(s: EvidenceNeSy) -> None:
    p = config.EVIDENCE_WEIGHTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": s.clf.state_dict(), "affect_raw_dim": s.clf.proj_e.in_features,
                "affect_dim": s.clf.proj_e.out_features, "fusion": config.NEURAL_FUSION,
                "hidden": config.MLP["hidden"]}, p / "neural.pt")
    s.card.save(p / "scorecard.npz")
    s.bank.to_files(p / "concept_bank.npz", p / "concept_bank.json")
    (p / "rules.json").write_text(json.dumps({"rule_sets": s.rule_sets}, indent=1),
                                  encoding="utf-8")
    (p / "meta.json").write_text(json.dumps(s.meta, indent=2), encoding="utf-8")


def load() -> EvidenceNeSy:
    p = config.EVIDENCE_WEIGHTS_DIR
    need = ["neural.pt", "scorecard.npz", "concept_bank.npz", "concept_bank.json",
            "rules.json", "meta.json"]
    missing = [n for n in need if not (p / n).exists()]
    if missing:
        raise FileNotFoundError(f"{p} is missing {missing}; run fit(persist=True) first")
    ck = torch.load(p / "neural.pt", map_location=_dev)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    classes = meta.get("classes", list(config.LABELS))
    clf = clf.Classifier(ck["affect_raw_dim"], fusion=ck["fusion"], affect_dim=ck["affect_dim"],
                               num_classes=len(classes), hidden=ck["hidden"]).to(_dev)
    clf.load_state_dict(ck["state_dict"])
    clf.eval()
    return EvidenceNeSy(
        bank=cl.ConceptBank.from_files(p / "concept_bank.npz", p / "concept_bank.json"),
        rule_sets=json.loads((p / "rules.json").read_text(encoding="utf-8"))["rule_sets"],
        clf=clf, card=cp.ConceptScorecard.load(p / "scorecard.npz"), w=float(meta["w"]),
        meta=meta, classes=classes)
