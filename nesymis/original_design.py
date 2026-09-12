"""
reported result can be reproduced: fit once,
persist, reload for batch evaluation or live single-image inference.

Components (all inference-only over a frozen CLIP backbone):
  * neural Path-A'           : a gated 2-layer classifier over [v;u] -> 5 logits
  * DYNAMIC symbolic rules   : learned DNF over a multimodal CONCEPT bank
                               (text-phrase + image-scene anchors, gap-corrected
                               and clustered into named concept prototypes that fire
                               on the caption OR the image) -- concepts, thresholds,
                               and rule structure all induced from train+val
  * decision policy          : linear evidence layer, verifiable-reward bootstrap

This deployed model is AFFECT-FREE. A systematic study (experiments/affect_*.py) shows
affective signals -- learned affect head, curated appraisal axes, induced cross-modal
incongruity, and learned bilinear interactions -- add nothing beyond the induced
content concepts (affect is entangled with content in memes), so no affect head is
used and no distilled/silver labels are required. The classifier keeps its capacity
with the (inert) affect-fusion slot fed zeros.

fit()  trains classifier + rules + policy and saves to artifacts/weights/.
load() restores them (auto-fits if any checkpoint is missing).
predict_image() runs the LIVE path on a raw image (EasyOCR -> CLIP -> neural ->
rules -> policy), returning the label + interpretable concept evidence.

Shared by train.py / test.py / infer.py and the Results scripts -> all use the
identical model, so results stay consistent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nesymis import config
from nesymis.data import dataset as ds
from nesymis.fusion import decision_layer as rl
from nesymis.neural import classifier as clf
from nesymis.symbolic import concept_layer as cl
from nesymis.symbolic import affect_predicates as ap
from nesymis.symbolic import rule_induction, rules

_dev = "cuda" if torch.cuda.is_available() else "cpu"
AFFECT_DIM = 6                     # 6-d affect head outputs (the adopted setting)
FUSION_DEFAULT = "plain"       # deployed affect-free: plain MLP over [v;u] (no FiLM gate)
_OCR_READER = None             # lazily-built EasyOCR singleton (reused across images)


def best_fusion() -> str:
    """Fusion block selected by the E1 grid (artifacts/affect_best_fusion.json), else gate."""
    p = config.ARTIFACTS_DIR / "affect_best_fusion.json"
    if p.exists():
        try:
            return json.loads(p.read_text())["fusion"]
        except Exception:  # noqa: BLE001
            pass
    return FUSION_DEFAULT


def _abs_path(path: str) -> str:
    """Absolute path string (open_rgb/config.abspath then resolve correctly)."""
    return path if os.path.isabs(path) else os.path.abspath(path)


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _OCR_READER


def _live_ocr(abs_image_path: str) -> str:
    from nesymis.data.imageio import open_rgb
    from nesymis.data.ocr_text import _prep, _read
    return _read(_get_ocr_reader(), _prep(open_rgb(abs_image_path)))


# --------------------------------------------------------------------------- #
# The runnable system
# --------------------------------------------------------------------------- #
@dataclass
class OriginalNeSy:
    clf: clf.Classifier            # neural Path-A' (gated classifier over [v;u]; affect slot inert)
    head: object                         # affect head -- None in the affect-free deployed model
    policy: object                       # decision layer (LinearEvidencePolicy deployed; PolicyNet legacy)
    rprec: dict                          # per-class learned-rule precision (train+val)
    rprec_vec: np.ndarray                # rprec as a (4,) vector over STEREO_CLASSES
    fusion: str
    rule_sets: dict                      # learned DNF rules per class (readable literals over concepts)
    bank: cl.ConceptBank                 # deployed multimodal concept-abstraction grounding
    gd_threshold: float = 0.5            # legacy gender_directed threshold (unused when affect_free)
    policy_type: str = "linear_bootstrap"     # deployed: linear evidence layer, correctness bootstrap
    affect_free: bool = True             # deployed default: no affect signal (entanglement finding)

    def _emo(self, n: int, D=None) -> np.ndarray:
        """Affect input: zeros when affect-free (the classifier keeps its capacity via
        the inert fusion slot); otherwise the supplied 6-d vector (study/legacy only)."""
        return np.zeros((n, AFFECT_DIM), np.float32) if (self.affect_free or D is None) else D

    # ---- batch prediction over precomputed features (used by evaluation) ----
    def neural_logits(self, vu: np.ndarray, D: np.ndarray = None) -> np.ndarray:
        D = self._emo(len(vu), D)
        return clf.predict_logits(self.clf, vu.astype(np.float32), D.astype(np.float32))

    def policy_pred(self, neural_logits: np.ndarray, fired_mat: np.ndarray, D: np.ndarray = None) -> np.ndarray:
        state = rl.build_state(neural_logits, fired_mat, self.rprec_vec, self._emo(len(neural_logits), D))
        return rl.greedy_pred(self.policy, neural_logits, state)

    def _concept_features(self, text_emb, image_emb, D=None, index=None):
        """Continuous feature frame: concept scores (fired by caption OR image). Affect
        columns are added only for a legacy affect-ful system; affect-free rules use none."""
        cont = self.bank.frame(text_emb, image_emb, index=index)
        if not self.affect_free and D is not None:
            for j, nm in enumerate(config.AFFECT_FEATURES):
                cont[f"emo:{nm}"] = D[:, j]
        return cont

    def symbolic_eval(self, emb, D: np.ndarray = None) -> pd.DataFrame:
        """Learned-rule evaluation for all rows of `emb` (rules.evaluate schema)."""
        cont = self._concept_features(emb.text_caption, emb.image, D, index=emb.df.index)
        return rule_induction.evaluate_features(self.rule_sets, cont, None)

    # ---- live single-image inference (raw image -> label + evidence) ----
    def predict_image(self, image_path: str, ocr_text: str | None = None,
                      caption_text: str | None = None, run_ocr: bool = True) -> dict:
        from nesymis.encoders.clip_encoder import encode_images, encode_texts

        ap = _abs_path(image_path)
        v = encode_images([ap])                                   # (1, 512)
        if ocr_text is None:
            ocr_text = _live_ocr(ap) if run_ocr else ""
        caption_text = ocr_text if caption_text is None else caption_text
        u_cap = encode_texts([caption_text or ""])
        vu = np.concatenate([v, encode_texts([ocr_text or ""])], axis=1).astype(np.float32)

        logits = self.neural_logits(vu)                         # affect-free (affect slot = zeros)
        neural = int(logits.argmax(1)[0])

        cont = self._concept_features(u_cap, v)
        rd = rule_induction.evaluate_features(self.rule_sets, cont, None)
        fired_mat = np.stack([rd[f"{c}_fired"].to_numpy() for c in config.STEREO_CLASSES], 1)
        final = int(self.policy_pred(logits, fired_mat)[0])

        # readable clause-level evidence for the fired classes (all literals continuous concepts)
        fired_clauses = []
        for j, cl_name in enumerate(config.STEREO_CLASSES):
            if not fired_mat[0, j]:
                continue
            for clause in self.rule_sets[cl_name]:
                lits = [rule_induction.parse_literal(ln) for ln in clause["literals"]]
                vals = [cont[f].iloc[0] > cut for f, cut, _ in lits]
                vals = [(not v if neg else v) for v, (_, _, neg) in zip(vals, lits)]
                if all(vals):
                    fired_clauses.append(f"{cl_name}: {' AND '.join(clause['literals'])}")

        return {
            "label": config.IDX_TO_LABEL[final],
            "neural_label": config.IDX_TO_LABEL[neural],
            "policy_changed": final != neural,
            "fired_rules": list(rd.loc[0, "fired_rules"]),
            "fired_clauses": fired_clauses,
            "ocr_text": ocr_text or "",
        }


def fit(emb: ds.Embeddings | None = None, verbose: bool = False, persist: bool = True) -> OriginalNeSy:
    """Train the affect-free deployed model on the cached embeddings."""
    if emb is None:
        emb = ds.load_embeddings()
    vu = emb.features("both").astype(np.float32)
    y = emb.labels
    tr, va = emb.split_mask("train"), emb.split_mask("val")
    trainval = emb.df["split"].isin(["train", "val"]).to_numpy()
    fusion = config.NEURAL_FUSION            # deployed: "plain" MLP over [v;u], no affect, no gating

    # AFFECT-FREE: affect is dropped (the entanglement study finds it non-contributing).
    # With fusion="plain" the neural branch is a plain MLP over the frozen [v;u]; the
    # affect argument is ignored, so no affect head or distilled/silver labels are needed.
    D = np.zeros((len(y), AFFECT_DIM), np.float32)
    # `logits` carries out-of-fold values on train+val and the full-data model's
    # values on test, so the decision layer below is fitted on neural evidence
    # that still contains the branch's real errors (see clf.crossfit_logits).
    clf, logits = clf.crossfit_logits(vu, D, y, tr, va, fusion=fusion, affect_dim=AFFECT_DIM,
                                     verbose=verbose)

    # deployed symbolic grounding: multimodal concept-abstraction layer -> learned DNF
    # rules over concepts (no affect literals; each concept fires on the caption OR the image)
    bank = cl.build(emb, trainval)
    cont = bank.frame(emb.text_caption, emb.image, index=emb.df.index)
    rule_sets, L, _ = rule_induction.induce(cont, y, trainval, affect=None, save=True)
    rule_df = rule_induction.evaluate(rule_sets, L, emb.df.index)
    rprec = rules.rule_precision(rule_df.loc[trainval], y[trainval])
    rprec_vec = np.array([rprec[c] for c in config.STEREO_CLASSES], np.float32)
    fired = np.stack([rule_df[f"{c}_fired"].to_numpy() for c in config.STEREO_CLASSES], 1)

    # decision layer: linear evidence policy, verifiable-reward bootstrap only.
    state = rl.build_state(logits, fired, rprec_vec, D)
    policy, pf = rl.train_rlvr(logits, state, y, tr, va, verbose=verbose,
                               policy_ctor=rl.LinearEvidencePolicy)

    system = OriginalNeSy(clf=clf, head=None, policy=policy, rprec=rprec, rprec_vec=rprec_vec,
                         fusion=fusion, rule_sets=rule_sets, bank=bank,
                         gd_threshold=0.5, policy_type="linear_bootstrap", affect_free=True)
    if persist:
        save(system)
        # keep the exact logits the decision layer was fitted on, so control
        # experiments train their policies on the same footing (see table17)
        config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.OOF_LOGITS_PATH, logits)
    if verbose:
        print(f"[affect-pipeline] fit done (AFFECT-FREE) - fusion={fusion}  "
              f"val bootstrap-F1={pf:.3f} (linear evidence layer)")
    return system


def save(s: OriginalNeSy) -> None:
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": s.clf.state_dict(),
                "affect_raw_dim": s.clf.proj_e.in_features,
                "affect_dim": s.clf.proj_e.out_features,
                "fusion": s.fusion, "hidden": config.MLP["hidden"]},
               config.AFFECT_CLF_PATH)
    if isinstance(s.policy, rl.LinearEvidencePolicy):
        pol_ck = {"state_dict": s.policy.state_dict(), "arch": "linear", "in_dim": 20}
    else:
        pol_ck = {"state_dict": s.policy.state_dict(), "arch": "mlp",
                  "in_dim": s.policy.body[0].in_features,
                  "hidden": s.policy.body[0].out_features}
    torch.save(pol_ck, config.POLICY_PATH)
    config.AFFECT_META_PATH.write_text(json.dumps(
        {"fusion": s.fusion, "rprec": s.rprec, "rprec_vec": s.rprec_vec.tolist(),
         "gd_threshold": s.gd_threshold, "policy_type": s.policy_type,
         "affect_free": s.affect_free}, indent=2), encoding="utf-8")
    config.SYMBOLIC_RULES_PATH.write_text(json.dumps(
        {"rule_sets": {c: [{k: v for k, v in clause.items() if k != "idx"} for clause in rs]
                       for c, rs in s.rule_sets.items()}}, indent=1), encoding="utf-8")
    s.bank.to_files(config.CONCEPT_BANK_PATH, config.CONCEPT_BANK_META_PATH)


def _all_weights_exist() -> bool:
    return all(p.exists() for p in (config.AFFECT_CLF_PATH,
                                    config.POLICY_PATH, config.AFFECT_META_PATH,
                                    config.SYMBOLIC_RULES_PATH, config.CONCEPT_BANK_PATH,
                                    config.CONCEPT_BANK_META_PATH))


def load(verbose: bool = False) -> OriginalNeSy:
    """Restore the persisted complete model; fit + persist it once if anything is missing."""
    if not _all_weights_exist():
        if verbose:
            print("[affect-pipeline] no saved system found - fitting once on cached embeddings ...")
        return fit(verbose=verbose)

    meta = json.loads(config.AFFECT_META_PATH.read_text())
    ck = torch.load(config.AFFECT_CLF_PATH, map_location=_dev)
    clf = clf.Classifier(ck["affect_raw_dim"], fusion=ck["fusion"], affect_dim=ck["affect_dim"],
                               hidden=ck["hidden"]).to(_dev)
    clf.load_state_dict(ck["state_dict"]); clf.eval()

    pk = torch.load(config.POLICY_PATH, map_location=_dev)
    if pk.get("arch") == "linear":
        policy = rl.LinearEvidencePolicy(pk["in_dim"]).to(_dev)
    else:
        policy = rl.PolicyNet(pk["in_dim"], hidden=pk["hidden"]).to(_dev)
    policy.load_state_dict(pk["state_dict"]); policy.eval()

    sym = json.loads(config.SYMBOLIC_RULES_PATH.read_text(encoding="utf-8"))
    bank = cl.ConceptBank.from_files(config.CONCEPT_BANK_PATH, config.CONCEPT_BANK_META_PATH)

    ptype = meta.get("policy_type", "bootstrap")
    if verbose:
        print(f"[affect-pipeline] loaded saved system from {config.WEIGHTS_DIR}  "
              f"(fusion={meta['fusion']}, policy={ptype.upper()})")
    return OriginalNeSy(clf=clf, head=None, policy=policy, rprec=meta["rprec"],
                       rprec_vec=np.array(meta["rprec_vec"], np.float32), fusion=meta["fusion"],
                       rule_sets=sym["rule_sets"], bank=bank,
                       gd_threshold=float(meta.get("gd_threshold", 0.5)), policy_type=ptype,
                       affect_free=bool(meta.get("affect_free", True)))


if __name__ == "__main__":
    fit(verbose=True)
