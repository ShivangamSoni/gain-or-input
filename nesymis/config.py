"""
Central configuration for NeSy-MIS.

Single source of truth for paths, dataset construction parameters, the CLIP
backbone, the 5-class label space, and the symbolic prompts/thresholds. Every
other module imports from here so experiments stay reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path

# cuBLAS only honours this if it is set before the CUDA context is created, so it
# lives at config import -- every entry point imports config before touching a
# device. Without it, matmul reductions vary run to run and, because training
# keeps the best-val checkpoint, a float-level difference can select a different
# epoch and move a minority-class F1 visibly.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MANIFEST_PATH = ARTIFACTS_DIR / "manifest.csv"
OCR_CACHE_PATH = ARTIFACTS_DIR / "ocr_cache.json"
IMAGE_EMB_PATH = ARTIFACTS_DIR / "image_emb.npy"
# Two text embedding caches: OCR (noisy, feeds the neural path) and caption
# (clean, feeds the symbolic path's text grounding). This information split is
# what makes the two paths complementary -> NeSy can correct the neural path.
TEXT_OCR_EMB_PATH = ARTIFACTS_DIR / "text_emb_ocr.npy"
TEXT_CAPTION_EMB_PATH = ARTIFACTS_DIR / "text_emb_caption.npy"
EMB_IDS_PATH = ARTIFACTS_DIR / "emb_ids.json"          # row order of the .npy caches
RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
NESY_POLICY_PATH = ARTIFACTS_DIR / "nesy_policy.json"   # selected fusion policy + rule precision

# Complete model — trained weights persisted by nesymis.original_design and loaded by
# train.py / test.py / infer.py. Live in WEIGHTS_DIR (regenerable), so the whole
# trained system is co-located and rebuilt if any piece is missing.
AFFECT_CLF_PATH = WEIGHTS_DIR / "affect_clf.pt"                 # affect-aware neural Path-A'
AFFECT_HEAD_PATH = WEIGHTS_DIR / "affect_head.pt"               # affect head (live inference)
POLICY_PATH = WEIGHTS_DIR / "policy.pt"                           # decision policy
AFFECT_META_PATH = WEIGHTS_DIR / "nesymis_affect.json"        # fusion + rule precision + gd thr
SYMBOLIC_RULES_PATH = WEIGHTS_DIR / "symbolic_rules.json"         # learned DNF rules (over concepts+affect)
POOL_VECTORS_PATH = WEIGHTS_DIR / "pool_vectors.npz"              # sem:/ex: predicate vectors (legacy pool grounding)
CONCEPT_BANK_PATH = WEIGHTS_DIR / "concept_bank.npz"             # deployed grounding: concept centroids + gap centers
CONCEPT_BANK_META_PATH = WEIGHTS_DIR / "concept_bank.json"        # concept names + membership metadata

# Combined paper (branch 15Sep): the corrected benchmark -- one OCR pipeline for every row --
# and the evidence system trained on it (nesymis/concept_routed.py). The original design
# above is kept unchanged so its own result can be reported.
CORRECTED_MANIFEST_PATH = ARTIFACTS_DIR / "manifest_corrected.csv"
TEXT_OCR_CORRECTED_EMB_PATH = ARTIFACTS_DIR / "text_emb_ocr_corrected.npy"
EVIDENCE_WEIGHTS_DIR = WEIGHTS_DIR / "evidence"


def rel(path: Path) -> str:
    """Project-root-relative POSIX path (stable across machines / OSes)."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def abspath(relpath: str) -> Path:
    return PROJECT_ROOT / relpath


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Label space (fixed canonical order)
# --------------------------------------------------------------------------- #
LABELS = ["non_stereotype", "kitchen", "leadership", "working", "shopping"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABELS)

# Pretty names for tables/plots.
LABEL_DISPLAY = {
    "non_stereotype": "Non-Stereotype",
    "kitchen": "Kitchen",
    "leadership": "Leadership",
    "working": "Working",
    "shopping": "Shopping",
}

# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
# class name -> stereotype source folder under dataset/
STEREO_SOURCES = {
    "kitchen": "women_kitchen_memes_categorized",
    "leadership": "women_leadership_memes_categorized",
    "working": "working_women_memes_categorized",
    "shopping": "women_shopping_memes_categorized",
}
STEREO_SUBSPLITS = ["same", "different", "image"]
IMAGE_EXTS = ["*.jpeg", "*.jpg", "*.png"]

# non-stereotype (GOAT-Bench harmful/offensive) sources: (source_name, subdir)
NON_STEREO_SOURCES = [
    ("goat_harmful", "non_misogyny/harmfulness"),
    ("goat_offensive", "non_misogyny/offensiveness"),
]
# Target number of non-stereotype samples (paper Table I = 1133).
NON_STEREO_TARGET = 1133

# Stratified split fractions (paper test set ~= 490 samples => 15%).
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------- #
# Text composition
# --------------------------------------------------------------------------- #
# How the per-sample text fed to CLIP is built from the available pieces:
#   "fused"   -> caption + OCR  (stereotype);  GOAT jsonl text (non-stereotype)
#   "caption" -> caption only   (stereotype);  GOAT jsonl text (non-stereotype)
#   "ocr"     -> OCR only        (stereotype); GOAT jsonl text (non-stereotype)
#   "empty"   -> "" (stereotype);              GOAT jsonl text (non-stereotype)
#
# NeSy information split (the key design choice):
#   * NEURAL path text   -> OCR (noisy, automatically extractable): leaves fixable
#                            errors for the symbolic path to correct.
#   * SYMBOLIC path text -> caption (clean knowledge): complementary signal.
NEURAL_TEXT_SOURCE = "ocr"
SYMBOLIC_TEXT_SOURCE = "caption"
TEXT_SOURCE = "ocr"   # default for compose_text() / degenerate-variant building

# --------------------------------------------------------------------------- #
# CLIP backbone (frozen)
# --------------------------------------------------------------------------- #
CLIP_MODEL = "openai/clip-vit-base-patch32"
EMB_DIM = 512

# --------------------------------------------------------------------------- #
# Symbolic prompts (manually designed, per the paper's symbol grounding).
# Image-side predicates use the image embedding; *_text uses the text embedding.
# Prompt sets are ensembled (mean cosine) for robustness.
# --------------------------------------------------------------------------- #
PROMPTS = {
    "woman_present": [
        "a photo of a woman",
        "a woman",
        "a meme about a woman",
    ],
    # visual context (scene) prompts
    "kitchen_context": ["a kitchen", "a photo of a kitchen", "cooking in the kitchen"],
    "leadership_context": ["a business leader", "a boss in an office", "a politician giving a speech"],
    "working_context": ["a person at work", "an office workplace", "a working professional"],
    "shopping_context": ["a shopping mall", "a person shopping", "buying clothes in a store"],
    # visual stereotype prompts (woman + stereotypical role)
    "kitchen_stereotype": ["a woman cooking in the kitchen", "a woman doing housework", "a woman as a housewife"],
    "leadership_stereotype": ["a woman as a boss", "a woman leader giving orders", "a woman making executive decisions"],
    "working_stereotype": ["a woman working at a job", "a woman in the workforce", "a career woman"],
    "shopping_stereotype": ["a woman shopping", "a woman buying things", "a woman with shopping bags"],
    # textual stereotype prompts (meme text vs stereotype statement)
    "kitchen_text_stereotype": [
        "women belong in the kitchen",
        "a woman's place is in the kitchen cooking and cleaning",
    ],
    "leadership_text_stereotype": [
        "women are bad leaders and too affective to lead",
        "women cannot make good decisions in charge",
    ],
    "working_text_stereotype": [
        "women should not work and belong at home",
        "women are not capable at their jobs",
    ],
    "shopping_text_stereotype": [
        "women only care about shopping and spending money",
        "women are obsessed with shopping",
    ],
}

# --------------------------------------------------------------------------- #
# Symbolic thresholds (Table II defaults; re-calibrated empirically later and
# written back to artifacts/thresholds.json by tests/threshold_calibration.py).
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "woman_present": 0.55,
    "context": 0.555,        # {class}_context
    "stereotype": 0.57,      # {class}_stereotype  (visual)
    "text_stereotype": 0.70, # {class}_text_stereotype (textual)
}
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"


def load_thresholds() -> dict:
    """Calibrated thresholds (artifacts/thresholds.json) if present, else defaults."""
    import json

    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return dict(THRESHOLDS)


STEREO_CLASSES = ["kitchen", "leadership", "working", "shopping"]

# --------------------------------------------------------------------------- #
# Affect component
# --------------------------------------------------------------------------- #
# Canonical tone order.
AFFECT_NAMES = ["sarcasm", "contempt", "anger", "humor", "neutral"]
AFFECT_FEATURES = AFFECT_NAMES + ["gender_directed"]   # 6-d affect vector
AFFECT_DIM = len(AFFECT_FEATURES)

# Affect head: frozen CLIP [v;u] -> 5 tone (sigmoid) + gender_directed.
AFFECT_OUTPUTS_PATH = ARTIFACTS_DIR / "affect_outputs.npy"   # (N,6) in [0,1]
AFFECT_HIDDEN_PATH = ARTIFACTS_DIR / "affect_hidden.npy"     # (N,512) hidden
AFFECT_HEAD = {
    "hidden": 512,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "batch_size": 128,
    "n_folds": 5,        # out-of-fold predictions for train rows (avoid stacked leakage)
}

# Affect-aware fusion (Path-A'): concat / gate(FiLM) / cross-attention.
AFFECT_FUSION = {"d_model": 256, "nhead": 4, "dropout": 0.1}  # cross-attention block

# Deployed neural branch. "plain" = a plain MLP over [v;u] with no affect and no
# FiLM gating (the affect-free deployed model). "gate"/"concat"/"crossattn" are
# retained for the affect-entanglement study only. Single source of truth: the
# deployed pipeline and every Results table read this so a full run stays
# consistent.
NEURAL_FUSION = "plain"

OOF_LOGITS_PATH = ARTIFACTS_DIR / "oof_logits.npy"   # decision-layer training logits

# Literal thresholds. True  => each concept's cut-points are optimised on the
# calibration split (F_beta one-vs-rest, support-constrained); False => the fixed
# 50/75/90 quantile grid. The fixed grid assumes every concept separates its
# class at the same point of its own score distribution, which is an assumption
# rather than a finding -- see experiments/phase2_e21_thresholds.py.
LEARNED_THRESHOLDS = False   # REVERTED. Learned per-concept cuts match the fixed grid on
                             # the full split and lose -0.052 macro-F1 on the novel slice
                             # (bundled with width 4; see phase2_e25_fullprotocol.py).

# Rule complexity. True => clause length and DNF width selected per class by
# cross-validation inside the calibration split, with the DNF chosen by beam
# search. Kept OFF: E2.2 found that the stronger search overfits the calibration
# objective (val 0.857 / test 0.845, against greedy's val 0.852 / test 0.857) and
# that CV on subsampled folds is biased toward rules that are too simple. The
# restricted greedy search is doing useful capacity control on a corpus this
# size, so it stays the default and the caps are justified empirically rather
# than assumed. See experiments/phase2_e22_complexity.py.
LEARNED_COMPLEXITY = False

# Folds used to produce out-of-fold neural logits for fitting the decision layer
# (stacking). 0/None => fit it on in-sample logits, which lets a well-trained
# neural branch hide its errors from the layer that exists to correct them.
POLICY_CROSSFIT = 5

# --------------------------------------------------------------------------- #
# Neural classifier (Path-A) hyperparameters
# --------------------------------------------------------------------------- #
MLP = {
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    # Path-A is trained as a properly-tuned neural baseline, selected on VALIDATION
    # macro-F1 by experiments/phase0_baseline_strength.py (never on test). The previously
    # deployed unweighted linear head was the weakest of the eight configurations
    # swept, which understated the neural branch and inflated the symbolic margin.
    "hidden": 512,           # None => single linear layer; 512 => one ReLU layer
    "class_weighted": True,  # inverse-frequency CE weights (Path-A imbalance handling)
    "standardize": True,     # per-feature train-split standardization of [v;u]
}
