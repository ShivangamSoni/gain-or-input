# inputaudit — an input-availability audit for multimodal classifiers

Is a reported gain in the model, or in the input? Twelve checks in two tiers, cheapest first.
Each returns its statistics and a verdict: **FLAG** (the problem the check targets is present),
**ok**, or **note** (reported, no decision rule). Thresholds are screening defaults, not universal
rules: detection thresholds (D2's enrichment test, D3's lift, D6's excess, S5's ratio) are the
smallest values that flag at most 5% of clean cases in planted-leak calibration
(`experiments/threshold_calibration.py`); the rest are stated materiality choices. Every statistic is
returned so you can apply your own.

| check | question | flag when (default) |
|---|---|---|
| **D1** provenance | Is each text field produced the same way for every source? | a field is produced differently across sources |
| **D2** field relations | Does a relation between text fields reveal the label? | support ≥ 20, class precision ≥ 0.95, and enrichment over the class base rate (binomial p ≤ 0.001, Bonferroni) |
| **D3** surface screen | Does text *shape* alone (no words) predict the label? | lift ≥ 0.15 & share ≥ 0.75 (notable); ≥ 0.30 & ≥ 0.85 (severe); bootstrap intervals reported |
| **D4** source leak | Does shape identify the source, and the source the label? | AUC ≥ 0.80 and Cramér's V ≥ 0.50 |
| **D5** modality check | Does each modality matter, under a weak **and** a strong encoder? | one modality alone ≥ 0.90 macro-F1 |
| **D6** near-duplicate leakage | Would looking up a test item's duplicates (same text, or embedding cosine ≥ 0.90) answer it? | share answered, minus its label-permutation null, ≥ 0.05 |
| **S1** input inventory | Do compared components see the same inputs, all available at inference? | the system reads an input its baseline does not, or one unavailable |
| **S2** same-input test | Does the gain survive equal inputs? | there is a gain, and a significant part of it disappears with equal inputs (swing interval above zero); the equal-input margin is the architecture's share |
| **S3** headroom by type | What information would fix the remaining errors? | residual below 2% of the test set |
| **S4** influence | Does the explanatory component change decisions? | it changes < 2% of decisions |
| **S5** faithfulness | Do the named reasons drive the decision? | delete-top-3 / delete-random-3 < 2 |
| **S6** fresh partitions | Does a test-informed choice hold on fresh data? | a criterion fixed in advance is not met |

D5 also encodes a rule: a modality is only reported as uninformative if the **strongest** encoder
agrees; a weak encoder's verdict alone is "encoder-limited". When D6 flags, D3 is re-run on the
duplicate-free test rows, which separates shape shortcuts from lookup of seen items.

## Install

Needs `numpy`, `pandas`, `scikit-learn`; PyTorch is optional (without it, the D3/D5 probe is a
class-balanced logistic regression instead of the small MLP used in the paper). The MLP probe is
deterministic on a given device (the package fixes the cuBLAS workspace before CUDA starts); CPU
and GPU runs of the same seeds can differ by up to about 0.02 macro-F1, which changed no verdict
on the NeSy-MIS corpus. The package is
standalone — it does not import the NeSy-MIS model code.

## Data tier (command line)

A manifest is any CSV with a label column and one or more text columns; optionally a `split`
column (train/val/test; otherwise a stratified 70/15/15 split is made) and a source column.

```bash
python -m inputaudit data --manifest memes.csv --label label --text ocr,caption \
    --split split --source source --provenance provenance.json \
    --emb clip:image=clip_img.npy --emb clip:text:ocr=clip_ocr.npy \
    --emb siglip:image=sig_img.npy --emb siglip:text:ocr=sig_ocr.npy \
    --out audit
```

* `provenance.json`: `{"caption": {"siteA": "post title scraped with the image", "siteB": "dataset meme text"}, "ocr": {"siteA": "EasyOCR", "siteB": "EasyOCR"}}`
* `--emb ENCODER:VIEW=path.npy`, row-aligned; VIEW is `image` or `text:<field>`. List encoders
  weakest first. Text embeddings of a field are D3's text reference; the first text field's are
  D5's text view and, from the first encoder listed, D6's near-duplicate criterion (without
  them, D6 matches exact text only).
* Writes `audit.md` (verdict table + details) and `audit.json`.

## System tier (Python)

You supply the parts only you can write; the package does the statistics.

```python
from inputaudit import system_tier as st

st.s1_inventory(components={"neural": ["image", "ocr"], "symbolic": ["image", "caption"]},
                system=["neural", "symbolic"], baseline=["neural"],
                available_at_inference=["image", "ocr", "caption"])

# run(name, config, seed) -> (pred_system, pred_baseline) on the test rows
st.s2_same_input(run, {"as built": {"symbolic": "caption"}, "equal": {"symbolic": "ocr"}},
                 seeds=[42, 1, 2], y_test=y_te, k=5, equal="equal")

st.s3_headroom(y_te, base_preds, {"full caption": cap_preds, "image only": img_preds})
st.s4_influence(y_te, pred_full, pred_without_component)
st.s5_faithfulness(predict_from_Z, Z_test, contributions)        # deletion / sufficiency
parts = st.s6_partitions(y, pool_idx=np.flatnonzero(~test_mask))  # then:
st.s6_verdict(deltas, criterion={"macro_f1": -0.01, "acc": -0.01})
```

## Validation

`experiments/tool_validation.py` runs the data tier on the NeSy-MIS corpus (as first assembled and
under the uniform-OCR protocol) and on eleven public meme tasks. It reproduces the paper's
surface-screen scores and verdicts on all eleven tasks exactly, and on our corpus the shape-only
scores, source AUCs, and D1/D2 flags. Reports: `artifacts/tool_validation_first.md`,
`artifacts/tool_validation_uniform.md`, `artifacts/tool_validation.json`.

Known answers: `experiments/threshold_calibration.py` plants shape cues and near-copies of test items at
controlled strength into clean tasks (D3, D6), and draws D2's and D4's chance behaviour and S5's
random-ranking null. `experiments/known_answer_mmhs.py` and `experiments/known_answer_architectures.py` build
systems on MMHS150K whose added component reads the tweet text (an input its baseline never sees)
or the baseline's own inputs through another encoder, with that component built as an MLP probe, a
fusion transformer, a nearest-neighbour classifier, a concept scorecard and a zero-shot VLM, and
check S1 and S2 against the answer fixed in advance.
