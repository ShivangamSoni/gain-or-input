# Reproducing the paper

Every number in the paper is written by one script into one JSON file under `artifacts/`, and every
generated table is written from those files by a script under `paper/EAAI/tables/`. The consistency
check `paper/EAAI/check_numbers.py` recomputes each number the prose quotes and fails if the text
and the artefact disagree, so a re-run can be compared with ours line by line.

Data comes first: see `DATA.md`. WBMS is available from its authors on request, so the case-study
corpus cannot be redistributed here; `artifacts/release/corpus_splits.csv` identifies every row by
the SHA-256 of its image so that the splits can be rebuilt from a copy obtained from the authors.

## Environment

Python 3.12. The versions these results were produced with:

| package | version | | package | version |
|---|---|---|---|---|
| torch | 2.6.0+cu124 | | scikit-learn | 1.8.0 |
| numpy | 2.4.4 | | transformers | 5.9.0 |
| pandas | 3.0.2 | | open_clip_torch | 3.3.0 |
| scipy | 1.17.1 | | easyocr | 1.7.2 |
| pillow | 12.2.0 | | requests | 2.32.5 |

The audit tool itself (`inputaudit/`) needs only numpy, pandas and scikit-learn; torch is optional
and only changes which probe D3 and D5 use.

Hardware: one NVIDIA RTX 4070 Laptop (8 GB). The probes fix the cuBLAS workspace before CUDA
starts, so a given device reproduces its own runs exactly; CPU and GPU runs of the same seeds can
differ by up to about 0.02 macro-F1, which changed no verdict here. The zero-shot vision-language
model runs locally through Ollama (`minicpm-v4.5`, temperature 0, 4,096-token context).

## Order

1. **Build the corpus.** `python -m nesymis.data.build_manifest` writes `artifacts/manifest.csv`
   (3,263 rows, fixed splits). `python experiments/uniform_ocr_protocol.py` re-runs one OCR pipeline
   over every image (the uniform-OCR protocol) and writes `artifacts/manifest_corrected.csv`.
2. **Encode.** `python -m nesymis.encoders.clip_encoder` caches the frozen CLIP ViT-B/32 image and
   text embeddings (rebuilding the manifest invalidates every cache); `experiments/d5_two_encoders.py`
   and `experiments/repair_backbones.py` cache the stronger encoders they need.
3. **Run the experiments.** Each script below is self-contained and writes one artefact.
4. **Compare with ours.** Every result file is written under the name of the script that
   produced it, and `artifacts/release/artifact_hashes.csv` carries the SHA-256 of each one as we
   produced it, so a re-run can be compared file by file.

## What produces what

**`experiments/NAME.py` writes `artifacts/NAME.json`.** That is the whole rule; these are the
scripts that produce a result file:

* `experiments/cost_measurements.py` -> `artifacts/cost_measurements.json`
* `experiments/d2_d4_external_tasks.py` -> `artifacts/d2_d4_external_tasks.json`
* `experiments/d3_surface_screen.py` -> `artifacts/d3_surface_screen.json`
* `experiments/d5_modality_probes.py` -> `artifacts/d5_modality_probes.json`
* `experiments/d5_seed_stability.py` -> `artifacts/d5_seed_stability.json`
* `experiments/d5_two_encoders.py` -> `artifacts/d5_two_encoders.json`
* `experiments/d6_duplicates.py` -> `artifacts/d6_duplicates.json`
* `experiments/duplicate_free_split.py` -> `artifacts/duplicate_free_split.json`
* `experiments/grouped_cv_operating_point.py` -> `artifacts/grouped_cv_operating_point.json`
* `experiments/known_answer_architectures.py` -> `artifacts/known_answer_architectures.json`
* `experiments/known_answer_baseline_and_pooling.py` -> `artifacts/known_answer_baseline_and_pooling.json`
* `experiments/known_answer_mami.py` -> `artifacts/known_answer_mami.json`
* `experiments/known_answer_mmhs.py` -> `artifacts/known_answer_mmhs.json`
* `experiments/novel_slice_criteria.py` -> `artifacts/novel_slice_criteria.json`
* `experiments/original_design_multiseed.py` -> `artifacts/original_design_multiseed.json`
* `experiments/per_class_intervals.py` -> `artifacts/per_class_intervals.json`
* `experiments/release_manifest.py` -> `artifacts/release_manifest.json`
* `experiments/repair_backbones.py` -> `artifacts/repair_backbones.json`
* `experiments/repair_benchmark_split.py` -> `artifacts/repair_benchmark_split.json`
* `experiments/repair_fresh_partitions.py` -> `artifacts/repair_fresh_partitions.json`
* `experiments/repair_nulls_and_cost.py` -> `artifacts/repair_nulls_and_cost.json`
* `experiments/repair_source_controlled.py` -> `artifacts/repair_source_controlled.json`
* `experiments/s2_text_regimes.py` -> `artifacts/s2_text_regimes.json`
* `experiments/threshold_calibration.py` -> `artifacts/threshold_calibration.json`
* `experiments/threshold_sensitivity.py` -> `artifacts/threshold_sensitivity.json`
* `experiments/tool_validation.py` -> `artifacts/tool_validation.json`
* `experiments/uniform_ocr_protocol.py` -> `artifacts/uniform_ocr_protocol.json`

The other files in `experiments/` are imported by these and write nothing of their own:
`common.py` (the shared context), `repair_core.py`, the `data_*.py` dataset loaders,
`external_transfer.py`, `mami_full_train.py`, `neural_scaling.py`, `concept_scorecard_eval.py`,
`evidence_feasibility.py` and `repair_operating_point_study.py`.

`artifacts/release/artifact_hashes.csv` lists the SHA-256 of each result file as we produced it,
under these same names, so a re-run can be compared with ours file by file.

## Costs

The data tier is seconds: about 30 s for all six checks on 3,263 memes on a CPU. The system tier
costs one training run per input configuration (35 s here). The expensive parts are the external
encoders (minutes per dataset), the known-answer tests on MMHS150K (about 20 minutes, plus about
2.5 hours for the vision-language model's 3,000 answers), our OCR over MAMI's 11,000 images (about
95 minutes), and the 11-seed protocols (tens of minutes each).
