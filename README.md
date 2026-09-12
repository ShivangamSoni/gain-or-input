# gain-or-input

Is a reported gain in the model, or in the input? This repository holds

* **`inputaudit/`** -- the audit: twelve checks in two tiers, each with a statistic, a screening
  default and a cost (`python -m inputaudit data --manifest memes.csv --label label --text ocr`).
  It needs only numpy, pandas and scikit-learn; torch is optional.
* **`nesymis/`** -- the neuro-symbolic meme classifier the paper audits and repairs, reduced to the
  modules the experiments import.
* **`experiments/`** -- the scripts behind the paper's numbers. Each writes one result file:
  `artifacts/<name>.json` comes from `experiments/<name>.py`.
* **`REPRODUCE.md`**, **`DATA.md`** -- environment, run order, and how to obtain the data.
* **`artifacts/release/`** -- each corpus's split identified by the SHA-256 of every image, and the
  hash of every result file the paper reads.

No meme image, caption or OCR text is distributed here; see `DATA.md`.

See `inputaudit/README.md` for the checks, their defaults and how each default was calibrated.
