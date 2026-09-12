"""inputaudit -- an input-availability audit for multimodal classifiers.

Two tiers, cheapest first. The DATA tier needs only a manifest (and, for D5, embeddings); the
SYSTEM tier needs the system under test. Every check returns a dict with its statistics and a
verdict: "FLAG" (a problem the check is designed to catch), "ok", or "note" (informative, no
decision rule).

  Data tier                                    System tier
  D1 provenance      are text fields made      S1 input inventory   does every compared
                     the same way per source?                       component see the same inputs?
  D2 field relations does a relation between   S2 same-input test   does the gain survive equal
                     fields reveal the label?                       inputs?
  D3 surface screen  does text shape alone     S3 headroom by type  what kind of information would
                     predict the label?                             fix the remaining errors?
  D4 source leak     does shape identify the   S4 influence         does the explanatory component
                     source, and the source                         change decisions?
                     the label?
  D5 modality check  does each modality        S5 faithfulness      do the named reasons drive the
                     matter, under a weak AND                       decision (deletion/sufficiency)?
                     a strong encoder?         S6 fresh partitions  does a test-informed choice hold
                                                                    on data that did not inform it?

    python -m inputaudit data --manifest m.csv --label label --text caption,ocr --source source
    from inputaudit import data_tier, system_tier

Standalone: numpy, pandas, scikit-learn; PyTorch optional (the D3/D5 probe falls back to
logistic regression without it).
"""
from inputaudit import data_tier, system_tier  # noqa: F401

__version__ = "0.1.0"
