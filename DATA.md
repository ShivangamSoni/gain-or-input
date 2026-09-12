# Data

No meme image, caption or OCR text is distributed with this code. Every dataset below is obtained
from the people who published it, under their own terms. What we add is the split of each corpus we
used and the code that rebuilds everything from it: `artifacts/release/corpus_splits.csv` identifies
each meme of the case-study corpus by the SHA-256 of its image file, and
`artifacts/release/mami_splits.csv` gives the MAMI split by the organisers' file names.

## The case-study corpus (3,263 memes)

| part | rows | obtain |
|---|---|---|
| WBMS -- women's stereotype memes, four classes (kitchen, leadership, working, shopping) | 2,130 | from the WBMS authors on request (Kanwar et al.) |
| GOAT-Bench -- harmfulness and offensiveness portions | 1,133 | from the GOAT-Bench release (Lin et al.) |

WBMS file names carry the post caption that the original design routed to the symbolic layer, which
is why no file name appears in anything we release. Place the images under `dataset/` in the layout
`ARCHITECTURE.md` describes, then `python -m nesymis.data.build_manifest` rebuilds `manifest.csv`; join
`corpus_splits.csv` on the SHA-256 of each image to confirm you have the same rows in the same
splits. The two text fields are the post caption supplied with each image and the OCR text; the
uniform-OCR protocol (`experiments/uniform_ocr_protocol.py`) re-reads every image with one EasyOCR
pipeline, and is the protocol every reported result uses.

## Public tasks used in the data tier

Eleven task configurations across seven datasets, each under its own licence and access procedure:
HarMeme (harmful; 3-class intensity), PrideMM (hate; 4-class target), MAMI (misogynous), MMSD2.0
(sarcasm), Hateful Memes, EXIST 2024 (sexist; 6-class), MMHS150K (sexist; 6-class hate type). The
code expects them under `dataset/<NAME>/` with the layout each publisher ships; the loaders in
`experiments/` name the files they read. Downloading them is the user's responsibility and none of them
is redistributed here.

MMHS150K also carries the known-answer tests of the system tier (`experiments/known_answer_mmhs.py`,
`experiments/known_answer_architectures.py`), because its tweets carry both their own post text and the
organisers' OCR of the text inside the image -- the same two-channel shape as our case study, on
data we did not collect.

## Models

Frozen CLIP ViT-B/32 and, for the two-encoder rule in D5, SigLIP SO400M and larger CLIP variants,
all downloaded from their public releases at run time. The zero-shot vision-language model in the
known-answer tests is `minicpm-v4.5` served locally by Ollama. No model weights are distributed
here; the only trained parts are the small heads this code fits.

## What we do not release

The dataset files, the embedding caches built from them, OCR text, captions, and the trained
weights that embed them. `experiments/release_manifest.py` regenerates everything we do release:
the split files above and the SHA-256 of each result file the paper reads.
