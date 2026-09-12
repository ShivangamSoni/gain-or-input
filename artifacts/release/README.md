# Release manifest

These files identify the rows behind every result in the paper without carrying any of the data.

* `corpus_splits.csv` -- the case-study corpus (3,263 memes). `sha256` is the SHA-256 of the image
  file's bytes; `row` is the row's position in our manifest, which fixes the order the cached
  embeddings are stored in. WBMS memes are available from the WBMS authors on request and GOAT from
  its own release; hash the images you receive and join on `sha256`. No file name, caption or OCR
  text appears here, because WBMS file names contain the post caption.
  The WBMS-4 setting is the rows whose `source` starts with `wbms_`, with the same split.
* `mami_splits.csv` -- MAMI rows by the organisers' file names, with our split: the official test
  split, and the official training split divided into train and a 15% stratified validation carve
  (seed 42).
* `artifact_hashes.csv` -- SHA-256 and size of each result file the paper's numbers are read from,
  so a re-run can be compared with ours file by file.

Reproduction order and the environment are described in REPRODUCE.md.
