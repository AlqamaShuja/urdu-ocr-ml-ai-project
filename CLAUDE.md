# CLAUDE.md - project brief for AI coding agents

You are working on a Master's thesis project. Read this file fully before acting.
It is the entry point; the two detail files are:

- **[docs/CRNN.md](docs/CRNN.md)** - Model A, the CRNN baseline. Built and working.
- **[docs/VLM.md](docs/VLM.md)** - Model B, the vision-language model. Not started.

---

## RULE 0 - KEEP THESE FILES CURRENT

**Whenever you change the project, update the matching file in the same session,
before you finish your turn.** This is not optional and not a nicety: these files
are the only memory across sessions. A change that is not written down here did
not happen as far as the next session is concerned.

| What you changed | Update |
| --- | --- |
| CRNN code, config, results | `docs/CRNN.md` |
| VLM code, model choice, results | `docs/VLM.md` |
| Project status, rules, environment, open issues | this file |
| Any decision, run, or finding, with its evidence | `reports/experiment_log.md` (append-only) |
| Dataset facts, counts, licences, exclusions | `reports/dataset_card.md` |

`reports/experiment_log.md` is **append-only**. Never edit a past entry; add a new
one that corrects it. Date every entry.

Do not write results you have not measured. Do not write "complete" for something
you have not run.

---

## The project

**Title:** Comparative Evaluation of CRNN and Fine-Tuned Vision-Language Models
for Printed and Handwritten Urdu Text Recognition.
Master's in ML/AI, NUST (PNEC).

**Question:** how do a CRNN+CTC baseline and a parameter-efficiently fine-tuned
small VLM compare on Urdu text recognition, measured separately on printed and
handwritten text under one identical evaluation protocol?

**Four experiment groups.** No assumed winner.

| # | Model | Domain | Data | Granularity | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | CRNN + CTC | Printed | Synthetic-R-Urdu | word | not started |
| 2 | VLM | Printed | Synthetic-R-Urdu | word | not started |
| 3 | CRNN + CTC | Handwritten | NUST-UHWR | line | **code ready, needs GPU run** |
| 4 | VLM | Handwritten | NUST-UHWR | line | not started |

The comparison is made **within** each domain. Printed and handwritten numbers are
never compared to each other, averaged, or put on a shared axis without stating the
granularity, because the two domains differ in granularity and image height.

**The user is a beginner in practical ML with limited time.** Explain steps in
simple Roman Urdu; write code, comments, filenames and documentation in English.
Work in small verifiable steps, not one large dump. After each step say what files
should exist, what to run, what output to expect, and how to troubleshoot.

---

## Hard rules - do not violate

1. **Never train, tune, or select on test data.** `src/crnn/train.py` does not open
   the test manifest at all. Keep it that way. Evaluate on test once, when model
   selection is final.
2. **Both models must be scored by `src/evaluation/`** - the same normalizer and
   the same metric code. If one side is scored differently, the experiment measures
   the scoring, not the models.
3. **Never call prompt-only inference "fine-tuning."** Label zero-shot and
   fine-tuned results separately and explicitly.
4. **Never reverse an Urdu string** because the script is right-to-left. Text is
   stored in logical order. Display direction is the renderer's job.
5. **Never claim the local 2 GB Quadro M620 can fine-tune a VLM.** It cannot.
6. **Do not invent** dataset sizes, licences, citations, model capabilities or
   results. Verify from the official source or from an actual run, and record where
   the number came from.
7. **Do not add a web frontend or backend.** This is a research project.
8. **Ask before large downloads, paid cloud usage, or anything hard to reverse.**
9. **No external language model on one side only.** The CRNN uses greedy CTC
   decoding with no language model, because the VLM already contains one. Adding a
   beam-search LM to the CRNN alone would measure language modelling, not reading.

---

## Current status (last updated 2026-11-07)

| Step | Status |
| --- | --- |
| Environment setup and verification | **complete** |
| Dataset verification from official pages | **complete** |
| NUST-UHWR downloaded, audited, manifests built | **complete** |
| Shared normalization + CER/WER code | **complete**, 67 tests passing |
| CRNN model, dataset, decode, train, evaluate, demo | **complete** |
| CRNN tiny overfit test | **PASSED** - CER 0.0000, 8/8 exact |
| CRNN real training run on GPU | **NOT DONE - this is the next task** |
| Printed data (Synthetic-R-Urdu) | not downloaded |
| VLM - anything at all | **not started**, see docs/VLM.md |

**There is no usable trained model yet.** The checkpoints in `outputs/checkpoints/`
are `smoke_local_*` (outputs nothing) and `local_demo_*` (CER 0.81, predicts only
the most common character). **Neither may appear in any results table.**

---

## Open issues

1. **The Colab notebook's checkpoint claim is wrong.** `notebooks/01_colab_crnn_training.ipynb`
   says in markdown that checkpoints are written to Drive every epoch. They are
   not: `train.py` writes to `outputs/` inside the session, and a separate cell
   copies to Drive *after* training. A session that dies mid-training loses
   everything. **Fix:** point `output.checkpoints_dir` at the mounted Drive path,
   or add a per-epoch copy. The user already lost a run to this.
2. **Printed-domain licence.** Synthetic-R-Urdu states no licence anywhere. The
   user's decision: not a blocker for local work and a local demo. A ready-to-send
   enquiry is at `reports/licence_enquiry_iiit.md`, to be sent if the thesis is
   submitted, published, or pushed publicly with the data or trained weights.
   **Note: the GitHub repo is now public**, so revisit this before the printed data
   or models trained on it are committed.
3. **Full-page images are out of scope.** The CRNN reads one text line. Page
   support needs line segmentation, which is a separate stage and deliberately not
   part of the comparison. `scripts/predict_image.py` warns on page-like input.

---

## Environment

Windows 11, Python **3.12.10**, local venv at `.venv`. Beware: Python 3.15 is also
installed and is the `py` launcher default - always use the venv's `python`.

```bat
cd E:\alqama_nust\urdu_ocr_project
.venv\Scripts\activate.bat
```

(The user often uses **cmd**, where `Activate.ps1` silently does nothing. In
PowerShell use `.\.venv\Scripts\Activate.ps1`.)

Local PyTorch is **CPU-only on purpose** (`torch==2.15.0+cpu`). `CUDA available:
False` locally is correct. Package versions are pinned in `requirements.txt` to
mature lines (numpy 2.2, pandas 2.3, opencv 4.14, jiwer 3.1) because an unpinned
install pulled in pandas 3.x and opencv 5.x, whose APIs differ from published
tutorials.

Hardware: i5-7440HQ (4 cores, mobile), 24 GB RAM, Quadro M620 **2 GB** (unusable
for training). **All real training runs on Google Colab free tier (T4).**

### Verification commands

```bat
python scripts/check_env.py          REM expect: Result: PASSED
python scripts/test_evaluation.py    REM expect: 67 passed, 0 failed
```

Run `test_evaluation.py` after **any** change under `src/evaluation/` - that code
produces every number in the thesis.

---

## Repository layout

```text
CLAUDE.md                 this file
docs/CRNN.md              Model A detail
docs/VLM.md               Model B detail
README.md                 human-facing setup and run instructions
configs/                  YAML, one per experiment
data/
  raw/                    datasets, gitignored
  manifests/              CSV manifests + vocabulary, committed
src/
  data/                   manifest building, vocabulary
  crnn/                   model, dataset, decode, train, evaluate
  vlm/                    EMPTY - Model B goes here
  evaluation/             normalize_text.py, metrics.py  <- shared by both models
scripts/                  env check, dataset audit, tests, overfit test, demo
notebooks/                Colab notebooks
outputs/                  checkpoints, predictions, metrics, figures (gitignored)
reports/                  dataset_card.md, experiment_log.md, audit, licence draft
```

---

## Where the reasoning lives

Do not re-derive these. Read them.

| File | Holds |
| --- | --- |
| `reports/dataset_card.md` | every dataset fact, with its source and date; licences; the exclusions and why; the normalization evidence |
| `reports/experiment_log.md` | append-only decision and run history, dated |
| `reports/dataset_audit_nust_uhwr.md` | generated audit output |
| `docs/CRNN.md` | architecture, width arithmetic, gotchas, results so far |
| `docs/VLM.md` | what must be verified before starting, constraints, plan |

### Decisions already made - do not silently reopen

- **Printed stays word-level** (Synthetic-R-Urdu), handwritten stays line-level
  (NUST-UHWR). Two separate tasks.
- **Normalization: NFKC, plus folding Arabic letter variants to Urdu forms.**
  Diacritics kept. Digits NOT folded. Genuine Urdu letters (آ ؤ ئ ھ ے) NOT folded.
  Evidence in `reports/dataset_card.md`.
- **`04745.jpg` excluded from validation** - byte-identical to training image
  `04746.jpg` with a label that does not match its image.
- **Greedy CTC decoding only**, no external language model.

If the user asks to change one of these, say what it affects and update
`reports/dataset_card.md` and `reports/experiment_log.md` with the new decision.

---

## Data

**NUST-UHWR** (handwritten, Nastaliq), Zenodo record **21674269**, CC BY 4.0.
(The original project brief's URL, record 10548922, is dead - 404.)
`UHWR.zip`, 158,934,492 bytes, MD5 `e367ff83c147671f4ae569bc63480af9`.
10,606 line images, all exactly **64 px tall**, widths 200-2,170.

**Usable counts after audit: 8,483 train / 1,060 val / 1,061 test = 10,604.**
Two images are excluded: `03598.jpg` (unlabelled, no manifest references it) and
`04745.jpg` (label conflict, see above). These counts must appear in every results
table with the exclusions stated.

Vocabulary: **109 characters, 110 CTC outputs** (blank at index 0), built from the
**training split only**. Measured: 0 characters appear only in val or test, so
there is no vocabulary-induced CER floor.

**Known data limitation to report in threats to validity:** the Zenodo record
documents label noise - spelling variation, punctuation inconsistency, unclear
characters. CER has a non-zero floor on this dataset, so a gap of a fraction of a
percent between the two models is not meaningful. Also, identical transcriptions
appear across splits (83 train/val, 84 train/test, 8 val/test); different writers,
same sentence, reported rather than removed.

---

## Metrics

From `src/evaluation/metrics.py`. Both models, same code.

- **CER and WER are corpus-aggregated**, not averaged per sample.
- **Spaces count as characters** in CER.
- Every run reports **normalized (primary)** and **raw (stricter secondary)**.
- Substitutions, deletions and insertions are reported alongside the rate.
- **`1 - CER` is never reported as "accuracy"** - CER can exceed 1.0.
- Also reported: inference latency (at batch size 1, with device stated),
  parameter count, peak GPU memory, training time.

**Latency and memory must be measured on the same device class for both models**,
or those columns are not comparable. If the CRNN is trained on CPU and the VLM runs
on GPU, the CRNN's inference must still be measured on GPU for the comparison.
