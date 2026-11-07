# Experiment log

Append-only. One entry per setup change, dataset action or training run.
Never edit a past entry; add a correcting entry instead.

---

## 2026-11-02 - Step 1: local environment setup

**Machine:** local Windows PC (Windows 11 Pro 10.0.26200), Intel Core i5-7440HQ
(4 cores), 24 GB RAM, NVIDIA Quadro M620 2 GB (driver 553.46), 305 GB free on E:.

**Verified present:**

| Tool | Version |
| --- | --- |
| Python (default `python`) | 3.12.10 |
| pip | 26.2.2 (upgraded from 25.0.1) |
| Git | 2.55.0.windows.2 |

**Notes on the base system:**

- Python 3.15 is also installed and is the `py` launcher default (`py -0p` shows
  `-V:3.15 *`). The project venv was built from Python 3.12 explicitly. Use
  `python` from the activated venv, not `py`, to avoid landing on 3.15.
- Jupyter was **not** installed system-wide, contrary to the initial project
  assumption. It is now installed inside the project venv instead, which is the
  better outcome for reproducibility.

**Virtual environment:** `.venv`, Python 3.12.10.

**Installed (local, CPU only):** numpy 2.5.4, pandas 3.1.0, pillow 13.0.0,
opencv-python 5.0.0.93, matplotlib 3.11.2, tqdm 4.70.1, pyyaml 6.0.3,
jiwer 4.0.0, jupyter 1.1.1, ipykernel 7.4.0, torch 2.15.0+cpu.

All are Python 3.12 compatible; no version conflict required a 3.11 fallback.

**Jupyter kernel registered:** `urdu-ocr`, display name "Python (Urdu OCR)".

**Verification result (`scripts/check_env.py`):** 3 of 4 groups pass.

- Python interpreter: PASS
- Core packages: PASS
- Urdu Unicode round-trip: PASS (17 code points, logical order preserved)
- PyTorch: **FAIL** - `OSError [WinError 126]` loading `torch/lib/shm.dll`

**Open issue:** the Microsoft Visual C++ 2015-2022 x64 Redistributable is not
installed. Confirmed by `C:\Windows\System32\vcruntime140_1.dll` being absent and
the registry key `HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64` not
existing. All nine torch DLLs are present, so the wheel itself is intact. Fix
pending: install the redistributable (admin required), then re-run
`scripts/check_env.py`.

**Not done yet:** no dataset verified or downloaded, no Colab session started, no
model code written, no training run.

---

## 2026-11-02 - Step 1b: dependency versions pinned

**Reason:** the first install was unpinned and resolved to the newest release of
every package, giving pandas 3.1.0, opencv-python 5.0.0.93, pillow 13.0.0,
matplotlib 3.11.2, numpy 2.5.4, jiwer 4.0.0 and ipykernel 7.4.0. Those are major
version jumps whose APIs differ from published OCR tutorials and documentation.
For a beginner-run project on a deadline that is debugging cost with no research
benefit, so each package was pinned to its mature line.

**Downgraded:**

| Package | Was | Now |
| --- | --- | --- |
| numpy | 2.5.4 | 2.2.6 |
| pandas | 3.1.0 | 2.3.3 |
| opencv-python | 5.0.0.93 | 4.14.0.94 |
| pillow | 13.0.0 | 11.3.0 |
| matplotlib | 3.11.2 | 3.10.9 |
| jiwer | 4.0.0 | 3.1.0 |
| ipykernel | 7.4.0 | 6.31.0 |

**Unchanged:** tqdm 4.70.1, pyyaml 6.0.3, jupyter 1.1.1, torch 2.15.0+cpu
(now pinned explicitly in the README install command).

**Verified:** `pip check` reports "No broken requirements found." All eight core
packages import at the pinned versions. The `urdu-ocr` Jupyter kernel was
re-registered after the ipykernel downgrade.

**Still blocked:** PyTorch cannot load until the Microsoft Visual C++ 2015-2022
x64 Redistributable is installed. Unrelated to the pinning; see the previous entry.

---

## 2026-11-02 - Step 1c: environment verified, Step 1 complete

**Fix applied:** the Microsoft Visual C++ 2015-2022 x64 Redistributable was
installed. `C:/Windows/System32/vcruntime140_1.dll` is now present (50,112 bytes).

**`scripts/check_env.py` result: PASSED (exit code 0), all four groups.**

| Check | Result |
| --- | --- |
| Python interpreter | 3.12.10, inside `.venv` |
| Core packages | all 8 import at pinned versions |
| torch | 2.15.0+cpu loads |
| CPU tensor op | correct (18.0) |
| CTC loss forward+backward | works, loss 4.5164, gradient populated |
| CUDA available | False - expected and correct for a CPU build |
| Urdu Unicode round-trip | preserved, 17 code points, logical order |

The CTC check matters beyond setup: CTC loss is the training objective of the
CRNN baseline, so this confirms the core of Model A is available and
differentiable on CPU before any model code is written.

**`requirements-lock.txt` generated:** 126 pinned packages including
`torch==2.15.0+cpu`. Note the `+cpu` local version tag - reinstalling from plain
PyPI would silently fetch a different build, so the CPU index URL must be used.

**Step 1 is complete.** Local environment is reproducible and verified.

**Still not done:** no dataset verified, downloaded or inspected; no Colab session
run; no model code; no training. Next action is Step 2, dataset verification from
the official pages only.

---

## 2026-11-02 - Step 2a: dataset candidates verified from official pages

Nothing downloaded. Official pages read and recorded in reports/dataset_card.md.

**Corrections to the project brief:**

1. The brief's NUST-UHWR URL (zenodo.org/records/10548922) returns HTTP 404 on
   both the page and the REST API. The dataset is at Zenodo record 21674269,
   DOI 10.5281/zenodo.21674269, by Ul-Hasan and Shafait. Verified: CC BY 4.0,
   open access, UHWR.zip 158,934,492 bytes, MD5 e367ff83c147671f4ae569bc63480af9.
2. The brief described NUST-UHWR as having 10,606 lines with splits. The record
   states 10,606 images but its manifests label only 10,605: image 03598.jpg is
   unreferenced and has no transcription. Stated splits 8,484/1,061/1,061;
   manifest splits 8,483/1,061/1,061. The record documents this itself.
3. IIIT Synthetic-R-Urdu is confirmed word-level (838,500 stated total, but the
   three split counts sum to 838,530) and states NO licence at all. The brief
   assumed a licence would be verifiable; it is not. Must be resolved with the
   maintainers before training on it.
4. IIIT-Urdu is confirmed line-level, 1,610 images, CC BY 4.0, and the page states
   it "can be used only for testing purposes". Its Train and Test download links
   point to the same test.zip. There is no training data, as the brief suspected.

**Blocking design question identified:** no line-level printed Urdu TRAINING set
exists among the candidates, while the handwritten set is line-level. Options A/B/C
are written up in reports/dataset_card.md. Awaiting the researcher's decision
before any download.

---

## 2026-11-02 - Step 2b: printed-domain design decided (Option A)

**Decision by the researcher: Option A. The printed task stays word-level.**

Printed = Synthetic-R-Urdu (word-level, official splits, subsampled).
Handwritten = NUST-UHWR (line-level, official splits retained).
Two separate tasks; CRNN-versus-VLM is compared within each domain only.

**Consequences recorded in reports/dataset_card.md:**

- Printed and handwritten CER/WER are never compared, averaged or plotted on a
  shared axis without stating the granularity.
- Printed WER degenerates to single-word WER and must be labelled as such. CER is
  the primary printed metric.
- The domains differ in image height (32 px printed, 64 px handwritten), so each
  gets its own CRNN input configuration, documented rather than forced to match.
- IIIT-Urdu is dropped from the headline comparison: it is line-level printed data
  and cannot test word-level printed models. Qualitative side observation only.
- The Synthetic-R-Urdu licence gap is now on the critical path.

**Decision by the researcher: no downloads yet.** Nothing fetched. Planned order
when instructed: NUST-UHWR first (licence clear), Synthetic-R-Urdu only after the
licence reply arrives.

**Created:** reports/licence_enquiry_iiit.md - a ready-to-send enquiry to
ocr.nltm@research.iiit.ac.in, the only contact address published at
https://ilocr.iiit.ac.in/contact/. Not sent yet.

---

## 2026-11-02 - Step 2c: NUST-UHWR downloaded, verified and audited

**Downloaded:** UHWR.zip from Zenodo record 21674269.
Size 158,934,492 bytes, matching the record exactly.
**MD5 verified: e367ff83c147671f4ae569bc63480af9 - MATCH.**
Extracted to data/raw/nust_uhwr/UHWR/ (330 MB on disk). Not committed to git.

Synthetic-R-Urdu was deliberately NOT downloaded: its licence is still unknown
and the enquiry has not been answered.

**Audit tool created:** scripts/audit_nust_uhwr.py. Run with `python -I` and the
data path as an argument, since the dataset is downloaded data.
Report: reports/dataset_audit_nust_uhwr.md.

**Every figure stated by the Zenodo record checked out:** 10,606 images on disk,
8,483/1,061/1,061 manifest records (10,605 labelled), 0 missing, 0 corrupt, all
images exactly 64 px high, widths 200-2,170, 0 empty labels, 116 distinct
characters exactly matching chars.txt, 0 image IDs shared between splits.

Image-to-label alignment was spot-checked visually: 07955.jpg renders exactly its
transcription, so there is no systematic off-by-one shift.

**New finding, NOT documented by the Zenodo record: a label conflict.**
04745.jpg and 04746.jpg are byte-identical (both MD5
b0b7bf6baf997080550f0dff00e2891e) but carry different transcriptions and sit in
different splits (train and val respectively). Identical pixels cannot depict two
sentences, so one label is wrong. Resolved by opening the image: it shows the
TRAIN label. The validation record therefore points at the wrong picture.

**Decision: exclude 04745.jpg from validation, keep 04746.jpg in train.** This
closes the train/validation leak and removes a record that would have punished a
correct reading. The audit script was extended to detect label conflicts in
duplicate groups, not just duplication.

**Usable counts for this project: 8,483 train / 1,060 val / 1,061 test = 10,604.**
The test split is untouched; the leak was train-to-validation only, so headline
test metrics are unaffected.

**Reported, not removed:** identical transcriptions appear in train and val (83),
train and test (84), val and test (8). Different writers, same sentence - valid in
a handwriting corpus, but it lets a model score partly from memorised text.
Goes in threats to validity.

**Normalization evidence gathered (decision still open):** NFC changes 318 labels,
NFKC 452, NFD 4,967. NFD is ruled out. NFKC expands the Arabic Presentation Forms
U+FDF2, U+FDFA and U+FE85 (154 occurrences) into constituent letters, which is a
FAIRNESS issue: a generative VLM will emit spelled-out letters, not the ligature
code point, and without NFKC would be scored wrong for a correct reading. Also
present are Arabic letter variants that normalization does not fold
(U+064A, U+0647, U+0629, U+0623, U+0624) and 12 stray Latin letters.
NFKC plus whitespace collapsing is recommended in reports/dataset_card.md,
awaiting confirmation.

**Audit verdict: 2 problems found, both now resolved by documented exclusions.**
Not yet done: manifests not built, no vocabulary, no model code, no training.

---

## 2026-11-02 - Step 2d: normalization decided, shared evaluation code, manifests

**Decision by the researcher: NFKC plus folding Arabic letter variants to Urdu forms.**

**One correction to my earlier framing.** I had listed U+0624 (ؤ) as an Arabic
variant to fold. It is not: it is standard Urdu, used in words such as jao and
banao, and its 171 occurrences confirm genuine usage rather than annotation noise.
It is NOT folded. The same applies to U+0622 آ (1,870), U+0626 ئ (5,028),
U+06BE ھ (6,638) and U+06D2 ے (22,196), all genuine Urdu letters.

**Evidence that justified folding the rest** - measured over 554,720 characters.
In each case the Arabic form is a rounding error beside the Urdu form, which is
the signature of annotator keyboard inconsistency, not a distinction the writers
made on paper:

| Arabic form | Count | Urdu form | Count | Arabic share |
| --- | --- | --- | --- | --- |
| U+064A ي | 26 | U+06CC ی | 43,659 | 0.06% |
| U+0647 ه | 14 | U+06C1 ہ | 23,699 | 0.06% |
| U+0629 ة | 22 | U+06C1 ہ | 23,699 | 0.09% |
| U+0623 أ | 2 | U+0627 ا | 53,874 | 0.004% |
| U+0643 ك | 0 | U+06A9 ک | 29,743 | absent |

U+0643 never occurs in the ground truth but is mapped anyway, because the
normalizer also runs over model predictions and a VLM trained largely on Arabic
may emit the Arabic form.

**Digits are NOT folded.** Both ASCII 0-9 (2,388) and extended Arabic-Indic
U+06F0-U+06F9 (124) occur, so annotation is inconsistent - but unlike the letter
pairs, these glyphs look clearly different on the page. A model reading "4" where
the image shows "۴" has genuinely misread it, and folding would hide that error.

**Diacritics are KEPT** in the primary metric. They are 878 of 554,720 characters
(0.16%). A diacritic-insensitive secondary score is available via
strip_diacritics().

**Created:**

- src/evaluation/normalize_text.py - the single place normalization happens.
- src/evaluation/metrics.py - corpus-aggregated CER/WER with S/D/I counts,
  reporting both normalized (primary) and raw (stricter secondary) numbers.
  Spaces count as characters. 1-CER is never reported as "accuracy", since CER
  can exceed 1.0.
- scripts/test_evaluation.py - **59 assertions, all passing.** Includes the
  fairness case: a model that spells out the U+FDFA ligature scores CER 0.0
  against the single ligature code point.
- src/data/prepare_dataset.py - manifest builder, exclusions declared in code
  rather than applied silently.
- src/data/vocabulary.py - CTC vocabulary, blank at index 0.

**Manifests built:** data/manifests/nust_uhwr_{train,val,test}.csv
**8,483 / 1,060 / 1,061 = 10,604 rows.** 04745.jpg excluded from val as decided.
Both text_raw and text_normalized are stored, so the normalization decision can be
revisited without re-reading the dataset. Paths are stored relative with forward
slashes so the same manifest works locally on Windows and in Colab on Linux.
Verified: 0 nulls, all image paths resolve, all heights 64, widths 200-2170.

**Vocabulary built from the TRAIN split only: 109 characters, CTC output size 110.**
116 raw characters became 109 because NFKC expanded the 3 presentation forms and
4 Arabic variants folded into letters that already existed. Encode/decode
round-trips losslessly.

**Important result: 0 characters appear only in validation or test.** Building the
vocabulary from train alone therefore costs nothing here - there is no
vocabulary-induced CER floor for the CRNN. This was worth measuring rather than
assuming, since such a floor would apply to the CRNN but not to the VLM.

**Not done yet:** no model code, no training, no printed data.

---

## 2026-11-02 - Step 3: CRNN implemented, tiny overfit test PASSED

**Scope decision by the researcher:** the licence enquiry is not a blocker. The
work is local experimentation and a local demo, not dataset redistribution, so
Synthetic-R-Urdu use is low risk. The enquiry draft is kept for if and when the
thesis is submitted or the work is published. Printed data download deferred;
CRNN is being built on handwritten data first.

**Sequence length verified before writing the model, not assumed.** CTC needs
T >= label length + one frame per pair of identical adjacent characters. Measured
over all 10,604 lines:

| Width downsampling | T range | Violations | Smallest margin |
| --- | --- | --- | --- |
| W/2 | 100-1085 | 0 | 79 frames |
| W/4 | 50-542 | 0 | 27 frames |
| W/8 | 25-271 | 59 | -21 frames |

W/8 would make the loss infinite for 59 samples. **W/4 chosen** as the most
aggressive safe option. LineImageDataset re-checks this per row at load time.

**Created:**

- src/crnn/model.py - CNN -> BiLSTM -> CTC head. Height 64 collapses to 1, width
  to W/4. Also supports height 32 for the printed data later.
  **8,761,710 parameters.** Output shapes verified at widths 200/374/912/2170.
- src/crnn/dataset.py - manifest-backed dataset, per-batch width padding with the
  BACKGROUND value (not zero: these are dark-on-light images, so zero padding
  would paste black blocks onto short lines), and true per-sample frame counts.
- src/crnn/decode.py - greedy CTC decoding only, deliberately. A beam search with
  an external language model would lower CRNN error, but the VLM already contains
  a language model; adding one to a single side would measure language modelling
  rather than recognition. Stated in the results.
- scripts/overfit_test.py - the pipeline sanity check.

**Bug found and fixed in the shared metrics code.** The first overfit run crashed:
jiwer's RemoveEmptyStrings transform drops empty *sentences* from the list, so
when the freshly initialised model predicted "" for some samples the references
and hypotheses stopped aligning (8 versus 1). Removed that transform; jiwer then
correctly scores an empty hypothesis as all-deletions. **Four regression tests
added. Test suite now 67 assertions, all passing.** This is exactly the class of
bug a tiny overfit test exists to catch, and it would have silently corrupted
every reported number had it only appeared later.

**Overfit test result: PASSED.**
8 narrowest training samples, 300 steps, AdamW lr 1e-3, seed 42, CPU.

| Step | Loss | CER |
| --- | --- | --- |
| 1 | 8.4727 | 0.9708 |
| 50 | 2.9389 | 1.0000 |
| 100 | 2.1222 | 1.0000 |
| 125 | 1.1536 | 0.5693 |
| 150 | 0.2446 | 0.1642 |
| 200 | 0.0129 | 0.0000 |
| 300 | 0.0036 | 0.0000 |

Final: CER 0.0000, WER 0.0000, **8/8 exact matches**, 384 s wall clock.

The CER plateau at 1.0 through step ~100 while loss fell is normal CTC behaviour:
the model first learns to emit blanks everywhere, which lowers loss without
producing any output, and only then starts emitting characters.

**This proves the pipeline can learn. It says nothing about generalisation** - the
model memorised 8 images, which was the intent.

**Compute evidence for moving to Colab.** 384 s for 300 steps on 8 *narrow* images
is about 1.3 s/step on 4 CPU cores. Median width is 911 px against roughly 300 px
here, so realistic steps will be several times slower. A rough estimate for the
full 8,483-sample training split at batch 16 is on the order of half an hour per
epoch on this CPU, so a real run is tens of hours. GPU training in Colab is
required, as planned. This is an estimate from one measurement, not a benchmark.

**Not done yet:** no real training run, no validation loop, no checkpointing,
no printed data, no VLM.

---

## 2026-11-07 - Step 4: training, evaluation and demo scripts; local smoke run

**Created:**

- configs/crnn_handwritten.yaml - every result-affecting setting in one place.
- src/crnn/train.py - training loop, per-epoch validation, checkpointing, CSV log.
- src/crnn/evaluate.py - held-out split evaluation with latency and per-sample CER.
- scripts/predict_image.py - single-image demo (NOT part of the evaluation).
- notebooks/01_colab_crnn_training.ipynb - full GPU run, 25 cells.

**Design points worth recording:**

- train.py never opens the test manifest. Only train and val are loaded, and the
  best checkpoint is selected on validation CER alone.
- Checkpoints store the config, vocabulary, seed, epoch and metrics, so a saved
  model carries the settings that produced it.
- evaluate.py takes the vocabulary from the checkpoint rather than rebuilding it
  from the manifests. A vocabulary that drifted since training would decode to the
  wrong characters silently, since output indices are fixed at training time.
- CTC loss is computed in float32 even under AMP; it sums many log probabilities
  and is unstable in half precision.
- Latency is measured at batch size 1 and reported with the batch size and device
  stated, because a per-image time taken from a large batch is not comparable.

**Local smoke run: `--limit-train 300 --limit-val 100 --epochs 2 --batch-size 8`**

| Epoch | train_loss | val_loss | val_CER | val_WER | train_s | val_s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.6920 | 3.2535 | 1.0000 | 1.0000 | 372.2 | 48.0 |
| 2 | 3.2095 | 3.3110 | 1.0000 | 1.0000 | 336.5 | 36.5 |

Exit code 0. **CER 1.0 is the expected outcome here, not a failure.** Inspecting
the saved predictions confirms the cause: 100 of 100 validation predictions are
completely empty, so the model is still in the all-blanks phase that CTC passes
through before it starts emitting characters. The overfit test needed about 125
steps to leave that phase on 8 repeated samples; this run did 76 steps on 300
varied ones, which is nowhere near enough. The purpose of the run was to verify
the loop, validation, checkpointing and logging, and all of those worked.

**Also verified mechanically with that checkpoint:**

- evaluate.py on the val split (NOT test): metrics, edit counts (2,084 deletions
  against 2,084 reference characters, consistent with empty predictions), latency,
  throughput, worst-sample listing, CSV and JSON output.
- predict_image.py on a real line image, and on a synthetic 12-line page, where
  the aspect-ratio guard fired as intended: "aspect ratio is 1.3 ... This looks
  like a paragraph or a page, which this model cannot read."

**CPU timing, and why the real run goes to Colab.** 372 s for 38 steps at batch 8
is about 9.8 s/step. The full training split is 8,483 samples, which at batch 16
is 530 steps per epoch with roughly twice the work per step, so on the order of
3 hours per epoch and around 88 hours for 30 epochs on this CPU. An estimate from
one measurement, not a benchmark, but the conclusion is not marginal.

**Still not done:** no real training run, so there is no usable model yet; test
split still untouched; no printed data; no VLM.

---
