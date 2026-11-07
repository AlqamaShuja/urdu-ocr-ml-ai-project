# Comparative Evaluation of CRNN and Fine-Tuned Vision-Language Models for Printed and Handwritten Urdu Text Recognition

Master's-level Machine Learning & AI project, NUST (PNEC).

## Research question

How do a CRNN + CTC baseline and a parameter-efficiently fine-tuned small
vision-language model (VLM) compare on Urdu text recognition, measured separately
on **printed** and **handwritten** text under an identical evaluation protocol?

Four experiment groups, no assumed winner:

| # | Model | Domain |
| --- | --- | --- |
| 1 | CRNN + CTC | Printed Urdu |
| 2 | Fine-tuned VLM | Printed Urdu |
| 3 | CRNN + CTC | Handwritten Urdu |
| 4 | Fine-tuned VLM | Handwritten Urdu |

Reported metrics: CER, WER, inference latency, parameter count, peak GPU memory,
training time. Printed and handwritten results are always reported separately.

## Project status

| Step | Description | Status |
| --- | --- | --- |
| 1 | Environment setup and verification | **complete** (2026-11-02) |
| 2a | Dataset candidates verified from official pages | **complete** (2026-11-02) |
| 2b | Handwritten data downloaded, audited, manifests built | **complete** (2026-11-02) |
| 2c | Printed data (licence enquiry pending) | **blocked** |
| 3 | Shared normalization and CER/WER code | **complete**, 67 tests passing |
| 4a | CRNN model, dataset, CTC decoding | **complete** (2026-11-02) |
| 4b | CRNN tiny overfit test | **PASSED**, CER 0.0000, 8/8 exact |
| 4c | Training, evaluation and demo scripts | **complete**, smoke run verified |
| 4d | CRNN real training run (Colab GPU) | **next** - no trained model yet |
| 5 | VLM zero-shot, then LoRA/PEFT fine-tuning | not started |
| 6 | Controlled comparison and report deliverables | not started |

**Handwritten data is ready.** NUST-UHWR is downloaded, checksum-verified, audited
and turned into manifests: 8,483 train / 1,060 validation / 1,061 test.

**Printed data is blocked.** Synthetic-R-Urdu states no licence anywhere, so an
enquiry must be answered before it can be used. See
[reports/licence_enquiry_iiit.md](reports/licence_enquiry_iiit.md).

No model has been trained yet.

## Experiment design

Two domains are run as two separate tasks, because the available data differs in
granularity. CRNN is compared against the VLM *within* each domain; printed and
handwritten numbers are never compared to each other.

| # | Model | Domain | Data | Granularity |
| --- | --- | --- | --- | --- |
| 1 | CRNN + CTC | Printed | Synthetic-R-Urdu | word |
| 2 | VLM (zero-shot, then LoRA) | Printed | Synthetic-R-Urdu | word |
| 3 | CRNN + CTC | Handwritten | NUST-UHWR | line |
| 4 | VLM (zero-shot, then LoRA) | Handwritten | NUST-UHWR | line |

Within each row pair both models see identical train/validation/test examples and
are scored by identical code in [src/evaluation/](src/evaluation/). The reasoning
behind this design, and its consequences, is in
[reports/dataset_card.md](reports/dataset_card.md).

## Where each task runs

| Task | Machine | Why |
| --- | --- | --- |
| Coding, notebooks, Git | Local Windows PC | No GPU needed |
| Dataset inspection, manifest building, audits | Local | CPU and disk work |
| Text normalization, CER/WER evaluation | Local | Pure CPU |
| CRNN tiny overfit test (a handful of samples) | Local CPU | Minutes, proves the pipeline learns |
| Full CRNN training | Google Colab GPU | Too slow on 4 CPU cores |
| VLM zero-shot inference | Google Colab GPU | Needs far more than 2 GB VRAM |
| VLM LoRA/PEFT fine-tuning | Google Colab GPU | Not feasible locally |
| Charts, error analysis, write-up | Local | CPU only |

The local NVIDIA Quadro M620 has **2 GB VRAM**. It is not used for training in this
project and cannot fine-tune a modern VLM. The local PyTorch install is CPU-only
on purpose, so no CUDA/driver compatibility is assumed.

## Local setup (Windows PowerShell)

Run these from the project root, `E:\alqama_nust\urdu_ocr_project`.

### 1. Verify the base tools

```powershell
python --version        # expect Python 3.12.x
python -m pip --version
git --version
```

### 2. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

The prompt should now start with `(.venv)`. Confirm the right interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

### 3. Install the local packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install PyTorch (CPU)

PyTorch is deliberately not in `requirements.txt`, because on Windows the CPU build
comes from a separate index:

```powershell
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### Why the versions are pinned

Every package in `requirements.txt` is pinned to an exact version. An unpinned
install resolves to the newest release of each package, which here pulled in
pandas 3.x and opencv 5.x. Those are major-version jumps whose APIs differ from
almost all published OCR tutorials and documentation, which costs debugging time
for no research benefit. The pins stay on the mature line of each package:
numpy 2.2, pandas 2.3, opencv 4.14, pillow 11.3, matplotlib 3.10, jiwer 3.1.

Confirm nothing conflicts after any dependency change:

```powershell
python -m pip check      # expect "No broken requirements found."
```

### 5. Register the Jupyter kernel

```powershell
python -m ipykernel install --user --name urdu-ocr --display-name "Python (Urdu OCR)"
```

Always select the **Python (Urdu OCR)** kernel in notebooks, so they use this
virtual environment rather than a global Python.

### 6. Verify everything

```powershell
python scripts/check_env.py
```

Expect `Result: PASSED`. The script checks the interpreter, the core packages, a
CPU tensor op, a CTC loss forward/backward pass, and Urdu Unicode round-tripping.
`CUDA available: False` is the expected and correct outcome locally.

### 7. Lock the versions

```powershell
python scripts/freeze_env.py
```

This writes `requirements-lock.txt` with the exact resolved versions. Commit it.

## Colab setup

Open `notebooks/00_colab_gpu_smoke_test.ipynb` in Google Colab and run it at the
start of **every** session, before any training. It reports the assigned GPU, the
session's PyTorch/CUDA versions and the usable VRAM, all of which must be recorded
in `reports/experiment_log.md` because Colab assigns different hardware each time.

## Data pipeline

Reproduce the handwritten data from scratch. Downloaded datasets are untrusted
data, so each script takes the data path as an argument and is run with `python -I`
(isolated mode), which stops anything inside the data directory from being imported.

```powershell
# 1. Download NUST-UHWR (159 MB, CC BY 4.0) into its own empty directory
mkdir data\raw\nust_uhwr
curl -L -o data\raw\nust_uhwr\UHWR.zip `
  "https://zenodo.org/records/21674269/files/UHWR.zip?download=1"

# 2. Verify the checksum published on the Zenodo record before trusting the file
certutil -hashfile data\raw\nust_uhwr\UHWR.zip MD5
#    must print e367ff83c147671f4ae569bc63480af9

# 3. Extract
tar -xf data\raw\nust_uhwr\UHWR.zip -C data\raw\nust_uhwr\

# 4. Audit before using it for anything
python -I scripts/audit_nust_uhwr.py `
  --root data/raw/nust_uhwr/UHWR `
  --out reports/dataset_audit_nust_uhwr.md

# 5. Build the manifests both models read
python -m src.data.prepare_dataset --dataset nust_uhwr `
  --root data/raw/nust_uhwr/UHWR --out data/manifests

# 6. Build the CRNN vocabulary from the TRAIN split only
python -m src.data.vocabulary --dataset nust_uhwr --manifests data/manifests
```

Step 4 exits non-zero when it finds a problem. Two were found and are resolved by
documented exclusions in [reports/dataset_card.md](reports/dataset_card.md), so a
non-zero exit is currently expected; read the report rather than ignoring it.

### Verifying the evaluation code

```powershell
python scripts/test_evaluation.py     # expect "67 passed, 0 failed"
```

This covers the normalizer and the CER/WER arithmetic. Run it after any change to
[src/evaluation/](src/evaluation/), because that code produces every number in the
thesis.

### Verifying the CRNN pipeline

Before any GPU training, confirm the pipeline can learn by memorising a handful of
images. Roughly six minutes on CPU:

```powershell
python scripts/overfit_test.py --samples 8 --steps 300
```

Expect `PASSED`, with CER reaching 0.0000 and 8/8 exact matches. A model that
memorises eight images is useless - the point is that if this fails, something in
data loading, label encoding, CTC or decoding is broken, and finding that out here
costs minutes instead of GPU hours.

CER staying at 1.0 for the first ~100 steps while the loss falls is normal: CTC
models first learn to emit blanks everywhere, then start emitting characters.

## Training and evaluation

### 1. Local smoke run (verify the loop, ~12 minutes on CPU)

```powershell
python -m src.crnn.train --config configs/crnn_handwritten.yaml `
  --limit-train 300 --limit-val 100 --epochs 2 --batch-size 8 --run-name smoke_local
```

Expect it to finish with exit code 0. **`val_CER 1.0000` here is correct, not a
failure:** 76 steps is far too few to leave the all-blanks phase CTC passes
through before it emits any characters. This run checks the plumbing, not quality.

### 2. Full training (Colab GPU)

Open [notebooks/01_colab_crnn_training.ipynb](notebooks/01_colab_crnn_training.ipynb)
in Colab and work through it. It verifies the GPU, caches the dataset on Drive with
an MD5 check, runs its own short smoke run first, then trains and saves a
checkpoint to Drive after every epoch so a disconnect does not lose the run.

Training on this CPU would take roughly 3 hours per epoch, so the real run needs a
GPU. The notebook measures the actual time rather than relying on that estimate.

### 3. Final evaluation on the held-out test split

```powershell
python -m src.crnn.evaluate --checkpoint outputs/checkpoints/crnn_handwritten_best.pt `
  --split test
```

**Run this once**, after training is finished and nothing further will be tuned.
Seeing a test number and then changing a setting turns the test set into a second
validation set. Writes per-sample predictions and a metrics JSON to `outputs/`.

### 4. Read a single image (demo)

```powershell
python scripts/predict_image.py `
  --checkpoint outputs/checkpoints/crnn_handwritten_best.pt --image my_line.jpg
```

**One text line per image.** The model has never seen a full page; given one it
squashes every line into 64 pixels and the output is meaningless. The script warns
when an image looks like a page. This is a demo - reported numbers come from step 3.

## Repository layout

```text
.
├── configs/            # YAML experiment configs (one per run)
├── data/
│   ├── raw/            # Downloaded datasets - never committed
│   ├── processed/       # Preprocessed images - never committed
│   └── manifests/      # CSV/JSON: image path, transcription, official split
├── notebooks/          # Exploration and the Colab smoke test
├── scripts/            # Environment checks and utilities
├── src/
│   ├── data/           # Dataset inspection, manifest building, vocabulary
│   ├── crnn/           # CRNN model, dataset, training, CTC decoding, evaluation
│   ├── vlm/            # VLM data prep, LoRA fine-tuning, evaluation
│   └── evaluation/     # Shared text normalization, CER/WER, comparison
├── outputs/            # Checkpoints, predictions, metrics, figures - not committed
└── reports/            # Dataset card, experiment log
```

`src/evaluation/` is shared by both models on purpose: the CRNN and the VLM must be
scored by identical normalization and metric code for the comparison to be fair.

## Troubleshooting

### torch WinError 126 (`shm.dll or one of its dependencies`)

```
OSError: [WinError 126] The specified module could not be found.
Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

The PyTorch wheel is fine. The missing dependency is the **Microsoft Visual C++
2015-2022 x64 Redistributable**, which PyTorch's compiled DLLs link against and
which Windows does not always ship. Confirm it with:

```powershell
Test-Path C:\Windows\System32\vcruntime140_1.dll   # False means it is missing
```

Install it (choose one), then restart the terminal:

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 -e
```

or download and run <https://aka.ms/vs/17/release/vc_redist.x64.exe>.

Re-verify:

```powershell
python scripts/check_env.py
```

### PowerShell refuses to run `Activate.ps1`

```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

Either use `cmd` with `.venv\Scripts\activate.bat`, or allow local scripts for
your own user only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Urdu text prints as `?????` in the terminal

A terminal display issue, not a data problem. `scripts/check_env.py` forces UTF-8
on stdout. If it still looks wrong, use a font with Urdu coverage, or inspect the
code points instead of the rendered glyphs. Never "fix" this by reversing or
re-encoding the stored string.

### A notebook cannot find the project packages

The notebook is on the wrong kernel. Select **Python (Urdu OCR)** via
*Kernel > Change kernel*, and confirm inside the notebook with:

```python
import sys; print(sys.executable)   # must point into .venv
```

## Non-negotiable project rules

- Official test splits are preserved. Never train, tune or select prompts on test data.
- Printed and handwritten domains are never merged into one evaluation set.
- Prompt-only inference is labelled **zero-shot**, never "fine-tuned".
- Urdu strings are stored in logical order. The Unicode string is never reversed
  just because the script renders right-to-left.
- Dataset sizes, licences, splits and results are taken from the official source or
  from an actual run. Nothing is estimated or invented.
- No web frontend or backend. This is an experimental research project.
