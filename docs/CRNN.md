# Model A - CRNN + CTC

Read [../CLAUDE.md](../CLAUDE.md) first. **Rule 0 applies: update this file whenever
you change CRNN code, config, or results.**

**Status: code complete and verified. No real training run has happened yet.**

---

## What it is

A CRNN built from scratch in this repository. **Nothing is downloaded** - the model
is randomly initialised and learns only from the project's data. 8,761,710
parameters.

Three stages:

1. **CNN** - turns pixels into features and collapses image *height* to 1 while
   keeping *width* as a sequence. A 64-pixel-tall line becomes a row of feature
   vectors, one per horizontal position.
2. **BiLSTM** - reads that row in both directions. Bidirectional matters for Urdu:
   letters join, so a letter's shape depends on neighbours on both sides.
3. **CTC head** - one score per character per frame, plus the blank. CTC handles
   alignment, so no character-level bounding boxes are needed.

The feature sequence runs left to right, which is *visual* order. Urdu renders
right to left, so the leftmost frame corresponds to the *end* of the logical
string. This is handled by keeping transcriptions in logical order and letting the
BiLSTM learn the mapping. **Never reverse the stored string.**

---

## Width arithmetic - verified, do not change casually

Height collapses to 1. Width is downsampled by exactly 4: two 2x2 max-pools halve
it, and every later pool uses stride (2, 1), touching height only.

```
T = floor(floor(W / 2) / 2)      # src/crnn/model.py: output_width()
```

CTC needs `T >= label length + one frame per pair of identical adjacent
characters`. Measured over all 10,604 NUST-UHWR lines:

| Downsampling | T range | Violations | Smallest margin |
| --- | --- | --- | --- |
| W/2 | 100-1085 | 0 | 79 frames |
| **W/4 (chosen)** | **50-542** | **0** | **27 frames** |
| W/8 | 25-271 | **59** | -21 frames |

W/8 would make the loss infinite for 59 samples. **If you change the CNN's pooling,
re-run this check against the manifests before training.** `LineImageDataset` also
re-checks it per row at load time and raises with a clear message.

Input height must be **32 or 64** - the final height-collapsing convolution has a
fixed kernel. 64 goes 64→32→16→8→4→1; the 32 variant exists for the printed
dataset, which ships at height 32.

---

## Files

| File | Purpose |
| --- | --- |
| `src/crnn/model.py` | architecture, `output_width()`, `n_parameters()` |
| `src/crnn/dataset.py` | manifest-backed dataset, padding, CTC lengths |
| `src/crnn/decode.py` | greedy CTC decoding only |
| `src/crnn/train.py` | training loop, validation, checkpointing, CSV log |
| `src/crnn/evaluate.py` | held-out split evaluation, latency, per-sample CER |
| `scripts/overfit_test.py` | pipeline sanity check |
| `scripts/predict_image.py` | single-image demo, **not** part of evaluation |
| `configs/crnn_handwritten.yaml` | all result-affecting settings |
| `notebooks/01_colab_crnn_training.ipynb` | full GPU run |

---

## Gotchas that cost time before - do not undo these

- **Inputs are standardised with the measured TRAINING-split pixel statistics**
  (mean 0.953208, std 0.186242, in `src/crnn/dataset.py`). Raw images are 93%
  near-white, so an unstandardised input is an almost-constant 0.95 with the ink
  carrying all the signal in ~7% of pixels, and the first convolution has no
  BatchNorm to rescue it. `scripts/predict_image.py` applies the identical
  transform - a mismatch there produces nonsense silently.
- **Padding is expressed in the standardised space**, not as raw 1.0. White becomes
  +0.2512. Padding with 1.0 after standardising would make padded columns a large
  positive signal. Never pad with zero either: these are dark ink on light paper,
  so zero padding would paste black blocks onto short lines.
- **`input_lengths` come from each sample's TRUE width**, never the padded batch
  width. Otherwise CTC is told to account for padding as if it were image.
- **CTC loss runs in float32 even under AMP.** It sums many log probabilities and
  is numerically unstable in half precision. `train.py` does `log_probs.float()`.
- **`evaluate.py` takes the vocabulary from the checkpoint**, not from the
  manifests. Output indices are fixed at training time; a vocabulary that drifted
  since would decode to the wrong characters silently.
- **`print(..., flush=True)` on the epoch line.** Python block-buffers stdout when
  it is not a terminal, so without this nothing appears in a notebook cell or a log
  file until the run ends. Use `python -u` when launching too.
- **`RemoveEmptyStrings` must stay out of the jiwer word transform** in
  `src/evaluation/metrics.py`. It drops empty *sentences* from the list, which
  breaks reference/hypothesis alignment the moment an untrained model predicts "".
  This was a real bug caught by the overfit test; four regression tests guard it.
- **`float(loss.detach())`**, not `float(loss)`, in the training loop.
- **Batches are width-bucketed** (`WidthBucketSampler`). With random batching 33.7%
  of every batch tensor was blank padding, which BatchNorm folds into its
  statistics. This is a correctness fix, not a speed option. Note it did **not**
  lift the training plateau on its own - see "Results so far".

---

## Commands

```bat
REM 1. Pipeline sanity check, ~6 min CPU. Expect PASSED, CER 0.0000, 8/8 exact.
python scripts/overfit_test.py --samples 8 --steps 300

REM 2. Local smoke run of the training loop, ~12 min CPU.
REM    val_CER 1.0000 here is CORRECT, not a failure - see "CTC learning stages".
python -m src.crnn.train --config configs/crnn_handwritten.yaml ^
  --limit-train 300 --limit-val 100 --epochs 2 --batch-size 8 --run-name smoke_local

REM 3. Full run - Colab GPU, via the notebook.
python -m src.crnn.train --config configs/crnn_handwritten.yaml ^
  --device cuda --amp --num-workers 2 --epochs 30 --batch-size 16 ^
  --run-name crnn_handwritten

REM 4. Final evaluation. RUN ONCE, when model selection is final.
python -m src.crnn.evaluate ^
  --checkpoint outputs/checkpoints/crnn_handwritten_best.pt --split test

REM 5. Demo on one image (one TEXT LINE per image, not a page).
python scripts/predict_image.py ^
  --checkpoint outputs/checkpoints/crnn_handwritten_best.pt --image my_line.jpg
```

`train.py` never opens the test manifest. Best checkpoint is selected on
validation CER alone.

---

## CTC learning stages - read before calling a run "broken"

CTC passes through predictable stages. A high CER early is not a bug:

1. **All blanks.** Loss falls while every prediction is empty and CER sits at 1.0.
   The overfit test needed about 125 steps to leave this stage on 8 repeated
   samples; real data takes far longer.
2. **Dominant character.** The model emits the most frequent character and a rough
   word rhythm, e.g. `ک ک ک ک`. CER around 0.8, WER slightly above 1.0 because it
   inserts more word-like tokens than it gets right.
3. **Real characters**, then words.

**Diagnosis rule:** if loss falls but CER stays at 1.0 and predictions are empty,
the model is in stage 1 - train longer. If loss falls and predictions are
non-empty but CER stays high, suspect label encoding or decoding. If loss does not
fall at all, suspect the gradient path, learning rate, or CTC lengths.

---

## The training plateau - history, so it is not rediscovered

The first full GPU run (30 epochs, T4, batch 16, lr 3e-4, AMP, random batching)
**stalled**: training loss sat at 2.89 at epoch 3 and 2.84 at epoch 30, best val
CER 0.7258. Train and validation loss were equal, so it was not overfitting.

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| Width padding polluting BatchNorm | 33.7% of each batch tensor was padding | **real bug, fixed** - but a 5-epoch bucketed diagnostic matched the old run epoch for epoch, so **not the cause of the plateau** |
| Inputs never standardised | measured mean 0.953, std 0.186, 93% of pixels near-white | **real, fixed** - not yet proven to lift the plateau |
| AMP destabilising LSTM/CTC | not isolated yet | untested |
| Learning rate | 3e-4 and 1e-3 both plateau at ~2.85-2.90 | unlikely to be the whole story |

**Every checkpoint trained before the normalisation fix is invalid** - it expects
unstandardised input. Discard them, do not evaluate or demo with them.

**Next diagnostic must be 5 epochs, not 30.** The question is only whether train
loss goes below 2.8. Do not spend a full run until it does.

## Results so far - NONE are reportable

| Run | Setup | Result | Verdict |
| --- | --- | --- | --- |
| `overfit_test` | 8 samples, 300 steps, CPU | **CER 0.0000, 8/8 exact**, 384 s | pipeline proven to learn |
| `smoke_local` | 300 train / 100 val, 2 epochs, CPU | CER 1.0000, 100/100 empty | stage 1, as expected |
| `local_demo` | 1000 train / 100 val, 10 epochs, lr 1e-3, CPU | CER 0.8143, 1.84 h | stage 2, demo only |

**`local_demo_best.pt` must never appear in a results table.** It predicts the
most common character and nothing else. It exists so the demo script produces
*something*.

### Why the real run needs a GPU

Measured: 372 s for 38 steps at batch 8 on CPU, about 9.8 s/step. The full training
split at batch 16 is 530 steps per epoch with roughly twice the work per step, so
on the order of **3 hours per epoch, ~88 hours for 30 epochs** on this CPU - and
that CPU is a mobile i5 that will thermally throttle over days.

`local_demo` did about 1,250 steps. The planned Colab run is about **16,000 steps
on eight times the data**. CPU is not a slower route to the same answer within any
practical budget.

---

## Next steps

1. **Fix the Colab checkpoint path** (open issue 1 in CLAUDE.md) so checkpoints
   land on Drive every epoch. The user already lost a run to this.
2. Run `notebooks/01_colab_crnn_training.ipynb` end to end on a free T4.
3. Record in `reports/experiment_log.md`: GPU name and VRAM, torch version, git
   commit, epochs run, best epoch, best validation CER, training time, peak GPU
   memory.
4. Evaluate once on test; save predictions and metrics.
5. Measure CRNN inference latency **on GPU** as well, so the latency and memory
   columns are comparable with the VLM's.
6. Only then start the printed domain (height 32, word-level) - it is a data swap,
   not new modelling.
