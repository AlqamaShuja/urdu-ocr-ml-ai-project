"""Tiny overfit test: can the training pipeline learn at all?

    python scripts/overfit_test.py --samples 8 --steps 300

Trains on a handful of images, over and over, with no validation and no
regularisation beyond what the model already has. The point is **not** a good
model. A model that memorises eight images is useless. The point is that if the
pipeline cannot memorise eight images, something is broken, and finding that out
in three minutes on a CPU is far cheaper than finding it out after an hour of GPU
training.

It exercises the whole chain end to end: manifest reading, image loading, label
encoding, padding and lengths, the CNN's shape arithmetic, the CTC loss, greedy
decoding, and the shared CER metric.

**Pass criterion: final CER on the training samples is near zero.** If loss falls
but CER stays high, decoding or label encoding is wrong, not the model. If loss
does not fall at all, the gradient path is broken.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import ctc_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from src.crnn.dataset import LineImageDataset, collate_batch  # noqa: E402
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN  # noqa: E402
from src.data.vocabulary import BLANK_INDEX, Vocabulary  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nust_uhwr")
    parser.add_argument("--manifests", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cer-threshold", type=float, default=0.02,
                        help="pass if final CER on the training samples is below this")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 68)
    print("CRNN tiny overfit test")
    print("=" * 68)

    vocabulary = Vocabulary.from_json(args.manifests / f"{args.dataset}_vocab.json")

    # The narrowest images are chosen on purpose: fewer frames means a faster step,
    # and width has nothing to do with whether the pipeline can learn. Narrow lines
    # are also the tightest CTC case, so this is the harder end, not the easier one.
    full = LineImageDataset(
        args.manifests / f"{args.dataset}_train.csv",
        vocabulary,
        project_root=PROJECT_ROOT,
        expected_height=args.height,
    )
    narrowest = sorted(range(len(full)), key=lambda i: full.widths[i])[: args.samples]
    samples = [full[i] for i in narrowest]
    batch = collate_batch(samples)

    device = torch.device("cpu")
    model = CRNN(n_classes=vocabulary.size, input_height=args.height).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"  samples        : {len(samples)} (the narrowest in the training split)")
    print(f"  widths         : {[s.image.shape[2] for s in samples]}")
    print(f"  label lengths  : {[s.target_length for s in samples]}")
    print(f"  frames (T)     : {batch.input_lengths.tolist()}")
    print(f"  padded to width: {batch.images.shape[3]}")
    print(f"  vocabulary     : {vocabulary.size} CTC outputs (blank at {BLANK_INDEX})")
    print(f"  parameters     : {model.n_parameters():,}")
    print(f"  device         : {device}")
    print(f"  steps          : {args.steps}, lr {args.lr}, seed {args.seed}")
    print()

    images = batch.images.to(device)
    targets = batch.targets.to(device)
    input_lengths = batch.input_lengths.to(device)
    target_lengths = batch.target_lengths.to(device)

    print(f"{'step':>6}  {'loss':>10}  {'CER':>8}   elapsed")
    print("-" * 44)

    started = time.perf_counter()
    first_loss: float | None = None
    model.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        log_probs = model(images)
        loss = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=BLANK_INDEX,
            zero_infinity=False,  # an infinite loss must be loud, not silently zeroed
        )
        if not torch.isfinite(loss):
            print(f"\nFAIL: loss became {loss.item()} at step {step}.")
            print("An infinite CTC loss means some target is longer than its frame count.")
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()

        if step % 25 == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                predictions = greedy_decode(model(images), vocabulary, batch.input_lengths)
            model.train()
            cer = evaluate(batch.texts, predictions).cer
            print(f"{step:>6}  {loss.item():>10.4f}  {cer:>8.4f}   "
                  f"{time.perf_counter() - started:>6.1f}s")

    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        predictions = greedy_decode(model(images), vocabulary, batch.input_lengths)
    result = evaluate(batch.texts, predictions)

    print()
    print("Final predictions versus ground truth")
    print("-" * 68)
    for image_id, reference, hypothesis in zip(batch.image_ids, batch.texts, predictions):
        mark = "OK  " if reference == hypothesis else "DIFF"
        print(f"  [{mark}] {image_id}")
        print(f"         ref : {reference}")
        print(f"         pred: {hypothesis}")
    print()

    print("Result")
    print("-" * 68)
    print(f"  first loss   : {first_loss:.4f}")
    print(f"  final loss   : {loss.item():.4f}")
    print(f"  final CER    : {result.cer:.4f}  (threshold {args.cer_threshold})")
    print(f"  final WER    : {result.wer:.4f}")
    print(f"  exact matches: {sum(r == h for r, h in zip(batch.texts, predictions))}"
          f"/{len(predictions)}")
    print(f"  wall clock   : {elapsed:.1f}s on {device}")
    print()

    if result.cer <= args.cer_threshold:
        print("PASSED. The pipeline can learn: data loading, label encoding, CTC and")
        print("decoding all work together. This says nothing about generalisation -")
        print("the model has simply memorised these samples, which is the intent.")
        return 0

    print("FAILED. The pipeline did not memorise a handful of samples.")
    if first_loss is not None and loss.item() < first_loss * 0.5:
        print("Loss fell but CER stayed high, which points at label encoding or")
        print("decoding rather than the model itself.")
    else:
        print("Loss barely moved, which points at the gradient path, the learning")
        print("rate, or the CTC input/target lengths.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
