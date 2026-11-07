"""Is BatchNorm the reason evaluation is far worse than training?

    python scripts/diagnose_batchnorm.py --checkpoint outputs/checkpoints/diag3_capacity_last.pt ^
        --split train --limit 200 --device cuda

### The question

Training loss is computed with the model in `train()` mode, where BatchNorm
normalises each batch by **that batch's own** mean and variance. Evaluation runs in
`eval()` mode, where BatchNorm instead uses **running averages** accumulated during
training.

Those running averages are only useful if every batch looked roughly alike. Here
they do not: batches are width-bucketed, so one batch is entirely narrow images and
another is entirely wide ones, with very different statistics. The running average
is then a compromise that matches no real batch, and the model can work in training
and fall apart in evaluation.

That would explain the central puzzle: a model whose training loss fell to 0.14 -
about 87% confidence per character - still decoding at CER 0.36 on the very samples
it had memorised.

### What this script does

Scores the same data three ways and prints the CERs side by side:

1. **eval mode** - BatchNorm uses running statistics. This is what every reported
   number so far used.
2. **batch-statistics mode** - BatchNorm uses each batch's own statistics, exactly
   as during training. Dropout stays off, so BatchNorm is the only thing that
   changes.
3. **batch-statistics, width-bucketed batches** - as above, but batches are
   width-homogeneous as in training.

### Reading the result

- **CER drops sharply in modes 2 and 3** -> confirmed. BatchNorm running statistics
  are the bug. Fix by replacing BatchNorm with a normalisation that has no running
  state and no batch dependence (GroupNorm), which is the standard choice for
  variable-sized inputs.
- **CER stays about the same** -> BatchNorm is not the problem. Look at decoding or
  the CTC target construction next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.dataset import (  # noqa: E402
    LineImageDataset,
    WidthBucketSampler,
    collate_batch,
)
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN  # noqa: E402
from src.data.vocabulary import Vocabulary  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def set_batchnorm_mode(model: nn.Module, *, use_batch_statistics: bool) -> int:
    """Switch only the BatchNorm layers, leaving dropout and the rest in eval mode.

    Returns how many layers were switched, so the output cannot quietly report on a
    model that has no BatchNorm at all.
    """
    switched = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.train(use_batch_statistics)
            switched += 1
    return switched


def score(
    model: CRNN,
    loader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
) -> tuple[float, float, int, int]:
    references: list[str] = []
    predictions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch.images.to(device))
            predictions.extend(greedy_decode(log_probs, vocabulary, batch.input_lengths))
            references.extend(batch.texts)
    result = evaluate(references, predictions)
    exact = sum(r == p for r, p in zip(references, predictions))
    return result.cer, result.wer, exact, len(references)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_cfg, model_cfg = config["data"], config["model"]
    vocabulary = Vocabulary(chars=tuple(checkpoint["vocabulary_chars"]))

    dataset = LineImageDataset(
        PROJECT_ROOT / data_cfg["manifests_dir"] / f"{data_cfg['dataset']}_{args.split}.csv",
        vocabulary,
        project_root=PROJECT_ROOT,
        text_column=data_cfg["text_column"],
        expected_height=int(data_cfg["image_height"]),
        limit=args.limit,
    )

    model = CRNN(
        n_classes=vocabulary.size,
        input_height=int(data_cfg["image_height"]),
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 72)
    print("BatchNorm diagnosis")
    print("=" * 72)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  epoch      : {checkpoint['epoch']}, val CER at save {checkpoint['val_cer']:.4f}")
    print(f"  data       : {args.split} split, {len(dataset)} samples")
    print(f"  device     : {device}")
    print()

    sequential = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )
    bucketed = DataLoader(
        dataset,
        batch_sampler=WidthBucketSampler(dataset.widths, args.batch_size, shuffle=False),
        collate_fn=collate_batch,
    )

    rows: list[tuple[str, float, float, int, int]] = []

    n_bn = set_batchnorm_mode(model, use_batch_statistics=False)
    rows.append(("1. eval mode (running stats)", *score(model, sequential, vocabulary, device)))

    set_batchnorm_mode(model, use_batch_statistics=True)
    rows.append(("2. batch stats, mixed widths", *score(model, sequential, vocabulary, device)))
    rows.append(("3. batch stats, bucketed", *score(model, bucketed, vocabulary, device)))

    set_batchnorm_mode(model, use_batch_statistics=False)

    print(f"  BatchNorm layers switched: {n_bn}")
    if n_bn == 0:
        print("  WARNING: this model has no BatchNorm; the comparison is meaningless.")
    print()
    print(f"  {'mode':<32} {'CER':>8} {'WER':>8} {'exact':>8}")
    print("  " + "-" * 58)
    for label, cer, wer, exact, total in rows:
        print(f"  {label:<32} {cer:>8.4f} {wer:>8.4f} {exact:>5}/{total}")
    print()

    eval_cer = rows[0][1]
    best_batch_cer = min(rows[1][1], rows[2][1])
    if eval_cer - best_batch_cer > 0.05:
        print("  VERDICT: BatchNorm running statistics are a major problem.")
        print(f"  CER improves by {eval_cer - best_batch_cer:.4f} when BatchNorm uses each")
        print("  batch's own statistics instead of the running averages.")
        print()
        print("  Fix: replace BatchNorm2d with GroupNorm in src/crnn/model.py. GroupNorm")
        print("  keeps no running state and does not depend on batch composition, which")
        print("  is the standard choice when input sizes vary as much as they do here.")
    else:
        print("  VERDICT: BatchNorm is NOT the main problem - the modes score alike.")
        print("  Look at CTC target construction and decoding next.")

    # Running statistics themselves, for the record.
    print()
    print("  BatchNorm running statistics (should look stable, not extreme):")
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mean, var = module.running_mean, module.running_var
            print(f"    {name:<12} mean {mean.mean():+.3f} (range {mean.min():+.2f} "
                  f"to {mean.max():+.2f})  var {var.mean():.3f} "
                  f"(range {var.min():.3f} to {var.max():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
