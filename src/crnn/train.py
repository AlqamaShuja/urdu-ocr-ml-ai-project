"""Train the CRNN and select a checkpoint on validation CER.

Local smoke run, a few minutes on CPU:

    python -m src.crnn.train --config configs/crnn_handwritten.yaml ^
        --limit-train 300 --limit-val 100 --epochs 2 --batch-size 8 ^
        --run-name smoke_local

Full run, on a Colab GPU:

    python -m src.crnn.train --config configs/crnn_handwritten.yaml ^
        --device cuda --amp --num-workers 2

### What this script guarantees

- **The test split is never read.** Only train and validation manifests are opened.
  Model selection uses validation CER alone, so the test set stays untouched until
  the final evaluation.
- **The checkpoint is self-describing.** It stores the config, the vocabulary, the
  seed, the epoch and the validation metrics, so a result can be traced back to the
  settings that produced it without consulting notes.
- **Metrics come from the shared code** in `src/evaluation/`, the same code the VLM
  is scored with.
- **Resource use is measured, not estimated.** Wall clock per epoch and peak memory
  go into the run log, because the project reports them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.functional import ctc_loss
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.dataset import (  # noqa: E402
    LineImageDataset,
    WidthBucketSampler,
    collate_batch,
)
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN  # noqa: E402
from src.data.vocabulary import BLANK_INDEX, Vocabulary  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def _random_batch_padding_fraction(widths: list[int], batch_size: int) -> float:
    """Padding share a RANDOM batching of these widths would produce.

    Reported next to the bucketed figure so the run log shows what bucketing
    actually bought. Averaged over a few random shuffles.
    """
    import statistics

    samples = []
    for _ in range(5):
        order = list(range(len(widths)))
        random.shuffle(order)
        total_padding = total_cells = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            widest = max(widths[i] for i in batch)
            total_cells += widest * len(batch)
            total_padding += sum(widest - widths[i] for i in batch)
        samples.append(total_padding / total_cells)
    return statistics.mean(samples)


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak GPU memory in MB, or None on CPU where it is not measurable this way."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_cer: float
    val_wer: float
    train_seconds: float
    val_seconds: float
    peak_memory_mb: float | None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default=None, help="overrides config run_name")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    parser.add_argument("--no-bucket", action="store_true",
                        help="disable width-bucketed batching (not recommended)")
    flip = parser.add_mutually_exclusive_group()
    flip.add_argument("--flip", dest="flip", action="store_true", default=None,
                      help="mirror images left-to-right (see dataset.py)")
    flip.add_argument("--no-flip", dest="flip", action="store_false",
                      help="feed images in their original orientation")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    return parser


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Apply command line overrides, returning a note for each one applied."""
    applied: list[str] = []
    mapping = {
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "lr": ("train", "lr"),
        "seed": ("train", "seed"),
        "num_workers": ("train", "num_workers"),
        "limit_train": ("train", "limit_train"),
        "limit_val": ("train", "limit_val"),
    }
    for attr, (section, key) in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            applied.append(f"{section}.{key}: {config[section][key]} -> {value}")
            config[section][key] = value
    if args.amp:
        applied.append(f"train.amp: {config['train']['amp']} -> True")
        config["train"]["amp"] = True
    if args.no_bucket:
        applied.append("train.bucket_by_width: True -> False")
        config["train"]["bucket_by_width"] = False
    if args.flip is not None:
        applied.append(f"data.flip_horizontal: {config['data']['flip_horizontal']} -> {args.flip}")
        config["data"]["flip_horizontal"] = args.flip
    if args.run_name:
        applied.append(f"run_name: {config['run_name']} -> {args.run_name}")
        config["run_name"] = args.run_name
    return applied


def run_validation(
    model: CRNN,
    loader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
) -> tuple[float, float, float, list[str], list[str], list[str]]:
    """Validation pass. Returns loss, CER, WER and the aligned ids/refs/predictions."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    image_ids: list[str] = []
    references: list[str] = []
    predictions: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch.images.to(device, non_blocking=True)
            log_probs = model(images)
            loss = ctc_loss(
                log_probs.float(),
                batch.targets.to(device),
                batch.input_lengths.to(device),
                batch.target_lengths.to(device),
                blank=BLANK_INDEX,
                zero_infinity=True,  # one bad sample must not destroy the epoch's number
            )
            total_loss += float(loss)
            n_batches += 1

            predictions.extend(greedy_decode(log_probs, vocabulary, batch.input_lengths))
            references.extend(batch.texts)
            image_ids.extend(batch.image_ids)

    result = evaluate(references, predictions)
    return (
        total_loss / max(n_batches, 1),
        result.cer,
        result.wer,
        image_ids,
        references,
        predictions,
    )


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    overrides = apply_overrides(config, args)

    data_cfg, model_cfg, train_cfg, out_cfg = (
        config["data"], config["model"], config["train"], config["output"]
    )
    run_name = config["run_name"]
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but no CUDA device is available.")
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    if train_cfg["amp"] and device.type != "cuda":
        print("note: amp requested but device is CPU; running in full precision.\n")

    set_seed(int(train_cfg["seed"]))

    manifests = PROJECT_ROOT / data_cfg["manifests_dir"]
    dataset_name = data_cfg["dataset"]
    vocabulary = Vocabulary.from_json(manifests / f"{dataset_name}_vocab.json")

    common = {
        "vocabulary": vocabulary,
        "project_root": PROJECT_ROOT,
        "text_column": data_cfg["text_column"],
        "expected_height": int(data_cfg["image_height"]),
        "flip_horizontal": bool(data_cfg.get("flip_horizontal", True)),
    }
    # Only train and val are opened. The test manifest is deliberately not touched.
    train_set = LineImageDataset(
        manifests / f"{dataset_name}_train.csv", limit=train_cfg["limit_train"], **common
    )
    val_set = LineImageDataset(
        manifests / f"{dataset_name}_val.csv", limit=train_cfg["limit_val"], **common
    )

    batch_size = int(train_cfg["batch_size"])
    loader_kwargs = {
        "collate_fn": collate_batch,
        "num_workers": int(train_cfg["num_workers"]),
        "pin_memory": device.type == "cuda",
    }

    bucket = bool(train_cfg.get("bucket_by_width", True))
    if bucket:
        # Width-homogeneous batches. See WidthBucketSampler for why this is a
        # correctness fix rather than a speed option: BatchNorm statistics are
        # computed over padding too, and mixed-width batches destabilise them.
        train_sampler = WidthBucketSampler(train_set.widths, batch_size, shuffle=True)
        val_sampler = WidthBucketSampler(val_set.widths, batch_size, shuffle=False)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, **loader_kwargs)
        padding_before = _random_batch_padding_fraction(train_set.widths, batch_size)
        padding_after = train_sampler.padding_fraction(train_set.widths)
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs
        )
        padding_before = padding_after = None

    model = CRNN(
        n_classes=vocabulary.size,
        input_height=int(data_cfg["image_height"]),
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_cer = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_cer = float(checkpoint.get("best_val_cer", float("inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best val CER {best_cer:.4f}\n")

    checkpoints_dir = PROJECT_ROOT / out_cfg["checkpoints_dir"]
    metrics_dir = PROJECT_ROOT / out_cfg["metrics_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir / f"{run_name}_train_log.csv"

    print("=" * 72)
    print(f"CRNN training: {run_name}")
    print("=" * 72)
    print(f"  device        : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  mixed precision: {use_amp}")
    print(f"  train samples : {len(train_set)}")
    print(f"  val samples   : {len(val_set)}  (test split NOT opened)")
    print(f"  vocabulary    : {vocabulary.size} CTC outputs")
    print(f"  image flip    : {data_cfg.get('flip_horizontal', True)}"
          f"  (Urdu is right-to-left; see src/crnn/dataset.py)")
    print(f"  parameters    : {model.n_parameters():,}")
    print(f"  batch size    : {train_cfg['batch_size']}, lr {train_cfg['lr']}, "
          f"seed {train_cfg['seed']}")
    if bucket:
        print(f"  batching      : width-bucketed. Padding in a batch tensor "
              f"{padding_before:.1%} -> {padding_after:.1%}")
    else:
        print("  batching      : RANDOM. BatchNorm statistics will be polluted by "
              "padding; see WidthBucketSampler.")
    print(f"  epochs        : {start_epoch}..{train_cfg['epochs']}")
    if overrides:
        print("  overrides     : " + "; ".join(overrides))
    print()

    history: list[EpochResult] = []
    print(f"{'epoch':>6} {'train_loss':>11} {'val_loss':>9} {'val_CER':>8} {'val_WER':>8} "
          f"{'train_s':>8} {'val_s':>7}  best")
    print("-" * 72, flush=True)

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        started = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch.images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                log_probs = model(images)
            # CTC is run in float32 even under AMP: the loss sums many log
            # probabilities and is numerically fragile in half precision.
            loss = ctc_loss(
                log_probs.float(),
                batch.targets.to(device),
                batch.input_lengths.to(device),
                batch.target_lengths.to(device),
                blank=BLANK_INDEX,
                zero_infinity=True,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            # detach first: the loss still carries a graph here, and converting it
            # to a Python float without detaching warns and can hold the graph alive.
            epoch_loss += float(loss.detach())
            n_batches += 1

        train_seconds = time.perf_counter() - started
        train_loss = epoch_loss / max(n_batches, 1)

        val_started = time.perf_counter()
        val_loss, val_cer, val_wer, image_ids, references, predictions = run_validation(
            model, val_loader, vocabulary, device
        )
        val_seconds = time.perf_counter() - val_started

        is_best = val_cer < best_cer
        if is_best:
            best_cer = val_cer

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_cer=val_cer,
            val_wer=val_wer,
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            peak_memory_mb=peak_memory_mb(device),
        )
        history.append(result)

        # flush: when stdout is a file or a notebook cell rather than a terminal,
        # Python block-buffers it and no progress appears until the run ends.
        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>9.4f} {val_cer:>8.4f} "
              f"{val_wer:>8.4f} {train_seconds:>8.1f} {val_seconds:>7.1f}  "
              f"{'<- best' if is_best else ''}", flush=True)

        # Written every epoch, so an interrupted run still leaves a usable log.
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch", "train_loss", "val_loss", "val_cer", "val_wer",
                    "train_seconds", "val_seconds", "peak_memory_mb",
                ],
            )
            writer.writeheader()
            for row in history:
                writer.writerow(vars(row))

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "val_loss": val_loss,
            "best_val_cer": best_cer,
            "config": config,
            "vocabulary_chars": list(vocabulary.chars),
            "n_parameters": model.n_parameters(),
            "seed": int(train_cfg["seed"]),
            "device": str(device),
        }
        torch.save(checkpoint, checkpoints_dir / f"{run_name}_last.pt")
        if is_best:
            torch.save(checkpoint, checkpoints_dir / f"{run_name}_best.pt")
            # Validation predictions from the selected epoch, for error analysis.
            predictions_path = (
                PROJECT_ROOT / "outputs" / "predictions" / f"{run_name}_val_best.csv"
            )
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            with predictions_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_id", "reference", "prediction"])
                writer.writerows(zip(image_ids, references, predictions))

    print()
    total_train = sum(r.train_seconds for r in history)
    best = min(history, key=lambda r: r.val_cer) if history else None

    print("Summary")
    print("-" * 72)
    if best is not None:
        print(f"  best epoch      : {best.epoch}")
        print(f"  best val CER    : {best.val_cer:.4f}")
        print(f"  best val WER    : {best.val_wer:.4f}")
    print(f"  total train time: {total_train:.1f}s over {len(history)} epoch(s)")
    if device.type == "cuda":
        print(f"  peak GPU memory : {peak_memory_mb(device):.0f} MB")
    print(f"  log             : {log_path}")
    print(f"  best checkpoint : {checkpoints_dir / f'{run_name}_best.pt'}")
    print()
    print("  The test split was not opened. Run src/crnn/evaluate.py once model")
    print("  selection is final.")

    summary_path = metrics_dir / f"{run_name}_train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "device": str(device),
                "mixed_precision": use_amp,
                "n_parameters": model.n_parameters(),
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "epochs_run": len(history),
                "best_epoch": best.epoch if best else None,
                "best_val_cer": best.val_cer if best else None,
                "best_val_wer": best.val_wer if best else None,
                "total_train_seconds": total_train,
                "peak_memory_mb": peak_memory_mb(device),
                "bucket_by_width": bucket,
                "padding_fraction_random": padding_before,
                "padding_fraction_bucketed": padding_after,
                "overrides": overrides,
                "config": config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
