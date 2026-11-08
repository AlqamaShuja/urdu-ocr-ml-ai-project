"""Evaluate a trained CRNN checkpoint on a held-out split.

    python -m src.crnn.evaluate --checkpoint outputs/checkpoints/crnn_handwritten_best.pt ^
        --split test

**Run this on `test` exactly once, when model selection is finished.** Every
hyperparameter choice, every early-stopping decision and every prompt choice must
already have been made on the validation split. Re-running on test after seeing a
test number and then changing something turns the test set into a validation set,
and the reported result stops meaning what it claims to mean.

What it writes:

- `outputs/predictions/<run>_<split>.csv` - one row per sample with the reference,
  the prediction and that sample's CER, for error analysis.
- `outputs/metrics/<run>_<split>.json` - the metrics, with the edit counts behind
  them, plus latency and the environment the numbers were measured in.

Latency is measured per image at batch size 1 and reported with the batch size and
device stated, because a per-image time from a large batch is not comparable to one
from a single-image call.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.dataset import LineImageDataset, collate_batch  # noqa: E402
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN  # noqa: E402
from src.data.vocabulary import Vocabulary  # noqa: E402
from src.evaluation.metrics import evaluate as score  # noqa: E402
from src.evaluation.metrics import per_sample_cer  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="for smoke tests only")
    parser.add_argument("--latency-samples", type=int, default=50,
                        help="images timed one at a time for the latency number")
    parser.add_argument("--run-name", default=None, help="overrides the checkpoint's run name")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but no CUDA device is available.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_cfg, model_cfg = config["data"], config["model"]
    run_name = args.run_name or config["run_name"]

    # The vocabulary is taken from the checkpoint, not rebuilt from the manifests.
    # A vocabulary that drifted since training would silently decode to the wrong
    # characters, since the model's output indices are fixed at training time.
    vocabulary = Vocabulary(chars=tuple(checkpoint["vocabulary_chars"]))

    manifests = PROJECT_ROOT / data_cfg["manifests_dir"]
    dataset_name = data_cfg["dataset"]
    dataset = LineImageDataset(
        manifests / f"{dataset_name}_{args.split}.csv",
        vocabulary,
        project_root=PROJECT_ROOT,
        text_column=data_cfg["text_column"],
        expected_height=int(data_cfg["image_height"]),
        flip_horizontal=bool(data_cfg.get("flip_horizontal", True)),
        limit=args.limit,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
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
    print(f"CRNN evaluation: {run_name} on the {args.split.upper()} split")
    print("=" * 72)
    print(f"  checkpoint    : {args.checkpoint}")
    print(f"  trained epochs: {checkpoint['epoch']}  (val CER at save: "
          f"{checkpoint['val_cer']:.4f})")
    print(f"  samples       : {len(dataset)}")
    print(f"  vocabulary    : {vocabulary.size} CTC outputs")
    print(f"  parameters    : {model.n_parameters():,}")
    print(f"  device        : {device}")
    print(f"  decoding      : greedy, no external language model")
    print()
    if args.split == "test":
        print("  NOTE: this is the held-out test split. Report this number as final")
        print("  and do not tune anything after seeing it.")
        print()

    image_ids: list[str] = []
    references: list[str] = []
    predictions: list[str] = []

    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch.images.to(device))
            predictions.extend(greedy_decode(log_probs, vocabulary, batch.input_lengths))
            references.extend(batch.texts)
            image_ids.extend(batch.image_ids)
    batched_seconds = time.perf_counter() - started

    result = score(references, predictions)
    sample_cers = per_sample_cer(references, predictions)

    # Latency at batch size 1, which is the honest number for a per-image claim.
    n_latency = min(args.latency_samples, len(dataset))
    with torch.no_grad():
        for i in range(min(3, n_latency)):  # warm-up, excluded from the timing
            model(collate_batch([dataset[i]]).images.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_started = time.perf_counter()
        for i in range(n_latency):
            single = collate_batch([dataset[i]])
            log_probs = model(single.images.to(device))
            greedy_decode(log_probs, vocabulary, single.input_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_seconds = time.perf_counter() - latency_started

    ms_per_image = latency_seconds / n_latency * 1000

    print("Results")
    print("-" * 72)
    print(f"  {result.summary()}")
    print()
    print(f"  latency      : {ms_per_image:.1f} ms/image at batch size 1 on {device}")
    print(f"  throughput   : {len(dataset) / batched_seconds:.1f} images/s at "
          f"batch size {args.batch_size}")
    print(f"  exact matches: {sum(r == p for r, p in zip(references, predictions))}"
          f"/{len(references)}")
    print()

    worst = sorted(
        ((cer, i) for i, cer in enumerate(sample_cers) if cer is not None),
        reverse=True,
    )[:5]
    print("Five worst samples (for error analysis)")
    print("-" * 72)
    for cer, index in worst:
        print(f"  CER {cer:.3f}  {image_ids[index]}")
        print(f"    ref : {references[index]}")
        print(f"    pred: {predictions[index]}")
    print()

    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / f"{run_name}_{args.split}.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "reference", "prediction", "cer", "exact_match"])
        for image_id, reference, prediction, cer in zip(
            image_ids, references, predictions, sample_cers
        ):
            writer.writerow([image_id, reference, prediction, cer, reference == prediction])

    metrics_path = metrics_dir / f"{run_name}_{args.split}.json"
    payload = {
        "run_name": run_name,
        "model": "CRNN+CTC",
        "decoding": "greedy, no external language model",
        "split": args.split,
        "dataset": dataset_name,
        "domain": "handwritten",
        "granularity": "line",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "n_parameters": model.n_parameters(),
        "latency_ms_per_image_batch1": round(ms_per_image, 3),
        "latency_samples": n_latency,
        "throughput_images_per_s": round(len(dataset) / batched_seconds, 3),
        "throughput_batch_size": args.batch_size,
        "exact_matches": sum(r == p for r, p in zip(references, predictions)),
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        **result.as_row(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  predictions : {predictions_path}")
    print(f"  metrics     : {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
