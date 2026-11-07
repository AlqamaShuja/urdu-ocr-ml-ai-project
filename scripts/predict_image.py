"""Read one or more Urdu text-line images with a trained CRNN checkpoint.

    python scripts/predict_image.py --checkpoint outputs/checkpoints/crnn_handwritten_best.pt ^
        --image path/to/line.jpg

    python scripts/predict_image.py --checkpoint ... --image a.jpg b.jpg c.jpg

**This is a demo, not part of the evaluation.** Numbers reported in the thesis come
from `src/crnn/evaluate.py` over an official split. This script exists so a single
image can be tried by hand.

### What it expects

**One line of text per image.** The model was trained on single text lines and has
never seen a full page. Give it a page and the output is meaningless, not merely
worse: the page would be squashed to 64 pixels tall, stacking every line on top of
every other.

Unlike the training loader, this script **does** resize, because a hand-supplied
image will rarely already be 64 pixels tall. It scales to the training height while
keeping the aspect ratio, and warns when the image looks like a page rather than a
line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN, output_width  # noqa: E402
from src.data.vocabulary import Vocabulary  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# A single text line is far wider than it is tall. Anything squarer is probably a
# paragraph or a page, which this model cannot read.
MIN_LINE_ASPECT_RATIO = 3.0


def load_image(path: Path, target_height: int) -> tuple[torch.Tensor, list[str]]:
    """Load one image, scale it to the training height, and flag anything suspect."""
    warnings: list[str] = []
    with Image.open(path) as handle:
        image = handle.convert("L")
        width, height = image.size

        aspect = width / height
        if aspect < MIN_LINE_ASPECT_RATIO:
            warnings.append(
                f"aspect ratio is {aspect:.1f} (width/height). A single text line is "
                f"normally wider than {MIN_LINE_ASPECT_RATIO:.0f}:1. This looks like a "
                "paragraph or a page, which this model cannot read - split it into "
                "lines first."
            )

        if height != target_height:
            new_width = max(1, round(width * target_height / height))
            warnings.append(
                f"resized {width}x{height} to {new_width}x{target_height} "
                "(aspect ratio preserved)"
            )
            image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)

        array = np.asarray(image, dtype=np.float32) / 255.0

    # The model needs at least one frame per character it must emit. Very narrow
    # images simply cannot produce a long transcription.
    frames = output_width(array.shape[1])
    if frames < 5:
        warnings.append(
            f"only {frames} frames after downsampling; the image is too narrow to "
            "produce meaningful text"
        )

    return torch.from_numpy(array).unsqueeze(0), warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but no CUDA device is available.")
    if not args.checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 "Train one first with: python -m src.crnn.train --config configs/crnn_handwritten.yaml")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_cfg, model_cfg = config["data"], config["model"]
    target_height = int(data_cfg["image_height"])
    vocabulary = Vocabulary(chars=tuple(checkpoint["vocabulary_chars"]))

    model = CRNN(
        n_classes=vocabulary.size,
        input_height=target_height,
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model : {config['run_name']}, epoch {checkpoint['epoch']}, "
          f"validation CER {checkpoint['val_cer']:.4f}")
    print(f"Input : expects ONE text line per image, {target_height} px tall")
    print()

    for path in args.image:
        if not path.is_file():
            print(f"{path}\n  SKIPPED: file not found\n")
            continue

        image, warnings = load_image(path, target_height)
        for warning in warnings:
            print(f"  warning: {warning}")

        with torch.no_grad():
            log_probs = model(image.unsqueeze(0).to(device))
            text = greedy_decode(log_probs, vocabulary)[0]

        print(f"{path.name}")
        print(f"  {text if text else '(empty - the model emitted only blanks)'}")
        print()

    print("Reminder: this is a demo. Reported metrics come from")
    print("src/crnn/evaluate.py over an official dataset split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
