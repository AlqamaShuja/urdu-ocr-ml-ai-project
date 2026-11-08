"""A/B test: does mirroring the image left-to-right let the model learn?

    python scripts/ab_test_flip.py --samples 64 --steps 400

Trains the same model twice from the same seed on the same samples. The only
difference is whether the image is mirrored left-to-right before the model sees it.

### Why this should matter

Urdu is written right to left, so the first character of a transcription sits at
the right-hand end of the image. The CNN scans left to right, so without a flip the
target sequence runs **backwards** relative to the feature sequence.

CTC aligns a target to the input monotonically: character *i* must align at or
after character *i-1*. A backwards target cannot be expressed that way at all, so
the model can only learn the marginal character distribution, never to read.

If that diagnosis is right, the flipped run should pull clearly ahead. If the two
runs track each other, the diagnosis is wrong and the real cause is elsewhere.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import ctc_loss
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn.dataset import LineImageDataset, WidthBucketSampler, collate_batch  # noqa: E402
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN  # noqa: E402
from src.data.vocabulary import BLANK_INDEX, Vocabulary  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(
    *,
    flip: bool,
    samples: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    manifests: Path,
    dataset_name: str,
    height: int,
    report_every: int,
) -> tuple[list[tuple[int, float, float]], list[str], list[str]]:
    set_seed(seed)

    vocabulary = Vocabulary.from_json(manifests / f"{dataset_name}_vocab.json")
    dataset = LineImageDataset(
        manifests / f"{dataset_name}_train.csv",
        vocabulary,
        project_root=PROJECT_ROOT,
        expected_height=height,
        limit=samples,
        flip_horizontal=flip,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=WidthBucketSampler(dataset.widths, batch_size, shuffle=True),
        collate_fn=collate_batch,
    )

    model = CRNN(n_classes=vocabulary.size, input_height=height).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: list[tuple[int, float, float]] = []
    step = 0
    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(batch.images.to(device))
            loss = ctc_loss(
                log_probs.float(),
                batch.targets.to(device),
                batch.input_lengths.to(device),
                batch.target_lengths.to(device),
                blank=BLANK_INDEX,
                zero_infinity=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1

            if step % report_every == 0 or step == steps:
                model.eval()
                refs: list[str] = []
                hyps: list[str] = []
                with torch.no_grad():
                    for b in loader:
                        lp = model(b.images.to(device))
                        hyps.extend(greedy_decode(lp, vocabulary, b.input_lengths))
                        refs.extend(b.texts)
                model.train()
                history.append((step, float(loss.detach()), evaluate(refs, hyps).cer))

    # Final predictions, for a qualitative look.
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for b in loader:
            lp = model(b.images.to(device))
            hyps.extend(greedy_decode(lp, vocabulary, b.input_lengths))
            refs.extend(b.texts)
    return history, refs, hyps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nust_uhwr")
    parser.add_argument("--manifests", type=Path, default=PROJECT_ROOT / "data" / "manifests")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--report-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    common = dict(
        samples=args.samples,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        manifests=args.manifests,
        dataset_name=args.dataset,
        height=args.height,
        report_every=args.report_every,
    )

    print("=" * 72)
    print("A/B test: image mirrored left-to-right, or not")
    print("=" * 72)
    print(f"  samples {args.samples}, steps {args.steps}, batch {args.batch_size}, "
          f"lr {args.lr}, seed {args.seed}, device {device}")
    print("  Identical seed and data in both arms. Only the flip differs.")
    print()

    results = {}
    for flip in (False, True):
        label = "FLIPPED (mirrored)" if flip else "NOT flipped (current behaviour)"
        print(f"--- {label} ---")
        started = time.perf_counter()
        history, refs, hyps = run(flip=flip, **common)
        print(f"  {'step':>6} {'loss':>9} {'train CER':>10}")
        for step, loss, cer in history:
            print(f"  {step:>6} {loss:>9.4f} {cer:>10.4f}")
        print(f"  ({time.perf_counter() - started:.0f}s)")
        print()
        results[flip] = (history, refs, hyps)

    print("=" * 72)
    print("Comparison")
    print("=" * 72)
    print(f"  {'step':>6} {'loss no-flip':>14} {'loss FLIPPED':>14} "
          f"{'CER no-flip':>13} {'CER FLIPPED':>13}")
    for (step, l0, c0), (_, l1, c1) in zip(results[False][0], results[True][0]):
        print(f"  {step:>6} {l0:>14.4f} {l1:>14.4f} {c0:>13.4f} {c1:>13.4f}")
    print()

    final_no = results[False][0][-1]
    final_yes = results[True][0][-1]
    print(f"  final loss : {final_no[1]:.4f} (no flip) vs {final_yes[1]:.4f} (flipped)")
    print(f"  final CER  : {final_no[2]:.4f} (no flip) vs {final_yes[2]:.4f} (flipped)")
    print()

    if final_yes[2] < final_no[2] - 0.05:
        print("  VERDICT: flipping helps clearly. The right-to-left/CTC monotonicity")
        print("  diagnosis is confirmed - keep flip_horizontal=True.")
    elif final_no[2] < final_yes[2] - 0.05:
        print("  VERDICT: flipping HURTS. The diagnosis is wrong; revert it.")
    else:
        print("  VERDICT: no clear difference at this budget. Inconclusive - rerun with")
        print("  more steps before drawing a conclusion either way.")

    print()
    print("  Sample predictions, flipped arm:")
    refs, hyps = results[True][1], results[True][2]
    for reference, hypothesis in list(zip(refs, hyps))[:3]:
        print(f"    ref : {reference}")
        print(f"    pred: {hypothesis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
