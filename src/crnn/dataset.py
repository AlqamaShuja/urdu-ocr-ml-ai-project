"""Dataset and batching for the CRNN, reading the project manifests.

Both models read the same manifests, so the CRNN and the VLM are guaranteed to
train and be evaluated on identical examples.

### Right-to-left script and CTC

**Images are mirrored left-to-right before being fed to the model.** This is not
cosmetic; without it the model cannot learn at all.

Urdu is written right to left, so the **first** character of a transcription is at
the **right-hand** end of the image. The CNN scans left to right, so its first
feature frame covers the **last** character of the string, and its last frame
covers the first.

CTC can only align a target to the input **monotonically**: target character *i*
must align to a frame at or after the frame for character *i-1*. With an unflipped
Urdu line the required alignment runs backwards, which CTC cannot express. The
model is left able to learn the marginal character distribution - mostly the
commonest letter and a word rhythm - but never to read.

Mirroring the image puts the first character in the first frame and restores a
monotonic alignment. UTRNet, a published Urdu OCR model, applies the same
left-right flip.

**The stored text is never touched.** Transcriptions stay in logical order in the
manifests, in the vocabulary, in the metrics and in every saved prediction. Only
the pixels are mirrored, inside the loader, and `scripts/predict_image.py` applies
the identical flip so a hand-supplied image is treated the same way.

### Variable widths

Line images share a height but not a width - NUST-UHWR runs from 200 to 2,170
pixels. Three things follow:

1. Each batch is padded to its own widest member, not to a global maximum, so a
   batch of narrow lines stays cheap.
2. Padding uses the **background** value, not zero. These images are dark ink on a
   light background, so zero-padding would paste black blocks onto the line and
   teach the model that ink appears at the end of every short image.
3. The true frame count per sample is kept and handed to CTC as `input_lengths`.
   Padding must never be presented to the loss as if it were image.

### Sorting

`width_bucketed_indices` groups images of similar width together, so batches
contain little padding. It is useful for speed but it removes shuffling, so it is
off by default and should not be used for the final training run without noting it.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.crnn.model import output_width
from src.data.vocabulary import Vocabulary

__all__ = [
    "NUST_UHWR_PIXEL_MEAN",
    "NUST_UHWR_PIXEL_STD",
    "LineImageDataset",
    "Sample",
    "WidthBucketSampler",
    "collate_batch",
    "width_bucketed_indices",
]

# Images are dark ink on light paper, so the background is the high end of the
# raw 0-1 range. Padding uses it so added columns stay blank rather than inky.
RAW_BACKGROUND_VALUE = 1.0

# Pixel statistics measured over all 8,483 NUST-UHWR TRAINING images
# (515,685,248 pixels). Validation and test were not touched, so no information
# leaks from them into the input scaling.
#
# WHY THIS MATTERS. Raw images have mean 0.953 and standard deviation 0.186:
# 93% of pixels are near-white paper and the ink that carries all the signal is
# about 7%. A CNN fed an almost-constant 0.95 is badly conditioned, and this
# model's first convolution has no BatchNorm after it (following the original
# CRNN design), so nothing downstream rescues it. Standardising to mean 0 and
# unit variance is the standard fix.
NUST_UHWR_PIXEL_MEAN = 0.953208
NUST_UHWR_PIXEL_STD = 0.186242


@dataclass(frozen=True)
class Sample:
    """One manifest row, resolved to tensors."""

    image: torch.Tensor  # (1, H, W), standardised to roughly mean 0 / unit variance
    targets: torch.Tensor  # (L,) int64 vocabulary indices
    input_length: int  # frames the CNN will produce for the true width
    target_length: int
    image_id: str
    text: str  # the normalized reference this sample is scored against
    pad_value: float  # background in the SAME space as `image`, for padding


class LineImageDataset(Dataset[Sample]):
    """Text line or word images listed in a project manifest CSV.

    Args:
        manifest: path to `data/manifests/<dataset>_<split>.csv`.
        vocabulary: built from the training split only.
        project_root: manifests store paths relative to it.
        text_column: manifest column used as the reference. Defaults to
            `text_normalized`, which is what the metrics also compare against.
        expected_height: every image must already be this tall. NUST-UHWR ships at
            64 and Synthetic-R-Urdu at 32, so no resizing happens here; a
            mismatch is an error rather than a silent rescale.
        limit: keep only the first N rows. For smoke tests and overfit runs.
    """

    def __init__(
        self,
        manifest: Path,
        vocabulary: Vocabulary,
        *,
        project_root: Path,
        text_column: str = "text_normalized",
        expected_height: int = 64,
        limit: int | None = None,
        pixel_mean: float = NUST_UHWR_PIXEL_MEAN,
        pixel_std: float = NUST_UHWR_PIXEL_STD,
        flip_horizontal: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        self.vocabulary = vocabulary
        self.project_root = Path(project_root)
        self.text_column = text_column
        self.expected_height = expected_height
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.flip_horizontal = flip_horizontal
        # Background in standardised space. Padding must be expressed here, not as
        # raw 1.0, or padded columns would read as a huge positive signal.
        self.pad_value = (RAW_BACKGROUND_VALUE - pixel_mean) / pixel_std

        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"no rows read from {self.manifest}")
        self.rows = rows

        # Checked once up front: a target longer than the frames available makes
        # the CTC loss infinite, and that is far easier to diagnose here than
        # mid-training.
        for row in self.rows:
            width = int(row["width"])
            text = row[self.text_column]
            frames = output_width(width)
            needed = len(text) + sum(1 for a, b in zip(text, text[1:]) if a == b)
            if frames < needed:
                raise ValueError(
                    f"{row['image_id']}: width {width} yields {frames} frames but the "
                    f"label needs at least {needed}. CTC loss would be infinite."
                )

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def widths(self) -> list[int]:
        return [int(row["width"]) for row in self.rows]

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]
        path = self.project_root / row["image_path"]

        with Image.open(path) as handle:
            image = handle.convert("L")  # grayscale, 8-bit
            array = np.asarray(image, dtype=np.float32) / 255.0

        if self.flip_horizontal:
            # Mirror the image left-to-right. See the module docstring section
            # "Right-to-left script and CTC" - without this the target sequence
            # runs backwards relative to the feature sequence, and CTC's
            # monotonic alignment cannot express that at all.
            array = array[:, ::-1]

        # Standardise to roughly mean 0, unit variance. See the module constants
        # for why: raw inputs sit at mean 0.953 with std 0.186.
        array = (np.ascontiguousarray(array) - self.pixel_mean) / self.pixel_std

        if array.shape[0] != self.expected_height:
            raise ValueError(
                f"{row['image_id']}: expected height {self.expected_height}, "
                f"got {array.shape[0]}"
            )

        text = row[self.text_column]
        targets = self.vocabulary.encode(text, skip_unknown=False)

        return Sample(
            image=torch.from_numpy(array).unsqueeze(0),  # (1, H, W)
            targets=torch.tensor(targets, dtype=torch.int64),
            input_length=output_width(array.shape[1]),
            target_length=len(targets),
            image_id=row["image_id"],
            text=text,
            pad_value=self.pad_value,
        )


@dataclass(frozen=True)
class Batch:
    """A padded batch, with the lengths CTC needs to ignore the padding."""

    images: torch.Tensor  # (B, 1, H, W_max)
    targets: torch.Tensor  # (sum of target lengths,) concatenated
    input_lengths: torch.Tensor  # (B,) true frames per sample
    target_lengths: torch.Tensor  # (B,)
    image_ids: list[str]
    texts: list[str]


def collate_batch(samples: list[Sample]) -> Batch:
    """Pad a list of samples to a batch.

    Targets are concatenated into one 1-D tensor rather than padded into a matrix.
    `ctc_loss` accepts that form and it needs no padding value, which removes any
    chance of a pad index being mistaken for a real character.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")

    height = samples[0].image.shape[1]
    max_width = max(sample.image.shape[2] for sample in samples)

    images = torch.full(
        (len(samples), 1, height, max_width),
        fill_value=samples[0].pad_value,
        dtype=torch.float32,
    )
    for i, sample in enumerate(samples):
        width = sample.image.shape[2]
        images[i, :, :, :width] = sample.image

    return Batch(
        images=images,
        targets=torch.cat([sample.targets for sample in samples]),
        input_lengths=torch.tensor([s.input_length for s in samples], dtype=torch.int64),
        target_lengths=torch.tensor([s.target_length for s in samples], dtype=torch.int64),
        image_ids=[sample.image_id for sample in samples],
        texts=[sample.text for sample in samples],
    )


def width_bucketed_indices(widths: list[int], batch_size: int) -> list[list[int]]:
    """Group indices so each batch holds similar widths, minimising padding."""
    order = sorted(range(len(widths)), key=lambda i: widths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


class WidthBucketSampler(Sampler[list[int]]):
    """Batch sampler that puts images of similar width in the same batch.

    **This is a correctness fix, not only a speed optimisation.**

    Widths here run from 200 to 2,170 pixels. With random batching, a batch holding
    both extremes pads the narrow image with roughly 1,970 columns of blank
    background. CTC ignores that padding because it is given the true
    `input_lengths` - but **BatchNorm does not**. BatchNorm computes its statistics
    over the whole padded tensor, so those statistics are dominated by padding, and
    the proportion of padding changes from batch to batch. The normalisation the
    network sees is then unstable, and training stalls.

    That failure is exactly what happened on the first full GPU run: training loss
    flattened at about 2.85 after three epochs and barely moved for the next
    twenty-seven, while CER sat around 0.73. The earlier tiny overfit test passed
    only because it used the eight narrowest images, where padding is negligible.

    Batches are formed from width-sorted indices, then the **order of the batches**
    is shuffled every epoch. So each batch is width-homogeneous, while the sequence
    of batches is still random. Within-batch composition is fixed across epochs,
    which is a mild reduction in randomness and is recorded in the run log.
    """

    def __init__(self, widths: list[int], batch_size: int, *, shuffle: bool = True) -> None:
        self.batches = width_bucketed_indices(widths, batch_size)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for index in order:
            yield self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)

    def padding_fraction(self, widths: list[int]) -> float:
        """Share of each batch tensor that is padding, averaged over batches.

        Useful for reporting the effect of bucketing: compare it against the
        unbucketed value before and after.
        """
        total_padding = 0
        total_cells = 0
        for batch in self.batches:
            widest = max(widths[i] for i in batch)
            total_cells += widest * len(batch)
            total_padding += sum(widest - widths[i] for i in batch)
        return total_padding / total_cells if total_cells else 0.0
