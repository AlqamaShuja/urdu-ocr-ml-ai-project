"""CRNN: a CNN feature extractor, a BiLSTM sequence model, and a CTC output head.

The three stages, and why each exists:

1. **CNN.** Slides over the image and turns pixels into features. Crucially it
   reduces the image *height* to 1 while keeping the *width* as a sequence. A
   64-pixel-tall line image becomes a row of feature vectors, one per horizontal
   position, which is what a sequence model can consume.

2. **BiLSTM.** Reads that row of vectors in both directions. Bidirectional matters
   for Urdu: the shape a letter takes depends on its neighbours on both sides,
   because letters join. A left-to-right-only model would have to guess the
   current letter's form before seeing what follows it.

3. **CTC head.** A linear layer producing one score per character per frame, plus
   the blank. CTC then handles alignment, so no character-level bounding boxes are
   needed - only the image and its transcription.

### Reading order and right-to-left script

The feature sequence runs left to right across the image, which is the *visual*
order. Urdu renders right to left, so the leftmost frame corresponds to the *end*
of the logical string. This is handled where it belongs - in the label encoding,
by keeping transcriptions in logical order and letting the BiLSTM learn the
mapping - and never by reversing the stored Unicode string. See
`src/evaluation/normalize_text.py`.

### Width arithmetic, verified against the data

Width is downsampled by exactly 4: two 2x2 max-pools halve it, and every later
pool uses stride (2, 1), which touches height only.

    T = floor(floor(W / 2) / 2)

CTC requires T to be at least the label length plus one frame per pair of
identical adjacent characters. Measured over all 10,604 NUST-UHWR lines:

| Downsampling | T range | Violations | Smallest margin |
| --- | --- | --- | --- |
| W/2 | 100-1085 | 0 | 79 frames |
| **W/4** | **50-542** | **0** | **27 frames** |
| W/8 | 25-271 | **59** | -21 frames |

W/8 would make the loss infinite for 59 samples, so W/4 is the most aggressive
safe choice. Use `output_width()` rather than recomputing this by hand.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["CRNN", "output_width"]


def output_width(input_width: int) -> int:
    """Frames the CNN produces for an image of this width.

    Must be used to build the `input_lengths` passed to CTC. In a padded batch the
    lengths come from each sample's *true* width, never the padded width, or the
    model would be told to account for blank padding as if it were image.
    """
    return input_width // 2 // 2


class CRNN(nn.Module):
    """CRNN for fixed-height, variable-width text line or word images.

    Args:
        n_classes: CTC output units, blank included. For NUST-UHWR this is 110
            (109 characters + 1 blank). Take it from the vocabulary, never hardcode.
        input_height: image height in pixels. Must be 32 or 64; the CNN reduces it
            to exactly 1. NUST-UHWR ships at 64, Synthetic-R-Urdu at 32.
        input_channels: 1 for grayscale.
        lstm_hidden: units per direction in each BiLSTM layer.
        lstm_layers: stacked BiLSTM layers.
        dropout: applied between BiLSTM layers; ignored when `lstm_layers` is 1.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        input_height: int = 64,
        input_channels: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_height not in (32, 64):
            raise ValueError(f"input_height must be 32 or 64, got {input_height}")
        if n_classes < 2:
            raise ValueError(f"n_classes must include the blank, got {n_classes}")

        self.n_classes = n_classes
        self.input_height = input_height

        def conv_block(in_ch: int, out_ch: int, batch_norm: bool = False) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return layers

        layers: list[nn.Module] = []
        # Height 64 -> 32, width W -> W/2
        layers += conv_block(input_channels, 64)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Height 32 -> 16, width W/2 -> W/4. Width is not reduced again after this.
        layers += conv_block(64, 128)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Height 16 -> 8, width unchanged
        layers += conv_block(128, 256, batch_norm=True)
        layers += conv_block(256, 256)
        layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        # Height 8 -> 4, width unchanged
        layers += conv_block(256, 512, batch_norm=True)
        layers += conv_block(512, 512)
        layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

        # Collapse the remaining height to 1 with a valid convolution. For a
        # 64-pixel input the height is now 4; for a 32-pixel input it is 2.
        remaining_height = input_height // 16
        layers += [
            nn.Conv2d(512, 512, kernel_size=(remaining_height, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ]
        self.cnn = nn.Sequential(*layers)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Bidirectional, so the LSTM emits 2 * lstm_hidden features per frame.
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to per-frame log-probabilities.

        Args:
            images: `(batch, channels, height, width)`.

        Returns:
            `(time, batch, n_classes)` log-probabilities - the layout
            `torch.nn.functional.ctc_loss` expects, so no transpose is needed at
            the call site.
        """
        if images.dim() != 4:
            raise ValueError(f"expected a 4D batch, got shape {tuple(images.shape)}")
        if images.shape[2] != self.input_height:
            raise ValueError(
                f"expected height {self.input_height}, got {images.shape[2]}"
            )

        features = self.cnn(images)  # (B, 512, 1, T)
        if features.shape[2] != 1:
            raise RuntimeError(
                f"CNN left height {features.shape[2]} instead of 1; check input_height"
            )
        features = features.squeeze(2)  # (B, 512, T)
        features = features.permute(2, 0, 1)  # (T, B, 512)

        sequence, _ = self.rnn(features)  # (T, B, 2 * hidden)
        logits = self.classifier(sequence)  # (T, B, n_classes)
        return logits.log_softmax(dim=2)

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Parameter count, one of the metrics the project reports."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
