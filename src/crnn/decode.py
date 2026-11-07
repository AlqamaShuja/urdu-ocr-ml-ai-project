"""CTC decoding: turn per-frame probabilities into text.

Only **greedy** (best-path) decoding is implemented, and that is a deliberate
choice, not a limitation to fix later.

A beam search with an Urdu language model would lower the CRNN's error rate. But
the project compares a CRNN against a vision-language model, and the VLM already
carries a language model inside it. Adding an external language model to only one
of the two would mean the comparison measured language modelling rather than
recognition. So both systems are decoded without an external language model, and
the fact is stated in the results.

### How greedy CTC decoding works

The network emits a probability distribution over characters for every frame.
Greedy decoding:

1. Takes the single most likely symbol at each frame.
2. Collapses runs of the same symbol into one.
3. Removes the blanks.

Step 2 before step 3 is what makes genuine double letters possible. To write "pp"
the network emits `p, blank, p`: collapsing gives `p, blank, p` and dropping
blanks gives `pp`. Without the separating blank it would collapse to a single `p`.
"""

from __future__ import annotations

import torch

from src.data.vocabulary import BLANK_INDEX, Vocabulary

__all__ = ["collapse_path", "greedy_decode", "greedy_decode_indices"]


def collapse_path(path: list[int], blank_index: int = BLANK_INDEX) -> list[int]:
    """Apply the CTC collapse rule to one frame-wise index sequence.

    Collapses repeats first, then drops blanks - the order matters, see the module
    docstring.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for index in path:
        if index != previous:
            collapsed.append(index)
        previous = index
    return [index for index in collapsed if index != blank_index]


def greedy_decode_indices(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor | None = None,
    blank_index: int = BLANK_INDEX,
) -> list[list[int]]:
    """Decode a batch to index sequences.

    Args:
        log_probs: `(time, batch, n_classes)`, as `CRNN.forward` returns.
        input_lengths: `(batch,)` true frame count per sample. Required whenever
            the batch was padded - without it the padding frames are decoded as if
            they were image, which appends junk to every short sample.
        blank_index: the CTC blank, 0 by convention here.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"expected (time, batch, classes), got {tuple(log_probs.shape)}")

    best = log_probs.argmax(dim=2)  # (T, B)
    best = best.detach().cpu()
    n_frames, batch_size = best.shape

    if input_lengths is None:
        lengths = [n_frames] * batch_size
    else:
        if len(input_lengths) != batch_size:
            raise ValueError(
                f"input_lengths has {len(input_lengths)} entries for a batch of {batch_size}"
            )
        lengths = [min(int(length), n_frames) for length in input_lengths]

    return [
        collapse_path(best[: lengths[i], i].tolist(), blank_index)
        for i in range(batch_size)
    ]


def greedy_decode(
    log_probs: torch.Tensor,
    vocabulary: Vocabulary,
    input_lengths: torch.Tensor | None = None,
) -> list[str]:
    """Decode a batch straight to strings, in logical order.

    The returned strings are not reversed for right-to-left display. Rendering
    direction is the terminal's or the document's job; reversing here would
    corrupt the text.
    """
    return [
        vocabulary.decode(indices)
        for indices in greedy_decode_indices(log_probs, input_lengths)
    ]
