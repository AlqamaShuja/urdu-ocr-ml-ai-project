"""CER and WER for this project. Shared by the CRNN and the VLM without exception.

Both models are scored by `evaluate()` here, over the same examples, with the same
normalizer from `normalize_text`. That is what makes the comparison a comparison.

Two numbers are reported for every run, as the project brief requires:

  - **normalized** - the primary metric, after `normalize_for_eval`.
  - **raw** - the stricter secondary metric, on the untouched strings.

Reporting both means a reader can see exactly how much of the result the
normalization choice is responsible for.

Definitions used here, so they are not ambiguous in the write-up:

  - **CER** = (substitutions + deletions + insertions) / characters in the
    reference, aggregated over the whole split, not averaged per sample. A
    per-sample average would weight a 19-character line the same as a
    116-character line.
  - **WER** = the same, over whitespace-separated tokens.
  - Spaces count as characters in CER. Urdu word boundaries carry information, so
    a model that runs words together has made a real error.
  - `1 - CER` is **not** reported as "accuracy". CER can exceed 1.0 when a model
    inserts more than it gets right, which would make such an "accuracy" negative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import jiwer

from .normalize_text import normalize_for_eval

__all__ = ["EditCounts", "EvalResult", "evaluate", "per_sample_cer"]

# Treat each string as a flat character sequence, with spaces preserved as
# characters. Strip() only trims the ends, which normalization already did.
_CHAR_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
# `RemoveEmptyStrings` is deliberately absent. It drops empty *sentences* from the
# list, which silently breaks the alignment between references and hypotheses: an
# untrained model predicts "" for some samples, and dropping those leaves 8
# references against 7 hypotheses. Without it, jiwer maps an empty hypothesis to
# zero words and counts the whole reference as deletions, which is correct.
# Strip and RemoveMultipleSpaces already guarantee no empty tokens inside a
# sentence, so the transform was never needed.
_WORD_TRANSFORM = jiwer.Compose(
    [
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


@dataclass(frozen=True)
class EditCounts:
    """Substitutions, deletions, insertions and hits behind an error rate."""

    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def reference_length(self) -> int:
        """Units in the reference. Insertions are in the hypothesis, not here."""
        return self.hits + self.substitutions + self.deletions

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class EvalResult:
    """Everything one evaluation run reports."""

    n_samples: int
    cer: float
    wer: float
    cer_raw: float
    wer_raw: float
    char_counts: EditCounts
    word_counts: EditCounts
    n_empty_references: int = 0
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, object]:
        """Flatten to one dict, for writing a results CSV."""
        row: dict[str, object] = {
            "n_samples": self.n_samples,
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "cer_raw": round(self.cer_raw, 6),
            "wer_raw": round(self.wer_raw, 6),
        }
        for prefix, counts in (("char", self.char_counts), ("word", self.word_counts)):
            for key, value in asdict(counts).items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_reference_length"] = counts.reference_length
        row["n_empty_references"] = self.n_empty_references
        return row

    def summary(self) -> str:
        c, w = self.char_counts, self.word_counts
        return (
            f"n={self.n_samples}  "
            f"CER={self.cer:.4f} (raw {self.cer_raw:.4f})  "
            f"WER={self.wer:.4f} (raw {self.wer_raw:.4f})\n"
            f"  chars: S={c.substitutions} D={c.deletions} I={c.insertions} "
            f"H={c.hits} ref_len={c.reference_length}\n"
            f"  words: S={w.substitutions} D={w.deletions} I={w.insertions} "
            f"H={w.hits} ref_len={w.reference_length}"
        )


def _counts_from(output: jiwer.CharacterOutput | jiwer.WordOutput) -> EditCounts:
    return EditCounts(
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
    )


def evaluate(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalizer: Callable[[str], str] = normalize_for_eval,
) -> EvalResult:
    """Score one set of predictions against its ground truth.

    `references` and `hypotheses` must be aligned and the same length - element i
    of one is the ground truth for element i of the other.

    Samples whose reference normalizes to an empty string are dropped from the
    metric, because an error rate over zero reference characters is undefined.
    The number dropped is reported rather than hidden; in a clean dataset it is 0.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must align: got {len(references)} and {len(hypotheses)}"
        )
    if not references:
        raise ValueError("nothing to evaluate: references is empty")

    notes: list[str] = []

    norm_refs: list[str] = []
    norm_hyps: list[str] = []
    raw_refs: list[str] = []
    raw_hyps: list[str] = []
    n_empty = 0

    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalizer(reference)
        if not normalized_reference:
            n_empty += 1
            continue
        norm_refs.append(normalized_reference)
        norm_hyps.append(normalizer(hypothesis))
        # The raw score keeps the strings untouched apart from end-trimming, which
        # is needed only so an empty hypothesis is not treated as a single space.
        raw_refs.append(reference.strip())
        raw_hyps.append(hypothesis.strip())

    if not norm_refs:
        raise ValueError("every reference normalized to an empty string")
    if n_empty:
        notes.append(
            f"{n_empty} sample(s) dropped: reference is empty after normalization"
        )

    char_output = jiwer.process_characters(
        norm_refs,
        norm_hyps,
        reference_transform=_CHAR_TRANSFORM,
        hypothesis_transform=_CHAR_TRANSFORM,
    )
    word_output = jiwer.process_words(
        norm_refs,
        norm_hyps,
        reference_transform=_WORD_TRANSFORM,
        hypothesis_transform=_WORD_TRANSFORM,
    )
    char_output_raw = jiwer.process_characters(
        raw_refs,
        raw_hyps,
        reference_transform=_CHAR_TRANSFORM,
        hypothesis_transform=_CHAR_TRANSFORM,
    )
    word_output_raw = jiwer.process_words(
        raw_refs,
        raw_hyps,
        reference_transform=_WORD_TRANSFORM,
        hypothesis_transform=_WORD_TRANSFORM,
    )

    return EvalResult(
        n_samples=len(norm_refs),
        cer=char_output.cer,
        wer=word_output.wer,
        cer_raw=char_output_raw.cer,
        wer_raw=word_output_raw.wer,
        char_counts=_counts_from(char_output),
        word_counts=_counts_from(word_output),
        n_empty_references=n_empty,
        notes=notes,
    )


def per_sample_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    normalizer: Callable[[str], str] = normalize_for_eval,
) -> list[float | None]:
    """CER for each sample individually, for error analysis and worst-case triage.

    Returns `None` where the reference normalizes to empty. Do not average this
    list to get the corpus CER - that would weight short and long lines equally.
    Use `evaluate()` for the reported number.
    """
    scores: list[float | None] = []
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalizer(reference)
        if not normalized_reference:
            scores.append(None)
            continue
        output = jiwer.process_characters(
            [normalized_reference],
            [normalizer(hypothesis)],
            reference_transform=_CHAR_TRANSFORM,
            hypothesis_transform=_CHAR_TRANSFORM,
        )
        scores.append(output.cer)
    return scores
