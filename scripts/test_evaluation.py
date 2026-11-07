"""Tests for the shared normalization and metric code.

    python scripts/test_evaluation.py

Plain assertions, no pytest needed. These guard the code that produces every
number in the thesis, so they run before any training and after any change to
`src/evaluation/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from src.evaluation.metrics import evaluate, per_sample_cer  # noqa: E402
from src.evaluation.normalize_text import (  # noqa: E402
    normalize_for_eval,
    normalize_text,
    strip_diacritics,
)

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ OK ] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"\n         {detail}" if detail else ""))


def eq(name: str, actual: object, expected: object) -> None:
    check(name, actual == expected, f"expected {expected!r}, got {actual!r}")


def close(name: str, actual: float, expected: float, tol: float = 1e-9) -> None:
    check(name, abs(actual - expected) < tol, f"expected {expected}, got {actual}")


print("=" * 64)
print("Shared evaluation code tests")
print("=" * 64)

# --- Presentation forms -------------------------------------------------------
# The fairness case for NFKC: a generative model emits letters, not ligatures.
print("\n1. Arabic Presentation Forms expand (the VLM fairness requirement)")
eq("U+FDF2 ligature becomes the letters of Allah",
   normalize_for_eval("ﷲ"), "اللہ")
eq("U+FDFA ligature expands to the full phrase",
   normalize_for_eval("ﷺ"), "صلی اللہ علیہ وسلم")
eq("U+FE85 presentation waw-hamza becomes standard U+0624",
   normalize_for_eval("ﺅ"), "ؤ")
check(
    "a model spelling out the ligature scores 0 CER against the ligature",
    evaluate(["ﷲ"], ["اللہ"]).cer == 0.0,
    "this is the entire reason NFKC is applied",
)

# --- Arabic variant folding ---------------------------------------------------
print("\n2. Arabic letter variants fold to Urdu forms")
eq("ARABIC YEH -> FARSI YEH", normalize_for_eval("ي"), "ی")
eq("ALEF MAKSURA -> FARSI YEH", normalize_for_eval("ى"), "ی")
eq("ARABIC KAF -> KEHEH", normalize_for_eval("ك"), "ک")
eq("ARABIC HEH -> HEH GOAL", normalize_for_eval("ه"), "ہ")
eq("TEH MARBUTA -> HEH GOAL", normalize_for_eval("ة"), "ہ")
eq("ALEF WITH HAMZA ABOVE -> ALEF", normalize_for_eval("أ"), "ا")
eq("ALEF WITH HAMZA BELOW -> ALEF", normalize_for_eval("إ"), "ا")

print("\n3. Genuine Urdu letters are NOT folded")
for code, label in (
    (0x0622, "ALEF WITH MADDA"),
    (0x0624, "WAW WITH HAMZA (jao, banao)"),
    (0x0626, "YEH WITH HAMZA (koi, gai)"),
    (0x06BE, "HEH DOACHASHMEE"),
    (0x06D2, "YEH BARREE"),
    (0x06CC, "FARSI YEH"),
    (0x06C1, "HEH GOAL"),
    (0x06A9, "KEHEH"),
):
    eq(f"U+{code:04X} {label} survives", normalize_for_eval(chr(code)), chr(code))

print("\n4. Digits are NOT folded (the glyphs genuinely differ)")
eq("extended Arabic-Indic digits survive", normalize_for_eval("۴۳"), "۴۳")
eq("ASCII digits survive", normalize_for_eval("43"), "43")
check(
    "reading 4 where the image shows the Urdu 4 is still an error",
    evaluate(["۴"], ["4"]).cer == 1.0,
    "folding digits would have hidden a real misread",
)

# --- Invisible characters and whitespace --------------------------------------
print("\n5. Invisible characters and whitespace")
eq("zero width joiner removed", normalize_for_eval("a‍b"), "ab")
eq("BOM removed", normalize_for_eval("﻿ab"), "ab")
eq("RTL mark removed", normalize_for_eval("a‏b"), "ab")
eq("whitespace runs collapse", normalize_for_eval("  a   b  "), "a b")
eq("tabs and newlines collapse to one space", normalize_for_eval("a\t\nb"), "a b")

# --- Logical order ------------------------------------------------------------
print("\n6. Logical order is preserved (never reversed for RTL)")
urdu = "اردو"  # a r d w, in logical order
eq("string order is untouched", normalize_for_eval(urdu), urdu)
eq("first code point stays first", normalize_for_eval(urdu)[0], "ا")

# --- Diacritics ---------------------------------------------------------------
print("\n7. Diacritics kept by default, strippable on request")
with_fatha = "بَ"  # beh + fatha
eq("kept in the primary normalizer", normalize_for_eval(with_fatha), with_fatha)
eq("strip_diacritics removes the combining mark", strip_diacritics(with_fatha), "ب")
eq("opt-in flag removes it too",
   normalize_text(with_fatha, remove_diacritics=True), "ب")

# --- Metric arithmetic --------------------------------------------------------
print("\n8. CER and WER arithmetic")
perfect = evaluate(["abc", "def"], ["abc", "def"])
close("identical strings give CER 0", perfect.cer, 0.0)
close("identical strings give WER 0", perfect.wer, 0.0)

one_sub = evaluate(["abcd"], ["abcX"])
close("one substitution in 4 chars gives CER 0.25", one_sub.cer, 0.25)
eq("substitution is counted as such", one_sub.char_counts.substitutions, 1)

deletion = evaluate(["abcd"], ["abc"])
close("one deletion in 4 chars gives CER 0.25", deletion.cer, 0.25)
eq("deletion is counted as such", deletion.char_counts.deletions, 1)

insertion = evaluate(["abc"], ["abcd"])
close("one insertion over 3 ref chars gives CER 1/3", insertion.cer, 1 / 3)
eq("insertion is counted as such", insertion.char_counts.insertions, 1)

check(
    "CER can exceed 1.0, so 1-CER is not an accuracy",
    evaluate(["ab"], ["abcdefgh"]).cer > 1.0,
    "this is why the project never reports 1-CER as accuracy",
)

print("\n9. Corpus aggregation, not per-sample averaging")
# One 1-char line wrong, one 99-char line perfect. Aggregating over the corpus
# gives 1/100; averaging per sample would give 0.5, which is badly misleading.
agg = evaluate(["a", "b" * 99], ["x", "b" * 99])
close("corpus CER weights by reference length", agg.cer, 1 / 100)
samples = per_sample_cer(["a", "b" * 99], ["x", "b" * 99])
eq("per-sample CER is available separately", samples, [1.0, 0.0])

print("\n10. Spaces count as characters in CER")
spaced = evaluate(["ab cd"], ["abcd"])
close("a missing space is a deletion", spaced.cer, 1 / 5)
eq("counted as a deletion", spaced.char_counts.deletions, 1)

print("\n11. Normalized versus raw reporting")
both = evaluate(["ﷲ"], ["اللہ"])
close("normalized CER is 0", both.cer, 0.0)
check("raw CER is non-zero, exposing the normalizer's effect", both.cer_raw > 0.0)

print("\n12. Guard rails")
try:
    evaluate(["a", "b"], ["a"])
    check("mismatched lengths raise", False, "no exception raised")
except ValueError:
    check("mismatched lengths raise", True)
try:
    evaluate([], [])
    check("empty input raises", False, "no exception raised")
except ValueError:
    check("empty input raises", True)

empty_ref = evaluate(["", "abcd"], ["x", "abcd"])
eq("empty reference is dropped, not silently scored", empty_ref.n_empty_references, 1)
eq("remaining sample is still scored", empty_ref.n_samples, 1)
close("the dropped sample does not affect CER", empty_ref.cer, 0.0)
check("the drop is reported in notes", bool(empty_ref.notes))

print("\n13. Empty hypotheses (an untrained model predicts nothing)")
# Regression test. jiwer's RemoveEmptyStrings transform drops empty *sentences*
# from the list, which broke the reference/hypothesis alignment the first time the
# overfit test ran against a freshly initialised model.
blank_one = evaluate(["abc"], [""])
close("an empty prediction gives CER 1.0", blank_one.cer, 1.0)
eq("scored as deletions, not substitutions", blank_one.char_counts.deletions, 3)
close("an empty prediction gives WER 1.0", blank_one.wer, 1.0)

mixed = evaluate(["abc", "de"], ["", "de"])
eq("a batch with one empty prediction keeps all samples", mixed.n_samples, 2)
close("and scores the rest normally", mixed.cer, 3 / 5)

all_blank = evaluate(["abc", "de"], ["", ""])
close("all-empty predictions give CER 1.0", all_blank.cer, 1.0)
close("all-empty predictions give WER 1.0", all_blank.wer, 1.0)

eq("per-sample CER handles empty predictions", per_sample_cer(["abc"], [""]), [1.0])

print("\n14. Results row is CSV-ready")
row = one_sub.as_row()
for key in ("cer", "wer", "cer_raw", "wer_raw", "char_substitutions", "char_reference_length"):
    check(f"as_row contains {key}", key in row)

print()
print("=" * 64)
print(f"{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    for name in FAILED:
        print(f"  FAILED: {name}")
    print("=" * 64)
    raise SystemExit(1)
print("=" * 64)
