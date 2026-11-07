"""Shared Urdu text normalization for every model in this project.

This module is deliberately the *only* place normalization happens. Both the CRNN
and the VLM are scored by passing their predictions and the ground truth through
the same function here. If one model were normalized differently from the other,
the comparison would measure the normalizer rather than the models.

The canonical entry point is `normalize_for_eval`. Call that unless you have a
documented reason not to.

Pipeline order, which matters:

  1. **NFKC**. Expands Arabic Presentation Forms, where one code point encodes a
     whole ligature, into their constituent letters.
  2. **Remove invisible formatting characters** - zero-width joiners, bidi marks
     and the byte order mark. These change nothing a reader sees, so scoring them
     would measure invisible bytes.
  3. **Fold Arabic letter variants to their Urdu forms.** Unicode does not do this,
     because these are different letters, not compatibility variants. See the
     justification below.
  4. **Optionally strip diacritics** (off by default).
  5. **Collapse whitespace** and strip the ends.

Step 1 before step 3 is not optional. NFKC turns a presentation form into letters
that are themselves Arabic variants, which step 3 then folds. For example
U+FDF2 (one code point) becomes the letters of "allah" containing U+0647 ARABIC
HEH, which step 3 maps to U+06C1 HEH GOAL, yielding the correct Urdu spelling.

Decisions recorded on 2026-11-02, with the evidence that justified each, are in
`reports/dataset_card.md`.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ARABIC_TO_URDU",
    "INVISIBLE_CHARS",
    "normalize_for_eval",
    "normalize_text",
    "strip_diacritics",
]

# --- Step 2: invisible formatting characters ---------------------------------
# None of these appear in the NUST-UHWR ground truth, but a generative model can
# emit them, so they are removed from both sides for symmetry.
INVISIBLE_CHARS = {
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    "‎": "LEFT-TO-RIGHT MARK",
    "‏": "RIGHT-TO-LEFT MARK",
    "‪": "LEFT-TO-RIGHT EMBEDDING",
    "‫": "RIGHT-TO-LEFT EMBEDDING",
    "‬": "POP DIRECTIONAL FORMATTING",
    "‭": "LEFT-TO-RIGHT OVERRIDE",
    "‮": "RIGHT-TO-LEFT OVERRIDE",
    "﻿": "ZERO WIDTH NO-BREAK SPACE / BOM",
}

# --- Step 3: Arabic letter variants folded to Urdu forms ----------------------
#
# Urdu and Arabic share the Arabic block but prefer different code points for the
# same letter. Unicode normalization does not fold them, so this mapping is a
# research decision, not a standards-mandated step.
#
# Justification, measured over the 10,605 NUST-UHWR transcriptions (554,720
# characters). In every case the Arabic form is a rounding error beside the Urdu
# form, which is the signature of annotator keyboard inconsistency rather than a
# distinction the writers actually made on paper:
#
#   U+064A ARABIC YEH        26 vs  43,659 U+06CC FARSI YEH      (0.06%)
#   U+0647 ARABIC HEH        14 vs  23,699 U+06C1 HEH GOAL       (0.06%)
#   U+0629 TEH MARBUTA       22 vs  23,699 U+06C1 HEH GOAL       (0.09%)
#   U+0623 ALEF WITH HAMZA    2 vs  53,874 U+0627 ALEF           (0.004%)
#   U+0643 ARABIC KAF          0 vs 29,743 U+06A9 KEHEH          (absent)
#
# U+0643 never occurs in the ground truth but is still mapped: this normalizer
# also runs over model predictions, and a vision-language model trained largely on
# Arabic text may well emit the Arabic form.
ARABIC_TO_URDU = {
    "ى": "ی",  # ALEF MAKSURA        -> FARSI YEH
    "ي": "ی",  # ARABIC YEH          -> FARSI YEH
    "ك": "ک",  # ARABIC KAF          -> KEHEH
    "ه": "ہ",  # ARABIC HEH          -> HEH GOAL
    "ة": "ہ",  # TEH MARBUTA         -> HEH GOAL
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
}

# Characters deliberately NOT folded, with the reason. Listed so that a future
# reader can see these were considered rather than overlooked.
#
#   U+0622 آ  ALEF WITH MADDA   - a distinct Urdu letter, 1,870 occurrences.
#   U+0624 ؤ  WAW WITH HAMZA    - standard Urdu, 171 occurrences (jao, banao).
#   U+0626 ئ  YEH WITH HAMZA    - standard Urdu, 5,028 occurrences (koi, gai).
#   U+06BE ھ  HEH DOACHASHMEE   - a distinct Urdu letter, 6,638 occurrences.
#   U+06D2 ے  YEH BARREE        - a distinct Urdu letter, 22,196 occurrences.
#
# Digits are also NOT folded. Both ASCII 0-9 (2,388 occurrences) and extended
# Arabic-Indic U+06F0-U+06F9 (124) appear, so the annotation is inconsistent - but
# unlike the letter pairs above, these glyphs look clearly different on the page.
# A model that reads "4" where the image shows "۴" has genuinely misread it, and
# folding would hide that error.

_WHITESPACE_RUN = re.compile(r"\s+")
_INVISIBLE_RUN = re.compile(f"[{''.join(INVISIBLE_CHARS)}]")
_ARABIC_TABLE = str.maketrans(ARABIC_TO_URDU)


def strip_diacritics(text: str) -> str:
    """Remove combining marks: harakat, honorifics and other Urdu diacritics.

    Not applied by default. Diacritics are only 878 of 554,720 characters (0.16%)
    in NUST-UHWR and are optional in written Urdu, so they are kept in the primary
    metric and this function backs a secondary, diacritic-insensitive score.
    """
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def normalize_text(
    text: str,
    *,
    form: str = "NFKC",
    remove_invisible: bool = True,
    fold_arabic_variants: bool = True,
    remove_diacritics: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize one Urdu string. See the module docstring for the pipeline.

    The string stays in logical order throughout. It is never reversed, even
    though Urdu renders right to left: reversing would corrupt the data, and
    display direction is the renderer's job.
    """
    if form:
        text = unicodedata.normalize(form, text)
    if remove_invisible:
        text = _INVISIBLE_RUN.sub("", text)
    if fold_arabic_variants:
        text = text.translate(_ARABIC_TABLE)
    if remove_diacritics:
        text = strip_diacritics(text)
    if collapse_whitespace:
        text = _WHITESPACE_RUN.sub(" ", text).strip()
    return text


def normalize_for_eval(text: str) -> str:
    """The project's canonical normalization. Use this everywhere.

    Settings, as decided on 2026-11-02:
      - NFKC, so ligature presentation forms become letters. This is a fairness
        requirement: a generative VLM emits spelled-out letters, not the single
        ligature code point, and without this it would be marked wrong for output
        that is correct.
      - Invisible formatting characters removed.
      - Arabic letter variants folded to Urdu forms.
      - Diacritics KEPT, so the primary metric stays strict.
      - Whitespace collapsed and trimmed.
    """
    return normalize_text(text)
