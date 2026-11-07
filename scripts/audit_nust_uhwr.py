"""Audit the extracted NUST-UHWR dataset before any training.

Usage (from the project root, with .venv active):

    python -I scripts/audit_nust_uhwr.py ^
        --root data/raw/nust_uhwr/UHWR ^
        --out reports/dataset_audit_nust_uhwr.md

The dataset is downloaded data, so this script lives outside the data directory,
takes the data path as an argument, and is run with `python -I` so that nothing
inside the data directory can be imported.

Checks performed, in order:
  1. Manifest parsing and record counts per split.
  2. Referenced images that are missing from disk.
  3. Images on disk that no manifest references.
  4. Duplicate image IDs within and across splits (split leakage).
  5. Duplicate image bytes across splits (leakage that renaming would hide).
  6. Identical transcriptions across splits (text-level overlap).
  7. Unreadable or corrupt images; image height and width statistics.
  8. Empty or whitespace-only transcriptions; label length statistics.
  9. Character inventory versus the dataset's own chars.txt.
 10. Effect of Unicode normalization (NFC / NFKC) on the transcriptions.

It writes a Markdown report and prints a PASS/FAIL summary. It never modifies the
dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required. Run: python -m pip install -r requirements.txt")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SPLITS = ("train", "val", "test")

# Characters that are a normalization concern for Urdu specifically. Arabic and
# Urdu share code points but prefer different ones for the same letter, and the
# Arabic Presentation Forms block encodes whole ligatures as single characters.
# Any of these in the ground truth must be a deliberate, documented decision.
ARABIC_VARIANTS = {
    "ي": "ARABIC YEH (Urdu normally uses U+06CC FARSI YEH)",
    "ك": "ARABIC KAF (Urdu normally uses U+06A9 KEHEH)",
    "ه": "ARABIC HEH (Urdu normally uses U+06C1 HEH GOAL)",
    "ة": "ARABIC TEH MARBUTA",
    "أ": "ARABIC ALEF WITH HAMZA ABOVE",
    "ؤ": "ARABIC WAW WITH HAMZA ABOVE",
}
ZERO_WIDTH = {
    "​": "ZERO WIDTH SPACE",
    "‌": "ZERO WIDTH NON-JOINER",
    "‍": "ZERO WIDTH JOINER",
    "‎": "LEFT-TO-RIGHT MARK",
    "‏": "RIGHT-TO-LEFT MARK",
    "﻿": "ZERO WIDTH NO-BREAK SPACE / BOM",
}


@dataclass
class Record:
    split: str
    rel_path: str
    image_id: str
    label: str
    line_no: int


@dataclass
class Audit:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def problem(self, text: str) -> None:
        self.problems.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


def read_manifest(path: Path, split: str, audit: Audit) -> list[Record]:
    """Parse one manifest. Format is: relative/image/path<TAB>transcription."""
    raw = path.read_text(encoding="utf-8")
    records: list[Record] = []
    for line_no, line in enumerate(raw.split("\n"), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            audit.problem(
                f"{split}.txt line {line_no}: expected exactly 1 tab, found {len(parts) - 1}"
            )
            continue
        rel_path, label = parts
        records.append(
            Record(
                split=split,
                rel_path=rel_path.strip(),
                image_id=Path(rel_path.strip()).name,
                label=label,
                line_no=line_no,
            )
        )
    if not raw.endswith("\n"):
        audit.note(
            f"{split}.txt has no trailing newline, so `wc -l` under-reports it by 1."
        )
    return records


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: list[int]) -> str:
    if not values:
        return "no data"
    ordered = sorted(values)
    n = len(ordered)
    return (
        f"min {ordered[0]}, median {ordered[n // 2]}, "
        f"mean {sum(ordered) / n:.1f}, max {ordered[-1]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="extracted UHWR directory")
    parser.add_argument("--out", required=True, type=Path, help="Markdown report to write")
    parser.add_argument(
        "--skip-image-checks",
        action="store_true",
        help="skip per-image decoding and hashing (fast structural pass only)",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    images_dir = root / "Dataset" / "images"
    if not images_dir.is_dir():
        sys.exit(f"Not found: {images_dir}")

    audit = Audit()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("# NUST-UHWR dataset audit")
    emit()
    emit(f"- Dataset root: `{root}`")
    emit(f"- Generated by: `scripts/audit_nust_uhwr.py`")
    emit()

    # ---- 1. Manifests and counts -------------------------------------------------
    emit("## 1. Manifest record counts")
    emit()
    by_split: dict[str, list[Record]] = {}
    for split in SPLITS:
        manifest = root / f"{split}.txt"
        if not manifest.is_file():
            audit.problem(f"missing manifest: {manifest}")
            by_split[split] = []
            continue
        by_split[split] = read_manifest(manifest, split, audit)

    all_records = [r for split in SPLITS for r in by_split[split]]
    stated = {"train": 8484, "val": 1061, "test": 1061}

    emit("| Split | Records parsed | Stated in Zenodo description | Match |")
    emit("| --- | --- | --- | --- |")
    for split in SPLITS:
        found = len(by_split[split])
        want = stated[split]
        emit(f"| {split} | {found} | {want} | {'yes' if found == want else 'NO'} |")
    emit(f"| **total** | **{len(all_records)}** | **10,606** | "
         f"**{'yes' if len(all_records) == 10606 else 'NO'}** |")
    emit()

    # ---- 2 and 3. Image/manifest cross-reference ---------------------------------
    emit("## 2. Images versus manifests")
    emit()
    on_disk = {p.name for p in images_dir.iterdir() if p.is_file()}
    referenced = {r.image_id for r in all_records}

    missing = sorted(referenced - on_disk)
    unreferenced = sorted(on_disk - referenced)

    emit(f"- Image files on disk: **{len(on_disk)}**")
    emit(f"- Distinct images referenced by manifests: **{len(referenced)}**")
    emit(f"- Referenced but missing from disk: **{len(missing)}**")
    emit(f"- On disk but referenced by no manifest: **{len(unreferenced)}**")
    emit()
    if missing:
        audit.problem(f"{len(missing)} referenced images are missing from disk")
        emit(f"Missing: {', '.join(missing[:20])}")
        emit()
    if unreferenced:
        emit(f"Unreferenced: {', '.join(unreferenced[:20])}")
        emit()
        emit("The Zenodo record documents `03598.jpg` as unreferenced and without a")
        emit("recoverable transcription. Unlabelled images cannot be trained or")
        emit("evaluated on, so they are excluded; this exclusion is reported in the")
        emit("dataset card and in every results table.")
        emit()
        audit.note(f"{len(unreferenced)} unlabelled image(s) excluded: {', '.join(unreferenced)}")

    # ---- 4. Duplicate IDs / split leakage ----------------------------------------
    emit("## 3. Split leakage checks")
    emit()
    id_counts = Counter(r.image_id for r in all_records)
    dup_ids = {name: count for name, count in id_counts.items() if count > 1}
    emit(f"- Image IDs appearing in more than one manifest record: **{len(dup_ids)}**")
    if dup_ids:
        audit.problem(f"{len(dup_ids)} image IDs are duplicated across manifest records")
        for name in list(dup_ids)[:10]:
            where = [r.split for r in all_records if r.image_id == name]
            emit(f"  - `{name}` in {where}")
    emit()

    # ---- 7 and 5. Per-image decoding and byte hashing -----------------------------
    heights: list[int] = []
    widths: list[int] = []
    unreadable: list[str] = []
    hash_to_ids: dict[str, list[str]] = {}

    if args.skip_image_checks:
        emit("## 4. Image integrity")
        emit()
        emit("Skipped (`--skip-image-checks`).")
        emit()
    else:
        print("  ... decoding and hashing 10k+ images, this takes a minute", file=sys.stderr)
        for record in all_records:
            path = images_dir / record.image_id
            if not path.is_file():
                continue
            try:
                with Image.open(path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    img.verify()  # detects truncated or corrupt JPEG data
            except Exception:  # noqa: BLE001 - any decode failure is a finding
                unreadable.append(record.image_id)
                continue
            hash_to_ids.setdefault(md5_of(path), []).append(record.image_id)

        emit("## 4. Image integrity and geometry")
        emit()
        emit(f"- Unreadable or corrupt images: **{len(unreadable)}**")
        if unreadable:
            audit.problem(f"{len(unreadable)} images failed to decode")
            emit(f"  {', '.join(unreadable[:20])}")
        height_counts = Counter(heights)
        emit(f"- Distinct image heights: **{len(height_counts)}** -> "
             f"{dict(height_counts.most_common(5))}")
        emit(f"- Width distribution: {describe(widths)}")
        emit(f"- Height distribution: {describe(heights)}")
        emit()
        if len(height_counts) == 1:
            emit(f"All images share height {next(iter(height_counts))} px, as the record "
                 "states. The CRNN can therefore use a fixed input height with no "
                 "resizing, and only width varies.")
        else:
            audit.note("Image heights are not uniform; the CRNN must resize to a fixed height.")
        emit()

        # Byte-identical duplicates spanning splits are leakage even if the IDs differ.
        # Worse, if two identical images carry *different* labels, at least one label
        # is wrong: the pixels cannot depict two different sentences. That is a label
        # conflict, and it makes both records unusable until resolved by inspection.
        split_of = {r.image_id: r.split for r in all_records}
        label_of = {r.image_id: r.label.strip() for r in all_records}
        cross_split_dups = []
        within_split_dups = []
        label_conflicts = []
        for ids in hash_to_ids.values():
            if len(ids) < 2:
                continue
            splits_involved = {split_of[i] for i in ids}
            if len({label_of[i] for i in ids}) > 1:
                label_conflicts.append((ids, sorted(splits_involved)))
            if len(splits_involved) > 1:
                cross_split_dups.append((ids, sorted(splits_involved)))
            else:
                within_split_dups.append(ids)

        emit("## 5. Byte-identical duplicate images")
        emit()
        emit(f"- Duplicate groups **crossing** splits (leakage): **{len(cross_split_dups)}**")
        emit(f"- Duplicate groups **within** one split: **{len(within_split_dups)}**")
        emit(f"- Duplicate groups with **conflicting labels**: **{len(label_conflicts)}**")
        emit()
        if cross_split_dups:
            audit.problem(
                f"{len(cross_split_dups)} identical-image groups span different splits (leakage)"
            )
            for ids, splits_involved in cross_split_dups[:10]:
                emit(f"  - {ids} across {splits_involved}")
            emit()
        if label_conflicts:
            audit.problem(
                f"{len(label_conflicts)} identical-image group(s) carry different labels, "
                "so at least one label in each group is wrong"
            )
            emit("**Label conflicts.** Identical pixels cannot depict two different")
            emit("sentences, so one record in each group is mislabelled. Each group below")
            emit("must be resolved by looking at the image, not by guessing:")
            emit()
            for ids, splits_involved in label_conflicts[:10]:
                emit(f"- group {ids}, splits {splits_involved}:")
                for image_id in ids:
                    emit(f"    - `{image_id}` ({split_of[image_id]}): {label_of[image_id]}")
            emit()
        if within_split_dups:
            audit.note(
                f"{len(within_split_dups)} duplicate group(s) inside a single split - "
                "not leakage, but it inflates that split's effective size."
            )

    # ---- 6. Transcription overlap across splits ----------------------------------
    emit("## 6. Transcription overlap across splits")
    emit()
    label_sets = {s: {r.label.strip() for r in by_split[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = label_sets[a] & label_sets[b]
        emit(f"- Identical transcriptions in both `{a}` and `{b}`: **{len(shared)}**")
    emit()
    emit("Identical text in two splits is not image leakage - different writers can")
    emit("produce the same sentence. It does mean a model can score partly from")
    emit("memorised text rather than from reading the image, so the count is reported")
    emit("rather than silently removed.")
    emit()

    # ---- 8. Label quality ---------------------------------------------------------
    emit("## 7. Transcription quality")
    emit()
    empty = [r for r in all_records if not r.label.strip()]
    lengths = [len(r.label) for r in all_records]
    emit(f"- Empty or whitespace-only transcriptions: **{len(empty)}**")
    if empty:
        audit.problem(f"{len(empty)} records have an empty transcription")
        for r in empty[:10]:
            emit(f"  - {r.split}.txt line {r.line_no}: `{r.image_id}`")
    emit(f"- Label length in characters: {describe(lengths)}")
    leading_ws = sum(1 for r in all_records if r.label != r.label.strip())
    emit(f"- Labels with leading or trailing whitespace: **{leading_ws}**")
    if leading_ws:
        audit.note(
            f"{leading_ws} labels carry surrounding whitespace; the shared normalizer "
            "must strip it identically for both models."
        )
    emit()

    # ---- 9. Character inventory ---------------------------------------------------
    emit("## 8. Character inventory")
    emit()
    observed = Counter()
    for record in all_records:
        observed.update(record.label)
    emit(f"- Distinct characters across all transcriptions: **{len(observed)}**")

    chars_file = root / "chars.txt"
    if chars_file.is_file():
        declared = {line for line in chars_file.read_text(encoding="utf-8").split("\n") if line != ""}
        emit(f"- Characters declared in `chars.txt`: **{len(declared)}**")
        extra = sorted(set(observed) - declared)
        unused = sorted(declared - set(observed))
        emit(f"- In transcriptions but not in `chars.txt`: **{len(extra)}**")
        if extra:
            audit.note(
                "chars.txt does not cover every character in the transcriptions; build "
                "the CRNN vocabulary from the training split, not from chars.txt."
            )
            emit("  " + ", ".join(f"U+{ord(c):04X}" for c in extra[:30]))
        emit(f"- In `chars.txt` but never used: **{len(unused)}**")
        if unused:
            emit("  " + ", ".join(f"U+{ord(c):04X}" for c in unused[:30]))
    emit()

    emit("### Characters needing a documented normalization decision")
    emit()
    emit("| Character | Code point | Count | Why it matters |")
    emit("| --- | --- | --- | --- |")
    flagged = 0
    for char, reason in {**ARABIC_VARIANTS, **ZERO_WIDTH}.items():
        count = observed.get(char, 0)
        if count:
            flagged += 1
            shown = "(invisible)" if char in ZERO_WIDTH else char
            emit(f"| {shown} | U+{ord(char):04X} | {count} | {reason} |")
    presentation = {c: n for c, n in observed.items() if 0xFB50 <= ord(c) <= 0xFEFF}
    for char, count in sorted(presentation.items(), key=lambda kv: -kv[1]):
        flagged += 1
        emit(f"| {char} | U+{ord(char):04X} | {count} | "
             f"Arabic Presentation Form: one code point for a whole ligature |")
    latin = {c: n for c, n in observed.items() if c.isascii() and c.isalpha()}
    for char, count in sorted(latin.items(), key=lambda kv: -kv[1]):
        flagged += 1
        emit(f"| {char} | U+{ord(char):04X} | {count} | "
             f"Latin letter in Urdu ground truth: likely annotation noise |")
    if flagged == 0:
        emit("| - | - | 0 | none found |")
    emit()
    if flagged:
        audit.note(
            f"{flagged} character classes need an explicit normalization rule before "
            "CER/WER is computed; the same rule must apply to both models."
        )

    # ---- 10. Unicode normalization effect ----------------------------------------
    emit("## 9. Effect of Unicode normalization")
    emit()
    emit("| Form | Labels changed | Distinct chars after |")
    emit("| --- | --- | --- |")
    for form in ("NFC", "NFD", "NFKC"):
        changed = 0
        chars_after = set()
        for record in all_records:
            normalized = unicodedata.normalize(form, record.label)
            if normalized != record.label:
                changed += 1
            chars_after.update(normalized)
        emit(f"| {form} | {changed} | {len(chars_after)} |")
    emit()
    emit("NFKC folds Arabic Presentation Forms into their constituent letters, which")
    emit("is usually what an OCR evaluation wants, but it also changes other")
    emit("characters. Whichever form is chosen, it is applied by the same shared code")
    emit("to both models' predictions and to the ground truth, and the raw")
    emit("transcription is kept alongside it in the manifests.")
    emit()

    # ---- Summary ------------------------------------------------------------------
    emit("## Summary")
    emit()
    if audit.problems:
        emit(f"**{len(audit.problems)} problem(s) found:**")
        emit()
        for item in audit.problems:
            emit(f"- {item}")
    else:
        emit("**No blocking problems found.**")
    emit()
    if audit.notes:
        emit(f"**{len(audit.notes)} item(s) to carry forward:**")
        emit()
        for item in audit.notes:
            emit(f"- {item}")
    emit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.out}", file=sys.stderr)

    return 1 if audit.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
