"""Build project manifests from a raw dataset.

    python -I -m src.data.prepare_dataset --dataset nust_uhwr \
        --root data/raw/nust_uhwr/UHWR --out data/manifests

A manifest is one CSV per split, and it is the single source of truth for every
later step. The CRNN loader and the VLM loader both read these files, so the two
models are guaranteed to see the same examples.

Columns written:

| Column            | Meaning |
| ----------------- | ------- |
| `image_id`        | file name, e.g. `04746.jpg` |
| `image_path`      | path relative to the project root, forward slashes |
| `split`           | train / val / test, as published by the dataset |
| `domain`          | printed / handwritten |
| `granularity`     | word / line |
| `text_raw`        | the transcription exactly as shipped |
| `text_normalized` | `normalize_for_eval(text_raw)` |
| `n_chars_raw`     | length of `text_raw` |
| `n_chars_norm`    | length of `text_normalized` |
| `width`, `height` | pixel dimensions |

Both the raw and the normalized text are kept. The raw column means the
normalization decision can be revisited later without re-reading the dataset, and
it backs the stricter raw-text metric the project brief asks for.

Exclusions are declared in `EXCLUSIONS` below rather than applied silently, and
every excluded record is reported in the build summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.normalize_text import normalize_for_eval  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

SPLITS = ("train", "val", "test")

FIELDNAMES = [
    "image_id",
    "image_path",
    "split",
    "domain",
    "granularity",
    "text_raw",
    "text_normalized",
    "n_chars_raw",
    "n_chars_norm",
    "width",
    "height",
]


@dataclass(frozen=True)
class Exclusion:
    image_id: str
    split: str
    reason: str


# Decided during the 2026-11-02 audit; see reports/dataset_card.md for the
# evidence behind each one.
EXCLUSIONS: dict[str, tuple[Exclusion, ...]] = {
    "nust_uhwr": (
        Exclusion(
            image_id="04745.jpg",
            split="val",
            reason=(
                "byte-identical to training image 04746.jpg (MD5 "
                "b0b7bf6baf997080550f0dff00e2891e) and its label does not match its "
                "image. Visual inspection confirmed the image shows the TRAIN label, "
                "so this validation record is mislabelled. Excluding it closes a "
                "train/validation leak and removes a record that would penalise a "
                "correct reading."
            ),
        ),
    ),
}

DATASET_META = {
    "nust_uhwr": {"domain": "handwritten", "granularity": "line"},
}


def read_split(root: Path, split: str) -> list[tuple[str, str]]:
    """Read one shipped manifest. Format: relative/path<TAB>transcription.

    The shipped files have no trailing newline, so splitting on newline and
    skipping blanks is the correct way to count them; `wc -l` under-reports by one.
    """
    path = root / f"{split}.txt"
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rel_path, _, label = line.partition("\t")
        if not _:
            raise ValueError(f"{path.name}: line without a tab separator: {line[:60]!r}")
        rows.append((rel_path.strip(), label))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_META))
    parser.add_argument("--root", required=True, type=Path, help="extracted dataset root")
    parser.add_argument("--out", required=True, type=Path, help="manifest output directory")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = DATASET_META[args.dataset]
    exclusions = {(e.split, e.image_id): e for e in EXCLUSIONS.get(args.dataset, ())}
    applied: list[Exclusion] = []

    print(f"Building manifests for {args.dataset}")
    print(f"  source : {root}")
    print(f"  output : {out_dir.resolve()}")
    print()

    summary: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        rows = read_split(root, split)
        written = 0
        char_counter: Counter[str] = Counter()
        normalized_changed = 0

        out_path = out_dir / f"{args.dataset}_{split}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()

            for rel_path, text_raw in rows:
                image_id = Path(rel_path).name
                if (split, image_id) in exclusions:
                    applied.append(exclusions[(split, image_id)])
                    continue

                image_path = root / rel_path
                if not image_path.is_file():
                    raise FileNotFoundError(f"{split}: manifest references missing {image_path}")

                with Image.open(image_path) as img:
                    width, height = img.width, img.height

                text_normalized = normalize_for_eval(text_raw)
                if text_normalized != text_raw:
                    normalized_changed += 1
                char_counter.update(text_normalized)

                writer.writerow(
                    {
                        "image_id": image_id,
                        # Stored relative to the project root with forward slashes, so
                        # the same manifest works on Windows locally and on Linux in Colab.
                        "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                        "split": split,
                        "domain": meta["domain"],
                        "granularity": meta["granularity"],
                        "text_raw": text_raw,
                        "text_normalized": text_normalized,
                        "n_chars_raw": len(text_raw),
                        "n_chars_norm": len(text_normalized),
                        "width": width,
                        "height": height,
                    }
                )
                written += 1

        summary[split] = {
            "shipped_records": len(rows),
            "written": written,
            "excluded": len(rows) - written,
            "labels_changed_by_normalization": normalized_changed,
            "distinct_chars_after_normalization": len(char_counter),
        }
        print(f"  {split:5}  {written:>6} rows  "
              f"(from {len(rows)} shipped, {len(rows) - written} excluded)  "
              f"-> {out_path.name}")

    print()
    if applied:
        print(f"Exclusions applied ({len(applied)}):")
        for exclusion in applied:
            print(f"  - {exclusion.image_id} from {exclusion.split}")
            print(f"    {exclusion.reason}")
        print()

    expected = {e.split for e in EXCLUSIONS.get(args.dataset, ())}
    missed = expected - {e.split for e in applied}
    if missed:
        print(f"WARNING: declared exclusions were not found in splits {sorted(missed)}. "
              "The source data may have changed.")
        print()

    total = sum(int(s["written"]) for s in summary.values())
    print(f"Total rows written: {total}")

    # A machine-readable record of what this build did, so a results table can
    # cite exact counts without re-deriving them.
    build_record = {
        "dataset": args.dataset,
        "source_root": str(root),
        "domain": meta["domain"],
        "granularity": meta["granularity"],
        "normalization": "normalize_for_eval (NFKC + invisible removal + Arabic variant folding + whitespace collapse)",
        "splits": summary,
        "total_rows": total,
        "exclusions": [
            {"image_id": e.image_id, "split": e.split, "reason": e.reason} for e in applied
        ],
    }
    record_path = out_dir / f"{args.dataset}_build.json"
    record_path.write_text(json.dumps(build_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Build record: {record_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
