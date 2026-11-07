"""Character vocabulary for the CRNN, with the CTC blank token.

    python -I -m src.data.vocabulary --dataset nust_uhwr --manifests data/manifests

**The vocabulary is built from the training split only.** Reading validation or
test labels to decide the output layer would leak information from data the model
is not allowed to see, even though it looks harmless. The consequence is faced
rather than avoided: any character that appears only in validation or test cannot
be predicted at all, which puts a floor under CER. That floor is measured and
reported here instead of being hidden by a vocabulary built over everything.

### Why a blank token

CTC aligns a sequence of per-frame predictions to a shorter label without being
told which frame holds which character. It needs one extra symbol, the *blank*,
which means "no character emitted here". Decoding collapses repeated symbols and
then removes blanks, so the blank is also what lets a genuine double letter
survive: "pp" is emitted as p, blank, p rather than p, p.

The blank is index 0 by convention, matching the default of
`torch.nn.functional.ctc_loss`, so no `blank=` argument has to be threaded
through the training code.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

BLANK_TOKEN = "<blank>"
BLANK_INDEX = 0


@dataclass(frozen=True)
class Vocabulary:
    """Maps characters to CTC output indices and back.

    Index 0 is always the blank. Real characters start at index 1.
    """

    chars: tuple[str, ...]

    @property
    def size(self) -> int:
        """Number of CTC output units, blank included. This is the layer width."""
        return len(self.chars) + 1

    @property
    def char_to_index(self) -> dict[str, int]:
        return {char: i for i, char in enumerate(self.chars, start=1)}

    @property
    def index_to_char(self) -> dict[int, str]:
        return {i: char for i, char in enumerate(self.chars, start=1)}

    def encode(self, text: str, *, skip_unknown: bool = True) -> list[int]:
        """Text to CTC target indices.

        Characters outside the vocabulary are skipped by default, because a target
        sequence cannot contain an index the model has no output unit for. Set
        `skip_unknown=False` to make an unknown character an error instead.
        """
        table = self.char_to_index
        indices: list[int] = []
        for char in text:
            index = table.get(char)
            if index is None:
                if skip_unknown:
                    continue
                raise KeyError(f"character not in vocabulary: {char!r} (U+{ord(char):04X})")
            indices.append(index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """Indices to text. Expects already-collapsed indices, blanks removed."""
        table = self.index_to_char
        return "".join(table[i] for i in indices if i != BLANK_INDEX)

    def to_json(self, path: Path) -> None:
        payload = {
            "blank_token": BLANK_TOKEN,
            "blank_index": BLANK_INDEX,
            "size": self.size,
            "n_chars": len(self.chars),
            "chars": list(self.chars),
            "codepoints": [f"U+{ord(c):04X}" for c in self.chars],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Vocabulary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("blank_index") != BLANK_INDEX:
            raise ValueError(f"unexpected blank index in {path}: {payload.get('blank_index')}")
        return cls(chars=tuple(payload["chars"]))

    @classmethod
    def from_texts(cls, texts: list[str]) -> Vocabulary:
        """Build from training text. Sorted by code point for a stable ordering."""
        observed = set()
        for text in texts:
            observed.update(text)
        return cls(chars=tuple(sorted(observed)))


def read_column(path: Path, column: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row[column] for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--column", default="text_normalized",
                        help="manifest column to build from (default: text_normalized)")
    args = parser.parse_args()

    manifests: Path = args.manifests
    train_texts = read_column(manifests / f"{args.dataset}_train.csv", args.column)
    vocab = Vocabulary.from_texts(train_texts)

    print(f"Vocabulary for {args.dataset}, built from the TRAIN split only")
    print(f"  source column : {args.column}")
    print(f"  train samples : {len(train_texts)}")
    print(f"  distinct chars: {len(vocab.chars)}")
    print(f"  CTC output size: {vocab.size}  (= {len(vocab.chars)} characters + 1 blank)")
    print()

    # The point of building from train only: measure what it costs.
    print("Coverage of the held-out splits")
    print()
    known = set(vocab.chars)
    floor_rows = []
    for split in ("val", "test"):
        texts = read_column(manifests / f"{args.dataset}_{split}.csv", args.column)
        counts: Counter[str] = Counter()
        total = 0
        for text in texts:
            counts.update(text)
            total += len(text)
        unseen = {c: n for c, n in counts.items() if c not in known}
        unseen_total = sum(unseen.values())
        floor = unseen_total / total if total else 0.0
        floor_rows.append((split, len(texts), total, unseen, unseen_total, floor))
        print(f"  {split:5} {len(texts):>5} samples, {total:>7} chars, "
              f"{len(unseen)} char type(s) unseen in train, "
              f"{unseen_total} occurrence(s) -> CER floor {floor:.6f} ({floor * 100:.4f}%)")
        for char, count in sorted(unseen.items(), key=lambda kv: -kv[1]):
            name = unicodedata.name(char, "UNNAMED")
            print(f"        U+{ord(char):04X} {char!r} x{count}  {name}")
    print()

    if all(not row[3] for row in floor_rows):
        print("  No character appears only in validation or test. Building the")
        print("  vocabulary from the training split alone costs nothing here, so")
        print("  there is no vocabulary-induced CER floor.")
    else:
        print("  The characters above cannot be produced by the CRNN at all. The")
        print("  resulting floor is small but must be stated in the results, because")
        print("  it applies to the CRNN and not to the VLM, whose tokenizer is not")
        print("  restricted to this dataset's training characters.")
    print()

    out_path = manifests / f"{args.dataset}_vocab.json"
    vocab.to_json(out_path)
    print(f"Written: {out_path}")

    # Round-trip check: a vocabulary that cannot reproduce its own training text
    # would corrupt every CTC target silently.
    reloaded = Vocabulary.from_json(out_path)
    assert reloaded.chars == vocab.chars, "vocabulary did not survive a JSON round-trip"
    sample = train_texts[0]
    assert reloaded.decode(reloaded.encode(sample)) == sample, "encode/decode is not lossless"
    print("Round-trip check: encode -> decode reproduces training text exactly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
