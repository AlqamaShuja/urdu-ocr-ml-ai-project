"""Local environment smoke test for the Urdu OCR research project.

Run this after creating the virtual environment and installing requirements:

    python scripts/check_env.py

It checks, and never fails hard on, four things:
  1. Python version and interpreter path (must be the project .venv).
  2. Core data/imaging packages import correctly.
  3. PyTorch imports, runs a tiny CPU tensor op, and reports CUDA availability.
  4. Urdu Unicode text can be held and printed without mangling.

Nothing here downloads data or trains a model.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys

# Windows consoles default to a legacy code page, which cannot print Urdu.
# Force UTF-8 so the Unicode check below reflects Python's handling, not the terminal's.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "pyyaml"),
    ("jiwer", "jiwer"),
]

# A short Urdu line used only to verify encoding, not as data.
URDU_SAMPLE = "اردو متن کی شناخت"

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def check_python() -> bool:
    section("1. Python interpreter")
    print(f"  version    : {platform.python_version()}")
    print(f"  executable : {sys.executable}")
    print(f"  platform   : {platform.platform()}")

    ok = sys.version_info[:2] == (3, 12)
    print(f"  {PASS if ok else WARN} expected Python 3.12, found {sys.version_info.major}.{sys.version_info.minor}")

    in_venv = sys.prefix != sys.base_prefix
    print(f"  {PASS if in_venv else WARN} running inside a virtual environment: {in_venv}")
    if not in_venv:
        print("         -> activate .venv first, otherwise packages go to the global Python")
    return ok


def check_core_packages() -> bool:
    section("2. Core packages")
    all_ok = True
    for module_name, pip_name in CORE_PACKAGES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            all_ok = False
            print(f"  {FAIL} {pip_name:<16} import failed: {exc}")
            continue
        try:
            version = importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        print(f"  {PASS} {pip_name:<16} {version}")
    return all_ok


def check_torch() -> bool:
    section("3. PyTorch (CPU)")
    try:
        import torch
    except ImportError as exc:
        print(f"  {FAIL} torch is not installed: {exc}")
        print("         -> see README.md, section 'Install PyTorch (CPU)'")
        return False
    except OSError as exc:
        # Typically "WinError 126 ... shm.dll or one of its dependencies":
        # torch's wheel is fine but the Visual C++ runtime it links against is missing.
        print(f"  {FAIL} torch is installed but cannot load: {exc}")
        print("         -> most likely the Microsoft Visual C++ 2015-2022 x64")
        print("            Redistributable is missing. See README.md,")
        print("            section 'Troubleshooting: torch WinError 126'.")
        return False

    print(f"  {PASS} torch            {torch.__version__}")

    # Tiny CPU op: confirms the build actually runs, not just that it imports.
    a = torch.ones(2, 3)
    b = torch.full((2, 3), 2.0)
    total = float((a + b).sum())
    expected = 18.0
    op_ok = abs(total - expected) < 1e-6
    print(f"  {PASS if op_ok else FAIL} CPU tensor op    sum((1+2) over 2x3) = {total} (expected {expected})")

    # A CTC loss step is the core of the CRNN baseline, so verify it is present
    # and differentiable on CPU before relying on it later.
    try:
        log_probs = torch.randn(8, 1, 10).log_softmax(dim=2).requires_grad_()
        targets = torch.tensor([[1, 2, 3]])
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths=torch.tensor([8]),
            target_lengths=torch.tensor([3]),
            blank=0,
        )
        loss.backward()
        ctc_ok = log_probs.grad is not None
        print(f"  {PASS if ctc_ok else FAIL} CTC loss         forward+backward works, loss = {loss.item():.4f}")
    except Exception as exc:  # noqa: BLE001 - report any failure rather than crash
        ctc_ok = False
        print(f"  {FAIL} CTC loss         failed: {exc}")

    # Informational only. Local training is planned on CPU; GPU work happens in Colab.
    cuda_available = torch.cuda.is_available()
    print(f"  {WARN if not cuda_available else PASS} CUDA available   {cuda_available}")
    if cuda_available:
        print(f"         device: {torch.cuda.get_device_name(0)}")
        print("         note: local GPU is still not used for VLM fine-tuning (2 GB VRAM).")
    else:
        print("         expected for a CPU build. Heavy training runs in Google Colab.")

    return op_ok and ctc_ok


def check_urdu_unicode() -> bool:
    section("4. Urdu Unicode handling")
    text = URDU_SAMPLE
    print(f"  sample      : {text}")
    print(f"  length      : {len(text)} Python characters (code points, not glyphs)")
    print(f"  codepoints  : {' '.join(f'U+{ord(ch):04X}' for ch in text[:6])} ...")

    roundtrip = text.encode("utf-8").decode("utf-8")
    ok = roundtrip == text
    print(f"  {PASS if ok else FAIL} UTF-8 encode/decode round-trip preserves the string")
    print("  note: the string is stored in logical order. Do NOT reverse it for RTL display.")
    return ok


def main() -> int:
    print("=" * 60)
    print("Urdu OCR project - local environment check")
    print("=" * 60)

    results = {
        "python": check_python(),
        "core packages": check_core_packages(),
        "pytorch": check_torch(),
        "urdu unicode": check_urdu_unicode(),
    }

    section("Summary")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"\nResult: FAILED ({', '.join(failed)}). Fix the items above before Step 2.")
        return 1

    print("\nResult: PASSED. Local environment is ready for dataset inspection (Step 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
