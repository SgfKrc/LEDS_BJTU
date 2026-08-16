"""Create the isolated Gemma 4 Unified PyTorch pipeline environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv-gemma4-pipeline"
NATIVE_VENV_DIR = ROOT / ".venv-gemma4-native"
REQUIREMENTS = ROOT / "packaging" / "requirements-gemma4-pipeline-sidecar.txt"
TRANSFORMERS_VERSION = "5.10.1"
DEFAULT_TORCH_SPEC = os.environ.get(
    "QLH_GEMMA4_PIPELINE_TORCH_SPEC", "torch>=2.10,<2.14",
)


def _python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _ready() -> bool:
    python = _python_path()
    if not python.is_file():
        return False
    check = (
        "import accelerate, safetensors, torch, transformers; "
        "from transformers import (Gemma4UnifiedConfig, "
        "Gemma4UnifiedForConditionalGeneration); "
        "from transformers.masking_utils import (create_causal_mask, "
        "create_sliding_window_causal_mask); "
        f"ok = transformers.__version__ == '{TRANSFORMERS_VERSION}' "
        "and hasattr(torch, 'no_grad'); "
        "raise SystemExit(0 if ok else 1)"
    )
    completed = subprocess.run(
        [str(python), "-c", check],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return completed.returncode == 0


def _pip_base(python: Path, *, wheelhouse: Path | None, proxy: str) -> list[str]:
    command = [str(python), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(["--no-index", f"--find-links={wheelhouse}"])
    elif proxy:
        command.extend(["--proxy", proxy])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the isolated Gemma 4 PyTorch pipeline venv",
    )
    parser.add_argument("--wheelhouse", type=Path, default=None)
    parser.add_argument(
        "--torch-index-url",
        default=os.environ.get("QLH_GEMMA4_PIPELINE_TORCH_INDEX_URL", ""),
        help="platform-specific Torch index, for example the matching CUDA wheel index",
    )
    parser.add_argument("--torch-spec", default=DEFAULT_TORCH_SPEC)
    parser.add_argument(
        "--proxy",
        default=os.environ.get("QLH_GEMMA4_PIPELINE_PROXY", ""),
        help="optional user-managed HTTP proxy; no project proxy is assumed",
    )
    args = parser.parse_args(argv)
    if _ready():
        print(f"Gemma 4 pipeline environment ready: {VENV_DIR}")
        return 0
    wheelhouse = (
        args.wheelhouse.expanduser().absolute().resolve(strict=False)
        if args.wheelhouse else None
    )
    if wheelhouse is not None and not wheelhouse.is_dir():
        print("wheelhouse does not exist", file=sys.stderr)
        return 2
    if wheelhouse is None and not args.torch_index_url:
        print(
            "a platform-specific --torch-index-url or --wheelhouse is required",
            file=sys.stderr,
        )
        return 2
    python = _python_path()
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if python.resolve(strict=False) == (
        NATIVE_VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ).resolve(strict=False):
        print("the MTMD native venv cannot host the PyTorch pipeline", file=sys.stderr)
        return 2
    torch_command = _pip_base(python, wheelhouse=wheelhouse, proxy=args.proxy)
    if wheelhouse is None:
        torch_command.extend(["--index-url", str(args.torch_index_url)])
    torch_command.append(str(args.torch_spec))
    completed = subprocess.run(torch_command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    dependency_command = _pip_base(
        python, wheelhouse=wheelhouse, proxy=args.proxy,
    )
    dependency_command.extend(["-r", str(REQUIREMENTS)])
    completed = subprocess.run(dependency_command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not _ready():
        print("Gemma 4 pipeline dependency/capability check failed", file=sys.stderr)
        return 1
    print(f"Gemma 4 pipeline environment ready: {VENV_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
