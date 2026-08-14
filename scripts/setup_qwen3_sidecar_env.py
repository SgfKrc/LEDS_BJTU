"""Create the project-local QW3-R2 Transformers sidecar environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv-qwen3-sidecar"
REQUIREMENTS = ROOT / "packaging" / "requirements-qwen3-sidecar.txt"
PIPELINE_REQUIREMENTS = ROOT / "packaging" / "requirements-qwen3-pipeline-sidecar.txt"
DEFAULT_TORCH_SPEC = os.environ.get("QLH_QWEN3_TORCH_SPEC", "torch>=2.0")


def _python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _ready(*, pipeline: bool = False) -> bool:
    python = _python_path()
    if not python.is_file():
        return False
    check = "import transformers; from packaging.version import Version; ok = Version(transformers.__version__) >= Version('4.51.0')"
    if pipeline:
        check += "; import torch, torchvision, accelerate, safetensors, PIL; ok = ok and hasattr(torch, 'no_grad') and hasattr(torchvision, 'transforms') and bool(PIL.__version__)"
    check += "; raise SystemExit(0 if ok else 1)"
    completed = subprocess.run(
        [str(python), "-c", check],
        capture_output=True, text=True, timeout=60, check=False,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the isolated Qwen3 Transformers sidecar venv")
    parser.add_argument("--wheelhouse", type=Path, default=None)
    parser.add_argument("--pipeline", action="store_true", help="also install torch/accelerate for Qwen3 layer execution")
    parser.add_argument(
        "--torch-index-url",
        default=os.environ.get("QLH_QWEN3_TORCH_INDEX_URL", ""),
        help="optional platform-specific Torch index, e.g. https://download.pytorch.org/whl/cu126",
    )
    parser.add_argument(
        "--torch-spec",
        default=DEFAULT_TORCH_SPEC,
        help="Torch requirement used for --pipeline (default: QLH_QWEN3_TORCH_SPEC or torch>=2.0)",
    )
    args = parser.parse_args(argv)
    if _ready(pipeline=args.pipeline):
        print(f"Qwen3 sidecar environment ready: {VENV_DIR} (pipeline={args.pipeline})")
        return 0
    wheelhouse = args.wheelhouse.expanduser().absolute().resolve(strict=False) if args.wheelhouse else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        print("wheelhouse does not exist", file=sys.stderr)
        return 2
    if not _python_path().is_file():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if args.pipeline:
        if wheelhouse is None and not args.torch_index_url:
            print(
                "--pipeline requires --torch-index-url/QLH_QWEN3_TORCH_INDEX_URL "
                "or --wheelhouse so a platform-specific Torch build is explicit",
                file=sys.stderr,
            )
            return 2
        torch_command = [str(_python_path()), "-m", "pip", "install"]
        if wheelhouse is not None:
            torch_command.extend(["--no-index", f"--find-links={wheelhouse}"])
        elif args.torch_index_url:
            torch_command.extend(["--index-url", str(args.torch_index_url)])
        torch_command.extend([str(args.torch_spec)])
        torch_install = subprocess.run(torch_command, check=False)
        if torch_install.returncode != 0:
            return torch_install.returncode
    command = [str(_python_path()), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(["--no-index", f"--find-links={wheelhouse}"])
    command.extend(["-r", str(PIPELINE_REQUIREMENTS if args.pipeline else REQUIREMENTS)])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not _ready(pipeline=args.pipeline):
        print("Qwen3 sidecar dependency check failed", file=sys.stderr)
        return 1
    print(f"Qwen3 sidecar environment ready: {VENV_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
