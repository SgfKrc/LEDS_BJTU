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


def _python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _ready() -> bool:
    python = _python_path()
    if not python.is_file():
        return False
    completed = subprocess.run(
        [str(python), "-c", "import transformers; from packaging.version import Version; raise SystemExit(0 if Version(transformers.__version__) >= Version('4.51.0') else 1)"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the isolated Qwen3 Transformers sidecar venv")
    parser.add_argument("--wheelhouse", type=Path, default=None)
    args = parser.parse_args(argv)
    if _ready():
        print(f"Qwen3 sidecar environment ready: {VENV_DIR}")
        return 0
    wheelhouse = args.wheelhouse.expanduser().absolute().resolve(strict=False) if args.wheelhouse else None
    if wheelhouse is not None and not wheelhouse.is_dir():
        print("wheelhouse does not exist", file=sys.stderr)
        return 2
    if not _python_path().is_file():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    command = [str(_python_path()), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(["--no-index", f"--find-links={wheelhouse}"])
    command.extend(["-r", str(REQUIREMENTS)])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not _ready():
        print("Qwen3 sidecar dependency check failed", file=sys.stderr)
        return 1
    print(f"Qwen3 sidecar environment ready: {VENV_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
