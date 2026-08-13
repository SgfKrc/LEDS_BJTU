"""Create and validate the project-local test environment.

The default mode is a fully isolated virtual environment. ``--reuse-runtime``
creates a lightweight overlay that can read the existing runtime packages while
all pip writes still go to ``.venv-test``. The overlay is useful on CUDA
development machines where duplicating PyTorch is expensive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv-test"
REQUIREMENTS = ROOT / "requirements-test.txt"


def _python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _uses_system_site_packages() -> bool | None:
    config = VENV_DIR / "pyvenv.cfg"
    if not config.is_file():
        return None
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "include-system-site-packages":
            return value.strip().lower() == "true"
    return None


def _ready() -> bool:
    python = _python_path()
    if not python.is_file():
        return False
    try:
        # With no index and a dry run, pip parses the authoritative requirement
        # files but cannot download or mutate anything.
        dependency_check = subprocess.run(
            [
                str(python), "-m", "pip", "install",
                "--dry-run", "--no-index", "-r", str(REQUIREMENTS),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if dependency_check.returncode != 0:
            return False
        pip_check = subprocess.run(
            [str(python), "-m", "pip", "check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return pip_check.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _remove_test_environment() -> None:
    resolved = VENV_DIR.resolve(strict=False)
    if resolved.parent != ROOT.resolve() or resolved.name != ".venv-test":
        raise RuntimeError(f"refusing to remove unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _install_command(
    *, proxy: str = "", wheelhouse: Path | None = None,
) -> list[str]:
    command = [str(_python_path()), "-m", "pip", "install"]
    if wheelhouse is not None:
        command.extend(["--no-index", f"--find-links={wheelhouse}"])
    elif proxy:
        command.extend(["--proxy", proxy])
    command.extend(["-r", str(REQUIREMENTS)])
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the isolated .venv-test environment",
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("QLH_TEST_PROXY", ""),
        help="pip proxy URL; QLH_TEST_PROXY is used by default",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        default=None,
        help="offline wheel directory (uses --no-index)",
    )
    parser.add_argument(
        "--reuse-runtime",
        action="store_true",
        help="reuse system runtime packages but keep all pip writes in .venv-test",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the existing environment without changing it",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="replace the existing .venv-test (cannot be combined with --check)",
    )
    return parser


def _print_usage() -> None:
    python = _python_path()
    print(f"[test-env] ready: {VENV_DIR}")
    print(f"[test-env] run: {python} scripts/run_test_channels.py")
    print(f"[test-env] direct pytest: {python} -m pytest")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check and args.recreate:
        print("[test-env] --check and --recreate cannot be combined", file=sys.stderr)
        return 2

    wheelhouse = None
    if args.wheelhouse is not None:
        wheelhouse = args.wheelhouse.expanduser().resolve(strict=False)
        if not wheelhouse.is_dir():
            print(f"[test-env] wheelhouse does not exist: {wheelhouse}", file=sys.stderr)
            return 2

    if args.check:
        if not _ready():
            print("[test-env] environment is missing or unhealthy", file=sys.stderr)
            return 1
        _print_usage()
        return 0

    if args.recreate:
        _remove_test_environment()

    existing_mode = _uses_system_site_packages()
    if existing_mode is not None and existing_mode != args.reuse_runtime:
        current = "overlay" if existing_mode else "isolated"
        requested = "overlay" if args.reuse_runtime else "isolated"
        print(
            f"[test-env] existing environment is {current}, requested {requested}; "
            "rerun with its original mode or add --recreate",
            file=sys.stderr,
        )
        return 2

    if _ready():
        _print_usage()
        return 0

    if not _python_path().is_file():
        mode = "runtime overlay" if args.reuse_runtime else "fully isolated"
        print(f"[test-env] creating {mode}: {VENV_DIR}")
        try:
            venv.EnvBuilder(
                with_pip=True,
                system_site_packages=args.reuse_runtime,
            ).create(VENV_DIR)
        except Exception as exc:
            print(f"[test-env] creation failed: {exc}", file=sys.stderr)
            return 1

    print(f"[test-env] installing: {REQUIREMENTS.name}")
    if args.proxy and wheelhouse is None:
        print("[test-env] pip proxy enabled")
    completed = subprocess.run(
        _install_command(proxy=args.proxy, wheelhouse=wheelhouse),
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        print("[test-env] dependency installation failed", file=sys.stderr)
        return completed.returncode
    if not _ready():
        print("[test-env] dependency health check failed", file=sys.stderr)
        return 1
    _print_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
