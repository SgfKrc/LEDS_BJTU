"""Run the parallel, external-resource, and real-model test channels."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (src, current) if value
    )
    return env


def _run_pytest(arguments: Sequence[str], env: dict[str, str]) -> int:
    command = [sys.executable, "-m", "pytest", *arguments]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def _unit_args(workers: int) -> list[str]:
    arguments = [
        "tests",
        "-q",
        "-m",
        "not external and not requires_db and not real_model and not slow and not requires_gpu",
    ]
    if importlib.util.find_spec("xdist") is None:
        print(
            "[test-channels] pytest-xdist is missing; the unit channel will "
            "fall back to serial execution. Install requirements-test.txt "
            "in the project virtual environment to enable parallel tests.",
            flush=True,
        )
    else:
        arguments.extend(["-n", str(workers), "--dist", "loadscope"])
    return arguments


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--channel',
        choices=('all', 'unit', 'external', 'smoke'),
        default='all',
        help='channel to run; all uses unit -> external -> smoke order',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=int(os.environ.get('QLH_TEST_WORKERS', '4')),
        help='parallel unit-test workers (default: 4)',
    )
    parser.add_argument(
        '--allow-system-python',
        action='store_true',
        help=(
            'bypass the virtual-environment guard for disposable CI images or '
            'dependency diagnostics'
        ),
    )
    return parser.parse_args(argv)


def _in_virtual_environment() -> bool:
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)


def _check_python_environment(*, allow_system_python: bool) -> bool:
    if _in_virtual_environment() or allow_system_python:
        return True
    test_python = ROOT / '.venv-test' / (
        'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    )
    print(
        '[test-channels] refusing to run with system Python because test '
        'dependencies can change the shared runtime.',
        file=sys.stderr,
    )
    print(
        '[test-channels] prepare it with: python scripts/setup_test_env.py',
        file=sys.stderr,
    )
    print(
        f'[test-channels] then run: {test_python} '
        'scripts/run_test_channels.py',
        file=sys.stderr,
    )
    print(
        '[test-channels] use --allow-system-python only in a disposable '
        'environment.',
        file=sys.stderr,
    )
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise SystemExit('--workers must be between 1 and 16')
    if not _check_python_environment(
        allow_system_python=args.allow_system_python,
    ):
        return 2

    env = _pytest_env()
    external_args = [
        'tests',
        '-q',
        '-m',
        'not real_model and (external or requires_db or slow or requires_gpu)',
    ]
    if importlib.util.find_spec('xdist') is not None:
        external_args.extend(['-n', '0'])

    channels = (
        ('unit', _unit_args(args.workers)),
        ('external', external_args),
    )
    failures: list[str] = []

    for name, command in channels:
        if args.channel not in ('all', name):
            continue
        print(f'\n[test-channels] {name} channel', flush=True)
        if _run_pytest(command, env) != 0:
            failures.append(name)

    if args.channel in ('all', 'smoke'):
        if os.environ.get('QLH_RUN_REAL_MODEL_SMOKE') != '1':
            print(
                '\n[test-channels] smoke channel skipped: set '
                'QLH_RUN_REAL_MODEL_SMOKE=1 to load real weights.',
                flush=True,
            )
        else:
            print('\n[test-channels] smoke channel', flush=True)
            smoke_args = [
                'tests/test_real_model_smoke.py',
                '-q',
                '-m',
                'real_model',
            ]
            if importlib.util.find_spec('xdist') is not None:
                smoke_args.extend(['-n', '0'])
            if _run_pytest(smoke_args, env) != 0:
                failures.append('smoke')

    if failures:
        print('\n[test-channels] failed: ' + ', '.join(failures), flush=True)
        return 1
    print('\n[test-channels] completed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
