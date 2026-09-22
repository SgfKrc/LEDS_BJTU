"""Run the parallel, external-resource, and real-model test channels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _write_console(text: str) -> None:
    """Write subprocess output without letting a legacy Windows code page abort a run."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    encoded = text.encode(encoding, errors="replace")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(encoded)
        stream.flush()
    else:
        sys.stdout.write(encoded.decode(encoding, errors="replace"))
        sys.stdout.flush()


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (src, current) if value
    )
    return env


def _run_pytest(
    arguments: Sequence[str],
    env: dict[str, str],
    *,
    log_path: Path,
) -> tuple[int, float]:
    command = [sys.executable, "-m", "pytest", *arguments]
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                _write_console(line)
                log.write(line)
            returncode = process.wait()
        except Exception as exc:  # noqa: BLE001 - runner failures need an artifact
            log.write(f"\n[test-channels] runner error: {exc!r}\n")
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            returncode = 2
    return returncode, time.monotonic() - started


def _quality_args(*, junitxml: Path, order_seed: int) -> list[str]:
    return [
        "--junitxml",
        str(junitxml),
        "--maxfail=0",
        "--tb=long",
        "--qlh-order-seed",
        str(order_seed),
    ]


def _unit_args(
    workers: int,
    *,
    dist: str = "loadscope",
    junitxml: Path | None = None,
    order_seed: int | None = None,
) -> list[str]:
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
        arguments.extend(["-n", str(workers), "--dist", dist])
    if junitxml is not None and order_seed is not None:
        arguments.extend(_quality_args(junitxml=junitxml, order_seed=order_seed))
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
    parser.add_argument(
        '--dist',
        choices=('loadscope', 'load'),
        default='loadscope',
        help=(
            'unit-channel xdist distribution: loadscope is the stable default; '
            'load enables true test-level concurrency for race audits'
        ),
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help=(
            'run each selected channel this many complete times; failures are '
            'recorded and are never hidden by retries'
        ),
    )
    parser.add_argument(
        '--order-seed',
        type=int,
        default=(
            int(os.environ['QLH_TEST_ORDER_SEED'])
            if os.environ.get('QLH_TEST_ORDER_SEED')
            else None
        ),
        help='deterministic test-order seed; generated and recorded when omitted',
    )
    parser.add_argument(
        '--artifacts-dir',
        type=Path,
        default=Path(
            os.environ.get('QLH_TEST_ARTIFACTS_DIR', str(ROOT / 'build' / 'audit'))
        ),
        help='directory for per-run JUnit XML, logs, and manifest.json',
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
    if not 1 <= args.repeat <= 100:
        raise SystemExit('--repeat must be between 1 and 100')
    if not _check_python_environment(
        allow_system_python=args.allow_system_python,
    ):
        return 2

    order_seed = args.order_seed
    if order_seed is None:
        order_seed = random.SystemRandom().randrange(1, 2**32)
    artifact_base = args.artifacts_dir.expanduser().resolve()
    artifact_base.mkdir(parents=True, exist_ok=True)
    session_name = (
        'test-channels-'
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f'{os.getpid()}'
    )
    artifact_dir = artifact_base / session_name
    suffix = 1
    while artifact_dir.exists():
        suffix += 1
        artifact_dir = artifact_base / f'{session_name}-{suffix}'
    artifact_dir.mkdir()
    print(f'[test-channels] artifacts: {artifact_dir}', flush=True)
    print(f'[test-channels] order seed: {order_seed}', flush=True)

    env = _pytest_env()
    external_args = [
        'tests',
        '-q',
        '-m',
        'not real_model and (external or requires_db or slow or requires_gpu)',
    ]
    if importlib.util.find_spec('xdist') is not None:
        external_args.extend(['-n', '0'])

    channels = ('unit', 'external')
    failures: list[str] = []
    runs: list[dict[str, object]] = []

    def record_manifest() -> None:
        (artifact_dir / 'manifest.json').write_text(
            json.dumps(
                {
                    'channel': args.channel,
                    'workers': args.workers,
                    'dist': args.dist,
                    'repeat': args.repeat,
                    'order_seed': order_seed,
                    'runs': runs,
                },
                ensure_ascii=False,
                indent=2,
            )
            + '\n',
            encoding='utf-8',
        )

    # Create an evidence file even when the smoke channel is intentionally
    # skipped, so every invocation has a machine-readable outcome.
    record_manifest()

    for repetition in range(1, args.repeat + 1):
        run_order_seed = (order_seed + repetition - 1) % (2**32)
        for name in channels:
            if args.channel not in ('all', name):
                continue
            print(f'\n[test-channels] {name} channel (run {repetition}/{args.repeat})', flush=True)
            junitxml = artifact_dir / f'{name}-{repetition:03d}.xml'
            log_path = artifact_dir / f'{name}-{repetition:03d}.log'
            if name == 'unit':
                command = _unit_args(
                    args.workers,
                    dist=args.dist,
                    junitxml=junitxml,
                    order_seed=run_order_seed,
                )
            else:
                command = [
                    *external_args,
                    *_quality_args(junitxml=junitxml, order_seed=run_order_seed),
                ]
            returncode, duration = _run_pytest(command, env, log_path=log_path)
            runs.append({
                'channel': name,
                'run': repetition,
                'order_seed': run_order_seed,
                'returncode': returncode,
                'duration_s': round(duration, 3),
                'junitxml': junitxml.name,
                'log': log_path.name,
                'command': [sys.executable, '-m', 'pytest', *command],
            })
            record_manifest()
            if returncode != 0:
                failures.append(f'{name}#{repetition}')

    if args.channel in ('all', 'smoke'):
        if os.environ.get('QLH_RUN_REAL_MODEL_SMOKE') != '1':
            print(
                '\n[test-channels] smoke channel skipped: set '
                'QLH_RUN_REAL_MODEL_SMOKE=1 to load real weights.',
                flush=True,
            )
        else:
            print('\n[test-channels] smoke channel', flush=True)
            for repetition in range(1, args.repeat + 1):
                smoke_args = [
                    'tests',
                    '-q',
                    '-m',
                    'real_model',
                ]
                if importlib.util.find_spec('xdist') is not None:
                    smoke_args.extend(['-n', '0'])
                smoke_args.extend(_quality_args(
                    junitxml=artifact_dir / f'smoke-{repetition:03d}.xml',
                    order_seed=(order_seed + repetition - 1) % (2**32),
                ))
                log_path = artifact_dir / f'smoke-{repetition:03d}.log'
                returncode, duration = _run_pytest(smoke_args, env, log_path=log_path)
                runs.append({
                    'channel': 'smoke',
                    'run': repetition,
                    'order_seed': (order_seed + repetition - 1) % (2**32),
                    'returncode': returncode,
                    'duration_s': round(duration, 3),
                    'junitxml': f'smoke-{repetition:03d}.xml',
                    'log': log_path.name,
                    'command': [sys.executable, '-m', 'pytest', *smoke_args],
                })
                record_manifest()
                if returncode != 0:
                    failures.append(f'smoke#{repetition}')

    if failures:
        print('\n[test-channels] failed: ' + ', '.join(failures), flush=True)
        print(f'[test-channels] evidence: {artifact_dir}', flush=True)
        return 1
    print('\n[test-channels] completed', flush=True)
    print(f'[test-channels] evidence: {artifact_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
