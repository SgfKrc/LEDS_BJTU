"""Layer-handoff (L -> L) relay probe.

The probe drives the experimental ``llama-relay-gen`` binary (kept in the gitignored
``build/cross-framework-layer-poc`` tree) and decides acceptance through
:func:`src.relay_contract.judge_relay_generation`, i.e. **per-token argmax only** -- never a
cosine threshold, because the ``llama_batch.embd`` path is not bit-reproducible even inside a
single process (experiment §7.11.2).

Design rules:

* missing assets produce a named ``missing_assets`` report and nothing is executed, so the
  probe can never claim a relay run it did not perform;
* the same contract rules as the main path are reused instead of re-deriving them here;
* the runner is injectable, so tests exercise parsing and judgement without loading models.

This is an acceptance probe, not a production relay supervisor.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.relay_contract import RELAY_ACCEPTANCE, judge_relay_generation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "build" / "cross-framework-layer-poc"
DEFAULT_RUNNER = DEFAULT_EXPERIMENT_ROOT / "llama.cpp" / "build-cpu" / "bin" / "llama-relay-gen.exe"
DEFAULT_UPSTREAM_MODEL = DEFAULT_EXPERIMENT_ROOT / "out" / "qwen35-2b-f16.gguf"
DEFAULT_DOWNSTREAM_MODEL = DEFAULT_EXPERIMENT_ROOT / "out" / "qwen35-2b-f16-cut.gguf"
DEFAULT_PROMPT = DEFAULT_EXPERIMENT_ROOT / "out" / "prompt-single.txt"
DEFAULT_GEN = 16
DEFAULT_THREADS = 4

_SEQUENCE_PATTERN = re.compile(r"^(baseline|relay)\s*\((\d+)\)\s*:\s*([0-9\s]*)$", re.MULTILINE)

#: Windows loader failures that mean "the runner never started": ``STATUS_DLL_NOT_FOUND`` and
#: ``STATUS_ENTRYPOINT_NOT_FOUND``. The second is what a mismatched MSYS2/MinGW toolchain
#: produces (observed locally as 3221225785 with zero output).
_STATUS_LOAD_FAILURES = frozenset({0xC0000135, 0xC0000139})

#: Runner signature: (command, timeout_seconds) -> (returncode, combined_output).
Runner = Callable[[list[str], float], "tuple[int, str]"]


@dataclass(frozen=True)
class RelaySequences:
    baseline: list[int]
    relay: list[int]

    @property
    def length(self) -> int:
        return min(len(self.baseline), len(self.relay))


@dataclass(frozen=True)
class RelayProbePlan:
    runner: Path
    upstream_model: Path
    downstream_model: Path
    prompt: Path
    n_gen: int = DEFAULT_GEN
    threads: int = DEFAULT_THREADS
    timeout_seconds: float = 900.0

    @property
    def command(self) -> list[str]:
        return [
            str(self.runner),
            str(self.upstream_model),
            str(self.downstream_model),
            "--prompt",
            str(self.prompt),
            "--gen",
            str(self.n_gen),
            "--threads",
            str(self.threads),
        ]

    def missing_assets(self) -> list[str]:
        return [
            str(path)
            for path in (self.runner, self.upstream_model, self.downstream_model, self.prompt)
            if not path.is_file()
        ]


def build_plan(
    root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    runner: str | Path = DEFAULT_RUNNER,
    upstream_model: str | Path = DEFAULT_UPSTREAM_MODEL,
    downstream_model: str | Path = DEFAULT_DOWNSTREAM_MODEL,
    prompt: str | Path = DEFAULT_PROMPT,
    n_gen: int = DEFAULT_GEN,
    threads: int = DEFAULT_THREADS,
    timeout_seconds: float = 900.0,
) -> RelayProbePlan:
    base = Path(root).expanduser()
    return RelayProbePlan(
        runner=Path(runner).expanduser() if runner else base / "llama.cpp" / "build-cpu" / "bin" / "llama-relay-gen.exe",
        upstream_model=Path(upstream_model).expanduser() if upstream_model else base / "out" / "qwen35-2b-f16.gguf",
        downstream_model=Path(downstream_model).expanduser() if downstream_model else base / "out" / "qwen35-2b-f16-cut.gguf",
        prompt=Path(prompt).expanduser() if prompt else base / "out" / "prompt-single.txt",
        n_gen=max(1, int(n_gen)),
        threads=max(1, int(threads)),
        timeout_seconds=max(1.0, float(timeout_seconds)),
    )


def parse_sequences(stdout: str) -> RelaySequences | None:
    """Extract the ``baseline (N): ...`` / ``relay (N): ...`` token id sequences.

    Returns ``None`` when either line is missing, so a truncated or failed run is never
    mistaken for a completed comparison.
    """
    found: dict[str, list[int]] = {}
    for kind, _declared, payload in _SEQUENCE_PATTERN.findall(stdout or ""):
        tokens = [int(part) for part in payload.split()]
        if tokens:
            found[kind] = tokens
    if "baseline" not in found or "relay" not in found:
        return None
    return RelaySequences(baseline=found["baseline"], relay=found["relay"])


def plan_report(plan: RelayProbePlan) -> dict[str, Any]:
    """Dry-run report: what would run, and which assets are missing."""
    missing = plan.missing_assets()
    return {
        "status": "dry_run",
        "criterion": RELAY_ACCEPTANCE,
        "command": " ".join(plan.command),
        "runner": str(plan.runner),
        "upstream_model": str(plan.upstream_model),
        "downstream_model": str(plan.downstream_model),
        "prompt": str(plan.prompt),
        "gen": plan.n_gen,
        "threads": plan.threads,
        "missing_assets": missing,
        "ready": not missing,
        "note": "acceptance is per-token argmax; cosine is never a gate",
    }


def _default_runner(command: list[str], timeout: float) -> tuple[int, str]:
    # Decode as UTF-8 with replacement: the MSYS2/MinGW llama.cpp binaries emit UTF-8, while
    # text=True would decode with the Windows locale codec (gbk) and lose the entire stream on
    # the first CJK byte, which silently turns a good run into "unparsable_output".
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def run_probe(plan: RelayProbePlan, *, runner: Runner | None = None) -> dict[str, Any]:
    """Execute the baseline/relay pair and judge the result with the relay contract."""
    report = plan_report(plan)
    missing = report["missing_assets"]
    if missing:
        report["status"] = "missing_assets"
        return report

    execute = runner or _default_runner
    try:
        returncode, output = execute(plan.command, plan.timeout_seconds)
    except subprocess.TimeoutExpired:
        report["status"] = "runner_timeout"
        return report
    except (OSError, UnicodeError) as exc:
        report.update({"status": "runner_unavailable", "error": str(exc)})
        return report

    report["runner_returncode"] = returncode
    report["stdout_tail"] = output[-4000:]
    if returncode in _STATUS_LOAD_FAILURES and not output.strip():
        # MSYS2/MinGW binaries need their toolchain bin on PATH; without it the process cannot
        # start at all (loader failure) and prints nothing at all.
        report.update(
            {
                "status": "runner_dll_missing",
                "hint": r"add the toolchain bin to PATH before running (e.g. C:\msys64\ucrt64\bin)",
            }
        )
        return report
    sequences = parse_sequences(output)
    if sequences is None:
        report["status"] = "unparsable_output"
        return report

    verdict = judge_relay_generation(sequences.baseline, sequences.relay)
    report["status"] = "accepted" if verdict.accepted else "rejected"
    report["verdict"] = verdict.to_dict()
    report["baseline_tokens"] = sequences.baseline
    report["relay_tokens"] = sequences.relay
    if returncode != 0 and verdict.accepted:
        # The sequences matched but the process failed: never report a clean acceptance.
        report["status"] = "rejected"
        report["verdict"]["accepted"] = False
        report["verdict"]["reason"] = "runner_failed_after_match"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L -> L relay acceptance probe (per-token argmax)")
    parser.add_argument("--root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--runner", default=None, help="path to llama-relay-gen")
    parser.add_argument("--upstream-model", default=None)
    parser.add_argument("--downstream-model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--gen", type=int, default=DEFAULT_GEN)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    plan = build_plan(
        args.root,
        runner=args.runner or DEFAULT_RUNNER,
        upstream_model=args.upstream_model or DEFAULT_UPSTREAM_MODEL,
        downstream_model=args.downstream_model or DEFAULT_DOWNSTREAM_MODEL,
        prompt=args.prompt or DEFAULT_PROMPT,
        n_gen=args.gen,
        threads=args.threads,
        timeout_seconds=args.timeout,
    )
    report = plan_report(plan) if args.dry_run else run_probe(plan)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0 if report.get("status") in {"dry_run", "accepted"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
