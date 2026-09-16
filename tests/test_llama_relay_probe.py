"""Tests for the L -> L relay acceptance probe.

The probe's runner is injected here, so parsing and judgement are exercised without loading
any model; the only filesystem work is empty placeholder assets under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

from scripts.llama_relay_probe import (
    build_plan,
    parse_sequences,
    plan_report,
    run_probe,
)

MATCHING_OUTPUT = (
    "=== 汇总 ===\n"
    "baseline (16): 11751 13 198 32\n"
    "relay    (16): 11751 13 198 32\n"
    "RESULT: 全部 16 步一致（relay == baseline）\n"
)
DIVERGING_OUTPUT = (
    "=== 汇总 ===\n"
    "baseline (16): 11751 13 198 32\n"
    "relay    (16): 11751 13 999 32\n"
)


def _plan(tmp_path: Path, *, create: bool = True, **overrides):
    names = {
        "runner": "llama-relay-gen.exe",
        "upstream_model": "upstream.gguf",
        "downstream_model": "downstream.gguf",
        "prompt": "prompt.txt",
    }
    if create:
        for name in names.values():
            (tmp_path / name).write_bytes(b"placeholder")
    paths = {key: tmp_path / name for key, name in names.items()}
    paths.update(overrides)
    return build_plan(root=tmp_path, **paths)


def test_plan_report_is_a_dry_run_that_names_the_argmax_criterion(tmp_path: Path):
    report = plan_report(_plan(tmp_path))

    assert report["status"] == "dry_run"
    assert report["criterion"] == "per_token_argmax"
    assert report["ready"] is True
    assert "llama-relay-gen.exe" in report["command"]
    assert "--gen" in report["command"]


def test_plan_report_lists_missing_assets_without_failing(tmp_path: Path):
    report = plan_report(_plan(tmp_path, create=False))

    assert report["ready"] is False
    assert len(report["missing_assets"]) == 4
    assert report["status"] == "dry_run"


def test_parse_sequences_reads_both_lines():
    sequences = parse_sequences(MATCHING_OUTPUT)

    assert sequences is not None
    assert sequences.baseline == [11751, 13, 198, 32]
    assert sequences.relay == [11751, 13, 198, 32]
    assert sequences.length == 4


def test_parse_sequences_returns_none_when_a_line_is_missing():
    assert parse_sequences("baseline (16): 1 2 3\n") is None
    assert parse_sequences("") is None


def test_run_probe_refuses_to_execute_with_missing_assets(tmp_path: Path):
    calls: list[list[str]] = []

    def spy(command: list[str], timeout: float):
        calls.append(command)
        return 0, MATCHING_OUTPUT

    report = run_probe(_plan(tmp_path, create=False), runner=spy)

    assert report["status"] == "missing_assets"
    assert calls == []


def test_run_probe_accepts_matching_sequences_through_the_contract(tmp_path: Path):
    report = run_probe(_plan(tmp_path), runner=lambda command, timeout: (0, MATCHING_OUTPUT))

    assert report["status"] == "accepted"
    assert report["verdict"]["accepted"] is True
    assert report["verdict"]["criterion"] == "per_token_argmax"
    assert report["verdict"]["reason"] == "all_tokens_match"
    assert report["relay_tokens"] == report["baseline_tokens"]


def test_run_probe_rejects_a_diverging_relay_sequence(tmp_path: Path):
    report = run_probe(_plan(tmp_path), runner=lambda command, timeout: (0, DIVERGING_OUTPUT))

    assert report["status"] == "rejected"
    assert report["verdict"]["reason"] == "token_mismatch"
    assert report["verdict"]["matched_steps"] == 2
    assert report["verdict"]["diagnostics"] == "first_divergence_step=2"


def test_run_probe_never_reports_acceptance_after_a_runner_failure(tmp_path: Path):
    report = run_probe(_plan(tmp_path), runner=lambda command, timeout: (1, MATCHING_OUTPUT))

    assert report["status"] == "rejected"
    assert report["verdict"]["accepted"] is False
    assert report["verdict"]["reason"] == "runner_failed_after_match"
    assert report["runner_returncode"] == 1


def test_run_probe_names_a_dll_load_failure_instead_of_guessing(tmp_path: Path):
    # MSYS2/MinGW binaries fail to start (0xC0000135) when the toolchain bin is not on PATH,
    # and they print nothing at all; the probe must name that instead of reporting empty output.
    report = run_probe(_plan(tmp_path), runner=lambda command, timeout: (3221225785, ""))

    assert report["status"] == "runner_dll_missing"
    assert "PATH" in report["hint"]
    assert report["criterion"] == "per_token_argmax"


def test_run_probe_reports_unparsable_and_timed_out_runs(tmp_path: Path):
    unparsable = run_probe(_plan(tmp_path), runner=lambda command, timeout: (0, "boom\n"))

    def timing_out(command: list[str], timeout: float):
        raise TimeoutError("simulated")

    import subprocess

    def timing_out_real(command: list[str], timeout: float):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    timed_out = run_probe(_plan(tmp_path), runner=timing_out_real)
    unavailable = run_probe(
        _plan(tmp_path),
        runner=lambda command, timeout: (_ for _ in ()).throw(OSError("no runner")),
    )

    assert unparsable["status"] == "unparsable_output"
    assert timed_out["status"] == "runner_timeout"
    assert unavailable["status"] == "runner_unavailable"
    assert "no runner" in unavailable["error"]


def test_build_plan_clamps_generation_and_thread_bounds(tmp_path: Path):
    plan = build_plan(root=tmp_path, runner=tmp_path / "r.exe", upstream_model=tmp_path / "a",
                      downstream_model=tmp_path / "b", prompt=tmp_path / "p", n_gen=0, threads=0)

    assert plan.n_gen == 1
    assert plan.threads == 1
    assert plan.timeout_seconds > 0
