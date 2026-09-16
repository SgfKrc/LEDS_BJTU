"""Fallback-path tests for the relay probe (CORE-RELAY-01).

The point of these cases is that a rejection must never *promise* a fallback: with
``verify_fallback`` the probe has to actually run the non-relay (whole-model, token) path and
report whether it worked. The runners are injected, so nothing loads a model here.
"""

from __future__ import annotations

from pathlib import Path

from scripts.llama_relay_probe import build_plan, run_probe

DIVERGING_OUTPUT = (
    "=== 汇总 ===\n"
    "baseline (4): 11751 13 198 32\n"
    "relay    (4): 11751 13 999 32\n"
)
MATCHING_OUTPUT = (
    "=== 汇总 ===\n"
    "baseline (4): 11751 13 198 32\n"
    "relay    (4): 11751 13 198 32\n"
)


def _plan(tmp_path: Path, *, create: bool = True, with_fallback: bool = True, **overrides):
    names = {
        "runner": "llama-relay-gen.exe",
        "fallback_runner": "llama-relay-check.exe",
        "upstream_model": "upstream.gguf",
        "downstream_model": "downstream.gguf",
        "prompt": "prompt.txt",
    }
    if create:
        for key, name in names.items():
            if key == "fallback_runner" and not with_fallback:
                continue
            (tmp_path / name).write_bytes(b"placeholder")
    paths = {key: tmp_path / name for key, name in names.items()}
    paths.update(overrides)
    return build_plan(root=tmp_path, n_gen=4, **paths)


def test_no_fallback_block_when_the_relay_run_is_accepted(tmp_path: Path):
    report = run_probe(
        _plan(tmp_path),
        runner=lambda command, timeout: (0, MATCHING_OUTPUT),
        verify_fallback=True,
    )

    assert report["status"] == "accepted"
    assert "fallback" not in report


def test_rejection_actually_runs_the_single_process_path(tmp_path: Path):
    calls: list[list[str]] = []

    def runner(command: list[str], timeout: float):
        calls.append(command)
        if "--dump" in command:
            Path(command[command.index("--dump") + 1]).write_bytes(b"\x00" * 64)
            return 0, "ok\n"
        return 0, DIVERGING_OUTPUT

    report = run_probe(_plan(tmp_path), runner=runner, verify_fallback=True)

    assert report["status"] == "rejected"
    fallback = report["fallback"]
    assert fallback["engaged"] is True
    assert fallback["strategy"] == "single_process_llama_cpp"
    assert fallback["single_process_ok"] is True
    # The fallback must be the non-relay token path over the *upstream* (whole) model.
    dump_call = [c for c in calls if "--dump" in c][0]
    assert "--tokens" in dump_call
    assert str(tmp_path / "upstream.gguf") in dump_call
    assert "--gen" not in dump_call


def test_missing_fallback_assets_are_reported_not_promised(tmp_path: Path):
    report = run_probe(
        _plan(tmp_path, with_fallback=False),
        runner=lambda command, timeout: (0, DIVERGING_OUTPUT),
        verify_fallback=True,
    )

    fallback = report["fallback"]
    assert fallback["engaged"] is True
    assert fallback["single_process_ok"] is False
    assert fallback["missing_assets"]


def test_failing_single_process_path_is_not_reported_as_available(tmp_path: Path):
    def runner(command: list[str], timeout: float):
        if "--dump" in command:
            return 1, "boom\n"  # never writes the dump file
        return 0, DIVERGING_OUTPUT

    report = run_probe(_plan(tmp_path), runner=runner, verify_fallback=True)

    fallback = report["fallback"]
    assert fallback["single_process_ok"] is False
    assert fallback["reason"] == "single_process_path_failed"


def test_missing_relay_assets_also_verify_the_fallback(tmp_path: Path):
    report = run_probe(_plan(tmp_path, create=False), verify_fallback=True)

    assert report["status"] == "missing_assets"
    fallback = report["fallback"]
    assert fallback["engaged"] is True
    assert fallback["single_process_ok"] is False
    assert fallback["missing_assets"]
