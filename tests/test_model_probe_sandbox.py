"""Contracts for the OS-level template probe sandbox."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from scripts.model_tools.sandbox_runner import run_sandboxed


def test_sandbox_runner_injects_non_secret_runtime_markers() -> None:
    from scripts.model_tools.sandbox_runner import _sandbox_environment

    environment = _sandbox_environment(
        {"PATH": "safe", "HF_TOKEN": "must-not-be-used"},
        {
            "backend": "fixture-sandbox",
            "network_disabled": True,
        },
    )

    assert environment["QLH_OS_SANDBOX_BACKEND"] == "fixture-sandbox"
    assert environment["QLH_OS_SANDBOX_NETWORK_DISABLED"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["HF_TOKEN"] == "must-not-be-used"


@pytest.mark.skipif(os.name != "nt", reason="restricted-token backend is Windows-specific")
def test_windows_backend_starts_a_low_integrity_worker(tmp_path: Path) -> None:
    worker = Path(__file__).parents[1] / "scripts" / "model_tools" / "small_model_template_probe_worker.py"
    model_path = tmp_path / "empty-model"
    model_path.mkdir()
    request = {
        "schema_version": 1,
        "operation": "template_probe",
        "model_path": str(model_path),
        "controller_python": str(Path(sys.executable).with_name("controller.exe")),
        "trust_remote_code": False,
    }
    result = run_sandboxed(
        Path(sys.executable),
        worker,
        input_text=json.dumps(request),
        cwd=Path(__file__).parents[1],
        env={"PATH": os.environ.get("PATH", "")},
        timeout_seconds=15,
    )

    assert result.returncode == 0
    assert result.sandbox["os_level"] is True
    assert result.sandbox["low_privilege"] is True
    payloads = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert payloads[-1]["sandbox"]["backend"] == result.sandbox["backend"]
