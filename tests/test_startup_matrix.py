from __future__ import annotations

import os
from pathlib import Path

from scripts import startup_matrix


def test_pytest_profiles_use_isolated_serial_commands():
    python_path = Path(".venv-test") / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    command = startup_matrix._pytest_command(
        python_path,
        startup_matrix.NO_TORCH_TESTS,
    )

    assert Path(command[0]).name in {"python", "python.exe"}
    assert command[1:4] == ["-m", "pytest", startup_matrix.NO_TORCH_TESTS[0]]
    if startup_matrix.importlib.util.find_spec("xdist") is not None:
        assert command[-2:] == ["-n", "0"]


def test_matrix_runs_selected_profiles_and_collects_failures(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    test_python = tmp_path / "test-python"
    edge_python = tmp_path / "edge-python"
    test_python.write_text("", encoding="ascii")
    edge_python.write_text("", encoding="ascii")

    monkeypatch.setattr(startup_matrix, "_is_virtual_environment", lambda _path: True)

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if any(str(item).endswith("edge_preflight.py") for item in command):
            return {
                "ok": True,
                "returncode": 0,
                "stdout": '{"ok": true, "checks": {"cold_start": true}}',
                "stderr": "",
            }
        return {"ok": False, "returncode": 1, "stdout": "pytest output", "stderr": "failure"}

    monkeypatch.setattr(startup_matrix, "_run_subprocess", fake_run)

    report = startup_matrix.run_matrix(
        "all",
        test_python=test_python,
        edge_python=edge_python,
    )

    assert report["ok"] is False
    assert [item["profile"] for item in report["results"]] == [
        "full",
        "no-torch",
        "tui",
        "edge",
    ]
    assert len(calls) == 4
    assert report["results"][-1]["preflight"]["ok"] is True


def test_edge_profile_rejects_non_json_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(
        startup_matrix,
        "_run_subprocess",
        lambda *_args, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": "not json",
            "stderr": "",
        },
    )

    result = startup_matrix._run_edge_profile(tmp_path / "python", timeout_s=1)

    assert result["profile"] == "edge"
    assert result["ok"] is False
    assert "did not emit JSON" in result["error"]
