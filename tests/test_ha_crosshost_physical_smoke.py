from __future__ import annotations

from types import SimpleNamespace

from scripts import ha_crosshost_physical_smoke as smoke


def test_y700_probe_uses_dynamic_adb_serial_and_arm64_model_gate(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "model=TB321FU\n"
                "sdk=35\n"
                "abi=arm64-v8a,armeabi-v7a\n"
                "nproc=8\n"
                "available_mem_kb=123\n"
                "model_root=/data/data/com.termux/files/home/storage/shared/Download/QLH/models\n"
                "model_root_exists=true\n"
                "gguf_count=6\n"
            ),
            stderr="",
        ),
    )

    result = smoke._run_y700("100.99.211.13:40397", adb="adb")

    assert result["status"] == "passed"
    assert result["transport"] == "adb_wireless_debugging"
    assert result["serial"] == "100.99.211.13:40397"
    assert result["model_gate"] == "ready_for_arm64_model_smoke"
    assert result["observations"]["gguf_count"] == "6"


def test_y700_probe_records_ssh_failure_without_raising(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="connection timed out",
        ),
    )

    result = smoke._run_y700("100.99.211.13:40397", adb="adb")

    assert result["status"] == "failed"
    assert result["error_code"] == "adb_failed"
    assert result["model_gate"] == "blocked_no_gguf"


def test_y700_probe_discovers_online_dynamic_serial(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["devices", "-l"]:
            return SimpleNamespace(
                returncode=0,
                stdout="List of devices attached\n100.99.211.13:40397\tdevice product:TB321FU\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout="model=TB321FU\nabi=arm64-v8a\ngguf_count=1\n",
            stderr="",
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    result = smoke._run_y700(adb="adb", host="100.99.211.13")

    assert result["status"] == "passed"
    assert result["serial"] == "100.99.211.13:40397"
    assert calls[0] == ["adb", "devices", "-l"]
    assert calls[1][0:3] == ["adb", "-s", "100.99.211.13:40397"]


def test_y700_probe_requires_dynamic_serial_when_no_device_is_online(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="List of devices attached\n",
            stderr="",
        ),
    )

    result = smoke._run_y700(adb="adb", host="100.99.211.13")

    assert result["status"] == "failed"
    assert result["error_code"] == "dynamic_adb_serial_required"
    assert "dynamic_port" in result["hint"]
