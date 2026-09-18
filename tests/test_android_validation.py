"""Android validation runner tests; no SDK, emulator, or device required."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.android_validation as validation


def test_adb_devices_parses_only_online_devices(monkeypatch):
    monkeypatch.setattr(
        validation,
        "_run",
        lambda *args, **kwargs: {
            "stdout": (
                "List of devices attached\n"
                "emulator-5554\tdevice product:sdk_gphone_x86_64 model:emu\n"
                "offline-1\toffline\n"
            ),
            "stderr": "",
            "returncode": 0,
            "ok": True,
        },
    )

    assert validation._adb_devices("adb") == [
        {
            "serial": "emulator-5554",
            "state": "device",
            "product": "sdk_gphone_x86_64",
            "model": "emu",
        }
    ]


def test_no_device_is_not_reported_as_native_worker(monkeypatch):
    def fake_tool(name):
        return None if name in {"adb", "java"} else None

    monkeypatch.setattr(validation, "_tool", fake_tool)
    result = validation.validate(["--skip-unit"])

    assert result["status"] == "failed"
    assert result["evidence"]["apk_device_control_plane"] is False
    assert result["evidence"]["arm64_native_worker"] is False
