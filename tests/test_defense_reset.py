from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import defense_reset as reset
import demo_ownership as ownership


def _receipt(path: Path, *, pid: int = 424242, port: int | None = None) -> Path:
    run_id = "a" * 24
    receipt = path / f"{run_id}.json"
    payload = {
        "schema": ownership.RECEIPT_SCHEMA,
        "run_id": run_id,
        "source": "defense_demo",
        "created_by_pid": os.getpid(),
        "ports": [] if port is None else [port],
        "processes": [{
            "pid": pid,
            "create_time": 1234.5,
            "kind": "frontend",
            "root_pid": pid,
        }],
    }
    path.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt


def _config(tmp_path: Path, *, apply: bool = False) -> reset.ResetConfig:
    return reset.ResetConfig(apply=apply, report_path=ROOT / "build" / "defense-reset" / "test.json")


def test_default_dry_run_does_not_stop_or_remove_owned_process(monkeypatch, tmp_path):
    receipt = _receipt(tmp_path)
    monkeypatch.setattr(reset, "inspect_identity", lambda _identity: ("matched", object()))
    monkeypatch.setattr(
        reset,
        "_stop_identities",
        lambda *_args, **_kwargs: pytest.fail("dry run attempted to stop a process"),
    )

    report = reset.run_reset(_config(tmp_path), ownership_root=tmp_path)

    assert report["status"] == "dry_run"
    assert report["mode"] == "dry_run"
    assert report["summary"]["matched_processes"] == 1
    assert report["summary"]["stopped_processes"] == 0
    assert receipt.is_file()


def test_mismatched_identity_fails_closed_and_keeps_receipt(monkeypatch, tmp_path):
    receipt = _receipt(tmp_path)
    monkeypatch.setattr(reset, "inspect_identity", lambda _identity: ("mismatch", None))
    monkeypatch.setattr(
        reset,
        "_stop_identities",
        lambda *_args, **_kwargs: pytest.fail("mismatched receipt attempted to stop a process"),
    )

    report = reset.run_reset(_config(tmp_path, apply=True), ownership_root=tmp_path)

    assert report["status"] == "review_required"
    assert report["summary"]["mismatched_processes"] == 1
    assert report["summary"]["stopped_processes"] == 0
    assert receipt.is_file()


def test_apply_removes_only_stale_receipt_and_owned_atomic_temp(monkeypatch, tmp_path):
    receipt = _receipt(tmp_path)
    owned_temp = tmp_path / f"{'b' * 24}.tmp"
    owned_temp.write_text("partial", encoding="utf-8")
    unrelated = tmp_path / "keep.log"
    unrelated.write_text("evidence", encoding="utf-8")
    evidence = tmp_path.parent / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reset, "inspect_identity", lambda _identity: ("dead", None))

    first = reset.run_reset(_config(tmp_path, apply=True), ownership_root=tmp_path)
    second = reset.run_reset(_config(tmp_path, apply=True), ownership_root=tmp_path)

    assert first["status"] == "reset_complete"
    assert first["summary"]["removed_receipts"] == 1
    assert first["summary"]["removed_temp_files"] == 1
    assert second["status"] == "reset_complete"
    assert second["summary"]["receipt_files"] == 0
    assert not receipt.exists()
    assert not owned_temp.exists()
    assert unrelated.read_text(encoding="utf-8") == "evidence"
    assert evidence.read_text(encoding="utf-8") == "{}"


def test_invalid_receipt_is_retained_for_review(tmp_path):
    path = tmp_path / f"{'c' * 24}.json"
    path.write_text('{"schema":"tampered"}', encoding="utf-8")

    report = reset.run_reset(_config(tmp_path, apply=True), ownership_root=tmp_path)

    assert report["status"] == "review_required"
    assert report["summary"]["invalid_receipts"] == 1
    assert path.is_file()


def test_receipt_loader_rejects_extra_authority_fields(tmp_path):
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["processes"][0]["command"] = "taskkill arbitrary"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="process identity invalid"):
        ownership.load_receipt(receipt)


def test_current_python_process_cannot_be_claimed_as_demo_frontend():
    import psutil

    process = psutil.Process()
    identity = {
        "pid": process.pid,
        "create_time": process.create_time(),
        "kind": "frontend",
        "root_pid": process.pid,
    }

    state, matched = ownership.inspect_identity(identity)

    assert state == "mismatch"
    assert matched is None


def test_command_allowlist_accepts_only_expected_demo_shapes():
    frontend = str(ownership.FRONTEND_ROOT)
    root = str(ownership.ROOT)
    assert ownership._command_matches(
        "frontend",
        ["node.exe", str(ownership.FRONTEND_ROOT / "node_modules" / "vite" / "bin" / "vite.js"), "--host", "127.0.0.1"],
        frontend,
    )
    assert ownership._command_matches(
        "backend",
        ["python.exe", "-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1"],
        root,
    )
    assert ownership._command_matches(
        "failure-worker",
        ["python.exe", str(ownership.ROOT / "scripts" / "demo" / "failure_worker.py"), "--port", "12345"],
        root,
    )
    assert not ownership._command_matches(
        "frontend",
        ["node.exe", "malicious.js", "invite", "--host", "127.0.0.1"],
        frontend,
    )
    assert not ownership._command_matches(
        "backend",
        ["python.exe", "malicious.py", "uvicorn", "src.api_server:app", "--host", "127.0.0.1"],
        root,
    )


def test_report_path_must_stay_inside_repository(tmp_path):
    with pytest.raises(reset.ResetError, match="仓库内"):
        reset.run_reset(reset.ResetConfig(report_path=tmp_path / "outside.json"), ownership_root=tmp_path)


@pytest.mark.skipif(
    os.environ.get("QLH_RUN_DEFENSE_RESET_INTEGRATION") != "1"
    or "PYTEST_XDIST_WORKER" in os.environ,
    reason="must run serially; starts the current CyberGothic Vite fixture and simulates a launcher crash",
)
def test_crashed_fixture_is_dry_run_safe_recovered_and_idempotent():
    existing = list(ownership.OWNERSHIP_ROOT.glob("*.json")) if ownership.OWNERSHIP_ROOT.is_dir() else []
    if existing:
        pytest.fail("integration requires no pre-existing ownership receipts")

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    ledger = ownership.OwnershipLedger("defense_demo", ports=[port])
    frontend = subprocess.Popen(
        [
            "npm.cmd" if os.name == "nt" else "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ownership.FRONTEND_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    ledger.register("frontend", frontend.pid)

    def ready() -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as response:
                return 200 <= response.status < 400
        except OSError:
            return False

    def wait_ready(timeout: float) -> bool:
        # Vite 首次请求要按需编译页面，单次 0.5s 探测会偶发假阴性；
        # 只做有界重试，语义仍是"fixture 是否仍在提供服务"。
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready():
                return True
            time.sleep(0.2)
        return ready()

    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if ready() and list(ownership.OWNERSHIP_ROOT.glob("*.json")):
                break
            time.sleep(0.2)
        else:
            pytest.fail("fixture launcher did not become ready with an ownership receipt")

        # Stop only the receipt monitor, modelling an abruptly lost launcher.
        ledger._stop.set()
        assert ledger._thread is not None
        ledger._thread.join(timeout=2.0)
        time.sleep(0.5)
        assert wait_ready(5.0) is True

        dry = reset.run_reset(reset.ResetConfig())
        reset.write_report(dry, ROOT / "build" / "defense-reset" / "a4-dry-run.json")
        assert dry["status"] == "dry_run"
        assert dry["summary"]["stopped_processes"] == 0
        assert wait_ready(5.0) is True

        applied = reset.run_reset(reset.ResetConfig(apply=True))
        reset.write_report(applied, ROOT / "build" / "defense-reset" / "a4-crash-recovery.json")
        assert applied["status"] == "reset_complete"
        assert applied["summary"]["stopped_processes"] >= 1
        assert not list(ownership.OWNERSHIP_ROOT.glob("*.json"))
        assert ready() is False

        repeated = reset.run_reset(reset.ResetConfig(apply=True))
        reset.write_report(repeated, reset.DEFAULT_REPORT)
        assert repeated["status"] == "reset_complete"
        assert repeated["summary"]["receipt_files"] == 0
    finally:
        reset.run_reset(reset.ResetConfig(apply=True))
        if frontend.poll() is None:
            frontend.kill()
            frontend.wait(timeout=5.0)
        ledger.close()
