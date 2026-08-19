"""SYNC-SSH-S1 fake helper/server loopback tests using disposable git checkouts."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging"))

import ssh_sync_contract as contract  # noqa: E402
import ssh_sync_helper as helper  # noqa: E402
from signing import canonical_json  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip()


def _write_verify_key(path: Path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private = Ed25519PrivateKey.generate()
    raw = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    path.write_text(json.dumps({
        "key_id": "s1-test-key",
        "public_key": base64.b64encode(raw).decode("ascii"),
    }), encoding="utf-8")
    return private


def _signed_frame(private, commit_sha: str, *, repo: str) -> dict:
    body = {
        "schema": "qlh.patch_frame.v1",
        "ts": "2026-08-19T00:00:00+00:00",
        "repo": repo,
        "branch": "dev",
        "commit_sha": commit_sha,
        "proxy_port": 7897,
        "note": "S1 loopback",
    }
    return {
        **body,
        "signature": base64.b64encode(private.sign(canonical_json(body))).decode("ascii"),
        "key_id": "s1-test-key",
    }


@pytest.fixture
def patch_topology(tmp_path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    worker = tmp_path / "worker"
    state_dir = tmp_path / "helper-state"
    verify_key = tmp_path / "patch-main.pub.json"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(tmp_path, "clone", str(remote), str(seed))
    _git(seed, "checkout", "-b", "dev")
    _git(seed, "config", "user.name", "QLH S1 Test")
    _git(seed, "config", "user.email", "s1@example.invalid")
    seed.joinpath("app.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "app.txt")
    _git(seed, "commit", "-m", "base")
    _git(seed, "push", "-u", "origin", "dev")
    _git(tmp_path, "clone", "--branch", "dev", str(remote), str(worker))
    initial = _git(worker, "rev-parse", "HEAD")

    seed.joinpath("app.txt").write_text("target\n", encoding="utf-8")
    _git(seed, "commit", "-am", "target")
    target = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "dev")

    private = _write_verify_key(verify_key)
    frame = _signed_frame(private, target, repo=str(remote))
    operation_id = hashlib.sha256(("s1\0" + target).encode("ascii")).hexdigest()
    request = contract.build_helper_request("apply", {
        "frame": frame,
        "operation_id": operation_id,
    })
    server = helper.FakeSshServer(repo=worker, state_dir=state_dir, verify_key=verify_key)
    return {
        "remote": remote,
        "worker": worker,
        "state_dir": state_dir,
        "verify_key": verify_key,
        "frame": frame,
        "request": request,
        "operation_id": operation_id,
        "server": server,
        "initial": initial,
        "target": target,
    }


def test_fake_ssh_loopback_applies_signed_reachable_patch(patch_topology):
    result = patch_topology["server"].exchange(patch_topology["request"])
    assert result == {
        "schema": contract.HELPER_RESULT_SCHEMA,
        "action": "apply",
        "status": "ok",
        "commit_sha": patch_topology["target"],
    }
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["target"]
    ledger = json.loads((patch_topology["state_dir"] / "ssh-sync-helper-ledger.json").read_text(encoding="utf-8"))
    assert ledger["schema"] == contract.HELPER_LEDGER_SCHEMA
    assert ledger["operations"][patch_topology["operation_id"]]["commit_sha"] == patch_topology["target"]


def test_tampered_signed_frame_fails_closed_before_fetch_or_reset(patch_topology, monkeypatch):
    calls: list[list[str]] = []
    original = helper._git

    def observed(repo, args, **kwargs):
        calls.append(args)
        return original(repo, args, **kwargs)

    monkeypatch.setattr(helper, "_git", observed)
    frame = dict(patch_topology["frame"])
    frame["note"] = "tampered"
    request = contract.build_helper_request("apply", {
        "frame": frame,
        "operation_id": patch_topology["operation_id"],
    })
    result = patch_topology["server"].exchange(request)
    assert result["error_code"] == "patch_signature_invalid"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]
    assert not any(args[0] in {"fetch", "reset"} for args in calls)


def test_signed_frame_for_another_repository_fails_closed(patch_topology):
    frame = dict(patch_topology["frame"])
    frame["repo"] = "test://other/repository"
    request = contract.build_helper_request("apply", {
        "frame": frame,
        "operation_id": patch_topology["operation_id"],
    })
    result = patch_topology["server"].exchange(request)
    assert result["error_code"] == "patch_signature_invalid"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]


def test_dirty_checkout_fails_closed_then_allows_same_operation_after_manual_cleanup(patch_topology):
    dirty = patch_topology["worker"] / "untracked-local.txt"
    dirty.write_text("do not overwrite\n", encoding="utf-8")
    result = patch_topology["server"].exchange(patch_topology["request"])
    assert result["error_code"] == "workspace_dirty"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]
    assert not (patch_topology["state_dir"] / "ssh-sync-helper-ledger.json").exists()

    dirty.unlink()
    replay = patch_topology["server"].exchange(patch_topology["request"])
    assert replay["status"] == "ok"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["target"]


def test_lost_ack_replays_from_durable_receipt_without_second_reset(patch_topology, monkeypatch):
    with pytest.raises(helper.FakeSshDisconnected):
        patch_topology["server"].exchange(patch_topology["request"], fault="disconnect_after_apply")
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["target"]

    resets: list[list[str]] = []
    original = helper._git

    def observed(repo, args, **kwargs):
        if args[0] == "reset":
            resets.append(args)
        return original(repo, args, **kwargs)

    monkeypatch.setattr(helper, "_git", observed)
    result = patch_topology["server"].exchange(patch_topology["request"])
    assert result["status"] == "ok"
    assert resets == []


def test_operation_receipt_cannot_be_reused_for_a_different_commit(patch_topology):
    helper.HelperLedger(patch_topology["state_dir"]).put(
        patch_topology["operation_id"],
        contract.build_helper_result("apply", commit_sha=patch_topology["initial"]),
    )
    result = patch_topology["server"].exchange(patch_topology["request"])
    assert result["error_code"] == "remote_helper_rejected"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]


def test_lock_held_and_disconnect_before_do_not_change_checkout(patch_topology):
    lock_path = patch_topology["state_dir"] / "ssh-sync-maintenance.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(contract.build_maintenance_lock("f" * 64)), encoding="utf-8")
    locked = patch_topology["server"].exchange(patch_topology["request"])
    assert locked["error_code"] == "maintenance_lock_held"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]

    lock_path.unlink()
    with pytest.raises(helper.FakeSshDisconnected):
        patch_topology["server"].exchange(patch_topology["request"], fault="disconnect_before")
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]


def test_fetch_verify_and_fixed_command_use_deployment_owned_paths(patch_topology, monkeypatch, capsys):
    fetch = contract.build_helper_request("fetch", {"commit_sha": patch_topology["target"]})
    assert patch_topology["server"].exchange(fetch)["commit_sha"] == patch_topology["target"]
    verify = contract.build_helper_request("verify", {"commit_sha": patch_topology["target"]})
    assert patch_topology["server"].exchange(verify)["error_code"] == "remote_helper_rejected"

    config_path = patch_topology["state_dir"].parent / "fixed-helper.json"
    config_path.write_text(json.dumps({
        "schema": "qlh.sync_ssh.helper-config.v1",
        "repo": str(patch_topology["worker"]),
        "state_dir": str(patch_topology["state_dir"]),
        "verify_key": str(patch_topology["verify_key"]),
    }), encoding="utf-8")
    monkeypatch.setenv("QLH_SSH_SYNC_HELPER_CONFIG", str(config_path))
    encoded = base64.urlsafe_b64encode(json.dumps(patch_topology["frame"], sort_keys=True, separators=(",", ":")).encode("utf-8")).decode("ascii").rstrip("=")
    exit_code = helper.main([
        "--protocol", contract.HELPER_SCHEMA, "apply",
        "--frame-b64url", encoded,
        "--maintenance-operation", patch_topology["operation_id"],
    ])
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["status"] == "ok"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["target"]

    monkeypatch.setenv(
        "SSH_ORIGINAL_COMMAND",
        f"{contract.REMOTE_HELPER} --protocol {contract.HELPER_SCHEMA} status",
    )
    assert helper.main(["--forced-command"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "status"
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "cmd.exe /c whoami")
    assert helper.main(["--forced-command"]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "helper_action_invalid"


def test_helper_rejects_state_or_verify_material_inside_checkout(patch_topology):
    inside = patch_topology["worker"] / "unsafe-state"
    result = helper.FakeSshServer(
        repo=patch_topology["worker"], state_dir=inside,
        verify_key=patch_topology["verify_key"],
    ).exchange(patch_topology["request"])
    assert result["error_code"] == "repository_unmanaged"
    assert _git(patch_topology["worker"], "rev-parse", "HEAD") == patch_topology["initial"]
