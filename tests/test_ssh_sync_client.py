"""SYNC-SSH-S2 controlled-client tests.  No SSH server is contacted."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ssh_sync_client as client  # noqa: E402
import ssh_sync_contract as contract  # noqa: E402


def _openssh_ed25519(raw: bytes) -> str:
    wire = (
        len(b"ssh-ed25519").to_bytes(4, "big") + b"ssh-ed25519" +
        len(raw).to_bytes(4, "big") + raw
    )
    return "ssh-ed25519 " + base64.b64encode(wire).decode("ascii")


@pytest.fixture
def profile(tmp_path):
    host_key = _openssh_ed25519(bytes(range(32)))
    return contract.load_profile({
        "schema": contract.PROFILE_SCHEMA,
        "host": "100.100.52.106",
        "port": 22,
        "user": "qlh_sync",
        "host_public_key": host_key,
        "host_fingerprint": contract.host_fingerprint(host_key),
        "identity_file": str(tmp_path / "identity"),
        "identity_public_key": _openssh_ed25519(bytes(range(32, 64))),
    })


def _runner(*, stdout: str, returncode: int = 0, observed: list | None = None):
    def run(argv, timeout):
        if observed is not None:
            observed.append((list(argv), timeout))
        return subprocess.CompletedProcess(argv, returncode, stdout, "ignored stderr")
    return run


def _result(action: str, *, status: str = "ok") -> str:
    if status == "ok":
        raw = contract.build_helper_result(action, commit_sha="a" * 40)
    else:
        raw = contract.build_helper_result(action, status="error", error_code="workspace_dirty")
    return json.dumps(raw)


def test_client_writes_dedicated_known_hosts_and_runs_only_fixed_helper(profile, tmp_path):
    observed: list = []
    result = client.execute_helper(
        profile, contract.build_helper_request("status"),
        known_hosts_file=tmp_path / "state" / "known_hosts", runner=_runner(stdout=_result("status"), observed=observed),
    )
    assert result["status"] == "ok"
    known_hosts = (tmp_path / "state" / "known_hosts").read_text(encoding="utf-8")
    assert known_hosts == profile.known_hosts_entry()
    argv, timeout = observed[0]
    assert timeout == client.DEFAULT_TIMEOUT_SECONDS
    assert argv[-4:] == ["qlh-patch-helper", "--protocol", contract.HELPER_SCHEMA, "status"]
    assert "PasswordAuthentication=no" in argv
    assert "StrictHostKeyChecking=yes" in argv


def test_client_preserves_valid_helper_error_even_when_ssh_exits_nonzero(profile, tmp_path):
    result = client.execute_helper(
        profile, contract.build_helper_request("verify", {"commit_sha": "a" * 40}),
        known_hosts_file=tmp_path / "known_hosts", runner=_runner(stdout=_result("verify", status="error"), returncode=1),
    )
    assert result["error_code"] == "workspace_dirty"
    assert result["next_action"] == "manual_review"


def test_client_maps_timeout_and_invalid_output_to_stable_failures(profile, tmp_path):
    def timed_out(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    timeout = client.execute_helper(
        profile, contract.build_helper_request("fetch", {"commit_sha": "a" * 40}),
        known_hosts_file=tmp_path / "known_hosts-a", runner=timed_out,
    )
    assert timeout["error_code"] == "transport_unavailable"
    malformed = client.execute_helper(
        profile, contract.build_helper_request("status"),
        known_hosts_file=tmp_path / "known_hosts-b", runner=_runner(stdout="server banner\n"),
    )
    assert malformed["error_code"] == "remote_helper_rejected"
    missing_helper = client.execute_helper(
        profile, contract.build_helper_request("status"),
        known_hosts_file=tmp_path / "known_hosts-b2", runner=_runner(stdout="", returncode=1),
    )
    assert missing_helper["error_code"] == "remote_helper_rejected"
    transport = client.execute_helper(
        profile, contract.build_helper_request("status"),
        known_hosts_file=tmp_path / "known_hosts-b3", runner=_runner(stdout="", returncode=255),
    )
    assert transport["error_code"] == "transport_unavailable"
    inconsistent = client.execute_helper(
        profile, contract.build_helper_request("status"),
        known_hosts_file=tmp_path / "known_hosts-c", runner=_runner(stdout=_result("status"), returncode=1),
    )
    assert inconsistent["error_code"] == "remote_helper_rejected"


def test_client_rebuilds_requests_and_rejects_extra_or_unsafe_fields(profile, tmp_path):
    with pytest.raises(client.SshSyncClientError):
        client.normalize_request({"schema": contract.HELPER_SCHEMA, "action": "status", "command": "whoami"})
    with pytest.raises(client.SshSyncClientError):
        client.normalize_request({"schema": contract.HELPER_SCHEMA, "action": "fetch", "commit_sha": "a" * 40, "extra": True})
    with pytest.raises(client.SshSyncClientError):
        client.execute_helper(
            profile, contract.build_helper_request("status"),
            known_hosts_file=tmp_path / "known_hosts", timeout_seconds=0,
        )


def test_acceptance_sequence_is_explicit_and_apply_is_not_implicit():
    frame = {
        "schema": "qlh.patch_frame.v1", "branch": "dev", "commit_sha": "b" * 40,
        "signature": "opaque", "key_id": "patch-main",
    }
    requests = client.build_acceptance_requests(
        commit_sha="b" * 40, frame=frame, operation_id="c" * 64,
    )
    assert [request["action"] for request in requests] == ["status", "fetch", "apply", "verify"]
    assert requests[2]["maintenance_lock"]["operation_id"] == "c" * 64
