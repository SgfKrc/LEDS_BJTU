"""SYNC-SSH-S0 contract tests.  No SSH daemon or remote checkout is touched."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ssh_sync_contract as contract  # noqa: E402


def _openssh_ed25519(raw: bytes) -> str:
    wire = (
        len(b"ssh-ed25519").to_bytes(4, "big") + b"ssh-ed25519" +
        len(raw).to_bytes(4, "big") + raw
    )
    return "ssh-ed25519 " + base64.b64encode(wire).decode("ascii")


@pytest.fixture
def profile_mapping(tmp_path):
    host_key = _openssh_ed25519(bytes(range(32)))
    identity_key = _openssh_ed25519(bytes(range(32, 64)))
    return {
        "schema": contract.PROFILE_SCHEMA,
        "host": "[fd7a:115c:a1e0::643b:346b]",
        "port": 2202,
        "user": "qlh_sync",
        "host_public_key": host_key,
        "host_fingerprint": contract.host_fingerprint(host_key),
        "identity_file": str(tmp_path / "qlh_sync_ed25519"),
        "identity_public_key": identity_key,
    }


def _frame():
    return {
        "schema": "qlh.patch_frame.v1",
        "commit_sha": "a" * 40,
        "branch": "dev",
        "signature": "opaque-but-reverified-remotely",
        "key_id": "patch-main",
    }


def test_profile_requires_matching_ed25519_host_key_and_fingerprint(profile_mapping):
    profile = contract.load_profile(profile_mapping)
    assert profile.host == "fd7a:115c:a1e0::643b:346b"
    assert profile.authority == "[fd7a:115c:a1e0::643b:346b]:2202"
    assert profile.known_hosts_entry().startswith("[fd7a:115c:a1e0::643b:346b]:2202 ssh-ed25519 ")

    profile_mapping["host_fingerprint"] = "SHA256:" + "A" * 43
    with pytest.raises(contract.SshSyncContractError, match="does not match") as exc:
        contract.load_profile(profile_mapping)
    assert exc.value.code == "host_fingerprint_mismatch"


@pytest.mark.parametrize("field", ["password", "proxy_command", "auto_accept_host_key"])
def test_profile_rejects_password_auto_trust_and_remote_command(profile_mapping, field):
    profile_mapping[field] = "unsafe"
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.load_profile(profile_mapping)
    assert exc.value.code == "password_auth_forbidden"


def test_ssh_argv_uses_only_strict_key_auth_and_fixed_helper(profile_mapping, tmp_path):
    profile = contract.load_profile(profile_mapping)
    request = contract.build_helper_request("verify", {"commit_sha": "b" * 40})
    argv = contract.build_ssh_argv(profile, request, known_hosts_file=tmp_path / "known_hosts")
    rendered = "\0".join(argv)
    assert argv[0] == "ssh"
    assert "PasswordAuthentication=no" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "qlh-patch-helper" in argv
    assert "--commit-sha" in argv
    assert "password" not in rendered.lower().replace("passwordauthentication=no", "")
    assert "shell" not in rendered.lower()


def test_helper_rejects_arbitrary_actions_and_unsafe_commit():
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.build_helper_request("sh", {"command": "whoami"})
    assert exc.value.code == "helper_action_invalid"
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.build_helper_request("fetch", {"commit_sha": "dev;whoami"})
    assert exc.value.code == "commit_sha_invalid"


def test_ssh_argv_rejects_an_empty_known_hosts_path(profile_mapping):
    profile = contract.load_profile(profile_mapping)
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.build_ssh_argv(profile, contract.build_helper_request("status"), known_hosts_file="")
    assert exc.value.code == "known_hosts_invalid"


def test_apply_requires_remote_owned_maintenance_lock_and_forbids_restart(profile_mapping, tmp_path):
    operation_id = "c" * 64
    request = contract.build_helper_request("apply", {
        "frame": _frame(), "operation_id": operation_id,
    })
    assert request["maintenance_lock"] == {
        "schema": contract.LOCK_SCHEMA,
        "operation_id": operation_id,
        "scope": "managed_checkout",
        "owner": "remote_helper",
    }
    argv = contract.build_ssh_argv(
        contract.load_profile(profile_mapping), request,
        known_hosts_file=tmp_path / "known_hosts",
    )
    assert argv[-2:] == ["--maintenance-operation", operation_id]
    unsafe = _frame()
    unsafe["restart_requested"] = True
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.build_helper_request("apply", {"frame": unsafe, "operation_id": operation_id})
    assert exc.value.code == "restart_forbidden"


def test_fixture_is_stable_and_contains_no_private_material():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "ssh_sync_contract_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture == contract.protocol_fixture()
    serialized = json.dumps(fixture).lower()
    assert "private" not in serialized
    assert "identity_file" not in serialized


def test_helper_result_freezes_retry_and_pull_push_fallback():
    result = contract.validate_helper_result({
        "schema": contract.HELPER_RESULT_SCHEMA,
        "action": "fetch",
        "status": "error",
        "error_code": "transport_unavailable",
        "retryable": True,
        "next_action": "fallback_pull_push",
    }, expected_action="fetch")
    assert result["next_action"] == "fallback_pull_push"


def test_helper_result_rejects_forged_retry_semantics_and_details():
    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.validate_helper_result({
            "schema": contract.HELPER_RESULT_SCHEMA,
            "action": "apply",
            "status": "error",
            "error_code": "patch_signature_invalid",
            "retryable": True,
            "next_action": "retry_later",
        }, expected_action="apply")
    assert exc.value.code == "helper_result_invalid"

    with pytest.raises(contract.SshSyncContractError) as exc:
        contract.validate_helper_result({
            "schema": contract.HELPER_RESULT_SCHEMA,
            "action": "status",
            "status": "ok",
            "detail": "C:/Users/private/checkout",
        }, expected_action="status")
    assert exc.value.code == "helper_result_invalid"
