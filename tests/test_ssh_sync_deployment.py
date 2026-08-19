"""Local-only tests for the SYNC-SSH-S2 restricted helper provisioner."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ssh_sync_deployment as deploy  # noqa: E402


def _openssh_ed25519(raw: bytes) -> str:
    wire = (
        len(b"ssh-ed25519").to_bytes(4, "big") + b"ssh-ed25519" +
        len(raw).to_bytes(4, "big") + raw
    )
    return "ssh-ed25519 " + base64.b64encode(wire).decode("ascii") + " qlh-sync-s2\n"


@pytest.fixture
def topology(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    tools = repo / "tools"
    tools.mkdir()
    (tools / "ssh_sync_helper.py").write_text("# helper\n", encoding="utf-8")
    verify = repo / "release.pub.json"
    verify.write_text(json.dumps({"key_id": "release", "public_key": base64.b64encode(bytes(range(32))).decode("ascii")}), encoding="utf-8")
    client = tmp_path / "client.pub"
    client.write_text(_openssh_ed25519(bytes(range(32, 64))), encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_text("", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    return {"repo": repo, "verify": verify, "client": client, "python": python, "home": home, "state": tmp_path / "state"}


def _plan(topology):
    plan = deploy.build_install_plan(
        repo=topology["repo"], state_dir=topology["state"], verify_key=topology["verify"],
        client_public_key=topology["client"], home=topology["home"], python_executable=topology["python"],
    )
    plan["client_public_key_path"] = str(topology["client"])
    return plan


def test_plan_is_external_and_apply_writes_only_deployment_owned_files(topology):
    plan = _plan(topology)
    assert Path(plan["state_dir"]).is_relative_to(topology["repo"]) is False
    assert plan["key_rule_present"] is False
    result = deploy.install(plan)
    assert result["status"] == "installed"
    config = json.loads(Path(plan["config"]).read_text(encoding="utf-8"))
    assert config["repo"] == str(topology["repo"].resolve())
    assert Path(plan["trust_key"]).is_file()
    wrapper = Path(plan["wrapper"]).read_text(encoding="utf-8")
    assert "--forced-command" in wrapper
    keys = Path(plan["authorized_keys"]).read_text(encoding="utf-8")
    assert deploy.MANAGED_MARKER in keys
    assert "restrict,command=" in keys
    assert "qlh-sync-s2" in keys
    assert not (topology["repo"] / "helper-config.json").exists()


def test_install_is_idempotent_but_refuses_to_take_over_normal_key(topology):
    plan = _plan(topology)
    assert deploy.install(plan)["authorized_key_added"] is True
    assert deploy.install(plan)["authorized_key_added"] is False
    other = topology["home"] / ".ssh" / "authorized_keys"
    other.write_text(_openssh_ed25519(bytes(range(32, 64))), encoding="utf-8")
    with pytest.raises(deploy.DeploymentError, match="already authorized"):
        _plan(topology)


def test_plan_rejects_state_inside_checkout_or_invalid_public_material(topology):
    with pytest.raises(deploy.DeploymentError, match="outside"):
        deploy.build_install_plan(
            repo=topology["repo"], state_dir=topology["repo"] / "state", verify_key=topology["verify"],
            client_public_key=topology["client"], home=topology["home"], python_executable=topology["python"],
        )
    topology["client"].write_text("ssh-rsa not-ed25519\n", encoding="utf-8")
    with pytest.raises(deploy.DeploymentError, match="Ed25519"):
        _plan(topology)


def test_write_profile_uses_only_exact_ed25519_known_host(topology, tmp_path):
    identity = tmp_path / "identity"
    identity.write_text("private placeholder", encoding="utf-8")
    identity.with_suffix(".pub").write_text(_openssh_ed25519(bytes(range(32, 64))), encoding="utf-8")
    remote = _openssh_ed25519(bytes(range(64, 96))).strip()
    known = tmp_path / "known_hosts"
    known.write_text("100.100.52.106 " + remote + "\n", encoding="utf-8")
    profile = deploy.write_profile(
        output=tmp_path / "profile.json", host="100.100.52.106", user="surface",
        identity_file=identity, known_hosts=known,
    )
    assert profile["host_fingerprint"].startswith("SHA256:")
    assert profile["identity_file"] == str(identity.resolve())
