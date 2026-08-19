#!/usr/bin/env python3
"""Local provisioning for the restricted SYNC-SSH-S2 helper.

Run this script on the machine that hosts the managed checkout.  It creates
only deployment-owned files outside that checkout and appends one *new*,
dedicated Ed25519 key rule to ``authorized_keys``.  The existing user's normal
SSH key is never rewritten.  ``install`` is a dry plan unless ``--apply`` is
provided explicitly.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ssh_sync_contract import PROFILE_SCHEMA, host_fingerprint


DEPLOYMENT_SCHEMA = "qlh.sync_ssh.deployment.v1"
DEFAULT_STATE_DIR_NAME = "QLH/ssh-sync"
MANAGED_MARKER = "# QLH-SYNC-SSH-S2"


class DeploymentError(ValueError):
    """A local deployment input or state check failed without SSH side effects."""


def _resolved(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _require_external(path: Path, repo: Path, *, label: str) -> None:
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise DeploymentError(f"{label} must be outside the managed checkout")


def _atomic_write(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if mode is not None and os.name != "nt":
            temporary.chmod(mode)
        os.replace(temporary, path)
        if mode is not None and os.name != "nt":
            path.chmod(mode)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DeploymentError(f"cannot write {path.name}") from exc


def _read_ed25519_public_key(path: str | os.PathLike[str]) -> tuple[str, str]:
    try:
        line = _resolved(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DeploymentError("client public key cannot be read") from exc
    fields = line.split()
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise DeploymentError("client public key must be an OpenSSH Ed25519 key")
    try:
        wire = base64.b64decode(fields[1].encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise DeploymentError("client public key base64 is invalid") from exc
    if len(wire) != 51 or wire[:4] != (11).to_bytes(4, "big") or wire[4:15] != b"ssh-ed25519":
        raise DeploymentError("client public key is not a complete Ed25519 key")
    if wire[15:19] != (32).to_bytes(4, "big") or len(wire[19:]) != 32:
        raise DeploymentError("client public key is not a complete Ed25519 key")
    return f"ssh-ed25519 {fields[1]}", fields[1]


def _validate_verify_key(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        encoded = raw["public_key"]
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (OSError, KeyError, TypeError, UnicodeEncodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise DeploymentError("patch verification public key is invalid") from exc
    if not isinstance(raw.get("key_id"), str) or not raw["key_id"].strip() or len(decoded) != 32:
        raise DeploymentError("patch verification public key is invalid")
    return {"key_id": raw["key_id"], "public_key": encoded}


def build_authorized_key_line(*, public_key: str, wrapper_path: Path) -> str:
    """Render a restrictive per-key OpenSSH rule with a no-space command path."""
    _normalized, encoded = _read_ed25519_public_key_from_value(public_key)
    command = str(wrapper_path)
    if '"' in command or any(character in command for character in "\r\n\x00"):
        raise DeploymentError("helper wrapper path is unsafe")
    # ``restrict`` disables forwarding and PTY; the command receives the
    # original client command only via SSH_ORIGINAL_COMMAND inside the wrapper.
    return f'restrict,command="{command}" ssh-ed25519 {encoded} qlh-sync-s2\n'


def _read_ed25519_public_key_from_value(value: str) -> tuple[str, str]:
    fields = value.strip().split()
    if len(fields) < 2:
        raise DeploymentError("client public key is invalid")
    temporary = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    try:
        temporary.write(value)
        temporary.close()
        return _read_ed25519_public_key(temporary.name)
    finally:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)


def _find_matching_key_line(content: str, encoded_key: str) -> str | None:
    needle = f"ssh-ed25519 {encoded_key}"
    return next((line for line in content.splitlines() if needle in line), None)


def _append_restricted_key(*, authorized_keys: Path, public_key: str, wrapper_path: Path) -> bool:
    _normalized, encoded_key = _read_ed25519_public_key_from_value(public_key)
    line = build_authorized_key_line(public_key=public_key, wrapper_path=wrapper_path)
    current = authorized_keys.read_text(encoding="utf-8") if authorized_keys.exists() else ""
    existing = _find_matching_key_line(current, encoded_key)
    if existing == line.rstrip("\n"):
        return False
    if existing is not None:
        raise DeploymentError("refuse to restrict an existing non-dedicated SSH key")
    suffix = "" if not current or current.endswith("\n") else "\n"
    managed = f"{MANAGED_MARKER} schema={DEPLOYMENT_SCHEMA}\n{line}"
    _atomic_write(authorized_keys, current + suffix + managed, mode=0o600)
    return True


def _wrapper_content(*, python_executable: Path, helper_path: Path, config_path: Path) -> str:
    values = (python_executable, helper_path, config_path)
    if any('"' in str(value) or any(character in str(value) for character in "\r\n\x00") for value in values):
        raise DeploymentError("deployment path is unsafe")
    return (
        "@echo off\n"
        "setlocal EnableExtensions DisableDelayedExpansion\n"
        f"set \"QLH_SSH_SYNC_HELPER_CONFIG={config_path}\"\n"
        f"\"{python_executable}\" \"{helper_path}\" --forced-command\n"
        "exit /b %ERRORLEVEL%\n"
    )


def build_install_plan(
    *, repo: str | os.PathLike[str], state_dir: str | os.PathLike[str], verify_key: str | os.PathLike[str],
    client_public_key: str | os.PathLike[str], home: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate inputs and return paths/content facts without modifying them."""
    checkout = _resolved(repo)
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise DeploymentError("repo must be a managed git checkout")
    helper = checkout / "tools" / "ssh_sync_helper.py"
    if not helper.is_file():
        raise DeploymentError("managed checkout does not contain ssh_sync_helper.py")
    state_dir_path = _resolved(state_dir)
    _require_external(state_dir_path, checkout, label="state directory")
    source_verify_key = _resolved(verify_key)
    _validate_verify_key(source_verify_key)
    public_key, encoded = _read_ed25519_public_key(client_public_key)
    home_dir = _resolved(home or Path.home())
    authorized_keys = home_dir / ".ssh" / "authorized_keys"
    executable = _resolved(python_executable or sys.executable)
    if not executable.is_file():
        raise DeploymentError("Python executable is unavailable")
    config = state_dir_path / "helper-config.json"
    trust = state_dir_path / "trust" / source_verify_key.name
    wrapper = state_dir_path / "qlh-patch-helper.cmd"
    config_value = {
        "schema": "qlh.sync_ssh.helper-config.v1",
        "repo": str(checkout),
        "state_dir": str(state_dir_path),
        "verify_key": str(trust),
    }
    existing = authorized_keys.read_text(encoding="utf-8") if authorized_keys.exists() else ""
    matching = _find_matching_key_line(existing, encoded)
    desired = build_authorized_key_line(public_key=public_key, wrapper_path=wrapper).rstrip("\n")
    if matching is not None and matching != desired:
        raise DeploymentError("client public key is already authorized without this restricted helper rule")
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "repo": str(checkout),
        "state_dir": str(state_dir_path),
        "source_verify_key": str(source_verify_key),
        "trust_key": str(trust),
        "config": str(config),
        "wrapper": str(wrapper),
        "authorized_keys": str(authorized_keys),
        "client_public_key_path": str(_resolved(client_public_key)),
        "client_key_fingerprint": host_fingerprint(public_key),
        "key_rule_present": matching == desired,
        "python_executable": str(executable),
        "helper": str(helper),
        "config_value": config_value,
    }


def install(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a previously validated plan.  Repository files are never written."""
    state_dir = _resolved(str(plan["state_dir"]))
    trust = _resolved(str(plan["trust_key"]))
    source_key = _resolved(str(plan["source_verify_key"]))
    config = _resolved(str(plan["config"]))
    wrapper = _resolved(str(plan["wrapper"]))
    authorized_keys = _resolved(str(plan["authorized_keys"]))
    helper = _resolved(str(plan["helper"]))
    executable = _resolved(str(plan["python_executable"]))
    repo = _resolved(str(plan["repo"]))
    _require_external(state_dir, repo, label="state directory")
    _atomic_write(trust, source_key.read_text(encoding="utf-8"), mode=0o600)
    _atomic_write(config, json.dumps(plan["config_value"], ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", mode=0o600)
    _atomic_write(wrapper, _wrapper_content(python_executable=executable, helper_path=helper, config_path=config), mode=0o700)
    public_key_path = Path(str(plan.get("client_public_key_path", "")))
    if not public_key_path.is_file():
        raise DeploymentError("client public key source changed before install")
    public_key, _encoded = _read_ed25519_public_key(public_key_path)
    changed = _append_restricted_key(authorized_keys=authorized_keys, public_key=public_key, wrapper_path=wrapper)
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "status": "installed",
        "authorized_key_added": changed,
        "wrapper": str(wrapper),
        "config": str(config),
    }


def create_client_identity(identity_file: str | os.PathLike[str], *, apply: bool) -> dict[str, Any]:
    """Create a dedicated, noninteractive Ed25519 identity outside a checkout."""
    identity = _resolved(identity_file)
    public = Path(str(identity) + ".pub")
    if identity.exists() or public.exists():
        raise DeploymentError("refuse to overwrite an existing SSH identity")
    binary = shutil.which("ssh-keygen")
    if binary is None:
        raise DeploymentError("ssh-keygen is unavailable")
    plan = {
        "schema": DEPLOYMENT_SCHEMA,
        "identity_file": str(identity),
        "public_key": str(public),
        "algorithm": "ed25519",
        "passphrase": "none-required-for-BatchMode",
    }
    if not apply:
        return {"status": "planned", **plan}
    identity.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [binary, "-q", "-t", "ed25519", "-N", "", "-f", str(identity), "-C", "qlh-sync-s2"],
        shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
    )
    if result.returncode != 0 or not identity.is_file() or not public.is_file():
        raise DeploymentError("ssh-keygen could not create the client identity")
    if os.name != "nt":
        identity.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _read_ed25519_public_key(public)
    return {"status": "created", **plan}


def write_profile(
    *, output: str | os.PathLike[str], host: str, user: str, identity_file: str | os.PathLike[str],
    known_hosts: str | os.PathLike[str], port: int = 22,
) -> dict[str, Any]:
    """Create a strict S0 profile from a known-hosts Ed25519 entry."""
    identity = _resolved(identity_file)
    public, _encoded = _read_ed25519_public_key(Path(str(identity) + ".pub"))
    known_path = _resolved(known_hosts)
    host_token = host if port == 22 else f"[{host}]:{port}"
    matches = []
    for line in known_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == host_token and fields[1] == "ssh-ed25519":
            matches.append(" ".join(fields[1:3]))
    if len(matches) != 1:
        raise DeploymentError("known_hosts must have exactly one Ed25519 entry for this host")
    remote_key = matches[0]
    profile = {
        "schema": PROFILE_SCHEMA,
        "host": host,
        "port": port,
        "user": user,
        "host_public_key": remote_key,
        "host_fingerprint": host_fingerprint(remote_key),
        "identity_file": str(identity),
        "identity_public_key": public,
    }
    _atomic_write(_resolved(output), json.dumps(profile, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", mode=0o600)
    return profile


def _plan_for_cli(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_install_plan(
        repo=args.repo, state_dir=args.state_dir, verify_key=args.verify_key,
        client_public_key=args.client_public_key, home=args.home, python_executable=args.python_executable,
    )
    plan["client_public_key_path"] = str(_resolved(args.client_public_key))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLH restricted SSH helper provisioning")
    actions = parser.add_subparsers(dest="action", required=True)
    identity = actions.add_parser("create-client-identity")
    identity.add_argument("--identity-file", required=True)
    identity.add_argument("--apply", action="store_true")
    install_parser = actions.add_parser("install")
    install_parser.add_argument("--repo", required=True)
    install_parser.add_argument("--state-dir", required=True)
    install_parser.add_argument("--verify-key", required=True)
    install_parser.add_argument("--client-public-key", required=True)
    install_parser.add_argument("--home", default="")
    install_parser.add_argument("--python-executable", default="")
    install_parser.add_argument("--apply", action="store_true")
    profile = actions.add_parser("write-profile")
    profile.add_argument("--output", required=True)
    profile.add_argument("--host", required=True)
    profile.add_argument("--user", required=True)
    profile.add_argument("--identity-file", required=True)
    profile.add_argument("--known-hosts", required=True)
    profile.add_argument("--port", type=int, default=22)
    args = parser.parse_args(argv)
    try:
        if args.action == "create-client-identity":
            result = create_client_identity(args.identity_file, apply=args.apply)
        elif args.action == "write-profile":
            result = write_profile(
                output=args.output, host=args.host, user=args.user, identity_file=args.identity_file,
                known_hosts=args.known_hosts, port=args.port,
            )
        else:
            plan = _plan_for_cli(args)
            result = install(plan) if args.apply else {"status": "planned", **plan}
        print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0
    except (DeploymentError, OSError, ValueError) as exc:
        print(json.dumps({"schema": DEPLOYMENT_SCHEMA, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
