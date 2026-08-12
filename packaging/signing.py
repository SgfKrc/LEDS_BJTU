"""QLH release-manifest signing tooling (UP-N2 trusted publishing).

Implements the UP-N2 signing model from docs/安装包自动更新引导器方案.md §6:

1. The project release public key ships with the Launcher; private keys never
   enter the repository or a normal build environment.
2. The canonical manifest body is signed with Ed25519; the manifest carries
   ``key_id``, ``signature`` and ``signed_at``.
3. The Launcher verifies the signature first, and only then may install
   without ``--allow-unsigned``.
4. Authenticode may be layered on Windows; it protects the EXE, not the
   manifest, and never replaces this verifier.
5. Keys can rotate, but a new key must be authorized by the previous key or
   by the offline root key.

Trust model
-----------
``packaging/pubkeys/`` (shipped inside the Launcher bundle) contains:

- ``root.pub.json``        -- the offline root public key, trusted by itself.
- ``release-<id>.pub.json``-- one or more release keys.  Each carries an
  ``authorization`` signature created by its ``authorized_by`` key; the
  chain must be traceable back to the root key to be trusted.

Verification is fail-closed: any missing, unknown, malformed or mismatched
signature material leaves ``signature_verified=False`` with a reason in
``signature_error``.  No signature field is ever promoted to trust.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # cryptography is the only non-stdlib dependency of this module.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature
except Exception as exc:  # pragma: no cover - environment dependent
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    InvalidSignature = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


class SigningError(RuntimeError):
    """Expected signing/verification failure with a user-renderable message."""

    def __init__(self, message: str, *, code: str = "SIGNING_ERROR"):
        self.code = code
        super().__init__(message)


# --------------------------------------------------------------------------
# Canonical manifest body
# --------------------------------------------------------------------------

def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 manifest body used as the signing input.

    Keys are sorted and separators are compact, matching how serve.py
    serializes /latest.json so that signers and verifiers agree.
    """
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def manifest_body(mapping: Mapping[str, Any]) -> bytes:
    """The bytes that are signed: the manifest with ``signature`` removed.

    ``key_id`` and ``signed_at`` stay inside the body so they are protected
    by the signature as well.
    """
    body = {key: value for key, value in mapping.items() if key != "signature"}
    return canonical_json(body)


# --------------------------------------------------------------------------
# Key files (public keys are committed; private keys never are)
# --------------------------------------------------------------------------

def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64decode(value: str, *, what: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise SigningError(f"invalid base64 in {what}") from exc


def _public_key_to_file_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _private_key_to_file_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def generate_keypair(
    output_dir: str | os.PathLike[str],
    *,
    key_id: str,
    role: str = "release",
    created_at: str | None = None,
) -> Path:
    """Generate an Ed25519 keypair and write private + public key files.

    The private key is written with 0600 permissions (best effort on
    Windows) and must never be committed or copied into build machines.
    """
    if Ed25519PrivateKey is None:  # pragma: no cover - environment dependent
        raise SigningError(f"cryptography 不可用：{_CRYPTO_IMPORT_ERROR}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not key_id or any(ch in key_id for ch in "/\\"):
        raise SigningError(f"invalid key_id: {key_id!r}")
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    created = created_at or datetime.now(timezone.utc).isoformat()
    public_mapping: dict[str, Any] = {
        "key_id": key_id,
        "public_key": _b64encode(_public_key_to_file_bytes(public)),
        "role": role,
        "created_at": created,
    }
    if role == "release":
        public_mapping["authorized_by"] = ""
        public_mapping["authorization"] = ""
    private_path = directory / f"{key_id}.key"
    private_path.write_bytes(_private_key_to_file_bytes(private))
    try:
        os.chmod(private_path, 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    public_path = directory / f"{key_id}.pub.json"
    public_path.write_text(
        json.dumps(public_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return public_path


# --------------------------------------------------------------------------
# Key rotation authorization
# --------------------------------------------------------------------------

def authorization_body(public_mapping: Mapping[str, Any]) -> bytes:
    """The canonical bytes a new release key must be authorized over."""
    fields = {
        "key_id": public_mapping["key_id"],
        "public_key": public_mapping["public_key"],
        "role": public_mapping["role"],
        "created_at": public_mapping["created_at"],
        "authorized_by": public_mapping["authorized_by"],
    }
    return canonical_json(fields)


def load_private_key(path: str | os.PathLike[str]) -> tuple[str, Ed25519PrivateKey]:
    if Ed25519PrivateKey is None:  # pragma: no cover - environment dependent
        raise SigningError(f"cryptography 不可用：{_CRYPTO_IMPORT_ERROR}")
    key_path = Path(path)
    try:
        raw = key_path.read_bytes()
    except OSError as exc:
        raise SigningError(f"无法读取私钥文件: {key_path}: {exc}") from exc
    try:
        private = Ed25519PrivateKey.from_private_bytes(raw)
    except Exception as exc:
        raise SigningError(f"私钥文件不是 Ed25519 原始格式: {key_path}") from exc
    return key_path.stem, private


def load_public_key_file(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SigningError(f"无法解析公钥文件: {path}: {exc}") from exc
    if not isinstance(value, dict) or not value.get("key_id") or not value.get("public_key"):
        raise SigningError(f"公钥文件缺少 key_id/public_key: {path}")
    return value


def sign_authorization(
    public_mapping: Mapping[str, Any],
    *,
    issuer_private_key: Ed25519PrivateKey,
) -> str:
    """Sign a new release key's authorization with the issuer's private key."""
    return _b64encode(issuer_private_key.sign(authorization_body(public_mapping)))


def authorize_new_key(
    new_public_path: str | os.PathLike[str],
    *,
    issuer_private_path: str | os.PathLike[str],
    issuer_key_id: str,
) -> None:
    """Authorize ``new_public_path`` with the issuer key, in place.

    The new key's ``authorized_by`` becomes ``issuer_key_id`` and its
    ``authorization`` field is filled with the issuer's signature.  This is
    the only way a release key can become trusted after the root key.
    """
    public = load_public_key_file(new_public_path)
    public["authorized_by"] = issuer_key_id
    public["authorization"] = sign_authorization(
        public, issuer_private_key=load_private_key(issuer_private_path)[1],
    )
    Path(new_public_path).write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Manifest signing / verification
# --------------------------------------------------------------------------

def sign_manifest(
    mapping: Mapping[str, Any],
    *,
    private_key_path: str | os.PathLike[str],
    key_id: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``mapping`` with fresh key_id/signed_at/signature.

    A new signature never reuses a key_id or signed_at left over in the
    manifest: key_id comes from the explicit argument or the private key
    file stem, signed_at from the explicit argument or the current time.
    """
    resolved_key_id, private = load_private_key(private_key_path)
    result = dict(mapping)
    result["key_id"] = key_id or resolved_key_id
    result["signed_at"] = signed_at or datetime.now(timezone.utc).isoformat()
    result["signature"] = _b64encode(private.sign(manifest_body(result)))
    return result


def _parse_signed_at(value: Any) -> str:
    """Return the normalized signed_at or raise SigningError (fail-closed)."""
    if not isinstance(value, str) or not value.strip():
        raise SigningError("清单缺少 signed_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SigningError(f"signed_at 不是合法时间: {value!r}") from exc
    if parsed.tzinfo is None:
        raise SigningError(f"signed_at 缺少时区: {value!r}")
    return value


def _signature_result(
    verified: bool, error_code: str = "", reason: str = "",
) -> dict[str, Any]:
    return {
        "verified": verified,
        "error_code": error_code,
        "reason": reason,
    }


def verify_manifest_signature_details(
    mapping: Mapping[str, Any],
    *,
    trusted_keys_dir: str | os.PathLike[str] | None,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify a manifest and return stable code, localized reason and verdict.

    ``verify_manifest_signature`` remains the compatibility wrapper for
    callers that consume the historic ``(verified, reason)`` tuple.
    """
    if Ed25519PublicKey is None:  # pragma: no cover - environment dependent
        return _signature_result(
            False,
            "SIGNATURE_CRYPTO_UNAVAILABLE",
            f"cryptography 不可用：{_CRYPTO_IMPORT_ERROR}",
        )
    signature = mapping.get("signature")
    if not signature:
        return _signature_result(False, "SIGNATURE_MISSING", "清单没有签名")
    key_id = mapping.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        return _signature_result(False, "SIGNATURE_KEY_ID_MISSING", "清单缺少 key_id")
    try:
        _parse_signed_at(mapping.get("signed_at"))
    except SigningError as exc:
        return _signature_result(
            False,
            "SIGNATURE_SIGNED_AT_INVALID",
            str(exc),
        )
    if not trusted_keys_dir:
        return _signature_result(
            False,
            "SIGNATURE_TRUST_STORE_MISSING",
            "未配置可信公钥目录（Launcher 内置 pubkeys 缺失）",
        )
    try:
        public = _trusted_public_key(str(key_id), Path(trusted_keys_dir), now=now)
    except SigningError as exc:
        return _signature_result(False, exc.code, str(exc))
    try:
        raw_signature = _b64decode(str(signature), what="signature")
        public.verify(raw_signature, manifest_body(mapping))
    except InvalidSignature:
        return _signature_result(
            False, "SIGNATURE_INVALID", "清单签名校验失败",
        )
    except SigningError as exc:
        return _signature_result(False, exc.code, str(exc))
    except Exception as exc:
        return _signature_result(
            False, "SIGNATURE_CHECK_FAILED", f"签名校验异常: {exc}",
        )
    return _signature_result(True)


def verify_manifest_signature(
    mapping: Mapping[str, Any],
    *,
    trusted_keys_dir: str | os.PathLike[str] | None,
    now: float | None = None,
) -> tuple[bool, str]:
    """Compatibility wrapper returning the historic ``(verified, reason)``."""
    result = verify_manifest_signature_details(
        mapping, trusted_keys_dir=trusted_keys_dir, now=now,
    )
    return bool(result["verified"]), str(result["reason"])


def _trusted_public_key(
    key_id: str, keys_dir: Path, *, now: float | None = None,
) -> Ed25519PublicKey:
    """Load one release public key only if its authorization chain is valid.

    The chain starts at the offline root key (trusted by itself) and every
    release key must carry a signature from its ``authorized_by`` key.
    Unknown, revoked, expired or mis-authorized keys are rejected.
    """
    if Ed25519PublicKey is None:  # pragma: no cover - environment dependent
        raise SigningError("cryptography 不可用")
    now = now if now is not None else time.time()
    root_path = keys_dir / "root.pub.json"
    try:
        root = load_public_key_file(root_path)
    except SigningError:
        raise SigningError(
            "可信密钥目录缺少 root.pub.json",
            code="SIGNATURE_ROOT_KEY_MISSING",
        )
    trusted: dict[str, Ed25519PublicKey] = {}
    _add_trusted_key(root, "root", trusted, _parse_key_bytes(root), now=now)
    if key_id in trusted:
        return trusted[key_id]
    candidates: list[dict[str, Any]] = []
    for path in sorted(keys_dir.glob("release-*.pub.json")):
        try:
            candidates.append(load_public_key_file(path))
        except SigningError:
            continue
    # Resolve authorization chains in issuer-independent order: a release key
    # stays pending until its issuer is trusted, and is dropped forever once
    # its issuer is trusted but the authorization fails.
    failures: dict[str, SigningError] = {}
    while candidates:
        progressed = False
        remaining: list[dict[str, Any]] = []
        for candidate in candidates:
            issuer_id = str(candidate.get("authorized_by", ""))
            if issuer_id not in trusted:
                remaining.append(candidate)
                continue
            try:
                _add_trusted_key(
                    candidate, candidate["key_id"], trusted,
                    _parse_key_bytes(candidate), now=now,
                )
            except SigningError as exc:
                failures[candidate["key_id"]] = exc
                continue  # never trusted again
            progressed = True
        if key_id in trusted:
            return trusted[key_id]
        if not progressed:
            for candidate in remaining:
                if candidate.get("key_id") == key_id:
                    raise SigningError(
                        f"发布密钥授权者不受信任: {key_id} <- {candidate.get('authorized_by')}",
                        code="SIGNATURE_ISSUER_UNTRUSTED",
                    )
            break
        candidates = remaining
    if key_id in failures:
        raise failures[key_id]
    raise SigningError(
        f"未知发布密钥 key_id: {key_id}", code="SIGNATURE_KEY_UNKNOWN",
    )


def _parse_key_bytes(mapping: Mapping[str, Any]) -> Ed25519PublicKey:
    raw = _b64decode(str(mapping["public_key"]), what="public_key")
    if len(raw) != 32:
        raise SigningError("公钥长度非法")
    return Ed25519PublicKey.from_public_bytes(raw)


def _add_trusted_key(
    mapping: Mapping[str, Any],
    key_id: str,
    trusted: dict[str, Ed25519PublicKey],
    public: Ed25519PublicKey,
    *,
    now: float,
) -> None:
    """Register ``public`` under ``key_id`` after validating role and chain."""
    role = str(mapping.get("role", ""))
    valid_until = mapping.get("valid_until")
    if valid_until:
        try:
            deadline = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise SigningError(f"valid_until 不是合法时间: {valid_until!r}") from exc
        if now > deadline:
            raise SigningError(
                f"发布密钥已过期: {key_id}", code="SIGNATURE_KEY_EXPIRED",
            )
    if key_id == "root":
        if role != "root":
            raise SigningError("root.pub.json 的 role 必须为 root")
        trusted[key_id] = public
        return
    if role != "release":
        raise SigningError(f"未知密钥角色: {role}")
    issuer_id = mapping.get("authorized_by")
    if not isinstance(issuer_id, str) or not issuer_id:
        raise SigningError(f"发布密钥缺少授权者: {key_id}")
    if issuer_id not in trusted:
        raise SigningError(
            f"发布密钥授权者不受信任: {key_id} <- {issuer_id}",
            code="SIGNATURE_ISSUER_UNTRUSTED",
        )
    authorization = mapping.get("authorization")
    if not authorization:
        raise SigningError(
            f"发布密钥缺少授权签名: {key_id}",
            code="SIGNATURE_AUTHORIZATION_MISSING",
        )
    try:
        trusted[issuer_id].verify(
            _b64decode(str(authorization), what="authorization"),
            authorization_body(mapping),
        )
    except InvalidSignature:
        raise SigningError(
            f"发布密钥授权签名无效: {key_id}",
            code="SIGNATURE_AUTHORIZATION_INVALID",
        )
    except SigningError:
        raise
    except Exception as exc:
        raise SigningError(f"发布密钥授权校验异常: {key_id}: {exc}") from exc
    trusted[key_id] = public


def default_trusted_keys_dir() -> str | None:
    """Locate the bundled pubkeys directory (env override, bundle, source)."""
    override = os.environ.get("QLH_TRUSTED_KEYS_DIR", "").strip()
    if override:
        return override
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidate = bundled / "pubkeys"
        if candidate.is_dir():
            return str(candidate)
        return None
    candidate = Path(__file__).resolve().parent / "pubkeys"
    return str(candidate) if candidate.is_dir() else None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlh-signing",
        description="QLH release manifest signing tool (UP-N2).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser("keygen", help="generate an Ed25519 keypair")
    keygen.add_argument("--output-dir", required=True, help="directory for .key and .pub.json")
    keygen.add_argument("--key-id", required=True, help="key name, e.g. release-20260809")
    keygen.add_argument("--role", default="release", choices=("root", "release"))

    authorize = sub.add_parser("authorize", help="authorize a new release key with an issuer key")
    authorize.add_argument("--key", required=True, help="new release public key file (.pub.json)")
    authorize.add_argument("--issuer-key", required=True, help="issuer private key file (.key)")
    authorize.add_argument("--issuer-id", required=True, help="issuer key_id, usually the old release key or root")

    sign = sub.add_parser("sign", help="sign a manifest JSON file in place")
    sign.add_argument("--manifest", required=True, help="manifest JSON file (e.g. latest.json)")
    sign.add_argument("--key", required=True, help="release private key file (.key)")
    sign.add_argument("--key-id", help="override key_id (defaults to private key file stem)")

    verify = sub.add_parser("verify", help="verify a manifest signature (exit 0/1)")
    verify.add_argument("--manifest", required=True, help="manifest JSON file")
    verify.add_argument("--trusted-keys-dir", help="pubkeys directory (default: bundled)")
    return parser


def _cli(args: list[str] | None = None) -> int:
    parser = _build_parser()
    opts = parser.parse_args(args)
    try:
        if opts.command == "keygen":
            path = generate_keypair(opts.output_dir, key_id=opts.key_id, role=opts.role)
            print(f"生成密钥对：{path}（私钥 {Path(opts.output_dir) / (opts.key_id + '.key')} 请勿入库）")
            return 0
        if opts.command == "authorize":
            authorize_new_key(opts.key, issuer_private_path=opts.issuer_key, issuer_key_id=opts.issuer_id)
            print(f"已由 {opts.issuer_id} 授权：{opts.key}")
            return 0
        if opts.command == "sign":
            path = Path(opts.manifest)
            value = json.loads(path.read_text(encoding="utf-8"))
            signed = sign_manifest(value, private_key_path=opts.key, key_id=opts.key_id)
            path.write_text(
                json.dumps(signed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"已签名：{path}（key_id={signed['key_id']}）")
            return 0
        if opts.command == "verify":
            value = json.loads(Path(opts.manifest).read_text(encoding="utf-8"))
            trusted = opts.trusted_keys_dir or default_trusted_keys_dir()
            verified, reason = verify_manifest_signature(value, trusted_keys_dir=trusted)
            if verified:
                print("签名有效")
                return 0
            print(f"签名无效：{reason}")
            return 1
    except (SigningError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
