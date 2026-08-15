"""Isolated, offline tokenizer worker for a local MM1 token ledger."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping


TOOL = "qwen3_token_ledger"
SCHEMA_VERSION = 1
MAX_LEDGER_BYTES = 256 * 1024
MAX_LEDGER_TOKENS = 4096
MAX_TOKEN_ID = 2**31 - 1
MAX_TEXT_BYTES = 64 * 1024
MAX_SAMPLING_TOP_K = 4096
MAX_SAMPLING_SEED = 2**63 - 1
MAX_SAMPLING_TEMPERATURE = 2.0
MAX_POLICY_TTL_SECONDS = 7 * 24 * 60 * 60


def _base_result(
    status: str, *, gate_passed: bool = False,
    errors: list[dict[str, str]] | None = None,
    operation: str = "qwen3_token_ledger_decode",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": operation,
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": gate_passed,
        "status": status,
        "errors": errors or [],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_dir(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser().absolute().resolve(strict=False)
    return path if path.is_dir() else None


def _safe_file(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser().absolute().resolve(strict=False)
    return path if path.is_file() else None


def _sampling(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "mode", "temperature", "top_k", "top_p", "seed",
    }:
        raise ValueError("sampling fields are invalid")
    temperature = value["temperature"]
    top_k = value["top_k"]
    top_p = value["top_p"]
    seed = value["seed"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= MAX_SAMPLING_TEMPERATURE
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 0 <= top_k <= MAX_SAMPLING_TOP_K
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0.0 < float(top_p) <= 1.0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SAMPLING_SEED
    ):
        raise ValueError("sampling values are invalid")
    result = {
        "mode": "greedy" if float(temperature) == 0.0 else "multinomial",
        "temperature": float(temperature),
        "top_k": top_k,
        "top_p": float(top_p),
        "seed": seed,
    }
    if value["mode"] != result["mode"]:
        raise ValueError("sampling mode is invalid")
    return result


def _policy(value: Any, *, sampling: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "policy_kind", "policy_id", "policy_version", "issued_at",
        "expires_at", "replay_allowed", "route_scope", "sampling", "sampling_sha256",
        "snapshot_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("policy snapshot fields are invalid")
    unsigned = dict(value)
    snapshot_sha256 = str(unsigned.pop("snapshot_sha256") or "").lower()
    if (
        value["schema_version"] != 1
        or value["policy_kind"] != "qwen3_mm1_sampling_policy_snapshot"
        or value["route_scope"] != "mm1-bounded-decode"
        or not isinstance(value["replay_allowed"], bool)
        or not isinstance(value["policy_id"], str)
        or not isinstance(value["policy_version"], str)
        or not isinstance(value["issued_at"], int)
        or not isinstance(value["expires_at"], int)
        or len(snapshot_sha256) != 64
        or any(char not in "0123456789abcdef" for char in snapshot_sha256)
        or snapshot_sha256 != _digest(unsigned)
        or value["sampling"] != sampling
        or value["sampling_sha256"] != _digest(sampling)
    ):
        raise ValueError("policy snapshot identity does not match")
    try:
        issued = int(value["issued_at"])
        expires = int(value["expires_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("policy snapshot timestamps are invalid") from exc
    if (
        isinstance(value["issued_at"], bool)
        or isinstance(value["expires_at"], bool)
        or expires <= issued
        or expires - issued > MAX_POLICY_TTL_SECONDS
        or int(time.time()) < issued - 300
        or int(time.time()) >= expires
    ):
        raise ValueError("policy snapshot is expired or outside its lifetime")
    return {
        "policy_id": value["policy_id"],
        "policy_version": value["policy_version"],
        "snapshot_sha256": snapshot_sha256,
        "sampling_sha256": value["sampling_sha256"],
        "issued_at": issued,
        "expires_at": expires,
        "replay_allowed": value["replay_allowed"],
        "route_scope": value["route_scope"],
    }


def _load_ledger(path: Path, metadata: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    if path.stat().st_size != metadata["size_bytes"] or _sha256(path) != metadata["sha256"]:
        raise ValueError("ledger file evidence does not match metadata")
    if path.stat().st_size > MAX_LEDGER_BYTES:
        raise ValueError("ledger exceeds bounded size")
    with path.open("r", encoding="utf-8") as handle:
        ledger = json.load(handle)
    required = {
        "schema_version", "ledger_kind", "ledger_id", "text_chain_id", "contract_sha256",
        "prefill_generation", "token_count", "stop_reason", "records",
        "full_model_materialized", "ledger_sha256",
    }
    sampling_fields = {"sampling", "sampling_sha256", "draw_evidence"}
    quality_fields = {"quality_summary"}
    policy_fields = {"policy_snapshot", "policy_snapshot_sha256"}
    allowed_fields = {
        frozenset(required),
        frozenset(required | sampling_fields),
        frozenset(required | sampling_fields | quality_fields),
        frozenset(required | sampling_fields | policy_fields),
        frozenset(required | sampling_fields | quality_fields | policy_fields),
    }
    if not isinstance(ledger, dict) or frozenset(ledger) not in allowed_fields:
        raise ValueError("ledger fields do not match the schema")
    unsigned = dict(ledger)
    ledger_digest = str(unsigned.pop("ledger_sha256") or "")
    if len(ledger_digest) != 64 or ledger_digest != _digest(unsigned):
        raise ValueError("ledger digest does not match")
    if (
        ledger["schema_version"] != 1
        or ledger["ledger_kind"] != "qwen3_mm1_generated_token_ledger"
        or ledger["ledger_id"] != metadata["ledger_id"]
        or ledger["token_count"] != metadata["token_count"]
        or ledger["stop_reason"] != metadata["stop_reason"]
        or ledger["full_model_materialized"] is not False
    ):
        raise ValueError("ledger identity does not match metadata")
    expected_chain = str(request.get("expected_chain_id") or "")
    if expected_chain and ledger["text_chain_id"] != expected_chain:
        raise ValueError("ledger chain identity does not match")
    try:
        generation = int(request.get("expected_generation", -1))
        first_sequence = int(request.get("expected_first_sequence", -1))
        prefill_generation = int(ledger["prefill_generation"])
        token_count = int(ledger["token_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ledger dimensions are invalid") from exc
    if (
        token_count <= 0
        or token_count > MAX_LEDGER_TOKENS
        or generation != prefill_generation
        or not isinstance(ledger["records"], list)
        or len(ledger["records"]) != token_count
        or first_sequence <= 0
    ):
        raise ValueError("ledger generation or record count is invalid")
    sampling = None
    draws = None
    if sampling_fields.issubset(ledger):
        sampling = _sampling(ledger["sampling"])
        sampling_sha256 = str(ledger["sampling_sha256"] or "").lower()
        if sampling_sha256 != _digest(sampling):
            raise ValueError("sampling digest does not match")
        if (
            metadata.get("sampling_sha256") != sampling_sha256
            or metadata.get("draw_count") != token_count
            or not isinstance(ledger["draw_evidence"], list)
            or len(ledger["draw_evidence"]) != token_count
        ):
            raise ValueError("sampling metadata does not match")
        expected_sampling = str(request.get("expected_sampling_sha256") or "")
        if expected_sampling and expected_sampling != sampling_sha256:
            raise ValueError("sampling digest does not match replay expectation")
        draws = ledger["draw_evidence"]
    quality = None
    if "quality_summary" in ledger:
        quality = ledger["quality_summary"]
        if not isinstance(quality, dict) or set(quality) != {
            "schema_version", "step_count", "candidate_count_min", "candidate_count_max",
            "entropy_min", "entropy_max", "confidence_min", "confidence_max",
            "top_p_cutoff_min", "top_p_cutoff_max", "sampling_sha256", "sha256",
        }:
            raise ValueError("sampling quality summary fields are invalid")
        try:
            quality_step_count = int(quality["step_count"])
            candidate_min = int(quality["candidate_count_min"])
            candidate_max = int(quality["candidate_count_max"])
            cutoff_min = int(quality["top_p_cutoff_min"])
            cutoff_max = int(quality["top_p_cutoff_max"])
            entropy_min = float(quality["entropy_min"])
            entropy_max = float(quality["entropy_max"])
            confidence_min = float(quality["confidence_min"])
            confidence_max = float(quality["confidence_max"])
        except (TypeError, ValueError) as exc:
            raise ValueError("sampling quality summary values are invalid") from exc
        quality_sha256 = str(quality["sha256"] or "").lower()
        if (
            quality["schema_version"] != 1
            or quality_step_count != token_count
            or quality["sampling_sha256"] != ledger["sampling_sha256"]
            or metadata.get("quality_sha256") != quality_sha256
            or len(quality_sha256) != 64
            or any(char not in "0123456789abcdef" for char in quality_sha256)
            or candidate_min <= 0
            or candidate_min > candidate_max
            or cutoff_min <= 0
            or cutoff_min > cutoff_max
            or cutoff_max > candidate_max
            or not math.isfinite(entropy_min)
            or not math.isfinite(entropy_max)
            or entropy_min < 0.0
            or entropy_min > entropy_max
            or not math.isfinite(confidence_min)
            or not math.isfinite(confidence_max)
            or not 0.0 <= confidence_min <= confidence_max <= 1.0
        ):
            raise ValueError("sampling quality summary does not match")
        expected_quality = str(request.get("expected_quality_sha256") or "")
        if expected_quality and expected_quality != quality_sha256:
            raise ValueError("quality digest does not match replay expectation")
    expected_sampling = str(request.get("expected_sampling_sha256") or "")
    if expected_sampling and sampling is None:
        raise ValueError("replay expects sampler evidence")
    expected_quality = str(request.get("expected_quality_sha256") or "")
    if expected_quality and quality is None:
        raise ValueError("replay expects quality evidence")
    policy = None
    if policy_fields.issubset(ledger):
        if sampling is None:
            raise ValueError("policy snapshot requires sampler evidence")
        policy = _policy(ledger["policy_snapshot"], sampling=sampling)
        policy_sha256 = str(ledger["policy_snapshot_sha256"] or "").lower()
        if (
            policy_sha256 != policy["snapshot_sha256"]
            or metadata.get("policy_snapshot_sha256") != policy_sha256
            or metadata.get("policy_id") != policy["policy_id"]
            or metadata.get("policy_version") != policy["policy_version"]
        ):
            raise ValueError("policy snapshot metadata does not match")
        expected_policy = str(request.get("expected_policy_snapshot_sha256") or "")
        if expected_policy and expected_policy != policy_sha256:
            raise ValueError("policy snapshot digest does not match replay expectation")
        if (
            request.get("operation") == "qwen3_token_ledger_replay"
            and policy["replay_allowed"] is not True
        ):
            raise ValueError("policy snapshot does not allow replay")
    expected_policy = str(request.get("expected_policy_snapshot_sha256") or "")
    if expected_policy and policy is None:
        raise ValueError("replay expects policy snapshot evidence")
    token_ids: list[int] = []
    for index, record in enumerate(ledger["records"], start=1):
        if not isinstance(record, dict) or set(record) != {
            "step_index", "generation", "sequence_length", "token_id", "artifact_sha256",
        }:
            raise ValueError("ledger record fields are invalid")
        artifact_sha = str(record["artifact_sha256"] or "").lower()
        try:
            step = int(record["step_index"])
            record_generation = int(record["generation"])
            sequence = int(record["sequence_length"])
            token_id = int(record["token_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("ledger record dimensions are invalid") from exc
        if (
            step != index
            or record_generation != generation + index
            or sequence != first_sequence + index - 1
            or token_id < 0
            or token_id > MAX_TOKEN_ID
            or len(artifact_sha) != 64
            or any(char not in "0123456789abcdef" for char in artifact_sha)
        ):
            raise ValueError("ledger record sequence is invalid")
        if draws is not None and sampling is not None:
            draw = draws[index - 1]
            if not isinstance(draw, dict) or set(draw) != {"step_index", "sha256"}:
                raise ValueError("sampling draw fields are invalid")
            expected_draw_sha = _digest({
                "sampling_sha256": ledger["sampling_sha256"],
                "step_index": index,
                "artifact_sha256": artifact_sha,
                "token_id": token_id,
            })
            if draw.get("step_index") != index or draw.get("sha256") != expected_draw_sha:
                raise ValueError("sampling draw evidence does not match")
        token_ids.append(token_id)
    ledger["_token_ids"] = token_ids
    if policy is not None:
        ledger["_policy_projection"] = policy
    return ledger


def execute_request(
    request: Mapping[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    operation = str(request.get("operation") or "")
    if (
        request.get("schema_version") != SCHEMA_VERSION
        or request.get("tool") != TOOL
        or operation not in {"qwen3_token_ledger_decode", "qwen3_token_ledger_replay"}
        or request.get("read_only") is not True
        or request.get("network_access") != "disabled"
    ):
        return _base_result(
            "invalid_request", operation=operation or "qwen3_token_ledger_decode",
            errors=[{"code": "protocol_invalid", "message": "token ledger protocol is invalid"}],
        )
    model_path = _safe_dir(request.get("model_path"))
    ledger_path = _safe_file(request.get("ledger_path"))
    metadata = request.get("ledger_metadata")
    if (
        ledger_path is None
        or not isinstance(metadata, Mapping)
        or (operation == "qwen3_token_ledger_decode" and model_path is None)
    ):
        return _base_result(
            "invalid_request", operation=operation,
            errors=[{"code": "artifact_missing", "message": "model or ledger artifact is unavailable"}],
        )
    try:
        text_max = int(request.get("text_max_bytes", MAX_TEXT_BYTES))
        if not 0 < text_max <= MAX_TEXT_BYTES:
            raise ValueError("text limit is invalid")
        ledger = _load_ledger(ledger_path, metadata, request)
        ledger_result = {
            "ledger_id": ledger["ledger_id"],
            "sha256": metadata["sha256"],
            "size_bytes": metadata["size_bytes"],
            "token_count": ledger["token_count"],
            "stop_reason": ledger["stop_reason"],
            **({
                "sampling_sha256": ledger["sampling_sha256"],
                "draw_count": len(ledger["draw_evidence"]),
            } if "sampling_sha256" in ledger else {}),
            **({"quality_summary": ledger["quality_summary"]}
               if "quality_summary" in ledger else {}),
            **({"policy": ledger["_policy_projection"]}
               if "_policy_projection" in ledger else {}),
        }
        if operation == "qwen3_token_ledger_replay":
            return {
                **_base_result("replay_validated", gate_passed=True, operation=operation),
                "ledger": ledger_result,
                "replay": {
                    "sampler_validated": "sampling_sha256" in ledger,
                    "quality_validated": "quality_summary" in ledger,
                    "policy_validated": "_policy_projection" in ledger,
                    "full_model_materialized": False,
                    "weights_loaded": False,
                },
                "full_model_materialized": False,
                "weights_loaded": False,
            }
        transformers = module_loader("transformers")
        sidecar_python = Path(sys.executable).absolute().resolve(strict=False)
        controller_python = Path(str(request.get("controller_python", ""))).absolute().resolve(strict=False)
        if sidecar_python == controller_python:
            raise RuntimeError("tokenizer worker is not isolated")
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        if auto_tokenizer is None:
            raise RuntimeError("Transformers AutoTokenizer is unavailable")
        tokenizer = auto_tokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise RuntimeError("tokenizer does not provide decode")
        text = decode(ledger.pop("_token_ids"), skip_special_tokens=True)
        if not isinstance(text, str):
            raise RuntimeError("tokenizer returned a non-text result")
        encoded = text.encode("utf-8", errors="strict")
        if len(encoded) > text_max:
            raise ValueError("decoded text exceeds bounded UTF-8 size")
        text_sha = hashlib.sha256(encoded).hexdigest()
        return {
            **_base_result("decoded", gate_passed=True, operation=operation),
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "local_files_only": True,
                "trust_remote_code": False,
                "weights_loaded": False,
            },
            "ledger": ledger_result,
            "text": text,
            "text_sha256": text_sha,
            "text_bytes": len(encoded),
            "full_model_materialized": False,
            "weights_loaded": False,
        }
    except Exception as exc:
        return _base_result(
            "decode_failed" if operation == "qwen3_token_ledger_decode" else "replay_failed",
            operation=operation,
            errors=[{"code": "tokenizer_decode_failed", "message": exc.__class__.__name__}],
        )
    finally:
        gc.collect()


def main() -> int:
    raw = sys.stdin.buffer.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("token ledger request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    result = execute_request(request)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result.get("valid") is not False else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        result = _base_result(
            "invalid_request",
            errors=[{"code": "invalid_request", "message": exc.__class__.__name__}],
        )
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
