"""MM1 adapter for the existing local two/three-segment Qwen3 sidecar chain."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import secrets
import time
from typing import Any, Mapping, Sequence

from qwen3_multimodal_runtime import (
    Qwen3MultimodalRuntimeError,
    validate_mm1_first_segment_artifact_binding,
    validate_mm1_staged_text_contract,
)
from qwen3_pipeline_multisidecar import (
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
)


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _dtype(value: Any) -> str:
    return str(value or "").lower().removeprefix("torch.")


def _device(value: Any) -> str:
    result = str(value or "").lower()
    return "cuda" if result == "cuda" or result.startswith("cuda:") else result


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                "path" in lowered
                or "file" in lowered
                or "pixel" in lowered
                or lowered in {"prompt", "prompt_text", "prompt_content"}
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_sensitive",
                    "MM1 multi-sidecar metadata contains a sensitive field",
                )
            _reject_sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive(item)


def _decode_digest(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid",
            "MM1 decode binding is not canonical JSON",
        ) from exc
    if len(encoded) > 64 * 1024:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid",
            "MM1 decode binding exceeds 64 KiB",
        )
    return hashlib.sha256(encoded).hexdigest()


def _decode_identifier(value: Any, field: str) -> str:
    result = str(value or "")
    if not result or len(result) > 128 or any(
        character in result for character in "/\\\x00"
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", f"{field} is invalid",
        )
    return result


MAX_MM1_TOKEN_LEDGER_TOKENS = 4096
MAX_MM1_TOKEN_LEDGER_BYTES = 256 * 1024
MAX_MM1_TOKEN_ID = 2**31 - 1
MAX_MM1_SAMPLING_TOP_K = 4096
MAX_MM1_SAMPLING_SEED = 2**63 - 1
MAX_MM1_SAMPLING_TEMPERATURE = 2.0
MAX_MM1_POLICY_TTL_SECONDS = 7 * 24 * 60 * 60


def _ledger_digest(value: Mapping[str, Any]) -> str:
    return _decode_digest(value)


def _ledger_identifier(value: Any) -> str:
    return _decode_identifier(value, "ledger_id")


def _normalize_mm1_sampling(
    *, temperature: float, top_k: int, top_p: float, seed: int,
) -> dict[str, Any]:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 0.0 <= float(temperature) <= MAX_MM1_SAMPLING_TEMPERATURE
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 0 <= top_k <= MAX_MM1_SAMPLING_TOP_K
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0.0 < float(top_p) <= 1.0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_MM1_SAMPLING_SEED
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_bounded_decode_contract_invalid",
            "MM1 sampling parameters are outside their bounded contract",
        )
    normalized_temperature = float(temperature)
    return {
        "mode": "greedy" if normalized_temperature == 0.0 else "multinomial",
        "temperature": normalized_temperature,
        "top_k": int(top_k),
        "top_p": float(top_p),
        "seed": int(seed),
    }


def _sampling_digest(sampling: Mapping[str, Any]) -> str:
    return _decode_digest(sampling)


def _policy_snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    unsigned = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }
    return _decode_digest(unsigned)


def build_mm1_sampling_policy_snapshot(
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    policy_version: str = "mm1-sampling-v1",
    policy_id: str | None = None,
    issued_at: int | None = None,
    expires_at: int | None = None,
    replay_allowed: bool = True,
    route_scope: str = "mm1-bounded-decode",
) -> dict[str, Any]:
    sampling = _normalize_mm1_sampling(
        temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
    )
    version = _decode_identifier(policy_version, "policy_version")
    identifier = _decode_identifier(
        policy_id or f"mm1policy_{secrets.token_hex(16)}", "policy_id",
    )
    now = int(time.time())
    issued = now if issued_at is None else issued_at
    expires = issued + 3600 if expires_at is None else expires_at
    if (
        isinstance(issued, bool)
        or not isinstance(issued, int)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or expires <= issued
        or expires - issued > MAX_MM1_POLICY_TTL_SECONDS
        or not isinstance(replay_allowed, bool)
        or route_scope != "mm1-bounded-decode"
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid",
            "MM1 sampling policy lifetime or route scope is invalid",
        )
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "policy_kind": "qwen3_mm1_sampling_policy_snapshot",
        "policy_id": identifier,
        "policy_version": version,
        "issued_at": issued,
        "expires_at": expires,
        "replay_allowed": replay_allowed,
        "route_scope": route_scope,
        "sampling": sampling,
        "sampling_sha256": _sampling_digest(sampling),
    }
    snapshot["snapshot_sha256"] = _policy_snapshot_digest(snapshot)
    _reject_sensitive(snapshot)
    return snapshot


def validate_mm1_sampling_policy_snapshot(
    snapshot: Mapping[str, Any], *, now: int | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version", "policy_kind", "policy_id", "policy_version", "issued_at",
        "expires_at", "replay_allowed", "route_scope", "sampling", "sampling_sha256",
        "snapshot_sha256",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != required:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy fields are invalid",
        )
    if (
        snapshot["schema_version"] != 1
        or snapshot["policy_kind"] != "qwen3_mm1_sampling_policy_snapshot"
        or snapshot["route_scope"] != "mm1-bounded-decode"
        or not isinstance(snapshot["replay_allowed"], bool)
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy identity is invalid",
        )
    if not isinstance(snapshot["policy_id"], str) or not isinstance(snapshot["policy_version"], str):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy identifiers are invalid",
        )
    policy_id = _decode_identifier(snapshot["policy_id"], "policy_id")
    policy_version = _decode_identifier(snapshot["policy_version"], "policy_version")
    if not isinstance(snapshot["issued_at"], int) or not isinstance(snapshot["expires_at"], int):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy timestamps are invalid",
        )
    try:
        issued = int(snapshot["issued_at"])
        expires = int(snapshot["expires_at"])
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy timestamps are invalid",
        ) from exc
    if (
        isinstance(snapshot["issued_at"], bool)
        or isinstance(snapshot["expires_at"], bool)
        or expires <= issued
        or expires - issued > MAX_MM1_POLICY_TTL_SECONDS
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy lifetime is invalid",
        )
    sampling = snapshot["sampling"]
    if not isinstance(sampling, Mapping):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy payload is invalid",
        )
    normalized = _normalize_mm1_sampling(
        temperature=sampling.get("temperature"),
        top_k=sampling.get("top_k"),
        top_p=sampling.get("top_p"),
        seed=sampling.get("seed"),
    )
    if (
        dict(sampling) != normalized
        or snapshot["sampling_sha256"] != _sampling_digest(normalized)
        or snapshot["snapshot_sha256"] != _policy_snapshot_digest(snapshot)
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_invalid", "MM1 sampling policy digest does not match",
        )
    current = int(time.time()) if now is None else now
    if current < issued - 300 or current >= expires:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_policy_expired", "MM1 sampling policy is outside its valid time window",
        )
    return {
        **dict(snapshot),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "issued_at": issued,
        "expires_at": expires,
        "sampling": normalized,
    }


def _sampling_draw_digest(
    *, sampling: Mapping[str, Any], step_index: int, artifact_sha256: str, token_id: int,
) -> str:
    return _decode_digest({
        "sampling_sha256": _sampling_digest(sampling),
        "step_index": step_index,
        "artifact_sha256": artifact_sha256,
        "token_id": token_id,
    })


def _quality_step_digest(
    *, sampling: Mapping[str, Any], step_index: int, artifact_sha256: str,
    quality: Mapping[str, Any],
) -> str:
    return _decode_digest({
        "sampling_sha256": _sampling_digest(sampling),
        "step_index": step_index,
        "artifact_sha256": artifact_sha256,
        "quality": dict(quality),
    })


def _quality_summary_from_state(
    *, sampling: Mapping[str, Any], state: Mapping[str, Any], quality_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "step_count": int(state["step_count"]),
        "candidate_count_min": int(state["candidate_count_min"]),
        "candidate_count_max": int(state["candidate_count_max"]),
        "entropy_min": round(float(state["entropy_min"]), 8),
        "entropy_max": round(float(state["entropy_max"]), 8),
        "confidence_min": round(float(state["confidence_min"]), 8),
        "confidence_max": round(float(state["confidence_max"]), 8),
        "top_p_cutoff_min": int(state["top_p_cutoff_min"]),
        "top_p_cutoff_max": int(state["top_p_cutoff_max"]),
        "sampling_sha256": _sampling_digest(sampling),
        "sha256": quality_sha256,
    }


def _quality_state() -> dict[str, Any]:
    return {
        "step_count": 0,
        "candidate_count_min": MAX_MM1_TOKEN_ID,
        "candidate_count_max": 0,
        "entropy_min": float("inf"),
        "entropy_max": 0.0,
        "confidence_min": float("inf"),
        "confidence_max": 0.0,
        "top_p_cutoff_min": MAX_MM1_TOKEN_ID,
        "top_p_cutoff_max": 0,
    }


def _quality_state_update(
    state: dict[str, Any], quality: Mapping[str, Any], *, step_index: int,
) -> None:
    try:
        candidate_count = int(quality["candidate_count"])
        entropy = float(quality["entropy"])
        confidence = float(quality["confidence"])
        top_p_cutoff = int(quality["top_p_cutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_quality_invalid", "MM1 sampling quality dimensions are invalid",
        ) from exc
    if (
        candidate_count <= 0
        or candidate_count > MAX_MM1_TOKEN_ID
        or top_p_cutoff <= 0
        or top_p_cutoff > candidate_count
        or not math.isfinite(entropy)
        or entropy < 0.0
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_quality_invalid", "MM1 sampling quality values are outside limits",
        )
    if quality.get("step_index", step_index) != step_index:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_sampling_quality_invalid", "MM1 sampling quality step is not sequential",
        )
    state["step_count"] += 1
    state["candidate_count_min"] = min(state["candidate_count_min"], candidate_count)
    state["candidate_count_max"] = max(state["candidate_count_max"], candidate_count)
    state["entropy_min"] = min(state["entropy_min"], entropy)
    state["entropy_max"] = max(state["entropy_max"], entropy)
    state["confidence_min"] = min(state["confidence_min"], confidence)
    state["confidence_max"] = max(state["confidence_max"], confidence)
    state["top_p_cutoff_min"] = min(state["top_p_cutoff_min"], top_p_cutoff)
    state["top_p_cutoff_max"] = max(state["top_p_cutoff_max"], top_p_cutoff)


def _build_mm1_token_ledger(
    *,
    staged: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    stop_reason: str,
    sampling: Mapping[str, Any] | None = None,
    draw_evidence: Sequence[Mapping[str, Any]] | None = None,
    quality_summary: Mapping[str, Any] | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not records or len(records) > MAX_MM1_TOKEN_LEDGER_TOKENS:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 token ledger length is outside limits",
        )
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(records, start=1):
        if not isinstance(value, Mapping):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 token ledger record is invalid",
            )
        try:
            step = int(value["step_index"])
            generation = int(value["generation"])
            sequence_length = int(value["sequence_length"])
            token_id = int(value["token_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 token ledger record dimensions are invalid",
            ) from exc
        artifact_sha256 = str(value.get("artifact_sha256") or "").lower()
        if (
            step != index
            or generation != staged["generation"] + index
            or sequence_length != staged["input_layout"]["total_sequence"] + index
            or token_id < 0
            or token_id > MAX_MM1_TOKEN_ID
            or len(artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_sha256)
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 token ledger record does not advance the chain",
            )
        normalized.append({
            "step_index": step,
            "generation": generation,
            "sequence_length": sequence_length,
            "token_id": token_id,
            "artifact_sha256": artifact_sha256,
        })
    if stop_reason not in {"eos", "max_new_tokens"}:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 token ledger stop reason is invalid",
        )
    normalized_sampling = dict(sampling or _normalize_mm1_sampling(
        temperature=0.0, top_k=0, top_p=1.0, seed=0,
    ))
    if set(normalized_sampling) != {"mode", "temperature", "top_k", "top_p", "seed"}:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 sampling contract fields are invalid",
        )
    try:
        validated_sampling = _normalize_mm1_sampling(
            temperature=normalized_sampling["temperature"],
            top_k=normalized_sampling["top_k"],
            top_p=normalized_sampling["top_p"],
            seed=normalized_sampling["seed"],
        )
    except Qwen3MultimodalRuntimeError as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 sampling contract values are invalid",
        ) from exc
    if validated_sampling != normalized_sampling:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 sampling mode does not match its values",
        )
    normalized_sampling = validated_sampling
    provided_draws = list(draw_evidence or ())
    if provided_draws and len(provided_draws) != len(normalized):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 sampling draw evidence count is invalid",
        )
    normalized_draws: list[dict[str, Any]] = []
    for index, value in enumerate(provided_draws, start=1):
        if not isinstance(value, Mapping) or value.get("step_index") != index:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 sampling draw evidence is invalid",
            )
        digest = str(value.get("sha256") or "").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 sampling draw evidence digest is invalid",
            )
        expected_digest = _sampling_draw_digest(
            sampling=normalized_sampling,
            step_index=index,
            artifact_sha256=normalized[index - 1]["artifact_sha256"],
            token_id=normalized[index - 1]["token_id"],
        )
        if digest != expected_digest:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 sampling draw evidence does not match",
            )
        normalized_draws.append({"step_index": index, "sha256": digest})
    if not normalized_draws:
        normalized_draws = [
            {"step_index": record["step_index"], "sha256": _sampling_draw_digest(
                sampling=normalized_sampling,
                step_index=record["step_index"],
                artifact_sha256=record["artifact_sha256"],
                token_id=record["token_id"],
            )}
            for record in normalized
        ]
    if quality_summary is None:
        quality_state = _quality_state()
        quality_digest = hashlib.sha256()
        for record in normalized:
            quality = {
                "candidate_count": 1,
                "entropy": 0.0,
                "confidence": 1.0,
                "top_p_cutoff": 1,
            }
            _quality_state_update(
                quality_state, quality, step_index=record["step_index"],
            )
            quality_digest.update(json.dumps({
                "step_index": record["step_index"],
                "artifact_sha256": record["artifact_sha256"],
                "quality": quality,
                "sampling_sha256": _sampling_digest(normalized_sampling),
            }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        normalized_quality = _quality_summary_from_state(
            sampling=normalized_sampling,
            state=quality_state,
            quality_sha256=quality_digest.hexdigest(),
        )
    else:
        normalized_quality = dict(quality_summary)
        expected_quality_fields = {
            "schema_version", "step_count", "candidate_count_min", "candidate_count_max",
            "entropy_min", "entropy_max", "confidence_min", "confidence_max",
            "top_p_cutoff_min", "top_p_cutoff_max", "sampling_sha256", "sha256",
        }
        if set(normalized_quality) != expected_quality_fields:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 sampling quality summary fields are invalid",
            )
        if (
            normalized_quality["schema_version"] != 1
            or normalized_quality["step_count"] != len(normalized)
            or normalized_quality["sampling_sha256"] != _sampling_digest(normalized_sampling)
            or len(str(normalized_quality["sha256"] or "")) != 64
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 sampling quality summary does not match",
            )
    values: dict[str, Any] = {
        "schema_version": 1,
        "ledger_kind": "qwen3_mm1_generated_token_ledger",
        "ledger_id": f"mm1ledger_{secrets.token_hex(16)}",
        "text_chain_id": staged["text_chain_id"],
        "contract_sha256": staged["contract_sha256"],
        "prefill_generation": staged["generation"],
        "token_count": len(normalized),
        "stop_reason": stop_reason,
        "records": normalized,
        "sampling": normalized_sampling,
        "sampling_sha256": _sampling_digest(normalized_sampling),
        "draw_evidence": normalized_draws,
        "quality_summary": normalized_quality,
        "full_model_materialized": False,
    }
    if policy_snapshot is not None:
        normalized_policy = validate_mm1_sampling_policy_snapshot(policy_snapshot)
        if normalized_policy["sampling"] != normalized_sampling:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_invalid", "MM1 ledger policy sampling does not match",
            )
        values["policy_snapshot"] = normalized_policy
        values["policy_snapshot_sha256"] = normalized_policy["snapshot_sha256"]
    values["ledger_sha256"] = _ledger_digest(values)
    encoded = json.dumps(
        values, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_MM1_TOKEN_LEDGER_BYTES:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_token_ledger_invalid", "MM1 token ledger exceeds its bounded size",
        )
    _reject_sensitive({key: value for key, value in values.items() if key != "records"})
    return values


def _write_mm1_token_ledger(
    *, artifact_root: Path, staged: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    stop_reason: str, prefix: str, sampling: Mapping[str, Any] | None = None,
    draw_evidence: Sequence[Mapping[str, Any]] | None = None,
    quality_summary: Mapping[str, Any] | None = None,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    ledger = _build_mm1_token_ledger(
        staged=staged, records=records, stop_reason=stop_reason,
        sampling=sampling, draw_evidence=draw_evidence,
        quality_summary=quality_summary,
        policy_snapshot=policy_snapshot,
    )
    temporary = artifact_root / f"{prefix}ledger-tmp-{secrets.token_hex(16)}.json"
    target = artifact_root / f"{prefix}ledger-{secrets.token_hex(16)}.json"
    encoded = json.dumps(
        ledger, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(target)
        size, sha256 = _file_evidence(target)
        if size != len(encoded):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_mismatch", "MM1 token ledger evidence changed while committing",
            )
        metadata = {
            "ledger_id": ledger["ledger_id"],
            "size_bytes": size,
            "sha256": sha256,
            "status": "committed",
            "content_kind": "generated_token_ledger",
            "token_count": ledger["token_count"],
            "stop_reason": ledger["stop_reason"],
            "sampling_sha256": ledger["sampling_sha256"],
            "draw_count": len(ledger["draw_evidence"]),
            "quality_sha256": ledger["quality_summary"]["sha256"],
        }
        if "policy_snapshot" in ledger:
            metadata.update({
                "policy_snapshot_sha256": ledger["policy_snapshot_sha256"],
                "policy_id": ledger["policy_snapshot"]["policy_id"],
                "policy_version": ledger["policy_snapshot"]["policy_version"],
            })
        return metadata, target
    except Exception:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def build_mm1_decode_artifact_binding(
    staged_contract: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifact_id: str,
    size_bytes: int,
    sha256: str,
    batch_size: int = 1,
    token_count: int = 1,
) -> dict[str, Any]:
    """Bind a local decode input artifact to the MM1 staged generation."""
    staged = validate_mm1_staged_text_contract(staged_contract, manifest=manifest)
    try:
        batch = int(batch_size)
        tokens = int(token_count)
        size = int(size_bytes)
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode dimensions are invalid",
        ) from exc
    if (
        batch != staged["input_layout"]["batch_size"]
        or batch <= 0
        or tokens <= 0
        or tokens > 8192
        or size <= 0
        or size > (1 << 40)
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode dimensions are outside limits",
        )
    digest = str(sha256 or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode artifact digest is invalid",
        )
    prefill_length = staged["input_layout"]["total_sequence"]
    values = {
        "schema_version": 1,
        "contract_kind": "qwen3_mm1_decode_input_artifact",
        "staged_contract_sha256": staged["contract_sha256"],
        "model_id": staged["model_id"],
        "manifest_sha256": staged["manifest_sha256"],
        "text_chain_id": staged["text_chain_id"],
        "phase": "decode",
        "prefill_generation": staged["generation"],
        "decode_generation": staged["generation"] + 1,
        "input_artifact": {
            "artifact_id": _decode_identifier(artifact_id, "artifact_id"),
            "size_bytes": size,
            "sha256": digest,
            "status": "committed",
            "serialization": "torch_pt",
            "content_kind": "decode_input_ids",
        },
        "tensor": {
            "shape": [batch, tokens],
            "dtype": "int64",
            "storage_device": "cpu",
        },
        "sequence": {
            "prefill_length": prefill_length,
            "decode_input_tokens": tokens,
            "decode_length": prefill_length + tokens,
        },
        "full_model_materialized": False,
    }
    values["contract_sha256"] = _decode_digest(values)
    return validate_mm1_decode_artifact_binding(
        values, staged_contract=staged, manifest=manifest,
    )


def validate_mm1_decode_artifact_binding(
    value: Mapping[str, Any],
    *,
    staged_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a path-free decode input binding and its generation fence."""
    staged = validate_mm1_staged_text_contract(staged_contract, manifest=manifest)
    required = {
        "schema_version", "contract_kind", "staged_contract_sha256", "model_id",
        "manifest_sha256", "text_chain_id", "phase", "prefill_generation",
        "decode_generation", "input_artifact", "tensor", "sequence",
        "full_model_materialized", "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode binding fields are invalid",
        )
    try:
        binding = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode binding is not JSON serializable",
        ) from exc
    if (
        binding["schema_version"] != 1
        or binding["contract_kind"] != "qwen3_mm1_decode_input_artifact"
        or binding["staged_contract_sha256"] != staged["contract_sha256"]
        or binding["model_id"] != staged["model_id"]
        or binding["manifest_sha256"] != staged["manifest_sha256"]
        or binding["text_chain_id"] != staged["text_chain_id"]
        or binding["phase"] != "decode"
        or binding["prefill_generation"] != staged["generation"]
        or binding["decode_generation"] != staged["generation"] + 1
        or binding["full_model_materialized"] is not False
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode generation identity does not match",
        )
    artifact = binding["input_artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "artifact_id", "size_bytes", "sha256", "status", "serialization", "content_kind",
    }:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode artifact metadata is invalid",
        )
    _decode_identifier(artifact.get("artifact_id"), "artifact_id")
    try:
        size = int(artifact["size_bytes"])
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode artifact size is invalid",
        ) from exc
    digest = str(artifact.get("sha256") or "")
    if (
        size <= 0 or size > (1 << 40)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or artifact["status"] != "committed"
        or artifact["serialization"] != "torch_pt"
        or artifact["content_kind"] != "decode_input_ids"
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode artifact evidence is invalid",
        )
    layout = staged["input_layout"]
    tensor = binding["tensor"]
    sequence = binding["sequence"]
    if not isinstance(tensor, Mapping) or set(tensor) != {
        "shape", "dtype", "storage_device",
    } or not isinstance(sequence, Mapping) or set(sequence) != {
        "prefill_length", "decode_input_tokens", "decode_length",
    }:
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode tensor or sequence metadata is invalid",
        )
    shape = tensor["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != layout["batch_size"]
        or not isinstance(shape[1], int)
        or shape[1] <= 0
        or shape[1] > 8192
        or tensor["dtype"] != "int64"
        or tensor["storage_device"] != "cpu"
        or sequence["prefill_length"] != layout["total_sequence"]
        or sequence["decode_input_tokens"] != shape[1]
        or sequence["decode_length"] != layout["total_sequence"] + shape[1]
    ):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode sequence metadata does not match",
        )
    unsigned = dict(binding)
    contract_digest = str(unsigned.pop("contract_sha256", ""))
    if len(contract_digest) != 64 or contract_digest != _decode_digest(unsigned):
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_binding_invalid", "MM1 decode binding digest does not match",
        )
    _reject_sensitive(binding)
    return binding


class Qwen3MultimodalMultiSidecarAdapter:
    """Bind an MM1 staged contract to a local 2/3-sidecar prefill chain."""

    def __init__(
        self,
        *,
        staged_contract: Mapping[str, Any],
        manifest: Mapping[str, Any],
        artifact_binding: Mapping[str, Any],
        sessions: Sequence[Any],
        artifact_root: str | Path,
    ) -> None:
        self.staged = validate_mm1_staged_text_contract(
            staged_contract, manifest=manifest,
        )
        self.manifest = manifest
        self.binding = validate_mm1_first_segment_artifact_binding(
            artifact_binding,
            staged_contract=self.staged,
            manifest=manifest,
        )
        self.sessions = list(sessions)
        segments = self.staged["segment_plan"]
        if len(self.sessions) != len(segments):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_session_mismatch",
                "MM1 sidecar session count does not match the staged plan",
            )
        for index, (session, segment) in enumerate(zip(self.sessions, segments)):
            self._validate_session_identity(session, segment, index=index)
        for current, following in zip(segments, segments[1:]):
            if (
                _dtype(current["dtype"]) != _dtype(following["dtype"])
                or _device(current["device"]) != _device(following["device"])
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_boundary_mismatch",
                    "local MM1 sidecar boundaries require matching dtype and device",
                )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_artifact_root_missing",
                "MM1 multi-sidecar artifact root is unavailable",
            )
        self.artifact_root = root
        self.lifecycle: list[str] = []
        self._outputs: dict[str, Path] = {}
        self._ledger_outputs: dict[str, Path] = {}
        self._ledger_metadata: dict[str, dict[str, Any]] = {}
        self._retained_prefix = f"mm1-{self.staged['contract_sha256'][:20]}-"
        chain_segments = [
            {
                "layer_range": list(segment["layer_range"]),
                "has_embedding": segment["has_embedding"],
                "has_lm_head": segment["has_lm_head"],
                "device": segment["device"],
                "dtype": segment["dtype"],
            }
            for segment in segments
        ]
        try:
            self.chain = Qwen3PipelineMultiSidecar(
                sessions=self.sessions,
                segments=chain_segments,
                artifact_root=root,
                chain_id=self.staged["text_chain_id"],
                generation=self.staged["generation"],
                node_ids=[segment["node_id"] for segment in segments],
                hidden_size=self.staged["input_layout"]["hidden_size"],
            )
        except Qwen3MultiSidecarError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_contract_invalid",
                "MM1 staged plan could not create a local sidecar chain",
            ) from exc

    def _validate_session_identity(
        self, session: Any, segment: Mapping[str, Any], *, index: int,
    ) -> None:
        identity = getattr(session, "identity", None)
        if not isinstance(identity, Mapping) or getattr(session, "phase", None) != "idle":
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_session_invalid",
                "MM1 sidecar session is unavailable or already used",
            )
        expected = {
            "model_id": self.staged["model_id"],
            "node_id": segment["node_id"],
            "layer_range": segment["layer_range"],
            "total_layers": self.staged["total_layers"],
            "has_embedding": segment["has_embedding"],
            "has_lm_head": segment["has_lm_head"],
            "generation": self.staged["generation"],
            "assignment_manifest_sha256": segment["assignment_manifest_sha256"],
        }
        if any(identity.get(key) != expected_value for key, expected_value in expected.items()):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_session_mismatch",
                f"MM1 sidecar session {index} does not match its assignment",
            )
        if (
            _dtype(identity.get("dtype")) != _dtype(segment["dtype"])
            or _device(identity.get("execution_device")) != _device(segment["device"])
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_session_mismatch",
                f"MM1 sidecar session {index} dtype or device does not match",
            )

    def _local_input(self, value: str | Path) -> Path:
        return self._local_artifact(value, self.binding["input_artifact"])

    def _local_artifact(self, value: str | Path, artifact: Mapping[str, Any]) -> Path:
        candidate = Path(value).expanduser().absolute().resolve(strict=False)
        try:
            candidate.relative_to(self.artifact_root)
        except ValueError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_artifact_scope",
                "MM1 chain input escapes the local data-plane root",
            ) from exc
        if not candidate.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_artifact_missing",
                "MM1 chain input artifact is unavailable",
            )
        if _file_evidence(candidate) != (artifact["size_bytes"], artifact["sha256"]):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_artifact_mismatch",
                "MM1 chain input artifact evidence does not match the binding",
            )
        return candidate

    def _validate_phase(
        self,
        *,
        phase: str,
        sequence_length: int,
        handoff_sequence_length: int,
        generation: int,
    ) -> tuple[list[dict[str, Any]], Path, tuple[int, str]]:
        reports = self.chain.execution_reports(phase)
        outputs = self.chain.artifact_refs(phase)
        references = self.chain.handoff_references(phase)
        segments = self.staged["segment_plan"]
        layout = self.staged["input_layout"]
        if (
            len(reports) != len(segments)
            or len(outputs) != len(segments)
            or len(references) != len(segments) - 1
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_execution_failed",
                f"MM1 {phase} chain returned incomplete segment evidence",
            )
        expected_shape = [
            layout["batch_size"], handoff_sequence_length, layout["hidden_size"],
        ]
        summaries: list[dict[str, Any]] = []
        for index, (report, output, segment) in enumerate(zip(reports, outputs, segments)):
            if not output.is_file():
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_artifact_missing",
                    "MM1 segment output artifact is unavailable",
                )
            actual_size, actual_sha256 = _file_evidence(output)
            execution = report.get("execution")
            hidden = report.get("hidden_handoff")
            kv = report.get("kv_contract")
            if not isinstance(execution, Mapping) or (
                execution.get("data_plane") != "local_artifact"
                or execution.get("artifact_bytes") != actual_size
                or execution.get("artifact_sha256") != actual_sha256
                or execution.get("segment_materialized") is not True
                or execution.get("full_model_materialized") is not False
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_artifact_mismatch",
                    "MM1 segment output evidence does not match its local artifact",
                )
            if not isinstance(kv, Mapping) or (
                kv.get("schema_version") != 1
                or kv.get("chain_id") != self.staged["text_chain_id"]
                or kv.get("segment_index") != index
                or kv.get("layer_range") != segment["layer_range"]
                or kv.get("sequence_length") != sequence_length
                or kv.get("batch_size") != layout["batch_size"]
                or _dtype(kv.get("dtype")) != _dtype(segment["dtype"])
                or _device(kv.get("device")) != _device(segment["device"])
                or kv.get("phase") != phase
                or kv.get("generation") != generation
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_kv_mismatch",
                    "MM1 segment KV summary does not match the staged contract",
                )
            reference = None
            if index < len(segments) - 1:
                if not isinstance(hidden, Mapping) or (
                    hidden.get("schema_version") != 1
                    or hidden.get("chain_id") != self.staged["text_chain_id"]
                    or hidden.get("from_segment") != index
                    or hidden.get("to_segment") != index + 1
                    or hidden.get("shape") != expected_shape
                    or hidden.get("batch_size") != layout["batch_size"]
                    or hidden.get("sequence_length") != handoff_sequence_length
                    or hidden.get("hidden_size") != layout["hidden_size"]
                    or _dtype(hidden.get("dtype")) != _dtype(segment["dtype"])
                    or _device(hidden.get("device")) != _device(segment["device"])
                ):
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_handoff_mismatch",
                        "MM1 hidden handoff does not match the next segment",
                    )
                reference = references[index]
                following = segments[index + 1]
                if not isinstance(reference, Mapping) or (
                    reference.get("mode") != "local"
                    or reference.get("chain_id") != self.staged["text_chain_id"]
                    or reference.get("generation") != generation
                    or reference.get("phase") != phase
                    or reference.get("from_segment") != index
                    or reference.get("to_segment") != index + 1
                    or reference.get("source_node_id") != segment["node_id"]
                    or reference.get("target_node_id") != following["node_id"]
                    or reference.get("size_bytes") != actual_size
                    or reference.get("sha256") != actual_sha256
                    or reference.get("status") != "committed"
                    or reference.get("full_model_materialized") is not False
                    or report.get("artifact_reference") != reference
                ):
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_reference_mismatch",
                        "MM1 local handoff reference does not match its segment boundary",
                    )
            elif hidden is not None or report.get("artifact_reference") is not None:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_handoff_mismatch",
                    "MM1 final segment returned an unexpected hidden handoff",
                )
            summaries.append({
                "segment_index": index,
                "node_id": segment["node_id"],
                "layer_range": list(segment["layer_range"]),
                "output_artifact": {
                    "size_bytes": actual_size,
                    "sha256": actual_sha256,
                    "status": "committed",
                },
                "hidden_handoff": dict(hidden) if isinstance(hidden, Mapping) else None,
                "artifact_reference": dict(reference) if isinstance(reference, Mapping) else None,
                "kv_contract": dict(kv),
                "full_model_materialized": False,
            })
        _reject_sensitive(summaries)
        final_path = outputs[-1]
        return summaries, final_path, _file_evidence(final_path)

    def _validate_prefill(self) -> tuple[list[dict[str, Any]], Path, tuple[int, str]]:
        length = self.staged["input_layout"]["total_sequence"]
        return self._validate_phase(
            phase="prefill",
            sequence_length=length,
            handoff_sequence_length=length,
            generation=self.staged["generation"],
        )

    @staticmethod
    def _validate_session_cleanup(sessions: Sequence[Any]) -> None:
        for session in sessions:
            if getattr(session, "phase", None) not in {"released", "aborted"}:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_cleanup_failed",
                    "MM1 sidecar session remained materialized after cleanup",
                )
            snapshot = getattr(session, "snapshot", None)
            report = snapshot() if callable(snapshot) else None
            if not isinstance(report, Mapping) or (
                report.get("cleanup_complete") is not True
                or report.get("segment_materialized") is not False
                or report.get("full_model_materialized") is not False
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_cleanup_failed",
                    "MM1 sidecar session did not prove cleanup",
                )

    @classmethod
    def _validate_chain_cleanup(
        cls, snapshot: Mapping[str, Any], sessions: Sequence[Any], *, expected_phase: str,
    ) -> None:
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("phase") != expected_phase
            or snapshot.get("cleanup_complete") is not True
            or snapshot.get("created_artifact_count") != 0
            or snapshot.get("full_model_materialized") is not False
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_cleanup_failed",
                "MM1 multi-sidecar chain did not prove cleanup",
            )
        cls._validate_session_cleanup(sessions)

    def _remove_retained(self) -> int:
        paths = set(self._outputs.values())
        paths.update(self._ledger_outputs.values())
        paths.update(self.artifact_root.glob(f"{self._retained_prefix}*.pt"))
        paths.update(self.artifact_root.glob(f"{self._retained_prefix}*.json"))
        removed = 0
        failures = 0
        for path in paths:
            try:
                existed = path.exists()
                path.unlink(missing_ok=True)
                removed += int(existed)
            except OSError:
                failures += 1
        self._outputs.clear()
        self._ledger_outputs.clear()
        self._ledger_metadata.clear()
        if failures:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_cleanup_failed",
                "MM1 retained artifact cleanup did not complete",
            )
        return removed

    def execute_prefill(
        self, *, input_ref: str | Path, cancel_after_commit: bool = False,
    ) -> dict[str, Any]:
        """Execute all staged text segments and return only path-free evidence."""
        if not isinstance(cancel_after_commit, bool):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_contract_invalid", "cancel flag must be boolean",
            )
        input_path = self._local_input(input_ref)
        retained: Path | None = None
        try:
            self.chain.prepare()
            self.lifecycle.append("prepare")
            self.chain.commit()
            self.lifecycle.append("commit")
            if cancel_after_commit:
                cancelled = self.chain.cancel()
                self.lifecycle.append("cancel")
                self._validate_chain_cleanup(cancelled, self.sessions, expected_phase="aborted")
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_cancelled",
                    "MM1 multi-sidecar execution was cancelled after commit",
                )
            self.chain.prefill(
                input_ref=input_path,
                batch_size=self.staged["input_layout"]["batch_size"],
                sequence_length=self.staged["input_layout"]["total_sequence"],
            )
            self.lifecycle.append("prefill")
            segment_reports, final_path, final_evidence = self._validate_prefill()
            retained = self.artifact_root / (
                f"{self._retained_prefix}{secrets.token_hex(16)}.pt"
            )
            final_path.replace(retained)
            if _file_evidence(retained) != final_evidence:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_artifact_mismatch",
                    "MM1 final artifact changed while being retained",
                )
            released = self.chain.release()
            self.lifecycle.append("release")
            self._validate_chain_cleanup(released, self.sessions, expected_phase="released")
            output_id = f"mm1final_{final_evidence[1][:32]}"
            self._outputs[output_id] = retained
            result = {
                "schema_version": 1,
                "status": "multimodal_text_chain_prefilled",
                "contract_sha256": self.staged["contract_sha256"],
                "artifact_binding_sha256": self.binding["contract_sha256"],
                "text_chain_id": self.staged["text_chain_id"],
                "generation": self.staged["generation"],
                "segment_count": len(self.staged["segment_plan"]),
                "lifecycle": list(self.lifecycle),
                "segment_reports": segment_reports,
                "final_artifact": {
                    "artifact_id": output_id,
                    "size_bytes": final_evidence[0],
                    "sha256": final_evidence[1],
                    "status": "committed",
                    "content_kind": "final_segment_output",
                },
                "final_kv_contract": dict(segment_reports[-1]["kv_contract"]),
                "sidecar_cleanup_complete": True,
                "artifact_cleanup_required": True,
                "segment_materialized": False,
                "full_model_materialized": False,
            }
            _reject_sensitive(result)
            return result
        except Exception as exc:
            if retained is not None:
                try:
                    retained.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 final artifact cleanup did not complete",
                    ) from cleanup_exc
            if self.chain.phase not in {"released", "aborted"}:
                try:
                    aborted = self.chain.abort()
                    self.lifecycle.append("abort")
                    self._validate_chain_cleanup(
                        aborted, self.sessions, expected_phase="aborted",
                    )
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, Qwen3MultimodalRuntimeError):
                        raise
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 multi-sidecar abort did not complete",
                    ) from cleanup_exc
            if isinstance(exc, Qwen3MultimodalRuntimeError):
                raise
            if isinstance(exc, Qwen3MultiSidecarError) and exc.reason_code in {
                "qwen3_multisidecar_handoff_mismatch",
                "qwen3_multisidecar_boundary_mismatch",
            }:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_handoff_mismatch",
                    "MM1 multi-sidecar hidden handoff failed before the next segment",
                ) from exc
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_execution_failed",
                "MM1 multi-sidecar execution failed",
            ) from exc

    def execute_prefill_decode(
        self,
        *,
        input_ref: str | Path,
        decode_input_ref: str | Path,
        decode_binding: Mapping[str, Any],
        cancel_after_commit: bool = False,
        cancel_after_prefill: bool = False,
    ) -> dict[str, Any]:
        """Run prefill and decode before releasing the per-segment KV artifacts."""
        if not isinstance(cancel_after_commit, bool) or not isinstance(cancel_after_prefill, bool):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_contract_invalid", "cancel flags must be boolean",
            )
        decode = validate_mm1_decode_artifact_binding(
            decode_binding,
            staged_contract=self.staged,
            manifest=self.manifest,
        )
        input_path = self._local_input(input_ref)
        decode_path = self._local_artifact(
            decode_input_ref, decode["input_artifact"],
        )
        retained: Path | None = None
        try:
            self.chain.prepare()
            self.lifecycle.append("prepare")
            self.chain.commit()
            self.lifecycle.append("commit")
            if cancel_after_commit:
                cancelled = self.chain.cancel()
                self.lifecycle.append("cancel")
                self._validate_chain_cleanup(cancelled, self.sessions, expected_phase="aborted")
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_cancelled",
                    "MM1 multi-sidecar execution was cancelled after commit",
                )
            layout = self.staged["input_layout"]
            self.chain.prefill(
                input_ref=input_path,
                batch_size=layout["batch_size"],
                sequence_length=layout["total_sequence"],
            )
            self.lifecycle.append("prefill")
            prefill_reports, _prefill_final, _prefill_evidence = self._validate_prefill()
            if cancel_after_prefill:
                cancelled = self.chain.cancel()
                self.lifecycle.append("cancel")
                self._validate_chain_cleanup(cancelled, self.sessions, expected_phase="aborted")
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_cancelled",
                    "MM1 multi-sidecar execution was cancelled after prefill",
                )
            sequence = decode["sequence"]
            self.chain.decode(
                input_ref=decode_path,
                batch_size=decode["tensor"]["shape"][0],
                sequence_length=sequence["decode_length"],
                input_sequence_length=sequence["decode_input_tokens"],
            )
            self.lifecycle.append("decode")
            decode_reports, final_path, final_evidence = self._validate_phase(
                phase="decode",
                sequence_length=sequence["decode_length"],
                handoff_sequence_length=sequence["decode_input_tokens"],
                generation=decode["decode_generation"],
            )
            retained = self.artifact_root / (
                f"{self._retained_prefix}{secrets.token_hex(16)}.pt"
            )
            final_path.replace(retained)
            if _file_evidence(retained) != final_evidence:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_artifact_mismatch",
                    "MM1 final decode artifact changed while being retained",
                )
            released = self.chain.release()
            self.lifecycle.append("release")
            self._validate_chain_cleanup(released, self.sessions, expected_phase="released")
            output_id = f"mm1decode_{final_evidence[1][:32]}"
            self._outputs[output_id] = retained
            result = {
                "schema_version": 1,
                "status": "multimodal_text_chain_decoded",
                "contract_sha256": self.staged["contract_sha256"],
                "artifact_binding_sha256": self.binding["contract_sha256"],
                "decode_binding_sha256": decode["contract_sha256"],
                "text_chain_id": self.staged["text_chain_id"],
                "prefill_generation": self.staged["generation"],
                "decode_generation": decode["decode_generation"],
                "segment_count": len(self.staged["segment_plan"]),
                "lifecycle": list(self.lifecycle),
                "decode_input_artifact": dict(decode["input_artifact"]),
                "prefill_segment_reports": prefill_reports,
                "decode_segment_reports": decode_reports,
                "final_artifact": {
                    "artifact_id": output_id,
                    "size_bytes": final_evidence[0],
                    "sha256": final_evidence[1],
                    "status": "committed",
                    "content_kind": "final_decode_output",
                },
                "final_kv_contract": dict(decode_reports[-1]["kv_contract"]),
                "decode_sequence": dict(sequence),
                "sidecar_cleanup_complete": True,
                "artifact_cleanup_required": True,
                "segment_materialized": False,
                "full_model_materialized": False,
            }
            _reject_sensitive(result)
            return result
        except Exception as exc:
            if retained is not None:
                try:
                    retained.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 final decode artifact cleanup did not complete",
                    ) from cleanup_exc
            if self.chain.phase not in {"released", "aborted"}:
                try:
                    aborted = self.chain.abort()
                    self.lifecycle.append("abort")
                    self._validate_chain_cleanup(
                        aborted, self.sessions, expected_phase="aborted",
                    )
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, Qwen3MultimodalRuntimeError):
                        raise
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 multi-sidecar decode abort did not complete",
                    ) from cleanup_exc
            if isinstance(exc, Qwen3MultimodalRuntimeError):
                raise
            if isinstance(exc, Qwen3MultiSidecarError) and exc.reason_code in {
                "qwen3_multisidecar_handoff_mismatch",
                "qwen3_multisidecar_boundary_mismatch",
            }:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_handoff_mismatch",
                    "MM1 decode hidden handoff failed before the next segment",
                ) from exc
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_execution_failed",
                "MM1 multi-sidecar prefill/decode execution failed",
            ) from exc

    @staticmethod
    def _select_next_token(
        logits: Any, *, sampling: Mapping[str, Any], generator: Any,
        quality_out: dict[str, Any] | None = None,
    ) -> int:
        try:
            import torch

            if (
                not isinstance(logits, torch.Tensor)
                or logits.device.type != "cpu"
                or logits.ndim != 3
                or logits.shape[0] != 1
                or logits.shape[1] <= 0
                or logits.shape[2] <= 0
            ):
                raise ValueError("final logits are not a bounded CPU tensor")
            values = logits[0, -1, :].to(dtype=torch.float32)
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError("final logits contain non-finite values")
            if sampling["mode"] == "greedy":
                if quality_out is not None:
                    quality_out.update({
                        "candidate_count": 1,
                        "entropy": 0.0,
                        "confidence": 1.0,
                        "top_p_cutoff": 1,
                    })
                return int(torch.argmax(values).item())
            temperature = float(sampling["temperature"])
            scaled = values / temperature
            top_k = int(sampling["top_k"])
            if top_k > 0 and top_k < scaled.numel():
                top_values, top_indices = torch.topk(scaled, top_k, dim=-1)
                filtered = torch.full_like(scaled, float("-inf"))
                filtered.scatter_(0, top_indices, top_values)
            else:
                filtered = scaled
            probabilities = torch.softmax(filtered, dim=-1)
            top_p = float(sampling["top_p"])
            if top_p < 1.0:
                sorted_probabilities, sorted_indices = torch.sort(
                    probabilities, descending=True,
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
                probabilities = torch.zeros_like(probabilities)
                probabilities.scatter_(0, sorted_indices, sorted_probabilities)
            total = probabilities.sum()
            if not bool(torch.isfinite(total).item()) or float(total.item()) <= 0.0:
                raise ValueError("sampling probabilities are empty")
            probabilities = probabilities / total
            positive = probabilities > 0
            candidate_count = int(positive.sum().item())
            entropy = float((
                -probabilities[positive] * torch.log(probabilities[positive])
            ).sum().item()) if candidate_count else 0.0
            confidence = float(probabilities.max().item())
            if quality_out is not None:
                quality_out.update({
                    "candidate_count": candidate_count,
                    "entropy": round(entropy, 8),
                    "confidence": round(confidence, 8),
                    "top_p_cutoff": candidate_count,
                })
            if generator is None:
                raise ValueError("sampling generator is unavailable")
            return int(torch.multinomial(probabilities, 1, generator=generator).item())
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("next-token sampling failed") from exc

    def _write_next_decode_input(
        self, logits: Any, *, sampling: Mapping[str, Any] | None = None,
        generator: Any = None, quality_out: dict[str, Any] | None = None,
    ) -> tuple[int, Path]:
        temporary = self.artifact_root / (
            f"{self._retained_prefix}next-tmp-{secrets.token_hex(16)}.pt"
        )
        target = self.artifact_root / (
            f"{self._retained_prefix}next-{secrets.token_hex(16)}.pt"
        )
        try:
            import torch
            active_sampling = sampling or _normalize_mm1_sampling(
                temperature=0.0, top_k=0, top_p=1.0, seed=0,
            )
            token_id = self._select_next_token(
                logits, sampling=active_sampling, generator=generator,
                quality_out=quality_out,
            )
            selected = torch.tensor([[token_id]], dtype=torch.long, device="cpu")
            if token_id < 0:
                raise ValueError("selected token is invalid")
            torch.save(
                {"input_ids": selected},
                str(temporary),
            )
            temporary.replace(target)
            if _file_evidence(target)[0] <= 0:
                raise ValueError("next decode artifact is empty")
            return token_id, target
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_selection_failed",
                "MM1 next-token selection failed in the local data plane",
            ) from exc

    def execute_bounded_decode(
        self,
        *,
        input_ref: str | Path,
        decode_input_ref: str | Path,
        decode_binding: Mapping[str, Any],
        max_new_tokens: int,
        eos_token_ids: Sequence[int] = (),
        cancel_after_step: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        policy_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run bounded local greedy or seeded sampling while token data stays in artifacts."""
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 4096
            or isinstance(eos_token_ids, (str, bytes))
            or not isinstance(eos_token_ids, Sequence)
            or len(eos_token_ids) > 64
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_contract_invalid",
                "MM1 bounded decode limits are invalid",
            )
        try:
            eos_ids = {int(value) for value in eos_token_ids}
        except (TypeError, ValueError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_contract_invalid",
                "MM1 EOS token identifiers are invalid",
            ) from exc
        if any(value < 0 for value in eos_ids) or len(eos_ids) != len(eos_token_ids):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_contract_invalid",
                "MM1 EOS token identifiers must be unique non-negative integers",
            )
        if cancel_after_step is not None and (
            isinstance(cancel_after_step, bool)
            or not isinstance(cancel_after_step, int)
            or not 1 <= cancel_after_step <= max_new_tokens
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_contract_invalid",
                "MM1 bounded decode cancellation step is invalid",
            )
        policy: dict[str, Any] | None = None
        requested_sampling = _normalize_mm1_sampling(
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
        )
        if policy_snapshot is not None:
            policy = validate_mm1_sampling_policy_snapshot(policy_snapshot)
            policy_sampling = policy["sampling"]
            legacy_defaults = _normalize_mm1_sampling(
                temperature=0.0, top_k=0, top_p=1.0, seed=0,
            )
            if requested_sampling == legacy_defaults and policy_sampling != legacy_defaults:
                sampling = dict(policy_sampling)
            elif requested_sampling != policy_sampling:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sampling_policy_mismatch",
                    "MM1 sampling arguments do not match the policy snapshot",
                )
            else:
                sampling = dict(policy_sampling)
        else:
            sampling = requested_sampling
            if sampling["mode"] != "greedy":
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_sampling_policy_required",
                    "MM1 non-greedy sampling requires an explicit policy snapshot",
                )
        decode = validate_mm1_decode_artifact_binding(
            decode_binding,
            staged_contract=self.staged,
            manifest=self.manifest,
        )
        layout = self.staged["input_layout"]
        sequence = decode["sequence"]
        if (
            layout["batch_size"] != 1
            or decode["tensor"]["shape"] != [1, 1]
            or sequence["decode_input_tokens"] != 1
            or sequence["decode_length"] != layout["total_sequence"] + 1
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_contract_invalid",
                "MM1 bounded greedy decode requires one batch and one input token per step",
            )
        input_path = self._local_input(input_ref)
        current_input = self._local_artifact(
            decode_input_ref, decode["input_artifact"],
        )
        retained: Path | None = None
        owned_inputs: set[Path] = set()
        trace_digest = hashlib.sha256()
        quality_digest = hashlib.sha256()
        quality_digest.update(_sampling_digest(sampling).encode("ascii"))
        quality_state = _quality_state()
        sampling_generator = None
        if sampling["mode"] == "multinomial":
            try:
                import torch

                sampling_generator = torch.Generator(device="cpu")
                sampling_generator.manual_seed(sampling["seed"])
            except Exception as exc:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_bounded_decode_contract_invalid",
                    "MM1 local sampling generator could not be initialized",
                ) from exc
        generated_count = 0
        final_quality: dict[str, Any] | None = None
        ledger_records: list[dict[str, Any]] = []
        draw_evidence: list[dict[str, Any]] = []
        ledger_metadata: dict[str, Any] | None = None
        ledger_path: Path | None = None
        consumer = Qwen3MultimodalDecodeArtifactConsumer(
            artifact_root=self.artifact_root,
        )
        try:
            self.chain.prepare()
            self.lifecycle.append("prepare")
            self.chain.commit()
            self.lifecycle.append("commit")
            self.chain.prefill(
                input_ref=input_path,
                batch_size=layout["batch_size"],
                sequence_length=layout["total_sequence"],
            )
            self.lifecycle.append("prefill")
            prefill_reports, _prefill_final, _prefill_evidence = self._validate_prefill()
            current_sequence = int(sequence["decode_length"])
            final_reports: list[dict[str, Any]] = []
            final_path: Path | None = None
            final_evidence: tuple[int, str] | None = None
            stop_reason = "max_new_tokens"
            for step_index in range(1, max_new_tokens + 1):
                owned_current = current_input if current_input in owned_inputs else None
                self.chain.decode(
                    input_ref=current_input,
                    batch_size=1,
                    sequence_length=current_sequence,
                    input_sequence_length=1,
                )
                self.lifecycle.append(f"decode_step_{step_index}")
                if owned_current is not None:
                    owned_current.unlink(missing_ok=True)
                    owned_inputs.discard(owned_current)
                generation = self.staged["generation"] + step_index
                reports, output_path, evidence = self._validate_phase(
                    phase="decode",
                    sequence_length=current_sequence,
                    handoff_sequence_length=1,
                    generation=generation,
                )
                metadata = {
                    "artifact_id": f"mm1step_{step_index}_{evidence[1][:24]}",
                    "size_bytes": evidence[0],
                    "sha256": evidence[1],
                    "status": "committed",
                    "content_kind": "final_decode_output",
                }
                quality, payload = consumer._inspect(
                    artifact_ref=output_path,
                    artifact_metadata=metadata,
                    kv_contract=reports[-1]["kv_contract"],
                    expected_generation=generation,
                    expected_decode_tokens=1,
                    enforce_replay=True,
                )
                if quality["output_kind"] != "logits" or payload.get("logits") is None:
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_bounded_decode_selection_failed",
                        "MM1 final decode segment returned no logits",
                    )
                step_quality: dict[str, Any] = {"step_index": step_index}
                selected_token, next_input = self._write_next_decode_input(
                    payload["logits"], sampling=sampling, generator=sampling_generator,
                    quality_out=step_quality,
                )
                _quality_state_update(
                    quality_state, step_quality, step_index=step_index,
                )
                quality_digest.update(json.dumps({
                    "step_index": step_index,
                    "artifact_sha256": evidence[1],
                    "quality": {
                        key: step_quality[key]
                        for key in (
                            "candidate_count", "entropy", "confidence", "top_p_cutoff",
                        )
                    },
                    "sampling_sha256": _sampling_digest(sampling),
                }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                del payload
                step_summary = {
                    "step_index": step_index,
                    "generation": generation,
                    "sequence_length": current_sequence,
                    "artifact": dict(quality["artifact"]),
                    "logits": dict(quality["logits"]),
                    "kv": dict(quality["kv"]),
                    "full_model_materialized": False,
                }
                trace_digest.update(json.dumps(
                    step_summary,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"))
                generated_count = step_index
                final_quality = step_summary
                ledger_records.append({
                    "step_index": step_index,
                    "generation": generation,
                    "sequence_length": current_sequence,
                    "token_id": selected_token,
                    "artifact_sha256": evidence[1],
                })
                draw_evidence.append({
                    "step_index": step_index,
                    "sha256": _sampling_draw_digest(
                        sampling=sampling,
                        step_index=step_index,
                        artifact_sha256=evidence[1],
                        token_id=selected_token,
                    ),
                })
                if cancel_after_step == step_index:
                    next_input.unlink(missing_ok=True)
                    cancelled = self.chain.cancel()
                    self.lifecycle.append("cancel")
                    self._validate_chain_cleanup(
                        cancelled, self.sessions, expected_phase="aborted",
                    )
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cancelled",
                        "MM1 bounded decode was cancelled after a committed step",
                    )
                should_stop = selected_token in eos_ids or step_index == max_new_tokens
                if should_stop:
                    next_input.unlink(missing_ok=True)
                    stop_reason = "eos" if selected_token in eos_ids else "max_new_tokens"
                    final_reports = reports
                    final_path = output_path
                    final_evidence = evidence
                    break
                owned_inputs.add(next_input)
                current_input = next_input
                current_sequence += 1
            if (
                final_path is None
                or final_evidence is None
                or not final_reports
                or final_quality is None
                or generated_count <= 0
            ):
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_bounded_decode_execution_failed",
                    "MM1 bounded decode produced no final artifact",
                )
            retained = self.artifact_root / (
                f"{self._retained_prefix}{secrets.token_hex(16)}.pt"
            )
            final_path.replace(retained)
            if _file_evidence(retained) != final_evidence:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_artifact_mismatch",
                    "MM1 bounded decode artifact changed while being retained",
                )
            ledger_metadata, ledger_path = _write_mm1_token_ledger(
                artifact_root=self.artifact_root,
                staged=self.staged,
                records=ledger_records,
                stop_reason=stop_reason,
                prefix=self._retained_prefix,
                sampling=sampling,
                draw_evidence=draw_evidence,
                quality_summary=_quality_summary_from_state(
                    sampling=sampling,
                    state=quality_state,
                    quality_sha256=quality_digest.hexdigest(),
                ),
                policy_snapshot=policy,
            )
            released = self.chain.release()
            self.lifecycle.append("release")
            self._validate_chain_cleanup(released, self.sessions, expected_phase="released")
            output_id = f"mm1generated_{final_evidence[1][:32]}"
            assert ledger_path is not None and ledger_metadata is not None
            self._outputs[output_id] = retained
            self._ledger_outputs[ledger_metadata["ledger_id"]] = ledger_path
            self._ledger_metadata[ledger_metadata["ledger_id"]] = dict(ledger_metadata)
            result = {
                "schema_version": 1,
                "status": "multimodal_text_chain_generated",
                "contract_sha256": self.staged["contract_sha256"],
                "artifact_binding_sha256": self.binding["contract_sha256"],
                "decode_binding_sha256": decode["contract_sha256"],
                "text_chain_id": self.staged["text_chain_id"],
                "prefill_generation": self.staged["generation"],
                "final_generation": final_quality["generation"],
                "segment_count": len(self.staged["segment_plan"]),
                "generated_token_count": generated_count,
                "stop_reason": stop_reason,
                "sampling": {
                    "mode": sampling["mode"],
                    "temperature": sampling["temperature"],
                    "top_k": sampling["top_k"],
                    "top_p": sampling["top_p"],
                    "seed": sampling["seed"],
                    "sha256": _sampling_digest(sampling),
                    "draw_count": len(draw_evidence),
                },
                "sampling_quality": _quality_summary_from_state(
                    sampling=sampling,
                    state=quality_state,
                    quality_sha256=quality_digest.hexdigest(),
                ),
                "sampling_policy": ({
                    "policy_id": policy["policy_id"],
                    "policy_version": policy["policy_version"],
                    "snapshot_sha256": policy["snapshot_sha256"],
                    "sampling_sha256": policy["sampling_sha256"],
                    "issued_at": policy["issued_at"],
                    "expires_at": policy["expires_at"],
                    "replay_allowed": policy["replay_allowed"],
                    "route_scope": policy["route_scope"],
                } if policy is not None else {
                    "mode": "legacy_greedy",
                    "implicit_sampling_rejected": True,
                }),
                "lifecycle": list(self.lifecycle),
                "prefill_segment_reports": prefill_reports,
                "decode_trace": {
                    "step_count": generated_count,
                    "first_generation": self.staged["generation"] + 1,
                    "final_generation": final_quality["generation"],
                    "first_sequence_length": sequence["decode_length"],
                    "final_sequence_length": final_quality["sequence_length"],
                    "sha256": trace_digest.hexdigest(),
                },
                "final_decode_quality": final_quality,
                "token_ledger": dict(ledger_metadata),
                "final_artifact": {
                    "artifact_id": output_id,
                    "size_bytes": final_evidence[0],
                    "sha256": final_evidence[1],
                    "status": "committed",
                    "content_kind": "final_decode_output",
                },
                "final_kv_contract": dict(final_reports[-1]["kv_contract"]),
                "sidecar_cleanup_complete": True,
                "artifact_cleanup_required": True,
                "segment_materialized": False,
                "full_model_materialized": False,
            }
            _reject_sensitive(result)
            return result
        except Exception as exc:
            for path in owned_inputs:
                path.unlink(missing_ok=True)
            if ledger_path is not None:
                ledger_path.unlink(missing_ok=True)
            if retained is not None:
                try:
                    retained.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 bounded decode artifact cleanup did not complete",
                    ) from cleanup_exc
            if self.chain.phase not in {"released", "aborted"}:
                try:
                    aborted = self.chain.abort()
                    self.lifecycle.append("abort")
                    self._validate_chain_cleanup(
                        aborted, self.sessions, expected_phase="aborted",
                    )
                except Exception as cleanup_exc:
                    if isinstance(cleanup_exc, Qwen3MultimodalRuntimeError):
                        raise
                    raise Qwen3MultimodalRuntimeError(
                        "qwen3_mm1_multisidecar_cleanup_failed",
                        "MM1 bounded decode abort did not complete",
                    ) from cleanup_exc
            if isinstance(exc, Qwen3MultimodalRuntimeError):
                raise
            if isinstance(exc, Qwen3MultiSidecarError) and exc.reason_code in {
                "qwen3_multisidecar_handoff_mismatch",
                "qwen3_multisidecar_boundary_mismatch",
            }:
                raise Qwen3MultimodalRuntimeError(
                    "qwen3_mm1_multisidecar_handoff_mismatch",
                    "MM1 bounded decode hidden handoff failed",
                ) from exc
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_bounded_decode_execution_failed",
                "MM1 bounded multi-step decode failed",
            ) from exc

    def output_path(self, artifact_id: str) -> Path:
        """Resolve a retained final output for an in-process data-plane consumer."""
        path = self._outputs.get(str(artifact_id))
        if path is None or not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_artifact_missing",
                "MM1 final output artifact is unavailable",
            )
        return path

    def ledger_path(self, ledger_id: str) -> Path:
        """Resolve a retained local token ledger for an isolated tokenizer worker."""
        path = self._ledger_outputs.get(str(ledger_id))
        if path is None or not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_missing", "MM1 token ledger is unavailable",
            )
        return path

    def decode_token_ledger(
        self,
        *,
        ledger_id: str,
        model: str | Path | None,
        timeout_seconds: float = 60.0,
        worker_runner: Any | None = None,
        text_max_bytes: int = 64 * 1024,
    ) -> dict[str, Any]:
        """Decode one retained ledger through the isolated tokenizer worker."""
        metadata = self._ledger_metadata.get(str(ledger_id))
        path = self._ledger_outputs.get(str(ledger_id))
        if metadata is None or path is None or not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_missing", "MM1 token ledger is unavailable",
            )
        from scripts.model_tools.qwen3_token_ledger import (
            run_qwen3_token_ledger_decode,
        )

        return run_qwen3_token_ledger_decode(
            model=(Path(model) if model is not None else None),
            ledger=path,
            ledger_metadata=metadata,
            text_max_bytes=text_max_bytes,
            expected_chain_id=self.staged["text_chain_id"],
            expected_generation=self.staged["generation"],
            expected_first_sequence=self.staged["input_layout"]["total_sequence"] + 1,
            timeout_seconds=timeout_seconds,
            worker_runner=worker_runner,
        )

    def replay_token_ledger(
        self,
        ledger_id: str,
        *,
        expected_sampling_sha256: str = "",
        expected_quality_sha256: str = "",
        expected_policy_snapshot_sha256: str = "",
        timeout_seconds: float = 60,
        worker_runner: Any = None,
    ) -> dict[str, Any]:
        """Validate a retained ledger through the isolated read-only replay boundary."""
        metadata = self._ledger_metadata.get(str(ledger_id))
        path = self._ledger_outputs.get(str(ledger_id))
        if metadata is None or path is None or not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_token_ledger_missing", "MM1 token ledger is unavailable",
            )
        from scripts.model_tools.qwen3_token_ledger import (
            run_qwen3_token_ledger_replay,
        )

        return run_qwen3_token_ledger_replay(
            ledger=path,
            ledger_metadata=metadata,
            expected_chain_id=self.staged["text_chain_id"],
            expected_generation=self.staged["generation"],
            expected_first_sequence=self.staged["input_layout"]["total_sequence"] + 1,
            expected_sampling_sha256=str(expected_sampling_sha256 or ""),
            expected_quality_sha256=str(expected_quality_sha256 or ""),
            expected_policy_snapshot_sha256=str(expected_policy_snapshot_sha256 or ""),
            timeout_seconds=timeout_seconds,
            worker_runner=worker_runner,
        )

    def cleanup(self, reason_code: str = "completed") -> dict[str, Any]:
        """Remove retained outputs and abort an unfinished chain."""
        reason = str(reason_code or "")
        if not reason or len(reason) > 128 or any(char in reason for char in "/\\\x00"):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_multisidecar_contract_invalid", "cleanup reason is invalid",
            )
        if self.chain.phase not in {"released", "aborted"}:
            aborted = self.chain.abort()
            self.lifecycle.append("abort")
            self._validate_chain_cleanup(aborted, self.sessions, expected_phase="aborted")
        removed = self._remove_retained()
        return {
            "completed": True,
            "reason_code": reason,
            "removed_artifacts": removed,
            "retained_artifacts": 0,
            "segment_materialized": False,
            "full_model_materialized": False,
        }

    def recover_after_restart(self) -> dict[str, Any]:
        """Abort sessions and remove generic plus MM1-specific chain artifacts."""
        recovered = self.chain.recover_after_restart()
        self.lifecycle.append("recover")
        self._validate_chain_cleanup(recovered, self.sessions, expected_phase="aborted")
        removed = self._remove_retained()
        return {
            "completed": True,
            "reason_code": "restart_recovery",
            "removed_retained_artifacts": removed,
            "segment_materialized": False,
            "full_model_materialized": False,
        }


class Qwen3MultimodalDecodeArtifactConsumer:
    """Validate and summarize one retained MM1 decode artifact locally."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        artifact_loader: Any | None = None,
    ) -> None:
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_root_missing",
                "MM1 decode consumer artifact root is unavailable",
            )
        if artifact_loader is not None and not callable(artifact_loader):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode artifact loader must be callable",
            )
        self.artifact_root = root
        self._loader = artifact_loader
        self._consumed_sha256: set[str] = set()

    def _local_artifact(self, value: str | Path) -> Path:
        path = Path(value).expanduser().absolute().resolve(strict=False)
        try:
            path.relative_to(self.artifact_root)
        except ValueError as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_scope_invalid",
                "MM1 decode artifact is outside the local artifact root",
            ) from exc
        if not path.is_file():
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_artifact_missing",
                "MM1 decode artifact is unavailable",
            )
        return path

    @staticmethod
    def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode artifact metadata is invalid",
            )
        artifact_id = _decode_identifier(value.get("artifact_id"), "artifact_id")
        size_bytes = value.get("size_bytes")
        sha256 = str(value.get("sha256") or "").lower()
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or value.get("status") != "committed"
            or value.get("content_kind") != "final_decode_output"
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode artifact metadata does not describe a committed output",
            )
        return {
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": "committed",
            "content_kind": "final_decode_output",
        }

    def _load(self, path: Path) -> Any:
        try:
            if self._loader is not None:
                return self._loader(path)
            import torch

            return torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_load_failed",
                "MM1 decode artifact could not be loaded locally",
            ) from exc

    @staticmethod
    def _tensor_summary(value: Any, label: str) -> dict[str, Any] | None:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        dtype = _dtype(getattr(value, "dtype", None))
        device = _device(getattr(value, "device", None))
        try:
            canonical_shape = [int(item) for item in shape]
        except (TypeError, ValueError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_tensor_invalid",
                f"MM1 decode {label} tensor shape is invalid",
            ) from exc
        if (
            len(canonical_shape) != 3
            or any(dimension <= 0 for dimension in canonical_shape)
            or not dtype
            or device != "cpu"
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_tensor_invalid",
                f"MM1 decode {label} tensor metadata is invalid",
            )
        finite: bool | None = None
        try:
            import torch

            finite = bool(torch.isfinite(value).all().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            finite = None
        if finite is False:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_tensor_invalid",
                f"MM1 decode {label} tensor contains a non-finite value",
            )
        return {
            "shape": canonical_shape,
            "dtype": dtype,
            "device": device,
            "finite": finite,
        }

    @staticmethod
    def _cache_sequence_length(cache: Any, expected: int) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            try:
                value = int(getter())
                if value == expected:
                    return value
            except (IndexError, TypeError, ValueError):
                pass
        candidates: list[int] = []

        def visit(value: Any) -> None:
            shape = getattr(value, "shape", None)
            if shape is not None:
                try:
                    candidates.extend(int(item) for item in shape)
                except (TypeError, ValueError):
                    return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(cache)
        if expected in candidates:
            return expected
        raise Qwen3MultimodalRuntimeError(
            "qwen3_mm1_decode_consume_kv_invalid",
            "MM1 decode KV sequence length does not match its contract",
        )

    @staticmethod
    def _kv_contract(
        value: Mapping[str, Any], *, expected_generation: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode KV contract is invalid",
            )
        try:
            generation = int(value.get("generation"))
            sequence_length = int(value.get("sequence_length"))
            batch_size = int(value.get("batch_size"))
            layer_range = [int(item) for item in value.get("layer_range", [])]
        except (TypeError, ValueError) as exc:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode KV dimensions are invalid",
            ) from exc
        if (
            value.get("phase") != "decode"
            or generation != expected_generation
            or sequence_length <= 0
            or batch_size <= 0
            or len(layer_range) != 2
            or layer_range[0] < 0
            or layer_range[1] <= layer_range[0]
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode KV contract does not match the requested generation",
            )
        return {
            "generation": generation,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "layer_count": layer_range[1] - layer_range[0],
            "dtype": _dtype(value.get("dtype")),
            "device": _device(value.get("device")),
        }

    def _inspect(
        self,
        *,
        artifact_ref: str | Path,
        artifact_metadata: Mapping[str, Any],
        kv_contract: Mapping[str, Any],
        expected_generation: int,
        expected_decode_tokens: int,
        enforce_replay: bool,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        """Inspect locally, optionally recording replay state for public consumption."""
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
            or isinstance(expected_decode_tokens, bool)
            or not isinstance(expected_decode_tokens, int)
            or expected_decode_tokens <= 0
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_contract_invalid",
                "MM1 decode consumer expectations are invalid",
            )
        metadata = self._metadata(artifact_metadata)
        if enforce_replay and metadata["sha256"] in self._consumed_sha256:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_repeated",
                "MM1 decode artifact was already consumed",
            )
        canonical_kv = self._kv_contract(
            kv_contract, expected_generation=expected_generation,
        )
        path = self._local_artifact(artifact_ref)
        if _file_evidence(path) != (metadata["size_bytes"], metadata["sha256"]):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_artifact_mismatch",
                "MM1 decode artifact evidence does not match its metadata",
            )
        payload = self._load(path)
        if not isinstance(payload, Mapping) or set(payload) != {
            "hidden_states", "logits", "past_key_values",
        }:
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_payload_invalid",
                "MM1 decode artifact payload does not match the sidecar schema",
            )
        logits = self._tensor_summary(payload.get("logits"), "logits")
        hidden = self._tensor_summary(payload.get("hidden_states"), "hidden_states")
        if (logits is None) == (hidden is None):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_payload_invalid",
                "MM1 decode artifact must contain exactly one output tensor",
            )
        output = logits if logits is not None else hidden
        assert output is not None
        if (
            output["shape"][0] != canonical_kv["batch_size"]
            or output["shape"][1] != expected_decode_tokens
            or output["dtype"] != canonical_kv["dtype"]
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_tensor_invalid",
                "MM1 decode output metadata does not match the decode contract",
            )
        cache = payload.get("past_key_values")
        if (
            cache is None
            or not isinstance(cache, (list, tuple))
            or len(cache) != canonical_kv["layer_count"]
        ):
            raise Qwen3MultimodalRuntimeError(
                "qwen3_mm1_decode_consume_kv_invalid",
                "MM1 decode artifact contains no bounded KV cache",
            )
        cache_sequence = self._cache_sequence_length(
            cache, canonical_kv["sequence_length"],
        )
        result = {
            "schema_version": 1,
            "status": "decode_artifact_consumed",
            "artifact": metadata,
            "decode_generation": expected_generation,
            "output_kind": "logits" if logits is not None else "hidden_states",
            "token_count": expected_decode_tokens,
            "batch_size": canonical_kv["batch_size"],
            "logits": logits,
            "hidden_states": hidden,
            "kv": {
                "present": True,
                "layer_count": len(cache),
                "sequence_length": cache_sequence,
                "dtype": canonical_kv["dtype"],
                "device": canonical_kv["device"],
            },
            "segment_materialized": False,
            "full_model_materialized": False,
        }
        _reject_sensitive(result)
        if enforce_replay:
            self._consumed_sha256.add(metadata["sha256"])
        return result, payload

    def consume(
        self,
        *,
        artifact_ref: str | Path,
        artifact_metadata: Mapping[str, Any],
        kv_contract: Mapping[str, Any],
        expected_generation: int,
        expected_decode_tokens: int,
    ) -> dict[str, Any]:
        """Return a path-free quality summary without exporting artifact tensors."""
        result, _payload = self._inspect(
            artifact_ref=artifact_ref,
            artifact_metadata=artifact_metadata,
            kv_contract=kv_contract,
            expected_generation=expected_generation,
            expected_decode_tokens=expected_decode_tokens,
            enforce_replay=True,
        )
        return result

    def reset(self) -> None:
        """Forget in-process replay state without touching retained artifacts."""
        self._consumed_sha256.clear()


__all__ = [
    "Qwen3MultimodalDecodeArtifactConsumer",
    "Qwen3MultimodalMultiSidecarAdapter",
    "build_mm1_decode_artifact_binding",
    "build_mm1_sampling_policy_snapshot",
    "validate_mm1_decode_artifact_binding",
    "validate_mm1_sampling_policy_snapshot",
]
