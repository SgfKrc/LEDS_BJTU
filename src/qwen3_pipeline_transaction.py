"""Fail-closed Qwen3 pipeline prepare/commit protocol simulation.

The production scheduler still admits only the established Qwen/Qwen2
runtime.  This module freezes the Qwen3 control-plane contract and exercises
the same prepare -> commit -> ready / abort -> release lifecycle without
sending network messages or materializing model weights.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Iterable


SCHEMA_VERSION = 1
MAX_CONTRACT_BYTES = 64 * 1024
MAX_ACK_BYTES = 8 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_ASSIGNMENT_PROBE_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9._/\-]{1,512}$")
_ACTIVE_PHASES = {"preparing", "committing"}
_BANNED_KEYS = {
    "prompt", "messages", "input_ids", "hidden_states", "past_key_values",
    "logits", "weights", "tensor", "tensors", "tokenizer_output",
}


class Qwen3PipelineProtocolError(ValueError):
    """The Qwen3 dry-run contract or transition is not admissible."""


def _canonical_bytes(value: Any, *, maximum: int, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3PipelineProtocolError(f"{label} is not JSON serializable") from exc
    if len(encoded) > maximum:
        raise Qwen3PipelineProtocolError(f"{label} exceeds serialization limit")
    return encoded


def _digest(value: Any, *, maximum: int = MAX_CONTRACT_BYTES, label: str = "contract") -> str:
    return hashlib.sha256(_canonical_bytes(value, maximum=maximum, label=label)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise Qwen3PipelineProtocolError(f"{label} must be a lowercase SHA-256")
    return normalized


def _reject_sensitive_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _BANNED_KEYS:
                raise Qwen3PipelineProtocolError(
                    f"Qwen3 dry-run contract cannot contain {key}"
                )
            _reject_sensitive_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_fields(item)


def build_qwen3_dry_run_contract(
    *,
    config_id: str,
    plan_id: str,
    generation: int,
    model_id: str,
    model_sha256: str,
    total_layers: int,
    hidden_size: int,
    segments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build one canonical two/three-node Qwen3 transaction contract."""
    config_id = str(config_id or "")
    plan_id = str(plan_id or "")
    model_id = str(model_id or "")
    if not config_id or not plan_id or not model_id:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run identity is incomplete")
    try:
        generation = int(generation)
        total_layers = int(total_layers)
        hidden_size = int(hidden_size)
    except (TypeError, ValueError) as exc:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run dimensions are invalid") from exc
    if generation <= 0 or total_layers <= 0 or hidden_size <= 0:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run dimensions are invalid")
    model_sha256 = _require_sha256(model_sha256, "model_sha256")
    raw_segments = list(segments)
    if len(raw_segments) not in {2, 3}:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run requires two or three segments")

    normalized: list[dict[str, Any]] = []
    expected_start = 0
    node_ids: set[str] = set()
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise Qwen3PipelineProtocolError("Qwen3 segment must be an object")
        _reject_sensitive_fields(raw)
        node_id = str(raw.get("node_id", "") or "")
        layer_range = raw.get("layer_range")
        if not node_id or node_id in node_ids:
            raise Qwen3PipelineProtocolError("Qwen3 dry-run node IDs must be unique")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            raise Qwen3PipelineProtocolError("Qwen3 segment layer_range is invalid")
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
            required_bytes = int(raw.get("required_bytes", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise Qwen3PipelineProtocolError("Qwen3 segment dimensions are invalid") from exc
        if start != expected_start or end <= start or end > total_layers:
            raise Qwen3PipelineProtocolError(
                "Qwen3 dry-run segments must be contiguous and in bounds"
            )
        if required_bytes <= 0:
            raise Qwen3PipelineProtocolError("Qwen3 segment required_bytes is invalid")
        has_embedding = bool(raw.get("has_embedding", False))
        has_lm_head = bool(raw.get("has_lm_head", False))
        if has_embedding != (index == 0):
            raise Qwen3PipelineProtocolError("only the first Qwen3 segment owns embedding")
        if has_lm_head != (index == len(raw_segments) - 1):
            raise Qwen3PipelineProtocolError("only the last Qwen3 segment owns LM Head")
        segment = {
            "segment_index": index,
            "node_id": node_id,
            "layer_range": [start, end],
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "assignment_manifest_sha256": _require_sha256(
                raw.get("assignment_manifest_sha256"),
                "assignment_manifest_sha256",
            ),
            "required_bytes": required_bytes,
            "execution_device": str(raw.get("execution_device", "cpu") or "cpu"),
            "dtype": str(raw.get("dtype", "float32") or "float32"),
        }
        if segment["execution_device"] not in {"cpu", "cuda"}:
            raise Qwen3PipelineProtocolError("Qwen3 segment execution_device is invalid")
        if segment["dtype"] not in {"float32", "float16", "bfloat16"}:
            raise Qwen3PipelineProtocolError("Qwen3 segment dtype is invalid")
        raw_probe = raw.get("assignment_probe")
        if raw_probe is not None:
            if not isinstance(raw_probe, dict):
                raise Qwen3PipelineProtocolError("Qwen3 assignment probe is invalid")
            relative_path = str(raw_probe.get("relative_path", "") or "").replace(
                "\\", "/",
            )
            try:
                file_size = int(raw_probe.get("file_size", 0) or 0)
                offset = int(raw_probe.get("offset", 0) or 0)
                length = int(raw_probe.get("length", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise Qwen3PipelineProtocolError(
                    "Qwen3 assignment probe dimensions are invalid"
                ) from exc
            if (
                not _SAFE_RELATIVE_PATH.fullmatch(relative_path)
                or relative_path.startswith("/")
                or relative_path.startswith("../")
                or "/../" in relative_path
            ):
                raise Qwen3PipelineProtocolError(
                    "Qwen3 assignment probe path is unsafe"
                )
            if (
                file_size <= 0
                or offset < 0
                or length <= 0
                or length > MAX_ASSIGNMENT_PROBE_BYTES
                or offset + length > file_size
            ):
                raise Qwen3PipelineProtocolError(
                    "Qwen3 assignment probe range is invalid"
                )
            segment["assignment_probe"] = {
                "relative_path": relative_path,
                "file_size": file_size,
                "offset": offset,
                "length": length,
                "sha256": _require_sha256(
                    raw_probe.get("sha256"), "assignment_probe.sha256",
                ),
            }
        segment["segment_sha256"] = _digest(segment, label="segment contract")
        normalized.append(segment)
        node_ids.add(node_id)
        expected_start = end
    if expected_start != total_layers:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run does not cover all layers")

    handoffs: list[dict[str, Any]] = []
    for left, right in zip(normalized, normalized[1:]):
        handoff = {
            "from_segment": left["segment_index"],
            "to_segment": right["segment_index"],
            "from_node_id": left["node_id"],
            "to_node_id": right["node_id"],
            "rank": 3,
            "hidden_size": hidden_size,
            "transport_device": "cpu",
            "transport_dtype": left["dtype"],
        }
        handoff["handoff_sha256"] = _digest(handoff, label="hidden handoff contract")
        handoffs.append(handoff)

    kv_contracts = []
    for segment in normalized:
        kv = {
            "segment_index": segment["segment_index"],
            "node_id": segment["node_id"],
            "layer_range": segment["layer_range"],
            "ownership": "node_local",
            "phases": ["prefill", "decode"],
            "generation_required": True,
            "sequence_length_required": True,
            "release_on_abort": True,
        }
        kv["kv_contract_sha256"] = _digest(kv, label="KV contract")
        kv_contracts.append(kv)

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": "qwen3_pipeline_dry_run",
        "dry_run": True,
        "config_id": config_id,
        "plan_id": plan_id,
        "generation": generation,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "model_type": "qwen3",
        "total_layers": total_layers,
        "hidden_size": hidden_size,
        "segments": normalized,
        "hidden_handoffs": handoffs,
        "kv_contracts": kv_contracts,
        "full_model_fallback": False,
        "network_dispatch": False,
        "weight_materialization": False,
    }
    contract["contract_sha256"] = _digest(contract)
    _canonical_bytes(contract, maximum=MAX_CONTRACT_BYTES, label="Qwen3 dry-run contract")
    return contract


def validate_qwen3_dry_run_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a received contract and prove every digest is canonical."""
    if not isinstance(contract, dict):
        raise Qwen3PipelineProtocolError("Qwen3 dry-run contract must be an object")
    _reject_sensitive_fields(contract)
    expected_sha = _require_sha256(contract.get("contract_sha256"), "contract_sha256")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if _digest(payload) != expected_sha:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run contract digest mismatch")
    rebuilt = build_qwen3_dry_run_contract(
        config_id=contract.get("config_id", ""),
        plan_id=contract.get("plan_id", ""),
        generation=contract.get("generation", 0),
        model_id=contract.get("model_id", ""),
        model_sha256=contract.get("model_sha256", ""),
        total_layers=contract.get("total_layers", 0),
        hidden_size=contract.get("hidden_size", 0),
        segments=contract.get("segments", []),
    )
    if rebuilt != contract:
        raise Qwen3PipelineProtocolError("Qwen3 dry-run contract is not canonical")
    return rebuilt


class Qwen3PipelineDryRunTransaction:
    """In-memory, no-I/O simulation of Qwen3 C2 transaction semantics."""

    def __init__(
        self,
        contract: dict[str, Any],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        now: float | None = None,
        network_dispatch: bool = False,
    ) -> None:
        self.contract = validate_qwen3_dry_run_contract(contract)
        self.network_dispatch = bool(network_dispatch)
        try:
            self.timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise Qwen3PipelineProtocolError("Qwen3 dry-run timeout is invalid") from exc
        if not 0 < self.timeout_seconds <= 300:
            raise Qwen3PipelineProtocolError("Qwen3 dry-run timeout is invalid")
        self.phase = "preparing"
        self.reason_code = ""
        self.reason = ""
        self.prepared_nodes: set[str] = set()
        self.ready_nodes: set[str] = set()
        self.released_nodes: set[str] = set()
        self.retry_counts: dict[str, int] = {}
        self._ack_digests: dict[tuple[str, str], str] = {}
        self._segments = {
            segment["node_id"]: segment for segment in self.contract["segments"]
        }
        self._kv_contracts = {
            item["node_id"]: item for item in self.contract["kv_contracts"]
        }
        self._handoffs = {
            node_id: sorted(
                item["handoff_sha256"]
                for item in self.contract["hidden_handoffs"]
                if node_id in {item["from_node_id"], item["to_node_id"]}
            )
            for node_id in self._segments
        }
        self.deadline = float(time.monotonic() if now is None else now) + self.timeout_seconds

    @property
    def worker_ids(self) -> set[str]:
        return set(self._segments)

    def _message(self, node_id: str, phase: str) -> dict[str, Any]:
        segment = self._segments[node_id]
        message = {
            "schema_version": SCHEMA_VERSION,
            "operation": "qwen3_pipeline_dry_run",
            "dry_run": True,
            "phase": phase,
            "node_id": node_id,
            "config_id": self.contract["config_id"],
            "plan_id": self.contract["plan_id"],
            "generation": self.contract["generation"],
            "contract_sha256": self.contract["contract_sha256"],
            "model_id": self.contract["model_id"],
            "model_sha256": self.contract["model_sha256"],
            "segment_index": segment["segment_index"],
            "layer_range": segment["layer_range"],
            "total_layers": self.contract["total_layers"],
            "has_embedding": segment["has_embedding"],
            "has_lm_head": segment["has_lm_head"],
            "segment_sha256": segment["segment_sha256"],
            "assignment_manifest_sha256": segment["assignment_manifest_sha256"],
            "kv_contract_sha256": self._kv_contracts[node_id]["kv_contract_sha256"],
            "hidden_handoff_sha256": self._handoffs[node_id],
            "required_bytes": segment["required_bytes"],
            "execution_device": segment["execution_device"],
            "dtype": segment["dtype"],
            "full_model_fallback": False,
            "network_dispatch": self.network_dispatch,
            "loopback_only": self.network_dispatch,
            "weight_materialization": False,
        }
        if "assignment_probe" in segment:
            message["assignment_probe"] = dict(segment["assignment_probe"])
        return message

    def prepare_messages(self) -> list[dict[str, Any]]:
        if self.phase != "preparing":
            raise Qwen3PipelineProtocolError("Qwen3 dry-run is not preparing")
        return [self._message(node_id, "prepare") for node_id in sorted(self.worker_ids)]

    def _abort(self, reason_code: str, reason: str) -> dict[str, Any]:
        if self.phase == "released":
            raise Qwen3PipelineProtocolError("completed Qwen3 dry-run cannot be aborted")
        self.phase = "aborted"
        self.reason_code = str(reason_code)
        self.reason = str(reason)[:1024]
        self.prepared_nodes.clear()
        self.ready_nodes.clear()
        return {
            "accepted": False,
            "phase": self.phase,
            "outbound": self.release_messages(),
        }

    def abort(self, reason_code: str, reason: str = "") -> dict[str, Any]:
        return self._abort(reason_code, reason)

    def release(self) -> dict[str, Any]:
        if self.phase != "ready":
            raise Qwen3PipelineProtocolError(
                "only a ready Qwen3 dry-run can be released normally"
            )
        self.phase = "releasing"
        return {
            "accepted": True,
            "phase": self.phase,
            "outbound": self.release_messages(),
        }

    def release_messages(self) -> list[dict[str, Any]]:
        if self.phase not in {"aborted", "releasing"}:
            return []
        messages = []
        for node_id in sorted(self.worker_ids - self.released_nodes):
            message = {**self._message(node_id, "release"), "release": True}
            if self.phase == "aborted":
                message.update({
                    "abort": True,
                    "reason_code": self.reason_code,
                })
            messages.append(message)
        return messages

    def handle_ack(self, node_id: str, ack: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        node_id = str(node_id or "")
        if node_id not in self.worker_ids or not isinstance(ack, dict):
            if self.phase not in _ACTIVE_PHASES:
                raise Qwen3PipelineProtocolError("Qwen3 dry-run is not accepting ACKs")
            return self._abort("qwen3_ack_node_mismatch", "ACK node is not in transaction")
        try:
            ack_digest = _digest(ack, maximum=MAX_ACK_BYTES, label="Qwen3 dry-run ACK")
        except Qwen3PipelineProtocolError as exc:
            if self.phase not in _ACTIVE_PHASES:
                raise
            return self._abort("qwen3_ack_oversize", str(exc))
        incoming_phase = str(ack.get("phase", "") or "")
        incoming_replay_key = (incoming_phase, node_id)
        previous = self._ack_digests.get(incoming_replay_key)
        if previous is not None:
            if previous == ack_digest:
                return {
                    "accepted": True,
                    "duplicate": True,
                    "phase": self.phase,
                    "outbound": [],
                }
            if self.phase not in _ACTIVE_PHASES:
                raise Qwen3PipelineProtocolError(
                    "completed Qwen3 dry-run received a changed duplicate ACK"
                )
            return self._abort("qwen3_ack_replay_mismatch", "duplicate ACK payload changed")
        if self.phase not in _ACTIVE_PHASES:
            raise Qwen3PipelineProtocolError("Qwen3 dry-run is not accepting ACKs")
        expected_phase = "prepare" if self.phase == "preparing" else "commit"
        replay_key = (expected_phase, node_id)
        segment = self._segments[node_id]
        expected = {
            "schema_version": SCHEMA_VERSION,
            "operation": "qwen3_pipeline_dry_run_ack",
            "dry_run": True,
            "phase": expected_phase,
            "node_id": node_id,
            "config_id": self.contract["config_id"],
            "plan_id": self.contract["plan_id"],
            "generation": self.contract["generation"],
            "contract_sha256": self.contract["contract_sha256"],
            "model_sha256": self.contract["model_sha256"],
            "segment_sha256": segment["segment_sha256"],
            "assignment_manifest_sha256": segment["assignment_manifest_sha256"],
            "kv_contract_sha256": self._kv_contracts[node_id]["kv_contract_sha256"],
            "hidden_handoff_sha256": self._handoffs[node_id],
            "layer_range": segment["layer_range"],
            "status": "prepared" if expected_phase == "prepare" else "ready",
            "full_model_materialized": False,
        }
        if any(ack.get(key) != value for key, value in expected.items()):
            return self._abort("qwen3_ack_contract_mismatch", "ACK does not match dry-run contract")
        if expected_phase == "prepare":
            try:
                available_bytes = int(ack.get("available_bytes", 0) or 0)
            except (TypeError, ValueError):
                available_bytes = 0
            if available_bytes < segment["required_bytes"]:
                return self._abort("qwen3_prepare_capacity_changed", "worker capacity changed after plan")
            probe = segment.get("assignment_probe")
            if probe is not None:
                report = ack.get("assignment_probe")
                manifest_report = ack.get("assignment_manifest")
                try:
                    bytes_received = int(report.get("bytes_received", -1))
                    attempts = int(report.get("attempts", 0))
                    manifest_bytes = int(
                        manifest_report.get("bytes_received", 0)
                    )
                except (AttributeError, TypeError, ValueError):
                    bytes_received, attempts, manifest_bytes = -1, 0, 0
                expected_report = {
                    "relative_path": probe["relative_path"],
                    "offset": probe["offset"],
                    "length": probe["length"],
                    "sha256": probe["sha256"],
                    "content_range": (
                        f"bytes {probe['offset']}-"
                        f"{probe['offset'] + probe['length'] - 1}/"
                        f"{probe['file_size']}"
                    ),
                }
                if (
                    not isinstance(manifest_report, dict)
                    or manifest_report.get("sha256")
                    != segment["assignment_manifest_sha256"]
                    or not 1 <= manifest_bytes <= 2 * 1024 * 1024
                ):
                    return self._abort(
                        "qwen3_assignment_manifest_mismatch",
                        "prepare ACK assignment manifest does not match segment contract",
                    )
                if (
                    not isinstance(report, dict)
                    or any(report.get(key) != value for key, value in expected_report.items())
                    or bytes_received != probe["length"]
                    or not 1 <= attempts <= 3
                ):
                    return self._abort(
                        "qwen3_assignment_probe_mismatch",
                        "prepare ACK assignment probe does not match segment contract",
                    )
        else:
            expected_kv_probe = {
                "segment_index": segment["segment_index"],
                "layer_range": segment["layer_range"],
                "cache_generation": self.contract["generation"],
                "sequence_length": 0,
                "dtype": segment["dtype"],
                "device": segment["execution_device"],
                "phase": "empty",
                "cleared": True,
            }
            if ack.get("kv_cache_probe") != expected_kv_probe:
                return self._abort(
                    "qwen3_kv_contract_mismatch",
                    "commit ACK KV cache probe does not match segment contract",
                )
        self._ack_digests[replay_key] = ack_digest
        current = float(time.monotonic() if now is None else now)
        self.deadline = current + self.timeout_seconds
        if expected_phase == "prepare":
            self.prepared_nodes.add(node_id)
            if self.prepared_nodes == self.worker_ids:
                self.phase = "committing"
                return {
                    "accepted": True,
                    "phase": self.phase,
                    "outbound": [self._message(worker, "commit") for worker in sorted(self.worker_ids)],
                }
        else:
            self.ready_nodes.add(node_id)
            if self.ready_nodes == self.worker_ids:
                self.phase = "ready"
        return {"accepted": True, "phase": self.phase, "outbound": []}

    def retry_messages(self) -> list[dict[str, Any]]:
        if self.phase == "preparing":
            pending = self.worker_ids - self.prepared_nodes
            phase = "prepare"
        elif self.phase == "committing":
            pending = self.worker_ids - self.ready_nodes
            phase = "commit"
        else:
            return []
        messages = []
        for node_id in sorted(pending):
            self.retry_counts[node_id] = self.retry_counts.get(node_id, 0) + 1
            message = self._message(node_id, phase)
            message["retry_count"] = self.retry_counts[node_id]
            messages.append(message)
        return messages

    def expire(self, *, now: float | None = None) -> dict[str, Any] | None:
        if self.phase not in _ACTIVE_PHASES:
            return None
        current = float(time.monotonic() if now is None else now)
        if current < self.deadline:
            return None
        return self._abort("qwen3_transaction_timeout", f"{self.phase} deadline expired")

    def disconnect(self, node_id: str) -> dict[str, Any] | None:
        disconnectible = set(_ACTIVE_PHASES)
        if self.network_dispatch:
            disconnectible.update({"ready", "releasing"})
        if self.phase not in disconnectible or node_id not in self.worker_ids:
            return None
        return self._abort("qwen3_worker_disconnected", f"worker {node_id} disconnected")

    def release_ack(self, node_id: str, payload: dict[str, Any]) -> bool:
        if self.phase not in {"aborted", "releasing"} or node_id not in self.worker_ids:
            return False
        expected = self._message(node_id, "release")
        required = {
            "node_id": node_id,
            "config_id": expected["config_id"],
            "plan_id": expected["plan_id"],
            "generation": expected["generation"],
            "contract_sha256": expected["contract_sha256"],
            "phase": "release",
            "status": "released",
            "release": True,
        }
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in required.items()):
            return False
        self.released_nodes.add(node_id)
        if self.released_nodes == self.worker_ids:
            self.phase = "released"
            self._ack_digests.clear()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "phase": self.phase,
            "config_id": self.contract["config_id"],
            "plan_id": self.contract["plan_id"],
            "generation": self.contract["generation"],
            "contract_sha256": self.contract["contract_sha256"],
            "worker_ids": sorted(self.worker_ids),
            "prepared_nodes": sorted(self.prepared_nodes),
            "ready_nodes": sorted(self.ready_nodes),
            "released_nodes": sorted(self.released_nodes),
            "reason_code": self.reason_code,
            "reason": self.reason,
            "network_dispatch": self.network_dispatch,
            "loopback_only": self.network_dispatch,
            "weight_materialization": False,
            "full_model_fallback": False,
        }


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_ACK_BYTES",
    "MAX_ASSIGNMENT_PROBE_BYTES",
    "MAX_CONTRACT_BYTES",
    "Qwen3PipelineDryRunTransaction",
    "Qwen3PipelineProtocolError",
    "build_qwen3_dry_run_contract",
    "validate_qwen3_dry_run_contract",
]
