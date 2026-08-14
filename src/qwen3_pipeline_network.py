"""Active-contract network handoffs for a Qwen3 sidecar chain."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Protocol

from qwen3_pipeline_data_plane import Qwen3ArtifactTransferRuntime
from qwen3_pipeline_peer_auth import Qwen3PeerRequestSigner
from qwen3_pipeline_transaction import validate_qwen3_dry_run_contract
from qwen3_pipeline_transfer import (
    MAX_TRANSFER_CHUNK_BYTES,
    Qwen3ArtifactTransferClient,
    Qwen3TransferError,
    TransferRequester,
)


QWEN3_ARTIFACT_REFERENCE_SCHEMA_VERSION = 1


class Qwen3NetworkError(RuntimeError):
    """A network handoff is outside its active canonical contract."""

    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)[:128]
        self.reason = str(reason)[:1024]
        super().__init__(self.reason)


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _inside(root: Path, value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser().absolute().resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Qwen3NetworkError(
            "qwen3_network_artifact_scope", "network artifact escapes the chain root",
        ) from exc
    if must_exist and not path.is_file():
        raise Qwen3NetworkError(
            "qwen3_network_artifact_missing", "network artifact is unavailable",
        )
    return path


def _reference(
    *,
    mode: str,
    artifact_id: str,
    source_node_id: str,
    target_node_id: str,
    chain_id: str,
    generation: int,
    phase: str,
    from_segment: int,
    to_segment: int,
    size_bytes: int,
    sha256: str,
    status: str = "committed",
) -> dict[str, Any]:
    value = {
        "schema_version": QWEN3_ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "mode": str(mode),
        "artifact_id": str(artifact_id),
        "source_node_id": str(source_node_id),
        "target_node_id": str(target_node_id),
        "chain_id": str(chain_id),
        "generation": int(generation),
        "phase": str(phase),
        "from_segment": int(from_segment),
        "to_segment": int(to_segment),
        "size_bytes": int(size_bytes),
        "sha256": str(sha256),
        "status": str(status),
        "full_model_materialized": False,
    }
    validate_qwen3_artifact_reference(value)
    return value


def validate_qwen3_artifact_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version", "mode", "artifact_id", "source_node_id",
        "target_node_id", "chain_id", "generation", "phase",
        "from_segment", "to_segment", "size_bytes", "sha256", "status",
        "full_model_materialized",
    }
    result = dict(value) if isinstance(value, Mapping) else {}
    if set(result) != expected:
        raise Qwen3NetworkError(
            "qwen3_network_reference_invalid", "artifact reference fields are invalid",
        )
    if (
        result.get("schema_version") != QWEN3_ARTIFACT_REFERENCE_SCHEMA_VERSION
        or result.get("mode") not in {"local", "network"}
        or not isinstance(result.get("artifact_id"), str)
        or not result["artifact_id"]
        or not isinstance(result.get("source_node_id"), str)
        or not result["source_node_id"]
        or not isinstance(result.get("target_node_id"), str)
        or not result["target_node_id"]
        or not isinstance(result.get("chain_id"), str)
        or len(result["chain_id"]) != 64
        or any(character not in "0123456789abcdef" for character in result["chain_id"])
        or result.get("phase") not in {"prefill", "decode"}
        or not all(
            isinstance(result.get(key), int) and not isinstance(result.get(key), bool)
            for key in ("generation", "from_segment", "to_segment", "size_bytes")
        )
        or result["generation"] < 0
        or result["from_segment"] not in {0, 1}
        or result["to_segment"] != result["from_segment"] + 1
        or result["size_bytes"] <= 0
        or not isinstance(result.get("sha256"), str)
        or len(result["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in result["sha256"])
        or result.get("status") != "committed"
        or result.get("full_model_materialized") is not False
    ):
        raise Qwen3NetworkError(
            "qwen3_network_reference_invalid", "artifact reference contract is invalid",
        )
    return result


def build_local_artifact_reference(
    path: str | Path,
    *,
    artifact_root: str | Path,
    source_node_id: str,
    target_node_id: str,
    chain_id: str,
    generation: int,
    phase: str,
    from_segment: int,
    to_segment: int,
) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
    artifact = _inside(root, path)
    size, digest = _file_evidence(artifact)
    identity = hashlib.sha256(
        f"{chain_id}:{generation}:{phase}:{from_segment}:{to_segment}:{digest}".encode("ascii"),
    ).hexdigest()[:32]
    return _reference(
        mode="local",
        artifact_id=f"local_{identity}",
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        chain_id=chain_id,
        generation=generation,
        phase=phase,
        from_segment=from_segment,
        to_segment=to_segment,
        size_bytes=size,
        sha256=digest,
    )


@dataclass(frozen=True)
class Qwen3ResolvedArtifact:
    """Internal path plus a path-free reference safe for control metadata."""

    path: Path
    reference: dict[str, Any]


class Qwen3NetworkTransferCoordinator:
    """Issue receives only for one node in one active canonical chain."""

    def __init__(
        self,
        *,
        local_node_id: str,
        runtime: Qwen3ArtifactTransferRuntime,
    ) -> None:
        if not str(local_node_id or ""):
            raise ValueError("Qwen3 network coordinator requires a local node identity")
        self.local_node_id = str(local_node_id)
        self.runtime = runtime
        self._lock = threading.RLock()
        self._contract: dict[str, Any] | None = None
        self._last_generation = -1
        self._transfers: dict[str, dict[str, Any]] = {}
        self._outputs: dict[str, dict[str, Any]] = {}
        self._control_peer_epoch: int = 0
        self._phase = "idle"
        self._ledger_load: Callable[[], Mapping[str, Any]] | None = None
        self._ledger_save: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None
        self._ledger: dict[str, Any] = {
            "schema_version": 1,
            "local_node_id": self.local_node_id,
            "last_generation": -1,
            "active_contract": {},
            "transfers": {},
            "outputs": {},
            "updated_at": "",
        }
        self._recovered_contract: dict[str, Any] = {}
        self._restart_pending = False
        self.sidecar_executor: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None
        self.runtime.authorization_gate = self.authorize_transfer
        self._reconcile_consumed_outputs()

    def configure_persistent_ledger(
        self,
        *,
        load: Callable[[], Mapping[str, Any]],
        save: Callable[[dict[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Attach the user-owned SQLite projection before activating a chain."""
        if not callable(load) or not callable(save):
            raise ValueError("Qwen3 network ledger callbacks are invalid")
        with self._lock:
            if self._contract is not None or self._transfers or self._outputs:
                raise Qwen3NetworkError(
                    "qwen3_network_ledger_active", "network ledger must be attached before activation",
                )
            try:
                loaded = dict(load())
            except Exception as exc:
                raise Qwen3NetworkError(
                    "qwen3_network_ledger_unavailable", "network ledger could not be loaded",
                ) from exc
            ledger_node = str(loaded.get("local_node_id", "") or "")
            if ledger_node and ledger_node != self.local_node_id:
                raise Qwen3NetworkError(
                    "qwen3_network_ledger_identity", "network ledger belongs to another node",
                )
            self._ledger_load = load
            self._ledger_save = save
            self._ledger = {
                "schema_version": 1,
                "local_node_id": self.local_node_id,
                "last_generation": int(loaded.get("last_generation", -1)),
                "active_contract": dict(loaded.get("active_contract", {})),
                "transfers": {
                    str(key): dict(value)
                    for key, value in dict(loaded.get("transfers", {})).items()
                },
                "outputs": {
                    str(key): dict(value)
                    for key, value in dict(loaded.get("outputs", {})).items()
                },
                "updated_at": str(loaded.get("updated_at", "")),
            }
            self._last_generation = max(
                self._last_generation, int(self._ledger.get("last_generation", -1)),
            )
            active = dict(self._ledger.get("active_contract", {}))
            self._recovered_contract = active
            self._restart_pending = bool(active and active.get("phase") != "released")
            for record in self._ledger["transfers"].values():
                if record.get("status") not in {
                    "cancelled", "released", "invalidated", "expired", "failed",
                }:
                    record["status"] = "invalidated"
            for record in self._ledger["outputs"].values():
                if record.get("status") not in {"released", "invalidated"}:
                    record["status"] = "invalidated"
                    record["lease_state"] = "invalidated"
            if active:
                active["phase"] = "recovered"
                active["restart_epoch"] = int(active.get("restart_epoch", 0)) + 1
                self._ledger["active_contract"] = active
            self._persist_ledger_locked()
            return self.ledger_snapshot()

    def _persist_ledger_locked(self) -> None:
        save = self._ledger_save
        if save is None:
            return
        self._ledger["local_node_id"] = self.local_node_id
        self._ledger["last_generation"] = int(self._last_generation)
        try:
            self._ledger = dict(save(dict(self._ledger)))
        except Exception as exc:
            raise Qwen3NetworkError(
                "qwen3_network_ledger_unavailable", "network ledger could not be persisted",
            ) from exc

    def _record_transfer_locked(
        self,
        transfer_id: str,
        transfer: Mapping[str, Any],
        *,
        status: str | None = None,
        received_bytes: int | None = None,
        input_reference: Mapping[str, Any] | None = None,
        output_reference_id: str | None = None,
        kv_contract: Mapping[str, Any] | None = None,
    ) -> None:
        if self._ledger_save is None:
            return
        descriptor = dict(transfer.get("descriptor", {}))
        previous = dict(self._ledger["transfers"].get(str(transfer_id), {}))
        record = {
            "transfer_id": str(transfer_id),
            "source_node_id": str(transfer.get("peer_node_id", previous.get("source_node_id", ""))),
            "target_node_id": str(transfer.get("target_node_id", previous.get("target_node_id", self.local_node_id))),
            "chain_id": str(descriptor.get("chain_id", previous.get("chain_id", ""))),
            "generation": int(descriptor.get("generation", previous.get("generation", 0))),
            "phase": str(descriptor.get("phase", previous.get("phase", ""))),
            "from_segment": int(descriptor.get("from_segment", previous.get("from_segment", -1))),
            "to_segment": int(descriptor.get("to_segment", previous.get("to_segment", -1))),
            "size_bytes": int(descriptor.get("size_bytes", previous.get("size_bytes", 0))),
            "sha256": str(descriptor.get("sha256", previous.get("sha256", ""))),
            "status": str(status or transfer.get("status", previous.get("status", "receiving"))),
            "received_bytes": int(
                received_bytes
                if received_bytes is not None
                else previous.get("received_bytes", descriptor.get("received_bytes", 0))
            ),
            "input_reference": dict(
                input_reference
                if input_reference is not None
                else previous.get("input_reference", {})
            ),
            "output_reference_id": str(
                output_reference_id
                if output_reference_id is not None
                else previous.get("output_reference_id", "")
            ),
            "kv_contract": dict(
                kv_contract
                if kv_contract is not None
                else previous.get("kv_contract", {})
            ),
            "updated_at": "",
        }
        self._ledger["transfers"][str(transfer_id)] = record
        self._persist_ledger_locked()

    def _record_output_locked(
        self,
        output_id: str,
        output: Mapping[str, Any],
        *,
        status: str,
        lease_state: str,
        next_transfer_id: str | None = None,
        confirmed_offset: int | None = None,
    ) -> None:
        if self._ledger_save is None:
            return
        previous = dict(self._ledger["outputs"].get(str(output_id), {}))
        reference = dict(output.get("reference", previous.get("reference", {})))
        self._ledger["outputs"][str(output_id)] = {
            "artifact_id": str(output_id),
            "reference": reference,
            "parent_transfer_id": str(output.get("parent_transfer_id", previous.get("parent_transfer_id", ""))),
            "status": str(status),
            "lease_state": str(lease_state),
            "next_transfer_id": str(
                next_transfer_id
                if next_transfer_id is not None
                else previous.get("next_transfer_id", "")
            ),
            "confirmed_offset": int(
                confirmed_offset
                if confirmed_offset is not None
                else previous.get("confirmed_offset", 0)
            ),
            "updated_at": "",
        }
        self._persist_ledger_locked()

    def _set_output_terminal_locked(self, output_id: str, *, status: str) -> None:
        if self._ledger_save is None:
            return
        record = self._ledger.get("outputs", {}).get(str(output_id))
        if isinstance(record, dict):
            record["status"] = str(status)
            record["lease_state"] = "released" if status == "released" else "invalidated"
            self._persist_ledger_locked()

    def ledger_snapshot(self) -> dict[str, Any]:
        with self._lock:
            ledger = self._ledger
            return {
                "schema_version": 1,
                "enabled": self._ledger_save is not None,
                "local_node_id": self.local_node_id,
                "last_generation": int(ledger.get("last_generation", -1)),
                "restart_pending": bool(self._restart_pending),
                "active_contract": dict(ledger.get("active_contract", {})),
                "transfer_count": len(ledger.get("transfers", {})),
                "output_count": len(ledger.get("outputs", {})),
                "invalidated_transfers": sum(
                    record.get("status") == "invalidated"
                    for record in ledger.get("transfers", {}).values()
                ),
                "invalidated_outputs": sum(
                    record.get("status") == "invalidated"
                    for record in ledger.get("outputs", {}).values()
                ),
            }

    def configure_sidecar_executor(
        self,
        executor: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None,
    ) -> None:
        """Install the target-local executor used when a route omits one."""
        with self._lock:
            self.sidecar_executor = executor

    def _reconcile_consumed_outputs(self) -> None:
        """Remove target output files left by a process that restarted."""
        root = self.runtime.receiver.root
        for pattern in ("qwen3-consume-*.pt", ".qwen3-consume-*.part"):
            for path in root.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue

    def _discard_transfer_locked(self, transfer_id: str, *, reason_code: str) -> dict[str, Any]:
        transfer_key = str(transfer_id)
        transfer = self._transfers.pop(transfer_key, None)
        if transfer is None:
            return {
                "transfer_id": transfer_key,
                "status": "missing",
                "cleanup_complete": True,
                "cleanup_failures": 0,
                "removed_artifacts": 0,
            }
        cleanup = getattr(transfer.get("executor"), "cleanup", None)
        if callable(cleanup):
            try:
                cleanup(dict(transfer.get("consume_request", {})), str(reason_code))
            except Exception:
                pass
        result = self.runtime.receiver.discard(transfer_key)
        removed = int(result.get("removed_artifacts", 0) or 0)
        failures = int(result.get("cleanup_failures", 0) or 0)
        for output_id, output in list(self._outputs.items()):
            if output.get("parent_transfer_id") != transfer_key:
                continue
            path = output.get("path")
            if isinstance(path, Path):
                try:
                    existed = path.exists()
                    path.unlink(missing_ok=True)
                    removed += int(existed)
                except OSError:
                    failures += 1
            output_status = "released" if reason_code == "release" else "invalidated"
            self._record_output_locked(
                output_id, output, status=output_status,
                lease_state="released" if output_status == "released" else "invalidated",
            )
            self._outputs.pop(output_id, None)
        terminal_status = {
            "release": "released",
            "expired": "expired",
            "peer_epoch_changed": "invalidated",
            "consume_failed": "failed",
        }.get(str(reason_code), "cancelled")
        self._record_transfer_locked(transfer_key, transfer, status=terminal_status)
        return {
            "transfer_id": transfer_key,
            "status": terminal_status,
            "cleanup_complete": failures == 0,
            "cleanup_failures": failures,
            "removed_artifacts": removed,
        }

    def cleanup_expired(self) -> dict[str, Any]:
        """Reconcile receiver TTL expiry into the persistent network ledger."""
        with self._lock:
            runtime_result = self.runtime.receiver.cleanup_expired()
            reconciled = 0
            cleanup_failures = int(runtime_result.get("cleanup_failures", 0) or 0)
            removed_artifacts = int(runtime_result.get("removed_artifacts", 0) or 0)
            for transfer_id in list(self._transfers):
                if self.runtime.receiver.session_status(transfer_id) != "expired":
                    continue
                result = self._discard_transfer_locked(
                    transfer_id, reason_code="expired",
                )
                reconciled += 1
                cleanup_failures += int(result.get("cleanup_failures", 0) or 0)
                removed_artifacts += int(result.get("removed_artifacts", 0) or 0)
            return {
                **runtime_result,
                "reconciled_transfers": reconciled,
                "removed_artifacts": removed_artifacts,
                "cleanup_failures": cleanup_failures,
                "cleanup_complete": cleanup_failures == 0,
            }

    def _fence_peer_locked(self, peer_node_id: str, *, reason_code: str) -> None:
        for transfer_id, transfer in list(self._transfers.items()):
            if transfer.get("peer_node_id") == str(peer_node_id):
                self._discard_transfer_locked(transfer_id, reason_code=reason_code)

    @staticmethod
    def _consume_key(
        *, phase: str, generation: int, batch_size: int, sequence_length: int,
        dtype: str, device: str, has_next_segment: bool,
    ) -> tuple[Any, ...]:
        return (
            str(phase), int(generation), int(batch_size), int(sequence_length),
            str(dtype).lower(), str(device).lower(), bool(has_next_segment),
        )

    @staticmethod
    def _dtype_matches(actual: Any, expected: Any) -> bool:
        aliases = {
            "float16": "float16", "fp16": "float16", "torch.float16": "float16",
            "bfloat16": "bfloat16", "bf16": "bfloat16", "torch.bfloat16": "bfloat16",
            "float32": "float32", "fp32": "float32", "torch.float32": "float32",
        }
        return aliases.get(str(actual).lower(), str(actual).lower()) == aliases.get(
            str(expected).lower(), str(expected).lower(),
        )

    @staticmethod
    def _device_matches(actual: Any, expected: Any) -> bool:
        actual_value = str(actual).lower()
        expected_value = str(expected).lower()
        if expected_value == "cuda":
            return actual_value == "cuda" or actual_value.startswith("cuda:")
        return actual_value == expected_value

    def activate(self, contract: dict[str, Any]) -> dict[str, Any]:
        canonical = validate_qwen3_dry_run_contract(contract)
        if canonical.get("execution_mode") != "node_local_sidecar":
            raise Qwen3NetworkError(
                "qwen3_network_contract_invalid", "network handoff requires node_local_sidecar mode",
            )
        if self.local_node_id not in {
            segment["node_id"] for segment in canonical["segments"]
        }:
            raise Qwen3NetworkError(
                "qwen3_network_peer_scope", "local node is not in the active Qwen3 topology",
            )
        with self._lock:
            if self._contract is not None:
                if self._contract["contract_sha256"] == canonical["contract_sha256"]:
                    return self.snapshot()
                raise Qwen3NetworkError(
                    "qwen3_network_contract_active", "another Qwen3 network contract is active",
                )
            recovered_same_contract = bool(
                self._restart_pending
                and self._recovered_contract.get("contract_sha256") == canonical["contract_sha256"]
                and int(self._recovered_contract.get("generation", -1)) == int(canonical["generation"])
            )
            if int(canonical["generation"]) <= self._last_generation and not recovered_same_contract:
                raise Qwen3NetworkError(
                    "qwen3_network_generation_stale", "Qwen3 network generation is stale",
                )
            self._contract = canonical
            self._last_generation = max(self._last_generation, int(canonical["generation"]))
            self._transfers.clear()
            self._outputs.clear()
            self._phase = "prepared"
            self._restart_pending = False
            self._ledger["active_contract"] = {
                "contract_sha256": canonical["contract_sha256"],
                "generation": int(canonical["generation"]),
                "phase": "prepared",
                "segment_count": len(canonical["segments"]),
                "restart_epoch": int(
                    self._recovered_contract.get("restart_epoch", 0)
                    if recovered_same_contract else 0
                ),
            }
            self._persist_ledger_locked()
            return self.snapshot()

    def _active(self) -> dict[str, Any]:
        contract = self._contract
        if contract is None:
            raise Qwen3NetworkError(
                "qwen3_network_contract_inactive", "Qwen3 network contract is not active",
            )
        return contract

    def authorize_control_peer(
        self,
        peer_node_id: str,
        *,
        contract: dict[str, Any] | None = None,
        chain_id: str | None = None,
        generation: int | None = None,
        peer_epoch: int = 0,
    ) -> dict[str, Any]:
        """Fence one control call to the immediately preceding live peer."""
        with self._lock:
            canonical = (
                validate_qwen3_dry_run_contract(contract)
                if contract is not None
                else self._active()
            )
            if chain_id is not None and canonical["contract_sha256"] != str(chain_id):
                raise Qwen3NetworkError(
                    "qwen3_network_contract_mismatch", "network control chain is stale",
                )
            if generation is not None and int(canonical["generation"]) != int(generation):
                raise Qwen3NetworkError(
                    "qwen3_network_generation_stale", "network control generation is stale",
                )
            local_index = next(
                (
                    int(segment["segment_index"])
                    for segment in canonical["segments"]
                    if segment["node_id"] == self.local_node_id
                ),
                -1,
            )
            if local_index <= 0:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope",
                    "local node has no authenticated upstream network peer",
                )
            expected_peer = canonical["segments"][local_index - 1]["node_id"]
            if str(peer_node_id) != expected_peer:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope",
                    "network control peer is outside the adjacent topology",
                )
            if isinstance(peer_epoch, bool) or int(peer_epoch) < 0:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope", "network control peer epoch is invalid",
                )
            candidate_epoch = int(peer_epoch)
            if self._control_peer_epoch and candidate_epoch <= 0:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope", "network control peer registration epoch is stale",
                )
            if self._control_peer_epoch and candidate_epoch < self._control_peer_epoch:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope", "network control peer registration epoch is stale",
                )
            if candidate_epoch > self._control_peer_epoch:
                self._fence_peer_locked(
                    expected_peer,
                    reason_code="peer_epoch_changed",
                )
            self._control_peer_epoch = candidate_epoch
            return canonical

    @staticmethod
    def _expected_generation(contract: dict[str, Any], phase: str) -> int:
        if phase == "prefill":
            return int(contract["generation"])
        if phase == "decode":
            return int(contract["generation"]) + 1
        raise Qwen3NetworkError(
            "qwen3_network_phase_invalid", "Qwen3 network phase is invalid",
        )

    def begin_phase(self, phase: str, generation: int) -> dict[str, Any]:
        with self._lock:
            contract = self._active()
            expected_state = "prepared" if phase == "prefill" else "prefilled"
            if (
                phase not in {"prefill", "decode"}
                or self._phase != expected_state
                or int(generation) != self._expected_generation(contract, phase)
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_phase_invalid", "network execution phase is not active",
                )
            self._phase = phase
            if self._ledger.get("active_contract"):
                self._ledger["active_contract"]["phase"] = phase
                self._persist_ledger_locked()
            return self.snapshot()

    def finish_phase(self, phase: str, generation: int) -> dict[str, Any]:
        with self._lock:
            contract = self._active()
            if (
                phase not in {"prefill", "decode"}
                or self._phase != phase
                or int(generation) != self._expected_generation(contract, phase)
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_phase_invalid", "network execution phase cannot finish",
                )
            if any(
                transfer["descriptor"]["phase"] == phase
                and transfer.get("status") not in {"committed", "consumed"}
                for transfer in self._transfers.values()
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_phase_incomplete",
                    "network execution phase still has incomplete transfers",
                )
            self._phase = "prefilled" if phase == "prefill" else "decoded"
            if self._ledger.get("active_contract"):
                self._ledger["active_contract"]["phase"] = self._phase
                self._persist_ledger_locked()
            return self.snapshot()

    def begin_receive(
        self,
        *,
        base_url: str,
        source_peer_id: str,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        size_bytes: int,
        sha256: str,
        ttl_seconds: float = 60,
        peer_epoch: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            contract = self._active()
            try:
                source = contract["segments"][int(from_segment)]
                target = contract["segments"][int(to_segment)]
            except (IndexError, TypeError, ValueError) as exc:
                raise Qwen3NetworkError(
                    "qwen3_network_boundary_invalid", "network handoff boundary is invalid",
                ) from exc
            if (
                chain_id != contract["contract_sha256"]
                or int(to_segment) != int(from_segment) + 1
                or source["segment_index"] != int(from_segment)
                or target["segment_index"] != int(to_segment)
                or source["node_id"] != str(source_peer_id)
                or target["node_id"] != self.local_node_id
                or int(generation) != self._expected_generation(contract, phase)
                or self._phase != phase
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_contract_mismatch", "network handoff does not match active topology",
                )
            plan = self.runtime.begin_receive(
                base_url=base_url,
                peer_node_id=source["node_id"],
                chain_id=contract["contract_sha256"],
                generation=int(generation),
                phase=phase,
                from_segment=int(from_segment),
                to_segment=int(to_segment),
                size_bytes=int(size_bytes),
                sha256=str(sha256),
                ttl_seconds=ttl_seconds,
                peer_epoch=int(peer_epoch),
            )
            self._transfers[plan["transfer_id"]] = {
                "peer_node_id": source["node_id"],
                "peer_epoch": int(peer_epoch),
                "target_node_id": target["node_id"],
                "ticket": plan["ticket"],
                "descriptor": dict(plan["descriptor"]),
                "status": "receiving",
            }
            self._record_transfer_locked(
                plan["transfer_id"], self._transfers[plan["transfer_id"]],
                status="receiving", received_bytes=0,
            )
            return plan

    def authorize_transfer(
        self, transfer_id: str, peer_node_id: str, peer_epoch: int = 0,
    ) -> None:
        with self._lock:
            if self._contract is None:
                raise Qwen3TransferError(
                    "qwen3_transfer_not_active", "Qwen3 network contract is not active",
                )
            transfer = self._transfers.get(str(transfer_id))
            if transfer is None:
                raise Qwen3TransferError(
                    "qwen3_transfer_scope_mismatch", "transfer is not owned by the active Qwen3 chain",
                )
            if transfer["peer_node_id"] != str(peer_node_id):
                raise Qwen3TransferError(
                    "qwen3_transfer_peer_mismatch", "transfer belongs to another active peer",
                )
            if int(transfer.get("peer_epoch", 0)) != int(peer_epoch):
                raise Qwen3TransferError(
                    "qwen3_transfer_peer_epoch_mismatch",
                    "transfer belongs to another TCP registration epoch",
                )
            if (
                transfer.get("status") != "receiving"
                or self._phase != transfer["descriptor"]["phase"]
            ):
                raise Qwen3TransferError(
                    "qwen3_transfer_not_active", "transfer phase is no longer active",
                )

    def record_transfer_progress(self, transfer_id: str, received_bytes: int) -> dict[str, Any]:
        """Persist the receiver-confirmed offset used by retry and restart fencing."""
        if isinstance(received_bytes, bool) or not isinstance(received_bytes, int):
            raise Qwen3NetworkError(
                "qwen3_network_progress_invalid", "network transfer progress is invalid",
            )
        with self._lock:
            transfer = self._transfers.get(str(transfer_id))
            if transfer is None or transfer.get("status") != "receiving":
                raise Qwen3NetworkError(
                    "qwen3_network_transfer_missing", "network transfer is not receiving",
                )
            size = int(transfer["descriptor"]["size_bytes"])
            previous = int(
                self._ledger.get("transfers", {})
                .get(str(transfer_id), {})
                .get("received_bytes", 0)
            )
            if not previous <= received_bytes <= size:
                raise Qwen3NetworkError(
                    "qwen3_network_progress_invalid", "network transfer progress moved backwards",
                )
            self._record_transfer_locked(
                str(transfer_id), transfer, status="receiving", received_bytes=received_bytes,
            )
            return {
                "transfer_id": str(transfer_id),
                "received_bytes": int(received_bytes),
                "size_bytes": size,
                "status": "receiving",
            }

    def commit_reference(self, transfer_id: str) -> dict[str, Any]:
        with self._lock:
            contract = self._active()
            transfer = self._transfers.get(str(transfer_id))
            if transfer is None:
                raise Qwen3NetworkError(
                    "qwen3_network_transfer_missing", "network transfer is not active",
                )
            if transfer.get("status") == "consumed":
                reference = transfer.get("input_reference")
                if isinstance(reference, Mapping):
                    return dict(reference)
                raise Qwen3NetworkError(
                    "qwen3_network_transfer_missing", "network input artifact is no longer available",
                )
            descriptor = transfer["descriptor"]
            try:
                path = self.runtime.receiver.artifact_path(str(transfer_id))
            except Qwen3TransferError as exc:
                if exc.reason_code == "qwen3_transfer_missing":
                    raise Qwen3NetworkError(
                        "qwen3_network_artifact_missing",
                        "committed network artifact is unavailable",
                    ) from exc
                raise
            if not path.is_file():
                raise Qwen3NetworkError(
                    "qwen3_network_artifact_missing",
                    "committed network artifact is unavailable",
                )
            size, digest = _file_evidence(path)
            if size != descriptor["size_bytes"] or digest != descriptor["sha256"]:
                self.runtime.receiver.discard(str(transfer_id))
                raise Qwen3NetworkError(
                    "qwen3_network_digest_mismatch", "committed network artifact changed",
                )
            reference = _reference(
                mode="network",
                artifact_id=str(transfer_id),
                source_node_id=transfer["peer_node_id"],
                target_node_id=self.local_node_id,
                chain_id=contract["contract_sha256"],
                generation=descriptor["generation"],
                phase=descriptor["phase"],
                from_segment=descriptor["from_segment"],
                to_segment=descriptor["to_segment"],
                size_bytes=size,
                sha256=digest,
            )
            transfer["status"] = "committed"
            transfer["input_reference"] = dict(reference)
            self._record_transfer_locked(
                str(transfer_id), transfer, status="committed",
                received_bytes=size, input_reference=reference,
            )
            return reference

    def resolve(self, transfer_id: str) -> Qwen3ResolvedArtifact:
        with self._lock:
            reference = self.commit_reference(transfer_id)
            try:
                path = self.runtime.receiver.artifact_path(str(transfer_id))
            except Qwen3TransferError as exc:
                raise Qwen3NetworkError(
                    "qwen3_network_artifact_missing", "committed network artifact is unavailable",
                ) from exc
            return Qwen3ResolvedArtifact(path=path, reference=reference)

    def consume_transfer(
        self,
        transfer_id: str,
        *,
        phase: str,
        generation: int,
        batch_size: int,
        sequence_length: int,
        dtype: str,
        device: str,
        has_next_segment: bool,
        executor: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Consume one committed artifact inside the target process.

        Only the target-owned callback receives the resolved path.  The returned
        contract is deliberately path-free so an upstream peer cannot learn the
        target filesystem layout.
        """
        if phase not in {"prefill", "decode"}:
            raise Qwen3NetworkError("qwen3_network_phase_invalid", "target execution phase is invalid")
        if (
            isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
            or isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0
        ):
            raise Qwen3NetworkError("qwen3_network_contract_mismatch", "target execution dimensions are invalid")
        with self._lock:
            contract = self._active()
            transfer = self._transfers.get(str(transfer_id))
            if transfer is None:
                raise Qwen3NetworkError("qwen3_network_transfer_missing", "network transfer is not committed")
            if transfer.get("status") == "consumed":
                key = self._consume_key(
                    phase=phase, generation=generation, batch_size=batch_size,
                    sequence_length=sequence_length, dtype=dtype, device=device,
                    has_next_segment=has_next_segment,
                )
                if transfer.get("consume_key") == key:
                    return dict(transfer.get("consume_result", {}))
                raise Qwen3NetworkError(
                    "qwen3_network_consume_duplicate_mismatch",
                    "target consume was already committed with another contract",
                )
            if transfer.get("status") == "consuming":
                raise Qwen3NetworkError(
                    "qwen3_network_consume_in_progress", "target consume is already in progress",
                )
            if transfer.get("status") != "committed":
                raise Qwen3NetworkError("qwen3_network_transfer_missing", "network transfer is not committed")
            descriptor = transfer["descriptor"]
            expected_next = int(descriptor["to_segment"]) < len(contract["segments"]) - 1
            if (
                descriptor["phase"] != phase
                or int(descriptor["generation"]) != int(generation)
                or self._phase != phase
                or bool(has_next_segment) != expected_next
                or not self._dtype_matches(dtype, contract["segments"][int(descriptor["to_segment"])] ["dtype"])
                or not self._device_matches(device, contract["segments"][int(descriptor["to_segment"])] ["execution_device"])
            ):
                raise Qwen3NetworkError("qwen3_network_contract_mismatch", "target execution contract does not match")
            resolved = self.resolve(str(transfer_id))
            request = {
                "transfer_id": str(transfer_id),
                "chain_id": contract["contract_sha256"],
                "generation": int(generation),
                "phase": str(phase),
                "batch_size": int(batch_size),
                "sequence_length": int(sequence_length),
                "dtype": str(dtype),
                "device": str(device),
                "has_next_segment": bool(has_next_segment),
                "reference": dict(resolved.reference),
                "segment_index": int(descriptor["to_segment"]),
                "layer_range": list(contract["segments"][int(descriptor["to_segment"])] ["layer_range"]),
                "target_node_id": self.local_node_id,
            }
            consume_key = self._consume_key(
                phase=phase, generation=generation, batch_size=batch_size,
                sequence_length=sequence_length, dtype=dtype, device=device,
                has_next_segment=has_next_segment,
            )
            effective_executor = executor if executor is not None else self.sidecar_executor
            transfer["status"] = "consuming"
            transfer["consume_key"] = consume_key
            transfer["consume_request"] = dict(request)
            transfer["executor"] = effective_executor
            self._record_transfer_locked(
                str(transfer_id), transfer, status="consuming",
                input_reference=resolved.reference,
            )
            metadata: Mapping[str, Any] = {}
        try:
            if effective_executor is not None:
                result = effective_executor(resolved.path, request)
                if result is not None:
                    if not isinstance(result, Mapping):
                        raise Qwen3NetworkError("qwen3_network_execution_invalid", "target executor result is invalid")
                    metadata = result
            hidden = metadata.get("hidden_handoff", {})
            if not isinstance(hidden, Mapping):
                hidden = {}
            kv = metadata.get("kv_contract", {})
            if not isinstance(kv, Mapping):
                kv = {}
            def _safe_label(value: Any, fallback: str) -> str:
                candidate = str(value if value is not None else fallback)
                if not candidate or len(candidate) > 64 or any(char in candidate for char in ("/", "\\", "\x00")):
                    raise Qwen3NetworkError(
                        "qwen3_network_execution_invalid", "target execution label is invalid",
                    )
                return candidate

            def _safe_shape(value: Any, fallback: list[int]) -> list[int]:
                candidate = value if value is not None else fallback
                if (
                    not isinstance(candidate, (list, tuple))
                    or len(candidate) > 16
                    or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in candidate)
                ):
                    raise Qwen3NetworkError(
                        "qwen3_network_execution_invalid", "target execution shape is invalid",
                    )
                return [int(item) for item in candidate]
            hidden_dtype = _safe_label(hidden.get("dtype", dtype), str(dtype))
            hidden_device = _safe_label(hidden.get("device", device), str(device))
            hidden_shape = _safe_shape(hidden.get("shape"), [int(batch_size), int(sequence_length)])
            kv_shape = _safe_shape(kv.get("shape"), [int(batch_size), int(sequence_length)])
            output_reference = None
            output_entry: dict[str, Any] | None = None
            output_path = metadata.get("output_path")
            if output_path is not None:
                output = _inside(self.runtime.receiver.root, output_path)
                if output == resolved.path or not output.is_file():
                    raise Qwen3NetworkError(
                        "qwen3_network_execution_invalid", "target output artifact is unavailable",
                    )
                output_size, output_sha256 = _file_evidence(output)
                if output_size <= 0:
                    raise Qwen3NetworkError(
                        "qwen3_network_execution_invalid", "target output artifact is empty",
                    )
                if bool(has_next_segment):
                    next_index = int(descriptor["to_segment"]) + 1
                    output_id = hashlib.sha256(
                        f"{transfer_id}:{output_sha256}".encode("ascii"),
                    ).hexdigest()[:32]
                    output_reference = _reference(
                        mode="network",
                        artifact_id=f"qout_{output_id}",
                        source_node_id=self.local_node_id,
                        target_node_id=contract["segments"][next_index]["node_id"],
                        chain_id=contract["contract_sha256"],
                        generation=int(generation),
                        phase=str(phase),
                        from_segment=int(descriptor["to_segment"]),
                        to_segment=next_index,
                        size_bytes=output_size,
                        sha256=output_sha256,
                    )
                    output_entry = {
                        "path": output,
                        "reference": dict(output_reference),
                        "parent_transfer_id": str(transfer_id),
                        "lease_state": "available",
                        "transfer_status": "registered",
                        "next_transfer_id": "",
                        "confirmed_offset": 0,
                    }
            result_contract = {
                "schema_version": 1,
                "transfer_id": str(transfer_id),
                "chain_id": contract["contract_sha256"],
                "generation": int(generation),
                "phase": str(phase),
                "from_segment": int(descriptor["from_segment"]),
                "to_segment": int(descriptor["to_segment"]),
                "execution": {
                    "artifact_bytes": int(resolved.reference["size_bytes"]),
                    "artifact_sha256": str(resolved.reference["sha256"]),
                    "input_consumed": True,
                    "output_registered": output_reference is not None,
                },
                "output_reference": dict(output_reference) if output_reference else None,
                "hidden_handoff": {
                    "dtype": hidden_dtype,
                    "device": hidden_device,
                    "shape": hidden_shape,
                    "has_next_segment": bool(has_next_segment),
                },
                "kv_contract": {
                    "present": bool(kv.get("present", phase == "decode")),
                    "shape": kv_shape,
                },
                "full_model_materialized": False,
            }
            with self._lock:
                current = self._transfers.get(str(transfer_id))
                if current is None or current.get("status") != "consuming":
                    raise Qwen3NetworkError(
                        "qwen3_network_consume_cancelled", "target consume was cancelled",
                    )
                current["status"] = "consumed"
                current["consume_result"] = dict(result_contract)
                current["output_reference"] = dict(output_reference) if output_reference else None
                if output_reference is not None and output_entry is not None:
                    self._outputs[output_reference["artifact_id"]] = output_entry
                    self._record_output_locked(
                        output_reference["artifact_id"], output_entry,
                        status="registered", lease_state="available",
                    )
                self._record_transfer_locked(
                    str(transfer_id), current, status="consumed",
                    received_bytes=int(resolved.reference["size_bytes"]),
                    input_reference=resolved.reference,
                    output_reference_id=(
                        output_reference["artifact_id"] if output_reference else ""
                    ),
                    kv_contract=result_contract["kv_contract"],
                )
                self.runtime.receiver.discard(str(transfer_id))
            return result_contract
        except Exception as exc:
            with self._lock:
                self._discard_transfer_locked(
                    str(transfer_id), reason_code="consume_failed",
                )
            if isinstance(exc, Qwen3NetworkError):
                raise
            # QW3.17：透传 sidecar 失败原因（错误码/消息，无路径/tensor），
            # 否则远端 consume 失败只剩笼统的 "target sidecar execution failed"
            raise Qwen3NetworkError(
                "qwen3_network_execution_failed",
                f"target sidecar execution failed: {exc}",
            ) from exc

    def consume(self, transfer_id: str, **fields) -> dict[str, Any]:
        """Local-controller spelling matching the remote control client API."""
        return self.consume_transfer(str(transfer_id), **fields)

    def lease_output_reference(self, output_id: str, next_transfer_id: str) -> dict[str, Any]:
        """Bind one registered output to exactly one resumable next-hop transfer."""
        transfer_key = str(next_transfer_id)
        if not transfer_key.startswith("qtx_") or len(transfer_key) != 36:
            raise Qwen3NetworkError(
                "qwen3_network_output_lease_invalid", "next-hop transfer identity is invalid",
            )
        with self._lock:
            output = self._outputs.get(str(output_id))
            if output is None:
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "registered output reference is unavailable",
                )
            current = str(output.get("next_transfer_id", ""))
            state = str(output.get("lease_state", "available"))
            transfer_status = str(output.get("transfer_status", "registered"))
            if current and current != transfer_key:
                raise Qwen3NetworkError(
                    "qwen3_network_output_lease_conflict", "output reference is leased by another transfer",
                )
            if state in {"released", "invalidated"}:
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "output reference lease is no longer active",
                )
            if current == transfer_key and transfer_status == "committed":
                reference = validate_qwen3_artifact_reference(output.get("reference", {}))
                return {
                    "output_id": str(output_id),
                    "next_transfer_id": transfer_key,
                    "confirmed_offset": int(reference["size_bytes"]),
                    "status": "committed",
                }
            output["lease_state"] = "leased"
            output["transfer_status"] = "transferring"
            output["next_transfer_id"] = transfer_key
            output.setdefault("confirmed_offset", 0)
            self._record_output_locked(
                str(output_id), output, status="transferring", lease_state="leased",
                next_transfer_id=transfer_key,
                confirmed_offset=int(output.get("confirmed_offset", 0)),
            )
            return {
                "output_id": str(output_id),
                "next_transfer_id": transfer_key,
                "confirmed_offset": int(output.get("confirmed_offset", 0)),
                "status": "leased",
            }

    def authorize_output_peer(
        self, output_id: str, peer_node_id: str, peer_epoch: int = 0,
    ) -> dict[str, Any]:
        """Fence output lease/progress calls to the reference's downstream peer."""
        if isinstance(peer_epoch, bool) or not isinstance(peer_epoch, int) or peer_epoch < 0:
            raise Qwen3NetworkError(
                "qwen3_network_output_peer_scope", "output peer epoch is invalid",
            )
        with self._lock:
            output = self._outputs.get(str(output_id))
            if output is None:
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "registered output reference is unavailable",
                )
            reference = validate_qwen3_artifact_reference(output.get("reference", {}))
            if reference["target_node_id"] != str(peer_node_id):
                raise Qwen3NetworkError(
                    "qwen3_network_output_peer_scope", "output reference belongs to another peer",
                )
            previous_epoch = int(output.get("consumer_peer_epoch", 0))
            if previous_epoch and peer_epoch < previous_epoch:
                raise Qwen3NetworkError(
                    "qwen3_network_output_peer_scope", "output peer registration epoch is stale",
                )
            output["consumer_peer_epoch"] = max(previous_epoch, int(peer_epoch))
            return reference

    def record_output_progress(
        self, output_id: str, next_transfer_id: str, confirmed_offset: int,
    ) -> dict[str, Any]:
        if isinstance(confirmed_offset, bool) or not isinstance(confirmed_offset, int):
            raise Qwen3NetworkError(
                "qwen3_network_output_offset_invalid", "output progress is invalid",
            )
        with self._lock:
            output = self._outputs.get(str(output_id))
            if output is None or output.get("next_transfer_id") != str(next_transfer_id):
                raise Qwen3NetworkError(
                    "qwen3_network_output_lease_conflict", "output progress does not match its lease",
                )
            reference = validate_qwen3_artifact_reference(output.get("reference", {}))
            previous = int(output.get("confirmed_offset", 0))
            if not previous <= confirmed_offset <= int(reference["size_bytes"]):
                raise Qwen3NetworkError(
                    "qwen3_network_output_offset_invalid", "output progress moved backwards",
                )
            output["confirmed_offset"] = int(confirmed_offset)
            status = (
                "committed"
                if output.get("transfer_status") == "committed"
                and confirmed_offset == int(reference["size_bytes"])
                else "transferring"
            )
            output["transfer_status"] = status
            self._record_output_locked(
                str(output_id), output, status=status, lease_state="leased",
                next_transfer_id=str(next_transfer_id), confirmed_offset=confirmed_offset,
            )
            return {
                "output_id": str(output_id),
                "next_transfer_id": str(next_transfer_id),
                "confirmed_offset": int(confirmed_offset),
                "size_bytes": int(reference["size_bytes"]),
                "status": status,
            }

    def commit_output_reference(self, output_id: str, next_transfer_id: str) -> dict[str, Any]:
        with self._lock:
            output = self._outputs.get(str(output_id))
            if output is None or output.get("next_transfer_id") != str(next_transfer_id):
                raise Qwen3NetworkError(
                    "qwen3_network_output_lease_conflict", "output commit does not match its lease",
                )
            reference = validate_qwen3_artifact_reference(output.get("reference", {}))
            if int(output.get("confirmed_offset", 0)) != int(reference["size_bytes"]):
                raise Qwen3NetworkError(
                    "qwen3_network_output_incomplete", "output transfer is not fully confirmed",
                )
            output["lease_state"] = "leased"
            output["transfer_status"] = "committed"
            self._record_output_locked(
                str(output_id), output, status="committed", lease_state="leased",
                next_transfer_id=str(next_transfer_id),
                confirmed_offset=int(reference["size_bytes"]),
            )
            return {
                "output_id": str(output_id),
                "next_transfer_id": str(next_transfer_id),
                "confirmed_offset": int(reference["size_bytes"]),
                "status": "committed",
            }

    def read_output_chunk(
        self,
        output_id: str,
        *,
        requester_peer_id: str,
        authenticated_peer_epoch: int = 0,
        offset: int,
        max_bytes: int,
    ) -> dict[str, Any]:
        """Read one bounded, path-free chunk from a registered next-hop output."""
        from qwen3_pipeline_transfer import MAX_TRANSFER_CHUNK_BYTES

        if (
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            or not 0 < max_bytes <= MAX_TRANSFER_CHUNK_BYTES
        ):
            raise Qwen3NetworkError(
                "qwen3_network_output_offset_invalid", "output chunk bounds are invalid",
            )
        with self._lock:
            output = self._outputs.get(str(output_id))
            if output is None:
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "registered output reference is unavailable",
                )
            try:
                reference = validate_qwen3_artifact_reference(output.get("reference", {}))
            except Qwen3NetworkError:
                self._outputs.pop(str(output_id), None)
                self._set_output_terminal_locked(str(output_id), status="invalidated")
                raise
            if reference["artifact_id"] != str(output_id):
                self._outputs.pop(str(output_id), None)
                self._set_output_terminal_locked(str(output_id), status="invalidated")
                raise Qwen3NetworkError(
                    "qwen3_network_output_invalid", "registered output identity changed",
                )
            if reference["source_node_id"] != self.local_node_id or reference["target_node_id"] != str(requester_peer_id):
                raise Qwen3NetworkError(
                    "qwen3_network_output_peer_scope", "output reference is not owned by this peer",
                )
            path = output.get("path")
            if not isinstance(path, Path):
                self._outputs.pop(str(output_id), None)
                self._set_output_terminal_locked(str(output_id), status="invalidated")
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "registered output path is unavailable",
                )
            try:
                safe_path = _inside(self.runtime.receiver.root, path)
                size, digest = _file_evidence(safe_path)
            except (OSError, Qwen3NetworkError) as exc:
                self._outputs.pop(str(output_id), None)
                self._set_output_terminal_locked(str(output_id), status="invalidated")
                if isinstance(exc, Qwen3NetworkError):
                    raise Qwen3NetworkError(
                        "qwen3_network_output_missing", "registered output artifact is unavailable",
                    ) from exc
                raise Qwen3NetworkError(
                    "qwen3_network_output_missing", "registered output artifact is unavailable",
                ) from exc
            if size != reference["size_bytes"] or digest != reference["sha256"]:
                self._outputs.pop(str(output_id), None)
                self._set_output_terminal_locked(str(output_id), status="invalidated")
                try:
                    safe_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise Qwen3NetworkError(
                    "qwen3_network_output_digest_mismatch", "registered output artifact changed",
                )
            if offset > size:
                raise Qwen3NetworkError(
                    "qwen3_network_output_offset_invalid", "output chunk offset exceeds artifact",
                )
            with safe_path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(min(max_bytes, size - offset))
            return {
                "reference": dict(reference),
                "offset": int(offset),
                "total_bytes": int(size),
                "sha256": digest,
                "eof": offset + len(data) >= size,
                "data": data,
                "peer_epoch": int(authenticated_peer_epoch),
            }

    def release_output_reference(self, output_id: str) -> dict[str, Any]:
        """Release one registered output after its next hop has committed it."""
        with self._lock:
            output = self._outputs.pop(str(output_id), None)
            if output is None:
                return {"output_id": str(output_id), "status": "missing", "cleanup_complete": True}
            path = output.get("path")
            failures = 0
            removed = 0
            if isinstance(path, Path):
                try:
                    existed = path.exists()
                    path.unlink(missing_ok=True)
                    removed = int(existed)
                except OSError:
                    failures = 1
            self._set_output_terminal_locked(str(output_id), status="released")
            return {
                "output_id": str(output_id),
                "status": "released",
                "cleanup_complete": failures == 0,
                "cleanup_failures": failures,
                "removed_artifacts": removed,
            }

    def cancel_transfer(self, transfer_id: str) -> dict[str, Any]:
        with self._lock:
            return self._discard_transfer_locked(str(transfer_id), reason_code="cancelled")

    def release(self) -> dict[str, Any]:
        with self._lock:
            removed = 0
            failures = 0
            for transfer_id in list(self._transfers):
                result = self._discard_transfer_locked(transfer_id, reason_code="release")
                removed += int(result.get("removed_artifacts", 0) or 0)
                failures += int(result.get("cleanup_failures", 0) or 0)
            self._transfers.clear()
            for output_id, output in list(self._outputs.items()):
                path = output.get("path")
                if isinstance(path, Path):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        failures += 1
                self._set_output_terminal_locked(output_id, status="released")
                self._outputs.pop(output_id, None)
            self._contract = None
            self._phase = "idle"
            if self._ledger.get("active_contract"):
                self._ledger["active_contract"]["phase"] = "released"
                self._persist_ledger_locked()
            return {
                "cleanup_complete": failures == 0,
                "removed_artifacts": removed,
                "cleanup_failures": failures,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            contract = self._contract
            return {
                "schema_version": 1,
                "active": contract is not None,
                "local_node_id": self.local_node_id,
                "chain_id": "" if contract is None else contract["contract_sha256"],
                "generation": 0 if contract is None else contract["generation"],
                "phase": self._phase,
                "transfer_count": len(self._transfers),
                "consuming_count": sum(transfer.get("status") == "consuming" for transfer in self._transfers.values()),
                "consumed_count": sum(transfer.get("status") == "consumed" for transfer in self._transfers.values()),
                "output_count": len(self._outputs),
                "ledger": self.ledger_snapshot(),
                "full_model_materialized": False,
            }


@dataclass(frozen=True)
class Qwen3NetworkTarget:
    node_id: str
    base_url: str
    coordinator: "Qwen3NetworkController"
    requester: TransferRequester | None = None


class Qwen3NetworkController(Protocol):
    local_node_id: str

    def activate(self, contract: dict[str, Any]) -> dict[str, Any]: ...
    def begin_phase(self, phase: str, generation: int) -> dict[str, Any]: ...
    def finish_phase(self, phase: str, generation: int) -> dict[str, Any]: ...
    def begin_receive(self, **fields) -> dict[str, Any]: ...
    def resolve(self, transfer_id: str) -> Qwen3ResolvedArtifact: ...
    def commit_reference(self, transfer_id: str) -> dict[str, Any]: ...
    def record_transfer_progress(self, transfer_id: str, received_bytes: int) -> dict[str, Any]: ...
    def lease_output_reference(self, output_id: str, next_transfer_id: str) -> dict[str, Any]: ...
    def record_output_progress(self, output_id: str, next_transfer_id: str, confirmed_offset: int) -> dict[str, Any]: ...
    def commit_output_reference(self, output_id: str, next_transfer_id: str) -> dict[str, Any]: ...
    def read_output_chunk(self, output_id: str, **fields) -> dict[str, Any]: ...
    def authorize_output_peer(self, output_id: str, peer_node_id: str, peer_epoch: int = 0) -> dict[str, Any]: ...
    def release_output_reference(self, output_id: str) -> dict[str, Any]: ...
    def consume(self, transfer_id: str, **fields) -> dict[str, Any]: ...
    def cancel_transfer(self, transfer_id: str) -> dict[str, Any]: ...
    def release(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...


class Qwen3NetworkHandoffTransport:
    """Bridge sidecar boundaries through target-owned transfer runtimes."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        targets: Mapping[str, Qwen3NetworkTarget],
        peer_signers: Mapping[str, Qwen3PeerRequestSigner],
        chunk_bytes: int = MAX_TRANSFER_CHUNK_BYTES,
        max_attempts: int = 2,
        target_execution: bool = False,
    ) -> None:
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = root
        self.targets = dict(targets)
        self.peer_signers = dict(peer_signers)
        self.chunk_bytes = int(chunk_bytes)
        if not 1 <= int(max_attempts) <= 3:
            raise ValueError("Qwen3 network transfer attempts are outside limits")
        self.max_attempts = int(max_attempts)
        self.target_execution = bool(target_execution)
        self._contract: dict[str, Any] | None = None
        self._active_transfers: list[tuple[Qwen3NetworkController, str]] = []
        self._pending_output_transfers: dict[str, dict[str, Any]] = {}
        self._activated_targets: set[str] = set()
        self._phase = "idle"

    def activate(self, contract: dict[str, Any]) -> dict[str, Any]:
        canonical = validate_qwen3_dry_run_contract(contract)
        expected_targets = {
            segment["node_id"] for segment in canonical["segments"][1:]
        }
        expected_sources = {
            segment["node_id"] for segment in canonical["segments"][:-1]
        }
        if not expected_targets.issubset(self.targets) or not expected_sources.issubset(self.peer_signers):
            raise Qwen3NetworkError(
                "qwen3_network_topology_unavailable", "network target or peer signer is missing",
            )
        if self._contract is not None and self._contract["contract_sha256"] != canonical["contract_sha256"]:
            raise Qwen3NetworkError(
                "qwen3_network_contract_active", "another network chain is active",
            )
        activated: list[Qwen3NetworkTransferCoordinator] = []
        try:
            for node_id in sorted(expected_targets):
                target = self.targets[node_id]
                if target.node_id != node_id or target.coordinator.local_node_id != node_id:
                    raise Qwen3NetworkError(
                        "qwen3_network_topology_unavailable", "network target identity is invalid",
                    )
                target.coordinator.activate(canonical)
                activated.append(target.coordinator)
                self._activated_targets.add(node_id)
        except Exception:
            for coordinator in reversed(activated):
                try:
                    coordinator.release()
                except Exception:
                    continue
            self._activated_targets.clear()
            raise
        self._contract = canonical
        self._phase = "prepared"
        return self.snapshot()

    def begin_phase(self, phase: str, generation: int) -> dict[str, Any]:
        contract = self._contract
        expected_state = "prepared" if phase == "prefill" else "prefilled"
        expected_generation = (
            -1 if contract is None else int(contract["generation"]) + int(phase == "decode")
        )
        if (
            contract is None
            or phase not in {"prefill", "decode"}
            or self._phase != expected_state
            or int(generation) != expected_generation
        ):
            raise Qwen3NetworkError(
                "qwen3_network_phase_invalid", "network handoff phase is invalid",
            )
        for node_id in sorted(self._activated_targets):
            self.targets[node_id].coordinator.begin_phase(phase, generation)
        self._phase = phase
        return self.snapshot()

    def finish_phase(self, phase: str, generation: int) -> dict[str, Any]:
        if self._contract is None or self._phase != phase:
            raise Qwen3NetworkError(
                "qwen3_network_phase_invalid", "network handoff phase cannot finish",
            )
        for node_id in sorted(self._activated_targets):
            self.targets[node_id].coordinator.finish_phase(phase, generation)
        self._phase = "prefilled" if phase == "prefill" else "decoded"
        return self.snapshot()

    def transfer(
        self,
        *,
        source_path: str | Path,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        source_node_id: str,
        target_node_id: str,
    ) -> Qwen3ResolvedArtifact:
        contract = self._contract
        if contract is None or contract["contract_sha256"] != chain_id:
            raise Qwen3NetworkError(
                "qwen3_network_contract_inactive", "network handoff contract is not active",
            )
        if self._phase != phase:
            raise Qwen3NetworkError(
                "qwen3_network_phase_invalid", "network handoff phase is not active",
            )
        try:
            source_segment = contract["segments"][int(from_segment)]
            target_segment = contract["segments"][int(to_segment)]
        except (IndexError, TypeError, ValueError) as exc:
            raise Qwen3NetworkError(
                "qwen3_network_boundary_invalid", "network handoff boundary is invalid",
            ) from exc
        if (
            int(to_segment) != int(from_segment) + 1
            or source_segment["node_id"] != source_node_id
            or target_segment["node_id"] != target_node_id
        ):
            raise Qwen3NetworkError(
                "qwen3_network_contract_mismatch", "network handoff topology changed",
            )
        source = _inside(self.artifact_root, source_path)
        size, digest = _file_evidence(source)
        target = self.targets[target_node_id]
        plan: dict[str, Any] | None = None
        try:
            plan = target.coordinator.begin_receive(
                base_url=target.base_url,
                source_peer_id=source_node_id,
                chain_id=chain_id,
                generation=int(generation),
                phase=phase,
                from_segment=int(from_segment),
                to_segment=int(to_segment),
                size_bytes=size,
                sha256=digest,
            )
            signer = self.peer_signers[source_node_id]
            client = Qwen3ArtifactTransferClient(
                self.artifact_root,
                target.requester,
                chunk_bytes=self.chunk_bytes,
                peer_proof_headers=signer.headers,
            )
            receipt = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    receipt = client.upload(source=source, plan=plan)
                    break
                except Qwen3TransferError as exc:
                    if (
                        exc.reason_code != "qwen3_transfer_connection_failed"
                        or attempt >= self.max_attempts
                    ):
                        raise
            if receipt is None:
                raise Qwen3NetworkError(
                    "qwen3_network_transfer_failed", "network handoff produced no receipt",
                )
            if receipt.get("status") != "committed":
                raise Qwen3NetworkError(
                    "qwen3_network_commit_failed", "network handoff did not commit",
                )
            resolved = target.coordinator.resolve(plan["transfer_id"])
            _inside(self.artifact_root, resolved.path)
            if resolved.reference["sha256"] != digest or resolved.reference["size_bytes"] != size:
                raise Qwen3NetworkError(
                    "qwen3_network_digest_mismatch", "network reference differs from source",
                )
            self._active_transfers.append((target.coordinator, plan["transfer_id"]))
            return resolved
        except Qwen3NetworkError:
            if plan is not None:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
            raise
        except Exception as exc:
            if plan is not None:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
            reason_code = getattr(exc, "reason_code", "qwen3_network_transfer_failed")
            raise Qwen3NetworkError(str(reason_code), str(exc)) from exc

    def consume_target(
        self,
        *,
        target_node_id: str,
        transfer_id: str,
        phase: str,
        generation: int,
        batch_size: int,
        sequence_length: int,
        dtype: str,
        device: str,
        has_next_segment: bool,
    ) -> dict[str, Any]:
        """Invoke target-owned sidecar execution through its path-free control API."""
        target = self.targets.get(str(target_node_id))
        if target is None:
            raise Qwen3NetworkError("qwen3_network_topology_unavailable", "network target is unavailable")
        consumer = getattr(target.coordinator, "consume", None)
        if not callable(consumer):
            raise Qwen3NetworkError(
                "qwen3_network_execution_unavailable",
                "target does not expose the path-free consume API",
            )
        result = consumer(
            str(transfer_id),
            phase=str(phase),
            generation=int(generation),
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            dtype=str(dtype),
            device=str(device),
            has_next_segment=bool(has_next_segment),
        )
        if not isinstance(result, Mapping) or "execution" not in result:
            raise Qwen3NetworkError(
                "qwen3_network_execution_invalid", "target consume response is invalid",
            )
        result = dict(result)
        output_reference = result.get("output_reference")
        if output_reference is not None:
            result["output_reference"] = validate_qwen3_artifact_reference(output_reference)
        encoded = str(result)
        if any(token in encoded.lower() for token in ("output_path", "\\\\", "/users/", "/home/")):
            raise Qwen3NetworkError(
                "qwen3_network_execution_invalid", "target consume response leaked a filesystem path",
            )
        return result

    def transfer_reference(
        self,
        *,
        source_path: str | Path,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        source_node_id: str,
        target_node_id: str,
    ) -> dict[str, Any]:
        """Upload an artifact and return only its path-free target reference."""
        contract = self._contract
        if contract is None or contract["contract_sha256"] != chain_id:
            raise Qwen3NetworkError(
                "qwen3_network_contract_inactive", "network handoff contract is not active",
            )
        source = _inside(self.artifact_root, source_path)
        size, digest = _file_evidence(source)
        target = self.targets.get(str(target_node_id))
        if target is None:
            raise Qwen3NetworkError("qwen3_network_topology_unavailable", "network target is unavailable")
        plan: dict[str, Any] | None = None
        try:
            plan = target.coordinator.begin_receive(
                base_url=target.base_url,
                source_peer_id=source_node_id,
                chain_id=chain_id,
                generation=int(generation),
                phase=phase,
                from_segment=int(from_segment),
                to_segment=int(to_segment),
                size_bytes=size,
                sha256=digest,
            )
            signer = self.peer_signers[source_node_id]
            client = Qwen3ArtifactTransferClient(
                self.artifact_root,
                target.requester,
                chunk_bytes=self.chunk_bytes,
                peer_proof_headers=signer.headers,
            )
            receipt = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    receipt = client.upload(source=source, plan=plan)
                    break
                except Qwen3TransferError as exc:
                    if exc.reason_code != "qwen3_transfer_connection_failed" or attempt >= self.max_attempts:
                        raise
            if not isinstance(receipt, Mapping) or receipt.get("status") != "committed":
                raise Qwen3NetworkError(
                    "qwen3_network_commit_failed", "network handoff did not commit",
                )
            reference = validate_qwen3_artifact_reference(
                target.coordinator.commit_reference(plan["transfer_id"]),
            )
            if reference["size_bytes"] != size or reference["sha256"] != digest:
                raise Qwen3NetworkError(
                    "qwen3_network_digest_mismatch", "network reference differs from source",
                )
            self._active_transfers.append((target.coordinator, plan["transfer_id"]))
            return reference
        except Qwen3NetworkError:
            if plan is not None:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
            raise
        except Exception as exc:
            if plan is not None:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
            raise Qwen3NetworkError(
                str(getattr(exc, "reason_code", "qwen3_network_transfer_failed")), str(exc),
            ) from exc

    def transfer_registered_output(
        self,
        *,
        output_reference: Mapping[str, Any],
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        source_node_id: str,
        target_node_id: str,
    ) -> dict[str, Any]:
        """Move a registered output to its next target without exchanging paths."""
        reference = validate_qwen3_artifact_reference(output_reference)
        contract = self._contract
        if contract is None or contract["contract_sha256"] != str(chain_id):
            raise Qwen3NetworkError(
                "qwen3_network_contract_inactive", "network handoff contract is not active",
            )
        if self._phase != str(phase):
            raise Qwen3NetworkError(
                "qwen3_network_phase_invalid", "network handoff phase is not active",
            )
        if (
            reference["source_node_id"] != str(source_node_id)
            or reference["target_node_id"] != str(target_node_id)
            or reference["chain_id"] != str(chain_id)
            or reference["generation"] != int(generation)
            or reference["phase"] != str(phase)
            or reference["from_segment"] != int(from_segment)
            or reference["to_segment"] != int(to_segment)
            or int(to_segment) != int(from_segment) + 1
        ):
            raise Qwen3NetworkError(
                "qwen3_network_contract_mismatch", "registered output does not match the next boundary",
            )
        try:
            source_target = self.targets[str(source_node_id)]
            target = self.targets[str(target_node_id)]
        except KeyError as exc:
            raise Qwen3NetworkError(
                "qwen3_network_topology_unavailable", "registered output source or target is unavailable",
            ) from exc
        source_controller = source_target.coordinator
        reader = getattr(source_controller, "read_output_chunk", None)
        if not callable(reader):
            raise Qwen3NetworkError(
                "qwen3_network_output_unavailable", "source does not expose output chunk reads",
            )
        plan: dict[str, Any] | None = None
        source_reader: Any = None
        lease_active = False
        connection_failed = False
        try:
            pending = self._pending_output_transfers.get(reference["artifact_id"])
            if pending is not None:
                if (
                    pending.get("reference") != reference
                    or pending.get("source_node_id") != str(source_node_id)
                    or pending.get("target_node_id") != str(target_node_id)
                ):
                    raise Qwen3NetworkError(
                        "qwen3_network_output_lease_conflict",
                        "pending output transfer belongs to another boundary",
                    )
                plan = dict(pending["plan"])
            else:
                plan = target.coordinator.begin_receive(
                    base_url=target.base_url,
                    source_peer_id=str(source_node_id),
                    chain_id=str(chain_id),
                    generation=int(generation),
                    phase=str(phase),
                    from_segment=int(from_segment),
                    to_segment=int(to_segment),
                    size_bytes=reference["size_bytes"],
                    sha256=reference["sha256"],
                )
                self._pending_output_transfers[reference["artifact_id"]] = {
                    "plan": dict(plan),
                    "reference": dict(reference),
                    "source_node_id": str(source_node_id),
                    "target_node_id": str(target_node_id),
                }
            source_reader = source_controller
            try:
                from qwen3_pipeline_control import Qwen3LoopbackNetworkControlClient
                if isinstance(source_controller, Qwen3LoopbackNetworkControlClient):
                    signer = self.peer_signers.get(str(target_node_id))
                    if signer is None:
                        raise Qwen3NetworkError(
                            "qwen3_network_topology_unavailable",
                            "source output reader has no target peer signer",
                        )
                    source_reader = source_controller.for_peer(signer)
            except ImportError:
                pass

            binder = getattr(source_reader, "bind_output_reference", None)
            if callable(binder):
                binder(reference)

            lease = getattr(source_reader, "lease_output_reference", None)
            if not callable(lease):
                raise Qwen3NetworkError(
                    "qwen3_network_output_lease_unavailable",
                    "source output reader does not support persistent leases",
                )
            lease(reference["artifact_id"], plan["transfer_id"])
            lease_active = True

            def provider(offset: int, limit: int) -> Mapping[str, Any]:
                try:
                    if isinstance(source_reader, Qwen3NetworkTransferCoordinator):
                        return source_reader.read_output_chunk(
                            reference["artifact_id"],
                            requester_peer_id=str(target_node_id),
                            offset=offset,
                            max_bytes=limit,
                        )
                    return reader_for(source_reader, offset, limit)
                except Qwen3NetworkError as exc:
                    if exc.reason_code == "qwen3_transfer_connection_failed":
                        raise Qwen3TransferError(exc.reason_code, exc.reason) from exc
                    raise

            def reader_for(controller: Any, offset: int, limit: int) -> Mapping[str, Any]:
                result = controller.read_output_chunk(
                    reference["artifact_id"], offset=offset, max_bytes=limit,
                )
                if not isinstance(result, Mapping):
                    raise Qwen3NetworkError(
                        "qwen3_network_output_response_invalid", "source output reader returned invalid data",
                    )
                return result

            signer = self.peer_signers.get(str(source_node_id))
            if signer is None:
                raise Qwen3NetworkError(
                    "qwen3_network_topology_unavailable", "network source peer signer is missing",
                )
            client = Qwen3ArtifactTransferClient(
                self.artifact_root,
                target.requester,
                chunk_bytes=self.chunk_bytes,
                peer_proof_headers=signer.headers,
            )

            def progress(offset: int) -> None:
                try:
                    target.coordinator.record_transfer_progress(plan["transfer_id"], offset)
                    source_reader.record_output_progress(
                        reference["artifact_id"], plan["transfer_id"], offset,
                    )
                except Qwen3NetworkError as exc:
                    if exc.reason_code == "qwen3_transfer_connection_failed":
                        raise Qwen3TransferError(exc.reason_code, exc.reason) from exc
                    raise

            receipt = None
            for attempt in range(1, self.max_attempts + 1):
                connection_failed = False
                try:
                    receipt = client.upload_chunks(
                        plan=plan,
                        total_bytes=reference["size_bytes"],
                        sha256=reference["sha256"],
                        chunk_provider=provider,
                        progress_callback=progress,
                    )
                    break
                except Qwen3TransferError as exc:
                    connection_failed = exc.reason_code == "qwen3_transfer_connection_failed"
                    if not connection_failed or attempt >= self.max_attempts:
                        raise
            if not isinstance(receipt, Mapping) or receipt.get("status") != "committed":
                raise Qwen3NetworkError(
                    "qwen3_network_commit_failed", "registered output did not commit",
                )
            committed = validate_qwen3_artifact_reference(
                target.coordinator.commit_reference(plan["transfer_id"]),
            )
            if (
                committed["size_bytes"] != reference["size_bytes"]
                or committed["sha256"] != reference["sha256"]
                or committed["chain_id"] != reference["chain_id"]
                or committed["from_segment"] != reference["from_segment"]
                or committed["to_segment"] != reference["to_segment"]
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_digest_mismatch", "target reference differs from registered output",
                )
            source_reader.commit_output_reference(
                reference["artifact_id"], plan["transfer_id"],
            )
            self._pending_output_transfers.pop(reference["artifact_id"], None)
            self._active_transfers.append((target.coordinator, plan["transfer_id"]))
            return {
                "input_reference": dict(reference),
                "target_reference": committed,
                "transfer_id": plan["transfer_id"],
                "full_model_materialized": False,
            }
        except Qwen3NetworkError as exc:
            connection_failed = connection_failed or exc.reason_code == "qwen3_transfer_connection_failed"
            if plan is not None and not connection_failed:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
                self._pending_output_transfers.pop(reference["artifact_id"], None)
                if lease_active and source_reader is not None:
                    try:
                        source_reader.release_output_reference(reference["artifact_id"])
                    except Exception:
                        pass
            raise
        except Exception as exc:
            connection_failed = connection_failed or getattr(exc, "reason_code", "") == "qwen3_transfer_connection_failed"
            if plan is not None and not connection_failed:
                try:
                    target.coordinator.cancel_transfer(plan["transfer_id"])
                except Exception:
                    pass
                self._pending_output_transfers.pop(reference["artifact_id"], None)
                if lease_active and source_reader is not None:
                    try:
                        source_reader.release_output_reference(reference["artifact_id"])
                    except Exception:
                        pass
            raise Qwen3NetworkError(
                str(getattr(exc, "reason_code", "qwen3_network_transfer_failed")), str(exc),
            ) from exc

    def release_registered_output(
        self, output_reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Release one source output after the downstream node has committed it."""
        reference = validate_qwen3_artifact_reference(output_reference)
        source_target = self.targets.get(reference["source_node_id"])
        if source_target is None:
            raise Qwen3NetworkError(
                "qwen3_network_topology_unavailable", "registered output source is unavailable",
            )
        source_controller: Any = source_target.coordinator
        try:
            from qwen3_pipeline_control import Qwen3LoopbackNetworkControlClient
            if isinstance(source_controller, Qwen3LoopbackNetworkControlClient):
                signer = self.peer_signers.get(reference["target_node_id"])
                if signer is None:
                    raise Qwen3NetworkError(
                        "qwen3_network_topology_unavailable",
                        "registered output release has no target peer signer",
                    )
                source_controller = source_controller.for_peer(signer)
        except ImportError:
            pass
        binder = getattr(source_controller, "bind_output_reference", None)
        if callable(binder):
            binder(reference)
        releaser = getattr(source_controller, "release_output_reference", None)
        if not callable(releaser):
            raise Qwen3NetworkError(
                "qwen3_network_output_lease_unavailable", "source output release is unavailable",
            )
        return dict(releaser(reference["artifact_id"]))

    def transfer_and_consume(
        self,
        *,
        source_path: str | Path,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        source_node_id: str,
        target_node_id: str,
        batch_size: int,
        sequence_length: int,
        dtype: str,
        device: str,
        has_next_segment: bool,
    ) -> dict[str, Any]:
        """Transfer then execute at target, keeping both boundaries path-free."""
        reference = self.transfer_reference(
            source_path=source_path, chain_id=chain_id, generation=generation,
            phase=phase, from_segment=from_segment, to_segment=to_segment,
            source_node_id=source_node_id, target_node_id=target_node_id,
        )
        try:
            result = self.consume_target(
                target_node_id=target_node_id,
                transfer_id=reference["artifact_id"],
                phase=phase,
                generation=generation,
                batch_size=batch_size,
                sequence_length=sequence_length,
                dtype=dtype,
                device=device,
                has_next_segment=has_next_segment,
            )
        except Exception:
            try:
                self.targets[target_node_id].coordinator.cancel_transfer(reference["artifact_id"])
            except Exception:
                pass
            raise
        return {
            "input_reference": reference,
            "consume": result,
            "output_reference": result.get("output_reference"),
        }

    def execute_target_chain(
        self,
        *,
        source_path: str | Path,
        phase: str,
        generation: int,
        batch_size: int,
        sequence_length: int,
    ) -> dict[str, Any]:
        """Execute every remote target segment through path-free handoffs."""
        contract = self._contract
        if not self.target_execution:
            raise Qwen3NetworkError(
                "qwen3_network_execution_unavailable",
                "target-chain execution is not enabled for this transport",
            )
        if contract is None or self._phase != str(phase):
            raise Qwen3NetworkError(
                "qwen3_network_phase_invalid", "target-chain phase is not active",
            )
        if (
            phase not in {"prefill", "decode"}
            or isinstance(generation, bool) or not isinstance(generation, int)
            or int(generation) != int(contract["generation"]) + int(phase == "decode")
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
            or isinstance(sequence_length, bool) or not isinstance(sequence_length, int)
            or sequence_length <= 0
        ):
            raise Qwen3NetworkError(
                "qwen3_network_contract_mismatch", "target-chain dimensions are invalid",
            )
        segments = list(contract["segments"])
        if len(segments) not in {2, 3}:
            raise Qwen3NetworkError(
                "qwen3_network_contract_mismatch", "target-chain segment count is unsupported",
            )
        source = _inside(self.artifact_root, source_path)
        executions: list[dict[str, Any]] = []
        released_outputs: list[str] = []

        first_target = segments[1]
        first = self.transfer_and_consume(
            source_path=source,
            chain_id=contract["contract_sha256"],
            generation=int(generation),
            phase=str(phase),
            from_segment=0,
            to_segment=1,
            source_node_id=segments[0]["node_id"],
            target_node_id=first_target["node_id"],
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            dtype=str(first_target["dtype"]),
            device=str(first_target["execution_device"]),
            has_next_segment=len(segments) > 2,
        )
        consume = dict(first["consume"])
        executions.append({
            "segment_index": 1,
            "node_id": first_target["node_id"],
            "input_reference": dict(first["input_reference"]),
            "execution": dict(consume["execution"]),
            "hidden_handoff": dict(consume.get("hidden_handoff", {})),
            "kv_contract": dict(consume.get("kv_contract", {})),
            "output_reference": (
                dict(consume["output_reference"])
                if consume.get("output_reference") is not None else None
            ),
        })
        current_output = consume.get("output_reference")

        for target_index in range(2, len(segments)):
            if current_output is None:
                raise Qwen3NetworkError(
                    "qwen3_network_execution_invalid",
                    "target-chain output reference is missing before the final segment",
                )
            reference = validate_qwen3_artifact_reference(current_output)
            target_segment = segments[target_index]
            moved = self.transfer_registered_output(
                output_reference=reference,
                chain_id=contract["contract_sha256"],
                generation=int(generation),
                phase=str(phase),
                from_segment=target_index - 1,
                to_segment=target_index,
                source_node_id=segments[target_index - 1]["node_id"],
                target_node_id=target_segment["node_id"],
            )
            try:
                consume = self.consume_target(
                    target_node_id=target_segment["node_id"],
                    transfer_id=moved["transfer_id"],
                    phase=str(phase),
                    generation=int(generation),
                    batch_size=int(batch_size),
                    sequence_length=int(sequence_length),
                    dtype=str(target_segment["dtype"]),
                    device=str(target_segment["execution_device"]),
                    has_next_segment=target_index < len(segments) - 1,
                )
            except Exception as exc:
                if getattr(exc, "reason_code", "") != "qwen3_transfer_connection_failed":
                    try:
                        self.targets[target_segment["node_id"]].coordinator.cancel_transfer(
                            moved["transfer_id"],
                        )
                    except Exception:
                        pass
                    try:
                        self.release_registered_output(reference)
                    except Exception:
                        pass
                raise
            released = self.release_registered_output(reference)
            if released.get("status") not in {"released", "missing"}:
                raise Qwen3NetworkError(
                    "qwen3_network_output_release_failed",
                    "target-chain source output was not released",
                )
            released_outputs.append(reference["artifact_id"])
            executions.append({
                "segment_index": int(target_index),
                "node_id": target_segment["node_id"],
                "input_reference": dict(moved["target_reference"]),
                "execution": dict(consume["execution"]),
                "hidden_handoff": dict(consume.get("hidden_handoff", {})),
                "kv_contract": dict(consume.get("kv_contract", {})),
                "output_reference": (
                    dict(consume["output_reference"])
                    if consume.get("output_reference") is not None else None
                ),
            })
            current_output = consume.get("output_reference")

        if current_output is not None:
            raise Qwen3NetworkError(
                "qwen3_network_execution_invalid",
                "final target segment returned another output reference",
            )
        return {
            "schema_version": 1,
            "chain_id": contract["contract_sha256"],
            "generation": int(generation),
            "phase": str(phase),
            "segment_count": len(segments),
            "target_execution_count": len(executions),
            "executions": executions,
            "released_output_ids": released_outputs,
            "completed": True,
            "full_model_materialized": False,
        }

    def cleanup(self) -> dict[str, Any]:
        removed = 0
        failures = 0
        for node_id in sorted(self._activated_targets):
            coordinator = self.targets[node_id].coordinator
            try:
                result = coordinator.release()
            except Exception:
                failures += 1
                continue
            removed += int(result.get("removed_artifacts", 0) or 0)
            failures += int(result.get("cleanup_failures", 0) or 0)
        self._active_transfers.clear()
        self._pending_output_transfers.clear()
        self._activated_targets.clear()
        self._contract = None
        self._phase = "idle"
        return {
            "cleanup_complete": failures == 0,
            "removed_artifacts": removed,
            "cleanup_failures": failures,
        }

    def snapshot(self) -> dict[str, Any]:
        contract = self._contract
        return {
            "schema_version": 1,
            "active": contract is not None,
            "chain_id": "" if contract is None else contract["contract_sha256"],
            "generation": 0 if contract is None else contract["generation"],
            "phase": self._phase,
            "target_count": len(self.targets),
            "transfer_count": len(self._active_transfers),
            "target_execution": self.target_execution,
            "mode": "network",
            "full_model_materialized": False,
        }


__all__ = [
    "QWEN3_ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "Qwen3NetworkError",
    "Qwen3NetworkController",
    "Qwen3NetworkHandoffTransport",
    "Qwen3NetworkTarget",
    "Qwen3NetworkTransferCoordinator",
    "Qwen3ResolvedArtifact",
    "build_local_artifact_reference",
    "validate_qwen3_artifact_reference",
]
