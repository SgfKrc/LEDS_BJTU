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
        self._control_peer_epoch: int = 0
        self._phase = "idle"
        self.runtime.authorization_gate = self.authorize_transfer

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
            if int(canonical["generation"]) <= self._last_generation:
                raise Qwen3NetworkError(
                    "qwen3_network_generation_stale", "Qwen3 network generation is stale",
                )
            self._contract = canonical
            self._last_generation = int(canonical["generation"])
            self._transfers.clear()
            self._phase = "prepared"
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
            if int(peer_epoch) > 0 and self._control_peer_epoch not in {0, int(peer_epoch)}:
                raise Qwen3NetworkError(
                    "qwen3_network_peer_scope", "network control peer registration epoch is stale",
                )
            self._control_peer_epoch = int(peer_epoch)
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
                and transfer.get("status") != "committed"
                for transfer in self._transfers.values()
            ):
                raise Qwen3NetworkError(
                    "qwen3_network_phase_incomplete",
                    "network execution phase still has incomplete transfers",
                )
            self._phase = "prefilled" if phase == "prefill" else "decoded"
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

    def resolve(self, transfer_id: str) -> Qwen3ResolvedArtifact:
        with self._lock:
            contract = self._active()
            transfer = self._transfers.get(str(transfer_id))
            if transfer is None:
                raise Qwen3NetworkError(
                    "qwen3_network_transfer_missing", "network transfer is not active",
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
            if transfer is None or transfer.get("status") not in {"committed", "consumed"}:
                raise Qwen3NetworkError("qwen3_network_transfer_missing", "network transfer is not committed")
            descriptor = transfer["descriptor"]
            if (
                descriptor["phase"] != phase
                or int(descriptor["generation"]) != int(generation)
                or self._phase != phase
            ):
                raise Qwen3NetworkError("qwen3_network_contract_mismatch", "target execution contract does not match")
            resolved = self.resolve(str(transfer_id))
            request = {
                "chain_id": contract["contract_sha256"],
                "generation": int(generation),
                "phase": str(phase),
                "batch_size": int(batch_size),
                "sequence_length": int(sequence_length),
                "dtype": str(dtype),
                "device": str(device),
                "has_next_segment": bool(has_next_segment),
                "reference": dict(resolved.reference),
            }
            metadata: Mapping[str, Any] = {}
            if executor is not None:
                result = executor(resolved.path, request)
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
            transfer["status"] = "consumed"
            return {
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
                },
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

    def consume(self, transfer_id: str, **fields) -> dict[str, Any]:
        """Local-controller spelling matching the remote control client API."""
        return self.consume_transfer(str(transfer_id), **fields)

    def cancel_transfer(self, transfer_id: str) -> dict[str, Any]:
        with self._lock:
            result = self.runtime.receiver.discard(str(transfer_id))
            self._transfers.pop(str(transfer_id), None)
            return result

    def release(self) -> dict[str, Any]:
        with self._lock:
            removed = 0
            failures = 0
            for transfer_id in list(self._transfers):
                result = self.runtime.receiver.discard(transfer_id)
                removed += int(result.get("removed_artifacts", 0) or 0)
                failures += int(result.get("cleanup_failures", 0) or 0)
            self._transfers.clear()
            self._contract = None
            self._phase = "idle"
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
        self._contract: dict[str, Any] | None = None
        self._active_transfers: list[tuple[Qwen3NetworkController, str]] = []
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
        return dict(result)

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
