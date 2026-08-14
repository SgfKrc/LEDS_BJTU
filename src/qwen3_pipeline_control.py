"""Authenticated loopback control adapter for Qwen3 network handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from qwen3_pipeline_network import (
    Qwen3NetworkError,
    Qwen3NetworkTransferCoordinator,
    Qwen3ResolvedArtifact,
    validate_qwen3_artifact_reference,
)
from qwen3_pipeline_peer_auth import (
    QWEN3_NETWORK_CONTROL_PREFIX,
    Qwen3PeerRequestSigner,
)
from qwen3_pipeline_transaction import (
    MAX_CONTRACT_BYTES,
    Qwen3PipelineProtocolError,
    validate_qwen3_dry_run_contract,
)
from qwen3_pipeline_transfer import (
    MAX_CONTROL_RESPONSE_BYTES,
    Qwen3TransferError,
    TransferRequester,
    default_transfer_request,
)


router = APIRouter(prefix=QWEN3_NETWORK_CONTROL_PREFIX, include_in_schema=False)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3NetworkError(
            "qwen3_network_control_invalid", "network control payload is not strict JSON",
        ) from exc
    if not encoded or len(encoded) > MAX_CONTRACT_BYTES:
        raise Qwen3NetworkError(
            "qwen3_network_control_oversize", "network control payload exceeds its limit",
        )
    return encoded


def _coordinator(request: Request) -> Qwen3NetworkTransferCoordinator:
    value = getattr(request.app.state, "qwen3_network_transfer_coordinator", None)
    if not isinstance(value, Qwen3NetworkTransferCoordinator):
        raise HTTPException(
            status_code=503,
            detail={"code": "qwen3_network_control_disabled"},
        )
    return value


def _peer(request: Request) -> str:
    value = request.scope.get("qlh_authenticated_peer_id")
    if not isinstance(value, str) or not value or len(value) > 128:
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_network_control_peer_auth_missing"},
        )
    return value


def _peer_identity(request: Request) -> tuple[str, int]:
    peer = _peer(request)
    epoch = request.scope.get("qlh_authenticated_peer_epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_network_control_peer_epoch_missing"},
        )
    return peer, int(epoch)


async def _payload(request: Request, expected: set[str]) -> dict[str, Any]:
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail={"code": "qwen3_network_control_content_type_unsupported"},
        )
    raw_length = request.headers.get("Content-Length", "")
    if raw_length:
        try:
            declared = int(raw_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "qwen3_network_control_content_length_invalid"},
            ) from exc
        if not 0 < declared <= MAX_CONTRACT_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "qwen3_network_control_oversize"},
            )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_CONTRACT_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "qwen3_network_control_oversize"},
            )
    if raw_length and declared != len(body):
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_network_control_content_length_invalid"},
        )
    digest = hashlib.sha256(body).hexdigest()
    authorization = request.headers.get("Authorization", "")
    scheme, separator, bearer = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or bearer != digest:
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_network_control_body_binding_mismatch"},
        )
    try:
        value = json.loads(bytes(body).decode("utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_network_control_invalid"},
        ) from exc
    if not isinstance(value, dict) or set(value) != expected:
        raise HTTPException(
            status_code=422,
            detail={"code": "qwen3_network_control_fields_invalid"},
        )
    return value


def _http_error(exc: Qwen3NetworkError) -> HTTPException:
    if exc.reason_code in {"qwen3_network_peer_scope"}:
        status = 403
    elif exc.reason_code in {
        "qwen3_network_contract_mismatch",
        "qwen3_network_generation_stale",
        "qwen3_network_phase_invalid",
        "qwen3_network_phase_incomplete",
        "qwen3_network_contract_active",
        "qwen3_network_contract_inactive",
    }:
        status = 409
    elif exc.reason_code in {
        "qwen3_network_artifact_missing",
        "qwen3_network_transfer_missing",
    }:
        status = 404
    else:
        status = 422
    return HTTPException(
        status_code=status,
        detail={"code": exc.reason_code, "message": exc.reason},
    )


def _base_generation(payload: Mapping[str, Any]) -> int:
    generation = payload.get("generation")
    phase = payload.get("phase")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or phase not in {"prefill", "decode"}
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": "qwen3_network_control_fields_invalid"},
        )
    return generation - int(phase == "decode")


async def _run(callable_, *args, **kwargs):
    try:
        return await run_in_threadpool(callable_, *args, **kwargs)
    except Qwen3NetworkError as exc:
        raise _http_error(exc) from exc
    except Qwen3TransferError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.reason_code, "message": exc.reason},
        ) from exc
    except Qwen3PipelineProtocolError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "qwen3_network_contract_invalid", "message": str(exc)},
        ) from exc


@router.post("/activate")
async def activate_network_contract(request: Request):
    payload = await _payload(request, {"contract"})
    coordinator = _coordinator(request)
    peer, peer_epoch = _peer_identity(request)
    contract = payload["contract"]
    await _run(coordinator.authorize_control_peer, peer, contract=contract, peer_epoch=peer_epoch)
    return JSONResponse(await _run(coordinator.activate, contract), headers={"Cache-Control": "no-store"})


@router.post("/begin-phase")
async def begin_network_phase(request: Request):
    payload = await _payload(request, {"chain_id", "generation", "phase"})
    coordinator = _coordinator(request)
    await _run(
        coordinator.authorize_control_peer,
        _peer_identity(request)[0],
        peer_epoch=_peer_identity(request)[1],
        chain_id=payload["chain_id"],
        generation=_base_generation(payload),
    )
    result = await _run(coordinator.begin_phase, payload["phase"], payload["generation"])
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/finish-phase")
async def finish_network_phase(request: Request):
    payload = await _payload(request, {"chain_id", "generation", "phase"})
    coordinator = _coordinator(request)
    await _run(
        coordinator.authorize_control_peer,
        _peer_identity(request)[0],
        peer_epoch=_peer_identity(request)[1],
        chain_id=payload["chain_id"],
        generation=_base_generation(payload),
    )
    result = await _run(coordinator.finish_phase, payload["phase"], payload["generation"])
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/begin-receive")
async def begin_network_receive(request: Request):
    fields = {
        "base_url", "chain_id", "generation", "phase", "from_segment",
        "to_segment", "size_bytes", "sha256", "ttl_seconds",
    }
    payload = await _payload(request, fields)
    coordinator = _coordinator(request)
    peer, peer_epoch = _peer_identity(request)
    await _run(
        coordinator.authorize_control_peer,
        peer,
        chain_id=payload["chain_id"],
        generation=_base_generation(payload),
        peer_epoch=peer_epoch,
    )
    result = await _run(
        coordinator.begin_receive,
        base_url=payload["base_url"],
        source_peer_id=peer,
        chain_id=payload["chain_id"],
        generation=payload["generation"],
        phase=payload["phase"],
        from_segment=payload["from_segment"],
        to_segment=payload["to_segment"],
        size_bytes=payload["size_bytes"],
        sha256=payload["sha256"],
        ttl_seconds=payload["ttl_seconds"],
        peer_epoch=peer_epoch,
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/resolve")
async def resolve_network_receive(request: Request):
    payload = await _payload(request, {"chain_id", "generation", "transfer_id"})
    coordinator = _coordinator(request)
    await _run(
        coordinator.authorize_control_peer,
        _peer_identity(request)[0],
        peer_epoch=_peer_identity(request)[1],
        chain_id=payload["chain_id"],
        generation=payload["generation"],
    )
    resolved = await _run(coordinator.resolve, payload["transfer_id"])
    return JSONResponse(
        {"reference": resolved.reference},
        headers={"Cache-Control": "no-store"},
    )


@router.post("/consume")
async def consume_network_receive(request: Request):
    """Execute the next sidecar segment with a target-local artifact path."""
    fields = {
        "chain_id", "generation", "phase", "transfer_id", "batch_size",
        "sequence_length", "dtype", "device", "has_next_segment",
    }
    payload = await _payload(request, fields)
    coordinator = _coordinator(request)
    peer, peer_epoch = _peer_identity(request)
    await _run(
        coordinator.authorize_control_peer,
        peer,
        chain_id=payload["chain_id"],
        generation=_base_generation({"generation": payload["generation"], "phase": payload["phase"]}),
        peer_epoch=peer_epoch,
    )
    executor = getattr(request.app.state, "qwen3_network_sidecar_executor", None)
    result = await _run(
        coordinator.consume_transfer,
        payload["transfer_id"],
        phase=payload["phase"],
        generation=payload["generation"],
        batch_size=payload["batch_size"],
        sequence_length=payload["sequence_length"],
        dtype=payload["dtype"],
        device=payload["device"],
        has_next_segment=payload["has_next_segment"],
        executor=executor if callable(executor) else None,
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
@router.post("/cancel")
async def cancel_network_receive(request: Request):
    payload = await _payload(request, {"chain_id", "generation", "transfer_id"})
    coordinator = _coordinator(request)
    await _run(
        coordinator.authorize_control_peer,
        _peer_identity(request)[0],
        peer_epoch=_peer_identity(request)[1],
        chain_id=payload["chain_id"],
        generation=payload["generation"],
    )
    result = await _run(coordinator.cancel_transfer, payload["transfer_id"])
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/release")
async def release_network_contract(request: Request):
    payload = await _payload(request, {"chain_id", "generation"})
    coordinator = _coordinator(request)
    await _run(
        coordinator.authorize_control_peer,
        _peer_identity(request)[0],
        peer_epoch=_peer_identity(request)[1],
        chain_id=payload["chain_id"],
        generation=payload["generation"],
    )
    result = await _run(coordinator.release)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


class Qwen3LoopbackNetworkControlClient:
    """Control one target process without sharing its coordinator object."""

    def __init__(
        self,
        *,
        node_id: str,
        base_url: str,
        artifact_root: str | Path,
        signer: Qwen3PeerRequestSigner,
        requester: TransferRequester | None = None,
    ) -> None:
        from qwen3_pipeline_loopback import validate_loopback_base_url

        self.local_node_id = str(node_id)
        self.base_url = validate_loopback_base_url(base_url)
        self.artifact_root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        self.signer = signer
        self._requester = requester or default_transfer_request
        self._contract: dict[str, Any] | None = None
        self._phase = "idle"

    def _call(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = _canonical_bytes(dict(payload))
        digest = hashlib.sha256(body).hexdigest()
        url = f"{self.base_url}{QWEN3_NETWORK_CONTROL_PREFIX}/{action}"
        headers = {
            "Authorization": f"Bearer {digest}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Accept-Encoding": "identity",
            **self.signer.headers("POST", url, digest),
        }
        try:
            response = self._requester("POST", url, headers, body)
        except Qwen3TransferError as exc:
            raise Qwen3NetworkError(exc.reason_code, exc.reason) from exc
        if len(response.content) > MAX_CONTROL_RESPONSE_BYTES:
            raise Qwen3NetworkError(
                "qwen3_network_control_response_oversize",
                "network control response exceeds its limit",
            )
        try:
            decoded = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Qwen3NetworkError(
                "qwen3_network_control_response_invalid",
                "network control response is invalid",
            ) from exc
        if response.status_code != 200:
            detail = decoded.get("detail", {}) if isinstance(decoded, dict) else {}
            code = detail.get("code", "qwen3_network_control_failed") if isinstance(detail, dict) else "qwen3_network_control_failed"
            message = detail.get("message", "network control request failed") if isinstance(detail, dict) else "network control request failed"
            raise Qwen3NetworkError(str(code), str(message))
        if not isinstance(decoded, dict):
            raise Qwen3NetworkError(
                "qwen3_network_control_response_invalid",
                "network control response is not an object",
            )
        return decoded

    def activate(self, contract: dict[str, Any]) -> dict[str, Any]:
        canonical = validate_qwen3_dry_run_contract(contract)
        result = self._call("activate", {"contract": canonical})
        self._contract = canonical
        self._phase = "prepared"
        return result

    def _identity(self) -> tuple[str, int]:
        if self._contract is None:
            raise Qwen3NetworkError(
                "qwen3_network_contract_inactive", "network control client is inactive",
            )
        return self._contract["contract_sha256"], int(self._contract["generation"])

    def begin_phase(self, phase: str, generation: int) -> dict[str, Any]:
        chain_id, _ = self._identity()
        result = self._call(
            "begin-phase",
            {"chain_id": chain_id, "generation": int(generation), "phase": str(phase)},
        )
        self._phase = str(phase)
        return result

    def finish_phase(self, phase: str, generation: int) -> dict[str, Any]:
        chain_id, _ = self._identity()
        result = self._call(
            "finish-phase",
            {"chain_id": chain_id, "generation": int(generation), "phase": str(phase)},
        )
        self._phase = "prefilled" if phase == "prefill" else "decoded"
        return result

    def begin_receive(self, **fields) -> dict[str, Any]:
        source_peer_id = str(fields.pop("source_peer_id", ""))
        if source_peer_id != self.signer.peer_node_id:
            raise Qwen3NetworkError(
                "qwen3_network_peer_scope", "network control signer differs from source peer",
            )
        payload = dict(fields)
        payload["ttl_seconds"] = float(payload.get("ttl_seconds", 60))
        return self._call("begin-receive", payload)

    def resolve(self, transfer_id: str) -> Qwen3ResolvedArtifact:
        chain_id, generation = self._identity()
        decoded = self._call(
            "resolve",
            {
                "chain_id": chain_id,
                "generation": generation,
                "transfer_id": str(transfer_id),
            },
        )
        reference = validate_qwen3_artifact_reference(decoded.get("reference", {}))
        if reference["artifact_id"] != str(transfer_id):
            raise Qwen3NetworkError(
                "qwen3_network_reference_invalid", "resolved transfer identity changed",
            )
        path = self.artifact_root.joinpath(f"{transfer_id}.pt").resolve(strict=False)
        try:
            path.relative_to(self.artifact_root)
        except ValueError as exc:
            raise Qwen3NetworkError(
                "qwen3_network_artifact_scope", "resolved artifact escapes target root",
            ) from exc
        if not path.is_file():
            raise Qwen3NetworkError(
                "qwen3_network_artifact_missing", "resolved artifact is unavailable",
            )
        return Qwen3ResolvedArtifact(path=path, reference=reference)

    def consume(
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
    ) -> dict[str, Any]:
        """Ask the target sidecar to consume a transfer without returning a path."""
        chain_id, _ = self._identity()
        result = self._call(
            "consume",
            {
                "chain_id": chain_id,
                "generation": int(generation),
                "phase": str(phase),
                "transfer_id": str(transfer_id),
                "batch_size": int(batch_size),
                "sequence_length": int(sequence_length),
                "dtype": str(dtype),
                "device": str(device),
                "has_next_segment": bool(has_next_segment),
            },
        )
        if any(key in json.dumps(result, ensure_ascii=True).lower() for key in ("path", "\\\\", "/users/", "/home/")):
            raise Qwen3NetworkError(
                "qwen3_network_control_response_invalid",
                "target execution response leaked a filesystem path",
            )
        return result

    def cancel_transfer(self, transfer_id: str) -> dict[str, Any]:
        chain_id, generation = self._identity()
        return self._call(
            "cancel",
            {
                "chain_id": chain_id,
                "generation": generation,
                "transfer_id": str(transfer_id),
            },
        )

    def release(self) -> dict[str, Any]:
        if self._contract is None:
            return {
                "cleanup_complete": True,
                "removed_artifacts": 0,
                "cleanup_failures": 0,
            }
        chain_id, generation = self._identity()
        try:
            return self._call(
                "release", {"chain_id": chain_id, "generation": generation},
            )
        finally:
            self._contract = None
            self._phase = "idle"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "active": self._contract is not None,
            "local_node_id": self.local_node_id,
            "chain_id": "" if self._contract is None else self._contract["contract_sha256"],
            "generation": 0 if self._contract is None else self._contract["generation"],
            "phase": self._phase,
            "mode": "loopback_process_control",
            "full_model_materialized": False,
        }


__all__ = [
    "Qwen3LoopbackNetworkControlClient",
    "router",
]
