"""Authenticated loopback HTTP adapter for Qwen3 pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from qwen3_pipeline_loopback import Qwen3LoopbackError, validate_loopback_base_url
from qwen3_pipeline_transfer import (
    MAX_TRANSFER_CHUNK_BYTES,
    QWEN3_TRANSFER_PREFIX,
    QWEN3_TRANSFER_SCHEMA_VERSION,
    Qwen3ArtifactReceiver,
    Qwen3TransferError,
    Qwen3TransferTicketSigner,
)


@dataclass
class Qwen3ArtifactTransferRuntime:
    """Own one node's bounded receive store and artifact ticket signer."""

    receiver: Qwen3ArtifactReceiver
    signer: Qwen3TransferTicketSigner
    authorization_gate: Callable[[str, str, int], None] | None = None

    @classmethod
    def create(
        cls,
        *,
        state_dir: str | Path,
        cluster_secret: str | bytes,
        clock: Callable[[], float] | None = None,
    ) -> "Qwen3ArtifactTransferRuntime":
        signer_options = {"clock": clock} if clock is not None else {}
        receiver_options = {"clock": clock} if clock is not None else {}
        signer = Qwen3TransferTicketSigner(cluster_secret, **signer_options)
        root = (
            Path(state_dir).expanduser().absolute().resolve(strict=False)
            / "qwen3"
            / "network_artifacts"
        )
        receiver = Qwen3ArtifactReceiver(
            root,
            signer=signer,
            **receiver_options,
        )
        receiver.reconcile_orphans()
        return cls(
            receiver=receiver,
            signer=signer,
        )

    def begin_receive(
        self,
        *,
        base_url: str,
        peer_node_id: str,
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
        try:
            safe_base_url = validate_loopback_base_url(base_url)
        except Qwen3LoopbackError as exc:
            raise Qwen3TransferError(exc.reason_code, exc.reason) from exc
        started = self.receiver.begin(
            peer_node_id=peer_node_id,
            chain_id=chain_id,
            generation=generation,
            phase=phase,
            from_segment=from_segment,
            to_segment=to_segment,
            size_bytes=size_bytes,
            sha256=sha256,
            ttl_seconds=ttl_seconds,
            peer_epoch=peer_epoch,
        )
        descriptor = started["descriptor"]
        return {
            "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
            "base_url": safe_base_url,
            "transfer_id": descriptor["transfer_id"],
            "ticket": started["ticket"],
            "descriptor": descriptor,
        }

    def snapshot(self) -> dict[str, Any]:
        result = self.receiver.snapshot()
        result["prefix"] = QWEN3_TRANSFER_PREFIX
        return result

    def cleanup_expired(self) -> dict[str, Any]:
        return self.receiver.cleanup_expired()

    def authorize_transfer(
        self, transfer_id: str, peer_node_id: str, peer_epoch: int = 0,
    ) -> None:
        gate = self.authorization_gate
        if gate is not None:
            gate(str(transfer_id), str(peer_node_id), int(peer_epoch))


router = APIRouter(prefix=QWEN3_TRANSFER_PREFIX, include_in_schema=False)


def _runtime(request: Request) -> Qwen3ArtifactTransferRuntime:
    runtime = getattr(request.app.state, "qwen3_artifact_transfer", None)
    if not isinstance(runtime, Qwen3ArtifactTransferRuntime):
        reason = getattr(
            request.app.state,
            "qwen3_artifact_transfer_reason",
            "not_initialized",
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "qwen3_transfer_disabled", "reason": str(reason)},
        )
    return runtime


def _authenticated_peer_identity(request: Request) -> tuple[str, int]:
    # This field must be injected by the authenticated peer transport. Never
    # derive node identity from a caller-controlled HTTP header.
    peer = request.scope.get("qlh_authenticated_peer_id")
    if not isinstance(peer, str) or not peer or len(peer) > 128:
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_transfer_peer_auth_missing"},
            headers={"WWW-Authenticate": "QLH-Peer"},
        )
    epoch = request.scope.get("qlh_authenticated_peer_epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_transfer_peer_epoch_missing"},
            headers={"WWW-Authenticate": "QLH-Peer"},
        )
    return peer, int(epoch)


def _bearer_ticket(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, ticket = authorization.partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not ticket
        or any(character.isspace() for character in ticket)
    ):
        raise HTTPException(
            status_code=401,
            detail={"code": "qwen3_transfer_ticket_missing"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ticket


def _http_error(exc: Qwen3TransferError) -> HTTPException:
    code = exc.reason_code
    if code in {
        "qwen3_transfer_ticket_invalid",
        "qwen3_transfer_ticket_signature",
        "qwen3_transfer_ticket_expired",
        "qwen3_transfer_identity_invalid",
    }:
        status = 401
    elif code in {
        "qwen3_transfer_peer_mismatch",
        "qwen3_transfer_scope_mismatch",
        "qwen3_transfer_peer_epoch_mismatch",
    }:
        status = 403
    elif code == "qwen3_transfer_missing":
        status = 404
    elif code in {
        "qwen3_transfer_incomplete",
        "qwen3_transfer_not_active",
        "qwen3_transfer_offset_mismatch",
        "qwen3_transfer_replay_mismatch",
    }:
        status = 409
    elif code == "qwen3_transfer_chunk_oversize":
        status = 413
    elif code == "qwen3_transfer_staging_failed":
        status = 507
    elif code == "qwen3_transfer_commit_failed":
        status = 500
    else:
        status = 422
    return HTTPException(
        status_code=status,
        detail={"code": code, "message": exc.reason},
    )


def _authorized(
    request: Request, transfer_id: str,
) -> tuple[Qwen3ArtifactTransferRuntime, str, str, int]:
    runtime = _runtime(request)
    runtime.cleanup_expired()
    ticket = _bearer_ticket(request)
    peer, peer_epoch = _authenticated_peer_identity(request)
    try:
        runtime.authorize_transfer(transfer_id, peer, peer_epoch)
    except Qwen3TransferError as exc:
        raise _http_error(exc) from exc
    return runtime, ticket, peer, peer_epoch


async def _bounded_chunk(request: Request) -> bytes:
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if content_type != "application/octet-stream":
        raise HTTPException(
            status_code=415,
            detail={"code": "qwen3_transfer_content_type_unsupported"},
        )
    raw_length = request.headers.get("Content-Length")
    if raw_length:
        try:
            declared = int(raw_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "qwen3_transfer_content_length_invalid"},
            ) from exc
        if declared < 0:
            raise HTTPException(
                status_code=400,
                detail={"code": "qwen3_transfer_content_length_invalid"},
            )
        if declared > MAX_TRANSFER_CHUNK_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "qwen3_transfer_chunk_oversize"},
            )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_TRANSFER_CHUNK_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "qwen3_transfer_chunk_oversize"},
            )
    if not body:
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_transfer_chunk_empty"},
        )
    return bytes(body)


@router.get("/status")
async def transfer_status(request: Request):
    runtime = getattr(request.app.state, "qwen3_artifact_transfer", None)
    if not isinstance(runtime, Qwen3ArtifactTransferRuntime):
        return {
            "enabled": False,
            "reason": getattr(
                request.app.state,
                "qwen3_artifact_transfer_reason",
                "not_initialized",
            ),
            "prefix": QWEN3_TRANSFER_PREFIX,
        }
    await run_in_threadpool(runtime.cleanup_expired)
    return await run_in_threadpool(runtime.snapshot)


@router.get("/{transfer_id}")
async def receive_status(transfer_id: str, request: Request):
    runtime, ticket, peer, peer_epoch = _authorized(request, transfer_id)
    try:
        result = await run_in_threadpool(
            runtime.receiver.status,
            transfer_id,
            ticket=ticket,
            authenticated_peer_id=peer,
            authenticated_peer_epoch=peer_epoch,
        )
    except Qwen3TransferError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(
        result,
        headers={"Upload-Offset": str(result["received_bytes"]), "Cache-Control": "no-store"},
    )


@router.patch("/{transfer_id}")
async def receive_chunk(
    transfer_id: str,
    request: Request,
    upload_offset: str = Header(alias="Upload-Offset"),
):
    runtime, ticket, peer, peer_epoch = _authorized(request, transfer_id)
    try:
        offset = int(upload_offset)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_transfer_offset_invalid"},
        ) from exc
    chunk = await _bounded_chunk(request)
    try:
        result = await run_in_threadpool(
            runtime.receiver.write,
            transfer_id,
            ticket=ticket,
            authenticated_peer_id=peer,
            authenticated_peer_epoch=peer_epoch,
            offset=offset,
            data=chunk,
        )
    except Qwen3TransferError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(
        result,
        headers={"Upload-Offset": str(result["received_bytes"]), "Cache-Control": "no-store"},
    )


@router.post("/{transfer_id}/commit")
async def commit_receive(transfer_id: str, request: Request):
    runtime, ticket, peer, peer_epoch = _authorized(request, transfer_id)
    if request.headers.get("Content-Length") not in {None, "", "0"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_transfer_commit_body_not_allowed"},
        )
    try:
        result = await run_in_threadpool(
            runtime.receiver.commit,
            transfer_id,
            ticket=ticket,
            authenticated_peer_id=peer,
            authenticated_peer_epoch=peer_epoch,
        )
    except Qwen3TransferError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.delete("/{transfer_id}")
async def cancel_receive(transfer_id: str, request: Request):
    runtime, ticket, peer, peer_epoch = _authorized(request, transfer_id)
    if request.headers.get("Content-Length") not in {None, "", "0"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "qwen3_transfer_cancel_body_not_allowed"},
        )
    try:
        result = await run_in_threadpool(
            runtime.receiver.cancel,
            transfer_id,
            ticket=ticket,
            authenticated_peer_id=peer,
            authenticated_peer_epoch=peer_epoch,
        )
    except Qwen3TransferError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


__all__ = [
    "Qwen3ArtifactTransferRuntime",
    "router",
]
