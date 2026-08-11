"""Attempt-scoped HTTP data plane for distributed diffusion image blobs."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from .distributed import (
    MAX_TRANSFER_GRANT_SECONDS,
    BlobAuthorizationError,
    BlobConflict,
    BlobNotFound,
    BlobTransferTokenSigner,
    BlobValidationError,
    DistributedBlobError,
    PersistentImageBlobStore,
)


DATA_PLANE_PREFIX = "/internal/v1/diffusion/data-plane"
MAX_TRANSFER_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass
class DiffusionDataPlaneRuntime:
    """Own the persistent store and grant signer for one node process."""

    store: PersistentImageBlobStore
    signer: BlobTransferTokenSigner

    @classmethod
    def create(
        cls,
        *,
        state_dir: str | Path,
        cluster_secret: str | bytes,
        store_options: Optional[dict[str, Any]] = None,
        clock=None,
    ) -> "DiffusionDataPlaneRuntime":
        secret = (
            cluster_secret.encode("utf-8")
            if isinstance(cluster_secret, str)
            else bytes(cluster_secret)
        )
        if len(secret) < 32:
            raise ValueError("diffusion data plane requires a 32+ byte cluster secret")
        root = Path(state_dir).expanduser().resolve() / "diffusion" / "distributed_blobs"
        options = dict(store_options or {})
        if clock is not None:
            options["clock"] = clock
        store = PersistentImageBlobStore(root, **options)
        signer_options = {"clock": clock} if clock is not None else {}
        return cls(
            store=store,
            signer=BlobTransferTokenSigner(secret, **signer_options),
        )

    def begin_upload(
        self,
        *,
        attempt_id: str,
        grant_ttl_seconds: float,
        **blob_fields: Any,
    ) -> dict[str, Any]:
        session = self.store.begin_upload(**blob_fields)
        grant = self.signer.issue(
            upload_id=session.upload_id,
            attempt_id=attempt_id,
            direction="upload",
            ttl_seconds=grant_ttl_seconds,
        )
        return {"upload": session.snapshot(), "grant": grant}

    def grant_download(
        self,
        blob_id: str,
        *,
        attempt_id: str,
        ttl_seconds: float,
    ) -> dict[str, Any]:
        lease = self.store.acquire_lease(
            blob_id,
            attempt_id=attempt_id,
            ttl_seconds=ttl_seconds,
        )
        grant = self.signer.issue(
            blob_id=blob_id,
            lease_id=lease.lease_id,
            attempt_id=attempt_id,
            direction="download",
            ttl_seconds=ttl_seconds,
        )
        return {"lease": lease.snapshot(), "grant": grant}

    def publish_output(
        self,
        data: bytes,
        *,
        attempt_id: str,
        base_url: str,
        grant_ttl_seconds: float,
        owner_scope: str,
        content_type: str,
        width: int,
        height: int,
        metadata: Optional[dict[str, Any]] = None,
        parent_blob_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Store a worker result and return its one-use-on-the-wire plan."""
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("diffusion output publication requires a base URL")
        descriptor = self.store.put_bytes(
            data,
            content_type=content_type,
            purpose="output",
            owner_scope=owner_scope,
            width=width,
            height=height,
            metadata=metadata,
            parent_blob_ids=parent_blob_ids,
            deduplicate=not parent_blob_ids,
        )
        download = self.grant_download(
            descriptor.blob_id,
            attempt_id=attempt_id,
            ttl_seconds=grant_ttl_seconds,
        )
        return {
            "descriptor": {
                key: descriptor.snapshot()[key]
                for key in (
                    "blob_id", "sha256", "size_bytes", "content_type", "width", "height", "purpose",
                )
            },
            "transfer_plan": {
                "base_url": base_url.rstrip("/"),
                "downloads": [{
                    "blob_id": descriptor.blob_id,
                    "lease_id": download["lease"]["lease_id"],
                    "grant": download["grant"],
                }],
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "prefix": DATA_PLANE_PREFIX,
            "max_chunk_bytes": MAX_TRANSFER_CHUNK_BYTES,
            "max_grant_seconds": MAX_TRANSFER_GRANT_SECONDS,
            "store": self.store.snapshot(),
        }

    def close(self) -> None:
        self.store.close()


router = APIRouter(prefix=DATA_PLANE_PREFIX, include_in_schema=False)


def _runtime(request: Request) -> DiffusionDataPlaneRuntime:
    runtime = getattr(request.app.state, "diffusion_data_plane", None)
    if not isinstance(runtime, DiffusionDataPlaneRuntime):
        reason = getattr(
            request.app.state,
            "diffusion_data_plane_reason",
            "not_initialized",
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "data_plane_disabled", "reason": str(reason)},
        )
    return runtime


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not token
        or any(character.isspace() for character in token)
    ):
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_transfer_grant"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _verify_grant(
    request: Request,
    *,
    attempt_id: str,
    direction: str,
) -> tuple[DiffusionDataPlaneRuntime, dict[str, Any]]:
    runtime = _runtime(request)
    try:
        payload = runtime.signer.verify(
            _bearer_token(request),
            direction=direction,
            attempt_id=attempt_id,
        )
    except BlobAuthorizationError as exc:
        raise HTTPException(
            status_code=401,
            detail={"code": exc.code},
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return runtime, payload


def _scope_matches(actual: Any, expected: str) -> bool:
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


def _http_error(exc: DistributedBlobError) -> HTTPException:
    if isinstance(exc, BlobAuthorizationError):
        status = 403
    elif isinstance(exc, BlobNotFound):
        status = 404
    elif isinstance(exc, BlobConflict):
        status = 409
    elif isinstance(exc, BlobValidationError):
        status = 422
    else:
        status = 500
    return HTTPException(
        status_code=status,
        detail={"code": exc.code, "message": str(exc)},
    )


async def _bounded_chunk(request: Request) -> bytes:
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if content_type != "application/octet-stream":
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_transfer_content_type"},
        )
    raw_length = request.headers.get("Content-Length")
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_content_length"},
            ) from exc
        if declared_length < 0:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_content_length"},
            )
        if declared_length > MAX_TRANSFER_CHUNK_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "transfer_chunk_too_large"},
            )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_TRANSFER_CHUNK_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"code": "transfer_chunk_too_large"},
            )
    if not body:
        raise HTTPException(status_code=400, detail={"code": "empty_transfer_chunk"})
    return bytes(body)


@router.get("/status")
async def data_plane_status(request: Request):
    runtime = getattr(request.app.state, "diffusion_data_plane", None)
    if not isinstance(runtime, DiffusionDataPlaneRuntime):
        return {
            "enabled": False,
            "reason": getattr(
                request.app.state,
                "diffusion_data_plane_reason",
                "not_initialized",
            ),
            "prefix": DATA_PLANE_PREFIX,
        }
    return await run_in_threadpool(runtime.snapshot)


@router.get("/attempts/{attempt_id}/uploads/{upload_id}")
async def upload_status(attempt_id: str, upload_id: str, request: Request):
    runtime, grant = _verify_grant(
        request,
        attempt_id=attempt_id,
        direction="upload",
    )
    if not _scope_matches(grant.get("upload_id"), upload_id):
        raise HTTPException(status_code=403, detail={"code": "transfer_scope_mismatch"})
    try:
        session = await run_in_threadpool(runtime.store.upload_session, upload_id)
    except DistributedBlobError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(
        session.snapshot(),
        headers={"Upload-Offset": str(session.received_bytes), "Cache-Control": "no-store"},
    )


@router.patch("/attempts/{attempt_id}/uploads/{upload_id}")
async def upload_chunk(
    attempt_id: str,
    upload_id: str,
    request: Request,
    upload_offset: str = Header(alias="Upload-Offset"),
):
    runtime, grant = _verify_grant(
        request,
        attempt_id=attempt_id,
        direction="upload",
    )
    if not _scope_matches(grant.get("upload_id"), upload_id):
        raise HTTPException(status_code=403, detail={"code": "transfer_scope_mismatch"})
    try:
        offset = int(upload_offset)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_upload_offset"},
        ) from exc
    chunk = await _bounded_chunk(request)
    try:
        session = await run_in_threadpool(
            runtime.store.write_upload,
            upload_id,
            offset=offset,
            data=chunk,
        )
    except DistributedBlobError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(
        session.snapshot(),
        headers={"Upload-Offset": str(session.received_bytes), "Cache-Control": "no-store"},
    )


@router.post("/attempts/{attempt_id}/uploads/{upload_id}/commit")
async def commit_upload(attempt_id: str, upload_id: str, request: Request):
    runtime, grant = _verify_grant(
        request,
        attempt_id=attempt_id,
        direction="upload",
    )
    if not _scope_matches(grant.get("upload_id"), upload_id):
        raise HTTPException(status_code=403, detail={"code": "transfer_scope_mismatch"})
    if request.headers.get("Content-Length") not in {None, "", "0"}:
        raise HTTPException(
            status_code=400,
            detail={"code": "commit_body_not_allowed"},
        )
    try:
        descriptor = await run_in_threadpool(runtime.store.commit_upload, upload_id)
    except DistributedBlobError as exc:
        raise _http_error(exc) from exc
    return JSONResponse(descriptor.snapshot(), headers={"Cache-Control": "no-store"})


@router.get("/attempts/{attempt_id}/blobs/{blob_id}")
async def download_blob_chunk(
    attempt_id: str,
    blob_id: str,
    request: Request,
    lease_id: str = Header(alias="X-Blob-Lease"),
    offset: int = Query(default=0, ge=0),
    length: int = Query(default=MAX_TRANSFER_CHUNK_BYTES, ge=1, le=MAX_TRANSFER_CHUNK_BYTES),
):
    runtime, grant = _verify_grant(
        request,
        attempt_id=attempt_id,
        direction="download",
    )
    if (
        not _scope_matches(grant.get("blob_id"), blob_id)
        or not _scope_matches(grant.get("lease_id"), lease_id)
    ):
        raise HTTPException(status_code=403, detail={"code": "transfer_scope_mismatch"})
    try:
        descriptor = await run_in_threadpool(runtime.store.descriptor, blob_id)
        if offset >= descriptor.size_bytes:
            raise HTTPException(
                status_code=416,
                detail={"code": "blob_offset_out_of_range"},
                headers={"Content-Range": f"bytes */{descriptor.size_bytes}"},
            )
        data = await run_in_threadpool(
            runtime.store.read_chunk,
            blob_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            offset=offset,
            length=length,
        )
    except HTTPException:
        raise
    except DistributedBlobError as exc:
        raise _http_error(exc) from exc
    end = offset + len(data) - 1
    partial = offset != 0 or len(data) != descriptor.size_bytes
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Content-Range": f"bytes {offset}-{end}/{descriptor.size_bytes}",
        "X-Blob-SHA256": descriptor.sha256,
        "X-Content-Type-Options": "nosniff",
    }
    return Response(
        data,
        status_code=206 if partial else 200,
        media_type=descriptor.content_type,
        headers=headers,
    )


__all__ = [
    "DATA_PLANE_PREFIX",
    "MAX_TRANSFER_CHUNK_BYTES",
    "DiffusionDataPlaneRuntime",
    "router",
]
