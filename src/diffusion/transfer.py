"""Verified HTTP transfer helpers for v3 diffusion Blob descriptors.

The wire protocol carries only descriptors and short-lived grants.  This
module consumes the grants in memory and returns verified bytes or a new local
descriptor; callers must never place the transfer plan in workflow metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .data_plane import DATA_PLANE_PREFIX, MAX_TRANSFER_CHUNK_BYTES
from .distributed import PersistentImageBlobStore


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BLOB_ID = re.compile(r"^img_[A-Za-z0-9_-]{16,96}$")
_LEASE_ID = re.compile(r"^bls_[A-Za-z0-9_-]{16,96}$")


class DiffusionTransferError(RuntimeError):
    """Safe, stable failure used by Provider retries and worker errors."""

    def __init__(self, message: str, *, code: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TransferResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes


TransferRequester = Callable[[str, str, Mapping[str, str], bytes | None], TransferResponse]


def _header(headers: Mapping[str, str], name: str) -> str:
    needle = name.lower()
    for key, value in headers.items():
        if str(key).lower() == needle:
            return str(value)
    return ""


def _default_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
) -> TransferResponse:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=30.0) as response:
            return TransferResponse(
                status_code=int(response.status),
                headers=dict(response.headers.items()),
                content=response.read(),
            )
    except HTTPError as exc:
        return TransferResponse(
            status_code=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            content=exc.read(),
        )
    except (URLError, OSError) as exc:
        raise DiffusionTransferError(
            "diffusion Blob endpoint is unreachable",
            code="transfer_connection_failed",
        ) from exc


class DiffusionBlobTransferClient:
    """Download descriptor bytes in verified, bounded HTTP ranges."""

    def __init__(
        self,
        requester: TransferRequester | None = None,
        *,
        chunk_bytes: int = MAX_TRANSFER_CHUNK_BYTES,
    ) -> None:
        if not 1 <= int(chunk_bytes) <= MAX_TRANSFER_CHUNK_BYTES:
            raise ValueError("chunk_bytes is outside the data-plane limit")
        self._requester = requester or _default_request
        self._chunk_bytes = int(chunk_bytes)

    @staticmethod
    def _descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
        descriptor = dict(value)
        expected = {
            "blob_id", "sha256", "size_bytes", "content_type", "width", "height", "purpose",
        }
        if set(descriptor) != expected:
            raise DiffusionTransferError(
                "diffusion Blob descriptor fields are invalid",
                code="invalid_blob_descriptor",
            )
        if _BLOB_ID.fullmatch(str(descriptor["blob_id"])) is None:
            raise DiffusionTransferError("diffusion Blob ID is invalid", code="invalid_blob_descriptor")
        if _SHA256.fullmatch(str(descriptor["sha256"])) is None:
            raise DiffusionTransferError("diffusion Blob digest is invalid", code="invalid_blob_descriptor")
        size = descriptor["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 16 * 1024 * 1024:
            raise DiffusionTransferError("diffusion Blob size is invalid", code="invalid_blob_descriptor")
        if descriptor["content_type"] not in {"image/png", "image/jpeg", "image/webp"}:
            raise DiffusionTransferError("diffusion Blob MIME type is invalid", code="invalid_blob_descriptor")
        for name in ("width", "height"):
            dimension = descriptor[name]
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise DiffusionTransferError("diffusion Blob dimensions are invalid", code="invalid_blob_descriptor")
        if (
            descriptor["width"] > 2048
            or descriptor["height"] > 2048
            or descriptor["width"] * descriptor["height"] > 2048 * 2048
        ):
            raise DiffusionTransferError("diffusion Blob dimensions are invalid", code="invalid_blob_descriptor")
        if not isinstance(descriptor["purpose"], str) or not descriptor["purpose"]:
            raise DiffusionTransferError("diffusion Blob purpose is invalid", code="invalid_blob_descriptor")
        return descriptor

    @staticmethod
    def _download_grant(
        descriptor: Mapping[str, Any],
        transfer_plan: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        base_url = transfer_plan.get("base_url")
        downloads = transfer_plan.get("downloads")
        if not isinstance(base_url, str) or not base_url or not isinstance(downloads, list):
            raise DiffusionTransferError("diffusion transfer plan is invalid", code="invalid_transfer_plan")
        matching = [
            item for item in downloads
            if isinstance(item, Mapping) and item.get("blob_id") == descriptor["blob_id"]
        ]
        if len(matching) != 1:
            raise DiffusionTransferError("diffusion transfer grant is missing", code="transfer_plan_mismatch")
        grant = matching[0]
        lease_id = str(grant.get("lease_id", ""))
        token = str(grant.get("grant", ""))
        if _LEASE_ID.fullmatch(lease_id) is None or not token:
            raise DiffusionTransferError("diffusion transfer grant is invalid", code="invalid_transfer_plan")
        return base_url.rstrip("/"), lease_id, token

    def download(
        self,
        *,
        attempt_id: str,
        descriptor: Mapping[str, Any],
        transfer_plan: Mapping[str, Any],
    ) -> bytes:
        expected = self._descriptor(descriptor)
        base_url, lease_id, grant = self._download_grant(expected, transfer_plan)
        if not isinstance(attempt_id, str) or not attempt_id:
            raise DiffusionTransferError("diffusion attempt ID is invalid", code="invalid_attempt_id")
        prefix = (
            f"{base_url}{DATA_PLANE_PREFIX}/attempts/{quote(attempt_id, safe='')}/"
            f"blobs/{quote(expected['blob_id'], safe='')}"
        )
        received = bytearray()
        while len(received) < expected["size_bytes"]:
            offset = len(received)
            length = min(self._chunk_bytes, expected["size_bytes"] - offset)
            url = f"{prefix}?{urlencode({'offset': offset, 'length': length})}"
            response = self._requester(
                "GET",
                url,
                {
                    "Authorization": f"Bearer {grant}",
                    "X-Blob-Lease": lease_id,
                    "Accept": "application/octet-stream",
                },
                None,
            )
            if response.status_code not in {200, 206}:
                raise DiffusionTransferError(
                    "diffusion Blob download was rejected",
                    code="transfer_http_status",
                )
            body = bytes(response.content)
            if not body or len(body) > length:
                raise DiffusionTransferError(
                    "diffusion Blob download returned an invalid range",
                    code="transfer_range_mismatch",
                )
            content_range = _header(response.headers, "Content-Range")
            expected_range = f"bytes {offset}-{offset + len(body) - 1}/{expected['size_bytes']}"
            if content_range != expected_range:
                raise DiffusionTransferError(
                    "diffusion Blob range does not match the descriptor",
                    code="transfer_range_mismatch",
                )
            if _header(response.headers, "X-Blob-SHA256") != expected["sha256"]:
                raise DiffusionTransferError(
                    "diffusion Blob digest header does not match the descriptor",
                    code="transfer_digest_mismatch",
                )
            content_type = _header(response.headers, "Content-Type").split(";", 1)[0]
            if content_type != expected["content_type"]:
                raise DiffusionTransferError(
                    "diffusion Blob MIME type does not match the descriptor",
                    code="transfer_content_type_mismatch",
                )
            received.extend(body)
        data = bytes(received)
        if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise DiffusionTransferError(
                "diffusion Blob bytes do not match the descriptor",
                code="transfer_digest_mismatch",
            )
        return data

    def download_to_store(
        self,
        *,
        attempt_id: str,
        descriptor: Mapping[str, Any],
        transfer_plan: Mapping[str, Any],
        destination_store: PersistentImageBlobStore,
        owner_scope: str,
        purpose: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent_blob_ids: tuple[str, ...] = (),
        deduplicate: bool = True,
    ) -> dict[str, Any]:
        expected = self._descriptor(descriptor)
        data = self.download(
            attempt_id=attempt_id,
            descriptor=expected,
            transfer_plan=transfer_plan,
        )
        local = destination_store.put_bytes(
            data,
            content_type=expected["content_type"],
            purpose=purpose or expected["purpose"],
            owner_scope=owner_scope,
            width=expected["width"],
            height=expected["height"],
            metadata=dict(metadata or {}),
            parent_blob_ids=parent_blob_ids,
            deduplicate=deduplicate,
        ).snapshot()
        if local["sha256"] != expected["sha256"]:
            raise DiffusionTransferError(
                "stored diffusion Blob digest changed unexpectedly",
                code="transfer_digest_mismatch",
            )
        return {
            key: local[key]
            for key in (
                "blob_id", "sha256", "size_bytes", "content_type", "width", "height", "purpose",
            )
        }


__all__ = [
    "DiffusionBlobTransferClient",
    "DiffusionTransferError",
    "TransferRequester",
    "TransferResponse",
]
