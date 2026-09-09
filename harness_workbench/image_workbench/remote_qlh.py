"""Small QLH transport adapter for remote SD generation."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.parse import urljoin

from .contracts import GeneratedImage, ImageAdapter, ImageAdapterCapabilities, ImageAdapterError, ImageRequest


class RemoteQLHTransport(Protocol):
    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def get_json(self, path: str) -> Mapping[str, Any]:
        ...

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        ...


@dataclass(frozen=True, slots=True)
class RemoteQLHConfig:
    base_url: str
    model_id: str | None = None
    poll_interval_seconds: float = 0.25
    poll_timeout_seconds: float = 600.0


class RemoteQLHImageAdapter(ImageAdapter):
    """Map the harness txt2img contract to QLH's async job/blob contract."""

    def __init__(self, config: RemoteQLHConfig, transport: RemoteQLHTransport) -> None:
        base_url = config.base_url.rstrip("/") + "/"
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("remote QLH base_url must use http or https")
        self.config = RemoteQLHConfig(
            base_url=base_url,
            model_id=config.model_id,
            poll_interval_seconds=max(0.0, config.poll_interval_seconds),
            poll_timeout_seconds=max(0.1, config.poll_timeout_seconds),
        )
        self.transport = transport
        self._capabilities = ImageAdapterCapabilities(
            backend="qlh_remote_diffusion",
            model_ids=(config.model_id,) if config.model_id else (),
            evidence={"capability_probe": "not_run"},
        )

    def capabilities(self) -> ImageAdapterCapabilities:
        return self._capabilities

    def probe_capabilities(self) -> ImageAdapterCapabilities:
        try:
            payload = self.transport.get_json("/api/diffusion/capabilities")
        except Exception as exc:
            raise ImageAdapterError("QLH diffusion capability probe failed", code="remote_capability_probe_failed", status_code=502) from exc
        presets = payload.get("presets", []) if isinstance(payload, Mapping) else []
        model_ids = tuple(
            str(item.get("preset_id") or item.get("asset_id"))
            for item in presets
            if isinstance(item, Mapping) and (item.get("preset_id") or item.get("asset_id"))
        )
        self._capabilities = ImageAdapterCapabilities(
            backend="qlh_remote_diffusion",
            model_ids=model_ids or ((self.config.model_id,) if self.config.model_id else ()),
            supports_txt2img=True,
            supports_edit=True,
            supports_distributed=bool(payload.get("distributed", False)) if isinstance(payload, Mapping) else False,
            runtime_available=True,
            evidence={"capability_probe": "qlh:/api/diffusion/capabilities", "raw": dict(payload)},
        )
        return self._capabilities

    def generate(self, request: ImageRequest) -> GeneratedImage:
        request.validate()
        if self.config.model_id and request.model and request.model != self.config.model_id:
            raise ImageAdapterError(
                f"remote QLH adapter only serves model {self.config.model_id}",
                code="model_not_available",
                status_code=404,
            )
        try:
            submitted = self.transport.post_json("/api/diffusion/generate", request.as_remote_payload())
        except Exception as exc:
            raise ImageAdapterError("QLH diffusion request failed", code="remote_image_request_failed", status_code=502) from exc
        result = self._resolve_result(submitted)
        if result is None:
            raise ImageAdapterError("QLH returned no image blob", code="remote_image_result_missing", status_code=502)
        return result

    def _resolve_result(self, payload: Mapping[str, Any]) -> GeneratedImage | None:
        if not isinstance(payload, Mapping):
            return None
        direct = _decode_payload_image(payload)
        if direct is not None:
            return direct
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return None
        deadline = time.monotonic() + self.config.poll_timeout_seconds
        while True:
            job = self.transport.get_json(f"/api/diffusion/jobs/{job_id}")
            if not isinstance(job, Mapping):
                raise ImageAdapterError("QLH returned an invalid diffusion job", code="remote_job_invalid", status_code=502)
            state = str(job.get("state", ""))
            resolved = _decode_payload_image(job)
            if resolved is not None:
                return resolved
            blob = job.get("blob")
            if isinstance(blob, Mapping):
                blob_id = blob.get("blob_id")
                if isinstance(blob_id, str) and blob_id:
                    return self._read_blob(blob_id)
            if state in {"failed", "cancelled", "canceled"}:
                raise ImageAdapterError(
                    str(job.get("error") or "QLH diffusion job failed"),
                    code="remote_image_generation_failed",
                    status_code=502,
                )
            if state in {"completed", "done"}:
                return None
            if time.monotonic() >= deadline:
                raise ImageAdapterError("QLH diffusion job timed out", code="remote_image_timeout", status_code=504, retryable=True)
            time.sleep(self.config.poll_interval_seconds)

    def _read_blob(self, blob_id: str) -> GeneratedImage:
        try:
            data, mime_type = self.transport.get_bytes(f"/api/diffusion/blobs/{blob_id}")
        except Exception as exc:
            raise ImageAdapterError("QLH diffusion blob download failed", code="remote_blob_download_failed", status_code=502) from exc
        return GeneratedImage(data=data, mime_type=mime_type or "image/png")

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


class HttpRemoteQLHTransport:
    """Lazy httpx transport; importing this class does not add a dependency."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError("httpx is required for the remote QLH image adapter") from exc
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)
        return self._client

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._http().post(path.lstrip("/"), json=dict(payload))
        return _json_response(response)

    def get_json(self, path: str) -> Mapping[str, Any]:
        response = self._http().get(path.lstrip("/"))
        return _json_response(response)

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        response = self._http().get(path.lstrip("/"))
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/png").split(";", 1)[0]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def _json_response(response: Any) -> Mapping[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("remote response must be a JSON object")
    return payload


def _decode_payload_image(payload: Mapping[str, Any]) -> GeneratedImage | None:
    encoded = payload.get("b64_json") or payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            return _decode_payload_image(data[0])
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ImageAdapterError("QLH returned invalid image base64", code="remote_image_invalid", status_code=502) from exc
    return GeneratedImage(
        data=raw,
        mime_type=str(payload.get("mime_type") or payload.get("content_type") or "image/png"),
        width=_optional_int(payload.get("width")),
        height=_optional_int(payload.get("height")),
        seed=_optional_int(payload.get("seed")),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = ["HttpRemoteQLHTransport", "RemoteQLHConfig", "RemoteQLHImageAdapter", "RemoteQLHTransport"]
