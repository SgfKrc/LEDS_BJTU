"""Local txt2img boundary for the independent harness workbench."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import (
    GeneratedImage,
    ImageAdapter,
    ImageAdapterCapabilities,
    ImageAdapterError,
    ImageRequest,
)
from .manifest import AssetManifest, AssetManifestReport, validate_asset_manifest


@dataclass(frozen=True, slots=True)
class LocalImageEngineConfig:
    asset_root: Path | str
    model_id: str | None = None
    full_hash: bool = False
    backend_id: str = "local_diffusers_txt2img"


class LocalImageExecutor(Protocol):
    def generate(self, request: ImageRequest, manifest: AssetManifest) -> GeneratedImage:
        ...

    def close(self) -> None:
        ...


class UnavailableImageExecutor:
    def generate(self, request: ImageRequest, manifest: AssetManifest) -> GeneratedImage:
        raise ImageAdapterError(
            "local image runtime is not installed; provide the optional diffusers/CUDA executor",
            code="local_image_runtime_unavailable",
            status_code=503,
        )

    def close(self) -> None:
        return None


class LocalImageEngine(ImageAdapter):
    """Verify a local SD package before delegating to a real/fake executor."""

    def __init__(self, config: LocalImageEngineConfig, *, executor: LocalImageExecutor | None = None) -> None:
        self.config = LocalImageEngineConfig(
            asset_root=Path(config.asset_root).expanduser().resolve(),
            model_id=config.model_id,
            full_hash=config.full_hash,
            backend_id=config.backend_id,
        )
        self._executor = executor or UnavailableImageExecutor()

    @property
    def asset_root(self) -> Path:
        return Path(self.config.asset_root)

    def inspect(self) -> AssetManifestReport:
        return validate_asset_manifest(
            self.asset_root,
            expected_asset_id=self.config.model_id,
            full_hash=self.config.full_hash,
        )

    def capabilities(self) -> ImageAdapterCapabilities:
        report = self.inspect()
        executor_available = not isinstance(self._executor, UnavailableImageExecutor)
        optional_runtime_detected = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("diffusers") is not None
        # This ticket ships the boundary and an injectable executor.  Merely
        # finding torch/diffusers in the environment does not prove that the
        # harness-owned worker can load this selected asset.
        runtime_available = executor_available
        supports = report.valid and runtime_available
        return ImageAdapterCapabilities(
            backend=self.config.backend_id,
            model_ids=(report.asset_id,) if report.asset_id else (),
            supports_txt2img=supports,
            runtime_available=runtime_available,
            evidence={
                "manifest": report.as_dict(),
                "executor_injected": executor_available,
                "optional_runtime_detected": optional_runtime_detected,
            },
        )

    def generate(self, request: ImageRequest) -> GeneratedImage:
        request.validate()
        if self.config.model_id and request.model and request.model != self.config.model_id:
            raise ImageAdapterError(
                f"local image engine only serves model {self.config.model_id}",
                code="model_not_available",
                status_code=404,
            )
        report = self.inspect()
        if not report.valid or report.manifest is None:
            raise ImageAdapterError(
                "local SD asset manifest failed verification",
                code="asset_manifest_invalid",
                status_code=409,
            )
        try:
            result = self._executor.generate(request, report.manifest)
        except ImageAdapterError:
            raise
        except Exception as exc:
            raise ImageAdapterError(
                "local image executor failed",
                code="local_image_generation_failed",
                status_code=502,
            ) from exc
        if not isinstance(result, GeneratedImage):
            raise ImageAdapterError("local image executor returned an invalid result", code="invalid_image_result", status_code=502)
        return result

    def close(self) -> None:
        self._executor.close()


__all__ = ["LocalImageEngine", "LocalImageEngineConfig", "LocalImageExecutor"]
