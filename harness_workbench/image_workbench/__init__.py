"""Small-model harness image workbench.

The package is intentionally independent from ``src/``.  It is Koakumix's
image-generation boundary and delegates execution to an injected local
executor.
"""

from .assets import ImageAssetRecord, ImageAssetStore
from .contracts import (
    GeneratedImage,
    ImageAdapter,
    ImageAdapterCapabilities,
    ImageAdapterError,
    ImageRequest,
    ImageRequestError,
)
from .local_engine import LocalImageEngine, LocalImageEngineConfig
from .manifest import AssetManifestReport, validate_asset_manifest

__all__ = [
    "AssetManifestReport",
    "GeneratedImage",
    "ImageAdapter",
    "ImageAdapterCapabilities",
    "ImageAdapterError",
    "ImageAssetRecord",
    "ImageAssetStore",
    "ImageRequest",
    "ImageRequestError",
    "LocalImageEngine",
    "LocalImageEngineConfig",
    "validate_asset_manifest",
]
