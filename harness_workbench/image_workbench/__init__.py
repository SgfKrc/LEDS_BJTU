"""Small-model harness image workbench.

The package is intentionally independent from ``src/``.  It owns the stable
request/asset contracts and delegates actual image generation to either an
injected local executor or a narrow QLH HTTP transport.
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
from .remote_qlh import HttpRemoteQLHTransport, RemoteQLHConfig, RemoteQLHImageAdapter

__all__ = [
    "AssetManifestReport",
    "GeneratedImage",
    "HttpRemoteQLHTransport",
    "ImageAdapter",
    "ImageAdapterCapabilities",
    "ImageAdapterError",
    "ImageAssetRecord",
    "ImageAssetStore",
    "ImageRequest",
    "ImageRequestError",
    "LocalImageEngine",
    "LocalImageEngineConfig",
    "RemoteQLHImageAdapter",
    "RemoteQLHConfig",
    "validate_asset_manifest",
]
