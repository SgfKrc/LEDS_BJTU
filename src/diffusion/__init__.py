"""Optional Stable Diffusion 1.5 sidecar.

The diffusion engine is deliberately separate from ``model_module.ModelManager``.
This keeps the existing PyTorch/llama.cpp LLM paths and their worker contracts
unchanged while the SD 1.5 local path is validated.
"""

from .artifacts import DiffusionArtifact, DiffusionArtifactInspector
from .assets import (
    ASSET_CATALOG,
    DiffusionAssetManager,
    DiffusionAssetSpec,
    LOCAL_PROXY_FALLBACK,
    get_asset_spec,
    verify_asset_directory,
)
from .presets import DiffusionPreset, get_preset, list_presets
from .sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15EngineConfig,
    SD15GenerationRequest,
    SD15GenerationResult,
)
from .service import (
    DiffusionConflictError,
    DiffusionBlobInUseError,
    DiffusionBlobReferencedError,
    DiffusionInputError,
    DiffusionNotFoundError,
    DiffusionService,
    DiffusionServiceError,
    DiffusionUnsupportedError,
    DIFFUSION_MAX_IMAGE_PIXELS,
    DIFFUSION_MAX_UPLOAD_BYTES,
    ImageBlob,
    MemoryImageBlobStore,
    SD15EditRequest,
    SD15_LOAD_PROFILES,
    build_sd15_engine_config,
    build_sd15_generation_request,
)

__all__ = [
    "DiffusionArtifact",
    "DiffusionArtifactInspector",
    "DiffusionAssetManager",
    "DiffusionAssetSpec",
    "DiffusionConflictError",
    "DiffusionBlobInUseError",
    "DiffusionBlobReferencedError",
    "DiffusionInputError",
    "DiffusionNotFoundError",
    "DiffusionPreset",
    "DiffusionService",
    "DiffusionServiceError",
    "DiffusionUnsupportedError",
    "DIFFUSION_MAX_IMAGE_PIXELS",
    "DIFFUSION_MAX_UPLOAD_BYTES",
    "GenerationCancelled",
    "SD15Engine",
    "SD15EngineConfig",
    "SD15GenerationRequest",
    "SD15GenerationResult",
    "ASSET_CATALOG",
    "LOCAL_PROXY_FALLBACK",
    "MemoryImageBlobStore",
    "ImageBlob",
    "SD15EditRequest",
    "SD15_LOAD_PROFILES",
    "build_sd15_engine_config",
    "build_sd15_generation_request",
    "get_preset",
    "get_asset_spec",
    "list_presets",
    "verify_asset_directory",
]
