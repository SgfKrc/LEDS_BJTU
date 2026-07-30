"""Optional Stable Diffusion 1.5 sidecar.

The diffusion engine is deliberately separate from ``model_module.ModelManager``.
This keeps the existing PyTorch/llama.cpp LLM paths and their worker contracts
unchanged while the SD 1.5 local path is validated.
"""

from .artifacts import DiffusionArtifact, DiffusionArtifactInspector
from .presets import DiffusionPreset, get_preset, list_presets
from .sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15EngineConfig,
    SD15GenerationRequest,
    SD15GenerationResult,
)

__all__ = [
    "DiffusionArtifact",
    "DiffusionArtifactInspector",
    "DiffusionPreset",
    "GenerationCancelled",
    "SD15Engine",
    "SD15EngineConfig",
    "SD15GenerationRequest",
    "SD15GenerationResult",
    "get_preset",
    "list_presets",
]
