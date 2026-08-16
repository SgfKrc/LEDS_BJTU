"""Model asset inspection, maintenance, and conversion tools."""

from .gguf import inspect_gguf, verify_gguf
from .gguf_convert import GGUFConvertError, execute_conversion, plan_conversion
from .gemma4_native_probe import run_native_probe
from .gemma4_native_artifacts import resolve_ollama_gemma4_12b, run_ollama_gemma4_12b_preflight
from .gemma4_pipeline_adapter import (
    Gemma4AdapterError,
    Gemma4PipelineAdapter,
    load_gemma4_text_layer_assignment,
    select_gemma4_assignment_keys,
    validate_gemma4_assignment,
)
from .qwen3_sidecar_probe import run_qwen3_sidecar_probe
from .qwen3_multimodal_processor_probe import run_qwen3_multimodal_processor_probe
from .qwen3_multimodal_vision_text_smoke import run_qwen3_multimodal_vision_text_smoke
from .qwen3_multimodal_vision_tower_probe import run_qwen3_multimodal_vision_tower_probe
from .qwen3_pipeline_smoke import run_qwen3_pipeline_smoke
from .qwen3_pipeline_chain_smoke import run_qwen3_pipeline_chain_smoke
from .qwen3_pipeline_chain import (
    build_hidden_handoff,
    build_kv_contract,
    execute_segment_chain,
    validate_hidden_handoff,
    validate_kv_contract,
    validate_segment_plan,
)
from .qwen3_pipeline_adapter import (
    Qwen3AdapterError,
    Qwen3PipelineAdapter,
    load_qwen3_layer_assignment,
    render_without_thinking,
    select_qwen3_assignment_keys,
    validate_qwen3_assignment,
)
from .llm_smoke_matrix import discover_units, fixed_prompts, run_smoke_matrix
from .llama_quantize_toolchain import resolve_quantizer, verify_managed_package
from .maintenance import clean_models, model_disk_usage
from .sweep import sweep_models
from .sync_status import build_inventory, compare_inventories

__all__ = [
    "build_inventory",
    "clean_models",
    "compare_inventories",
    "discover_units",
    "fixed_prompts",
    "GGUFConvertError",
    "run_native_probe",
    "resolve_ollama_gemma4_12b",
    "run_ollama_gemma4_12b_preflight",
    "Gemma4AdapterError",
    "Gemma4PipelineAdapter",
    "load_gemma4_text_layer_assignment",
    "select_gemma4_assignment_keys",
    "validate_gemma4_assignment",
    "run_qwen3_sidecar_probe",
    "run_qwen3_multimodal_processor_probe",
    "run_qwen3_multimodal_vision_tower_probe",
    "run_qwen3_multimodal_vision_text_smoke",
    "run_qwen3_pipeline_smoke",
    "run_qwen3_pipeline_chain_smoke",
    "build_hidden_handoff",
    "build_kv_contract",
    "execute_segment_chain",
    "validate_hidden_handoff",
    "validate_kv_contract",
    "validate_segment_plan",
    "Qwen3AdapterError",
    "Qwen3PipelineAdapter",
    "load_qwen3_layer_assignment",
    "render_without_thinking",
    "select_qwen3_assignment_keys",
    "validate_qwen3_assignment",
    "execute_conversion",
    "inspect_gguf",
    "model_disk_usage",
    "plan_conversion",
    "resolve_quantizer",
    "run_smoke_matrix",
    "sweep_models",
    "verify_gguf",
    "verify_managed_package",
]
