"""Path-free contracts for the first Qwen3 multimodal pipeline stage.

MM1 describes the boundary between a visual component (native PyTorch
weights or an external ``mmproj``) and the first Qwen text segment.  It is a
metadata contract only: image bytes, prompts, tensors, filesystem paths and
transfer tickets never enter the contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


MM1_SCHEMA_VERSION = 1
MM1_MAX_CONTRACT_BYTES = 64 * 1024
MM1_MAX_COMPONENTS = 16
MM1_MAX_ARTIFACT_BYTES = 1 << 40
MM1_MAX_VISUAL_TOKENS = 262_144
MM1_MAX_SEQUENCE_LENGTH = 262_144
MM1_MAX_BATCH_SIZE = 64

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MODEL_FAMILIES = {"qwen3_vl", "qwen3_5"}
_RUNTIMES = {"transformers_sidecar", "llama_cpp_mtmd"}
_FORMATS = {"safetensors", "gguf", "json", "tokenizer"}
_COMPONENT_KINDS = {"text_weights", "vision_weights", "mmproj", "processor", "mtp"}
_MODALITIES = {"image", "video"}
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_TRANSFER_REFERENCE_FIELDS = {
    "artifact_id", "mode", "source_node_id", "target_node_id", "chain_id",
    "generation", "phase", "from_segment", "to_segment", "size_bytes", "sha256",
}
_SENSITIVE_KEYS = {
    "path", "model_path", "input_path", "output_path", "filesystem_path",
    "ticket", "grant", "prompt", "messages", "input_ids", "hidden_states",
    "past_key_values", "tensors", "payload",
}


class Qwen3MultimodalContractError(ValueError):
    """The MM1 component or hidden handoff contract is not admissible."""


def _canonical(value: Any, *, label: str, maximum: int = MM1_MAX_CONTRACT_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalContractError(f"{label} is not JSON serializable") from exc
    if len(encoded) > maximum:
        raise Qwen3MultimodalContractError(f"{label} exceeds serialization limit")
    return encoded


def _digest(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical(value, label=label)).hexdigest()


def _safe_id(value: Any, field: str) -> str:
    result = str(value or "")
    if _SAFE_ID.fullmatch(result) is None:
        raise Qwen3MultimodalContractError(f"{field} is invalid")
    return result


def _sha256(value: Any, field: str) -> str:
    result = str(value or "").lower()
    if _SHA256.fullmatch(result) is None:
        raise Qwen3MultimodalContractError(f"{field} must be a lowercase SHA-256")
    return result


def _positive_int(value: Any, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise Qwen3MultimodalContractError(f"{field} is invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalContractError(f"{field} is invalid") from exc
    if result <= 0 or maximum is not None and result > maximum:
        raise Qwen3MultimodalContractError(f"{field} is outside limits")
    return result


def _non_negative_int(value: Any, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise Qwen3MultimodalContractError(f"{field} is invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalContractError(f"{field} is invalid") from exc
    if result < 0 or maximum is not None and result > maximum:
        raise Qwen3MultimodalContractError(f"{field} is outside limits")
    return result


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise Qwen3MultimodalContractError(
                    f"MM1 contract cannot contain {key}",
                )
            _reject_sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive(item)


def _exact(value: Any, fields: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Qwen3MultimodalContractError(f"{field} fields are invalid")
    return dict(value)


def _normalize_modalities(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > len(_MODALITIES):
        raise Qwen3MultimodalContractError(f"{field} is invalid")
    result = [str(item) for item in value]
    if len(set(result)) != len(result) or result != sorted(result):
        raise Qwen3MultimodalContractError(f"{field} must be unique and ordered")
    if any(item not in _MODALITIES for item in result):
        raise Qwen3MultimodalContractError(f"{field} contains unsupported modality")
    return result


def _validate_component(value: Any, field: str) -> dict[str, Any]:
    component = _exact(
        value,
        {"component_id", "artifact_id", "component_kind", "format", "revision", "size_bytes", "sha256"},
        field,
    )
    _safe_id(component["component_id"], f"{field}.component_id")
    _safe_id(component["artifact_id"], f"{field}.artifact_id")
    if component["component_kind"] not in _COMPONENT_KINDS:
        raise Qwen3MultimodalContractError(f"{field}.component_kind is unsupported")
    if component["format"] not in _FORMATS:
        raise Qwen3MultimodalContractError(f"{field}.format is unsupported")
    _safe_id(component["revision"], f"{field}.revision")
    _positive_int(component["size_bytes"], f"{field}.size_bytes", maximum=MM1_MAX_ARTIFACT_BYTES)
    _sha256(component["sha256"], f"{field}.sha256")
    return component


def _validate_profile_parts(
    text: Any, vision: Any, processor: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    text = _exact(text, {"hidden_size", "num_hidden_layers", "dtype"}, "text")
    text["hidden_size"] = _positive_int(text["hidden_size"], "text.hidden_size", maximum=65_536)
    text["num_hidden_layers"] = _positive_int(text["num_hidden_layers"], "text.num_hidden_layers", maximum=4096)
    if text["dtype"] not in _DTYPE_BYTES:
        raise Qwen3MultimodalContractError("text.dtype is unsupported")
    vision = _exact(
        vision,
        {"hidden_size", "output_hidden_size", "depth", "patch_size", "spatial_merge_size", "temporal_patch_size", "modalities"},
        "vision",
    )
    for name in ("hidden_size", "output_hidden_size", "depth", "patch_size", "spatial_merge_size", "temporal_patch_size"):
        vision[name] = _positive_int(vision[name], f"vision.{name}", maximum=65_536)
    vision["modalities"] = _normalize_modalities(vision["modalities"], "vision.modalities")
    processor = _exact(
        processor,
        {"processor_id", "revision", "modalities", "image_token_id", "video_token_id", "max_images", "max_video_frames", "min_pixels", "max_pixels"},
        "processor",
    )
    _safe_id(processor["processor_id"], "processor.processor_id")
    _safe_id(processor["revision"], "processor.revision")
    processor["modalities"] = _normalize_modalities(processor["modalities"], "processor.modalities")
    for name in ("image_token_id", "video_token_id"):
        processor[name] = _non_negative_int(processor[name], f"processor.{name}", maximum=1_000_000_000)
    processor["max_images"] = _positive_int(processor["max_images"], "processor.max_images", maximum=64)
    processor["max_video_frames"] = _positive_int(processor["max_video_frames"], "processor.max_video_frames", maximum=4096)
    processor["min_pixels"] = _positive_int(processor["min_pixels"], "processor.min_pixels", maximum=1_000_000_000)
    processor["max_pixels"] = _positive_int(processor["max_pixels"], "processor.max_pixels", maximum=1_000_000_000)
    if processor["min_pixels"] > processor["max_pixels"]:
        raise Qwen3MultimodalContractError("processor pixel limits are inverted")
    if not set(processor["modalities"]).issubset(vision["modalities"]):
        raise Qwen3MultimodalContractError("processor modality is absent from vision profile")
    return text, vision, processor


def build_mm1_model_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a bounded, path-free profile from a Qwen3-VL/Qwen3.5 config."""
    if not isinstance(config, Mapping):
        raise Qwen3MultimodalContractError("model config must be an object")
    model_family = str(config.get("model_type", "") or "").lower()
    if model_family not in _MODEL_FAMILIES:
        raise Qwen3MultimodalContractError("model config is not an MM1 model family")
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        text_config = config
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, Mapping):
        raise Qwen3MultimodalContractError("model config has no vision_config")
    try:
        text = {
            "hidden_size": text_config["hidden_size"],
            "num_hidden_layers": text_config["num_hidden_layers"],
            "dtype": str(text_config.get("dtype", "bfloat16")).lower(),
        }
        vision = {
            "hidden_size": vision_config["hidden_size"],
            "output_hidden_size": vision_config["out_hidden_size"],
            "depth": vision_config["depth"],
            "patch_size": vision_config["patch_size"],
            "spatial_merge_size": vision_config["spatial_merge_size"],
            "temporal_patch_size": vision_config["temporal_patch_size"],
            "modalities": ["image", "video"],
        }
        processor = {
            "processor_id": model_family,
            "revision": str(config.get("transformers_version", "config")),
            "modalities": ["image", "video"],
            "image_token_id": config["image_token_id"],
            "video_token_id": config["video_token_id"],
            "max_images": 4,
            "max_video_frames": 128,
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 2048 * 2048,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Qwen3MultimodalContractError("model config is missing MM1 fields") from exc
    # The processor defaults are deliberately conservative and are overridden
    # only by an explicit, separately validated processor manifest.
    return {
        "model_family": model_family,
        "text": text,
        "vision": vision,
        "processor": processor,
    }


def build_mm1_model_manifest(
    *,
    model_id: str,
    model_family: str,
    runtime: str,
    revision: str,
    components: list[Mapping[str, Any]],
    text: Mapping[str, Any],
    vision: Mapping[str, Any],
    processor: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        "schema_version": MM1_SCHEMA_VERSION,
        "manifest_kind": "qwen3_multimodal_components",
        "model_id": _safe_id(model_id, "model_id"),
        "model_family": str(model_family),
        "runtime": str(runtime),
        "revision": _safe_id(revision, "revision"),
        "components": sorted([dict(item) for item in components], key=lambda item: str(item.get("component_id", ""))),
        "text": dict(text),
        "vision": dict(vision),
        "processor": dict(processor),
    }
    values["manifest_sha256"] = _digest(values, label="MM1 manifest")
    return validate_mm1_model_manifest(values)


def validate_mm1_model_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _exact(
        value,
        {"schema_version", "manifest_kind", "model_id", "model_family", "runtime", "revision", "components", "text", "vision", "processor", "manifest_sha256"},
        "MM1 manifest",
    )
    if manifest["schema_version"] != MM1_SCHEMA_VERSION or manifest["manifest_kind"] != "qwen3_multimodal_components":
        raise Qwen3MultimodalContractError("MM1 manifest version/kind is unsupported")
    _safe_id(manifest["model_id"], "model_id")
    if manifest["model_family"] not in _MODEL_FAMILIES:
        raise Qwen3MultimodalContractError("model_family is unsupported")
    if manifest["runtime"] not in _RUNTIMES:
        raise Qwen3MultimodalContractError("runtime is unsupported")
    if manifest["model_family"] == "qwen3_5" and manifest["runtime"] != "transformers_sidecar":
        raise Qwen3MultimodalContractError("Qwen3.5 requires transformers_sidecar runtime")
    _safe_id(manifest["revision"], "revision")
    components = manifest["components"]
    if not isinstance(components, list) or not 1 <= len(components) <= MM1_MAX_COMPONENTS:
        raise Qwen3MultimodalContractError("MM1 components are invalid")
    normalized_components = [_validate_component(item, f"components[{index}]") for index, item in enumerate(components)]
    component_ids = [item["component_id"] for item in normalized_components]
    if component_ids != sorted(component_ids) or len(set(component_ids)) != len(component_ids):
        raise Qwen3MultimodalContractError("MM1 components must be unique and ordered")
    by_artifact: dict[str, dict[str, Any]] = {}
    for component in normalized_components:
        previous = by_artifact.setdefault(component["artifact_id"], component)
        if any(component[name] != previous[name] for name in ("format", "revision", "size_bytes", "sha256")):
            raise Qwen3MultimodalContractError("shared MM1 artifact identity changed")
    kinds = {item["component_kind"] for item in normalized_components}
    if "processor" not in kinds or "text_weights" not in kinds:
        raise Qwen3MultimodalContractError("MM1 manifest is missing processor or text weights")
    if manifest["runtime"] == "llama_cpp_mtmd":
        if manifest["model_family"] != "qwen3_vl" or "mmproj" not in kinds:
            raise Qwen3MultimodalContractError("llama_cpp_mtmd requires a Qwen3-VL mmproj")
    elif "vision_weights" not in kinds:
        raise Qwen3MultimodalContractError("transformers_sidecar requires vision weights")
    text, vision, processor = _validate_profile_parts(
        manifest["text"], manifest["vision"], manifest["processor"],
    )
    if not set(processor["modalities"]).issubset(set(vision["modalities"])):
        raise Qwen3MultimodalContractError("processor modalities are not vision modalities")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    if manifest["manifest_sha256"] != _digest(unsigned, label="MM1 manifest"):
        raise Qwen3MultimodalContractError("MM1 manifest digest mismatch")
    normalized = dict(manifest)
    normalized.update({"components": normalized_components, "text": text, "vision": vision, "processor": processor})
    _canonical(normalized, label="MM1 manifest")
    return normalized


def estimate_mm1_capacity(
    manifest: Mapping[str, Any],
    *,
    batch_size: int,
    visual_tokens: int,
    sequence_length: int,
    dtype: str | None = None,
    safety_margin: float = 1.2,
) -> dict[str, Any]:
    """Estimate visual/text node memory and the path-free handoff size."""
    safe = validate_mm1_model_manifest(manifest)
    batch = _positive_int(batch_size, "batch_size", maximum=MM1_MAX_BATCH_SIZE)
    visual_count = _positive_int(visual_tokens, "visual_tokens", maximum=MM1_MAX_VISUAL_TOKENS)
    sequence = _positive_int(sequence_length, "sequence_length", maximum=MM1_MAX_SEQUENCE_LENGTH)
    try:
        margin = float(safety_margin)
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalContractError("safety_margin is invalid") from exc
    if margin < 1.0 or margin > 100.0:
        raise Qwen3MultimodalContractError("safety_margin is outside limits")
    selected_dtype = str(dtype or safe["text"]["dtype"]).lower()
    if selected_dtype not in _DTYPE_BYTES:
        raise Qwen3MultimodalContractError("capacity dtype is unsupported")
    dtype_bytes = _DTYPE_BYTES[selected_dtype]
    unique_artifacts = {
        item["artifact_id"]: item["size_bytes"]
        for item in safe["components"]
    }
    visual_artifacts = {
        item["artifact_id"] for item in safe["components"]
        if item["component_kind"] in {"vision_weights", "mmproj"}
    }
    text_artifacts = {
        item["artifact_id"] for item in safe["components"]
        if item["component_kind"] in {"text_weights", "mtp"}
    }
    visual_weight = sum(unique_artifacts[item] for item in visual_artifacts)
    text_weight = sum(unique_artifacts[item] for item in text_artifacts)
    visual_activation = batch * visual_count * safe["vision"]["output_hidden_size"] * dtype_bytes
    text_activation = batch * sequence * safe["text"]["hidden_size"] * dtype_bytes
    handoff_bytes = visual_activation
    return {
        "schema_version": MM1_SCHEMA_VERSION,
        "model_id": safe["model_id"],
        "model_family": safe["model_family"],
        "dtype": selected_dtype,
        "batch_size": batch,
        "visual_tokens": visual_count,
        "sequence_length": sequence,
        "visual_weight_bytes": visual_weight,
        "text_weight_bytes": text_weight,
        "unique_artifact_bytes": sum(unique_artifacts.values()),
        "visual_activation_bytes": visual_activation,
        "text_activation_bytes": text_activation,
        "handoff_bytes": handoff_bytes,
        "visual_required_bytes": int((visual_weight + visual_activation) * margin),
        "text_required_bytes": int((text_weight + text_activation) * margin),
        "full_model_materialized": False,
    }


def _validate_artifact(value: Any, field: str) -> dict[str, Any]:
    artifact = _exact(value, {"artifact_id", "mode", "size_bytes", "sha256", "status"}, field)
    _safe_id(artifact["artifact_id"], f"{field}.artifact_id")
    if artifact["mode"] not in {"local", "network"} or artifact["status"] != "committed":
        raise Qwen3MultimodalContractError(f"{field} state is invalid")
    _positive_int(artifact["size_bytes"], f"{field}.size_bytes", maximum=MM1_MAX_ARTIFACT_BYTES)
    _sha256(artifact["sha256"], f"{field}.sha256")
    return artifact


def build_mm1_handoff_contract(
    *,
    manifest: Mapping[str, Any],
    text_chain_id: str,
    generation: int,
    phase: str,
    source_node_id: str,
    target_node_id: str,
    artifact: Mapping[str, Any],
    shape: list[int] | tuple[int, int, int],
    dtype: str,
    device: str,
    modality: str,
    item_count: int = 1,
    frame_count: int = 0,
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise Qwen3MultimodalContractError("shape must be [batch,tokens,hidden]")
    values = {
        "schema_version": MM1_SCHEMA_VERSION,
        "contract_kind": "qwen3_multimodal_handoff",
        "model_id": safe_manifest["model_id"],
        "model_family": safe_manifest["model_family"],
        "runtime": safe_manifest["runtime"],
        "revision": safe_manifest["revision"],
        "manifest_sha256": safe_manifest["manifest_sha256"],
        "text_chain_id": _sha256(text_chain_id, "text_chain_id"),
        "generation": _non_negative_int(generation, "generation", maximum=2**31 - 1),
        "phase": str(phase),
        "source_node_id": _safe_id(source_node_id, "source_node_id"),
        "target_node_id": _safe_id(target_node_id, "target_node_id"),
        "boundary": "visual_to_text",
        "artifact": dict(artifact),
        "tensor": {"shape": list(shape), "dtype": str(dtype).lower(), "device": str(device).lower()},
        "media": {"modality": str(modality), "item_count": item_count, "frame_count": frame_count},
        "full_model_materialized": False,
    }
    values["contract_sha256"] = _digest(values, label="MM1 handoff contract")
    return validate_mm1_handoff_contract(values, safe_manifest)


def validate_mm1_handoff_contract(
    value: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    contract = _exact(
        value,
        {"schema_version", "contract_kind", "model_id", "model_family", "runtime", "revision", "manifest_sha256", "text_chain_id", "generation", "phase", "source_node_id", "target_node_id", "boundary", "artifact", "tensor", "media", "full_model_materialized", "contract_sha256"},
        "MM1 handoff contract",
    )
    if contract["schema_version"] != MM1_SCHEMA_VERSION or contract["contract_kind"] != "qwen3_multimodal_handoff":
        raise Qwen3MultimodalContractError("MM1 handoff version/kind is unsupported")
    for field in ("model_id", "source_node_id", "target_node_id"):
        _safe_id(contract[field], field)
    if contract["source_node_id"] == contract["target_node_id"]:
        raise Qwen3MultimodalContractError("MM1 handoff source and target must differ")
    if contract["model_id"] != safe_manifest["model_id"] or contract["model_family"] != safe_manifest["model_family"] or contract["runtime"] != safe_manifest["runtime"] or contract["revision"] != safe_manifest["revision"] or contract["manifest_sha256"] != safe_manifest["manifest_sha256"]:
        raise Qwen3MultimodalContractError("MM1 handoff model identity does not match manifest")
    _sha256(contract["text_chain_id"], "text_chain_id")
    _non_negative_int(contract["generation"], "generation", maximum=2**31 - 1)
    if contract["phase"] not in {"prefill", "decode"} or contract["boundary"] != "visual_to_text":
        raise Qwen3MultimodalContractError("MM1 handoff phase/boundary is invalid")
    _validate_artifact(contract["artifact"], "artifact")
    tensor = _exact(contract["tensor"], {"shape", "dtype", "device"}, "tensor")
    shape = tensor["shape"]
    if not isinstance(shape, list) or len(shape) != 3:
        raise Qwen3MultimodalContractError("tensor.shape must be [batch,tokens,hidden]")
    batch = _positive_int(shape[0], "tensor.shape[0]", maximum=MM1_MAX_BATCH_SIZE)
    tokens = _positive_int(shape[1], "tensor.shape[1]", maximum=MM1_MAX_VISUAL_TOKENS)
    hidden = _positive_int(shape[2], "tensor.shape[2]", maximum=65_536)
    if hidden != safe_manifest["vision"]["output_hidden_size"]:
        raise Qwen3MultimodalContractError("visual hidden size does not match manifest")
    if tensor["dtype"] not in _DTYPE_BYTES or tensor["device"] not in {"cpu", "cuda"}:
        raise Qwen3MultimodalContractError("tensor dtype/device is unsupported")
    media = _exact(contract["media"], {"modality", "item_count", "frame_count"}, "media")
    if media["modality"] not in safe_manifest["processor"]["modalities"]:
        raise Qwen3MultimodalContractError("media modality is not supported by processor")
    count = _positive_int(media["item_count"], "media.item_count", maximum=safe_manifest["processor"]["max_images"])
    frames = _non_negative_int(media["frame_count"], "media.frame_count", maximum=safe_manifest["processor"]["max_video_frames"])
    if media["modality"] == "image" and frames != 0:
        raise Qwen3MultimodalContractError("image handoff cannot declare video frames")
    if media["modality"] == "video" and frames <= 0:
        raise Qwen3MultimodalContractError("video handoff requires video frames")
    if contract["full_model_materialized"] is not False:
        raise Qwen3MultimodalContractError("MM1 handoff cannot materialize the full model")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256")
    if contract["contract_sha256"] != _digest(unsigned, label="MM1 handoff contract"):
        raise Qwen3MultimodalContractError("MM1 handoff digest mismatch")
    _reject_sensitive(contract)
    _canonical(contract, label="MM1 handoff contract")
    return dict(contract)


def build_mm1_transfer_binding(
    *,
    handoff: Mapping[str, Any],
    manifest: Mapping[str, Any],
    transfer_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a committed QW3 transfer reference to an MM1 visual handoff.

    The QW3 transport owns retry, cancellation, leases and filesystem cleanup.
    This adapter only joins the two path-free contracts after the receiver has
    committed the artifact, so an MM1 consumer never has to resolve a path or
    trust an uncommitted offset.
    """
    safe_handoff = validate_mm1_handoff_contract(handoff, manifest)
    reference = _validate_mm1_transfer_reference(transfer_reference)
    artifact = safe_handoff["artifact"]
    if (
        reference["artifact_id"] != artifact["artifact_id"]
        or reference["size_bytes"] != artifact["size_bytes"]
        or reference["sha256"] != artifact["sha256"]
        or reference["source_node_id"] != safe_handoff["source_node_id"]
        or reference["target_node_id"] != safe_handoff["target_node_id"]
        or reference["generation"] != safe_handoff["generation"]
        or reference["phase"] != safe_handoff["phase"]
    ):
        raise Qwen3MultimodalContractError(
            "MM1 transfer reference does not match visual handoff",
        )
    values = {
        "schema_version": MM1_SCHEMA_VERSION,
        "binding_kind": "qwen3_multimodal_transfer_binding",
        "handoff_contract_sha256": safe_handoff["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "transfer_reference": reference,
        "full_model_materialized": False,
    }
    values["binding_sha256"] = _digest(values, label="MM1 transfer binding")
    return validate_mm1_transfer_binding(values, manifest=manifest, handoff=safe_handoff)


def _validate_mm1_transfer_reference(value: Any) -> dict[str, Any]:
    reference = _exact(value, _TRANSFER_REFERENCE_FIELDS, "transfer_reference")
    _safe_id(reference["artifact_id"], "transfer_reference.artifact_id")
    if reference["mode"] not in {"local", "network"}:
        raise Qwen3MultimodalContractError("transfer_reference.mode is unsupported")
    _safe_id(reference["source_node_id"], "transfer_reference.source_node_id")
    _safe_id(reference["target_node_id"], "transfer_reference.target_node_id")
    _sha256(reference["chain_id"], "transfer_reference.chain_id")
    reference["generation"] = _non_negative_int(
        reference["generation"], "transfer_reference.generation", maximum=2**31 - 1,
    )
    if reference["phase"] not in {"prefill", "decode"}:
        raise Qwen3MultimodalContractError("transfer_reference.phase is unsupported")
    reference["from_segment"] = _non_negative_int(
        reference["from_segment"], "transfer_reference.from_segment", maximum=1024,
    )
    reference["to_segment"] = _positive_int(
        reference["to_segment"], "transfer_reference.to_segment", maximum=1024,
    )
    if reference["to_segment"] != reference["from_segment"] + 1:
        raise Qwen3MultimodalContractError("transfer_reference boundary is not adjacent")
    reference["size_bytes"] = _positive_int(
        reference["size_bytes"], "transfer_reference.size_bytes", maximum=MM1_MAX_ARTIFACT_BYTES,
    )
    _sha256(reference["sha256"], "transfer_reference.sha256")
    return reference


def validate_mm1_transfer_binding(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted MM1/QW3 binding without exposing local paths."""
    safe_manifest = validate_mm1_model_manifest(manifest)
    safe_handoff = validate_mm1_handoff_contract(handoff, safe_manifest)
    binding = _exact(
        value,
        {
            "schema_version", "binding_kind", "handoff_contract_sha256",
            "manifest_sha256", "transfer_reference", "full_model_materialized",
            "binding_sha256",
        },
        "MM1 transfer binding",
    )
    if binding["schema_version"] != MM1_SCHEMA_VERSION or binding["binding_kind"] != "qwen3_multimodal_transfer_binding":
        raise Qwen3MultimodalContractError("MM1 transfer binding version/kind is unsupported")
    if binding["handoff_contract_sha256"] != safe_handoff["contract_sha256"] or binding["manifest_sha256"] != safe_manifest["manifest_sha256"]:
        raise Qwen3MultimodalContractError("MM1 transfer binding identity does not match")
    reference = _validate_mm1_transfer_reference(binding["transfer_reference"])
    artifact = safe_handoff["artifact"]
    if (
        reference["artifact_id"] != artifact["artifact_id"]
        or reference["size_bytes"] != artifact["size_bytes"]
        or reference["sha256"] != artifact["sha256"]
        or reference["source_node_id"] != safe_handoff["source_node_id"]
        or reference["target_node_id"] != safe_handoff["target_node_id"]
        or reference["generation"] != safe_handoff["generation"]
        or reference["phase"] != safe_handoff["phase"]
    ):
        raise Qwen3MultimodalContractError("MM1 transfer reference does not match visual handoff")
    if binding["full_model_materialized"] is not False:
        raise Qwen3MultimodalContractError("MM1 transfer binding cannot materialize the full model")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256")
    if binding["binding_sha256"] != _digest(unsigned, label="MM1 transfer binding"):
        raise Qwen3MultimodalContractError("MM1 transfer binding digest mismatch")
    _reject_sensitive(binding)
    _canonical(binding, label="MM1 transfer binding")
    normalized = dict(binding)
    normalized["transfer_reference"] = reference
    return normalized


__all__ = [
    "MM1_MAX_ARTIFACT_BYTES",
    "MM1_MAX_CONTRACT_BYTES",
    "MM1_SCHEMA_VERSION",
    "Qwen3MultimodalContractError",
    "build_mm1_handoff_contract",
    "build_mm1_model_manifest",
    "build_mm1_model_profile",
    "build_mm1_transfer_binding",
    "estimate_mm1_capacity",
    "validate_mm1_handoff_contract",
    "validate_mm1_model_manifest",
    "validate_mm1_transfer_binding",
]
