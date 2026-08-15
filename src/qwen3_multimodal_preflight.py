"""Offline metadata preflight for Qwen3 multimodal processor workers.

The preflight intentionally stops before importing Transformers, decoding
media, or opening model weights.  It proves that a bounded set of local JSON
metadata agrees with an MM1 manifest and produces path-free worker contracts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from qwen3_multimodal_contract import (
    MM1_SCHEMA_VERSION,
    build_mm1_model_profile,
    validate_mm1_model_manifest,
)


MM1_PREFLIGHT_MAX_BYTES = 64 * 1024
MM1_PROCESSOR_JSON_MAX_BYTES = 2 * 1024 * 1024
MM1_MAX_MEDIA_PIXELS = 1_000_000_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SENSITIVE_KEYS = {
    "path", "model_path", "input_path", "output_path", "filesystem_path",
    "image_path", "video_path", "image_bytes", "video_bytes", "pixel_values",
    "inputs", "payload", "prompt", "messages", "input_ids", "tensors",
    "ticket", "grant",
}
_ASSET_FILES = {
    "model_config": "config.json",
    "image_processor": "preprocessor_config.json",
    "video_processor": "video_preprocessor_config.json",
    "tokenizer_config": "tokenizer_config.json",
}
_PROCESSOR_FIELDS = {
    "inspection_sha256", "processor_id", "revision", "processor_class",
    "tokenizer_class", "image_processor_type", "video_processor_type", "patch_size",
    "spatial_merge_size", "temporal_patch_size", "image_token_id",
    "video_token_id", "max_images", "max_video_frames", "image_min_pixels",
    "image_max_pixels", "video_min_pixels", "video_max_pixels",
}
_PROCESSOR_SMOKE_RUNTIME_FIELDS = {
    "transformers_version", "isolated", "local_files_only", "trust_remote_code",
    "processor_class", "image_processor_class", "video_processor_class",
    "tokenizer_class", "declared_tokenizer_class", "image_token_id",
    "video_token_id", "patch_size", "temporal_patch_size", "merge_size",
}
_PROCESSOR_SMOKE_CLEANUP_FIELDS = {
    "attempted", "completed", "objects_released", "weight_materialized",
    "full_model_materialized",
}


class Qwen3MultimodalPreflightError(ValueError):
    """The processor metadata or visual worker preflight is not admissible."""


def _canonical(value: Any, *, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalPreflightError(f"{label} is not JSON serializable") from exc
    if len(encoded) > MM1_PREFLIGHT_MAX_BYTES:
        raise Qwen3MultimodalPreflightError(f"{label} exceeds serialization limit")
    return encoded


def _digest(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical(value, label=label)).hexdigest()


def _exact(value: Any, fields: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Qwen3MultimodalPreflightError(f"{field} fields are invalid")
    return dict(value)


def _safe_id(value: Any, field: str) -> str:
    result = str(value or "")
    if _SAFE_ID.fullmatch(result) is None:
        raise Qwen3MultimodalPreflightError(f"{field} is invalid")
    return result


def _sha256(value: Any, field: str) -> str:
    result = str(value or "").lower()
    if _SHA256.fullmatch(result) is None:
        raise Qwen3MultimodalPreflightError(f"{field} must be a lowercase SHA-256")
    return result


def _int(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = MM1_MAX_MEDIA_PIXELS,
) -> int:
    if isinstance(value, bool):
        raise Qwen3MultimodalPreflightError(f"{field} is invalid")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3MultimodalPreflightError(f"{field} is invalid") from exc
    if result < minimum or result > maximum:
        raise Qwen3MultimodalPreflightError(f"{field} is outside limits")
    return result


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise Qwen3MultimodalPreflightError(
                    f"MM1 visual preflight cannot contain {key}",
                )
            _reject_sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive(item)


def _read_json(root: Path, name: str, *, label: str) -> tuple[dict[str, Any], bytes]:
    candidate = root / name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Qwen3MultimodalPreflightError(f"{label} is missing") from exc
    if resolved.parent != root or not resolved.is_file():
        raise Qwen3MultimodalPreflightError(f"{label} escapes model directory")
    size = resolved.stat().st_size
    if size <= 0 or size > MM1_PROCESSOR_JSON_MAX_BYTES:
        raise Qwen3MultimodalPreflightError(f"{label} size is outside limits")
    data = resolved.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3MultimodalPreflightError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Qwen3MultimodalPreflightError(f"{label} must be a JSON object")
    return value, data


def _processor_size(value: Mapping[str, Any], field: str) -> tuple[int, int]:
    size = value.get("size")
    if not isinstance(size, Mapping):
        raise Qwen3MultimodalPreflightError(f"{field}.size is invalid")
    shortest = _int(size.get("shortest_edge"), f"{field}.size.shortest_edge", minimum=1)
    longest = _int(size.get("longest_edge"), f"{field}.size.longest_edge", minimum=1)
    if shortest > longest:
        raise Qwen3MultimodalPreflightError(f"{field}.size limits are inverted")
    return shortest, longest


def _processor_geometry(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    field: str,
) -> tuple[int, int, int]:
    patch = _int(value.get("patch_size"), f"{field}.patch_size", minimum=1, maximum=65_536)
    temporal = _int(
        value.get("temporal_patch_size"),
        f"{field}.temporal_patch_size",
        minimum=1,
        maximum=65_536,
    )
    merge = _int(value.get("merge_size"), f"{field}.merge_size", minimum=1, maximum=65_536)
    vision = manifest["vision"]
    if (
        patch != vision["patch_size"]
        or temporal != vision["temporal_patch_size"]
        or merge != vision["spatial_merge_size"]
    ):
        raise Qwen3MultimodalPreflightError(
            f"{field} geometry does not match MM1 manifest",
        )
    return patch, temporal, merge


def inspect_mm1_processor_assets(
    model_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect bounded processor JSON files without loading code or weights."""
    safe_manifest = validate_mm1_model_manifest(manifest)
    try:
        root = Path(model_dir).resolve(strict=True)
    except OSError as exc:
        raise Qwen3MultimodalPreflightError("model directory is missing") from exc
    if not root.is_dir():
        raise Qwen3MultimodalPreflightError("model directory is invalid")

    documents: dict[str, dict[str, Any]] = {}
    assets: list[dict[str, Any]] = []
    for asset_kind, filename in _ASSET_FILES.items():
        value, data = _read_json(root, filename, label=asset_kind)
        documents[asset_kind] = value
        assets.append({
            "asset_kind": asset_kind,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    try:
        profile = build_mm1_model_profile(documents["model_config"])
    except ValueError as exc:
        raise Qwen3MultimodalPreflightError("model config cannot produce an MM1 profile") from exc
    if profile["model_family"] != safe_manifest["model_family"]:
        raise Qwen3MultimodalPreflightError("model config family does not match MM1 manifest")
    if profile["text"] != safe_manifest["text"] or profile["vision"] != safe_manifest["vision"]:
        raise Qwen3MultimodalPreflightError("model config shape does not match MM1 manifest")
    for token_name in ("image_token_id", "video_token_id"):
        if profile["processor"][token_name] != safe_manifest["processor"][token_name]:
            raise Qwen3MultimodalPreflightError(
                f"model config {token_name} does not match MM1 manifest",
            )

    image = documents["image_processor"]
    video = documents["video_processor"]
    if image.get("processor_class") != "Qwen3VLProcessor" or video.get("processor_class") != "Qwen3VLProcessor":
        raise Qwen3MultimodalPreflightError("processor class is unsupported")
    if image.get("image_processor_type") != "Qwen2VLImageProcessorFast":
        raise Qwen3MultimodalPreflightError("image processor type is unsupported")
    if video.get("video_processor_type") != "Qwen3VLVideoProcessor":
        raise Qwen3MultimodalPreflightError("video processor type is unsupported")
    tokenizer = documents["tokenizer_config"]
    if tokenizer.get("tokenizer_class") != "Qwen2Tokenizer":
        raise Qwen3MultimodalPreflightError("tokenizer class is unsupported")
    image_patch, image_temporal, image_merge = _processor_geometry(
        image, safe_manifest, field="image processor",
    )
    video_patch, video_temporal, video_merge = _processor_geometry(
        video, safe_manifest, field="video processor",
    )
    if (image_patch, image_temporal, image_merge) != (video_patch, video_temporal, video_merge):
        raise Qwen3MultimodalPreflightError("image/video processor geometry differs")
    image_min, image_max = _processor_size(image, "image processor")
    video_min, video_max = _processor_size(video, "video processor")
    declared = safe_manifest["processor"]
    processor = {
        "processor_id": declared["processor_id"],
        "revision": declared["revision"],
        "processor_class": image["processor_class"],
        "tokenizer_class": tokenizer["tokenizer_class"],
        "image_processor_type": image["image_processor_type"],
        "video_processor_type": video["video_processor_type"],
        "patch_size": image_patch,
        "spatial_merge_size": image_merge,
        "temporal_patch_size": image_temporal,
        "image_token_id": declared["image_token_id"],
        "video_token_id": declared["video_token_id"],
        "max_images": declared["max_images"],
        "max_video_frames": declared["max_video_frames"],
        "image_min_pixels": max(declared["min_pixels"], image_min),
        "image_max_pixels": min(declared["max_pixels"], image_max),
        "video_min_pixels": max(declared["min_pixels"], video_min),
        "video_max_pixels": min(declared["max_pixels"], video_max),
    }
    if (
        processor["image_min_pixels"] > processor["image_max_pixels"]
        or processor["video_min_pixels"] > processor["video_max_pixels"]
    ):
        raise Qwen3MultimodalPreflightError("processor/manifest pixel limits do not overlap")
    assets.sort(key=lambda item: item["asset_kind"])
    inspection = {
        "schema_version": MM1_SCHEMA_VERSION,
        "inspection_kind": "qwen3_visual_processor_assets",
        "model_id": safe_manifest["model_id"],
        "model_family": safe_manifest["model_family"],
        "runtime": safe_manifest["runtime"],
        "revision": safe_manifest["revision"],
        "manifest_sha256": safe_manifest["manifest_sha256"],
        "processor": processor,
        "assets": assets,
        "full_model_materialized": False,
        "weight_materialized": False,
    }
    inspection["inspection_sha256"] = _digest(inspection, label="MM1 processor inspection")
    return validate_mm1_processor_inspection(inspection, manifest=safe_manifest)


def validate_mm1_processor_inspection(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    inspection = _exact(
        value,
        {
            "schema_version", "inspection_kind", "model_id", "model_family",
            "runtime", "revision", "manifest_sha256", "processor", "assets",
            "full_model_materialized", "weight_materialized", "inspection_sha256",
        },
        "MM1 processor inspection",
    )
    if (
        inspection["schema_version"] != MM1_SCHEMA_VERSION
        or inspection["inspection_kind"] != "qwen3_visual_processor_assets"
    ):
        raise Qwen3MultimodalPreflightError("MM1 processor inspection version/kind is unsupported")
    for name in ("model_id", "model_family", "runtime", "revision", "manifest_sha256"):
        if inspection[name] != safe_manifest[name]:
            raise Qwen3MultimodalPreflightError(f"MM1 processor inspection {name} does not match")
    processor = _exact(inspection["processor"], _PROCESSOR_FIELDS - {"inspection_sha256"}, "processor")
    declared = safe_manifest["processor"]
    for name in (
        "processor_id", "revision", "image_token_id", "video_token_id",
        "max_images", "max_video_frames",
    ):
        if processor[name] != declared[name]:
            raise Qwen3MultimodalPreflightError(f"processor.{name} does not match manifest")
    if processor["processor_class"] != "Qwen3VLProcessor":
        raise Qwen3MultimodalPreflightError("processor class is unsupported")
    if processor["tokenizer_class"] != "Qwen2Tokenizer":
        raise Qwen3MultimodalPreflightError("tokenizer class is unsupported")
    if processor["image_processor_type"] != "Qwen2VLImageProcessorFast":
        raise Qwen3MultimodalPreflightError("image processor type is unsupported")
    if processor["video_processor_type"] != "Qwen3VLVideoProcessor":
        raise Qwen3MultimodalPreflightError("video processor type is unsupported")
    for name in ("patch_size", "spatial_merge_size", "temporal_patch_size"):
        processor[name] = _int(processor[name], f"processor.{name}", minimum=1, maximum=65_536)
    vision = safe_manifest["vision"]
    if (
        processor["patch_size"] != vision["patch_size"]
        or processor["spatial_merge_size"] != vision["spatial_merge_size"]
        or processor["temporal_patch_size"] != vision["temporal_patch_size"]
    ):
        raise Qwen3MultimodalPreflightError("processor geometry does not match manifest")
    for modality in ("image", "video"):
        minimum = _int(processor[f"{modality}_min_pixels"], f"processor.{modality}_min_pixels", minimum=1)
        maximum = _int(processor[f"{modality}_max_pixels"], f"processor.{modality}_max_pixels", minimum=1)
        if minimum > maximum:
            raise Qwen3MultimodalPreflightError(f"processor {modality} limits are inverted")
    assets = inspection["assets"]
    if not isinstance(assets, list) or len(assets) != len(_ASSET_FILES):
        raise Qwen3MultimodalPreflightError("processor inspection assets are invalid")
    normalized_assets = []
    for index, item in enumerate(assets):
        asset = _exact(item, {"asset_kind", "size_bytes", "sha256"}, f"assets[{index}]")
        if asset["asset_kind"] not in _ASSET_FILES:
            raise Qwen3MultimodalPreflightError("processor inspection asset kind is unsupported")
        asset["size_bytes"] = _int(
            asset["size_bytes"], f"assets[{index}].size_bytes",
            minimum=1, maximum=MM1_PROCESSOR_JSON_MAX_BYTES,
        )
        _sha256(asset["sha256"], f"assets[{index}].sha256")
        normalized_assets.append(asset)
    kinds = [item["asset_kind"] for item in normalized_assets]
    if kinds != sorted(_ASSET_FILES) or len(set(kinds)) != len(kinds):
        raise Qwen3MultimodalPreflightError("processor inspection assets must be unique and ordered")
    if inspection["full_model_materialized"] is not False or inspection["weight_materialized"] is not False:
        raise Qwen3MultimodalPreflightError("processor inspection cannot materialize weights")
    unsigned = dict(inspection)
    unsigned.pop("inspection_sha256")
    if inspection["inspection_sha256"] != _digest(unsigned, label="MM1 processor inspection"):
        raise Qwen3MultimodalPreflightError("MM1 processor inspection digest mismatch")
    _reject_sensitive(inspection)
    _canonical(inspection, label="MM1 processor inspection")
    normalized = dict(inspection)
    normalized["processor"] = processor
    normalized["assets"] = normalized_assets
    return normalized


def _selected_components(
    component_ids: Any,
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    if not isinstance(component_ids, list) or not component_ids:
        raise Qwen3MultimodalPreflightError("component_ids are invalid")
    selected = [_safe_id(item, "component_ids") for item in component_ids]
    if selected != sorted(selected) or len(set(selected)) != len(selected):
        raise Qwen3MultimodalPreflightError("component_ids must be unique and ordered")
    by_id = {item["component_id"]: item for item in manifest["components"]}
    if any(item not in by_id for item in selected):
        raise Qwen3MultimodalPreflightError("component_ids contain an unknown component")
    kinds = sorted(by_id[item]["component_kind"] for item in selected)
    required = (
        ["processor", "vision_weights"]
        if manifest["runtime"] == "transformers_sidecar"
        else ["mmproj", "processor"]
    )
    if kinds != required:
        raise Qwen3MultimodalPreflightError(
            "visual worker component assignment is not least-privilege",
        )
    return selected, kinds


def _validate_security(value: Any) -> dict[str, bool]:
    security = _exact(
        value,
        {"offline", "local_files_only", "trust_remote_code", "network_disabled"},
        "security",
    )
    if (
        security["offline"] is not True
        or security["local_files_only"] is not True
        or security["network_disabled"] is not True
        or security["trust_remote_code"] is not False
    ):
        raise Qwen3MultimodalPreflightError("visual worker security policy is not offline-safe")
    return security


def _validate_media(value: Any, processor: Mapping[str, Any]) -> dict[str, Any]:
    media = _exact(
        value,
        {"modality", "item_count", "frame_count", "width", "height", "pixel_count"},
        "media",
    )
    if media["modality"] not in {"image", "video"}:
        raise Qwen3MultimodalPreflightError("media modality is unsupported")
    media["item_count"] = _int(
        media["item_count"], "media.item_count", minimum=1,
        maximum=processor["max_images"],
    )
    media["width"] = _int(media["width"], "media.width", minimum=1, maximum=1_000_000)
    media["height"] = _int(media["height"], "media.height", minimum=1, maximum=1_000_000)
    media["pixel_count"] = _int(media["pixel_count"], "media.pixel_count", minimum=1)
    if media["width"] * media["height"] != media["pixel_count"]:
        raise Qwen3MultimodalPreflightError("media pixel_count does not match dimensions")
    if media["modality"] == "image":
        if media["frame_count"] != 0:
            raise Qwen3MultimodalPreflightError("image media cannot declare video frames")
    else:
        media["frame_count"] = _int(
            media["frame_count"], "media.frame_count", minimum=1,
            maximum=processor["max_video_frames"],
        )
    minimum = processor[f"{media['modality']}_min_pixels"]
    maximum = processor[f"{media['modality']}_max_pixels"]
    if not minimum <= media["pixel_count"] <= maximum:
        raise Qwen3MultimodalPreflightError("media pixel_count is outside processor limits")
    return media


def build_mm1_visual_worker_request(
    *,
    request_id: str,
    node_id: str,
    manifest: Mapping[str, Any],
    inspection: Mapping[str, Any],
    component_ids: list[str],
    modality: str,
    item_count: int,
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    safe_inspection = validate_mm1_processor_inspection(inspection, manifest=safe_manifest)
    selected, _kinds = _selected_components(component_ids, safe_manifest)
    processor = dict(safe_inspection["processor"])
    processor["inspection_sha256"] = safe_inspection["inspection_sha256"]
    media = {
        "modality": str(modality),
        "item_count": item_count,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "pixel_count": width * height,
    }
    request = {
        "schema_version": MM1_SCHEMA_VERSION,
        "request_kind": "qwen3_visual_worker_preflight",
        "request_id": _safe_id(request_id, "request_id"),
        "model_id": safe_manifest["model_id"],
        "model_family": safe_manifest["model_family"],
        "runtime": safe_manifest["runtime"],
        "revision": safe_manifest["revision"],
        "manifest_sha256": safe_manifest["manifest_sha256"],
        "node_id": _safe_id(node_id, "node_id"),
        "component_ids": selected,
        "processor": processor,
        "media": _validate_media(media, processor),
        "security": {
            "offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "network_disabled": True,
        },
        "full_model_materialized": False,
    }
    request["request_sha256"] = _digest(request, label="MM1 visual worker request")
    return validate_mm1_visual_worker_request(
        request, manifest=safe_manifest, inspection=safe_inspection,
    )


def validate_mm1_visual_worker_request(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    safe_manifest = validate_mm1_model_manifest(manifest)
    safe_inspection = validate_mm1_processor_inspection(inspection, manifest=safe_manifest)
    request = _exact(
        value,
        {
            "schema_version", "request_kind", "request_id", "model_id",
            "model_family", "runtime", "revision", "manifest_sha256", "node_id",
            "component_ids", "processor", "media", "security",
            "full_model_materialized", "request_sha256",
        },
        "MM1 visual worker request",
    )
    if request["schema_version"] != MM1_SCHEMA_VERSION or request["request_kind"] != "qwen3_visual_worker_preflight":
        raise Qwen3MultimodalPreflightError("MM1 visual worker request version/kind is unsupported")
    _safe_id(request["request_id"], "request_id")
    _safe_id(request["node_id"], "node_id")
    for name in ("model_id", "model_family", "runtime", "revision", "manifest_sha256"):
        if request[name] != safe_manifest[name]:
            raise Qwen3MultimodalPreflightError(f"MM1 visual worker request {name} does not match")
    selected, _kinds = _selected_components(request["component_ids"], safe_manifest)
    processor = _exact(request["processor"], _PROCESSOR_FIELDS, "processor")
    if processor["inspection_sha256"] != safe_inspection["inspection_sha256"]:
        raise Qwen3MultimodalPreflightError("processor inspection identity does not match")
    expected_processor = dict(safe_inspection["processor"])
    expected_processor["inspection_sha256"] = safe_inspection["inspection_sha256"]
    if processor != expected_processor:
        raise Qwen3MultimodalPreflightError("processor projection does not match inspection")
    media = _validate_media(request["media"], processor)
    security = _validate_security(request["security"])
    if request["full_model_materialized"] is not False:
        raise Qwen3MultimodalPreflightError("visual worker request cannot materialize the full model")
    unsigned = dict(request)
    unsigned.pop("request_sha256")
    if request["request_sha256"] != _digest(unsigned, label="MM1 visual worker request"):
        raise Qwen3MultimodalPreflightError("MM1 visual worker request digest mismatch")
    _reject_sensitive(request)
    _canonical(request, label="MM1 visual worker request")
    normalized = dict(request)
    normalized.update({
        "component_ids": selected,
        "processor": processor,
        "media": media,
        "security": security,
    })
    return normalized


def build_mm1_visual_worker_response(
    request: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> dict[str, Any]:
    safe_request = validate_mm1_visual_worker_request(
        request, manifest=manifest, inspection=inspection,
    )
    response = {
        "schema_version": MM1_SCHEMA_VERSION,
        "response_kind": "qwen3_visual_worker_preflight",
        "status": "ready_for_offline_start",
        "request_id": safe_request["request_id"],
        "request_sha256": safe_request["request_sha256"],
        "manifest_sha256": safe_request["manifest_sha256"],
        "model_id": safe_request["model_id"],
        "node_id": safe_request["node_id"],
        "processor_ready": True,
        "visual_worker_ready": True,
        "component_count": len(safe_request["component_ids"]),
        "security": dict(safe_request["security"]),
        "full_model_materialized": False,
        "weight_materialized": False,
    }
    response["response_sha256"] = _digest(response, label="MM1 visual worker response")
    return validate_mm1_visual_worker_response(response, request=safe_request)


def validate_mm1_visual_worker_response(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    response = _exact(
        value,
        {
            "schema_version", "response_kind", "status", "request_id",
            "request_sha256", "manifest_sha256", "model_id", "node_id",
            "processor_ready", "visual_worker_ready", "component_count", "security",
            "full_model_materialized", "weight_materialized", "response_sha256",
        },
        "MM1 visual worker response",
    )
    if response["schema_version"] != MM1_SCHEMA_VERSION or response["response_kind"] != "qwen3_visual_worker_preflight":
        raise Qwen3MultimodalPreflightError("MM1 visual worker response version/kind is unsupported")
    if response["status"] != "ready_for_offline_start":
        raise Qwen3MultimodalPreflightError("MM1 visual worker response status is unsupported")
    for name in ("request_id", "request_sha256", "manifest_sha256", "model_id", "node_id"):
        if response[name] != request[name]:
            raise Qwen3MultimodalPreflightError(f"MM1 visual worker response {name} does not match")
    if response["processor_ready"] is not True or response["visual_worker_ready"] is not True:
        raise Qwen3MultimodalPreflightError("MM1 visual worker response is not ready")
    response["component_count"] = _int(
        response["component_count"], "component_count", minimum=1, maximum=16,
    )
    if response["component_count"] != len(request["component_ids"]):
        raise Qwen3MultimodalPreflightError("MM1 visual worker response component count differs")
    response["security"] = _validate_security(response["security"])
    if response["security"] != request["security"]:
        raise Qwen3MultimodalPreflightError("MM1 visual worker response security differs")
    if response["full_model_materialized"] is not False or response["weight_materialized"] is not False:
        raise Qwen3MultimodalPreflightError("MM1 visual worker preflight cannot materialize weights")
    unsigned = dict(response)
    unsigned.pop("response_sha256")
    if response["response_sha256"] != _digest(unsigned, label="MM1 visual worker response"):
        raise Qwen3MultimodalPreflightError("MM1 visual worker response digest mismatch")
    _reject_sensitive(response)
    _canonical(response, label="MM1 visual worker response")
    return response


def build_mm1_visual_input_contract(
    media_tensor_reference: Mapping[str, Any],
    *,
    node_capacity_bytes: int,
    safety_margin: float = 1.2,
) -> dict[str, Any]:
    """MM1.9：以媒体张量参考为输入占位，构造视觉塔执行器的 path-free 输入契约。

    grid/token/字节预算与节点容量比对：required = 字节估计 × margin，
    超过 node_capacity 则 admitted=false（fail-closed，视觉塔不执行）。
    契约不含路径、像素值或权重。
    """
    reference = validate_mm1_media_tensor_reference(
        media_tensor_reference,
        model_id=str(media_tensor_reference["model_id"]),
        component_ids=list(media_tensor_reference["component_ids"]),
    )
    capacity = reference["capacity"]
    total_bytes = int(capacity.get("output_bytes_estimate") or 0)
    total_tokens = int(capacity.get("total_media_tokens") or 0)
    try:
        node_capacity = int(node_capacity_bytes)
        margin = float(safety_margin)
    except (TypeError, ValueError):
        raise Qwen3MultimodalPreflightError("MM1.9 node capacity is invalid")
    if node_capacity <= 0 or not 1.0 <= margin <= 10.0:
        raise Qwen3MultimodalPreflightError("MM1.9 node capacity or margin is out of bounds")
    required_bytes = int(total_bytes * margin)
    admitted = required_bytes <= node_capacity
    image_shape = list(reference["media"]["image"].get("pixel_values_shape") or [])
    video_shape = list(reference["media"]["video"].get("pixel_values_shape") or [])
    contract = {
        "schema_version": MM1_SCHEMA_VERSION,
        "contract_kind": "qwen3_visual_input_placeholder",
        "model_id": reference["model_id"],
        "media_reference_sha256": reference["reference_sha256"],
        "input": {
            "total_media_tokens": total_tokens,
            "total_bytes_estimate": total_bytes,
            "grid": {
                "image_shape": image_shape,
                "video_shape": video_shape,
            },
        },
        "capacity": {
            "node_capacity_bytes": node_capacity,
            "required_bytes": required_bytes,
            "safety_margin": margin,
            "admitted": bool(admitted),
        },
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    contract["contract_sha256"] = _digest(contract, label="MM1 visual input contract")
    return contract


def validate_mm1_visual_input_contract(
    value: Mapping[str, Any],
    *,
    media_tensor_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """只读校验视觉输入契约（身份/digest/预算不足即 admitted=false）。"""
    contract = _exact(
        value,
        {
            "schema_version", "contract_kind", "model_id",
            "media_reference_sha256", "input", "capacity",
            "weight_materialized", "full_model_materialized", "contract_sha256",
        },
        "MM1 visual input contract",
    )
    if (
        contract["schema_version"] != MM1_SCHEMA_VERSION
        or contract["contract_kind"] != "qwen3_visual_input_placeholder"
        or contract["model_id"] != media_tensor_reference["model_id"]
        or contract["media_reference_sha256"] != media_tensor_reference["reference_sha256"]
    ):
        raise Qwen3MultimodalPreflightError("MM1 visual input contract identity is invalid")
    if contract["weight_materialized"] is not False or contract["full_model_materialized"] is not False:
        raise Qwen3MultimodalPreflightError("MM1 visual input contract must stay weight-free")
    capacity = contract["capacity"]
    if capacity["admitted"] is True and int(capacity["required_bytes"]) > int(capacity["node_capacity_bytes"]):
        raise Qwen3MultimodalPreflightError("MM1 visual input contract admitted flag is inconsistent")
    if contract["contract_sha256"] != _digest(
        {key: val for key, val in contract.items() if key != "contract_sha256"},
        label="MM1 visual input contract",
    ):
        raise Qwen3MultimodalPreflightError("MM1 visual input contract digest is invalid")
    return contract


def build_mm1_media_tensor_reference(
    media_summary: Mapping[str, Any],
    *,
    model_id: str,
    component_ids: Sequence[str],
) -> dict[str, Any]:
    """MM1.8：把 MM1.7 媒体摘要投影为 path-free 的媒体张量参考。

    视觉组件占位：shape/dtype/grid/token 契约 + 容量预算（供视觉塔执行器
    输入占位与容量规划）。pixel_values、原始媒体、路径与 prompt 一律不进入；
    权重/媒体路径与摘要解耦（weight_materialized=false）。
    """
    if not isinstance(media_summary, Mapping) or not media_summary:
        raise Qwen3MultimodalPreflightError("MM1.8 media summary is required")
    image = media_summary.get("image")
    video = media_summary.get("video")
    if not isinstance(image, Mapping) or not isinstance(video, Mapping):
        raise Qwen3MultimodalPreflightError("MM1.8 media summary structure is invalid")
    if not image.get("pixel_values_shape") and not image.get("token_count_estimate"):
        raise Qwen3MultimodalPreflightError("MM1.8 image media entry is empty")
    safe_model_id = _safe_id(model_id, "model_id")
    safe_components = sorted(
        str(component) for component in component_ids if str(component)
    )
    if not safe_components:
        raise Qwen3MultimodalPreflightError("MM1.8 component_ids are required")

    def _tokens(value: Mapping[str, Any]) -> int:
        tokens = value.get("token_count_estimate")
        try:
            return int(tokens) if tokens is not None else 0
        except (TypeError, ValueError):
            return 0

    def _shape(value: Mapping[str, Any]) -> list[int]:
        shape = value.get("pixel_values_shape")
        if not isinstance(shape, (list, tuple)):
            return []
        return [int(item) for item in shape]

    image_tokens = _tokens(image)
    video_tokens = _tokens(video)
    reference = {
        "schema_version": MM1_SCHEMA_VERSION,
        "reference_kind": "qwen3_visual_media_tensor_placeholder",
        "model_id": safe_model_id,
        "component_ids": safe_components,
        "media": {
            "image": {
                "pixel_values_shape": _shape(image),
                "dtype": str(image.get("dtype") or ""),
                "token_count_estimate": image_tokens,
            },
            "video": {
                "pixel_values_shape": _shape(video),
                "dtype": str(video.get("dtype") or ""),
                "token_count_estimate": video_tokens,
            },
        },
        "capacity": {
            "total_media_tokens": image_tokens + video_tokens,
            "output_bytes_estimate": max(
                0, int(media_summary.get("output_bytes_estimate") or 0),
            ),
        },
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    reference["reference_sha256"] = _digest(reference, label="MM1 media tensor reference")
    return reference


def validate_mm1_media_tensor_reference(
    value: Mapping[str, Any],
    *,
    model_id: str,
    component_ids: Sequence[str],
) -> dict[str, Any]:
    """只读校验媒体张量参考（字段/身份/解耦）。"""
    reference = _exact(
        value,
        {
            "schema_version", "reference_kind", "model_id", "component_ids",
            "media", "capacity", "weight_materialized",
            "full_model_materialized", "reference_sha256",
        },
        "MM1 media tensor reference",
    )
    if (
        reference["schema_version"] != MM1_SCHEMA_VERSION
        or reference["reference_kind"] != "qwen3_visual_media_tensor_placeholder"
        or reference["model_id"] != model_id
        or sorted(str(item) for item in reference["component_ids"])
        != sorted(str(item) for item in component_ids)
    ):
        raise Qwen3MultimodalPreflightError("MM1 media tensor reference identity is invalid")
    if reference["weight_materialized"] is not False or reference["full_model_materialized"] is not False:
        raise Qwen3MultimodalPreflightError("MM1 media tensor reference must stay weight-free")
    media = reference["media"]
    for kind in ("image", "video"):
        entry = media.get(kind)
        if not isinstance(entry, Mapping):
            raise Qwen3MultimodalPreflightError(f"MM1 media tensor reference {kind} is invalid")
        if not isinstance(entry.get("pixel_values_shape"), list):
            raise Qwen3MultimodalPreflightError(f"MM1 media tensor reference {kind} shape is invalid")
        try:
            int(entry.get("token_count_estimate") or 0)
        except (TypeError, ValueError):
            raise Qwen3MultimodalPreflightError(f"MM1 media tensor reference {kind} tokens are invalid")
    if reference["reference_sha256"] != _digest(
        {key: val for key, val in reference.items() if key != "reference_sha256"},
        label="MM1 media tensor reference",
    ):
        raise Qwen3MultimodalPreflightError("MM1 media tensor reference digest is invalid")
    return reference


def build_mm1_processor_smoke_response(
    request: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    inspection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    media_summary: Mapping[str, Any] | None = None,
    media_tensor_reference: Mapping[str, Any] | None = None,
    visual_input_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a response after constructing AutoProcessor in the sidecar."""
    safe_request = validate_mm1_visual_worker_request(
        request, manifest=manifest, inspection=inspection,
    )
    runtime_value = _exact(runtime, _PROCESSOR_SMOKE_RUNTIME_FIELDS, "processor smoke runtime")
    response = {
        "schema_version": MM1_SCHEMA_VERSION,
        "response_kind": "qwen3_visual_worker_processor_smoke",
        "status": "ready_for_offline_start",
        "request_id": safe_request["request_id"],
        "request_sha256": safe_request["request_sha256"],
        "manifest_sha256": safe_request["manifest_sha256"],
        "model_id": safe_request["model_id"],
        "node_id": safe_request["node_id"],
        "processor_constructed": True,
        "visual_worker_ready": True,
        "component_count": len(safe_request["component_ids"]),
        "runtime": runtime_value,
        "cleanup": {
            "attempted": True,
            "completed": True,
            "objects_released": True,
            "weight_materialized": False,
            "full_model_materialized": False,
        },
        "media_summary": dict(media_summary) if media_summary else None,
        "media_tensor_reference": (
            dict(media_tensor_reference) if media_tensor_reference else None
        ),
        "visual_input_contract": (
            dict(visual_input_contract) if visual_input_contract else None
        ),
    }
    response["response_sha256"] = _digest(response, label="MM1 processor smoke response")
    return validate_mm1_processor_smoke_response(response, request=safe_request)


def validate_mm1_processor_smoke_response(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    response = _exact(
        value,
        {
            "schema_version", "response_kind", "status", "request_id",
            "request_sha256", "manifest_sha256", "model_id", "node_id",
            "processor_constructed", "visual_worker_ready", "component_count",
            "runtime", "cleanup", "media_summary",
            "media_tensor_reference", "visual_input_contract", "response_sha256",
        },
        "MM1 processor smoke response",
    )
    if (
        response["schema_version"] != MM1_SCHEMA_VERSION
        or response["response_kind"] != "qwen3_visual_worker_processor_smoke"
        or response["status"] != "ready_for_offline_start"
    ):
        raise Qwen3MultimodalPreflightError("MM1 processor smoke response version/status is unsupported")
    for name in ("request_id", "request_sha256", "manifest_sha256", "model_id", "node_id"):
        if response[name] != request[name]:
            raise Qwen3MultimodalPreflightError(f"MM1 processor smoke response {name} does not match")
    if response["processor_constructed"] is not True or response["visual_worker_ready"] is not True:
        raise Qwen3MultimodalPreflightError("MM1 processor smoke did not construct a ready worker")
    response["component_count"] = _int(
        response["component_count"], "component_count", minimum=1, maximum=16,
    )
    if response["component_count"] != len(request["component_ids"]):
        raise Qwen3MultimodalPreflightError("MM1 processor smoke component count differs")
    runtime = _exact(response["runtime"], _PROCESSOR_SMOKE_RUNTIME_FIELDS, "processor smoke runtime")
    version = str(runtime["transformers_version"] or "")
    version_tuple = []
    for part in version.split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        version_tuple.append(int(digits))
    if tuple(version_tuple or [0]) < (4, 51, 0):
        raise Qwen3MultimodalPreflightError("processor smoke Transformers version is too old")
    if (
        runtime["isolated"] is not True
        or runtime["local_files_only"] is not True
        or runtime["trust_remote_code"] is not False
    ):
        raise Qwen3MultimodalPreflightError("processor smoke runtime is not isolated/offline-safe")
    allowed_classes = {
        "processor_class": "Qwen3VLProcessor",
        "image_processor_class": "Qwen2VLImageProcessorFast",
        "video_processor_class": "Qwen3VLVideoProcessor",
    }
    for name, expected in allowed_classes.items():
        if runtime[name] != expected:
            raise Qwen3MultimodalPreflightError(f"processor smoke {name} is unsupported")
    if runtime["declared_tokenizer_class"] != request["processor"]["tokenizer_class"]:
        raise Qwen3MultimodalPreflightError("processor smoke declared tokenizer class differs")
    if runtime["tokenizer_class"] not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}:
        raise Qwen3MultimodalPreflightError("processor smoke tokenizer class is unsupported")
    for name in ("image_token_id", "video_token_id"):
        runtime[name] = _int(runtime[name], f"runtime.{name}", minimum=0, maximum=1_000_000_000)
        if runtime[name] != request["processor"][name]:
            raise Qwen3MultimodalPreflightError(f"processor smoke {name} differs")
    for name in ("patch_size", "temporal_patch_size", "merge_size"):
        runtime[name] = _int(runtime[name], f"runtime.{name}", minimum=1, maximum=65_536)
        expected = request["processor"]["spatial_merge_size"] if name == "merge_size" else request["processor"][name]
        if runtime[name] != expected:
            raise Qwen3MultimodalPreflightError(f"processor smoke {name} differs")
    cleanup = _exact(response["cleanup"], _PROCESSOR_SMOKE_CLEANUP_FIELDS, "processor smoke cleanup")
    if (
        cleanup["attempted"] is not True
        or cleanup["completed"] is not True
        or cleanup["objects_released"] is not True
        or cleanup["weight_materialized"] is not False
        or cleanup["full_model_materialized"] is not False
    ):
        raise Qwen3MultimodalPreflightError("processor smoke cleanup is incomplete")
    unsigned = dict(response)
    unsigned.pop("response_sha256")
    if response["response_sha256"] != _digest(unsigned, label="MM1 processor smoke response"):
        raise Qwen3MultimodalPreflightError("MM1 processor smoke response digest mismatch")
    _reject_sensitive(response)
    _canonical(response, label="MM1 processor smoke response")
    response["runtime"] = runtime
    response["cleanup"] = cleanup
    return response


__all__ = [
    "MM1_PREFLIGHT_MAX_BYTES",
    "MM1_PROCESSOR_JSON_MAX_BYTES",
    "Qwen3MultimodalPreflightError",
    "build_mm1_visual_worker_request",
    "build_mm1_visual_worker_response",
    "build_mm1_processor_smoke_response",
    "inspect_mm1_processor_assets",
    "validate_mm1_processor_inspection",
    "validate_mm1_visual_worker_request",
    "validate_mm1_visual_worker_response",
    "validate_mm1_processor_smoke_response",
]
