"""Assignment-scoped manifests for PyTorch pipeline workers.

The manifest is deliberately smaller than the active model manifest.  It
contains the config/tokenizer support files and only the safetensors shards
and keys needed by one assigned layer range.  ``model_sha256`` remains the
full model revision so a plan can never mix revisions, while ``manifest_sha256``
identifies the exact assignment payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from model_sync import compute_file_sha256
from pipeline_model_descriptor import (
    _ARCHITECTURE_LAYOUTS,
    _component_for_key,
    _decoder_config,
    _read_json_object,
    _safe_shard_path,
)


MANIFEST_SCHEMA_VERSION = 1
_ARTIFACT_SUFFIXES = (
    ".json", ".py", ".tiktoken", ".model", ".txt", ".jinja", ".spm", ".vocab",
)


class PipelineAssignmentManifestError(ValueError):
    """The assignment cannot be represented as a safe manifest."""


def _safe_relative(root: Path, relative: str) -> Path:
    normalized = str(relative or "").replace("\\", "/").lstrip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise PipelineAssignmentManifestError("unsafe assignment artifact path")
    path = (root / normalized).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PipelineAssignmentManifestError("assignment artifact escapes model root") from exc
    return path


def _index_weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        raw = _read_json_object(index_path).get("weight_map")
        if not isinstance(raw, dict) or not raw:
            raise PipelineAssignmentManifestError("safetensors index has no weight_map")
        return {str(key): str(value) for key, value in raw.items()}
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise PipelineAssignmentManifestError("safetensors is required for assignment manifest") from exc
    weight_map: dict[str, str] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise PipelineAssignmentManifestError(f"duplicate tensor key: {key}")
                weight_map[key] = shard.name
    if not weight_map:
        raise PipelineAssignmentManifestError("model has no safetensors weights")
    return weight_map


def _selected_keys(
    model_type: str,
    weight_map: dict[str, str],
    start_layer: int,
    end_layer: int,
    has_embedding: bool,
    has_lm_head: bool,
    tie_word_embeddings: bool = False,
) -> dict[str, list[str]]:
    layout = _ARCHITECTURE_LAYOUTS.get(model_type)
    if layout is None:
        raise PipelineAssignmentManifestError(f"unsupported model type: {model_type}")
    selected: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        component, layer_index = _component_for_key(key, layout)
        include = False
        if component == "layers":
            include = layer_index is not None and start_layer <= layer_index < end_layer
        elif component == "final_norm":
            include = True
        elif component == "embedding":
            include = bool(has_embedding) or (
                bool(has_lm_head) and bool(tie_word_embeddings)
            )
        elif component == "lm_head":
            include = bool(has_lm_head)
        elif component in {"visual", "mtp", "multimodal"}:
            include = False
        else:
            # ``other`` is charged with the output side of the plan, so keep
            # it with the final assignment rather than duplicating it.
            include = bool(has_lm_head)
        if include:
            selected.setdefault(filename, []).append(key)
    for keys in selected.values():
        keys.sort()
    if not selected:
        raise PipelineAssignmentManifestError("assignment selects no safetensors keys")
    return selected


def _support_files(root: Path) -> list[str]:
    paths: list[str] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in sorted(filenames):
            if filename in {"model.sha256", "model.sha256.meta.json"} or filename.endswith(".part"):
                continue
            relative = Path(directory, filename).relative_to(root).as_posix()
            if relative in {"model.safetensors.index.json", "config.json"}:
                continue
            if filename.lower().endswith(_ARTIFACT_SUFFIXES):
                paths.append(relative)
    return sorted(paths)


def _manifest_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_assignment_manifest(
    model_path: str | Path,
    *,
    model_id: str,
    model_sha256: str,
    config_id: str,
    plan_id: str,
    node_id: str,
    start_layer: int,
    end_layer: int,
    total_layers: int,
    has_embedding: bool = False,
    has_lm_head: bool = False,
) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise PipelineAssignmentManifestError("model directory does not exist")
    if not config_id or not plan_id or not node_id or not model_sha256:
        raise PipelineAssignmentManifestError("assignment identity is incomplete")
    start_layer, end_layer, total_layers = int(start_layer), int(end_layer), int(total_layers)
    if start_layer < 0 or end_layer <= start_layer or end_layer > total_layers:
        raise PipelineAssignmentManifestError("assignment layer range is invalid")
    config = _read_json_object(root / "config.json")
    model_type = str(config.get("model_type", "") or "").lower()
    layout = _ARCHITECTURE_LAYOUTS.get(model_type)
    if layout is None:
        raise PipelineAssignmentManifestError(f"unsupported model type: {model_type}")
    try:
        configured_layers = int(_decoder_config(config).get("num_hidden_layers", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise PipelineAssignmentManifestError("config num_hidden_layers is invalid") from exc
    if configured_layers != total_layers:
        raise PipelineAssignmentManifestError("assignment total_layers differs from config")
    tie_word_embeddings = bool(_decoder_config(config).get("tie_word_embeddings", False))
    selected = _selected_keys(
        model_type, _index_weight_map(root), start_layer, end_layer,
        bool(has_embedding), bool(has_lm_head), tie_word_embeddings,
    )
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise PipelineAssignmentManifestError("safetensors is required for assignment manifest") from exc
    for filename, keys in selected.items():
        shard = _safe_shard_path(root, filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
        missing = sorted(set(keys) - available)
        if missing:
            raise PipelineAssignmentManifestError(
                f"assignment index references missing tensor keys: {', '.join(missing[:4])}"
            )
    files: list[dict[str, Any]] = []

    def add_file(relative: str, *, keys: list[str] | None = None) -> None:
        path = _safe_relative(root, relative)
        if not path.is_file():
            raise PipelineAssignmentManifestError(f"assignment artifact missing: {relative}")
        item: dict[str, Any] = {
            "path": Path(relative).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": compute_file_sha256(path),
            "kind": "weights" if path.suffix.lower() == ".safetensors" else "support",
        }
        if keys is not None:
            item["keys"] = list(keys)
            item["key_set_sha256"] = hashlib.sha256(
                "\n".join(keys).encode("utf-8")
            ).hexdigest()
        files.append(item)

    add_file("config.json")
    # The worker always receives a filtered index, including models that use a
    # single unindexed safetensors file.  This prevents the loader from
    # scanning an entire shard and accidentally seeing unassigned layers.
    filtered_index = {
        "weight_map": {
            key: filename for filename, keys in selected.items() for key in keys
        },
        "metadata": {"total_size": 0},
    }
    filtered_index_bytes = json.dumps(
        filtered_index, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source_index = root / "model.safetensors.index.json"
    if source_index.is_file():
        add_file("model.safetensors.index.json")
    else:
        files.append({
            "path": "model.safetensors.index.json",
            "size_bytes": len(filtered_index_bytes),
            "sha256": hashlib.sha256(filtered_index_bytes).hexdigest(),
            "kind": "support",
        })
    files[-1]["size_bytes"] = len(filtered_index_bytes)
    files[-1]["sha256"] = hashlib.sha256(filtered_index_bytes).hexdigest()
    files[-1]["filtered_weight_map"] = filtered_index["weight_map"]
    for filename, keys in sorted(selected.items()):
        add_file(filename, keys=keys)
    for relative in _support_files(root):
        add_file(relative)
    files.sort(key=lambda item: item["path"])
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "pytorch_pipeline_assignment",
        "model_id": model_id,
        "model_sha256": model_sha256,
        "model_type": model_type,
        "total_layers": total_layers,
        "config_id": config_id,
        "plan_id": plan_id,
        "node_id": node_id,
        "layer_range": [start_layer, end_layer],
        "has_embedding": bool(has_embedding),
        "has_lm_head": bool(has_lm_head),
        "tie_word_embeddings": tie_word_embeddings,
        "output_weight_source": (
            f"{layout['embedding_prefixes'][0]}weight"
            if bool(has_lm_head) and tie_word_embeddings
            else "lm_head.weight"
            if bool(has_lm_head)
            else ""
        ),
        "files": files,
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def public_assignment_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe manifest projection; never expose local paths."""
    return json.loads(json.dumps(payload, ensure_ascii=True))
