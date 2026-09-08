"""Read-only local model metadata probe for S1.5."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import CapabilityState, ModelProfile


MAX_METADATA_BYTES = 8 * 1024 * 1024
_TOOL_SIGNAL = re.compile(r"\b(tool|tools|tool_calls?|function_call|functions?)\b", re.I)
_JSON_SIGNAL = re.compile(r"\b(json|arguments|parameters)\b", re.I)
_MM_SIGNAL = re.compile(r"\b(vision|visual|multimodal|mmproj|image|video|vl)\b", re.I)
_THINKING_SIGNAL = re.compile(r"enable_thinking|thinking|reasoning", re.I)
_TOKENIZER_NAMES = ("tokenizer", "vocab", "merges", "spiece", "sentencepiece")
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, "metadata_too_large"
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid"


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, "metadata_too_large"
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError):
        return None, "invalid"


def _file_inventory(path: Path, *, hash_weights: bool) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return an artifact digest without materializing tensors in memory."""

    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    entries: list[dict[str, Any]] = []
    names: list[str] = []
    errors: list[str] = []
    for item in sorted(files, key=lambda value: value.as_posix().lower()):
        try:
            relative = item.name if path.is_file() else item.relative_to(path).as_posix()
            size = item.stat().st_size
            suffix = item.suffix.lower()
            entry: dict[str, Any] = {"name": relative, "size_bytes": size}
            if hash_weights or suffix not in _WEIGHT_SUFFIXES:
                entry["sha256"] = _sha256_file(item)
            else:
                entry["sha256"] = None
            entries.append(entry)
            names.append(relative)
        except OSError:
            errors.append("artifact_file_unreadable:" + item.name)
    digest = _sha256_bytes(_canonical(entries))
    return digest, tuple(names), tuple(errors)


def _path_digest(path: Path) -> str:
    return _sha256_bytes(str(path.expanduser().absolute().resolve(strict=False)).encode("utf-8"))


def _template_from_metadata(tokenizer: Mapping[str, Any], sidecar: str | None) -> str:
    value = tokenizer.get("chat_template", "")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("default", "chat_template", "template"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return sidecar or ""


def _capability(status: str, *evidence: str) -> CapabilityState:
    return CapabilityState(status=status, evidence=tuple(item for item in evidence if item))


def _infer_prompt_family(model_id: str) -> str:
    value = model_id.lower()
    if "gemma" in value:
        return "gemma_chat_v1"
    if "qwen" in value or value.startswith("qw"):
        return "qwen_chat_v1"
    return "generic_chat_v1"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Static probe result; it never implies that generation was executed."""

    profile: ModelProfile
    errors: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    path_digest: str | None = None
    weights_loaded: bool = False
    network_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.as_dict(),
            "errors": list(self.errors),
            "files": list(self.files),
            "path_digest": self.path_digest,
            "weights_loaded": self.weights_loaded,
            "network_used": self.network_used,
        }


def probe_local_model(
    path: str | Path,
    *,
    model_id: str | None = None,
    backend: str = "llama_server",
    revision: str = "local-probe-v1",
    hash_weights: bool = False,
) -> ProbeResult:
    """Build a candidate profile from local metadata only.

    Directory artifacts use an inventory digest by default: metadata files are
    hashed, while large weight files contribute their names and sizes.  Set
    ``hash_weights=True`` when a complete content digest is worth the I/O cost.
    Neither mode loads tensor values into a framework.
    """

    artifact = Path(path).expanduser()
    inferred_id = model_id or artifact.stem if artifact.is_file() else model_id or artifact.name
    errors: list[str] = []
    files: tuple[str, ...] = ()
    if not artifact.exists() or not (artifact.is_file() or artifact.is_dir()):
        profile = ModelProfile(
            model_id=inferred_id or "unknown-model",
            revision=revision,
            backend=backend,
            status="rejected",
            evidence={"probe_mode": "static_local", "error": "asset_missing", "weights_loaded": False, "network_used": False},
        )
        return ProbeResult(profile=profile, errors=("asset_missing",), path_digest=_path_digest(artifact))

    artifact_sha256, files, inventory_errors = _file_inventory(artifact, hash_weights=hash_weights)
    errors.extend(inventory_errors)
    metadata: dict[str, Any] = {}
    tokenizer_data: dict[str, Any] = {}
    generation_data: dict[str, Any] = {}
    template = ""
    metadata_files: list[str] = []
    asset_format = "gguf" if artifact.is_file() and artifact.suffix.lower() == ".gguf" else "safetensors"

    if artifact.is_dir():
        for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
            value, error = _read_json(artifact / name)
            if value is not None:
                metadata_files.append(name)
                if name == "config.json" and isinstance(value, dict):
                    metadata = value
                elif name == "tokenizer_config.json" and isinstance(value, dict):
                    tokenizer_data = value
                elif name == "generation_config.json" and isinstance(value, dict):
                    generation_data = value
            elif error not in {None, "missing"}:
                errors.append(f"{name}:{error}")
        sidecar, sidecar_error = _read_text(artifact / "chat_template.jinja")
        if sidecar is not None:
            template = sidecar
            metadata_files.append("chat_template.jinja")
        elif sidecar_error not in {None, "missing"}:
            errors.append(f"chat_template.jinja:{sidecar_error}")
    else:
        # GGUF metadata is intentionally not parsed here.  A later adapter can
        # provide exact tokenizer/template data; S1.5 records the unknown state
        # instead of guessing from a filename.
        metadata_files.append(artifact.name)
        sidecars = (
            artifact.with_suffix(".json"),
            artifact.with_suffix(".chat_template.jinja"),
        )
        for sidecar_path in sidecars:
            if sidecar_path.suffix == ".json":
                value, error = _read_json(sidecar_path)
                if isinstance(value, dict):
                    metadata = value
                    metadata_files.append(sidecar_path.name)
                    tokenizer_data = value.get("tokenizer_config", {}) if isinstance(value.get("tokenizer_config"), dict) else {}
                    break
                if error not in {None, "missing"}:
                    errors.append(f"{sidecar_path.name}:{error}")
            else:
                value, error = _read_text(sidecar_path)
                if value is not None:
                    template = value
                    metadata_files.append(sidecar_path.name)
                    break
                if error not in {None, "missing"}:
                    errors.append(f"{sidecar_path.name}:{error}")

    template = _template_from_metadata(tokenizer_data, template)
    template_digest = _sha256_bytes(template.encode("utf-8")) if template else None
    tokenizer_files = tuple(
        name for name in files
        if any(part in name.lower() for part in _TOKENIZER_NAMES)
    )
    tokenizer_entries: list[dict[str, str]] = []
    for name in tokenizer_files:
        tokenizer_path = artifact / name if artifact.is_dir() else artifact.parent / name
        try:
            tokenizer_entries.append({"name": name, "sha256": _sha256_file(tokenizer_path)})
        except OSError:
            errors.append("tokenizer_file_unreadable:" + name)
    tokenizer_digest = _sha256_bytes(_canonical(tokenizer_entries)) if tokenizer_entries else None
    template_lower = template.lower()
    tool_signal = bool(_TOOL_SIGNAL.search(template))
    json_signal = bool(_JSON_SIGNAL.search(template))
    result_signal = bool(re.search(r"role[^\n]{0,80}tool|tool[^\n]{0,80}result", template, re.I))
    mm_signal = bool(_MM_SIGNAL.search(template_lower)) or bool(_MM_SIGNAL.search(json.dumps(metadata, ensure_ascii=False)))
    thinking_signal = bool(_THINKING_SIGNAL.search(template)) or "enable_thinking" in json.dumps(tokenizer_data, ensure_ascii=False)

    n_ctx = metadata.get("max_position_embeddings", metadata.get("max_seq_len", 0))
    if not isinstance(n_ctx, int) or n_ctx < 0:
        n_ctx = 0
    stop = generation_data.get("stop_strings", generation_data.get("eos_token"))
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = []
    model_name = inferred_id or "unknown-model"
    roles = ["answer", "summarizer"]
    if "qwen3" in model_name.lower() or "littlelamb" in model_name.lower() or "tool" in model_name.lower():
        roles = ["tool_router", "summarizer"]
    capabilities = {
        "json_output": _capability("declared", "chat_template_json_signal") if json_signal else _capability("unknown", "static_execution_not_run"),
        "tool_call_generation": _capability("declared", "chat_template_tool_signal") if tool_signal else _capability("unknown", "static_execution_not_run"),
        "tool_result_reinjection": _capability("declared", "chat_template_tool_result_signal") if result_signal else _capability("unknown", "static_execution_not_run"),
        "multimodal": _capability("declared", "metadata_multimodal_signal") if mm_signal else _capability("unknown", "static_execution_not_run"),
        "thinking_control": _capability("declared", "metadata_thinking_signal") if thinking_signal else _capability("unknown", "static_execution_not_run"),
    }
    evidence = {
        "probe_mode": "static_local",
        "asset_format": asset_format,
        "artifact_digest_mode": "full_stream" if hash_weights or artifact.is_file() else "inventory",
        "metadata_files": sorted(set(metadata_files)),
        "tokenizer_files": list(tokenizer_files),
        "weights_loaded": False,
        "network_used": False,
        "chat_template_present": bool(template),
        "model_type": metadata.get("model_type", "unknown"),
        "architectures": metadata.get("architectures", []),
        "fixture_set": "small-model-core-v1",
    }
    profile = ModelProfile(
        model_id=model_name,
        revision=revision,
        backend=backend,
        artifact_sha256=artifact_sha256,
        tokenizer_digest=tokenizer_digest,
        chat_template_digest=template_digest,
        context={"n_ctx": n_ctx, "input_budget": 0, "max_new_tokens": 0},
        generation={"temperature": 0.7, "top_p": 0.9, "stop": stop, "thinking": "declared" if thinking_signal else "unknown"},
        adaptation={
            "prompt_family": _infer_prompt_family(model_name),
            "tool_mode": "sidecar_candidate" if "tool_router" in roles else "host_router",
            "structured_output": "grammar_first" if tool_signal else "json_repair",
            "summary_mode": "state_schema_v1",
        },
        roles=tuple(roles),
        resources={"kv_cache": "unknown", "gpu_layers": "auto", "max_batch": "unknown"},
        capabilities=capabilities,
        status="rejected" if errors else "candidate",
        production_eligible=False,
        evidence=evidence,
    )
    return ProbeResult(
        profile=profile,
        errors=tuple(sorted(set(errors))),
        files=files,
        path_digest=_path_digest(artifact),
    )
