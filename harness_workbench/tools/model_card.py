"""Generate an auditable, offline model card from local model metadata.

The generator reads manifest health records, sidecar declarations and the GGUF
header only. It never loads tensor data, starts a runtime, or accesses a
network. The resulting card is an inventory and evidence boundary, not a
quality or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..model_profiles import ModelProfile, builtin_profiles
from .manifest_health import MANIFEST_HEALTH_SCHEMA, ManifestHealthReport, scan_manifest_health


MODEL_CARD_SCHEMA = "qlh.harness.model_card.v1"
MODEL_CARD_INPUT_SCHEMA = "qlh.model_card.v1"
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_GGUF_MAGIC = b"GGUF"
_GGUF_HEADER_LIMIT = 64 * 1024 * 1024
_GGUF_MAX_ITEMS = 10000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _path_free(value: Any) -> bool:
    if isinstance(value, str):
        return not _ABSOLUTE_PATH.search(value)
    if isinstance(value, Mapping):
        return all(_path_free(key) and _path_free(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return all(_path_free(child) for child in value)
    return True


def _relative(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    if _ABSOLUTE_PATH.search(text) or "://" in text:
        raise ValueError("model card paths must be relative")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("model card paths must be relative")
    return candidate.as_posix()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class ModelCardArtifact:
    path: str
    kind: str
    size_bytes: int
    sha256_status: str
    ignored: bool = False
    expected_sha256: str | None = None
    sidecar_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative(self.path))
        if self.sidecar_path is not None:
            object.__setattr__(self, "sidecar_path", _relative(self.sidecar_path))
        if self.size_bytes < 0 or not self.kind or not self.sha256_status:
            raise ValueError("invalid model card artifact")
        if not _path_free(self.metadata):
            raise ValueError("artifact metadata contains an unsafe path")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256_status": self.sha256_status,
            "ignored": self.ignored,
            "expected_sha256": self.expected_sha256,
            "sidecar_path": self.sidecar_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ModelCardMetadata:
    model_id: str
    revision: str | None = None
    backend: str | None = None
    profile_status: str | None = None
    production_eligible: bool | None = None
    parameter_hint: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    tokenizer: str | None = None
    chat_template: str | None = None
    capabilities: tuple[str, ...] = ()
    source: str | None = None
    notes: tuple[str, ...] = ()
    gguf_header: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model card model_id is required")
        if self.context_length is not None and self.context_length < 0:
            raise ValueError("context_length must be non-negative")
        if not _path_free(self.gguf_header):
            raise ValueError("GGUF metadata contains an unsafe path")
        object.__setattr__(self, "capabilities", tuple(dict.fromkeys(str(item) for item in self.capabilities if str(item).strip())))
        object.__setattr__(self, "notes", tuple(dict.fromkeys(str(item) for item in self.notes if str(item).strip())))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "backend": self.backend,
            "profile_status": self.profile_status,
            "production_eligible": self.production_eligible,
            "parameter_hint": self.parameter_hint,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "tokenizer": self.tokenizer,
            "chat_template": self.chat_template,
            "capabilities": list(self.capabilities),
            "source": self.source,
            "notes": list(self.notes),
            "gguf_header": dict(self.gguf_header),
        }


@dataclass(frozen=True, slots=True)
class ModelCardReport:
    model_id: str
    title: str
    status: str
    artifacts: tuple[ModelCardArtifact, ...]
    metadata: ModelCardMetadata
    limitations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    manifest_health: Mapping[str, Any]
    profile_digest: str | None = None
    schema: str = MODEL_CARD_SCHEMA
    runner_kind: str = "metadata"
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != MODEL_CARD_SCHEMA or not self.model_id or not self.title:
            raise ValueError("model card identity is invalid")
        if self.status not in {"complete", "incomplete", "invalid"}:
            raise ValueError("invalid model card status")
        if self.runner_kind != "metadata" or self.network_used or self.weights_loaded:
            raise ValueError("model card must remain offline and model-free")
        if len({artifact.path for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("model card artifact paths must be unique")
        if any(_ABSOLUTE_PATH.search(ref) or "://" in ref for ref in self.evidence_refs):
            raise ValueError("model card evidence refs must be local relative paths")
        if not _path_free(self.manifest_health):
            raise ValueError("manifest summary contains an unsafe path")

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "artifacts_present": bool(self.artifacts),
            "manifest_summary_present": isinstance(self.manifest_health, Mapping),
            "gguf_headers_readable": all(
                artifact.kind.lower() != "gguf"
                or artifact.metadata.get("gguf_header", {}).get("read_status") == "ok"
                for artifact in self.artifacts
            ),
            "evidence_refs_relative": all(not _ABSOLUTE_PATH.search(ref) and "://" not in ref for ref in self.evidence_refs),
            "offline_boundary": not self.network_used and not self.weights_loaded,
            "limitations_explicit": bool(self.limitations) and any("performance" in item.lower() for item in self.limitations),
        }

    @property
    def valid(self) -> bool:
        return self.status != "invalid" and all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "model_id": self.model_id,
            "title": self.title,
            "status": self.status,
            "runner_kind": self.runner_kind,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "metadata": self.metadata.as_dict(),
            "limitations": list(self.limitations),
            "evidence_refs": list(self.evidence_refs),
            "manifest_health": dict(self.manifest_health),
            "profile_digest": self.profile_digest,
            "checks": self.checks,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        metadata = self.metadata
        lines = [
            f"# {self.title}",
            "",
            f"- Model ID: `{self.model_id}`; status: `{self.status}`; card valid: `{str(self.valid).lower()}`",
            f"- Runner: `metadata`; weights loaded: `false`; network used: `false`; report digest: `{self.digest}`",
            "",
            "## Metadata",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| revision | `{metadata.revision or 'unknown'}` |",
            f"| backend | `{metadata.backend or 'unknown'}` |",
            f"| profile status | `{metadata.profile_status or 'unknown'}` |",
            f"| production eligible | `{str(metadata.production_eligible).lower() if metadata.production_eligible is not None else 'unknown'}` |",
            f"| parameters | `{metadata.parameter_hint or 'unknown'}` |",
            f"| quantization | `{metadata.quantization or 'unknown'}` |",
            f"| context length | `{metadata.context_length if metadata.context_length is not None else 'unknown'}` |",
            f"| tokenizer | `{metadata.tokenizer or 'unknown'}` |",
            f"| chat template | `{metadata.chat_template or 'unknown'}` |",
            f"| capabilities | `{', '.join(metadata.capabilities) or 'unknown'}` |",
            "",
            "## Artifacts",
            "",
            "| path | kind | size bytes | SHA status | ignored |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for artifact in self.artifacts:
            lines.append(f"| `{artifact.path}` | `{artifact.kind}` | {artifact.size_bytes} | `{artifact.sha256_status}` | {str(artifact.ignored).lower()} |")
        lines.extend(("", "## Manifest evidence", "", f"- Scanned files: `{self.manifest_health.get('scanned_file_count', 0)}`; manifests: `{self.manifest_health.get('manifest_count', 0)}`; manifest valid: `{str(self.manifest_health.get('valid', False)).lower()}`"))
        if self.manifest_health.get("warnings"):
            lines.append(f"- Warnings: `{'; '.join(self.manifest_health['warnings'])}`")
        if self.manifest_health.get("errors"):
            lines.append(f"- Errors: `{'; '.join(self.manifest_health['errors'])}`")
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {item}" for item in self.limitations)
        lines.extend(("", "## Evidence refs", ""))
        lines.extend(f"- `{item}`" for item in self.evidence_refs)
        lines.extend(("", "## Checks", ""))
        lines.extend(f"- `{name}`: **{'passed' if passed else 'failed'}**" for name, passed in self.checks.items())
        lines.append("")
        return "\n".join(lines)


def _read_u(data: bytes, offset: int, fmt: str) -> tuple[Any, int]:
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated GGUF header")
    return struct.unpack_from(fmt, data, offset)[0], offset + size


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_u(data, offset, "<Q")
    if length > len(data) - offset or length > _GGUF_HEADER_LIMIT:
        raise ValueError("invalid GGUF string length")
    raw = data[offset : offset + length]
    return raw.decode("utf-8", errors="replace"), offset + length


def _read_gguf_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if value_type == 8:
        return _read_string(data, offset)
    if value_type in formats:
        return _read_u(data, offset, formats[value_type])
    if value_type == 9:
        item_type, offset = _read_u(data, offset, "<I")
        count, offset = _read_u(data, offset, "<Q")
        if count > _GGUF_MAX_ITEMS:
            raise ValueError("GGUF array too large")
        values = []
        for _ in range(count):
            item, offset = _read_gguf_value(data, offset, item_type)
            values.append(item)
        return values, offset
    raise ValueError("unsupported GGUF metadata type")


def _skip_gguf_value(data: bytes, offset: int, value_type: int) -> int:
    """Advance over an unselected metadata value without materializing it."""
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if value_type == 8:
        _, offset = _read_string(data, offset)
        return offset
    if value_type in formats:
        _, offset = _read_u(data, offset, formats[value_type])
        return offset
    if value_type == 9:
        item_type, offset = _read_u(data, offset, "<I")
        count, offset = _read_u(data, offset, "<Q")
        if count > 1_000_000:
            raise ValueError("GGUF array too large")
        for _ in range(count):
            offset = _skip_gguf_value(data, offset, item_type)
        return offset
    raise ValueError("unsupported GGUF metadata type")


def _read_gguf_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = handle.read(_GGUF_HEADER_LIMIT)
    except OSError as exc:
        return {"read_status": "unreadable", "error": type(exc).__name__}
    if len(data) < 24 or data[:4] != _GGUF_MAGIC:
        return {"read_status": "not_gguf_or_truncated"}
    try:
        version, offset = _read_u(data, 4, "<I")
        if version not in {2, 3}:
            return {"read_status": "unsupported_version", "version": version}
        tensor_count, offset = _read_u(data, offset, "<Q")
        kv_count, offset = _read_u(data, offset, "<Q")
        if kv_count > _GGUF_MAX_ITEMS:
            return {"read_status": "metadata_limit_exceeded", "version": version, "metadata_kv_count": kv_count}
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key, offset = _read_string(data, offset)
            value_type, offset = _read_u(data, offset, "<I")
            if key in {
                "general.architecture", "general.name", "general.file_type", "general.quantization_version",
                "general.parameter_count", "llama.context_length", "qwen.context_length", "qwen2.context_length",
                "qwen3.context_length", "tokenizer.ggml.model", "tokenizer.chat_template",
            }:
                value, offset = _read_gguf_value(data, offset, value_type)
                metadata[key] = value
            else:
                offset = _skip_gguf_value(data, offset, value_type)
        return {
            "read_status": "ok",
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": kv_count,
            "metadata": metadata,
        }
    except (ValueError, UnicodeError, struct.error):
        return {"read_status": "invalid_or_truncated"}


def _infer_model_id(root_label: str, artifacts: Sequence[Mapping[str, Any]]) -> str:
    names = [str(item.get("path", "")) for item in artifacts]
    for name in names:
        if "qwen-1_8b" in name.lower() or "qwen_1_8b" in name.lower():
            return "QW1.8B"
    if root_label and root_label.lower() not in {"models", "."}:
        return root_label
    return "local-model-inventory"


def _profile_metadata(profile: ModelProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    capabilities = tuple(f"{name}:{state.status}" for name, state in profile.capabilities.items() if state.status != "unknown")
    context_length = profile.context.get("n_ctx") if _finite(profile.context.get("n_ctx")) else None
    return {
        "model_id": profile.model_id,
        "revision": profile.revision,
        "backend": profile.backend,
        "profile_status": profile.status,
        "production_eligible": profile.production_eligible,
        "context_length": int(context_length) if context_length is not None else None,
        "capabilities": capabilities,
        "chat_template": profile.adaptation.get("prompt_family"),
        "profile_digest": profile.digest,
        "status": profile.status,
    }


def _artifact_from_health(item: Mapping[str, Any], root: Path) -> ModelCardArtifact:
    metadata: dict[str, Any] = {}
    relative_path = str(item.get("path", ""))
    normalized = _relative(relative_path)
    candidate = root / Path(*PurePosixPath(normalized).parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError("model card artifact escapes the model root") from exc
    is_junction = getattr(candidate, "is_junction", lambda: False)
    if candidate.is_symlink() or is_junction():
        raise ValueError("model card artifacts cannot be reparse points")
    if str(item.get("kind", "")).lower() == "gguf":
        metadata["gguf_header"] = _read_gguf_header(candidate)
    return ModelCardArtifact(
        path=relative_path,
        kind=str(item.get("kind", "file")),
        size_bytes=int(item.get("size_bytes", 0) or 0),
        sha256_status=str(item.get("sha256_status", "unknown")),
        ignored=bool(item.get("ignored", False)),
        expected_sha256=item.get("expected_sha256") if isinstance(item.get("expected_sha256"), str) else None,
        sidecar_path=item.get("sidecar_path") if isinstance(item.get("sidecar_path"), str) else None,
        metadata=metadata,
    )


def build_model_card_from_health(
    root: str | Path,
    *,
    health_report: ManifestHealthReport | Mapping[str, Any] | None = None,
    profile: ModelProfile | None = None,
    title: str | None = None,
    source_ref: str | None = None,
) -> ModelCardReport:
    target = Path(root).expanduser().absolute()
    health = health_report or scan_manifest_health(target)
    health_value = health.as_dict() if isinstance(health, ManifestHealthReport) else dict(health)
    if health_value.get("schema") != MANIFEST_HEALTH_SCHEMA:
        raise ValueError("unsupported manifest health schema")
    if not _path_free(health_value):
        raise ValueError("manifest health input contains an unsafe path")
    raw_artifacts = health_value.get("artifacts", ())
    if not isinstance(raw_artifacts, list):
        raise ValueError("manifest health artifacts must be an array")
    artifacts = tuple(_artifact_from_health(item, target) for item in raw_artifacts if isinstance(item, Mapping))
    profile_value = _profile_metadata(profile)
    model_id = str(profile_value.get("model_id") or _infer_model_id(str(health_value.get("root_label", target.name)), raw_artifacts))
    gguf_headers = [artifact.metadata.get("gguf_header", {}) for artifact in artifacts if artifact.kind.lower() == "gguf"]
    gguf_header_errors = [header for header in gguf_headers if header.get("read_status") != "ok"]
    first_header = gguf_headers[0] if gguf_headers else {}
    header_values = first_header.get("metadata", {}) if isinstance(first_header, Mapping) else {}
    context_value = header_values.get("llama.context_length", header_values.get("qwen.context_length", header_values.get("qwen2.context_length", header_values.get("qwen3.context_length"))))
    context = int(context_value) if _finite(context_value) else profile_value.get("context_length")
    quantization = None
    for artifact in artifacts:
        match = re.search(r"(?:q|iq)[0-9][a-z0-9_]*", artifact.path.lower())
        if match:
            quantization = match.group(0).upper()
            break
    file_type = header_values.get("general.file_type")
    if quantization is None and file_type is not None:
        quantization = f"gguf_file_type_{file_type}"
    parameter_hint = header_values.get("general.parameter_count")
    if parameter_hint is None:
        for artifact in artifacts:
            match = re.search(r"(?<![0-9])(\d+(?:[._]\d+)?)b(?![a-z])", artifact.path.lower())
            if match:
                parameter_hint = match.group(1).replace("_", ".") + "B (filename hint)"
                break
    manifest_summary = {
        "root_label": health_value.get("root_label", target.name),
        "valid": bool(health_value.get("valid", False)),
        "scanned_file_count": int(health_value.get("scanned_file_count", 0) or 0),
        "ignored_file_count": int(health_value.get("ignored_file_count", 0) or 0),
        "artifact_count": len(artifacts),
        "manifest_count": len(health_value.get("manifests", ())) if isinstance(health_value.get("manifests", ()), list) else 0,
        "warnings": [str(item) for item in health_value.get("warnings", ())],
        "errors": [str(item) for item in health_value.get("errors", ())],
        "report_digest": health_value.get("report_digest"),
    }
    notes = ["Metadata-only inventory; no tensor contents were read."]
    if header_values.get("general.architecture"):
        notes.append(f"GGUF architecture declared as {header_values['general.architecture']}.")
    if manifest_summary["errors"]:
        notes.append("One or more manifest references are incomplete or invalid.")
    if gguf_header_errors:
        notes.append("One or more GGUF headers are unreadable, truncated or unsupported.")
    if profile_value.get("context_length") is not None and context != profile_value["context_length"]:
        notes.append(f"Profile runtime context is configured as {profile_value['context_length']}; GGUF declares {context}.")
    metadata = ModelCardMetadata(
        model_id=model_id,
        revision=profile_value.get("revision"),
        backend=profile_value.get("backend"),
        profile_status=profile_value.get("status"),
        production_eligible=profile_value.get("production_eligible"),
        parameter_hint=str(parameter_hint) if parameter_hint is not None else None,
        quantization=quantization,
        context_length=context,
        tokenizer=str(header_values.get("tokenizer.ggml.model")) if header_values.get("tokenizer.ggml.model") is not None else None,
        chat_template="GGUF declared" if header_values.get("tokenizer.chat_template") else profile_value.get("chat_template"),
        capabilities=tuple(profile_value.get("capabilities", ())),
        source=source_ref or "models/",
        notes=tuple(notes),
        gguf_header=first_header if isinstance(first_header, Mapping) else {},
    )
    limitations = [
        "This card contains identity and asset metadata only; it makes no quality or performance claim.",
        "weights_loaded=false and network_used=false; runtime behavior, latency and memory usage remain unmeasured.",
    ]
    if profile is None:
        limitations.append("No model profile was supplied; capabilities and backend remain unknown unless declared in GGUF metadata.")
    if manifest_summary["errors"]:
        limitations.append("Manifest health is incomplete; missing or invalid asset references must be repaired before release.")
    if gguf_header_errors:
        limitations.append("GGUF metadata could not be read safely; the affected artifact identity is not release-ready.")
    root_label = str(manifest_summary["root_label"])
    evidence = [f"{root_label}/", ".gitignore"]
    if source_ref:
        evidence.append(_relative(source_ref))
    evidence.extend(f"{root_label}/{artifact.path}" for artifact in artifacts)
    status = "invalid" if not artifacts else ("incomplete" if manifest_summary["errors"] or gguf_header_errors else "complete")
    return ModelCardReport(
        model_id=model_id,
        title=title or f"{model_id} model card",
        status=status,
        artifacts=artifacts,
        metadata=metadata,
        limitations=tuple(dict.fromkeys(limitations)),
        evidence_refs=tuple(dict.fromkeys(evidence)),
        manifest_health=manifest_summary,
        profile_digest=profile_value.get("profile_digest"),
    )


def _load_profile(path: str | Path) -> ModelProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelProfile.from_dict(payload)


def build_model_card_from_json(
    path: str | Path,
    *,
    profile_file: str | Path | None = None,
    profile: ModelProfile | None = None,
    title: str | None = None,
    root: str | Path | None = None,
) -> ModelCardReport:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("model card input must be a JSON object")
    if profile_file is not None and profile is not None:
        raise ValueError("provide profile_file or profile, not both")
    selected_profile = _load_profile(profile_file) if profile_file else profile
    if payload.get("schema") == "qlh.harness.manifest_health.v1":
        health_root = Path(root) if root is not None else Path(str(payload.get("root_label", "models")))
        return build_model_card_from_health(health_root, health_report=payload, profile=selected_profile, title=title, source_ref=source.name)
    if payload.get("schema") != MODEL_CARD_INPUT_SCHEMA:
        raise ValueError("unsupported model card input schema")
    root_value = payload.get("root", "models")
    if _ABSOLUTE_PATH.search(str(root_value)):
        raise ValueError("model card input root must be relative")
    if root is not None:
        root_value = root
    return build_model_card_from_health(root_value, profile=selected_profile, title=title, source_ref=source.name)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an offline model card from local manifest and GGUF metadata")
    parser.add_argument("--root", default="models", metavar="PATH", help="model asset root")
    parser.add_argument("--health-json", metavar="PATH", help="use an existing manifest-health JSON report")
    parser.add_argument("--profile", metavar="PATH", help="model profile JSON")
    parser.add_argument("--model-id", metavar="ID", help="use a built-in candidate profile by model id")
    parser.add_argument("--title", default="", help="card title")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile = _load_profile(args.profile) if args.profile else None
        if args.model_id:
            matches = [item for item in builtin_profiles() if item.model_id.lower() == args.model_id.lower() or args.model_id.lower() in {alias.lower() for alias in item.aliases}]
            if not matches:
                raise ValueError(f"unknown built-in model id: {args.model_id}")
            if profile is not None:
                raise ValueError("provide --profile or --model-id, not both")
            profile = matches[0]
        if args.health_json:
            report = build_model_card_from_json(args.health_json, profile=profile, title=args.title or None, root=args.root)
        else:
            report = build_model_card_from_health(args.root, profile=profile, title=args.title or None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    outputs = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        outputs += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        outputs += 1
    if not outputs:
        print(markdown_text, end="")
    return 0 if report.valid and all(report.checks.values()) else 1


__all__ = [
    "MODEL_CARD_INPUT_SCHEMA",
    "MODEL_CARD_SCHEMA",
    "ModelCardArtifact",
    "ModelCardMetadata",
    "ModelCardReport",
    "build_model_card_from_health",
    "build_model_card_from_json",
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
