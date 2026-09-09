"""Offline B1 and DSW-D1 probes for the newly registered model artifacts.

The default path only reads configuration, tokenizer metadata, GGUF headers and
manifests.  Template execution is an explicit opt-in and runs in the existing
isolated Transformers sidecar; it never loads model weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .gguf import GGUFError, inspect_gguf, verify_gguf
from .gguf_convert import plan_conversion

ROOT = Path(__file__).resolve().parents[2]
TOOL = "small_model_b1_probe"
DSW_TOOL = "dsw_d1_probe"
SCHEMA_VERSION = 1
MAX_METADATA_BYTES = 16 * 1024 * 1024
DEFAULT_B1_MODEL_IDS = ("qwen2.5-0.5b", "qwen3-0.6b", "minicpm4-0.5b")
THINKING_MARKERS = ("<think>", "</think>", "<|think|>")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, "metadata_too_large"
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None, "json_object_required"
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("default", "chat_template", "template"):
            if isinstance(value.get(key), str):
                return value[key]
    return ""


def _path_digest(path: Path) -> str:
    return _digest_bytes(str(path.absolute().resolve(strict=False)).encode("utf-8"))


def _stop_tokens(tokenizer: dict[str, Any], generation: dict[str, Any]) -> dict[str, list[Any]]:
    strings: list[str] = []
    ids: list[int] = []

    def add_string(value: Any) -> None:
        if isinstance(value, str) and value and value not in strings:
            strings.append(value)

    def add_ids(value: Any) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, int) and not isinstance(item, bool) and item not in ids:
                ids.append(item)

    add_string(tokenizer.get("eos_token"))
    add_string(generation.get("eos_token"))
    add_string(generation.get("stop_strings"))
    for value in (generation.get("eos_token_id"), tokenizer.get("eos_token_id")):
        add_ids(value)
    return {"strings": strings, "ids": ids}


def _thinking_report(template: str) -> dict[str, Any]:
    lowered = template.lower()
    switch_declared = "enable_thinking" in lowered
    return {
        "switch_declared": switch_declared,
        "status": "declared" if switch_declared else "not_declared",
        "runtime_status": "pending",
        "markers_in_template": any(marker in lowered for marker in THINKING_MARKERS),
        "evidence": "chat_template_enable_thinking_parameter" if switch_declared else "chat_template_has_no_enable_thinking_parameter",
    }


def _template_report(template: str) -> dict[str, Any]:
    lowered = template.lower()
    thinking = _thinking_report(template)
    return {
        "present": bool(template),
        "sha256": _digest_bytes(template.encode("utf-8")) if template else None,
        "length": len(template),
        "tool_tokens_declared": any(token in lowered for token in ("tool_call", "tools", "function_call")),
        "json_tokens_declared": any(token in lowered for token in ("tojson", "arguments", "parameters")),
        "thinking": thinking,
        "rendering": {
            "default": {"status": "not_run"},
            "thinking_disabled": {"status": "not_run"},
            "first_100_tokens": {"status": "not_run"},
        },
    }


def _manifest_report(root: Path, *, full_hash: bool) -> tuple[dict[str, Any], list[str]]:
    path = root / "model.manifest.json"
    manifest, error = _read_json(path)
    if manifest is None:
        return {"present": False, "valid": False, "reason": error}, ["manifest_missing"]
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return {"present": True, "valid": False, "reason": "files_missing"}, ["manifest_files_missing"]
    errors: list[str] = []
    checked = 0
    hashes_recorded = 0
    content_hashes_checked = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append("manifest_entry_invalid")
            continue
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("manifest_path_unsafe")
            continue
        target = root / relative
        if not target.is_file():
            errors.append("manifest_file_missing:" + relative.as_posix())
            continue
        checked += 1
        expected_size = entry.get("size_bytes", entry.get("size"))
        if isinstance(expected_size, int) and target.stat().st_size != expected_size:
            errors.append("manifest_size_mismatch:" + relative.as_posix())
        expected_hash = str(entry.get("sha256", "")).lower()
        if len(expected_hash) == 64:
            hashes_recorded += 1
            if full_hash or target.suffix.lower() not in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}:
                content_hashes_checked += 1
                if _digest_file(target) != expected_hash:
                    errors.append("manifest_sha256_mismatch:" + relative.as_posix())
        else:
            errors.append("manifest_sha256_missing:" + relative.as_posix())
    return {
        "present": True,
        "valid": not errors and checked == len(entries),
        "source": manifest.get("source"),
        "revision": manifest.get("revision", ""),
        "artifact_sha256": manifest.get("artifact_sha256"),
        "file_count": len(entries),
        "files_checked": checked,
        "hashes_recorded": hashes_recorded,
        "content_hashes_checked": content_hashes_checked,
        "full_hash": full_hash,
    }, errors


def _directory_probe(path: Path, *, full_hash: bool) -> dict[str, Any]:
    errors: list[str] = []
    config, config_error = _read_json(path / "config.json")
    tokenizer, tokenizer_error = _read_json(path / "tokenizer_config.json")
    generation, generation_error = _read_json(path / "generation_config.json")
    for name, value, error in (
        ("config.json", config, config_error),
        ("tokenizer_config.json", tokenizer, tokenizer_error),
        ("generation_config.json", generation, generation_error),
    ):
        if value is None and error not in {"missing", None}:
            errors.append(f"{name}:{error}")
    config = config or {}
    tokenizer = tokenizer or {}
    generation = generation or {}
    template = _text(tokenizer.get("chat_template"))
    if not template:
        sidecar = path / "chat_template.jinja"
        if sidecar.is_file():
            try:
                template = sidecar.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                errors.append("chat_template.jinja:invalid")
    manifest, manifest_errors = _manifest_report(path, full_hash=full_hash)
    errors.extend(manifest_errors)
    architecture = (config.get("architectures") or [config.get("model_type", "unknown")])[0]
    return {
        "format": "safetensors",
        "valid": not errors and bool(config) and bool(tokenizer),
        "model_type": config.get("model_type", "unknown"),
        "architecture": architecture,
        "architectures": config.get("architectures", []),
        "context_length": config.get("max_position_embeddings", config.get("max_seq_len", 0)),
        "template": _template_report(template),
        "stop_tokens": _stop_tokens(tokenizer, generation),
        "manifest": manifest,
        "errors": errors,
    }


def _gguf_probe(path: Path, *, full_hash: bool) -> dict[str, Any]:
    try:
        inspection = inspect_gguf(path)
        verification = verify_gguf(path, full_hash=full_hash)
    except (OSError, GGUFError) as exc:
        return {"format": "gguf", "valid": False, "errors": [exc.__class__.__name__]}
    metadata = inspection.get("metadata", {})
    template = _text(metadata.get("tokenizer.chat_template"))
    architecture = inspection.get("derived", {}).get("architecture", "unknown")
    stop_ids = []
    for key in (f"{architecture}.eos_token_id", "tokenizer.ggml.eos_token_id"):
        value = metadata.get(key)
        if isinstance(value, int):
            stop_ids.append(value)
    return {
        "format": "gguf",
        "valid": bool(inspection.get("valid")) and bool(verification.get("valid")),
        "architecture": architecture,
        "name": inspection.get("derived", {}).get("name"),
        "context_length": inspection.get("derived", {}).get("context_length"),
        "tensor_count": inspection.get("tensor_count", 0),
        "metadata_count": inspection.get("metadata_count", 0),
        "tensor_types": inspection.get("derived", {}).get("tensor_types", {}),
        "template": _template_report(template),
        "stop_tokens": {"strings": [], "ids": stop_ids},
        "verification": {
            "structure_valid": bool(verification.get("structure_valid")),
            "sha256": verification.get("sha256"),
            "sha256_expected": verification.get("sha256_expected"),
            "sha256_checked": bool(verification.get("sha256_checked")),
            "sidecar_present": bool(verification.get("sidecar")),
        },
        "errors": list(verification.get("errors", [])),
    }


def probe_artifact(path: str | Path, *, model_id: str | None = None, full_hash: bool = False) -> dict[str, Any]:
    target = Path(path).expanduser().absolute().resolve(strict=False)
    inferred_id = model_id or target.stem if target.is_file() else model_id or target.name
    base = {
        "model_id": inferred_id,
        "path_digest": _path_digest(target),
        "weights_loaded": False,
        "network_used": False,
        "read_only": True,
    }
    if not target.exists():
        return {**base, "format": "unknown", "valid": False, "errors": ["asset_missing"]}
    result = _gguf_probe(target, full_hash=full_hash) if target.is_file() else _directory_probe(target, full_hash=full_hash)
    return {**base, **result}


def _core_config(model_id: str) -> tuple[Any, Path, Path]:
    sys.path.insert(0, str(ROOT / "src"))
    import model_config as mc

    config = mc.get_builtin_model(model_id)
    if config is None:
        raise ValueError(f"unknown model_id: {model_id}")
    return config, Path(mc.resolve_model_path(config.model_path)), Path(mc.resolve_model_path(config.gguf_path))


def _converter_architecture_support() -> tuple[str, set[str]]:
    from .gguf_convert import _find_converter, _supported_architectures

    converter, status = _find_converter(None)
    return status, _supported_architectures(converter) or set()


def _architecture_probe(config: Any, gguf: dict[str, Any], source: dict[str, Any] | None = None) -> dict[str, Any]:
    expected_architectures = {
        "qwen2.5-0.5b": {"qwen2"},
        "qwen3-0.6b": {"qwen3"},
        "minicpm4-0.5b": {"minicpm"},
        "distilqwen25-ds3-0324-7b": {"qwen2"},
    }.get(config.model_id, set())
    expected_classes = {
        "qwen2.5-0.5b": "Qwen2ForCausalLM",
        "qwen3-0.6b": "Qwen3ForCausalLM",
        "minicpm4-0.5b": "MiniCPMForCausalLM",
        "distilqwen25-ds3-0324-7b": "Qwen2ForCausalLM",
    }
    converter_status, registered = _converter_architecture_support()
    actual = gguf.get("architecture")
    architecture_ok = actual in expected_architectures
    source_architectures = set(getattr(config, "model_type", "") and [expected_classes.get(config.model_id, "")] or [])
    if config.model_id == "minicpm4-0.5b":
        source_architectures = {"MiniCPMForCausalLM"}
    registered_name = bool(source_architectures & registered)
    return {
        "source_model_type": (source or {}).get("model_type", "unknown"),
        "source_architectures": list((source or {}).get("architectures", [])),
        "expected_gguf_architectures": sorted(expected_architectures),
        "actual_gguf_architecture": actual,
        "gguf_structure_valid": bool(gguf.get("valid")),
        "converter_status": converter_status,
        "converter_architecture_registered": registered_name,
        "runtime_load_status": "not_run",
        "status": "metadata_pass" if architecture_ok and gguf.get("valid") and registered_name else "rejected",
        "evidence": "GGUF header + converter registration; llama.cpp runtime load not executed",
    }


def _sidecar_python() -> Path | None:
    override = os.environ.get("QLH_TEMPLATE_PROBE_PYTHON", "").strip() or os.environ.get("QLH_QWEN3_SIDECAR_PYTHON", "").strip()
    if override:
        candidate = Path(override).expanduser().absolute().resolve(strict=False)
    elif os.name == "nt":
        candidate = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv-qwen3-sidecar" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _execute_template(path: Path, *, timeout_seconds: float, trust_remote_code: bool) -> dict[str, Any]:
    if path.is_file():
        return {"status": "skipped", "reason": "gguf_template_is_static_metadata_only"}
    python = _sidecar_python()
    if python is None:
        return {"status": "unavailable", "reason": "isolated_template_runtime_missing"}
    request = {
        "schema_version": SCHEMA_VERSION,
        "operation": "template_probe",
        "model_path": str(path),
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
        "trust_remote_code": trust_remote_code,
    }
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "NO_PROXY": "*"})
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    try:
        completed = subprocess.run(
            [str(python), str(Path(__file__).with_name("small_model_template_probe_worker.py"))],
            input=json.dumps(request, ensure_ascii=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "template_probe_timeout"}
    except OSError:
        return {"status": "failed", "reason": "template_probe_start_failed"}
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("operation") == "template_probe":
            return value
    return {"status": "failed", "reason": "template_probe_invalid_output"}


def _attach_template_runtime(report: dict[str, Any], runtime: dict[str, Any]) -> None:
    template = report.get("template")
    if not isinstance(template, dict):
        return
    template["runtime_probe"] = runtime
    rendered = runtime.get("rendering") if isinstance(runtime, dict) else None
    if isinstance(rendered, dict):
        template["rendering"].update(rendered)
        if "thinking_disabled" in rendered:
            template["rendering"]["first_100_tokens"] = dict(rendered["thinking_disabled"].get("first_100_tokens", {}))
        elif "default" in rendered:
            template["rendering"]["first_100_tokens"] = dict(rendered["default"].get("first_100_tokens", {}))
    thinking = runtime.get("thinking") if isinstance(runtime, dict) else None
    if isinstance(thinking, dict):
        template["thinking"].update(thinking)


def run_b1_probe(
    *,
    model_ids: Iterable[str] | None = None,
    paths: Iterable[str | Path] | None = None,
    execute_template: bool = False,
    run_gguf_smoke: bool = False,
    full_hash: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    selected_ids = list(model_ids or DEFAULT_B1_MODEL_IDS)
    if paths:
        selected = [(Path(item), None) for item in paths]
    else:
        selected = [(None, item) for item in selected_ids]
    models: list[dict[str, Any]] = []
    for path, model_id in selected:
        if model_id is None:
            source_path = path.expanduser().absolute().resolve(strict=False)
            gguf_path = source_path if source_path.suffix.lower() == ".gguf" else None
            config = None
            inferred = source_path.stem
        else:
            config, source_path, gguf_path = _core_config(model_id)
            inferred = model_id
        source = probe_artifact(source_path, model_id=inferred, full_hash=full_hash) if source_path else None
        gguf = probe_artifact(gguf_path, model_id=inferred, full_hash=full_hash) if gguf_path else None
        if execute_template and source_path and source_path.is_dir():
            _attach_template_runtime(source, _execute_template(source_path, timeout_seconds=timeout_seconds, trust_remote_code=(model_id == "minicpm4-0.5b")))
        compatibility = _architecture_probe(config, gguf, source) if config is not None and gguf is not None else {"status": "not_applicable"}
        models.append({"model_id": inferred, "safetensors": source, "gguf": gguf, "architecture": compatibility})
    smoke = None
    if run_gguf_smoke:
        from .llm_smoke_matrix import run_smoke_matrix

        smoke = run_smoke_matrix(
            model_ids=["minicpm4-0.5b"],
            formats=["gguf"],
            max_models=1,
            max_new_tokens=16,
            timeout_seconds=timeout_seconds,
            allow_cpu=True,
            require_complete=True,
        )
    static_passed = all(
        item.get("safetensors", {}).get("valid") and item.get("gguf", {}).get("valid") and item.get("architecture", {}).get("status") == "metadata_pass"
        for item in models
    )
    gate_passed = static_passed and (not run_gguf_smoke or bool(smoke and smoke.get("summary", {}).get("gate_passed")))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "matrix",
        "valid": static_passed,
        "gate_passed": gate_passed,
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "template_execution_requested": execute_template,
        "gguf_smoke_requested": run_gguf_smoke,
        "models": models,
        "gguf_smoke": smoke,
        "summary": {"models_total": len(models), "models_passed": sum(item["safetensors"].get("valid") and item["gguf"].get("valid") for item in models), "static_gate_passed": static_passed},
    }


def run_dsw_d1(
    *,
    model_id: str = "distilqwen25-ds3-0324-7b",
    outtype: str = "Q4_K_M",
    execute_template: bool = False,
    full_hash: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    config, source_path, gguf_path = _core_config(model_id)
    source = probe_artifact(source_path, model_id=model_id, full_hash=full_hash)
    gguf = probe_artifact(gguf_path, model_id=model_id, full_hash=full_hash)
    if execute_template:
        _attach_template_runtime(source, _execute_template(source_path, timeout_seconds=timeout_seconds, trust_remote_code=False))
    plan = plan_conversion(
        model_id=model_id,
        outtype=outtype,
        target=ROOT / f".qlh-dsw-d1-{model_id}-{outtype.lower()}.gguf",
    )
    architecture = _architecture_probe(config, gguf, source)
    valid = bool(source.get("valid")) and bool(gguf.get("valid")) and architecture.get("status") == "metadata_pass"
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": DSW_TOOL,
        "operation": "preflight",
        "model_id": model_id,
        "model_name": config.name,
        "valid": valid,
        "gate_passed": valid and bool(plan.get("valid")),
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "safetensors": source,
        "gguf": gguf,
        "architecture": architecture,
        "conversion": {
            "requested_outtype": outtype,
            "plan_valid": bool(plan.get("valid")),
            "toolchain": plan.get("toolchain", {}),
            "space": plan.get("space", {}),
            "errors": plan.get("errors", []),
        },
        "evidence": ["local_manifest_and_metadata", "existing_gguf_structure_and_sha_gate", "conversion_plan_read_only"],
    }


__all__ = ["DEFAULT_B1_MODEL_IDS", "probe_artifact", "run_b1_probe", "run_dsw_d1"]
