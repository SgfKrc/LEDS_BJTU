"""GGUF conversion preflight and explicitly confirmed execution transaction."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .gguf import GGUFError, inspect_gguf, verify_gguf
from .llama_quantize_toolchain import resolve_quantizer

SCHEMA_VERSION = 1
TOOL = "gguf_convert"
MIB = 1024 * 1024
SUPPORTED_CONVERTER_OUTTYPES = {"f32", "f16", "bf16", "q8_0", "auto"}
QUANTIZER_TYPES = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "IQ1_S", "IQ1_M"}
QUANTIZER_REQUIRED = QUANTIZER_TYPES - {"Q8_0"}
ESTIMATE_FACTORS = {
    "f32": 2.0,
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 0.55,
    "auto": 1.0,
    "Q2_K": 0.20,
    "Q3_K_S": 0.25,
    "Q3_K_M": 0.28,
    "Q4_0": 0.30,
    "Q4_1": 0.34,
    "Q4_K_S": 0.31,
    "Q4_K_M": 0.32,
    "Q5_0": 0.39,
    "Q5_1": 0.43,
    "Q5_K_S": 0.40,
    "Q5_K_M": 0.43,
    "Q6_K": 0.49,
    "Q8_0": 0.55,
}


class GGUFConvertError(ValueError):
    """Raised for an invalid conversion request."""


def _path_label(path: Path) -> str:
    return path.name or path.anchor or "<path>"


def _redact(message: str) -> str:
    value = re.sub(r"(?i)[A-Z]:[\\/][^\r\n\t\"']+", "<path>", str(message))
    return re.sub(r"(?<![A-Za-z0-9.])/(?:[^/\s\"']+/)+[^\s\"']+", "<path>", value)


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": _redact(message)[:2048]}


def _existing_parent(path: Path) -> Path | None:
    candidate = path.absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.absolute().resolve(strict=False).relative_to(root.absolute().resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_command(*parts: str) -> list[str]:
    """Return an argv plan with paths replaced by stable placeholders."""
    result: list[str] = []
    for part in parts:
        result.append(str(part))
    return result


def _registered_source(model_id: str | None) -> tuple[Path, str, dict[str, Any]]:
    if not model_id:
        raise GGUFConvertError("provide exactly one of --model-id or --source")
    from src.model_config import get_builtin_model, resolve_model_path

    model = get_builtin_model(model_id)
    if model is None:
        raise GGUFConvertError(f"unknown model_id: {model_id}")
    source = Path(resolve_model_path(model.model_path)).absolute()
    return source, model_id, {"name": model.name, "huggingface_id": model.huggingface_id}


def _source_summary(source: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if not source.is_absolute():
        errors.append(_error("source_not_absolute", "source must be an absolute path"))
        return {"label": _path_label(source), "exists": False}, errors
    if not source.is_dir():
        errors.append(_error("source_missing", "source directory does not exist"))
        return {"label": _path_label(source), "exists": False}, errors
    config_path = source / "config.json"
    tokenizer_path = source / "tokenizer_config.json"
    if not config_path.is_file():
        errors.append(_error("missing_config", "source is missing config.json"))
    if not tokenizer_path.is_file():
        errors.append(_error("missing_tokenizer_config", "source is missing tokenizer_config.json"))
    weight_files = sorted(item for item in source.iterdir() if item.is_file() and item.suffix.lower() in {".safetensors", ".bin"})
    if not weight_files:
        errors.append(_error("missing_weights", "source has no .safetensors or .bin weights"))
    weight_bytes = sum(item.stat().st_size for item in weight_files)
    files = [item for item in source.rglob("*") if item.is_file() and not item.is_symlink()]
    total_bytes = sum(item.stat().st_size for item in files)
    architectures: list[str] = []
    model_type: str | None = None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw_arch = config.get("architectures", [])
            if isinstance(raw_arch, list):
                architectures = [str(item) for item in raw_arch if isinstance(item, str)]
            model_type = str(config.get("model_type")) if config.get("model_type") is not None else None
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(_error("invalid_config", f"cannot parse config.json: {exc}"))
    index_valid: bool | None = None
    index_references = 0
    index_paths = [
        path
        for path in (source / "model.safetensors.index.json", source / "pytorch_model.bin.index.json")
        if path.is_file()
    ]
    if index_paths:
        index_valid = True
        referenced: set[str] = set()
        for index_path in index_paths:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("weight_map must be a non-empty object")
                for value in weight_map.values():
                    if not isinstance(value, str) or Path(value).name != value or Path(value).suffix.lower() not in {".safetensors", ".bin"}:
                        raise ValueError("weight_map contains an unsafe weight filename")
                    referenced.add(value)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                index_valid = False
                errors.append(_error("invalid_weight_index", f"cannot validate weight index: {exc}"))
        index_references = len(referenced)
        missing = [name for name in referenced if not (source / name).is_file()]
        if missing:
            index_valid = False
            errors.append(_error("missing_weight_shard", f"weight index references {len(missing)} missing shard(s)"))
    return {
        "label": _path_label(source),
        "exists": True,
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_bytes": weight_bytes,
        "total_bytes": total_bytes,
        "architectures": architectures,
        "model_type": model_type,
        "has_index": (source / "model.safetensors.index.json").is_file(),
        "weight_index_valid": index_valid,
        "weight_index_file_count": len(index_paths),
        "weight_index_referenced_shards": index_references,
    }, errors


def _find_converter(explicit: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        return (explicit.absolute(), "available") if explicit.is_file() else (None, "missing")
    candidates = [Path(__file__).resolve().parents[2] / "android" / "app" / "src" / "main" / "cpp" / "llama.cpp" / "convert_hf_to_gguf.py"]
    for name in ("convert-hf-to-gguf.py", "convert_hf_to_gguf.py"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute(), "available"
    return None, "missing"


def _find_quantizer(explicit: Path | None) -> tuple[Path | None, dict[str, Any]]:
    return resolve_quantizer(explicit)


def _supported_architectures(converter: Path | None) -> set[str] | None:
    if converter is None:
        return None
    conversion_dir = converter.parent / "conversion"
    if not conversion_dir.is_dir():
        return None
    names: set[str] = set()
    for file in conversion_dir.glob("*.py"):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "register":
                    continue
                names.update(arg.value for arg in decorator.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return names


def _dependency_probe(converter: Path | None) -> dict[str, Any]:
    if converter is None:
        return {"status": "unavailable", "checked": [], "missing": []}
    requirements = converter.parent / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if not requirements.is_file():
        return {"status": "not_probed", "checked": [], "missing": []}
    checked = ["numpy", "safetensors", "torch"]
    missing = [name for name in checked if importlib.util.find_spec(name) is None]
    local_gguf = (converter.parent / "gguf-py" / "gguf" / "__init__.py").is_file()
    if not local_gguf and importlib.util.find_spec("gguf") is None:
        missing.append("gguf")
    return {"status": "ready" if not missing else "missing", "checked": checked + ["gguf"], "missing": missing}


def _disk_info(target_parent: Path | None) -> dict[str, Any]:
    if target_parent is None:
        return {"available": False, "free_bytes": None}
    try:
        usage = shutil.disk_usage(target_parent)
        return {"available": True, "free_bytes": int(usage.free), "total_bytes": int(usage.total)}
    except OSError:
        return {"available": False, "free_bytes": None}


def plan_conversion(
    *,
    model_id: str | None = None,
    source: str | Path | None = None,
    target: str | Path | None = None,
    outtype: str = "Q4_K_M",
    converter: str | Path | None = None,
    quantizer: str | Path | None = None,
) -> dict[str, Any]:
    """Build a conversion plan without creating, modifying, or deleting files."""
    if bool(model_id) == bool(source):
        raise GGUFConvertError("provide exactly one of --model-id or --source")
    raw_outtype = str(outtype).replace("-", "_")
    outtype = raw_outtype.lower() if raw_outtype.lower() in SUPPORTED_CONVERTER_OUTTYPES else raw_outtype.upper()
    if outtype not in ESTIMATE_FACTORS:
        raise GGUFConvertError(f"unsupported output type: {outtype}")
    metadata: dict[str, Any] = {}
    if source is not None:
        source_path = Path(source).expanduser().absolute()
        source_id = None
    else:
        source_path, source_id, metadata = _registered_source(model_id)
    summary, errors = _source_summary(source_path)
    if source_id is None:
        source_id = source_path.name
    target_path = Path(target).expanduser().absolute() if target else source_path.parent / f"{source_path.name}-{outtype}.gguf"
    target_parent = _existing_parent(target_path.parent)
    if not target_path.parent.is_dir():
        errors.append(_error("target_parent_missing", "target parent directory must already exist"))
    if target_path.exists():
        errors.append(_error("target_exists", "target already exists; conversion will never overwrite an existing artifact"))
    target_sidecar = target_path.with_name(target_path.name + ".sha256")
    if target_sidecar.exists() or target_sidecar.is_symlink():
        errors.append(_error("target_sidecar_exists", "target SHA256 sidecar already exists; conversion will not overwrite it"))
    if target_path.suffix.lower() != ".gguf":
        errors.append(_error("target_extension", "target must use the .gguf extension"))
    if _is_inside(target_path, source_path):
        errors.append(_error("target_inside_source", "target must not be inside the source directory"))
    converter_path, converter_status = _find_converter(Path(converter).expanduser().absolute() if converter else None)
    quantizer_path, quantizer_info = _find_quantizer(Path(quantizer).expanduser().absolute() if quantizer else None)
    converter_supported = _supported_architectures(converter_path)
    dependencies = _dependency_probe(converter_path)
    architectures = set(summary.get("architectures", []))
    architecture_supported = None if converter_supported is None or not architectures else bool(architectures & converter_supported)
    if architecture_supported is False:
        errors.append(_error("unsupported_architecture", "source architecture is not registered by the selected converter"))
    if converter_status != "available":
        errors.append(_error("converter_missing", "HF to GGUF converter is not available"))
    if dependencies["status"] == "missing":
        errors.append(_error("converter_dependencies_missing", f"converter Python dependencies are missing: {', '.join(dependencies['missing'])}"))
    needs_quantizer = outtype in QUANTIZER_REQUIRED
    if needs_quantizer and quantizer_info["status"] != "available":
        code = "quantizer_invalid" if quantizer_info["status"] == "invalid" else "quantizer_missing"
        errors.append(_error(code, f"a valid llama-quantize is required for {outtype}"))
    source_weight_bytes = int(summary.get("weight_bytes", 0) or 0)
    estimated_output = int(source_weight_bytes * ESTIMATE_FACTORS[outtype]) if source_weight_bytes else 0
    estimated_intermediate = source_weight_bytes if needs_quantizer else 0
    scratch = estimated_intermediate + estimated_output + 512 * MIB
    required_free = scratch + 256 * MIB
    disk = _disk_info(target_parent)
    if disk.get("free_bytes") is not None and int(disk["free_bytes"]) < required_free:
        errors.append(_error("insufficient_space", "target volume does not have enough free space for staging and output"))
    if target_parent is None or not os.access(target_parent, os.W_OK):
        errors.append(_error("target_parent_not_writable", "nearest existing target parent is not writable"))
    source_token = "<source>"
    target_token = "<target>"
    converter_token = "<converter>"
    quantizer_token = "<quantizer>"
    if outtype in {"f32", "f16", "bf16", "q8_0", "auto"}:
        commands = [_safe_command("python", converter_token, source_token, "--outtype", outtype.lower(), "--outfile", target_token)]
        stages = ["convert_hf_to_gguf", "inspect_gguf", "verify_gguf", "atomic_publish"]
    else:
        intermediate = f"<staging>/{source_path.name}-{outtype}-f16.gguf"
        commands = [
            _safe_command("python", converter_token, source_token, "--outtype", "f16", "--outfile", intermediate),
            _safe_command(quantizer_token, intermediate, target_token, outtype),
        ]
        stages = ["convert_hf_to_gguf", "llama_quantize", "inspect_gguf", "verify_gguf", "atomic_publish"]
    valid = not errors and bool(summary.get("exists"))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "dry_run",
        "read_only": True,
        "writes_performed": False,
        "apply_supported": True,
        "request_valid": True,
        "valid": valid,
        "source": {"model_id": source_id, **metadata, **summary},
        "target": {"label": _path_label(target_path), "parent_exists": target_path.parent.is_dir(), "nearest_existing_parent": _path_label(target_parent) if target_parent else None, "extension": target_path.suffix.lower()},
        "output_type": outtype,
        "toolchain": {
            "converter": {"label": _path_label(converter_path) if converter_path else None, "status": converter_status, "supported_outtypes": sorted(SUPPORTED_CONVERTER_OUTTYPES)},
            "quantizer": {**quantizer_info, "required": needs_quantizer},
            "architecture_supported": architecture_supported,
            "dependencies": dependencies,
        },
        "space": {"source_weight_bytes": source_weight_bytes, "estimated_intermediate_bytes": estimated_intermediate, "estimated_output_bytes": estimated_output, "staging_bytes": scratch, "required_free_bytes": required_free, **disk},
        "plan": {"stages": stages, "commands": commands, "target_must_not_exist": True, "publish": "staging -> inspect -> verify -> atomic no-overwrite hard-link publish after explicit execution confirmation"},
        "errors": errors,
    }


def _tool_command(tool: Path, *args: str) -> list[str]:
    if tool.suffix.lower() == ".py":
        return [sys.executable, str(tool), *args]
    return [str(tool), *args]


def _run_stage(command: list[str], *, timeout_seconds: float, stage: str, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"stage": stage, "status": "failed", "code": "stage_timeout", "elapsed_ms": int((time.perf_counter() - started) * 1000)}
    except (OSError, ValueError):
        return {"stage": stage, "status": "failed", "code": "stage_start_failed", "elapsed_ms": int((time.perf_counter() - started) * 1000)}
    if completed.returncode != 0:
        return {"stage": stage, "status": "failed", "code": "stage_failed", "exit_code": int(completed.returncode), "elapsed_ms": int((time.perf_counter() - started) * 1000)}
    return {"stage": stage, "status": "passed", "elapsed_ms": int((time.perf_counter() - started) * 1000)}


def _validate_staged_artifact(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        return None, _error("artifact_missing", "conversion did not produce a regular non-empty GGUF file")
    try:
        inspection = inspect_gguf(path)
        verification = verify_gguf(path, full_hash=True)
    except (OSError, GGUFError):
        return None, _error("artifact_validation_failed", "conversion output could not be parsed or verified")
    if not inspection.get("valid") or not verification.get("valid"):
        return None, _error("artifact_validation_failed", "conversion output failed GGUF structure or SHA256 validation")
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": verification.get("sha256"),
        "structure_valid": bool(inspection.get("valid")),
        "verified": bool(verification.get("valid")),
        "tensor_count": int(inspection.get("tensor_count", 0)),
        "metadata_count": int(inspection.get("metadata_count", 0)),
    }, None


def _validate_intermediate(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        return _error("intermediate_missing", "converter did not produce a regular non-empty intermediate GGUF")
    try:
        inspection = inspect_gguf(path)
    except (OSError, GGUFError):
        return _error("intermediate_validation_failed", "intermediate GGUF could not be parsed")
    if not inspection.get("valid"):
        return _error("intermediate_validation_failed", "intermediate GGUF failed structure validation")
    return None


def _publish_new(staged: Path, target: Path, sha256: str) -> dict[str, str] | None:
    target_sidecar = target.with_name(target.name + ".sha256")
    if target.exists() or target.is_symlink():
        return _error("target_exists", "target appeared before atomic publish; refusing overwrite")
    if target_sidecar.exists() or target_sidecar.is_symlink():
        return _error("target_sidecar_exists", "target SHA256 sidecar appeared before atomic publish")
    staged_sidecar = staged.with_name(staged.name + ".sha256")
    try:
        staged_sidecar.write_text(f"{sha256}  {target.name}\n", encoding="ascii")
        os.link(staged_sidecar, target_sidecar, follow_symlinks=False)
        try:
            os.link(staged, target, follow_symlinks=False)
        except (OSError, ValueError):
            target_sidecar.unlink(missing_ok=True)
            raise
    except (OSError, ValueError):
        return _error("atomic_publish_failed", "same-volume atomic publish without overwrite is unavailable")
    try:
        staged.unlink()
        staged_sidecar.unlink()
    except OSError:
        try:
            target.unlink(missing_ok=True)
            target_sidecar.unlink(missing_ok=True)
        except OSError:
            pass
        return _error("staging_cleanup_failed", "published artifact or sidecar could not detach its staging link")
    return None


def execute_conversion(
    *,
    model_id: str | None = None,
    source: str | Path | None = None,
    target: str | Path | None = None,
    outtype: str = "Q4_K_M",
    converter: str | Path | None = None,
    quantizer: str | Path | None = None,
    timeout_seconds: float = 3600.0,
    confirmation: str | None = None,
) -> dict[str, Any]:
    """Execute a confirmed conversion through staging and no-overwrite publish."""
    if confirmation != "CONVERT":
        raise GGUFConvertError("execution requires --apply --confirm CONVERT")
    if target is None:
        raise GGUFConvertError("--target is required when executing a conversion")
    if isinstance(timeout_seconds, bool) or not 1 <= float(timeout_seconds) <= 24 * 3600:
        raise GGUFConvertError("timeout_seconds must be between 1 and 86400")
    report = plan_conversion(
        model_id=model_id,
        source=source,
        target=target,
        outtype=outtype,
        converter=converter,
        quantizer=quantizer,
    )
    report = {
        **report,
        "operation": "execute",
        "read_only": False,
        "apply_requested": True,
        "preflight_valid": bool(report.get("valid")),
        "writes_performed": False,
        "execution": {"started": False, "published": False, "stages": []},
    }
    if not report.get("valid"):
        return report
    source_path = Path(source).expanduser().absolute() if source is not None else _registered_source(model_id)[0]
    target_path = Path(target).expanduser().absolute()
    converter_path, converter_status = _find_converter(Path(converter).expanduser().absolute() if converter else None)
    quantizer_path, quantizer_info = _find_quantizer(Path(quantizer).expanduser().absolute() if quantizer else None)
    if converter_path is None or converter_status != "available":
        return {**report, "valid": False, "execution": {"started": False, "published": False, "stages": [], "error": _error("converter_missing", "converter disappeared after preflight")}}
    outtype_normalized = report["output_type"]
    needs_quantizer = outtype_normalized in QUANTIZER_REQUIRED
    if needs_quantizer and (quantizer_path is None or quantizer_info["status"] != "available"):
        code = "quantizer_invalid" if quantizer_info["status"] == "invalid" else "quantizer_missing"
        return {**report, "valid": False, "execution": {"started": False, "published": False, "stages": [], "error": _error(code, "quantizer disappeared or failed validation after preflight")}}
    parent = target_path.parent
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    stage_results: list[dict[str, Any]] = []
    report["execution"]["started"] = True
    report["valid"] = False
    try:
        with tempfile.TemporaryDirectory(prefix=".qlh-gguf-staging-", dir=str(parent)) as staging_dir:
            report["writes_performed"] = True
            staging = Path(staging_dir)
            intermediate = staging / "intermediate-f16.gguf"
            converted = staging / "converted.gguf"
            if needs_quantizer:
                stage_results.append(_run_stage(
                    _tool_command(converter_path, str(source_path), "--outtype", "f16", "--outfile", str(intermediate)),
                    timeout_seconds=float(timeout_seconds), stage="convert_hf_to_gguf", env=env,
                ))
                if stage_results[-1]["status"] == "passed":
                    intermediate_error = _validate_intermediate(intermediate)
                    stage_results.append({"stage": "inspect_intermediate", "status": "failed" if intermediate_error else "passed"})
                    if intermediate_error:
                        report["execution"]["stages"] = stage_results
                        report["execution"]["error"] = intermediate_error
                        return report
                    stage_results.append(_run_stage(
                            _tool_command(quantizer_path, str(intermediate), str(converted), outtype_normalized),
                            timeout_seconds=float(timeout_seconds), stage="llama_quantize", env=env,
                        ))
            else:
                stage_results.append(_run_stage(
                    _tool_command(converter_path, str(source_path), "--outtype", outtype_normalized, "--outfile", str(converted)),
                    timeout_seconds=float(timeout_seconds), stage="convert_hf_to_gguf", env=env,
                ))
            if not stage_results or stage_results[-1]["status"] != "passed":
                report["execution"]["stages"] = stage_results
                report["execution"]["error"] = _error(stage_results[-1]["code"], "conversion stage failed")
                return report
            artifact, artifact_error = _validate_staged_artifact(converted)
            stage_results.append({"stage": "inspect_verify", "status": "failed" if artifact_error else "passed"})
            if artifact_error:
                report["execution"]["stages"] = stage_results
                report["execution"]["error"] = artifact_error
                return report
            publish_error = _publish_new(converted, target_path, str(artifact["sha256"]))
            stage_results.append({"stage": "atomic_publish", "status": "failed" if publish_error else "passed"})
            report["execution"]["stages"] = stage_results
            if publish_error:
                report["execution"]["error"] = publish_error
                return report
            report["execution"].update({"published": True, "artifact": {**artifact, "sha256_sidecar": True}})
            report["valid"] = True
            return report
    except KeyboardInterrupt:
        report["execution"]["stages"] = stage_results
        report["execution"]["error"] = _error("cancelled", "conversion was cancelled and staging was removed")
        return report
    except (OSError, RuntimeError):
        report["execution"]["stages"] = stage_results
        report["execution"]["error"] = _error("staging_failed", "could not create or clean conversion staging")
        return report


__all__ = ["GGUFConvertError", "execute_conversion", "plan_conversion"]
