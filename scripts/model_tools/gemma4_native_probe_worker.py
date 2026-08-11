"""Isolated runtime half of the native Gemma 4 multimodal preflight."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


TOOL = "gemma4_native_probe"
SCHEMA_VERSION = 1
REQUIRED_MTMD_SYMBOLS = (
    "mtmd_context_params_default",
    "mtmd_free",
    "mtmd_init_from_file",
    "mtmd_support_audio",
    "mtmd_support_vision",
)


def _base_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "native_multimodal_preflight",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "project_llama_cpp_revision": request.get("project_llama_cpp_revision"),
        "artifact_probe_requested": bool(request.get("artifact_requested")),
        "status": "binding_unavailable",
        "gate_passed": False,
        "binding": {
            "status": "unavailable",
            "package_version": None,
            "mtmd_abi": False,
            "missing_symbols": list(REQUIRED_MTMD_SYMBOLS),
        },
        "artifacts": {
            "model": {"provided": False, "exists": False, "sha256_verified": False},
            "mmproj": {"provided": False, "exists": False, "sha256_verified": False},
            "verified": False,
        },
        "capabilities": {
            "context_initialized": False,
            "vision": False,
            "audio": False,
            "native_vision_preflight_passed": False,
            "native_audio_preflight_passed": False,
        },
        "errors": [],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_state(path_value: str, expected_sha256: str) -> tuple[dict[str, Any], Path | None]:
    if not path_value:
        return {"provided": False, "exists": False, "sha256_verified": False}, None
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file():
        return {"provided": True, "exists": False, "sha256_verified": False}, None
    actual = _sha256(path)
    return {
        "provided": True,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256_verified": actual == expected_sha256,
    }, path


def _binding(module_loader: Callable[[str], Any]) -> tuple[Any | None, Any | None, dict[str, Any], dict[str, str] | None]:
    try:
        binding = module_loader("llama_cpp")
    except Exception as exc:
        return None, None, {
            "status": "unavailable",
            "package_version": None,
            "mtmd_abi": False,
            "missing_symbols": list(REQUIRED_MTMD_SYMBOLS),
        }, {"code": "binding_import_failed", "message": exc.__class__.__name__}
    try:
        mtmd = module_loader("llama_cpp.mtmd_cpp")
    except Exception as exc:
        return binding, None, {
            "status": "incomplete",
            "package_version": getattr(binding, "__version__", None),
            "mtmd_abi": False,
            "missing_symbols": list(REQUIRED_MTMD_SYMBOLS),
        }, {"code": "mtmd_import_failed", "message": exc.__class__.__name__}

    missing = [name for name in REQUIRED_MTMD_SYMBOLS if not callable(getattr(mtmd, name, None))]
    return binding, mtmd, {
        "status": "available" if not missing else "incomplete",
        "package_version": getattr(binding, "__version__", None),
        "mtmd_abi": not missing,
        "missing_symbols": missing,
    }, None


def execute_request(
    request: dict[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    result = _base_result(request)
    binding, mtmd, binding_report, binding_error = _binding(module_loader)
    result["binding"] = binding_report
    if binding_error is not None:
        result["errors"].append(binding_error)
    if not binding_report["mtmd_abi"]:
        return result

    if not request.get("artifact_requested"):
        result["status"] = "binding_ready"
        return result

    model_state, model_path = _artifact_state(
        str(request.get("model_path", "")), str(request.get("model_sha256", "")),
    )
    mmproj_state, mmproj_path = _artifact_state(
        str(request.get("mmproj_path", "")), str(request.get("mmproj_sha256", "")),
    )
    result["artifacts"] = {
        "model": model_state,
        "mmproj": mmproj_state,
        "verified": bool(model_state.get("sha256_verified") and mmproj_state.get("sha256_verified")),
    }
    if not result["artifacts"]["verified"] or model_path is None or mmproj_path is None:
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "artifact_verification_failed", "message": "missing or digest-mismatched local artifact"})
        return result

    model = None
    mtmd_context = None
    try:
        llama_factory = getattr(binding, "Llama", None)
        if not callable(llama_factory):
            raise RuntimeError("Llama factory unavailable")
        model = llama_factory(
            model_path=str(model_path),
            n_ctx=int(request.get("n_ctx", 512)),
            n_batch=min(64, int(request.get("n_ctx", 512))),
            n_gpu_layers=0,
            verbose=False,
        )
        mtmd_context = mtmd.mtmd_init_from_file(
            str(mmproj_path).encode("utf-8"),
            model.model,
            mtmd.mtmd_context_params_default(),
        )
        if not mtmd_context:
            raise RuntimeError("MTMD context initialization returned null")
        vision = bool(mtmd.mtmd_support_vision(mtmd_context))
        audio = bool(mtmd.mtmd_support_audio(mtmd_context))
        capabilities = result["capabilities"]
        capabilities.update({
            "context_initialized": True,
            "vision": vision,
            "audio": audio,
            "native_vision_preflight_passed": vision,
            "native_audio_preflight_passed": audio,
        })
        result["gate_passed"] = vision and (audio or not request.get("require_audio"))
        result["status"] = "ready_for_image_smoke" if result["gate_passed"] else "required_capability_missing"
        if not vision:
            result["errors"].append({"code": "vision_not_supported", "message": "MTMD context does not report vision support"})
        elif request.get("require_audio") and not audio:
            result["errors"].append({"code": "audio_not_supported", "message": "MTMD context does not report audio support"})
    except Exception as exc:
        result["status"] = "initialization_failed"
        result["errors"].append({"code": "native_initialization_failed", "message": exc.__class__.__name__})
    finally:
        if mtmd_context is not None:
            try:
                mtmd.mtmd_free(mtmd_context)
            except Exception:
                pass
        if model is not None:
            close = getattr(model, "close", None)
            try:
                if callable(close):
                    close()
                else:
                    stack = getattr(model, "_stack", None)
                    stack_close = getattr(stack, "close", None)
                    if callable(stack_close):
                        stack_close()
            except Exception:
                pass
        del model
        gc.collect()
    return result


def main() -> int:
    raw = sys.stdin.buffer.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("native probe request exceeds 256 KiB")
    request = json.loads(raw.decode("utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION or request.get("operation") != "native_multimodal_preflight":
        raise ValueError("unsupported native probe protocol")
    if request.get("network_access") != "disabled" or request.get("read_only") is not True:
        raise ValueError("native probe must be read-only and network-disabled")
    print(json.dumps(execute_request(request), ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fallback = _base_result({})
        fallback["valid"] = False
        fallback["status"] = "invalid_request"
        fallback["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
        print(json.dumps(fallback, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
