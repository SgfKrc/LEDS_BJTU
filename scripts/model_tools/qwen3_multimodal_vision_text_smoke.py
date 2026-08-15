"""Controller for the isolated Qwen3-VL real vision semantics smoke (MM1.19)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TOOL = "qwen3_multimodal_vision_text_smoke"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 1800.0
MAX_IMAGES = 4


def _request_error(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_vision_text_real_semantics",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "invalid_request",
        "errors": [{"code": code, "message": message}],
    }


def evaluate_mm1_production_route(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a path-free, fail-closed route decision for MM1.19 evidence."""
    reasons: list[str] = []
    response = report.get("response") if isinstance(report, Mapping) else None
    if not isinstance(response, Mapping) or report.get("status") != "vision_semantics_loaded":
        reasons.append("semantic_smoke_not_passed")
    else:
        if response.get("full_model_materialized") is not False:
            reasons.append("full_model_materialization_true")
        if response.get("explicit_full_model_opt_in") is True:
            reasons.append("research_opt_in_only")
        image_count = response.get("image_count")
        if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 2:
            reasons.append("multi_image_baseline_missing")
        image_observations = response.get("images")
        if (
            not isinstance(image_observations, list)
            or not image_observations
            or any(
                not isinstance(item, Mapping) or item.get("semantic_gate_passed") is not True
                for item in image_observations
            )
        ):
            reasons.append("semantic_baseline_not_passed")
        observation = response.get("resource_observation")
        if not isinstance(observation, Mapping):
            reasons.append("resource_observation_missing")
        else:
            for key in ("rss_peak_bytes", "rss_peak_delta_bytes", "available_ram_before_bytes"):
                if not isinstance(observation.get(key), int) or observation.get(key) < 0:
                    reasons.append("resource_observation_incomplete")
                    break
    reasons.extend(["cuda_parity_pending", "distributed_hidden_handoff_pending", "long_video_pending"])
    native_admitted = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": "qwen3_mm1_production_route",
        "recommended_route": "native_sidecar" if native_admitted else "external_api",
        "native_sidecar": {
            "admitted": native_admitted,
            "reasons": reasons,
            "requires": [
                "no_full_model_materialization",
                "multi_image_semantics_baseline",
                "resource_observation",
                "cuda_parity",
                "distributed_hidden_handoff",
                "long_video",
            ],
        },
        "external_api": {
            "admitted": "conditional",
            "reason": "provider_policy_and_credentials_are_runtime_gates",
        },
    }


def _normalise_images(image: Path | None, images: Sequence[Path] | None) -> list[Path] | None:
    if images is not None and isinstance(images, (str, bytes)):
        return None
    try:
        values = list(images) if images is not None else []
    except TypeError:
        return None
    if image is not None:
        if not values:
            values = [image]
        elif image not in values:
            values.insert(0, image)
    if not values or len(values) > MAX_IMAGES:
        return None
    resolved: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            if not isinstance(value, Path):
                value = Path(value)
            path = value.expanduser().absolute().resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return None
        identity = str(path).lower()
        if identity in seen:
            return None
        seen.add(identity)
        resolved.append(path)
    return resolved


def _normalise_keywords(
    image_count: int,
    expected_keywords: Sequence[Sequence[str]] | None,
) -> list[list[str]] | None:
    if expected_keywords is None:
        return [["apple", "red", "wood"] for _ in range(image_count)]
    try:
        if isinstance(expected_keywords, (str, bytes)) or len(expected_keywords) != image_count:
            return None
    except TypeError:
        return None
    baselines: list[list[str]] = []
    for keywords in expected_keywords:
        try:
            invalid_shape = isinstance(keywords, (str, bytes)) or not 1 <= len(keywords) <= 8
        except TypeError:
            return None
        if invalid_shape:
            return None
        normalised: list[str] = []
        for keyword in keywords:
            if not isinstance(keyword, str) or not 1 <= len(keyword.strip()) <= 64:
                return None
            value = keyword.strip().lower()
            if value in normalised:
                return None
            normalised.append(value)
        baselines.append(normalised)
    return baselines


def run_qwen3_multimodal_vision_text_smoke(
    *,
    model: Path | None,
    image: Path | None = None,
    images: Sequence[Path] | None = None,
    expected_keywords: Sequence[Sequence[str]] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    text_chain_id: str = "a" * 64,
    generation: int = 4,
    allow_full_model_materialization: bool = False,
) -> dict[str, Any]:
    """Run an explicitly opt-in research smoke that materializes the full model.

    The default is fail-closed because this path is not compatible with the
    distributed/no-peak-memory runtime contract.
    """
    normalised_images = _normalise_images(image, images)
    if model is None or normalised_images is None:
        return _request_error("request_incomplete", "MM1.19 requires --model and one to four images")
    semantic_baselines = _normalise_keywords(len(normalised_images), expected_keywords)
    if semantic_baselines is None:
        return _request_error("semantic_baseline_invalid", "each image requires one to eight bounded keywords")
    if not isinstance(allow_full_model_materialization, bool):
        return _request_error("opt_in_invalid", "allow_full_model_materialization must be boolean")
    if not isinstance(text_chain_id, str) or len(text_chain_id) != 64:
        return _request_error("text_chain_id_invalid", "text_chain_id must be a 64-character digest")
    if isinstance(generation, bool) or not isinstance(generation, int) or not 0 <= generation <= 2**31 - 1:
        return _request_error("generation_invalid", "generation is outside the contract range")
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        return _request_error("sidecar_missing", "MM1.19 requires the isolated Qwen3 pipeline sidecar")
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_vision_text_real_semantics",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model.expanduser().absolute().resolve(strict=False)),
        "image_path": str(normalised_images[0]),
        "image_paths": [str(path) for path in normalised_images],
        "expected_keywords": semantic_baselines,
        "text_chain_id": text_chain_id,
        "generation": generation,
        "allow_full_model_materialization": allow_full_model_materialization,
    }
    worker = Path(__file__).with_name("qwen3_multimodal_vision_text_smoke_worker.py")
    try:
        proc = subprocess.run(
            [str(sidecar), str(worker)],
            input=json.dumps(request, separators=(",", ":")),
            capture_output=True, text=True, encoding="utf-8",
            timeout=float(timeout_seconds), cwd=str(ROOT),
        )
    except Exception:
        return _request_error("worker_failed", "vision text smoke worker did not return")
    try:
        report = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return _request_error("worker_output_invalid", "vision text smoke worker output is invalid")
    if not isinstance(report, dict) or report.get("tool") != TOOL or report.get("schema_version") != SCHEMA_VERSION:
        return _request_error("worker_report_invalid", "vision text smoke worker report is invalid")
    encoded = json.dumps(report, ensure_ascii=True, separators=(",", ":")).lower()
    sensitive_paths = [str(model.expanduser().absolute()), str(ROOT)]
    sensitive_paths.extend(str(path) for path in normalised_images)
    for path_value in sensitive_paths:
        if path_value.lower() in encoded:
            return _request_error("worker_report_sensitive", "vision text smoke worker report contains a path")
    if report.get("status") == "vision_semantics_loaded":
        response = report.get("response")
        if not isinstance(response, dict) or response.get("full_model_materialized") is not True:
            return _request_error("worker_report_inconsistent", "full-model smoke did not declare materialization")
        if response.get("image_count") != len(normalised_images):
            return _request_error("worker_report_inconsistent", "worker image count does not match request")
        if not isinstance(response.get("images"), list) or len(response["images"]) != len(normalised_images):
            return _request_error("worker_report_inconsistent", "worker image observations are incomplete")
        if not isinstance(response.get("resource_observation"), dict):
            return _request_error("worker_report_inconsistent", "worker resource observation is missing")
        report["production_route_evaluation"] = evaluate_mm1_production_route(report)
    return report


__all__ = ["evaluate_mm1_production_route", "run_qwen3_multimodal_vision_text_smoke"]
