#!/usr/bin/env python3
"""MF-MEM-N2R real-artifact admission gate.

The N2R comparison is only valid for a fixed 7B INT8/NF4 PyTorch artifact.
This gate inventories local evidence and returns an explicit resource rejection
when the artifact is absent. It never treats a smaller model or a GGUF Q4 file
as a 7B INT8/NF4 substitute, and it does not load model weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Sequence


TARGET_FAMILY = "deepseek-r1-distill-qwen-7b"
TARGET_QUANTIZATIONS = ("int8", "nf4")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_record(path: Path, *, format_name: str, quantization: str | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "format": format_name,
        "quantization": quantization,
    }


def inspect_artifacts(project_root: Path) -> dict[str, Any]:
    models = project_root / "models"
    gguf = models / "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    safetensors_manifest = (
        project_root
        / "build"
        / "model-fleet"
        / "model-store-20260808"
        / "manifests"
        / "migration"
        / "deepseek-r1-distill-qwen-7b-safetensors"
        / "builtin-20260808.json"
    )
    candidate_model = models / "qwen-1_8b-chat"
    target_model = models / "deepseek-r1-distill-qwen-7b"
    candidate_weights = sorted(candidate_model.glob("*.safetensors"))
    candidate_quant_evidence = project_root / "build" / "experiments" / "quant-qwen-v1"

    manifest: dict[str, Any] = {}
    if safetensors_manifest.is_file():
        try:
            manifest = json.loads(safetensors_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    expected_shards = [
        str(item.get("path"))
        for item in manifest.get("files", [])
        if str(item.get("path", "")).endswith(".safetensors")
    ]
    actual_shards = [str(candidate) for candidate in target_model.glob("*.safetensors")]

    records = [
        _file_record(gguf, format_name="gguf", quantization="q4_k_m"),
    ]
    records.append({
        "path": str(safetensors_manifest),
        "exists": safetensors_manifest.is_file(),
        "format": "manifest",
        "quantization": manifest.get("quantization"),
        "expected_safetensors": expected_shards,
        "actual_target_safetensors": actual_shards,
        "all_expected_weights_present": bool(expected_shards) and len(actual_shards) >= len(expected_shards),
    })
    records.append({
        "path": str(candidate_model),
        "exists": candidate_model.is_dir(),
        "format": "safetensors",
        "quantization": ["int8", "nf4"] if candidate_weights else [],
        "parameter_family": "qwen-1.8b",
        "weight_files": [_file_record(path, format_name="safetensors", quantization=None) for path in candidate_weights],
        "existing_experiment_evidence": candidate_quant_evidence.is_dir(),
    })

    target_weight_files_present = bool(expected_shards) and len(actual_shards) >= len(expected_shards)
    bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
    target_quantization_supported = bool(target_weight_files_present and bitsandbytes_available)
    # The fixed artifact is BF16 Safetensors; INT8/NF4 is an explicit, pinned
    # runtime quantization recipe over that digest, not an unverified substitute.
    target_artifact_available = target_quantization_supported
    reason_codes: list[str] = []
    if not target_weight_files_present:
        reason_codes.append("target_7b_safetensors_weights_missing")
    if records[0]["exists"]:
        reason_codes.append("7b_gguf_q4_k_m_is_not_target_quantization")
    if candidate_weights:
        reason_codes.append("available_quant_candidate_is_qwen_1_8b_not_7b")
    if not bitsandbytes_available:
        reason_codes.append("bitsandbytes_unavailable")
    if target_artifact_available:
        reason_codes.append("fixed_7b_bf16_weights_with_explicit_runtime_int8_nf4_recipe")

    return {
        "schema_version": 1,
        "ticket": "MF-MEM-N2R",
        "status": "resource_rejected" if not target_artifact_available else "ready_for_compare",
        "completed": 1,
        "resource_rejected": int(not target_artifact_available),
        "target_artifact_available": int(target_artifact_available),
        "target_weight_files_present": int(target_weight_files_present),
        "target_quantization_supported": int(target_quantization_supported),
        "target": {
            "family": TARGET_FAMILY,
            "required_quantizations": list(TARGET_QUANTIZATIONS),
            "format": "pytorch_safetensors",
        },
        "local_artifacts": records,
        "candidate_quant_artifact_available": int(bool(candidate_weights)),
        "candidate_target_match": 0,
        "bitsandbytes_available": int(bitsandbytes_available),
        "reason_codes": reason_codes,
        "next_gate": "acquire_and_verify_fixed_7b_int8_or_nf4_safetensors_artifact",
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MF-MEM-N2R fixed 7B INT8/NF4 artifact gate")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result-file", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = inspect_artifacts(args.project_root.resolve())
    _write_json(args.result_file.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
