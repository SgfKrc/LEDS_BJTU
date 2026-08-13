"""Metadata-only Qwen3 pipeline adapter worker.

This is the control-plane half of the isolated sidecar.  It validates the
assignment before a future weight loader is allowed to read a tensor.  The
actual model object is intentionally not created by this operation.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from .qwen3_pipeline_adapter import (
        QWEN3_ADAPTER_SCHEMA_VERSION,
        Qwen3AdapterError,
        validate_qwen3_assignment,
    )
except ImportError:  # direct sidecar script execution
    from qwen3_pipeline_adapter import (  # type: ignore
        QWEN3_ADAPTER_SCHEMA_VERSION,
        Qwen3AdapterError,
        validate_qwen3_assignment,
    )


TOOL = "qwen3_pipeline_sidecar"
MAX_INPUT_BYTES = 256 * 1024


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_pipeline_adapter_preflight",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "preflight_failed",
        "adapter": {
            "model_type": request.get("model_type"),
            "layer_range": request.get("layer_range"),
            "synthetic_forward_ready": False,
        },
        "errors": [],
    }
    try:
        if request.get("schema_version") != QWEN3_ADAPTER_SCHEMA_VERSION:
            raise Qwen3AdapterError("unsupported Qwen3 adapter schema")
        if request.get("operation") != "qwen3_pipeline_adapter_preflight":
            raise Qwen3AdapterError("unsupported Qwen3 adapter operation")
        if request.get("read_only") is not True or request.get("network_access") != "disabled":
            raise Qwen3AdapterError("Qwen3 adapter must be read-only and network-disabled")
        layer_range = request.get("layer_range")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            raise Qwen3AdapterError("layer_range must contain [start, end]")
        report = validate_qwen3_assignment(
            model_type=str(request.get("model_type", "")),
            total_layers=int(request.get("total_layers", 0)),
            start_layer=int(layer_range[0]),
            end_layer=int(layer_range[1]),
            has_embedding=bool(request.get("has_embedding", False)),
            has_lm_head=bool(request.get("has_lm_head", False)),
            keys=request.get("keys", []),
            tie_word_embeddings=bool(request.get("tie_word_embeddings", False)),
        )
        result["adapter"].update(report)
        result["adapter"]["synthetic_forward_ready"] = True
        result["gate_passed"] = True
        result["status"] = "ready_for_synthetic_forward"
    except (TypeError, ValueError, Qwen3AdapterError) as exc:
        result["errors"].append({"code": "assignment_rejected", "message": str(exc)[:4096]})
    return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Qwen3 adapter request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    print(json.dumps(execute_request(request), ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
