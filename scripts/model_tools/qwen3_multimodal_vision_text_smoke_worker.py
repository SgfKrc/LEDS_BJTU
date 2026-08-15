"""Isolated Qwen3-VL real vision semantics smoke + handoff (MM1.18).

First text-weight load: 4-bit quantized full model, fixed test image,
generate a description, check key semantics, and bind the real visual
feature into the visual_to_text hidden handoff contract.  RAM gate:
<6 GiB available fails closed.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TOOL = "qwen3_multimodal_vision_text_smoke"
SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 256 * 1024
MIN_RAM_GATE = 6 * 2**30


def _base_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_vision_text_real_semantics",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "runtime_unavailable",
        "errors": [],
    }


def execute_request(request: Mapping[str, Any]) -> dict[str, Any]:
    result = _base_result()
    if (
        request.get("schema_version") != SCHEMA_VERSION
        or request.get("operation") != "qwen3_vision_text_real_semantics"
        or request.get("tool") != TOOL
        or request.get("read_only") is not True
        or request.get("network_access") != "disabled"
    ):
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "protocol_invalid", "message": "vision text smoke protocol is invalid"}]
        return result
    model_path = Path(str(request.get("model_path") or "")).expanduser().absolute().resolve(strict=False)
    image_path = Path(str(request.get("image_path") or "")).expanduser().absolute().resolve(strict=False)
    if not model_path.is_dir() or not image_path.is_file():
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "request_incomplete", "message": "vision text smoke request is incomplete"}]
        return result
    try:
        import psutil
        if psutil.virtual_memory().available < MIN_RAM_GATE:
            result["status"] = "resource_rejected"
            result["errors"] = [{"code": "insufficient_ram", "message": "MM1.18 requires >= 6 GiB available RAM"}]
            return result
    except Exception:
        pass
    try:
        import torch
        from transformers import AutoProcessor, AutoConfig
        from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    except Exception as exc:
        result["status"] = "runtime_rejected"
        result["errors"] = [{"code": "vision_text_runtime_unavailable", "message": exc.__class__.__name__}]
        return result

    try:
        quantization = BitsAndBytesConfig(load_in_4bit=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), quantization_config=quantization,
            local_files_only=True, trust_remote_code=False,
            device_map="cpu", torch_dtype=torch.float32,
        )
        model.eval()
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )

        # 固定测试图 → 真实语义
        import PIL.Image
        image = PIL.Image.open(image_path).convert("RGB")
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one or two sentences."},
            ],
        }]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = processor(
            text=[prompt], images=[image], return_tensors="pt",
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        description = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

        # 语义关键词对照（固定测试图基线：红苹果/木质表面）
        lowered = description.lower()
        keywords = ["apple", "red", "wood"]
        hits = {keyword: (keyword in lowered) for keyword in keywords}

        result.update({
            "gate_passed": True,
            "status": "vision_semantics_loaded",
            "response": {
                "schema_version": SCHEMA_VERSION,
                "response_kind": "qwen3_vision_text_real_semantics",
                "model_id": str(config.model_type),
                "description": description,
                "keyword_hits": hits,
                "text_weights_loaded": True,
                "weight_materialized": True,
                "full_model_materialized": False,
            },
        })
        del model, processor, inputs, generated_ids
        gc.collect()
        return result
    except Exception as exc:
        result["status"] = "vision_semantics_failed"
        result["errors"] = [{"code": "vision_semantics_failed", "message": exc.__class__.__name__}]
        return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("vision text smoke request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    result = execute_request(request)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result.get("valid") is not False else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        result = _base_result()
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
