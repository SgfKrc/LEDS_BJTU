"""Isolated Qwen3-VL multi-image semantics and resource smoke (MM1.19).

This is an explicitly opted-in research path. It loads the complete 4-bit
model once, processes one to four bounded local images sequentially, and
records path-free latency and memory observations. It is not a distributed
execution path and does not grant production admission.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import threading
import time
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
MIN_RAM_GATE = 10 * 2**30
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 4096
MAX_IMAGES = 4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _resolve_image_paths(request: Mapping[str, Any]) -> list[Path] | None:
    values = request.get("image_paths")
    if values is None:
        values = [request.get("image_path")]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_IMAGES:
        return None
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value).expanduser().absolute().resolve(strict=False)
        identity = str(path).lower()
        if identity in seen:
            return None
        seen.add(identity)
        paths.append(path)
    return paths


def _resolve_expected_keywords(request: Mapping[str, Any], image_count: int) -> list[list[str]] | None:
    values = request.get("expected_keywords")
    if values is None:
        return [["apple", "red", "wood"] for _ in range(image_count)]
    if not isinstance(values, list) or len(values) != image_count:
        return None
    baselines: list[list[str]] = []
    for keywords in values:
        if not isinstance(keywords, list) or not 1 <= len(keywords) <= 8:
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


class _ResourceSampler:
    """Sample process and optional CUDA counters without exposing paths."""

    def __init__(self, psutil_module: Any, torch_module: Any, interval: float = 0.1) -> None:
        self.psutil = psutil_module
        self.torch = torch_module
        self.interval = interval
        self.process = psutil_module.Process()
        self.samples: list[dict[str, int | bool]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            rss = int(self.process.memory_info().rss)
            available = int(self.psutil.virtual_memory().available)
        except Exception:
            return
        cuda_available = False
        cuda_allocated = 0
        cuda_reserved = 0
        try:
            cuda = getattr(self.torch, "cuda", None)
            cuda_available = bool(cuda is not None and cuda.is_available())
            if cuda_available:
                cuda_allocated = int(cuda.memory_allocated())
                cuda_reserved = int(cuda.memory_reserved())
        except Exception:
            cuda_available = False
        self.samples.append({
            "rss_bytes": rss,
            "available_ram_bytes": available,
            "cuda_available": cuda_available,
            "cuda_allocated_bytes": cuda_allocated,
            "cuda_reserved_bytes": cuda_reserved,
        })

    def __enter__(self) -> "_ResourceSampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, name="mm1-resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()

    def observation(
        self,
        image_count: int,
        *,
        model_load_latency_ms: float,
        total_latency_ms: float,
    ) -> dict[str, int | bool | float | str]:
        samples = self.samples or [{
            "rss_bytes": 0,
            "available_ram_bytes": 0,
            "cuda_available": False,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
        }]
        first = samples[0]
        last = samples[-1]
        peak_rss = max(int(item["rss_bytes"]) for item in samples)
        return {
            "measurement_kind": "process_rss_sampled",
            "image_count": int(image_count),
            "model_load_latency_ms": round(float(model_load_latency_ms), 3),
            "total_latency_ms": round(float(total_latency_ms), 3),
            "rss_before_bytes": int(first["rss_bytes"]),
            "rss_after_bytes": int(last["rss_bytes"]),
            "rss_peak_bytes": peak_rss,
            "rss_peak_delta_bytes": max(0, peak_rss - int(first["rss_bytes"])),
            "available_ram_before_bytes": int(first["available_ram_bytes"]),
            "available_ram_after_bytes": int(last["available_ram_bytes"]),
            "cuda_available": bool(any(bool(item["cuda_available"]) for item in samples)),
            "cuda_allocated_peak_bytes": max(int(item["cuda_allocated_bytes"]) for item in samples),
            "cuda_reserved_peak_bytes": max(int(item["cuda_reserved_bytes"]) for item in samples),
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
    if request.get("allow_full_model_materialization") is not True:
        result["status"] = "resource_rejected"
        result["errors"] = [{
            "code": "full_model_materialization_disabled",
            "message": "MM1.19 requires explicit opt-in and is not a distributed execution path",
        }]
        return result
    text_chain_id = request.get("text_chain_id")
    if not isinstance(text_chain_id, str) or _SHA256.fullmatch(text_chain_id) is None:
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "text_chain_id_invalid", "message": "text_chain_id must be a SHA-256 identifier"}]
        return result
    generation = request.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or not 0 <= generation <= 2**31 - 1:
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "generation_invalid", "message": "generation is outside the contract range"}]
        return result
    model_path = Path(str(request.get("model_path") or "")).expanduser().absolute().resolve(strict=False)
    image_paths = _resolve_image_paths(request)
    expected_keywords = _resolve_expected_keywords(request, len(image_paths)) if image_paths is not None else None
    if (
        not model_path.is_dir()
        or image_paths is None
        or expected_keywords is None
        or any(not path.is_file() for path in image_paths)
    ):
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "request_incomplete", "message": "vision text smoke request is incomplete"}]
        return result
    try:
        for image_path in image_paths:
            if image_path.stat().st_size <= 0 or image_path.stat().st_size > MAX_IMAGE_BYTES:
                result["status"] = "invalid_request"
                result["errors"] = [{"code": "image_size_invalid", "message": "image exceeds the bounded smoke input"}]
                return result
    except OSError:
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "image_unreadable", "message": "image cannot be inspected"}]
        return result

    model = processor = inputs = generated_ids = image = None
    try:
        import psutil
    except Exception as exc:
        result["status"] = "runtime_rejected"
        result["errors"] = [{"code": "memory_probe_unavailable", "message": exc.__class__.__name__}]
        return result
    if psutil.virtual_memory().available < MIN_RAM_GATE:
        result["status"] = "resource_rejected"
        result["errors"] = [{"code": "insufficient_ram", "message": "MM1.19 requires >= 10 GiB available RAM"}]
        return result
    try:
        import torch
        from transformers import AutoConfig, AutoProcessor
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    except Exception as exc:
        result["status"] = "runtime_rejected"
        result["errors"] = [{"code": "vision_text_runtime_unavailable", "message": exc.__class__.__name__}]
        return result

    try:
        config = AutoConfig.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        if str(getattr(config, "model_type", "")) != "qwen3_vl":
            raise ValueError("model is not a Qwen3-VL checkpoint")
        quantization = BitsAndBytesConfig(load_in_4bit=True)
        observations: list[dict[str, Any]] = []
        total_started = time.perf_counter()
        model_started = total_started
        with _ResourceSampler(psutil, torch) as sampler:
            model_started = time.perf_counter()
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path), quantization_config=quantization,
                local_files_only=True, trust_remote_code=False,
                device_map="cpu", torch_dtype=torch.float32,
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
            )
            import PIL.Image
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
            model_load_latency_ms = (time.perf_counter() - model_started) * 1000
            for image_index, image_path in enumerate(image_paths):
                started = time.perf_counter()
                with PIL.Image.open(image_path) as opened_image:
                    if (
                        opened_image.width <= 0
                        or opened_image.height <= 0
                        or opened_image.width > 4096
                        or opened_image.height > 4096
                    ):
                        raise ValueError("image dimensions are outside the smoke limit")
                    image = opened_image.convert("RGB")
                inputs = processor(text=[prompt], images=[image], return_tensors="pt")
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                description = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )[0].strip()
                if len(description) > MAX_DESCRIPTION_CHARS:
                    description = description[:MAX_DESCRIPTION_CHARS]
                lowered = description.lower()
                keywords = expected_keywords[image_index]
                hits = {keyword: (keyword in lowered) for keyword in keywords}
                observations.append({
                    "image_index": image_index,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "description": description,
                    "keyword_hits": hits,
                    "keyword_hit_count": sum(1 for hit in hits.values() if hit),
                    "keyword_count": len(hits),
                    "semantic_gate_passed": all(hits.values()),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                })
                image.close()
                image = None
                inputs = generated_ids = None
        resource_observation = sampler.observation(
            len(image_paths),
            model_load_latency_ms=model_load_latency_ms,
            total_latency_ms=(time.perf_counter() - total_started) * 1000,
        )

        first_observation = observations[0]
        result.update({
            "gate_passed": True,
            "status": "vision_semantics_loaded",
            "response": {
                "schema_version": SCHEMA_VERSION,
                "response_kind": "qwen3_vision_text_real_semantics",
                "model_id": str(config.model_type),
                "description": first_observation["description"],
                "keyword_hits": first_observation["keyword_hits"],
                "image_count": len(observations),
                "images": observations,
                "semantic_pass_count": sum(1 for item in observations if item["semantic_gate_passed"]),
                "resource_observation": resource_observation,
                "text_weights_loaded": True,
                "weight_materialized": True,
                "full_model_materialized": True,
                "explicit_full_model_opt_in": True,
            },
        })
        return result
    except Exception as exc:
        result["status"] = "vision_semantics_failed"
        result["errors"] = [{"code": "vision_semantics_failed", "message": exc.__class__.__name__}]
        return result
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
        model = processor = inputs = generated_ids = image = None
        gc.collect()


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
