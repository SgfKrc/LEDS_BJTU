"""Validate the real SD 1.5 -> LLM lifecycle through inference-svc routes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from PIL import Image

from inference_svc_main import build_app


def _expect(response, status_code: int) -> dict[str, Any]:
    if response.status_code != status_code:
        raise RuntimeError(
            f"{response.request.method} {response.request.url.path} returned "
            f"{response.status_code}: {response.text[:1000]}"
        )
    if not response.content:
        return {}
    return response.json()


def _wait_job(
    client: TestClient,
    job_id: str,
    terminal_states: set[str],
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _expect(client.get(f"/v1/diffusion/jobs/{job_id}"), 200)
        if last.get("state") in terminal_states:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish: {last}")


def _wait_job_started(
    client: TestClient,
    job_id: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _expect(client.get(f"/v1/diffusion/jobs/{job_id}"), 200)
        if last.get("state") == "running" and int(
            last.get("progress", {}).get("step", 0)
        ) >= 1:
            return last
        if last.get("state") in {"completed", "failed", "cancelled"}:
            raise RuntimeError(f"job ended before cancellation: {last}")
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not start: {last}")


def _cuda_snapshot() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    torch.cuda.synchronize()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SD generation/cancellation and the switch back to a local LLM"
    )
    parser.add_argument("--model-path", default="models/sd15-original-v1")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cancel-steps", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--llm-model-id", default="qwen-1_8b")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"SD model directory not found: {model_path}")
    if args.steps <= 0 or args.cancel_steps <= args.steps:
        raise SystemExit("--cancel-steps must be greater than --steps, both positive")

    report: dict[str, Any] = {
        "model_path": str(model_path),
        "profile": args.profile,
        "cuda_before": _cuda_snapshot(),
    }
    app = build_app("master")
    with TestClient(app) as client:
        inspected = _expect(
            client.post(
                "/v1/diffusion/artifacts/inspect",
                json={"path": str(model_path), "compute_hash": False},
            ),
            200,
        )
        if inspected.get("artifact_kind") != "sd15_pipeline":
            raise RuntimeError(f"unexpected artifact: {inspected}")
        _expect(
            client.post(
                "/v1/diffusion/artifacts/register",
                json={
                    "path": str(model_path),
                    "artifact_id": "sd15_api_validation",
                    "name": "SD 1.5 API validation",
                },
            ),
            200,
        )

        load_started = time.perf_counter()
        loaded = _expect(
            client.post(
                "/v1/diffusion/load",
                json={
                    "artifact_id": "sd15_api_validation",
                    "profile": args.profile,
                    "safety_checker_required": True,
                },
            ),
            200,
        )
        report["sd_load_seconds"] = round(time.perf_counter() - load_started, 3)
        report["sd_engine_config"] = loaded.get("engine_config")

        generated = _expect(
            client.post(
                "/v1/diffusion/generate",
                json={
                    "preset_id": "sd15_original_v1",
                    "seed": 19950101,
                    "steps": args.steps,
                },
            ),
            202,
        )
        completed = _wait_job(
            client,
            generated["job_id"],
            {"completed", "failed"},
            timeout=args.timeout,
        )
        if completed["state"] != "completed":
            raise RuntimeError(f"generation failed: {completed}")
        blob_id = completed["blob"]["blob_id"]
        blob = client.get(f"/v1/diffusion/blobs/{blob_id}")
        if blob.status_code != 200 or blob.headers.get("content-type") != "image/png":
            raise RuntimeError(f"invalid blob response: {blob.status_code} {blob.headers}")
        with Image.open(io.BytesIO(blob.content)) as image:
            image.load()
            image_size = image.size
            rgb = image.convert("RGB")
            image_extrema = [list(values) for values in rgb.getextrema()]
            if rgb.getbbox() is None:
                raise RuntimeError(
                    "SD pipeline returned an all-black image; safety checker or generation failed"
                )
        report["generation"] = {
            "job_id": generated["job_id"],
            "elapsed_seconds": completed.get("metrics", {}).get("elapsed_seconds"),
            "progress": completed.get("progress"),
            "blob_bytes": len(blob.content),
            "blob_sha256": hashlib.sha256(blob.content).hexdigest(),
            "image_size": list(image_size),
            "image_extrema": image_extrema,
        }

        cancellable = _expect(
            client.post(
                "/v1/diffusion/generate",
                json={
                    "preset_id": "sd15_original_v1",
                    "seed": 19950102,
                    "steps": args.cancel_steps,
                },
            ),
            202,
        )
        started = _wait_job_started(
            client, cancellable["job_id"], timeout=args.timeout
        )
        cancel_started = time.perf_counter()
        accepted = _expect(
            client.post(f"/v1/diffusion/jobs/{cancellable['job_id']}/cancel"),
            200,
        )
        cancelled = _wait_job(
            client,
            cancellable["job_id"],
            {"cancelled", "completed", "failed"},
            timeout=args.timeout,
        )
        if not accepted.get("accepted") or cancelled["state"] != "cancelled":
            raise RuntimeError(f"cancellation did not win: {accepted} {cancelled}")
        if cancelled.get("blob") is not None:
            raise RuntimeError(f"cancelled job retained a blob: {cancelled}")
        report["cancellation"] = {
            "job_id": cancellable["job_id"],
            "requested_after_step": started["progress"]["step"],
            "terminal_progress": cancelled["progress"],
            "latency_seconds": round(time.perf_counter() - cancel_started, 3),
        }

        unloaded = _expect(client.post("/v1/diffusion/unload"), 200)
        if unloaded.get("loaded") or unloaded.get("state") != "unloaded":
            raise RuntimeError(f"SD unload failed: {unloaded}")
        report["cuda_after_sd_unload"] = _cuda_snapshot()

        if not args.skip_llm:
            llm_started = time.perf_counter()
            llm_loaded = _expect(
                client.post(
                    "/v1/models/load",
                    json={
                        "engine": "llama_cpp",
                        "quant_type": "int4",
                        "use_compile": False,
                        "model_id": args.llm_model_id,
                    },
                ),
                200,
            )
            current = _expect(client.get("/v1/models/current"), 200)
            if not current.get("loaded"):
                raise RuntimeError(f"LLM did not become loaded: {llm_loaded} {current}")
            chat = _expect(
                client.post(
                    "/v1/chat",
                    json={
                        "message": "Reply with exactly OK.",
                        "max_new_tokens": 8,
                        "temperature": 0.0,
                        "streaming_mode": "fast",
                    },
                ),
                200,
            )
            report["llm"] = {
                "load_seconds": round(time.perf_counter() - llm_started, 3),
                "model_id": current.get("model_id"),
                "engine": current.get("engine"),
                "response": chat.get("response") or chat.get("content"),
                "metrics": chat.get("metrics"),
            }
            _expect(client.post("/v1/models/unload", json={}), 200)

    report["cuda_final"] = _cuda_snapshot()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
