"""Run the deterministic SD 1.5 IP-Adapter reference-image quality gate.

The gate is deliberately separate from the normal img2img gate.  A reference
image is an IP-Adapter condition, not a latent source, so its acceptance
criteria include the complete adapter directory, the pinned source SHA, safety
checker results, output uniqueness, and clean unload behavior.

Use ``--seed-limit`` or ``--scale`` for a runtime smoke only.  Such a report
is recorded as ``partial_pass`` and cannot complete SD-N5.1A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import (  # noqa: E402
    DiffusionArtifactInspector,
    SD15EditRequest,
    SD15Engine,
    build_sd15_engine_config,
    build_sd15_generation_request,
    get_asset_spec,
    get_preset,
    verify_asset_directory,
)


DEFAULT_SCALES = (0.45, 0.60, 0.80)
DEFAULT_ADAPTER_ID = "sd15_ip_adapter_v1"
SAFETY_REPLACEMENT_SEEDS = (19950112, 19950113)
STEADY_MEMORY_GROWTH_LIMIT_BYTES = 256 * 1024 * 1024
UNLOAD_MEMORY_GROWTH_LIMIT_BYTES = 64 * 1024 * 1024
REFERENCE_PROMPTS = {
    "sd15_original_v1": (
        "portrait of the same blonde adult woman in a blue jacket, futuristic city "
        "at night, detailed 1990s anime style",
        "different person, dark hair, deformed face, extra eyes, blurry, low quality, "
        "text, watermark",
    ),
    "sd15_90s_retrovers_v1": (
        "retrovers, head-and-shoulders portrait of the same 30-year-old blonde woman "
        "wearing a fully buttoned blue courier jacket, futuristic city at night, "
        "detailed 1990s Japanese cel animation",
        "child, teenager, cleavage, nudity, nsfw, different person, dark hair, "
        "deformed face, extra eyes, blurry, low quality, text, watermark",
    ),
}


def _parse_float(value: str, *, label: str, lower: float, upper: float) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be a number") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise argparse.ArgumentTypeError(
            f"{label} must be between {lower} and {upper}"
        )
    return result


def _parse_scale(value: str) -> float:
    return _parse_float(value, label="scale", lower=0.05, upper=1.0)


def _parse_reviewer(value: str) -> dict[str, str]:
    try:
        name, decision = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "reviewer must be NAME=pass or NAME=fail"
        ) from exc
    name = name.strip()
    decision = decision.strip().lower()
    if not name or decision not in {"pass", "fail"}:
        raise argparse.ArgumentTypeError(
            "reviewer must be NAME=pass or NAME=fail"
        )
    return {"name": name, "decision": decision}


def _cuda_memory_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        torch.cuda.synchronize()
        return {
            "available": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except (ImportError, RuntimeError):
        return {"available": False}


def _load_source(path: Path, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageOps

    if not path.is_file():
        raise ValueError(f"source image not found: {path}")
    original = path.read_bytes()
    with Image.open(path) as opened:
        image_format = (opened.format or "").upper()
        if image_format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("source image must be PNG, JPEG, or WebP")
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("animated or multi-page source images are not supported")
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.width * image.height > 16 * 1024 * 1024:
        raise ValueError("source image exceeds the 16 megapixel limit")
    normalized_path = output_dir / "source-normalized.png"
    image.save(normalized_path, format="PNG")
    return image, {
        "filename": path.name,
        "input_format": image_format,
        "input_sha256": hashlib.sha256(original).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized_path.read_bytes()).hexdigest(),
        "width": image.width,
        "height": image.height,
    }


def _image_metrics(
    image: Any, path: Path, *,
    safety_flagged: bool,
    min_entropy: float = 3.0, min_stddev: float = 5.0,
) -> dict[str, Any]:
    from PIL import ImageStat

    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    extrema = rgb.getextrema()
    rgb.save(path, format="PNG")
    data = path.read_bytes()
    channel_stddev = [float(value) for value in stat.stddev]
    entropy = float(rgb.entropy())
    automatic_pass = (
        rgb.width > 0
        and rgb.height > 0
        and len(data) > 1024
        and entropy >= min_entropy
        and max(channel_stddev) >= min_stddev
        and any(low < high for low, high in extrema)
        and not safety_flagged
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "width": rgb.width,
        "height": rgb.height,
        "entropy": entropy,
        "channel_stddev": channel_stddev,
        "extrema": [list(item) for item in extrema],
        "safety_flagged": bool(safety_flagged),
        "automatic_pass": automatic_pass,
    }


def _memory_gate(
    images: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    snapshots = [item["cuda_memory"] for item in images]
    available = bool(
        snapshots
        and before.get("available")
        and after.get("available")
        and all(item.get("available") for item in snapshots)
    )
    if not available:
        return {
            "passed": False,
            "available": False,
            "reason": "CUDA memory snapshots are incomplete",
        }
    allocated = [int(item["allocated_bytes"]) for item in snapshots]
    reserved = [int(item["reserved_bytes"]) for item in snapshots]
    allocated_span = max(allocated) - min(allocated)
    reserved_span = max(reserved) - min(reserved)
    after_unload_growth = max(
        0,
        int(after["allocated_bytes"]) - int(before["allocated_bytes"]),
    )
    return {
        "passed": (
            allocated_span <= STEADY_MEMORY_GROWTH_LIMIT_BYTES
            and reserved_span <= STEADY_MEMORY_GROWTH_LIMIT_BYTES
            and after_unload_growth <= UNLOAD_MEMORY_GROWTH_LIMIT_BYTES
        ),
        "available": True,
        "allocated_span_bytes": allocated_span,
        "reserved_span_bytes": reserved_span,
        "after_unload_growth_bytes": after_unload_growth,
        "steady_growth_limit_bytes": STEADY_MEMORY_GROWTH_LIMIT_BYTES,
        "unload_growth_limit_bytes": UNLOAD_MEMORY_GROWTH_LIMIT_BYTES,
    }


def _full_matrix_reasons(
    *,
    scales: tuple[float, ...],
    seeds: tuple[int, ...],
    expected_seeds: tuple[int, ...],
    preset: Any,
    generation: Any,
    expected_prompt: str,
    expected_negative_prompt: str,
    expected_source_sha256: str,
    actual_source_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if scales != DEFAULT_SCALES:
        reasons.append("scale matrix is not the pinned low/medium/high set")
    if seeds != expected_seeds:
        reasons.append("seed matrix is incomplete")
    if generation.steps != preset.steps:
        reasons.append("steps differ from the pinned preset")
    if generation.prompt != expected_prompt:
        reasons.append("prompt differs from the pinned preset")
    if generation.negative_prompt != expected_negative_prompt:
        reasons.append("negative prompt differs from the pinned preset")
    if not expected_source_sha256:
        reasons.append("source SHA-256 is not pinned")
    elif expected_source_sha256 != actual_source_sha256:
        reasons.append("source SHA-256 does not match")
    return reasons


def _automatic_gate(
    images: list[dict[str, Any]],
    *,
    scales: tuple[float, ...],
    generated_seeds: tuple[int, ...],
    required_valid_per_scale: int,
    memory_passed: bool,
) -> dict[str, Any]:
    required_outputs = len(scales) * len(generated_seeds)
    valid_images = [item for item in images if item["automatic_pass"]]
    valid_per_scale = {
        str(scale): sum(
            1
            for item in valid_images
            if math.isclose(float(item["scale"]), scale, rel_tol=0.0, abs_tol=1e-9)
        )
        for scale in scales
    }
    required_valid_outputs = len(scales) * required_valid_per_scale
    unique_valid_images = len({item["sha256"] for item in valid_images})
    required_unique_images = max(1, required_valid_outputs - 1)
    passed = (
        len(images) == required_outputs
        and all(count >= required_valid_per_scale for count in valid_per_scale.values())
        and unique_valid_images >= required_unique_images
        and memory_passed
    )
    return {
        "passed": passed,
        "outputs": len(images),
        "required_outputs": required_outputs,
        "valid_outputs": len(valid_images),
        "required_valid_outputs": required_valid_outputs,
        "valid_outputs_per_scale": valid_per_scale,
        "required_valid_outputs_per_scale": required_valid_per_scale,
        "unique_images": unique_valid_images,
        "required_unique_images": required_unique_images,
        "safety_flagged_outputs": sum(
            1 for item in images if item["safety_flagged"]
        ),
    }


def _apply_manual_reviews(
    report: dict[str, Any], reviews: list[dict[str, str]]
) -> str:
    existing = list(report.get("manual_gate", {}).get("reviews", []))
    by_name = {item["name"]: item for item in existing if item.get("name")}
    for review in reviews:
        by_name[review["name"]] = review
    normalized = list(by_name.values())
    failures = [item for item in normalized if item["decision"] == "fail"]
    passes = {item["name"] for item in normalized if item["decision"] == "pass"}
    manual_pass = len(passes) >= 2 and not failures
    automatic_pass = bool(report.get("automatic_gate", {}).get("passed"))
    full_matrix = bool(report.get("full_matrix"))
    status = (
        "failed"
        if not automatic_pass or failures
        else "partial_pass"
        if not full_matrix
        else "passed"
        if manual_pass
        else "pending_manual_review"
    )
    report["manual_gate"] = {
        "passed": manual_pass,
        "required_reviewers": 2,
        "reviews": normalized,
        "updated_at": time.time(),
    }
    report["status"] = status
    return status


def _review_existing_report(path: Path, reviews: list[dict[str, str]]) -> int:
    if not reviews:
        raise SystemExit("--review-report requires at least one --reviewer NAME=pass|fail")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "reference" or "automatic_gate" not in report:
        raise SystemExit("--review-report must point to an IP-Adapter reference report")
    status = _apply_manual_reviews(report, reviews)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(path)}, ensure_ascii=False))
    return 0 if status == "passed" else 2 if status in {
        "partial_pass",
        "pending_manual_review",
    } else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-asset-id", default="sd15_90s_retrovers_v1")
    parser.add_argument("--adapter-asset-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--min-entropy", type=float, default=3.0,
        help="automatic gate 最小图像熵阈值（默认 3.0）",
    )
    parser.add_argument(
        "--min-stddev", type=float, default=5.0,
        help="automatic gate 最小通道标准差阈值（默认 5.0）",
    )
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--scale", action="append", type=_parse_scale, default=[])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--review-report", default="")
    parser.add_argument("--reviewer", action="append", type=_parse_reviewer, default=[])
    parser.add_argument("--require-manual", action="store_true")
    args = parser.parse_args()

    if args.review_report:
        return _review_existing_report(
            Path(args.review_report).expanduser().resolve(), args.reviewer
        )
    if not args.source_image:
        parser.error("--source-image is required unless --review-report is used")

    base_spec = get_asset_spec(args.base_asset_id)
    adapter_spec = get_asset_spec(args.adapter_asset_id)
    if adapter_spec.artifact_kind != "sd15_ip_adapter":
        raise SystemExit("adapter asset must have artifact_kind=sd15_ip_adapter")
    preset = get_preset(base_spec.preset_id)
    reference_prompt, reference_negative_prompt = REFERENCE_PROMPTS.get(
        base_spec.asset_id,
        (preset.prompt, preset.negative_prompt),
    )
    base_path = base_spec.target_path(ROOT)
    adapter_path = adapter_spec.target_path(ROOT)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "build" / "sd15-ip-adapter-quality" / args.base_asset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_source_sha256 = args.source_sha256.strip().lower()
    if expected_source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise SystemExit("--source-sha256 must contain exactly 64 hexadecimal characters")
    if args.steps < 0 or args.steps > 100:
        raise SystemExit("--steps must be between 0 and 100")
    gate_seeds = tuple(preset.seeds) + SAFETY_REPLACEMENT_SEEDS
    if args.seed_limit < 0 or args.seed_limit > len(gate_seeds):
        raise SystemExit(f"--seed-limit must be between 0 and {len(gate_seeds)}")
    if args.seed and args.seed_limit:
        raise SystemExit("--seed and --seed-limit cannot be combined")
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in args.seed):
        raise SystemExit("--seed must be between 0 and 4294967295")

    scales = tuple(dict.fromkeys(args.scale or DEFAULT_SCALES))
    seeds = tuple(dict.fromkeys(args.seed)) if args.seed else tuple(
        gate_seeds[: args.seed_limit or len(gate_seeds)]
    )
    base_verification = verify_asset_directory(base_path, args.base_asset_id, full_hash=True)
    adapter_verification = verify_asset_directory(adapter_path, args.adapter_asset_id, full_hash=True)
    if not base_verification["valid"] or not adapter_verification["valid"]:
        print(json.dumps({"base": base_verification, "adapter": adapter_verification}, ensure_ascii=False, indent=2))
        return 1
    source_image, source = _load_source(Path(args.source_image).expanduser().resolve(), output_dir)
    generation = build_sd15_generation_request(
        preset_id=preset.preset_id,
        prompt=args.prompt or reference_prompt,
        negative_prompt=args.negative_prompt or reference_negative_prompt,
        steps=args.steps or preset.steps,
    )
    partial_reasons = _full_matrix_reasons(
        scales=scales,
        seeds=seeds,
        expected_seeds=gate_seeds,
        preset=preset,
        generation=generation,
        expected_prompt=reference_prompt,
        expected_negative_prompt=reference_negative_prompt,
        expected_source_sha256=expected_source_sha256,
        actual_source_sha256=source["input_sha256"],
    )
    full_matrix = not partial_reasons
    inspector = DiffusionArtifactInspector()
    base_artifact = replace(
        inspector.inspect(str(base_path), compute_hash=False),
        sha256=base_verification["artifact_sha256"],
    )
    adapter_artifact = replace(
        inspector.inspect(str(adapter_path), compute_hash=False),
        sha256=adapter_verification["artifact_sha256"],
    )
    engine = SD15Engine(build_sd15_engine_config(args.profile, safety_checker_required=True))
    images: list[dict[str, Any]] = []
    started = time.time()
    memory_before = _cuda_memory_snapshot()
    base_load_seconds = 0.0
    try:
        load_started = time.perf_counter()
        engine.load(str(base_path), artifact=base_artifact)
        base_load_seconds = time.perf_counter() - load_started
        for scale in scales:
            for seed in seeds:
                request = SD15EditRequest(
                    mode="reference",
                    source_blob_id="quality-gate-source",
                    prompt=generation.prompt,
                    negative_prompt=generation.negative_prompt,
                    seed=seed,
                    width=generation.width,
                    height=generation.height,
                    steps=generation.steps,
                    guidance_scale=generation.guidance_scale,
                    strength=0.75,
                    scheduler=generation.scheduler,
                    edit_adapter_id=args.adapter_asset_id,
                    ip_adapter_scale=scale,
                )
                edit_started = time.perf_counter()
                result = engine.edit(request, image=source_image, adapter=adapter_artifact)
                edit_wall_seconds = time.perf_counter() - edit_started
                output_path = output_dir / f"scale-{scale:.2f}".replace(".", "p") / f"seed-{seed}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                metrics = _image_metrics(
                    result.image,
                    output_path,
                    safety_flagged=bool(result.metadata.get("safety_flagged", False)),
                    min_entropy=args.min_entropy,
                    min_stddev=args.min_stddev,
                )
                metrics.update(
                    {
                        "seed": seed,
                        "scale": scale,
                        "elapsed_seconds": result.elapsed_seconds,
                        "wall_seconds": edit_wall_seconds,
                        "adapter_setup_seconds": max(
                            0.0,
                            edit_wall_seconds - result.elapsed_seconds,
                        ),
                        "scheduler": result.metadata.get("scheduler"),
                        "ip_adapter_sha256": result.metadata.get("ip_adapter_sha256"),
                        "cuda_memory": _cuda_memory_snapshot(),
                    }
                )
                images.append(metrics)
                print(
                    json.dumps(
                        {
                            "completed": len(images),
                            "total": len(scales) * len(seeds),
                            "seed": seed,
                            "scale": scale,
                            "elapsed_seconds": result.elapsed_seconds,
                            "safety_flagged": metrics["safety_flagged"],
                            "automatic_pass": metrics["automatic_pass"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        engine.unload()
    memory_after = _cuda_memory_snapshot()
    memory_gate = _memory_gate(images, memory_before, memory_after)
    automatic_gate = _automatic_gate(
        images,
        scales=scales,
        generated_seeds=seeds,
        required_valid_per_scale=len(preset.seeds),
        memory_passed=memory_gate["passed"],
    )
    automatic_pass = bool(automatic_gate["passed"])
    manual_failures = [item for item in args.reviewer if item["decision"] == "fail"]
    manual_passes = {item["name"] for item in args.reviewer if item["decision"] == "pass"}
    manual_pass = len(manual_passes) >= 2 and not manual_failures
    status = (
        "failed"
        if not automatic_pass or manual_failures
        else "partial_pass"
        if not full_matrix
        else "passed"
        if manual_pass
        else "pending_manual_review"
    )
    report = {
        "schema_version": 1,
        "mode": "reference",
        "status": status,
        "full_matrix": full_matrix,
        "partial_reasons": partial_reasons,
        "base_asset_id": base_spec.asset_id,
        "base_artifact_id": base_spec.artifact_id,
        "base_repo_id": base_spec.repo_id,
        "base_revision": base_spec.revision,
        "adapter_asset_id": adapter_spec.asset_id,
        "adapter_artifact_id": adapter_spec.artifact_id,
        "adapter_repo_id": adapter_spec.repo_id,
        "adapter_revision": adapter_spec.revision,
        "source": source,
        "parameters": {
            "prompt": generation.prompt,
            "negative_prompt": generation.negative_prompt,
            "width": generation.width,
            "height": generation.height,
            "steps": generation.steps,
            "guidance_scale": generation.guidance_scale,
            "scheduler": generation.scheduler,
            "seeds": list(seeds),
            "scales": list(scales),
        },
        "asset_verification": {
            "base": {
                "valid": base_verification["valid"],
                "artifact_sha256": base_verification["artifact_sha256"],
                "integrity_scope": base_verification["integrity_scope"],
            },
            "adapter": {
                "valid": adapter_verification["valid"],
                "artifact_sha256": adapter_verification["artifact_sha256"],
                "integrity_scope": adapter_verification["integrity_scope"],
            },
        },
        "automatic_gate": automatic_gate,
        "memory_gate": memory_gate,
        "manual_gate": {
            "passed": manual_pass,
            "required_reviewers": 2,
            "reviews": args.reviewer,
        },
        "cuda_memory_before": memory_before,
        "cuda_memory_after_unload": memory_after,
        "base_load_seconds": base_load_seconds,
        "elapsed_seconds": time.time() - started,
        "images": images,
    }
    report_path = output_dir / "quality-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path)}, ensure_ascii=False))
    if not automatic_pass or manual_failures:
        return 1
    if args.require_manual and not manual_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
