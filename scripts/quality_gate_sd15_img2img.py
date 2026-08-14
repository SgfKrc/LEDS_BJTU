"""Run the deterministic SD 1.5 img2img quality gate.

The full gate uses every pinned preset seed and low/medium/high denoising
strengths. Use ``--seed-limit`` and a single ``--strength`` only for a quick
runtime smoke; such a run is recorded as partial and cannot complete SD-N5.1.
Keep ``steps * strength >= 1`` so Diffusers has at least one denoising step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import (  # noqa: E402
    SD15EditRequest,
    SD15Engine,
    build_sd15_engine_config,
    build_sd15_generation_request,
    get_asset_spec,
    get_preset,
    verify_asset_directory,
)


DEFAULT_STRENGTHS = (0.25, 0.55, 0.85)
STEADY_MEMORY_GROWTH_LIMIT_BYTES = 256 * 1024 * 1024
UNLOAD_MEMORY_GROWTH_LIMIT_BYTES = 64 * 1024 * 1024


def _parse_strength(value: str) -> float:
    try:
        strength = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("strength must be a number") from exc
    if not math.isfinite(strength) or not 0.05 <= strength <= 1.0:
        raise argparse.ArgumentTypeError("strength must be between 0.05 and 1.0")
    return strength


def _parse_reviewer(value: str) -> dict[str, str]:
    try:
        name, decision = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reviewer must be NAME=pass or NAME=fail") from exc
    name = name.strip()
    decision = decision.strip().lower()
    if not name or decision not in {"pass", "fail"}:
        raise argparse.ArgumentTypeError("reviewer must be NAME=pass or NAME=fail")
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
    normalized = normalized_path.read_bytes()
    return image, {
        "filename": path.name,
        "input_format": image_format,
        "input_sha256": hashlib.sha256(original).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "width": image.width,
        "height": image.height,
    }


def _image_metrics(
    image: Any, path: Path, *,
    min_entropy: float = 3.0, min_stddev: float = 5.0,
) -> dict[str, Any]:
    from PIL import ImageStat

    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    extrema = rgb.getextrema()
    rgb.save(path, format="PNG")
    data = path.read_bytes()
    channel_stddev = [float(value) for value in stat.stddev]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "width": rgb.width,
        "height": rgb.height,
        "entropy": float(rgb.entropy()),
        "channel_stddev": channel_stddev,
        "extrema": [list(item) for item in extrema],
        "automatic_pass": (
            rgb.width > 0
            and rgb.height > 0
            and float(rgb.entropy()) >= min_entropy
            and max(channel_stddev) >= min_stddev
            and any(low < high for low, high in extrema)
        ),
    }


def _strength_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


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
    strengths: tuple[float, ...],
    seeds: tuple[int, ...],
    preset: Any,
    generation: Any,
    expected_source_sha256: str,
    actual_source_sha256: str,
) -> list[str]:
    reasons = []
    if strengths != DEFAULT_STRENGTHS:
        reasons.append("strength matrix is not the pinned low/medium/high set")
    if seeds != tuple(preset.seeds):
        reasons.append("seed matrix is incomplete")
    if generation.steps != preset.steps:
        reasons.append("steps differ from the pinned preset")
    if generation.prompt != preset.prompt:
        reasons.append("prompt differs from the pinned preset")
    if generation.negative_prompt != preset.negative_prompt:
        reasons.append("negative prompt differs from the pinned preset")
    if not expected_source_sha256:
        reasons.append("source SHA-256 is not pinned")
    elif expected_source_sha256 != actual_source_sha256:
        reasons.append("source SHA-256 does not match")
    return reasons


def _apply_manual_reviews(
    report: dict[str, Any],
    reviews: list[dict[str, str]],
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
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read quality report: {path}: {exc}") from exc
    if report.get("mode") != "img2img" or "automatic_gate" not in report:
        raise SystemExit("--review-report must point to an img2img quality report")
    status = _apply_manual_reviews(report, reviews)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(path)}, ensure_ascii=False))
    return 0 if status == "passed" else 2 if status in {
        "partial_pass",
        "pending_manual_review",
    } else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", default="sd15_original_v1")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--source-image", default="")
    parser.add_argument(
        "--source-sha256",
        default="",
        help="pin the source image bytes; required for a full-matrix result",
    )
    parser.add_argument(
        "--review-report",
        default="",
        help="append manual reviewer decisions without rerunning inference",
    )
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
    parser.add_argument("--strength", action="append", type=_parse_strength, default=[])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--reviewer", action="append", type=_parse_reviewer, default=[])
    parser.add_argument("--require-manual", action="store_true")
    args = parser.parse_args()

    if args.review_report:
        return _review_existing_report(
            Path(args.review_report).expanduser().resolve(),
            args.reviewer,
        )
    if not args.source_image:
        parser.error("--source-image is required unless --review-report is used")

    spec = get_asset_spec(args.asset_id)
    preset = get_preset(spec.preset_id)
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else spec.target_path(ROOT)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "build" / "sd15-img2img-quality" / args.asset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.steps < 0 or args.steps > 100:
        raise SystemExit("--steps must be between 0 and 100")
    if args.seed_limit < 0 or args.seed_limit > len(preset.seeds):
        raise SystemExit(f"--seed-limit must be between 0 and {len(preset.seeds)}")
    expected_source_sha256 = args.source_sha256.strip().lower()
    if expected_source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise SystemExit("--source-sha256 must contain exactly 64 hexadecimal characters")

    strengths = tuple(dict.fromkeys(args.strength or DEFAULT_STRENGTHS))
    seeds = tuple(preset.seeds[: args.seed_limit or len(preset.seeds)])

    verification = verify_asset_directory(model_path, args.asset_id, full_hash=True)
    if not verification["valid"]:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 1

    source_image, source = _load_source(
        Path(args.source_image).expanduser().resolve(),
        output_dir,
    )
    generation = build_sd15_generation_request(
        preset_id=preset.preset_id,
        prompt=args.prompt or None,
        negative_prompt=args.negative_prompt or None,
        steps=args.steps or preset.steps,
    )
    partial_reasons = _full_matrix_reasons(
        strengths=strengths,
        seeds=seeds,
        preset=preset,
        generation=generation,
        expected_source_sha256=expected_source_sha256,
        actual_source_sha256=source["input_sha256"],
    )
    full_matrix = not partial_reasons

    engine = SD15Engine(
        build_sd15_engine_config(args.profile, safety_checker_required=True)
    )
    images: list[dict[str, Any]] = []
    started = time.time()
    memory_before = _cuda_memory_snapshot()
    try:
        engine.load(str(model_path))
        for strength in strengths:
            for seed in seeds:
                request = SD15EditRequest(
                    mode="img2img",
                    source_blob_id="quality-gate-source",
                    prompt=generation.prompt,
                    negative_prompt=generation.negative_prompt,
                    seed=seed,
                    width=generation.width,
                    height=generation.height,
                    steps=generation.steps,
                    guidance_scale=generation.guidance_scale,
                    strength=strength,
                    scheduler=generation.scheduler,
                )
                result = engine.edit(request, image=source_image)
                output_path = output_dir / (
                    f"strength-{_strength_label(strength)}-seed-{seed}.png"
                )
                metrics = _image_metrics(
                    result.image, output_path,
                    min_entropy=args.min_entropy, min_stddev=args.min_stddev,
                )
                metrics.update(
                    {
                        "seed": seed,
                        "strength": strength,
                        "elapsed_seconds": result.elapsed_seconds,
                        "scheduler": result.metadata.get("scheduler"),
                        "safety_flagged": result.metadata.get("safety_flagged", False),
                        "cuda_memory": _cuda_memory_snapshot(),
                    }
                )
                metrics["automatic_pass"] = (
                    metrics["automatic_pass"] and not metrics["safety_flagged"]
                )
                images.append(metrics)
                print(
                    json.dumps(
                        {
                            "completed": len(images),
                            "total": len(strengths) * len(seeds),
                            "seed": seed,
                            "strength": strength,
                            "elapsed_seconds": result.elapsed_seconds,
                            "automatic_pass": metrics["automatic_pass"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        engine.unload()
    memory_after = _cuda_memory_snapshot()

    unique_hashes = len({item["sha256"] for item in images})
    required_outputs = len(strengths) * len(seeds)
    memory_gate = _memory_gate(images, memory_before, memory_after)
    automatic_pass = (
        len(images) == required_outputs
        and all(item["automatic_pass"] for item in images)
        and unique_hashes >= max(1, required_outputs - 1)
        and memory_gate["passed"]
    )
    manual_failures = [item for item in args.reviewer if item["decision"] == "fail"]
    manual_passes = {
        item["name"] for item in args.reviewer if item["decision"] == "pass"
    }
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
        "mode": "img2img",
        "status": status,
        "full_matrix": full_matrix,
        "partial_reasons": partial_reasons,
        "asset_id": spec.asset_id,
        "artifact_id": spec.artifact_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "preset_id": preset.preset_id,
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
            "strengths": list(strengths),
        },
        "asset_verification": {
            "valid": verification["valid"],
            "integrity_scope": verification["integrity_scope"],
        },
        "automatic_gate": {
            "passed": automatic_pass,
            "outputs": len(images),
            "required_outputs": required_outputs,
            "unique_images": unique_hashes,
            "required_unique_images": max(1, required_outputs - 1),
        },
        "memory_gate": memory_gate,
        "manual_gate": {
            "passed": manual_pass,
            "required_reviewers": 2,
            "reviews": args.reviewer,
        },
        "cuda_memory_before": memory_before,
        "cuda_memory_after_unload": memory_after,
        "elapsed_seconds": time.time() - started,
        "images": images,
    }
    report_path = output_dir / "quality-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(report_path)}, ensure_ascii=False))
    if not automatic_pass or manual_failures:
        return 1
    if args.require_manual and not manual_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
