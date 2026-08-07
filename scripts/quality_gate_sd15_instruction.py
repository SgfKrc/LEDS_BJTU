"""Run the pinned InstructPix2Pix quality, identity, and memory gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion.artifacts import DiffusionArtifactInspector  # noqa: E402
from diffusion.assets import get_asset_spec, verify_asset_directory  # noqa: E402
from diffusion.sd15_engine import SD15Engine  # noqa: E402
from diffusion.service import SD15EditRequest, build_sd15_engine_config  # noqa: E402


DEFAULT_SOURCE = ROOT / "logs" / "sd15" / "sd15_original_v1_seed19950101.png"
DEFAULT_SOURCE_SHA256 = "a6fd131b5008b77f3c39f01d4a073529cf8225a8a204997d4f01de9217e93264"
DEFAULT_STEPS = 20
DEFAULT_IMAGE_GUIDANCE_SCALE = 1.0
DEFAULT_CASES = (
    ("red-trees", "turn the trees outside red", 19950101),
    ("winter", "make it a snowy winter day", 19950102),
    ("watercolor", "turn the room into a watercolor painting", 19950103),
    ("sunset", "make it a warm sunset scene", 19950104),
    ("night", "make it a clear moonlit night", 19950105),
    ("marble-floor", "turn the wooden floor into white marble", 19950106),
    ("white-frames", "turn the dark window frames white", 19950107),
    ("pencil-sketch", "turn the image into a pencil sketch", 19950108),
    ("fog", "add dense fog over the mountains outside", 19950109),
    ("spring", "make the landscape outside a bright spring day", 19950110),
)
MIN_MAE = 4.0
MAX_MAE = 120.0
MIN_ENTROPY = 3.0
MIN_EDGE_CORRELATION = 0.05
MAX_STEADY_ALLOCATED_SPAN = 64 * 1024 * 1024
MAX_AFTER_UNLOAD_ALLOCATED = 64 * 1024 * 1024


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise argparse.ArgumentTypeError("reviewer must be NAME=pass or NAME=fail")
    return {"name": name, "decision": decision}


def _image_metrics(source: Any, result: Any) -> dict[str, Any]:
    import numpy as np
    from PIL import ImageChops, ImageFilter, ImageStat

    source_rgb = source.convert("RGB")
    result_rgb = result.convert("RGB")
    difference = ImageChops.difference(source_rgb, result_rgb)
    mae = float(sum(ImageStat.Stat(difference).mean) / 3)
    entropy = float(result_rgb.entropy())
    channel_stddev = [float(value) for value in ImageStat.Stat(result_rgb).stddev]
    source_edges = np.asarray(
        source_rgb.convert("L").filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    ).reshape(-1)
    result_edges = np.asarray(
        result_rgb.convert("L").filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    ).reshape(-1)
    if float(source_edges.std()) <= 1e-6 or float(result_edges.std()) <= 1e-6:
        edge_correlation = 0.0
    else:
        edge_correlation = float(np.corrcoef(source_edges, result_edges)[0, 1])
    if not math.isfinite(edge_correlation):
        edge_correlation = 0.0
    automatic_pass = (
        MIN_MAE <= mae <= MAX_MAE
        and entropy >= MIN_ENTROPY
        and max(channel_stddev) >= 5.0
        and edge_correlation >= MIN_EDGE_CORRELATION
    )
    return {
        "mae": mae,
        "entropy": entropy,
        "channel_stddev": channel_stddev,
        "edge_correlation": edge_correlation,
        "automatic_pass": automatic_pass,
    }


def _apply_manual_reviews(
    report: dict[str, Any],
    reviews: list[dict[str, str]],
) -> str:
    existing = list(report.get("manual_gate", {}).get("reviews", []))
    by_name = {
        str(item["name"]).strip().casefold(): item
        for item in existing
        if item.get("name")
    }
    for review in reviews:
        recorded: dict[str, Any] = dict(review)
        recorded["reviewed_at"] = time.time()
        by_name[review["name"].strip().casefold()] = recorded
    normalized = list(by_name.values())
    failures = [item for item in normalized if item["decision"] == "fail"]
    passes = {
        str(item["name"]).strip().casefold()
        for item in normalized
        if item["decision"] == "pass"
    }
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


def _review_report(path: Path, reviews: list[dict[str, str]]) -> int:
    if not reviews:
        raise SystemExit("--review-report requires --reviewer NAME=pass|fail")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "instruction":
        raise SystemExit("--review-report must point to an instruction report")
    status = _apply_manual_reviews(report, reviews)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "report": str(path)}, ensure_ascii=False))
    return 0 if status == "passed" else 2 if status == "pending_manual_review" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-asset-id", default="sd15_original_v1")
    parser.add_argument(
        "--instruction-asset-id",
        default="sd15_instruct_pix2pix_v1",
    )
    parser.add_argument("--source-image", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument(
        "--output-dir",
        default="build/sd15-instruction-quality/full-original",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=DEFAULT_IMAGE_GUIDANCE_SCALE,
    )
    parser.add_argument("--case-limit", type=int, default=len(DEFAULT_CASES))
    parser.add_argument("--reviewer", action="append", type=_parse_reviewer, default=[])
    parser.add_argument("--review-report", default="")
    args = parser.parse_args()

    if args.review_report:
        return _review_report(Path(args.review_report).expanduser().resolve(), args.reviewer)
    if args.reviewer:
        parser.error("--reviewer requires --review-report so outputs are reviewed after generation")
    if not 1 <= args.case_limit <= len(DEFAULT_CASES):
        raise SystemExit(f"--case-limit must be between 1 and {len(DEFAULT_CASES)}")
    if not math.isfinite(args.image_guidance_scale) or not 0 <= args.image_guidance_scale <= 4:
        raise SystemExit("--image-guidance-scale must be between 0 and 4")

    import torch
    from PIL import Image, ImageOps

    if not torch.cuda.is_available():
        raise SystemExit("the instruction quality gate requires CUDA")
    source_path = Path(args.source_image).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"source image not found: {source_path}")
    source_sha256 = _sha256(source_path)
    if source_sha256.lower() != args.source_sha256.lower():
        raise SystemExit(
            f"source SHA mismatch: expected {args.source_sha256}, got {source_sha256}"
        )

    base_spec = get_asset_spec(args.base_asset_id)
    instruction_spec = get_asset_spec(args.instruction_asset_id)
    if base_spec.artifact_kind != "sd15_pipeline":
        raise SystemExit("base asset must be a standard SD1.5 pipeline")
    if instruction_spec.artifact_kind != "sd15_instruction_pipeline":
        raise SystemExit("instruction asset must be an InstructPix2Pix pipeline")
    base_path = base_spec.target_path(ROOT).resolve()
    instruction_path = instruction_spec.target_path(ROOT).resolve()
    base_report = verify_asset_directory(base_path, args.base_asset_id)
    instruction_report = verify_asset_directory(
        instruction_path,
        args.instruction_asset_id,
    )
    if not base_report["valid"] or not instruction_report["valid"]:
        raise SystemExit("base or instruction asset failed pinned verification")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB").resize(
            (512, 512),
            Image.Resampling.LANCZOS,
        )

    inspector = DiffusionArtifactInspector()
    base_artifact = replace(
        inspector.inspect(str(base_path)),
        sha256=base_report["artifact_sha256"],
    )
    instruction_artifact = replace(
        inspector.inspect(str(instruction_path)),
        sha256=instruction_report["artifact_sha256"],
    )
    cases = DEFAULT_CASES[: args.case_limit]
    full_matrix = (
        cases == DEFAULT_CASES
        and args.steps == DEFAULT_STEPS
        and args.image_guidance_scale == DEFAULT_IMAGE_GUIDANCE_SCALE
        and source_sha256 == DEFAULT_SOURCE_SHA256
    )
    engine = SD15Engine(build_sd15_engine_config("balanced"))
    rows: list[dict[str, Any]] = []
    memory_samples: list[dict[str, int]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        engine.load(str(base_path), artifact=base_artifact)
        base_load_seconds = time.perf_counter() - load_started
        for slug, instruction, seed in cases:
            request = SD15EditRequest(
                mode="instruction",
                source_blob_id="quality-source",
                prompt=instruction,
                instruction=instruction,
                edit_adapter_id=args.instruction_asset_id,
                image_guidance_scale=args.image_guidance_scale,
                seed=seed,
                width=512,
                height=512,
                steps=args.steps,
                guidance_scale=7.5,
            )
            result = engine.edit(
                request,
                image=source,
                adapter=instruction_artifact,
            )
            result_path = output_dir / f"{slug}-seed-{seed}.png"
            result.image.save(result_path)
            metrics = _image_metrics(source, result.image)
            safety_flagged = bool(result.metadata.get("safety_flagged"))
            rows.append(
                {
                    "slug": slug,
                    "instruction": instruction,
                    "seed": seed,
                    "result_path": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "elapsed_seconds": result.elapsed_seconds,
                    "safety_flagged": safety_flagged,
                    "metrics": metrics,
                    "automatic_pass": not safety_flagged and metrics["automatic_pass"],
                }
            )
            memory_samples.append(
                {
                    "allocated": int(torch.cuda.memory_allocated()),
                    "reserved": int(torch.cuda.memory_reserved()),
                }
            )
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        engine.unload()

    after_unload = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    steady = memory_samples[1:] if len(memory_samples) > 1 else memory_samples
    allocated_span = (
        max(item["allocated"] for item in steady)
        - min(item["allocated"] for item in steady)
        if steady
        else 0
    )
    memory_pass = (
        allocated_span <= MAX_STEADY_ALLOCATED_SPAN
        and after_unload["allocated"] <= MAX_AFTER_UNLOAD_ALLOCATED
    )
    unique_results = len({row["result_sha256"] for row in rows}) == len(rows)
    automatic_pass = (
        len(rows) == len(cases)
        and all(row["automatic_pass"] for row in rows)
        and unique_results
        and memory_pass
    )
    report = {
        "schema_version": 1,
        "mode": "instruction",
        "status": "failed",
        "full_matrix": full_matrix,
        "source_image": str(source_path),
        "source_sha256": source_sha256,
        "base_asset_id": args.base_asset_id,
        "base_artifact_sha256": base_report["artifact_sha256"],
        "instruction_asset_id": args.instruction_asset_id,
        "instruction_artifact_sha256": instruction_report["artifact_sha256"],
        "route": "StableDiffusionInstructPix2PixPipeline",
        "steps": args.steps,
        "guidance_scale": 7.5,
        "image_guidance_scale": args.image_guidance_scale,
        "base_load_seconds": base_load_seconds,
        "total_seconds": time.perf_counter() - started,
        "rows": rows,
        "automatic_gate": {
            "passed": automatic_pass,
            "unique_results": unique_results,
            "memory_passed": memory_pass,
            "thresholds": {
                "min_mae": MIN_MAE,
                "max_mae": MAX_MAE,
                "min_entropy": MIN_ENTROPY,
                "min_edge_correlation": MIN_EDGE_CORRELATION,
            },
        },
        "memory": {
            "samples": memory_samples,
            "allocated_span": allocated_span,
            "peak_reserved": peak_reserved,
            "after_unload": after_unload,
        },
        "manual_gate": {
            "passed": False,
            "required_reviewers": 2,
            "reviews": [],
        },
    }
    status = _apply_manual_reviews(report, args.reviewer)
    report_path = output_dir / "quality-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": status, "report": str(report_path)},
            ensure_ascii=False,
        )
    )
    return 0 if status == "passed" else 2 if status in {
        "pending_manual_review",
        "partial_pass",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
