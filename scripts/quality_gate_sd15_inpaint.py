"""Run the reproducible SD 1.5 inpaint mask-semantics and memory gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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


DEFAULT_SEEDS = (
    19950101,
    19950102,
    19950103,
    19950104,
    19950105,
    19950106,
    19950107,
    19950108,
    19950109,
    19950110,
)
MASK_SEQUENCE = (
    "black",
    "local",
    "white",
    "local",
    "black",
    "local",
    "white",
    "local",
    "black",
    "white",
)
DEFAULT_STEPS = 20
MAX_BLACK_MAE = 10.0
MAX_LOCAL_OUTSIDE_MAE = 10.0
MIN_LOCAL_INSIDE_MARGIN = 10.0
MIN_WHITE_MAE = 20.0
MAX_STEADY_ALLOCATED_SPAN = 64 * 1024 * 1024
MAX_AFTER_UNLOAD_ALLOCATED = 64 * 1024 * 1024


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _make_mask(kind: str, size: tuple[int, int]) -> Any:
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    if kind == "local":
        width, height = size
        ImageDraw.Draw(mask).ellipse(
            (
                int(width * 0.28),
                int(height * 0.20),
                int(width * 0.72),
                int(height * 0.74),
            ),
            fill=255,
        )
    elif kind == "white":
        mask.paste(255, (0, 0, *size))
    elif kind != "black":
        raise ValueError(f"unknown mask kind: {kind}")
    return mask


def _measure_mask_semantics(source: Any, result: Any, mask: Any) -> dict[str, float]:
    from PIL import ImageChops, ImageOps, ImageStat

    difference = ImageChops.difference(source.convert("RGB"), result.convert("RGB"))
    overall = sum(ImageStat.Stat(difference).mean) / 3
    extrema = mask.getextrema()
    inside = (
        sum(ImageStat.Stat(difference, mask=mask).mean) / 3
        if extrema[1] > 0
        else 0.0
    )
    inverse = ImageOps.invert(mask)
    outside = (
        sum(ImageStat.Stat(difference, mask=inverse).mean) / 3
        if extrema[0] < 255
        else 0.0
    )
    return {
        "overall_mae": float(overall),
        "inside_mae": float(inside),
        "outside_mae": float(outside),
        "inside_margin": float(inside - outside),
    }


def _semantic_pass(kind: str, metrics: dict[str, float]) -> bool:
    if kind == "black":
        return metrics["overall_mae"] <= MAX_BLACK_MAE
    if kind == "local":
        return (
            metrics["outside_mae"] <= MAX_LOCAL_OUTSIDE_MAE
            and metrics["inside_margin"] >= MIN_LOCAL_INSIDE_MARGIN
        )
    return metrics["overall_mae"] >= MIN_WHITE_MAE


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-asset-id", default="sd15_original_v1")
    parser.add_argument("--inpaint-asset-id", default="sd15_inpaint_v1")
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", default="build/sd15-inpaint-quality/full-original")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated seeds; changing the default matrix produces partial_pass",
    )
    parser.add_argument(
        "--prompt",
        default="a room with a vivid stained glass window overlooking mountains, photorealistic",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, distorted, low quality, malformed architecture",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image

    source_path = Path(args.source_image).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"source image not found: {source_path}")
    source_sha256 = _sha256(source_path)
    if source_sha256.lower() != args.source_sha256.lower():
        raise SystemExit(
            f"source SHA mismatch: expected {args.source_sha256}, got {source_sha256}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("the full inpaint gate requires an available CUDA device")

    base_spec = get_asset_spec(args.base_asset_id)
    inpaint_spec = get_asset_spec(args.inpaint_asset_id)
    if base_spec.artifact_kind != "sd15_pipeline":
        raise SystemExit("base asset must be a standard SD1.5 pipeline")
    if inpaint_spec.artifact_kind != "sd15_inpaint_pipeline":
        raise SystemExit("inpaint asset must be a dedicated SD1.5 inpaint pipeline")
    base_path = base_spec.target_path(ROOT).resolve()
    inpaint_path = inpaint_spec.target_path(ROOT).resolve()
    base_report = verify_asset_directory(base_path, args.base_asset_id)
    inpaint_report = verify_asset_directory(inpaint_path, args.inpaint_asset_id)
    if not base_report["valid"] or not inpaint_report["valid"]:
        raise SystemExit("base or inpaint asset failed pinned verification")

    seeds = tuple(args.seeds)
    full_matrix = seeds == DEFAULT_SEEDS and args.steps == DEFAULT_STEPS
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    inspector = DiffusionArtifactInspector()
    base_artifact = replace(
        inspector.inspect(str(base_path)),
        sha256=base_report["artifact_sha256"],
    )
    inpaint_artifact = replace(
        inspector.inspect(str(inpaint_path)),
        sha256=inpaint_report["artifact_sha256"],
    )
    engine = SD15Engine(build_sd15_engine_config("balanced"))
    rows: list[dict[str, Any]] = []
    memory_samples: list[dict[str, int]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        load_started = time.perf_counter()
        engine.load(str(base_path), artifact=base_artifact)
        base_load_seconds = time.perf_counter() - load_started
        for index, seed in enumerate(seeds):
            kind = MASK_SEQUENCE[index % len(MASK_SEQUENCE)]
            mask = _make_mask(kind, source.size)
            request = SD15EditRequest(
                mode="inpaint",
                source_blob_id="quality-source",
                mask_blob_id=f"quality-mask-{index}",
                edit_adapter_id=args.inpaint_asset_id,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=seed,
                width=512,
                height=512,
                steps=args.steps,
                guidance_scale=7.5,
                strength=1.0,
            )
            result = engine.edit(
                request,
                image=source,
                mask=mask,
                adapter=inpaint_artifact,
            )
            result_path = output_dir / f"{index:02d}-{kind}-seed-{seed}.png"
            mask_path = output_dir / f"{index:02d}-{kind}-mask.png"
            result.image.save(result_path)
            mask.save(mask_path)
            metrics = _measure_mask_semantics(source, result.image, mask)
            safety_flagged = bool(result.metadata.get("safety_flagged"))
            rows.append(
                {
                    "index": index,
                    "seed": seed,
                    "mask_kind": kind,
                    "result_path": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "mask_path": str(mask_path),
                    "mask_sha256": _sha256(mask_path),
                    "elapsed_seconds": result.elapsed_seconds,
                    "safety_flagged": safety_flagged,
                    "semantic_metrics": metrics,
                    "semantic_pass": (
                        not safety_flagged and _semantic_pass(kind, metrics)
                    ),
                }
            )
            memory_samples.append(
                {
                    "allocated": int(torch.cuda.memory_allocated()),
                    "reserved": int(torch.cuda.memory_reserved()),
                }
            )
        before_unload = memory_samples[-1] if memory_samples else {
            "allocated": 0,
            "reserved": 0,
        }
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        engine.unload()

    after_unload = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    steady_samples = memory_samples[1:] if len(memory_samples) > 1 else memory_samples
    allocated_span = (
        max(item["allocated"] for item in steady_samples)
        - min(item["allocated"] for item in steady_samples)
        if steady_samples
        else 0
    )
    memory_pass = (
        allocated_span <= MAX_STEADY_ALLOCATED_SPAN
        and after_unload["allocated"] <= MAX_AFTER_UNLOAD_ALLOCATED
    )
    automatic_pass = (
        len(rows) == len(seeds)
        and all(row["semantic_pass"] for row in rows)
        and len({row["result_sha256"] for row in rows}) == len(rows)
        and memory_pass
    )
    status = (
        "passed"
        if automatic_pass and full_matrix
        else "partial_pass"
        if automatic_pass
        else "failed"
    )
    report = {
        "status": status,
        "full_matrix": full_matrix,
        "automatic_pass": automatic_pass,
        "base_asset_id": args.base_asset_id,
        "base_artifact_sha256": base_report["artifact_sha256"],
        "inpaint_asset_id": args.inpaint_asset_id,
        "inpaint_artifact_sha256": inpaint_report["artifact_sha256"],
        "source_image": str(source_path),
        "source_sha256": source_sha256,
        "steps": args.steps,
        "seeds": list(seeds),
        "mask_sequence": [row["mask_kind"] for row in rows],
        "thresholds": {
            "max_black_mae": MAX_BLACK_MAE,
            "max_local_outside_mae": MAX_LOCAL_OUTSIDE_MAE,
            "min_local_inside_margin": MIN_LOCAL_INSIDE_MARGIN,
            "min_white_mae": MIN_WHITE_MAE,
            "max_steady_allocated_span": MAX_STEADY_ALLOCATED_SPAN,
            "max_after_unload_allocated": MAX_AFTER_UNLOAD_ALLOCATED,
        },
        "rows": rows,
        "memory": {
            "samples": memory_samples,
            "allocated_span": allocated_span,
            "before_unload": before_unload,
            "after_unload": after_unload,
            "peak_reserved": peak_reserved,
            "pass": memory_pass,
        },
        "base_load_seconds": base_load_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "quality-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"quality report: {report_path}")
    return 0 if status in {"passed", "partial_pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
