"""Run the deterministic SD 1.5 ten-seed quality gate.

The automatic gate catches corrupt, blank, duplicate, and malformed output.
Visual style and safety still require two human reviewers; their decisions are
recorded in the same JSON report rather than inferred from pixel statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import (  # noqa: E402
    SD15Engine,
    build_sd15_engine_config,
    build_sd15_generation_request,
    get_asset_spec,
    get_preset,
    verify_asset_directory,
)


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


def _apply_manual_reviews(
    report: dict[str, Any],
    reviews: list[dict[str, str]],
) -> str:
    """登记人工审核结论（不重跑推理）；两名不同 pass 且无 fail → passed。"""
    existing = list(report.get("manual_gate", {}).get("reviews", []))
    by_name = {item["name"]: item for item in existing if item.get("name")}
    for review in reviews:
        by_name[review["name"]] = review
    normalized = list(by_name.values())
    failures = [item for item in normalized if item["decision"] == "fail"]
    passes = {item["name"] for item in normalized if item["decision"] == "pass"}
    manual_pass = len(passes) >= 2 and not failures
    automatic_pass = bool(report.get("automatic_gate", {}).get("passed"))
    status = (
        "failed"
        if not automatic_pass or failures
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


def _image_metrics(image: Any, path: Path) -> dict[str, Any]:
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
        "mode": rgb.mode,
        "width": rgb.width,
        "height": rgb.height,
        "entropy": float(rgb.entropy()),
        "channel_stddev": channel_stddev,
        "extrema": [list(item) for item in extrema],
        "automatic_pass": (
            rgb.width > 0
            and rgb.height > 0
            and float(rgb.entropy()) >= 3.0
            and max(channel_stddev) >= 5.0
            and any(low < high for low, high in extrema)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pinned SD 1.5 quality gate")
    parser.add_argument("--asset-id", default="sd15_90s_retrovers_v1")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--reviewer", action="append", type=_parse_reviewer, default=[])
    parser.add_argument(
        "--review-report",
        default="",
        help="append manual reviewer decisions without rerunning inference",
    )
    parser.add_argument("--require-manual", action="store_true")
    args = parser.parse_args()

    # 登记模式：不重跑推理，只把审核结论写入既有报告
    if args.review_report:
        if not args.reviewer:
            raise SystemExit("--review-report requires at least one --reviewer NAME=pass|fail")
        report_path = Path(args.review_report).expanduser().resolve()
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read quality report: {report_path}: {exc}") from exc
        if "automatic_gate" not in report:
            raise SystemExit("--review-report must point to an SD 1.5 quality report")
        status = _apply_manual_reviews(report, args.reviewer)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": status, "report": str(report_path)}, ensure_ascii=False))
        return 0 if status == "passed" else 2 if status == "pending_manual_review" else 1

    spec = get_asset_spec(args.asset_id)
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else spec.target_path(ROOT)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "build" / "sd15-quality" / args.asset_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    verification = verify_asset_directory(model_path, args.asset_id, full_hash=True)
    if not verification["valid"]:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 1

    preset = get_preset(spec.preset_id)
    expected_names = {f"seed-{seed}.png" for seed in preset.seeds}
    for stale in output_dir.glob("seed-*.png"):
        if stale.name not in expected_names:
            stale.unlink()
    engine = SD15Engine(build_sd15_engine_config("balanced", safety_checker_required=True))
    images: list[dict[str, Any]] = []
    started = time.time()
    try:
        engine.load(str(model_path))
        for seed in preset.seeds:
            request = build_sd15_generation_request(
                preset_id=preset.preset_id,
                seed=seed,
                steps=args.steps or preset.steps,
            )
            result = engine.generate(request)
            metrics = _image_metrics(result.image, output_dir / f"seed-{seed}.png")
            metrics.update(
                {
                    "seed": seed,
                    "elapsed_seconds": result.elapsed_seconds,
                    "scheduler": result.metadata.get("scheduler"),
                    "safety_flagged": result.metadata.get("safety_flagged", False),
                }
            )
            metrics["automatic_pass"] = (
                metrics["automatic_pass"] and not metrics["safety_flagged"]
            )
            images.append(metrics)
    finally:
        engine.unload()

    unique_hashes = len({item["sha256"] for item in images})
    automatic_pass = (
        len(images) == len(preset.seeds)
        and all(item["automatic_pass"] for item in images)
        and unique_hashes >= max(1, len(images) - 1)
    )
    reviewers = args.reviewer
    manual_passes = {item["name"] for item in reviewers if item["decision"] == "pass"}
    manual_failures = [item for item in reviewers if item["decision"] == "fail"]
    manual_pass = len(manual_passes) >= 2 and not manual_failures
    status = (
        "failed"
        if not automatic_pass or manual_failures
        else "passed"
        if manual_pass
        else "pending_manual_review"
    )
    report = {
        "schema_version": 1,
        "asset_id": spec.asset_id,
        "artifact_id": spec.artifact_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license_id": spec.license_id,
        "preset_id": preset.preset_id,
        "prompt": preset.prompt,
        "negative_prompt": preset.negative_prompt,
        "parameters": {
            "width": preset.width,
            "height": preset.height,
            "steps": args.steps or preset.steps,
            "guidance_scale": preset.guidance_scale,
            "scheduler": preset.scheduler,
            "seeds": list(preset.seeds),
        },
        "safety_checker_required": True,
        "asset_verification": {
            "valid": verification["valid"],
            "integrity_scope": verification["integrity_scope"],
        },
        "automatic_gate": {
            "passed": automatic_pass,
            "unique_images": unique_hashes,
            "required_unique_images": max(1, len(images) - 1),
        },
        "manual_gate": {
            "passed": manual_pass,
            "required_reviewers": 2,
            "reviews": reviewers,
        },
        "status": status,
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
