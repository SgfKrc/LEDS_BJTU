"""Deterministic SD 1.5 prompt and scheduler matrix tools.

These commands are intentionally sidecar-only: they verify the local asset,
generate bounded PNG outputs, and write a JSON report without changing the
model directory or the primary SQLite store.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _image_metrics(image: Any, path: Path) -> dict[str, Any]:
    from PIL import ImageStat

    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    extrema = rgb.getextrema()
    rgb.save(path, format="PNG")
    data = path.read_bytes()
    channel_stddev = [float(value) for value in stat.stddev]
    entropy = float(rgb.entropy())
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "mode": rgb.mode,
        "width": rgb.width,
        "height": rgb.height,
        "entropy": entropy,
        "channel_stddev": channel_stddev,
        "extrema": [list(item) for item in extrema],
        "automatic_pass": (
            rgb.width > 0
            and rgb.height > 0
            and entropy >= 3.0
            and max(channel_stddev) >= 5.0
            and any(low < high for low, high in extrema)
        ),
    }


def _contact_sheet(image_paths: Iterable[Path], output: Path, *, columns: int = 4) -> None:
    from PIL import Image, ImageDraw

    paths = list(image_paths)
    if not paths:
        return
    thumbnails = [Image.open(path).convert("RGB") for path in paths]
    tile_width = max(image.width for image in thumbnails)
    tile_height = max(image.height for image in thumbnails) + 24
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, path) in enumerate(zip(thumbnails, paths)):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + tile_height - 20), path.stem[:32], fill="black")
    sheet.save(output, format="PNG")
    for image in thumbnails:
        image.close()


def _validate_common(*, steps: int, width: int, height: int, max_runs: int) -> None:
    if not 1 <= steps <= 100:
        raise ValueError("steps must be between 1 and 100")
    if not 64 <= width <= 768 or not 64 <= height <= 768 or width % 8 or height % 8:
        raise ValueError("width and height must be multiples of 8 between 64 and 768")
    if max_runs < 1 or max_runs > 64:
        raise ValueError("matrix is limited to 64 outputs")


def _write_report(report: dict[str, Any]) -> None:
    report_path = Path(report["output_dir"]) / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    report["report_path"] = str(report_path)
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def _build_request(preset: Any, *, prompt: str, seed: int, steps: int, scheduler: str) -> Any:
    from diffusion import SD15GenerationRequest

    return SD15GenerationRequest(
        prompt=prompt,
        negative_prompt=preset.negative_prompt,
        seed=seed,
        width=preset.width,
        height=preset.height,
        steps=steps,
        guidance_scale=preset.guidance_scale,
        scheduler=scheduler,
    )


def _run_matrix(
    *,
    asset_id: str,
    model_path: str | Path,
    output_dir: str | Path,
    preset: Any,
    jobs: list[dict[str, Any]],
    engine_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    from diffusion import SD15Engine, build_sd15_engine_config, verify_asset_directory
    from diffusion.assets import get_asset_spec

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = Path(model_path).expanduser().resolve()
    verification = verify_asset_directory(model, asset_id, full_hash=True)
    if not verification.get("valid"):
        return {
            "schema_version": 1,
            "tool": "sd15_matrix",
            "asset_id": asset_id,
            "model_path": str(model),
            "output_dir": str(output),
            "valid": False,
            "status": "asset_invalid",
            "asset_verification": verification,
            "jobs": [],
        }

    engine = engine_factory() if engine_factory else SD15Engine(build_sd15_engine_config("balanced", safety_checker_required=True))
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        engine.load(str(model))
        for index, job in enumerate(jobs):
            request = _build_request(
                preset,
                prompt=job["prompt"],
                seed=job["seed"],
                steps=job["steps"],
                scheduler=job["scheduler"],
            )
            result = engine.generate(request)
            path = output / f"{index:03d}-{job['label']}-seed{job['seed']}.png"
            metrics = _image_metrics(result.image, path)
            metrics.update({
                "label": job["label"],
                "seed": job["seed"],
                "steps": job["steps"],
                "scheduler": job["scheduler"] or result.metadata.get("scheduler"),
                "elapsed_seconds": result.elapsed_seconds,
                "safety_flagged": bool(result.metadata.get("safety_flagged", False)),
            })
            metrics["automatic_pass"] = metrics["automatic_pass"] and not metrics["safety_flagged"]
            results.append(metrics)
    finally:
        engine.unload()
    unique = len({item["sha256"] for item in results})
    report = {
        "schema_version": 1,
        "tool": "sd15_matrix",
        "valid": bool(results) and all(item["automatic_pass"] for item in results),
        "asset_id": asset_id,
        "artifact_id": get_asset_spec(asset_id).artifact_id,
        "model_path": str(model),
        "output_dir": str(output),
        "asset_verification": {
            "valid": verification["valid"],
            "integrity_scope": verification.get("integrity_scope"),
        },
        "automatic_gate": {
            "passed": bool(results) and all(item["automatic_pass"] for item in results),
            "outputs": len(results),
            "unique_images": unique,
        },
        "jobs": results,
        "elapsed_seconds": time.time() - started,
        "read_only_asset_tree": True,
    }
    _contact_sheet([Path(item["path"]) for item in results], output / "contact-sheet.png")
    report["contact_sheet"] = str(output / "contact-sheet.png")
    return report


def run_prompt_batch(
    *,
    asset_id: str,
    model_path: str | Path,
    output_dir: str | Path,
    preset: Any,
    prompts: list[str],
    seeds: list[int],
    steps: int,
    engine_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    if not prompts or not seeds:
        raise ValueError("prompts and seeds must not be empty")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must not contain blank values")
    if len(prompts) * len(seeds) > 64:
        raise ValueError("prompt batch is limited to 64 outputs")
    _validate_common(steps=steps, width=preset.width, height=preset.height, max_runs=len(prompts) * len(seeds))
    jobs = [
        {"label": f"prompt{prompt_index:02d}", "prompt": prompt, "seed": seed, "steps": steps, "scheduler": preset.scheduler}
        for prompt_index, prompt in enumerate(prompts)
        for seed in seeds
    ]
    report = _run_matrix(
        asset_id=asset_id,
        model_path=model_path,
        output_dir=output_dir,
        preset=preset,
        jobs=jobs,
        engine_factory=engine_factory,
    )
    report["tool"] = "sd15_prompt_batch"
    report["prompts"] = prompts
    report["seeds"] = seeds
    _write_report(report)
    return report


def run_sampler_matrix(
    *,
    asset_id: str,
    model_path: str | Path,
    output_dir: str | Path,
    preset: Any,
    prompt: str,
    schedulers: list[str],
    steps_list: list[int],
    seed: int,
    engine_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    if not schedulers or not steps_list:
        raise ValueError("schedulers and steps_list must not be empty")
    _validate_common(steps=max(steps_list), width=preset.width, height=preset.height, max_runs=len(schedulers) * len(steps_list))
    jobs = [
        {"label": f"{scheduler.replace('Scheduler', '').lower()}-{steps}", "prompt": prompt, "seed": seed, "steps": steps, "scheduler": scheduler}
        for scheduler in schedulers
        for steps in steps_list
    ]
    report = _run_matrix(
        asset_id=asset_id,
        model_path=model_path,
        output_dir=output_dir,
        preset=preset,
        jobs=jobs,
        engine_factory=engine_factory,
    )
    report["tool"] = "sd15_sampler_matrix"
    report["prompt"] = prompt
    report["schedulers"] = schedulers
    report["steps_list"] = steps_list
    report["seed"] = seed
    _write_report(report)
    return report


__all__ = ["run_prompt_batch", "run_sampler_matrix"]
