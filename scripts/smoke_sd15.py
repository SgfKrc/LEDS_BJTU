"""Run a deterministic local SD 1.5 smoke test on the CUDA sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusion import SD15Engine, SD15GenerationRequest, get_preset


def _cuda_memory_snapshot() -> dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize()
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except (ImportError, RuntimeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local SD1.5 smoke image")
    parser.add_argument("--model-path", default="models/sd15-original-v1")
    parser.add_argument("--preset", default="sd15_original_v1")
    parser.add_argument("--seed", type=int, default=19950101)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--output-dir", default="logs/sd15")
    parser.add_argument("--repeat", type=int, default=1, help="Continuous runs on one loaded pipeline")
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 20:
        raise SystemExit("--repeat 必须在 1-20 之间")

    preset = get_preset(args.preset)
    engine = SD15Engine()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    try:
        artifact = engine.load(args.model_path)
        for index in range(args.repeat):
            request = SD15GenerationRequest(
                prompt=preset.prompt,
                negative_prompt=preset.negative_prompt,
                seed=args.seed + index,
                width=preset.width,
                height=preset.height,
                steps=args.steps,
                guidance_scale=preset.guidance_scale,
            )
            result = engine.generate(request)
            output_path = output_dir / f"{preset.preset_id}_seed{result.seed}.png"
            result.image.save(output_path)
            runs.append(
                {
                    "output_path": str(output_path.resolve()),
                    "seed": result.seed,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "memory": _cuda_memory_snapshot(),
                }
            )
    finally:
        engine.unload()
    payload = {
        "output_path": runs[0]["output_path"],
        "artifact": artifact.to_dict(),
        "elapsed_seconds": runs[0]["elapsed_seconds"],
        "metadata": {
            "engine": "diffusers_sd15",
            "device": engine.device,
            "width": preset.width,
            "height": preset.height,
            "steps": args.steps,
            "guidance_scale": preset.guidance_scale,
        },
        "runs": runs,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
