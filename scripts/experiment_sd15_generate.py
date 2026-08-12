#!/usr/bin/env python3
"""EX-N3 标定：按 ps-sd-v1 提示词集生成 SD 1.5 图像（每轮 10 张，固定 seed）。

用法（CUDA 侧车）:
    .venv-packaging-cuda\\Scripts\\python.exe scripts/experiment_sd15_generate.py \\
        --prompt-set fixtures/prompt_sets/ps-sd-v1/prompts.jsonl \\
        --output-dir build/exp-calibration/round-1/images --seed 19950101
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_PATH = "models/sd15-original-v1"
PRESET_ID = "sd15_original_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="EX-N3 标定 SD 图像生成")
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=19950101)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.prompt_set).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusion.presets import get_preset
    from diffusion.sd15_engine import SD15Engine, SD15EngineConfig, SD15GenerationRequest

    preset = get_preset(PRESET_ID)
    engine = SD15Engine(
        SD15EngineConfig(
            quantization="none",
            enable_qkv_fusion=False,
            enable_attention_slicing=True,
            enable_torch_compile=False,
        )
    )
    engine.load(MODEL_PATH)
    manifest = []
    try:
        for index, item in enumerate(items):
            seed = args.seed + index
            request = SD15GenerationRequest(
                prompt=item["prompt"],
                negative_prompt=preset.negative_prompt,
                seed=seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance_scale=preset.guidance_scale,
            )
            result = engine.generate(request)
            image = result.image
            name = f"{item['id']}_seed{seed}.png"
            path = out_dir / name
            image.save(path)
            manifest.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "key_elements": item["key_elements"],
                "image": str(path),
                "seed": seed,
            })
            print(json.dumps({"id": item["id"], "image": name}, ensure_ascii=False))
    finally:
        engine.unload()

    (out_dir.parent / "prompts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"generated": len(manifest), "manifest": str(out_dir.parent / "prompts.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
