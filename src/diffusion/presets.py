"""Stable Diffusion 1.5 fixed presets used by local smoke and demos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List


@dataclass(frozen=True)
class DiffusionPreset:
    preset_id: str
    model_id: str
    prompt: str
    negative_prompt: str
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance_scale: float = 7.5
    scheduler: str = "DPMSolverMultistepScheduler"
    seeds: tuple[int, ...] = (19950101, 19950102, 19950103, 19950104)
    safety_checker_required: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise ValueError("SD 尺寸必须为正数且是 8 的倍数")
        if not 1 <= self.steps <= 100:
            raise ValueError("SD steps 必须在 1-100 之间")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale 不能为负数")
        if not self.seeds:
            raise ValueError("preset 至少需要一个 seed")


_PRESETS: Dict[str, DiffusionPreset] = {
    "sd15_original_v1": DiffusionPreset(
        preset_id="sd15_original_v1",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        prompt=(
            "a small observatory above a quiet mountain lake, warm window light, "
            "cinematic landscape, detailed environment, soft evening sky"
        ),
        negative_prompt=(
            "low quality, worst quality, blurry, deformed, malformed, text, "
            "watermark, logo, duplicate, oversaturated"
        ),
    ),
    "sd15_retrovers_space_courier_v1": DiffusionPreset(
        preset_id="sd15_retrovers_space_courier_v1",
        model_id="Aleksandra11/90style_anime_face_model",
        prompt=(
            "retrovers, portrait of an adult woman space courier on a rain-soaked "
            "neon train platform, 1990s Japanese cel animation, expressive eyes, "
            "hand-painted background, cinematic lighting, detailed face"
        ),
        negative_prompt=(
            "low quality, worst quality, blurry, deformed, malformed hands, "
            "extra fingers, text, watermark, logo, duplicate"
        ),
        steps=40,
    ),
}


def list_presets() -> List[DiffusionPreset]:
    return list(_PRESETS.values())


def get_preset(preset_id: str) -> DiffusionPreset:
    try:
        return _PRESETS[preset_id]
    except KeyError as exc:
        raise KeyError(f"未知 SD preset: {preset_id}") from exc


def with_seed(preset: DiffusionPreset, seed: int) -> DiffusionPreset:
    """Return a one-seed copy for a deterministic worker stage."""

    return replace(preset, seeds=(int(seed),))


__all__ = ["DiffusionPreset", "get_preset", "list_presets", "with_seed"]
