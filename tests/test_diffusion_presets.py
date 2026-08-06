import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.presets import get_preset, list_presets, with_seed
from diffusion.sd15_engine import SD15GenerationRequest


def test_sd15_presets_are_deterministic_and_512_safe():
    presets = {preset.preset_id: preset for preset in list_presets()}

    assert "sd15_original_v1" in presets
    assert "sd15_retrovers_space_courier_v1" in presets
    assert all(preset.width == 512 and preset.height == 512 for preset in presets.values())
    assert all(len(preset.seeds) == 10 for preset in presets.values())
    assert len({preset.seeds for preset in presets.values()}) == 1
    assert all(preset.scheduler == "DPMSolverMultistepScheduler" for preset in presets.values())


def test_preset_can_be_reduced_to_one_worker_seed():
    preset = get_preset("sd15_original_v1")
    worker_preset = with_seed(preset, 1234)

    assert worker_preset.preset_id == preset.preset_id
    assert worker_preset.seeds == (1234,)
    assert worker_preset.prompt == preset.prompt


def test_generation_request_rejects_unsafe_size_for_4060_profile():
    request = SD15GenerationRequest(prompt="test", width=1024, height=512)

    with pytest.raises(ValueError, match="不超过 768"):
        request.validate()


def test_generation_request_requires_prompt():
    request = SD15GenerationRequest(prompt=" ")

    with pytest.raises(ValueError, match="prompt 不能为空"):
        request.validate()
