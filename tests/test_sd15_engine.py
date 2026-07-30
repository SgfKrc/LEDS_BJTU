import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15GenerationRequest,
)


class _FakePipeline:
    def __init__(self, *, type_error: str = ""):
        self.type_error = type_error

    def __call__(self, *, callback_on_step_end, num_inference_steps, **_kwargs):
        if self.type_error:
            raise TypeError(self.type_error)
        for step in range(num_inference_steps):
            callback_on_step_end(self, step, None, {})
        return SimpleNamespace(images=["generated-image"])


def _loaded_fake_engine() -> SD15Engine:
    engine = SD15Engine()
    engine._pipeline = _FakePipeline()
    engine._device = "cpu"
    return engine


def test_generation_reports_every_step_and_returns_first_image():
    engine = _loaded_fake_engine()
    progress = []

    result = engine.generate(
        SD15GenerationRequest(prompt="test", steps=3),
        callback=lambda current, total: progress.append((current, total)),
    )

    assert result.image == "generated-image"
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert result.metadata["engine"] == "diffusers_sd15"


def test_generation_cancels_on_the_next_denoising_step():
    engine = _loaded_fake_engine()

    def cancel_after_first_step(current: int, _total: int) -> None:
        if current == 1:
            engine.cancel()

    with pytest.raises(GenerationCancelled, match="生成已取消"):
        engine.generate(SD15GenerationRequest(prompt="test", steps=3), callback=cancel_after_first_step)


def test_unrelated_pipeline_type_error_is_not_misreported_as_old_diffusers():
    engine = _loaded_fake_engine()
    engine._pipeline = _FakePipeline(type_error="a real pipeline programming error")

    with pytest.raises(TypeError, match="real pipeline programming error"):
        engine.generate(SD15GenerationRequest(prompt="test", steps=1))


def test_load_rejects_non_directory_before_importing_sidecar_dependencies(tmp_path):
    engine = SD15Engine()

    with pytest.raises(ValueError, match="Diffusers"):
        engine.load(str(tmp_path / "not-downloaded"))
