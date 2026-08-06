import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15EngineConfig,
    SD15GenerationRequest,
)
from diffusion.artifacts import DiffusionArtifact
from diffusion.service import SD15EditRequest


class _FakePipeline:
    def __init__(self, *, type_error: str = "", progress_logging=None):
        self.type_error = type_error
        self.progress_logging = progress_logging

    def __call__(self, *, callback_on_step_end, num_inference_steps, **_kwargs):
        if self.progress_logging is not None and self.progress_logging.enabled:
            raise OSError(22, "Invalid argument")
        if self.type_error:
            raise TypeError(self.type_error)
        for step in range(num_inference_steps):
            callback_on_step_end(self, step, None, {})
        return SimpleNamespace(images=["generated-image"])


class _ConfigurableFakePipeline:
    def __init__(self):
        self.calls = []
        self.unet = "unet"
        self.vae = SimpleNamespace(enable_slicing=lambda: self.calls.append("vae_slicing"))

    def enable_attention_slicing(self):
        self.calls.append("attention_slicing")

    def fuse_qkv_projections(self, *, unet, vae):
        self.calls.append(("qkv_fusion", unet, vae))

    def enable_model_cpu_offload(self):
        self.calls.append("cpu_offload")

    def to(self, device):
        self.calls.append(("to", device))
        return self


class _FakeTorch:
    def __init__(self):
        self.calls = []

    def compile(self, module, **kwargs):
        self.calls.append((module, kwargs))
        return "compiled-unet"


class _FakeLinear:
    def __init__(self, in_features=2, out_features=3, has_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = object() if has_bias else None
        self.training = True

    def state_dict(self):
        return {"weight": "weight", "bias": "bias"}


class _FakeModule:
    def __init__(self, children=None):
        object.__setattr__(self, "_children", children or {})

    def named_children(self):
        return list(self._children.items())

    def __setattr__(self, name, value):
        if name in self._children:
            self._children[name] = value
        object.__setattr__(self, name, value)


class _FakeReplacement:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.loaded_state = None
        self.training = None
        self.requires_grad = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def train(self, value):
        self.training = value

    def requires_grad_(self, value):
        self.requires_grad = value


class _FakeBnb:
    class nn:
        Linear8bitLt = _FakeReplacement


class _FakeTorchForQuantization:
    class nn:
        Linear = _FakeLinear


class _FakeDiffusersLogging:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []

    def is_progress_bar_enabled(self):
        return self.enabled

    def disable_progress_bar(self):
        self.calls.append("disable")
        self.enabled = False

    def enable_progress_bar(self):
        self.calls.append("enable")
        self.enabled = True


def _loaded_fake_engine() -> SD15Engine:
    engine = SD15Engine()
    progress_logging = _FakeDiffusersLogging(enabled=True)
    engine._pipeline = _FakePipeline(progress_logging=progress_logging)
    engine._diffusers_logging = progress_logging
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
    assert engine._diffusers_logging.calls == ["disable", "enable"]
    assert engine._diffusers_logging.enabled is True


def test_generation_suspends_progress_for_detached_windows_runtime():
    engine = _loaded_fake_engine()

    result = engine.generate(SD15GenerationRequest(prompt="test", steps=1))

    assert result.image == "generated-image"
    assert engine._diffusers_logging.calls == ["disable", "enable"]


def test_generation_cancels_on_the_next_denoising_step():
    engine = _loaded_fake_engine()

    def cancel_after_first_step(current: int, _total: int) -> None:
        if current == 1:
            engine.cancel()

    with pytest.raises(GenerationCancelled, match="生成已取消"):
        engine.generate(SD15GenerationRequest(prompt="test", steps=3), callback=cancel_after_first_step)


def test_img2img_reuses_loaded_components_and_reports_strength(monkeypatch):
    diffusers = pytest.importorskip("diffusers")

    class _Img2ImgPipeline:
        created_from = None

        @classmethod
        def from_pipe(cls, pipeline):
            cls.created_from = pipeline
            return cls()

        def enable_attention_slicing(self):
            return None

        def enable_vae_slicing(self):
            return None

        def __call__(self, *, image, strength, callback_on_step_end, num_inference_steps, **_kwargs):
            assert image.size == (512, 512)
            assert strength == 0.4
            for step in range(num_inference_steps):
                callback_on_step_end(self, step, None, {})
            return SimpleNamespace(images=['edited-image'])

    monkeypatch.setattr(diffusers, 'StableDiffusionImg2ImgPipeline', _Img2ImgPipeline)
    engine = _loaded_fake_engine()
    request = SD15EditRequest(
        mode='img2img',
        source_blob_id='img_source',
        prompt='change the lighting',
        strength=0.4,
        width=512,
        height=512,
        steps=2,
    )
    progress = []
    result = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        callback=lambda step, total: progress.append((step, total)),
    )

    assert result.image == 'edited-image'
    assert progress == [(1, 2), (2, 2)]
    assert result.metadata['engine'] == 'diffusers_sd15_img2img'
    assert result.metadata['strength'] == 0.4
    assert _Img2ImgPipeline.created_from is engine._pipeline


def test_unrelated_pipeline_type_error_is_not_misreported_as_old_diffusers():
    engine = _loaded_fake_engine()
    engine._pipeline = _FakePipeline(type_error="a real pipeline programming error")

    with pytest.raises(TypeError, match="real pipeline programming error"):
        engine.generate(SD15GenerationRequest(prompt="test", steps=1))


def test_load_rejects_non_directory_before_importing_sidecar_dependencies(tmp_path):
    engine = SD15Engine()

    with pytest.raises(ValueError, match="Diffusers"):
        engine.load(str(tmp_path / "not-downloaded"))


def test_pipeline_load_suspends_progress_and_restores_previous_state():
    logging = _FakeDiffusersLogging(enabled=True)

    with pytest.raises(RuntimeError, match="load failed"):
        with SD15Engine._suspended_diffusers_progress(logging):
            assert logging.enabled is False
            raise RuntimeError("load failed")

    assert logging.enabled is True
    assert logging.calls == ["disable", "enable"]


def test_pipeline_load_keeps_an_already_disabled_progress_bar_disabled():
    logging = _FakeDiffusersLogging(enabled=False)

    with SD15Engine._suspended_diffusers_progress(logging):
        assert logging.enabled is False

    assert logging.enabled is False
    assert logging.calls == ["disable"]


@pytest.mark.parametrize(
    "config, message",
    [
        (
            SD15EngineConfig(
                quantization="bitsandbytes_8bit_unet",
                enable_model_cpu_offload=True,
            ),
            "CPU offload",
        ),
        (
            SD15EngineConfig(enable_torch_compile=True, enable_model_cpu_offload=True),
            "CPU offload",
        ),
        (
            SD15EngineConfig(device="cpu", enable_torch_compile=True, enable_model_cpu_offload=False),
            "CUDA",
        ),
        (
            SD15EngineConfig(enable_qkv_fusion=True),
            "attention slicing",
        ),
    ],
)
def test_runtime_config_rejects_incompatible_device_ownership(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_configure_pipeline_applies_fusion_and_compile_to_resident_unet():
    engine = SD15Engine(
        SD15EngineConfig(
            enable_model_cpu_offload=False,
            enable_qkv_fusion=True,
            enable_attention_slicing=False,
            enable_torch_compile=True,
        )
    )
    engine._device = "cuda"
    pipeline = _ConfigurableFakePipeline()
    torch = _FakeTorch()

    configured = engine._configure_pipeline(pipeline, use_cuda=True, torch=torch)

    assert configured is pipeline
    assert pipeline.calls == [
        "vae_slicing",
        ("qkv_fusion", True, False),
        ("to", "cuda"),
    ]
    assert pipeline.unet == "compiled-unet"
    assert torch.calls == [("unet", {"mode": "reduce-overhead", "fullgraph": False})]


def test_default_pipeline_strategy_keeps_the_tested_cpu_offload_path():
    engine = SD15Engine()
    engine._device = "cuda"
    pipeline = _ConfigurableFakePipeline()

    configured = engine._configure_pipeline(pipeline, use_cuda=True, torch=_FakeTorch())

    assert configured is pipeline
    assert pipeline.calls == ["attention_slicing", "vae_slicing", "cpu_offload"]


def test_fp32_dreambooth_snapshot_does_not_request_a_missing_fp16_variant():
    engine = SD15Engine()
    kwargs = engine._pipeline_load_kwargs(
        dtype="float16",
        artifact=DiffusionArtifact(
            path="model",
            artifact_kind="sd15_pipeline",
            precision="fp32",
            loadable=True,
        ),
    )

    assert "variant" not in kwargs
    assert kwargs["torch_dtype"] == "float16"


def test_required_safety_checker_is_enforced_after_pipeline_load():
    engine = SD15Engine(SD15EngineConfig(safety_checker_required=True))

    with pytest.raises(RuntimeError, match="safety checker"):
        engine._validate_pipeline_safety(SimpleNamespace(safety_checker=None))

    engine._validate_pipeline_safety(SimpleNamespace(safety_checker=object()))


def test_sd15_capabilities_do_not_mislabel_attention_slicing_as_kv_paging():
    engine = SD15Engine(
        SD15EngineConfig(
            quantization="bitsandbytes_8bit_unet",
            enable_model_cpu_offload=False,
            enable_qkv_fusion=True,
            enable_attention_slicing=False,
        )
    )

    capabilities = engine.capabilities

    assert capabilities["weight_quantization"] == {
        "enabled": True,
        "strategy": "bitsandbytes_8bit_unet",
        "target": "unet",
        "scope": "torch.nn.Linear modules only",
        "module_count": 0,
    }
    assert capabilities["kv_paging"]["supported"] is False
    assert capabilities["attention_memory"]["strategy"] == "none"
    assert capabilities["operator_fusion"]["qkv_projection"] is True


def test_unet_8bit_quantization_replaces_only_linear_modules_recursively():
    nested = _FakeModule({"projection": _FakeLinear(3, 4, has_bias=False)})
    root = _FakeModule({"input": _FakeLinear(), "nested": nested})

    count = SD15Engine._replace_linear_with_8bit_modules(
        root,
        torch=_FakeTorchForQuantization(),
        bnb=_FakeBnb(),
    )

    assert count == 2
    assert isinstance(root._children["input"], _FakeReplacement)
    assert isinstance(nested._children["projection"], _FakeReplacement)
    assert root._children["input"].init_kwargs == {"bias": True, "has_fp16_weights": False, "threshold": 6.0}
    assert nested._children["projection"].init_kwargs == {"bias": False, "has_fp16_weights": False, "threshold": 6.0}
    assert root._children["input"].loaded_state == {"weight": "weight", "bias": "bias"}
    assert root._children["input"].training is True
    assert root._children["input"].requires_grad is False


def test_compile_rejection_explains_that_qkv_fusion_remains_available(monkeypatch):
    engine = SD15Engine(
        SD15EngineConfig(enable_torch_compile=True, enable_model_cpu_offload=False)
    )
    monkeypatch.setattr(
        SD15Engine,
        "_torch_compile_backend_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="QKV projection fusion"):
        engine._require_torch_compile_backend()


def test_load_starts_with_a_clean_quantized_module_count(tmp_path):
    engine = SD15Engine()
    engine._quantized_linear_count = 42

    with pytest.raises(ValueError, match="Diffusers"):
        engine.load(str(tmp_path / "not-downloaded"))

    assert engine.capabilities["weight_quantization"]["module_count"] == 0


def test_failed_configure_resets_quantization_state():
    engine = SD15Engine(
        SD15EngineConfig(
            quantization="bitsandbytes_8bit_unet",
            enable_model_cpu_offload=False,
        )
    )
    engine._quantized_linear_count = 42
    engine._pipeline = object()
    engine._artifact = object()

    engine._reset_failed_load()

    assert engine.is_loaded is False
    assert engine.artifact is None
    assert engine.device == "cpu"
    assert engine.capabilities["weight_quantization"]["module_count"] == 0


def test_unload_resets_the_runtime_device_state():
    engine = _loaded_fake_engine()
    engine._device = "cuda"

    engine.unload()

    assert engine.is_loaded is False
    assert engine.device == "cpu"
