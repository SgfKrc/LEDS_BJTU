import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15EngineConfig,
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
