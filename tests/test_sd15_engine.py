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

    def set_progress_bar_config(self, **kwargs):
        self.calls.append(("progress", kwargs))

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
        created_dtype = None

        @classmethod
        def from_pipe(cls, pipeline, *, torch_dtype=None):
            cls.created_from = pipeline
            cls.created_dtype = torch_dtype
            return cls()

        def enable_attention_slicing(self):
            return None

        def enable_vae_slicing(self):
            return None

        def __call__(self, *, image, strength, callback_on_step_end, num_inference_steps, **_kwargs):
            assert image.size == (512, 512)
            assert strength == 0.4
            for step in range(int(num_inference_steps * strength)):
                callback_on_step_end(self, step, None, {})
            return SimpleNamespace(images=['edited-image'])

    monkeypatch.setattr(diffusers, 'StableDiffusionImg2ImgPipeline', _Img2ImgPipeline)
    engine = _loaded_fake_engine()
    engine._pipeline.dtype = 'float16'
    request = SD15EditRequest(
        mode='img2img',
        source_blob_id='img_source',
        prompt='change the lighting',
        strength=0.4,
        width=512,
        height=512,
        steps=3,
    )
    progress = []
    result = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        callback=lambda step, total: progress.append((step, total)),
    )

    assert result.image == 'edited-image'
    assert progress == [(1, 1)]
    assert result.metadata['engine'] == 'diffusers_sd15_img2img'
    assert result.metadata['strength'] == 0.4
    assert _Img2ImgPipeline.created_from is engine._pipeline
    assert _Img2ImgPipeline.created_dtype == 'float16'


def test_inpaint_loads_a_dedicated_pipeline_and_reuses_it(monkeypatch, tmp_path):
    diffusers = pytest.importorskip("diffusers")

    class _InpaintPipeline:
        load_calls = []

        def __init__(self):
            self.safety_checker = object()
            self.scheduler = SimpleNamespace()
            self.vae = SimpleNamespace(enable_slicing=lambda: None)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.load_calls.append((Path(path), kwargs))
            return cls()

        def set_progress_bar_config(self, **_kwargs):
            return None

        def enable_attention_slicing(self):
            return None

        def to(self, _device):
            return self

        def __call__(
            self,
            *,
            image,
            mask_image,
            strength,
            callback_on_step_end,
            num_inference_steps,
            **_kwargs,
        ):
            assert image.mode == 'RGB'
            assert image.size == (512, 512)
            assert mask_image.mode == 'L'
            assert mask_image.size == (512, 512)
            assert strength == 0.5
            for step in range(int(num_inference_steps * strength)):
                callback_on_step_end(self, step, None, {})
            return SimpleNamespace(images=['inpainted-image'])

    monkeypatch.setattr(diffusers, 'StableDiffusionInpaintPipeline', _InpaintPipeline)
    engine = SD15Engine(
        SD15EngineConfig(
            device='cpu',
            dtype='float32',
            enable_model_cpu_offload=False,
        )
    )
    engine._pipeline = _FakePipeline()
    engine._diffusers_logging = _FakeDiffusersLogging()
    engine._device = 'cpu'
    artifact = DiffusionArtifact(
        path=str(tmp_path),
        artifact_kind='sd15_inpaint_pipeline',
        precision='fp16',
        sha256='d' * 64,
        loadable=True,
    )
    request = SD15EditRequest(
        mode='inpaint',
        source_blob_id='img_source',
        mask_blob_id='img_mask',
        prompt='replace the selected window',
        edit_adapter_id='sd15-inpaint',
        strength=0.5,
        width=512,
        height=512,
        steps=4,
    )
    progress = []

    first = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        mask=Image.new('L', (16, 16), 255),
        adapter=artifact,
        callback=lambda step, total: progress.append((step, total)),
    )
    second = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        mask=Image.new('L', (16, 16), 255),
        adapter=artifact,
    )

    assert first.image == second.image == 'inpainted-image'
    assert first.metadata['engine'] == 'diffusers_sd15_inpaint'
    assert first.metadata['inpaint_sha256'] == 'd' * 64
    assert first.metadata['mask_semantics'] == 'white=redraw, black=preserve'
    assert progress == [(1, 2), (2, 2)]
    assert len(_InpaintPipeline.load_calls) == 1
    assert _InpaintPipeline.load_calls[0][1]['local_files_only'] is True


def test_instruction_mode_loads_instruct_pix2pix_and_uses_image_guidance(
    monkeypatch,
    tmp_path,
):
    diffusers = pytest.importorskip("diffusers")

    class _InstructionPipeline:
        load_calls = []

        def __init__(self):
            self.safety_checker = object()
            self.scheduler = SimpleNamespace()
            self.vae = SimpleNamespace(enable_slicing=lambda: None)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.load_calls.append((Path(path), kwargs))
            return cls()

        def set_progress_bar_config(self, **_kwargs):
            return None

        def enable_attention_slicing(self):
            return None

        def to(self, _device):
            return self

        def __call__(
            self,
            *,
            prompt,
            image,
            image_guidance_scale,
            callback_on_step_end,
            num_inference_steps,
            **kwargs,
        ):
            assert prompt == 'make it a snowy winter day'
            assert image.mode == 'RGB'
            assert image.size == (512, 512)
            assert image_guidance_scale == 1.0
            assert 'strength' not in kwargs
            for step in range(num_inference_steps):
                callback_on_step_end(self, step, None, {})
            return SimpleNamespace(images=['instruction-image'])

    monkeypatch.setattr(
        diffusers,
        'StableDiffusionInstructPix2PixPipeline',
        _InstructionPipeline,
    )
    engine = SD15Engine(
        SD15EngineConfig(
            device='cpu',
            dtype='float32',
            enable_model_cpu_offload=False,
        )
    )
    engine._pipeline = _FakePipeline()
    engine._diffusers_logging = _FakeDiffusersLogging()
    engine._device = 'cpu'
    artifact = DiffusionArtifact(
        path=str(tmp_path),
        artifact_kind='sd15_instruction_pipeline',
        precision='fp16',
        sha256='e' * 64,
        loadable=True,
    )
    request = SD15EditRequest(
        mode='instruction',
        source_blob_id='img_source',
        prompt='make it a snowy winter day',
        instruction='make it a snowy winter day',
        edit_adapter_id='instruction-pipeline',
        image_guidance_scale=1.0,
        width=512,
        height=512,
        steps=2,
    )
    progress = []

    first = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        adapter=artifact,
        callback=lambda step, total: progress.append((step, total)),
    )
    second = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        adapter=artifact,
    )

    assert first.image == second.image == 'instruction-image'
    assert first.metadata['engine'] == 'diffusers_sd15_instruct_pix2pix'
    assert first.metadata['instruction_pipeline_sha256'] == 'e' * 64
    assert first.metadata['image_guidance_scale'] == 1.0
    assert progress == [(1, 2), (2, 2)]
    assert len(_InstructionPipeline.load_calls) == 1
    assert _InstructionPipeline.load_calls[0][1]['local_files_only'] is True
    assert engine._instruction_pipeline is not None

    engine.generate(SD15GenerationRequest(prompt='plain generation', steps=1))

    assert engine._instruction_pipeline is None


def test_reference_mode_loads_local_ip_adapter_and_reports_identity(monkeypatch):
    class _ReferencePipeline(_FakePipeline):
        def __init__(self):
            super().__init__()
            self.loaded = []
            self.scales = []
            self.calls = []

        def disable_attention_slicing(self):
            self.calls.append(('attention_slicing', 'disabled'))

        def enable_attention_slicing(self):
            self.calls.append(('attention_slicing', 'enabled'))

        def load_ip_adapter(self, path, **kwargs):
            self.loaded.append((path, kwargs))

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def unload_ip_adapter(self):
            self.loaded.append(('unloaded', {}))

        def __call__(self, *, callback_on_step_end, num_inference_steps, ip_adapter_image=None, **kwargs):
            if ip_adapter_image is not None:
                self.calls.append((ip_adapter_image.size, kwargs))
            for step in range(num_inference_steps):
                callback_on_step_end(self, step, None, {})
            return SimpleNamespace(
                images=['reference-image' if ip_adapter_image is not None else 'generated-image']
            )

    monkeypatch.setattr(
        'diffusion.sd15_engine.resolve_sd15_ip_adapter_layout',
        lambda _path: {
            'root': 'C:/models/ip-adapter',
            'subfolder': 'models',
            'weight_name': 'ip-adapter_sd15.safetensors',
            'image_encoder_folder': 'models/image_encoder',
        },
    )
    engine = SD15Engine()
    engine._pipeline = _ReferencePipeline()
    engine._diffusers_logging = _FakeDiffusersLogging()
    engine._device = 'cpu'
    adapter = DiffusionArtifact(
        path='C:/models/ip-adapter',
        artifact_kind='sd15_ip_adapter',
        precision='fp16',
        sha256='b' * 64,
        loadable=True,
    )
    request = SD15EditRequest(
        mode='reference',
        source_blob_id='img_reference',
        prompt='same character in a city',
        edit_adapter_id='ip-adapter',
        ip_adapter_scale=0.65,
        width=512,
        height=512,
        steps=2,
    )

    result = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
        adapter=adapter,
    )

    assert result.image == 'reference-image'
    assert result.metadata['engine'] == 'diffusers_sd15_ip_adapter'
    assert result.metadata['ip_adapter_scale'] == 0.65
    assert result.metadata['ip_adapter_sha256'] == 'b' * 64
    assert engine._pipeline.scales == [0.65]
    assert engine._pipeline.loaded[0][1]['local_files_only'] is True
    assert engine._pipeline.calls[0] == ('attention_slicing', 'disabled')
    assert engine._pipeline.calls[1][0] == (16, 16)

    engine.generate(SD15GenerationRequest(prompt='plain generation', steps=1))

    assert ('attention_slicing', 'enabled') in engine._pipeline.calls
    assert engine._ip_adapter_identity is None


def test_reference_mode_refreshes_cuda_cpu_offload_hooks_after_load_and_unload(monkeypatch):
    class _OffloadedReferencePipeline(_FakePipeline):
        def __init__(self):
            super().__init__()
            self.offload_devices = []
            self.lifecycle = []

        def disable_attention_slicing(self):
            return None

        def enable_attention_slicing(self):
            return None

        def load_ip_adapter(self, _path, **_kwargs):
            return None

        def set_ip_adapter_scale(self, _scale):
            return None

        def unload_ip_adapter(self):
            self.lifecycle.append('unload_adapter')

        def remove_all_hooks(self):
            self.lifecycle.append('remove_hooks')

        def enable_model_cpu_offload(self, *, device):
            self.offload_devices.append(device)

    monkeypatch.setattr(
        'diffusion.sd15_engine.resolve_sd15_ip_adapter_layout',
        lambda _path: {
            'root': 'C:/models/ip-adapter',
            'subfolder': 'models',
            'weight_name': 'ip-adapter_sd15.safetensors',
            'image_encoder_folder': 'models/image_encoder',
        },
    )
    engine = SD15Engine()
    engine._pipeline = _OffloadedReferencePipeline()
    engine._device = 'cuda'
    adapter = DiffusionArtifact(
        path='C:/models/ip-adapter',
        artifact_kind='sd15_ip_adapter',
        sha256='c' * 64,
        loadable=True,
    )

    engine._ensure_ip_adapter(engine._pipeline, adapter, scale=0.6)
    engine._unload_ip_adapter(engine._pipeline)

    assert engine._pipeline.offload_devices == ['cuda', 'cuda']
    assert engine._pipeline.lifecycle == ['remove_hooks', 'unload_adapter']


def test_img2img_rebinds_cuda_cpu_offload_to_shared_edit_pipeline(monkeypatch):
    class _OffloadedImg2ImgPipeline:
        def __init__(self):
            self.offload_devices = []

        def enable_model_cpu_offload(self, *, device):
            self.offload_devices.append(device)

        def __call__(
            self,
            *,
            callback_on_step_end,
            num_inference_steps,
            image,
            strength,
            **_kwargs,
        ):
            assert image.size == (512, 512)
            assert strength == 0.55
            callback_on_step_end(self, 0, None, {})
            return SimpleNamespace(images=['edited-image'])

    engine = SD15Engine()
    engine._pipeline = _FakePipeline()
    engine._diffusers_logging = _FakeDiffusersLogging()
    engine._device = 'cuda'
    edit_pipeline = _OffloadedImg2ImgPipeline()
    monkeypatch.setattr(engine, '_get_img2img_pipeline', lambda _pipeline: edit_pipeline)
    request = SD15EditRequest(
        mode='img2img',
        source_blob_id='img_source',
        prompt='change the lighting',
        strength=0.55,
        width=512,
        height=512,
        steps=2,
    )

    result = engine.edit(
        request,
        image=Image.new('RGB', (16, 16), 127),
    )

    assert result.image == 'edited-image'
    assert edit_pipeline.offload_devices == ['cuda']


def test_reference_mode_rejects_unvalidated_qkv_profile():
    engine = SD15Engine(
        SD15EngineConfig(enable_attention_slicing=False, enable_qkv_fusion=True)
    )
    engine._pipeline = _FakePipeline()
    engine._diffusers_logging = _FakeDiffusersLogging()
    adapter = DiffusionArtifact(
        path='C:/models/ip-adapter',
        artifact_kind='sd15_ip_adapter',
        loadable=True,
    )
    request = SD15EditRequest(
        mode='reference',
        source_blob_id='img_reference',
        prompt='same character',
        edit_adapter_id='ip-adapter',
        ip_adapter_scale=0.6,
        width=512,
        height=512,
    )

    with pytest.raises(RuntimeError, match='QKV-fused'):
        engine.edit(
            request,
            image=Image.new('RGB', (16, 16), 127),
            adapter=adapter,
        )


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
        ("progress", {"disable": True}),
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
    assert pipeline.calls == [
        ("progress", {"disable": True}),
        "attention_slicing",
        "vae_slicing",
        "cpu_offload",
    ]


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
