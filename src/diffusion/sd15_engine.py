"""Optional local Stable Diffusion 1.5 engine.

The implementation keeps Diffusers imports lazy so importing QLH for the
existing LLM engines does not require the SD sidecar dependencies.  It is
intentionally local-only in this first step; distributed fan-out will wrap the
same engine behind a complete Worker after the local smoke passes.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

from .artifacts import (
    DiffusionArtifact,
    DiffusionArtifactInspector,
    resolve_sd15_ip_adapter_layout,
)


class GenerationCancelled(RuntimeError):
    """Raised when a caller cancels a diffusion generation."""


@dataclass(frozen=True)
class SD15EngineConfig:
    device: str = "cuda"
    dtype: str = "float16"
    variant: Optional[str] = "fp16"
    use_safetensors: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = True
    quantization: str = "none"
    enable_qkv_fusion: bool = False
    enable_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    local_files_only: bool = True
    safety_checker_required: bool = True

    def validate(self) -> None:
        """Validate options whose combination changes device ownership."""

        if self.dtype not in {"float16", "float32"}:
            raise ValueError("SD15 dtype only supports float16 or float32")
        if self.quantization not in {"none", "bitsandbytes_8bit_unet"}:
            raise ValueError("Unsupported SD15 quantization strategy")
        if self.quantization != "none" and self.enable_model_cpu_offload:
            raise ValueError("A quantized U-Net cannot use model CPU offload")
        if self.quantization != "none" and not self.device.startswith("cuda"):
            raise ValueError("SD15 8-bit U-Net quantization requires a CUDA runtime")
        if self.enable_torch_compile and self.enable_model_cpu_offload:
            raise ValueError("A compiled U-Net cannot use model CPU offload")
        if self.enable_torch_compile and not self.device.startswith("cuda"):
            raise ValueError("SD15 torch.compile requires a CUDA runtime")
        if self.enable_qkv_fusion and self.enable_attention_slicing:
            raise ValueError("QKV fusion cannot be combined with attention slicing")
        if not self.torch_compile_mode.strip():
            raise ValueError("torch_compile_mode must not be empty")


@dataclass(frozen=True)
class SD15GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance_scale: float = 7.5
    scheduler: str = ""

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt 不能为空")
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise ValueError("SD width/height 必须是正数且为 8 的倍数")
        if self.width > 768 or self.height > 768:
            raise ValueError("4060 8GB 首期限制分辨率不超过 768")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps 必须在 1-100 之间")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale 不能为负数")
        if self.scheduler not in {"", "PNDMScheduler", "DPMSolverMultistepScheduler"}:
            raise ValueError(f"unsupported SD15 scheduler: {self.scheduler}")


@dataclass(frozen=True)
class SD15GenerationResult:
    image: Any
    seed: int
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SD15Engine:
    """A guarded, local Diffusers SD 1.5 pipeline for CUDA PC smoke tests."""

    def __init__(self, config: Optional[SD15EngineConfig] = None) -> None:
        self.config = config or SD15EngineConfig()
        self._pipeline: Any = None
        self._artifact: Optional[DiffusionArtifact] = None
        self._device = "cpu"
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._quantized_linear_count = 0
        self._diffusers_logging: Any = None
        self._ip_adapter_identity: Optional[str] = None
        self._ip_adapter_scale: Optional[float] = None
        self._inpaint_pipeline: Any = None
        self._inpaint_identity: Optional[str] = None

    @staticmethod
    @contextmanager
    def _suspended_diffusers_progress(diffusers_logging: Any) -> Iterator[None]:
        """Avoid invalid console handles in windowed or detached Windows runs."""

        was_enabled = bool(diffusers_logging.is_progress_bar_enabled())
        diffusers_logging.disable_progress_bar()
        try:
            yield
        finally:
            if was_enabled:
                diffusers_logging.enable_progress_bar()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def artifact(self) -> Optional[DiffusionArtifact]:
        return self._artifact

    @property
    def device(self) -> str:
        return self._device

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Report actual SD15-side capabilities without borrowing LLM labels.

        SD 1.5 denoises a changing latent at each U-Net step. It does not expose
        an autoregressive, cross-step KV cache, so QLH's token ``PagedKVCache``
        must not be advertised as an SD acceleration. Attention slicing remains
        the supported memory-bounded attention strategy for this pipeline.
        """

        return {
            "weight_quantization": {
                "enabled": self.config.quantization != "none",
                "strategy": self.config.quantization,
                "target": "unet" if self.config.quantization != "none" else None,
                "scope": "torch.nn.Linear modules only" if self.config.quantization != "none" else None,
                "module_count": self._quantized_linear_count,
            },
            "kv_paging": {
                "supported": False,
                "reason": "SD1.5 U-Net has no reusable autoregressive KV cache across denoising steps",
            },
            "attention_memory": {
                "configured": self.config.enable_attention_slicing,
                "enabled": (
                    self.config.enable_attention_slicing
                    and self._ip_adapter_identity is None
                ),
                "strategy": (
                    "temporarily_disabled_for_ip_adapter"
                    if self.config.enable_attention_slicing
                    and self._ip_adapter_identity is not None
                    else "diffusers_attention_slicing"
                    if self.config.enable_attention_slicing
                    else "none"
                ),
            },
            "operator_fusion": {
                "qkv_projection": self.config.enable_qkv_fusion,
                "torch_compile_unet": self.config.enable_torch_compile,
                "torch_compile_mode": self.config.torch_compile_mode if self.config.enable_torch_compile else None,
            },
            "reference_image": {
                "supported": True,
                "strategy": "sd15_ip_adapter",
                "adapter_loaded": self._ip_adapter_identity is not None,
                "adapter_identity": self._ip_adapter_identity,
                "scale": self._ip_adapter_scale,
            },
            "inpaint": {
                "supported": True,
                "strategy": "sd15_inpaint_pipeline",
                "pipeline_loaded": self._inpaint_identity is not None,
                "pipeline_identity": self._inpaint_identity,
            },
        }

    def _require_quantization_backend(self) -> Any:
        if self.config.quantization == "none":
            return None

        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "SD15 8-bit quantization requires bitsandbytes in the CUDA sidecar"
            ) from exc
        return bnb

    @staticmethod
    def _torch_compile_backend_available() -> bool:
        try:
            from torch.utils._triton import has_triton
        except ImportError:
            return False
        return bool(has_triton())

    def _require_torch_compile_backend(self) -> None:
        if not self._torch_compile_backend_available():
            raise RuntimeError(
                "SD15 torch.compile requires a working Triton/Inductor backend in the CUDA sidecar; "
                "use QKV projection fusion on this runtime"
            )

    def _reset_failed_load(self) -> None:
        """Return the engine to an unload-equivalent state after load failure."""

        self._pipeline = None
        self._artifact = None
        self._quantized_linear_count = 0
        self._device = "cpu"
        self._diffusers_logging = None
        self._inpaint_pipeline = None
        self._inpaint_identity = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _pipeline_load_kwargs(self, *, dtype: Any, artifact: DiffusionArtifact) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "use_safetensors": self.config.use_safetensors,
        }
        # Community DreamBooth snapshots are often stored as unqualified
        # FP32 safetensors. Asking Diffusers for an fp16 filename variant then
        # fails before torch_dtype can convert the weights.
        if self.config.variant and artifact.precision == "fp16":
            kwargs["variant"] = self.config.variant
        if not self.config.safety_checker_required:
            kwargs["safety_checker"] = None
        return kwargs

    def _validate_pipeline_safety(self, pipeline: Any) -> None:
        if (
            self.config.safety_checker_required
            and getattr(pipeline, "safety_checker", None) is None
        ):
            raise RuntimeError(
                "SD15 safety checker is required but missing from the local pipeline"
            )

    def _mixed_precision_safety_overrides(
        self,
        *,
        model_path: Any,
        artifact: DiffusionArtifact,
        dtype: Any,
    ) -> Dict[str, Any]:
        """Load an fp16-only safety component for an unqualified FP32 pipeline."""

        if not self.config.safety_checker_required or artifact.precision == "fp16":
            return {}
        safety_path = model_path / "safety_checker"
        fp16_weight = safety_path / "model.fp16.safetensors"
        standard_weights = (
            safety_path / "model.safetensors",
            safety_path / "pytorch_model.bin",
        )
        if not fp16_weight.is_file() or any(path.is_file() for path in standard_weights):
            return {}

        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
        from transformers import CLIPImageProcessor

        return {
            "safety_checker": StableDiffusionSafetyChecker.from_pretrained(
                safety_path,
                local_files_only=True,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16",
            ),
            "feature_extractor": CLIPImageProcessor.from_pretrained(
                model_path / "feature_extractor",
                local_files_only=True,
            ),
        }

    @staticmethod
    def _replace_linear_with_8bit_modules(module: Any, *, torch: Any, bnb: Any) -> int:
        """Replace U-Net Linear layers before its one permitted CUDA transfer.

        PipelineQuantizationConfig in Diffusers 0.35.2 dispatches a just-loaded
        8-bit U-Net through an Accelerate version that calls ``.to()`` a second
        time, which the current model wrapper correctly rejects. This local
        replacement uses the same bitsandbytes ``Linear8bitLt`` modules but lets
        this engine own the single CPU-to-CUDA transfer. Convolutional U-Net
        weights intentionally remain FP16: bitsandbytes only supplies an
        inference-safe replacement for Linear layers here.
        """

        replaced = 0
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                replacement = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                replacement.load_state_dict(child.state_dict())
                replacement.train(child.training)
                replacement.requires_grad_(False)
                setattr(module, name, replacement)
                replaced += 1
            else:
                replaced += SD15Engine._replace_linear_with_8bit_modules(child, torch=torch, bnb=bnb)
        return replaced

    def _configure_pipeline(
        self,
        pipeline: Any,
        *,
        use_cuda: bool,
        torch: Any,
        track_quantization: bool = True,
    ) -> Any:
        """Apply mutually validated memory and execution strategies."""

        set_progress = getattr(pipeline, "set_progress_bar_config", None)
        if callable(set_progress):
            # Diffusers 0.35 imports tqdm directly in pipeline_utils, so its
            # global logging switch does not suppress denoising progress. A
            # detached Windows build needs this pipeline-level switch.
            set_progress(disable=True)
        if self.config.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        if self.config.enable_vae_slicing:
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            elif hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
        if self.config.enable_qkv_fusion:
            pipeline.fuse_qkv_projections(unet=True, vae=False)

        if self.config.quantization != "none":
            if not use_cuda:
                raise RuntimeError("SD15 8-bit U-Net quantization requires an available CUDA device")
            bnb = self._require_quantization_backend()
            unet = getattr(pipeline, "unet", None)
            if unet is None:
                raise RuntimeError("SD15 pipeline has no U-Net to quantize")
            quantized_count = self._replace_linear_with_8bit_modules(
                unet,
                torch=torch,
                bnb=bnb,
            )
            if quantized_count == 0:
                raise RuntimeError("SD15 U-Net has no Linear layers eligible for 8-bit quantization")
            if track_quantization:
                self._quantized_linear_count = quantized_count

        if use_cuda and self.config.enable_model_cpu_offload:
            # Accelerate moves one component at a time and keeps baseline SD1.5
            # generation within the 8GB laptop GPU budget.
            pipeline.enable_model_cpu_offload()
            return pipeline

        pipeline = pipeline.to(self._device)
        if self.config.enable_torch_compile:
            unet = getattr(pipeline, "unet", None)
            if unet is None:
                raise RuntimeError("SD15 pipeline has no U-Net to compile")
            pipeline.unet = torch.compile(
                unet,
                mode=self.config.torch_compile_mode,
                fullgraph=False,
            )
        return pipeline

    def load(
        self,
        model_path: str,
        *,
        revision: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        artifact: Optional[DiffusionArtifact] = None,
    ) -> DiffusionArtifact:
        """Load a local Diffusers SD 1.5 pipeline.

        ``model_path`` must be a local, complete Diffusers directory.  Remote
        Hub loading is deliberately outside the first implementation: model
        download is performed by the pinned downloader before this engine is
        allowed to load anything.
        """

        with self._lock:
            self.config.validate()
            if self._pipeline is not None:
                raise RuntimeError("SD15 引擎已经加载模型，请先 unload")
            self._quantized_linear_count = 0

            from pathlib import Path

            if local_files_only is False or not self.config.local_files_only:
                raise ValueError("SD15 首期只允许加载已下载的本地 Diffusers 目录")

            local_model_path = Path(model_path).expanduser()
            if not local_model_path.is_dir():
                raise ValueError("离线 SD15 加载必须传入已下载的 Diffusers 目录")

            import torch

            use_cuda = self.config.device.startswith("cuda") and torch.cuda.is_available()
            if (self.config.quantization != "none" or self.config.enable_torch_compile) and not use_cuda:
                raise RuntimeError("SD15 quantization and torch.compile require an available CUDA device")
            if self.config.enable_torch_compile:
                self._require_torch_compile_backend()

            inspector = DiffusionArtifactInspector()
            inspected = artifact or inspector.inspect(local_model_path, compute_hash=False)
            if inspected.artifact_kind != "sd15_pipeline" or not inspected.loadable:
                raise ValueError(
                    "SD15 首期要求完整 Diffusers pipeline 目录；"
                    f"当前资产为 {inspected.artifact_kind}: "
                    + "; ".join(inspected.warnings)
                )

            try:
                from diffusers import StableDiffusionPipeline
                from diffusers.utils import logging as diffusers_logging
            except ImportError as exc:
                raise ImportError(
                    "SD15 sidecar 未安装，请使用当前 CUDA venv 安装 "
                    "packaging/requirements-sd15.txt"
                ) from exc

            self._device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda and self.config.dtype == "float16" else torch.float32
            pipeline: Any = None
            try:
                with self._suspended_diffusers_progress(diffusers_logging):
                    load_kwargs = self._pipeline_load_kwargs(dtype=dtype, artifact=inspected)
                    load_kwargs.update(
                        self._mixed_precision_safety_overrides(
                            model_path=local_model_path,
                            artifact=inspected,
                            dtype=dtype,
                        )
                    )
                    if revision:
                        load_kwargs["revision"] = revision
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        local_model_path,
                        **load_kwargs,
                    )
                    self._validate_pipeline_safety(pipeline)
                    pipeline = self._configure_pipeline(
                        pipeline,
                        use_cuda=use_cuda,
                        torch=torch,
                    )
            except Exception:
                if pipeline is not None:
                    del pipeline
                self._reset_failed_load()
                raise
            self._pipeline = pipeline
            self._artifact = inspected
            self._diffusers_logging = diffusers_logging
            return inspected

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._artifact = None
            self._quantized_linear_count = 0
            self._device = "cpu"
            self._diffusers_logging = None
            self._ip_adapter_identity = None
            self._ip_adapter_scale = None
            inpaint_pipeline = self._inpaint_pipeline
            self._inpaint_pipeline = None
            self._inpaint_identity = None
            if pipeline is not None:
                del pipeline
            if inpaint_pipeline is not None:
                del inpaint_pipeline
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def cancel(self) -> None:
        self._cancel_event.set()

    @contextmanager
    def _request_scheduler(self, pipeline: Any, request: Any) -> Iterator[None]:
        """Use a fresh request scheduler without leaking it into the next job."""

        requested = getattr(request, "scheduler", "")
        if not requested:
            yield
            return

        from diffusers import DPMSolverMultistepScheduler, PNDMScheduler

        scheduler_types = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "PNDMScheduler": PNDMScheduler,
        }
        scheduler_type = scheduler_types[requested]
        original_scheduler = getattr(pipeline, "scheduler", None)
        if original_scheduler is None:
            raise RuntimeError("SD15 pipeline has no scheduler")
        replacement_scheduler = scheduler_type.from_config(original_scheduler.config)
        pipeline.scheduler = replacement_scheduler
        try:
            yield
        finally:
            pipeline.scheduler = original_scheduler

    def _get_img2img_pipeline(self, pipeline: Any) -> Any:
        """Create the edit pipeline once while keeping the base weights shared."""

        edit_pipeline = getattr(pipeline, "_qlh_img2img_pipeline", None)
        if edit_pipeline is not None:
            return edit_pipeline

        from diffusers import StableDiffusionImg2ImgPipeline

        if hasattr(StableDiffusionImg2ImgPipeline, "from_pipe"):
            # Diffusers 0.35 defaults from_pipe(torch_dtype=...) to FP32 and
            # converts the shared components in place. Preserve the loaded
            # pipeline dtype or reference/img2img switching doubles memory and
            # also mutates the base pipeline precision.
            edit_pipeline = StableDiffusionImg2ImgPipeline.from_pipe(
                pipeline,
                torch_dtype=getattr(pipeline, "dtype", None),
            )
        else:
            edit_pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
        set_progress = getattr(edit_pipeline, "set_progress_bar_config", None)
        if callable(set_progress):
            set_progress(disable=True)
        if (
            self.config.enable_attention_slicing
            and hasattr(edit_pipeline, "enable_attention_slicing")
        ):
            edit_pipeline.enable_attention_slicing()
        if self.config.enable_vae_slicing and hasattr(edit_pipeline, "enable_vae_slicing"):
            edit_pipeline.enable_vae_slicing()
        pipeline._qlh_img2img_pipeline = edit_pipeline
        return edit_pipeline

    def _get_inpaint_pipeline(self, artifact: DiffusionArtifact) -> Any:
        """Load one pinned inpaint pipeline lazily and reuse it across edits."""

        if artifact.artifact_kind != "sd15_inpaint_pipeline" or not artifact.loadable:
            raise ValueError("inpaint mode requires a complete SD1.5 inpaint pipeline")
        identity = artifact.sha256 or artifact.path
        if self._inpaint_pipeline is not None and self._inpaint_identity == identity:
            return self._inpaint_pipeline
        if self.config.quantization != "none" or self.config.enable_qkv_fusion:
            raise RuntimeError(
                "SD15 inpaint has not passed the quantized or QKV-fused compatibility gate"
            )
        if self._device.startswith("cuda") and not self.config.enable_model_cpu_offload:
            raise RuntimeError(
                "SD15 inpaint currently requires model CPU offload to avoid two resident pipelines"
            )

        from pathlib import Path

        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from diffusers.utils import logging as diffusers_logging

        model_path = Path(artifact.path).expanduser()
        if not model_path.is_dir():
            raise ValueError("inpaint artifact must be a local Diffusers directory")

        use_cuda = self._device.startswith("cuda") and torch.cuda.is_available()
        dtype = (
            torch.float16
            if use_cuda and self.config.dtype == "float16"
            else torch.float32
        )
        replacement: Any = None
        try:
            with self._suspended_diffusers_progress(diffusers_logging):
                load_kwargs = self._pipeline_load_kwargs(
                    dtype=dtype,
                    artifact=artifact,
                )
                load_kwargs.update(
                    self._mixed_precision_safety_overrides(
                        model_path=model_path,
                        artifact=artifact,
                        dtype=dtype,
                    )
                )
                replacement = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    **load_kwargs,
                )
                self._validate_pipeline_safety(replacement)
                replacement = self._configure_pipeline(
                    replacement,
                    use_cuda=use_cuda,
                    torch=torch,
                    track_quantization=False,
                )
        except Exception:
            if replacement is not None:
                del replacement
            raise

        previous = self._inpaint_pipeline
        self._inpaint_pipeline = replacement
        self._inpaint_identity = identity
        if previous is not None:
            self._remove_model_cpu_offload_hooks(previous)
            del previous
        return replacement

    def _unload_ip_adapter(self, pipeline: Any) -> None:
        if self._ip_adapter_identity is None:
            return
        self._remove_model_cpu_offload_hooks(pipeline)
        unload = getattr(pipeline, "unload_ip_adapter", None)
        if not callable(unload):
            raise RuntimeError("current Diffusers pipeline cannot unload IP-Adapter")
        unload()
        self._ip_adapter_identity = None
        self._ip_adapter_scale = None
        if self.config.enable_attention_slicing:
            enable_slicing = getattr(pipeline, "enable_attention_slicing", None)
            if not callable(enable_slicing):
                raise RuntimeError(
                    "current Diffusers pipeline cannot restore attention slicing"
                )
            enable_slicing()
        self._refresh_model_cpu_offload(pipeline)

    def _remove_model_cpu_offload_hooks(self, pipeline: Any) -> None:
        """Detach hooks while dynamically registered modules are still visible."""

        if not (
            self.config.enable_model_cpu_offload
            and self._device.startswith("cuda")
        ):
            return
        remove_hooks = getattr(pipeline, "remove_all_hooks", None)
        if not callable(remove_hooks):
            raise RuntimeError(
                "current Diffusers pipeline cannot detach model CPU offload hooks"
            )
        remove_hooks()

    def _refresh_model_cpu_offload(self, pipeline: Any) -> None:
        """Rebuild Accelerate hooks after optional pipeline modules change.

        Diffusers installs model-offload hooks before an IP-Adapter image
        encoder exists.  Loading the adapter later registers that 2.5 GB
        encoder on CPU, so reference encoding otherwise runs entirely on the
        host.  Rebuilding hooks is the supported Diffusers operation and also
        drops stale hooks after the adapter is unloaded.
        """

        if not (
            self.config.enable_model_cpu_offload
            and self._device.startswith("cuda")
        ):
            return
        enable_offload = getattr(pipeline, "enable_model_cpu_offload", None)
        if not callable(enable_offload):
            raise RuntimeError(
                "current Diffusers pipeline cannot refresh model CPU offload"
            )
        enable_offload(device=self._device)

    def _ensure_ip_adapter(
        self,
        pipeline: Any,
        artifact: DiffusionArtifact,
        *,
        scale: float,
    ) -> None:
        if artifact.artifact_kind != "sd15_ip_adapter" or not artifact.loadable:
            raise ValueError("reference mode requires a complete local SD1.5 IP-Adapter directory")
        if self.config.quantization != "none" or self.config.enable_qkv_fusion:
            raise RuntimeError(
                "IP-Adapter is not enabled for quantized or QKV-fused SD15 profiles before GPU validation"
            )
        identity = artifact.sha256 or artifact.path
        if self._ip_adapter_identity != identity:
            self._unload_ip_adapter(pipeline)
            layout = resolve_sd15_ip_adapter_layout(artifact.path)
            slicing_suspended = False
            try:
                if self.config.enable_attention_slicing:
                    disable_slicing = getattr(
                        pipeline,
                        "disable_attention_slicing",
                        None,
                    )
                    if not callable(disable_slicing):
                        raise RuntimeError(
                            "current Diffusers pipeline cannot suspend attention slicing for IP-Adapter"
                        )
                    disable_slicing()
                    slicing_suspended = True
                pipeline.load_ip_adapter(
                    layout["root"],
                    subfolder=layout["subfolder"],
                    weight_name=layout["weight_name"],
                    image_encoder_folder=layout["image_encoder_folder"],
                    local_files_only=True,
                )
                self._refresh_model_cpu_offload(pipeline)
            except Exception:
                try:
                    pipeline.unload_ip_adapter()
                except Exception:
                    pass
                if slicing_suspended:
                    try:
                        pipeline.enable_attention_slicing()
                    except Exception:
                        pass
                self._ip_adapter_identity = None
                self._ip_adapter_scale = None
                raise
            self._ip_adapter_identity = identity
        pipeline.set_ip_adapter_scale(scale)
        self._ip_adapter_scale = scale

    def generate(
        self,
        request: SD15GenerationRequest,
        *,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> SD15GenerationResult:
        request.validate()
        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("SD15 引擎尚未加载模型")
            self._cancel_event.clear()
            self._unload_ip_adapter(pipeline)

            import torch

            generator = torch.Generator(device="cpu").manual_seed(int(request.seed))

            def on_step_end(_pipeline: Any, step: int, _timestep: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                if self._cancel_event.is_set():
                    raise GenerationCancelled("SD15 生成已取消")
                if callback:
                    callback(step + 1, request.steps)
                return kwargs

            started = time.perf_counter()
            try:
                diffusers_logging = self._diffusers_logging
                if diffusers_logging is None:
                    from diffusers.utils import logging as diffusers_logging

                with self._request_scheduler(pipeline, request):
                    with self._suspended_diffusers_progress(diffusers_logging):
                        output = pipeline(
                            prompt=request.prompt,
                            negative_prompt=request.negative_prompt or None,
                            height=request.height,
                            width=request.width,
                            num_inference_steps=request.steps,
                            guidance_scale=request.guidance_scale,
                            generator=generator,
                            callback_on_step_end=on_step_end,
                        )
            except TypeError as exc:
                message = str(exc)
                if "callback_on_step_end" in message and (
                    "unexpected keyword" in message or "got an unexpected" in message
                ):
                    # Old Diffusers versions without callback_on_step_end must
                    # fail explicitly instead of silently losing cancellation.
                    raise RuntimeError(
                        "当前 Diffusers 版本不支持可取消 SD15 step callback，请升级 sidecar"
                    ) from exc
                raise
            elapsed = time.perf_counter() - started
            images = getattr(output, "images", None) or []
            if not images:
                raise RuntimeError("SD15 pipeline 返回空图像")
            nsfw_flags = getattr(output, "nsfw_content_detected", None)
            return SD15GenerationResult(
                image=images[0],
                seed=int(request.seed),
                elapsed_seconds=elapsed,
                metadata={
                    "engine": "diffusers_sd15",
                    "artifact_sha256": self._artifact.sha256 if self._artifact else "",
                    "device": self._device,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "scheduler": request.scheduler or type(
                        getattr(pipeline, "scheduler", None)
                    ).__name__,
                    "safety_flagged": bool(nsfw_flags and nsfw_flags[0]),
                    "capabilities": self.capabilities,
                },
            )

    def edit(
        self,
        request: Any,
        *,
        image: Any,
        mask: Any = None,
        adapter: Optional[DiffusionArtifact] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> SD15GenerationResult:
        '''Run local img2img, inpaint, or IP-Adapter reference generation.'''

        mode = getattr(request, 'mode', '')
        if mode not in {'img2img', 'reference', 'inpaint'}:
            raise RuntimeError('SD15 engine does not support this edit mode yet')
        request.validate()
        with self._lock:
            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError('SD15 engine is not loaded')
            self._cancel_event.clear()
            import torch

            generator = torch.Generator(device='cpu').manual_seed(int(request.seed))

            def on_step_end(
                _pipeline: Any,
                step: int,
                _timestep: Any,
                kwargs: Dict[str, Any],
            ) -> Dict[str, Any]:
                if self._cancel_event.is_set():
                    raise GenerationCancelled('SD15 edit cancelled')
                if callback:
                    callback(step + 1, request.denoising_steps)
                return kwargs

            if mode == 'reference':
                if adapter is None:
                    raise ValueError('reference mode requires an IP-Adapter artifact')
                self._ensure_ip_adapter(
                    pipeline,
                    adapter,
                    scale=float(request.ip_adapter_scale),
                )
                edit_pipeline = pipeline
            elif mode == 'inpaint':
                if adapter is None:
                    raise ValueError('inpaint mode requires a dedicated inpaint artifact')
                if mask is None:
                    raise ValueError('inpaint mode requires a mask image')
                self._unload_ip_adapter(pipeline)
                edit_pipeline = self._get_inpaint_pipeline(adapter)
            else:
                self._unload_ip_adapter(pipeline)
                edit_pipeline = self._get_img2img_pipeline(pipeline)
                # from_pipe shares model modules with the base pipeline, but
                # Accelerate's hook chain belongs to one pipeline instance at
                # a time. Rebind it before img2img so shared modules do not
                # silently execute on CPU after a reference-mode transition.
                self._refresh_model_cpu_offload(edit_pipeline)

            try:
                from PIL import Image
                source_image = image.convert('RGB') if hasattr(image, 'convert') else image
                if mode in {'img2img', 'inpaint'}:
                    source_image = source_image.resize(
                        (request.width, request.height),
                        Image.Resampling.LANCZOS,
                    )
                mask_image = None
                if mode == 'inpaint':
                    mask_image = mask.convert('L') if hasattr(mask, 'convert') else mask
                    mask_image = mask_image.resize(
                        (request.width, request.height),
                        Image.Resampling.NEAREST,
                    )
                started = time.perf_counter()
                diffusers_logging = self._diffusers_logging
                if diffusers_logging is None:
                    from diffusers.utils import logging as diffusers_logging
                with self._request_scheduler(edit_pipeline, request):
                    with self._suspended_diffusers_progress(diffusers_logging):
                        call_kwargs = {
                            'prompt': request.prompt,
                            'negative_prompt': request.negative_prompt or None,
                            'num_inference_steps': request.steps,
                            'guidance_scale': request.guidance_scale,
                            'generator': generator,
                            'callback_on_step_end': on_step_end,
                        }
                        if mode == 'reference':
                            call_kwargs.update({
                                'ip_adapter_image': source_image,
                                'height': request.height,
                                'width': request.width,
                            })
                        elif mode == 'inpaint':
                            call_kwargs.update({
                                'image': source_image,
                                'mask_image': mask_image,
                                'strength': request.strength,
                                'height': request.height,
                                'width': request.width,
                            })
                        else:
                            call_kwargs.update({
                                'image': source_image,
                                'strength': request.strength,
                            })
                        output = edit_pipeline(**call_kwargs)
            except TypeError as exc:
                message = str(exc)
                if 'callback_on_step_end' in message and (
                    'unexpected keyword' in message or 'got an unexpected' in message
                ):
                    raise RuntimeError(
                        'current Diffusers version lacks cancellable SD15 callbacks'
                    ) from exc
                raise
            elapsed = time.perf_counter() - started
            images = getattr(output, 'images', None) or []
            if not images:
                raise RuntimeError('SD15 pipeline returned no image')
            nsfw_flags = getattr(output, 'nsfw_content_detected', None)
            return SD15GenerationResult(
                image=images[0],
                seed=int(request.seed),
                elapsed_seconds=elapsed,
                metadata={
                    'engine': (
                        'diffusers_sd15_ip_adapter'
                        if mode == 'reference'
                        else 'diffusers_sd15_inpaint'
                        if mode == 'inpaint'
                        else 'diffusers_sd15_img2img'
                    ),
                    'edit_mode': mode,
                    'strength': request.strength if mode in {'img2img', 'inpaint'} else None,
                    'mask_semantics': (
                        'white=redraw, black=preserve' if mode == 'inpaint' else None
                    ),
                    'inpaint_sha256': (
                        adapter.sha256 if mode == 'inpaint' and adapter else None
                    ),
                    'ip_adapter_scale': (
                        request.ip_adapter_scale if mode == 'reference' else None
                    ),
                    'ip_adapter_sha256': (
                        adapter.sha256 if mode == 'reference' and adapter else None
                    ),
                    'artifact_sha256': self._artifact.sha256 if self._artifact else '',
                    'device': self._device,
                    'width': request.width,
                    'height': request.height,
                    'steps': request.steps,
                    'guidance_scale': request.guidance_scale,
                    'scheduler': request.scheduler or type(getattr(edit_pipeline, 'scheduler', None)).__name__,
                    'safety_flagged': bool(nsfw_flags and nsfw_flags[0]),
                    'capabilities': self.capabilities,
                },
            )


__all__ = [
    "GenerationCancelled",
    "SD15Engine",
    "SD15EngineConfig",
    "SD15GenerationRequest",
    "SD15GenerationResult",
]
