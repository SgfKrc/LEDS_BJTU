"""Optional local Stable Diffusion 1.5 engine.

The implementation keeps Diffusers imports lazy so importing QLH for the
existing LLM engines does not require the SD sidecar dependencies.  It is
intentionally local-only in this first step; distributed fan-out will wrap the
same engine behind a complete Worker after the local smoke passes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .artifacts import DiffusionArtifact, DiffusionArtifactInspector


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
                "enabled": self.config.enable_attention_slicing,
                "strategy": "diffusers_attention_slicing" if self.config.enable_attention_slicing else "none",
            },
            "operator_fusion": {
                "qkv_projection": self.config.enable_qkv_fusion,
                "torch_compile_unet": self.config.enable_torch_compile,
                "torch_compile_mode": self.config.torch_compile_mode if self.config.enable_torch_compile else None,
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
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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

    def _configure_pipeline(self, pipeline: Any, *, use_cuda: bool, torch: Any) -> Any:
        """Apply mutually validated memory and execution strategies."""

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
            self._quantized_linear_count = self._replace_linear_with_8bit_modules(
                unet,
                torch=torch,
                bnb=bnb,
            )
            if self._quantized_linear_count == 0:
                raise RuntimeError("SD15 U-Net has no Linear layers eligible for 8-bit quantization")

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
            except ImportError as exc:
                raise ImportError(
                    "SD15 sidecar 未安装，请使用当前 CUDA venv 安装 "
                    "packaging/requirements-sd15.txt"
                ) from exc

            self._device = "cuda" if use_cuda else "cpu"
            dtype = torch.float16 if use_cuda and self.config.dtype == "float16" else torch.float32
            load_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "local_files_only": True,
                "use_safetensors": self.config.use_safetensors,
            }
            if self.config.variant:
                load_kwargs["variant"] = self.config.variant
            if not self.config.safety_checker_required:
                load_kwargs["safety_checker"] = None
            if revision:
                load_kwargs["revision"] = revision
            pipeline: Any = None
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(local_model_path, **load_kwargs)
                pipeline = self._configure_pipeline(pipeline, use_cuda=use_cuda, torch=torch)
            except Exception:
                if pipeline is not None:
                    del pipeline
                self._reset_failed_load()
                raise
            self._pipeline = pipeline
            self._artifact = inspected
            return inspected

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._artifact = None
            self._quantized_linear_count = 0
            self._device = "cpu"
            if pipeline is not None:
                del pipeline
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
                    "capabilities": self.capabilities,
                },
            )


__all__ = [
    "GenerationCancelled",
    "SD15Engine",
    "SD15EngineConfig",
    "SD15GenerationRequest",
    "SD15GenerationResult",
]
