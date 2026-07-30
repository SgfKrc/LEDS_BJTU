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
    local_files_only: bool = True
    safety_checker_required: bool = True


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

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def artifact(self) -> Optional[DiffusionArtifact]:
        return self._artifact

    @property
    def device(self) -> str:
        return self._device

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
            if self._pipeline is not None:
                raise RuntimeError("SD15 引擎已经加载模型，请先 unload")

            from pathlib import Path

            if local_files_only is False or not self.config.local_files_only:
                raise ValueError("SD15 首期只允许加载已下载的本地 Diffusers 目录")

            local_model_path = Path(model_path).expanduser()
            if not local_model_path.is_dir():
                raise ValueError("离线 SD15 加载必须传入已下载的 Diffusers 目录")

            import torch

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

            use_cuda = self.config.device.startswith("cuda") and torch.cuda.is_available()
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
            pipeline = StableDiffusionPipeline.from_pretrained(local_model_path, **load_kwargs)

            if self.config.enable_attention_slicing:
                pipeline.enable_attention_slicing()
            if self.config.enable_vae_slicing:
                vae = getattr(pipeline, "vae", None)
                if vae is not None and hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                elif hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
            if use_cuda and self.config.enable_model_cpu_offload:
                # Diffusers uses Accelerate to move one component at a time and
                # keeps 512x512 SD1.5 within the 8GB laptop GPU budget.
                pipeline.enable_model_cpu_offload()
            else:
                pipeline = pipeline.to(self._device)
            self._pipeline = pipeline
            self._artifact = inspected
            return inspected

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._artifact = None
            if pipeline is not None:
                del pipeline
            try:
                import torch

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
                },
            )


__all__ = [
    "GenerationCancelled",
    "SD15Engine",
    "SD15EngineConfig",
    "SD15GenerationRequest",
    "SD15GenerationResult",
]
