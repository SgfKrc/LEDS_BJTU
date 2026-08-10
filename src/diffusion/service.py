"""Local lifecycle service for the optional Stable Diffusion 1.5 sidecar."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from runtime_env import sd_runtime_diagnostics

from .artifacts import DiffusionArtifact, DiffusionArtifactInspector
from .assets import MANIFEST_NAME, DiffusionAssetManager, DiffusionAssetSpec
from .presets import get_preset
from .sd15_engine import (
    GenerationCancelled,
    SD15Engine,
    SD15EngineConfig,
    SD15GenerationRequest,
)


class DiffusionServiceError(RuntimeError):
    """Base error for stable API error mapping."""

    code = 'DIFFUSION_ERROR'


class DiffusionConflictError(DiffusionServiceError):
    """Raised when another lifecycle operation owns the local SD engine."""

    code = 'DIFFUSION_CONFLICT'


class DiffusionBlobInUseError(DiffusionConflictError):
    """Raised when a caller attempts to delete an actively leased blob."""

    code = 'DIFFUSION_BLOB_IN_USE'


class DiffusionBlobReferencedError(DiffusionConflictError):
    """Raised when a result blob still keeps a parent input alive."""

    code = 'DIFFUSION_BLOB_REFERENCED'


class DiffusionNotFoundError(DiffusionServiceError):
    """Raised when an artifact, job, or blob does not exist."""

    code = 'DIFFUSION_NOT_FOUND'


class DiffusionInputError(DiffusionServiceError):
    """Raised when an image or edit request violates the public contract."""

    code = 'DIFFUSION_INVALID_INPUT'


class DiffusionUnsupportedError(DiffusionServiceError):
    """Raised when a valid request has no installed executor yet."""

    code = 'DIFFUSION_UNSUPPORTED'


SD15_LOAD_PROFILES = (
    "balanced",
    "resident_fp16",
    "qkv_fp16",
    "unet_8bit",
    "unet_8bit_qkv",
)

# Keep the ingress limit in one place.  The gateway and both FastAPI surfaces
# import this value so a proxy cannot buffer a larger upload than the worker
# will ever accept.
DIFFUSION_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DIFFUSION_MAX_IMAGE_PIXELS = 16 * 1024 * 1024
DIFFUSION_UNLOAD_WAIT_SECONDS = 30.0


def build_sd15_engine_config(
    profile: str,
    *,
    safety_checker_required: bool = True,
) -> SD15EngineConfig:
    profiles = {
        "balanced": dict(
            enable_model_cpu_offload=True,
            enable_attention_slicing=True,
            enable_qkv_fusion=False,
            quantization="none",
        ),
        "resident_fp16": dict(
            enable_model_cpu_offload=False,
            enable_attention_slicing=True,
            enable_qkv_fusion=False,
            quantization="none",
        ),
        "qkv_fp16": dict(
            enable_model_cpu_offload=True,
            enable_attention_slicing=False,
            enable_qkv_fusion=True,
            quantization="none",
        ),
        "unet_8bit": dict(
            enable_model_cpu_offload=False,
            enable_attention_slicing=True,
            enable_qkv_fusion=False,
            quantization="bitsandbytes_8bit_unet",
        ),
        "unet_8bit_qkv": dict(
            enable_model_cpu_offload=False,
            enable_attention_slicing=False,
            enable_qkv_fusion=True,
            quantization="bitsandbytes_8bit_unet",
        ),
    }
    try:
        options = profiles[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported SD15 load profile: {profile}") from exc
    return SD15EngineConfig(
        device="cuda",
        dtype="float16",
        variant="fp16",
        enable_vae_slicing=True,
        local_files_only=True,
        safety_checker_required=bool(safety_checker_required),
        **options,
    )


def build_sd15_generation_request(
    *,
    preset_id: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    scheduler: Optional[str] = None,
) -> SD15GenerationRequest:
    preset = None
    if preset_id:
        try:
            preset = get_preset(preset_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    generation = SD15GenerationRequest(
        prompt=prompt if prompt is not None else (preset.prompt if preset else ""),
        negative_prompt=(
            negative_prompt
            if negative_prompt is not None
            else (preset.negative_prompt if preset else "")
        ),
        seed=int(seed if seed is not None else (preset.seeds[0] if preset else 0)),
        width=int(width if width is not None else (preset.width if preset else 512)),
        height=int(height if height is not None else (preset.height if preset else 512)),
        steps=int(steps if steps is not None else (preset.steps if preset else 28)),
        guidance_scale=float(
            guidance_scale
            if guidance_scale is not None
            else (preset.guidance_scale if preset else 7.5)
        ),
        scheduler=(
            scheduler
            if scheduler is not None
            else (preset.scheduler if preset else "")
        ),
    )
    generation.validate()
    return generation


@dataclass(frozen=True)
class RegisteredDiffusionArtifact:
    artifact_id: str
    name: str
    artifact: DiffusionArtifact
    registered_at: float

    def snapshot(self, *, include_path: bool = False) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "registered_at": self.registered_at,
            "artifact": self.artifact.to_dict(include_path=include_path),
        }


@dataclass(frozen=True)
class SD15EditRequest:
    mode: str
    source_blob_id: str
    prompt: str = ''
    negative_prompt: str = ''
    seed: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = 28
    guidance_scale: float = 7.5
    strength: float = 0.75
    mask_blob_id: Optional[str] = None
    instruction: Optional[str] = None
    edit_adapter_id: Optional[str] = None
    conditioning_scale: Optional[float] = None
    image_guidance_scale: Optional[float] = None
    ip_adapter_scale: Optional[float] = None
    scheduler: str = ''

    @property
    def denoising_steps(self) -> int:
        if self.mode in {'img2img', 'inpaint'}:
            return int(self.steps * self.strength)
        return self.steps

    def validate(self) -> None:
        if self.mode not in {'img2img', 'reference', 'inpaint', 'instruction'}:
            raise DiffusionInputError('unsupported edit mode')
        if not self.source_blob_id.strip():
            raise DiffusionInputError('source_blob_id is required')
        if not self.prompt.strip():
            raise DiffusionInputError('prompt is required')
        if self.mode == 'inpaint':
            if not self.mask_blob_id:
                raise DiffusionInputError('mask_blob_id is required for inpaint')
        elif self.mask_blob_id:
            raise DiffusionInputError('mask_blob_id is only valid for inpaint')
        normalized_instruction = (self.instruction or '').strip()
        if self.mode == 'instruction':
            if not normalized_instruction:
                raise DiffusionInputError('instruction is required for instruction mode')
            if self.prompt.strip() != normalized_instruction:
                raise DiffusionInputError(
                    'prompt must match instruction for instruction mode'
                )
        elif normalized_instruction:
            raise DiffusionInputError('instruction is only valid for instruction mode')
        normalized_adapter_id = (self.edit_adapter_id or '').strip()
        if self.mode in {'reference', 'instruction'}:
            if not normalized_adapter_id:
                raise DiffusionInputError(
                    f'edit_adapter_id is required for {self.mode} mode'
                )
        if self.mode == 'reference':
            if self.ip_adapter_scale is None:
                raise DiffusionInputError(
                    'ip_adapter_scale is required for reference mode'
                )
        elif normalized_adapter_id and self.mode not in {'instruction', 'inpaint'}:
            raise DiffusionInputError(
                'edit_adapter_id is only valid for reference, inpaint, or instruction mode'
            )
        if self.mode != 'reference' and self.ip_adapter_scale is not None:
            raise DiffusionInputError(
                'ip_adapter_scale is only valid for reference mode'
            )
        if self.mode == 'instruction':
            if self.image_guidance_scale is None:
                raise DiffusionInputError(
                    'image_guidance_scale is required for instruction mode'
                )
            if self.conditioning_scale is not None:
                raise DiffusionInputError(
                    'conditioning_scale is not valid for the InstructPix2Pix pipeline'
                )
        elif self.image_guidance_scale is not None:
            raise DiffusionInputError(
                'image_guidance_scale is only valid for instruction mode'
            )
        elif self.conditioning_scale is not None:
            raise DiffusionInputError(
                'conditioning_scale is only valid for a ControlNet instruction pipeline'
            )
        if not math.isfinite(self.strength) or not 0.05 <= self.strength <= 1.0:
            raise DiffusionInputError('strength must be finite and between 0.05 and 1.0')
        if self.steps < 1 or self.steps > 100:
            raise DiffusionInputError('steps must be between 1 and 100')
        if self.mode in {'img2img', 'inpaint'} and self.denoising_steps < 1:
            raise DiffusionInputError(
                'steps and strength must produce at least one denoising step'
            )
        if self.scheduler not in {
            '',
            'PNDMScheduler',
            'DPMSolverMultistepScheduler',
            'EulerDiscreteScheduler',
            'DDIMScheduler',
        }:
            raise DiffusionInputError('unsupported SD15 scheduler')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise DiffusionInputError('guidance_scale must be finite and non-negative')
        for name, value in (
            ('conditioning_scale', self.conditioning_scale),
            ('image_guidance_scale', self.image_guidance_scale),
            ('ip_adapter_scale', self.ip_adapter_scale),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise DiffusionInputError(f'{name} must be finite and non-negative')
        if self.ip_adapter_scale is not None and self.ip_adapter_scale > 2:
            raise DiffusionInputError('ip_adapter_scale must not exceed 2')
        if self.image_guidance_scale is not None and self.image_guidance_scale > 4:
            raise DiffusionInputError('image_guidance_scale must not exceed 4')
        for name, value in (('width', self.width), ('height', self.height)):
            if value is not None and (value < 64 or value > 768 or value % 8):
                raise DiffusionInputError(f'{name} must be a multiple of 8 between 64 and 768')

    def snapshot(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'source_blob_id': self.source_blob_id,
            'mask_blob_id': self.mask_blob_id,
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'seed': self.seed,
            'width': self.width,
            'height': self.height,
            'steps': self.steps,
            'denoising_steps': self.denoising_steps,
            'guidance_scale': self.guidance_scale,
            'strength': self.strength,
            'edit_adapter_id': self.edit_adapter_id,
            'conditioning_scale': self.conditioning_scale,
            'image_guidance_scale': self.image_guidance_scale,
            'ip_adapter_scale': self.ip_adapter_scale,
            'instruction': self.instruction,
            'scheduler': self.scheduler,
        }


@dataclass(frozen=True)
class ImageBlob:
    blob_id: str
    content_type: str
    data: bytes
    sha256: str
    created_at: float
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    purpose: str = 'output'
    owner_scope: str = 'local'
    width: Optional[int] = None
    height: Optional[int] = None
    parent_blob_ids: tuple[str, ...] = ()
    lease_count: int = 0
    reference_count: int = 0
    pending_delete: bool = False

    def descriptor(self) -> Dict[str, Any]:
        return {
            'purpose': self.purpose,
            'owner_scope': self.owner_scope,
            'width': self.width,
            'height': self.height,
            'parent_blob_ids': list(self.parent_blob_ids),
            'lease_count': self.lease_count,
            'reference_count': self.reference_count,
            'pending_delete': self.pending_delete,
            "blob_id": self.blob_id,
            "content_type": self.content_type,
            "size_bytes": len(self.data),
            "sha256": self.sha256,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }


class MemoryImageBlobStore:
    """Small bounded result store used before the distributed blob protocol exists."""

    def __init__(
        self,
        *,
        max_items: int = 16,
        max_total_bytes: int = 64 * 1024 * 1024,
        ttl_seconds: float = 30 * 60,
        max_upload_bytes: int = DIFFUSION_MAX_UPLOAD_BYTES,
        max_image_pixels: int = DIFFUSION_MAX_IMAGE_PIXELS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_upload_bytes <= 0 or max_image_pixels <= 0:
            raise ValueError('blob upload limits must be positive')
        if max_items <= 0 or max_total_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("blob store limits must be positive")
        self.max_items = int(max_items)
        self.max_total_bytes = int(max_total_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self.max_upload_bytes = int(max_upload_bytes)
        self.max_image_pixels = int(max_image_pixels)
        self._clock = clock
        self._items: Dict[str, ImageBlob] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()

    def _purge_expired_locked(self, now: float) -> None:
        # Removing a result can release the last reference to an expired
        # input, so repeat until no newly-unblocked parent remains.
        while True:
            expired = [
                blob_id
                for blob_id, blob in self._items.items()
                if (
                    blob.expires_at <= now
                    and blob.lease_count == 0
                    and blob.reference_count == 0
                )
            ]
            if not expired:
                return
            for blob_id in expired:
                self._remove_locked(blob_id)

    def _remove_locked(self, blob_id: str) -> Optional[ImageBlob]:
        blob = self._items.pop(blob_id, None)
        if blob is None:
            return None
        self._total_bytes -= len(blob.data)
        for parent_id in blob.parent_blob_ids:
            parent = self._items.get(parent_id)
            if parent is not None and parent.reference_count:
                self._items[parent_id] = replace(
                    parent,
                    reference_count=parent.reference_count - 1,
                )
        return blob

    def _evict_oldest_locked(self, *, exclude: frozenset[str] = frozenset()) -> bool:
        candidates = {
            blob_id: blob
            for blob_id, blob in self._items.items()
            if (
                blob_id not in exclude
                and blob.lease_count == 0
                and blob.reference_count == 0
            )
        }
        if not candidates:
            return False
        blob_id = min(
            candidates,
            key=lambda key: (candidates[key].created_at, key),
        )
        return self._remove_locked(blob_id) is not None

    @staticmethod
    def _normalize_upload(data: bytes, *, purpose: str, max_bytes: int, max_pixels: int) -> tuple[bytes, int, int]:
        if purpose not in {'input_image', 'mask'}:
            raise DiffusionInputError('purpose must be input_image or mask')
        if not data:
            raise DiffusionInputError('image upload is empty')
        if len(data) > max_bytes:
            raise DiffusionInputError('image upload exceeds the byte limit')
        try:
            from PIL import Image, ImageOps
            source = Image.open(io.BytesIO(data))
            image_format = (source.format or '').upper()
            if image_format not in {'PNG', 'JPEG', 'WEBP'}:
                raise DiffusionInputError('only PNG, JPEG, and WebP images are accepted')
            if getattr(source, 'n_frames', 1) != 1:
                raise DiffusionInputError('animated or multi-page images are not accepted')
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise DiffusionInputError('image dimensions exceed the pixel limit')
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                source.load()
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            if width * height > max_pixels:
                raise DiffusionInputError('image dimensions exceed the pixel limit after orientation')
            image = image.convert('L' if purpose == 'mask' else 'RGB')
            output = io.BytesIO()
            image.save(output, format='PNG', optimize=False)
            normalized = output.getvalue()
        except DiffusionInputError:
            raise
        except Exception as exc:
            raise DiffusionInputError('uploaded file is not a valid supported image') from exc
        if not normalized:
            raise DiffusionInputError('normalized image is empty')
        return normalized, width, height

    @staticmethod
    def encode_png(image: Any) -> bytes:
        """Serialize an image without mutating the bounded store."""

        output = io.BytesIO()
        image.save(output, format="PNG")
        data = output.getvalue()
        if not data:
            raise DiffusionServiceError("SD15 pipeline produced an empty PNG")
        return data

    def put_png(
        self,
        data: bytes,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        purpose: str = 'output',
        owner_scope: str = 'local',
        width: Optional[int] = None,
        height: Optional[int] = None,
        parent_blob_ids: tuple[str, ...] = (),
    ) -> ImageBlob:
        data = bytes(data)
        if not data:
            raise DiffusionServiceError("SD15 pipeline produced an empty PNG")
        if len(data) > self.max_total_bytes:
            raise DiffusionServiceError("generated image exceeds the local blob limit")

        now = self._clock()
        normalized_parent_ids = tuple(dict.fromkeys(parent_blob_ids))
        blob = ImageBlob(
            blob_id=f"img_{uuid.uuid4().hex}",
            content_type="image/png",
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            created_at=now,
            expires_at=now + self.ttl_seconds,
            purpose=purpose,
            owner_scope=owner_scope,
            width=width,
            height=height,
            parent_blob_ids=normalized_parent_ids,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._purge_expired_locked(now)
            for parent_id in normalized_parent_ids:
                if parent_id not in self._items:
                    raise DiffusionNotFoundError(
                        f'parent image blob not found: {parent_id}'
                    )
            while self._items and (
                len(self._items) >= self.max_items
                or self._total_bytes + len(data) > self.max_total_bytes
            ):
                if not self._evict_oldest_locked(
                    exclude=frozenset(normalized_parent_ids)
                ):
                    raise DiffusionConflictError('blob store is full of leased images')
            self._items[blob.blob_id] = blob
            for parent_id in normalized_parent_ids:
                parent = self._items[parent_id]
                self._items[parent_id] = replace(
                    parent,
                    reference_count=parent.reference_count + 1,
                )
            self._total_bytes += len(data)
        return blob

    def put_image(
        self,
        image: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageBlob:
        return self.put_png(self.encode_png(image), metadata=metadata)

    def put_upload(
        self,
        data: bytes,
        *,
        purpose: str,
        owner_scope: str,
    ) -> ImageBlob:
        original = bytes(data)
        normalized, width, height = self._normalize_upload(
            original,
            purpose=purpose,
            max_bytes=self.max_upload_bytes,
            max_pixels=self.max_image_pixels,
        )
        return self.put_png(
            normalized,
            purpose=purpose,
            owner_scope=owner_scope,
            width=width,
            height=height,
            metadata={
                'upload_sha256': hashlib.sha256(original).hexdigest(),
                'normalized_format': 'PNG',
            },
        )

    def get(self, blob_id: str) -> ImageBlob:
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            blob = self._items.get(blob_id)
            if blob is None:
                raise DiffusionNotFoundError(f"image blob not found: {blob_id}")
            return blob

    def acquire_lease(self, blob_id: str) -> ImageBlob:
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            blob = self._items.get(blob_id)
            if blob is None:
                raise DiffusionNotFoundError(f'image blob not found: {blob_id}')
            leased = replace(blob, lease_count=blob.lease_count + 1)
            self._items[blob_id] = leased
            return leased

    def release_lease(self, blob_id: str) -> ImageBlob:
        with self._lock:
            blob = self._items.get(blob_id)
            if blob is None:
                raise DiffusionNotFoundError(f'image blob not found: {blob_id}')
            if blob.lease_count <= 0:
                raise DiffusionConflictError(f'image blob has no active lease: {blob_id}')
            released = replace(blob, lease_count=blob.lease_count - 1)
            self._items[blob_id] = released
            return released

    def delete(self, blob_id: str) -> bool:
        with self._lock:
            self._purge_expired_locked(self._clock())
            blob = self._items.get(blob_id)
            if blob is None:
                return False
            if blob.lease_count:
                raise DiffusionBlobInUseError(f'image blob is in use: {blob_id}')
            if blob.reference_count:
                raise DiffusionBlobReferencedError(
                    f'image blob is referenced by {blob.reference_count} result(s): {blob_id}'
                )
            self._remove_locked(blob_id)
            return True

    def snapshot(self) -> Dict[str, Any]:
        now = self._clock()
        with self._lock:
            self._purge_expired_locked(now)
            return {
                "items": len(self._items),
                "total_bytes": self._total_bytes,
                "max_items": self.max_items,
                "max_total_bytes": self.max_total_bytes,
                "ttl_seconds": self.ttl_seconds,
                "max_upload_bytes": self.max_upload_bytes,
                "max_image_pixels": self.max_image_pixels,
                "referenced_items": sum(
                    1 for blob in self._items.values() if blob.reference_count
                ),
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total_bytes = 0


@dataclass
class DiffusionJob:
    job_id: str
    artifact_id: str
    request: Any
    kind: str = 'generate'
    source_blob_ids: tuple[str, ...] = ()
    owner_scope: str = 'local'
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress_step: int = 0
    progress_total: int = 0
    cancel_requested: bool = False
    blob: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_code: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        if self.kind == 'edit':
            parameters = self.request.snapshot()
        else:
            parameters = {
                'prompt': self.request.prompt,
                'negative_prompt': self.request.negative_prompt,
                'seed': self.request.seed,
                'width': self.request.width,
                'height': self.request.height,
                'steps': self.request.steps,
                'guidance_scale': self.request.guidance_scale,
                'scheduler': self.request.scheduler,
            }
        return {
            "job_id": self.job_id,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "owner_scope": self.owner_scope,
            "input_blob_ids": list(self.source_blob_ids),
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": {
                "step": self.progress_step,
                "total": self.progress_total,
            },
            "cancel_requested": self.cancel_requested,
            "parameters": parameters,
            "blob": dict(self.blob) if self.blob else None,
            "output_blob_id": self.blob.get("blob_id") if self.blob else None,
            "metrics": dict(self.metrics),
            "error": self.error,
            "error_code": self.error_code,
        }


class DiffusionService:
    """Own one local SD pipeline and one cancellable generation at a time."""

    TERMINAL_STATES = {"completed", "failed", "cancelled"}

    def __init__(
        self,
        *,
        inspector: Optional[DiffusionArtifactInspector] = None,
        engine_factory: Callable[[SD15EngineConfig], SD15Engine] = SD15Engine,
        blob_store: Optional[MemoryImageBlobStore] = None,
        asset_manager: Optional[DiffusionAssetManager] = None,
        max_jobs: int = 64,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")
        self._inspector = inspector or DiffusionArtifactInspector()
        self._engine_factory = engine_factory
        self._blob_store = blob_store or MemoryImageBlobStore(clock=clock)
        self._max_jobs = int(max_jobs)
        self._clock = clock
        self._artifacts: Dict[str, RegisteredDiffusionArtifact] = {}
        self._jobs: Dict[str, DiffusionJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sd15")
        self._engine: Optional[SD15Engine] = None
        self._loaded_artifact_id: Optional[str] = None
        self._engine_config: Optional[SD15EngineConfig] = None
        self._active_job_id: Optional[str] = None
        self._state = "unloaded"
        self._last_error: Optional[str] = None
        self._closed = False
        self._lock = threading.RLock()
        self._job_finished = threading.Condition(self._lock)
        self._asset_manager = asset_manager or DiffusionAssetManager(
            on_ready=self._register_catalog_asset,
        )
        # Custom inspectors are test/integration injection points and may not
        # recognize the real catalog directories. Auto-discovery belongs only
        # to the default production inspector.
        if asset_manager is None and inspector is None:
            self._asset_manager.discover_installed()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return bool(self._engine is not None and self._engine.is_loaded)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state in {"loading", "unloading"} or self._active_job_id is not None

    def inspect(self, path: str, *, compute_hash: bool = False) -> DiffusionArtifact:
        return self._inspector.inspect(path, compute_hash=compute_hash)

    @staticmethod
    def _default_artifact_id(artifact: DiffusionArtifact) -> str:
        identity = artifact.sha256 or hashlib.sha256(
            str(Path(artifact.path).resolve()).encode("utf-8")
        ).hexdigest()
        return f"sd15_{identity[:16]}"

    def register_artifact(
        self,
        path: str,
        *,
        artifact_id: Optional[str] = None,
        name: Optional[str] = None,
        compute_hash: bool = False,
        _trusted_sha256: Optional[str] = None,
    ) -> RegisteredDiffusionArtifact:
        artifact = self.inspect(path, compute_hash=compute_hash)
        if artifact.artifact_kind in {
            'sd15_ip_adapter',
            'sd15_inpaint_pipeline',
            'sd15_instruction_pipeline',
        } and not artifact.loadable:
            reason = '; '.join(artifact.warnings) or f'incomplete {artifact.artifact_kind} directory'
            raise ValueError(reason)
        if _trusted_sha256:
            artifact = replace(artifact, sha256=_trusted_sha256)
        if artifact.artifact_kind in {
            'sd15_ip_adapter',
            'sd15_inpaint_pipeline',
            'sd15_instruction_pipeline',
        } and not artifact.sha256:
            # Adapter identity participates in every result manifest.  A path
            # alone is not stable enough when users replace local weights.
            artifact = self.inspect(path, compute_hash=True)
        if artifact.artifact_kind == "unknown":
            reason = "; ".join(artifact.warnings) or "unrecognized SD artifact"
            raise ValueError(reason)
        resolved_id = (artifact_id or self._default_artifact_id(artifact)).strip()
        if not resolved_id or len(resolved_id) > 80:
            raise ValueError("artifact_id must contain 1-80 characters")
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in resolved_id):
            raise ValueError("artifact_id contains unsupported characters")
        registered = RegisteredDiffusionArtifact(
            artifact_id=resolved_id,
            name=(name or Path(artifact.path).name or resolved_id).strip()[:120],
            artifact=artifact,
            registered_at=self._clock(),
        )
        with self._lock:
            self._ensure_open_locked()
            existing = self._artifacts.get(resolved_id)
            if existing and Path(existing.artifact.path) != Path(artifact.path):
                raise DiffusionConflictError(f"artifact_id already points to another path: {resolved_id}")
            self._artifacts[resolved_id] = registered
        return registered

    def list_artifacts(self, *, include_path: bool = False) -> list[Dict[str, Any]]:
        with self._lock:
            values = list(self._artifacts.values())
        values.sort(key=lambda item: (item.name.lower(), item.artifact_id))
        return [item.snapshot(include_path=include_path) for item in values]

    def _register_catalog_asset(self, spec: DiffusionAssetSpec, path: Path) -> None:
        trusted_sha256 = ''
        try:
            manifest = json.loads((path / MANIFEST_NAME).read_text(encoding='utf-8'))
            candidate = str(manifest.get('artifact_sha256', ''))
            manifest_kind = str(manifest.get('asset', {}).get('artifact_kind', ''))
            if (
                len(candidate) == 64
                and all(char in '0123456789abcdef' for char in candidate.lower())
                and manifest_kind == spec.artifact_kind
            ):
                trusted_sha256 = candidate.lower()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
        self.register_artifact(
            str(path),
            artifact_id=spec.artifact_id,
            name=spec.name,
            compute_hash=False,
            _trusted_sha256=trusted_sha256 or None,
        )

    def asset_catalog(self) -> list[Dict[str, Any]]:
        self._asset_manager.discover_installed()
        return self._asset_manager.catalog()

    def asset_status(self, asset_id: str) -> Dict[str, Any]:
        return self._asset_manager.status(asset_id)

    def download_asset(
        self,
        asset_id: str,
        *,
        license_accepted: bool,
        proxy_fallback: str,
    ) -> Dict[str, Any]:
        try:
            return self._asset_manager.start_download(
                asset_id,
                license_accepted=license_accepted,
                proxy_fallback=proxy_fallback,
            )
        except RuntimeError as exc:
            raise DiffusionConflictError(str(exc)) from exc

    def import_asset(
        self,
        asset_id: str,
        path: str,
        *,
        license_accepted: bool,
    ) -> Dict[str, Any]:
        return self._asset_manager.import_asset(
            asset_id,
            path,
            license_accepted=license_accepted,
        )

    def _get_artifact_locked(self, artifact_id: str) -> RegisteredDiffusionArtifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise DiffusionNotFoundError(f"diffusion artifact not found: {artifact_id}")
        return artifact

    def load(self, artifact_id: str, config: SD15EngineConfig) -> Dict[str, Any]:
        config.validate()
        with self._lock:
            self._ensure_open_locked()
            artifact = self._get_artifact_locked(artifact_id)
            if artifact.artifact.artifact_kind != "sd15_pipeline" or not artifact.artifact.loadable:
                raise ValueError("current SD15 engine only loads complete local Diffusers pipeline directories")
            if self._active_job_id is not None or self._state in {"loading", "unloading"}:
                raise DiffusionConflictError("diffusion engine is busy")
            if self._engine is not None and self._engine.is_loaded:
                if self._loaded_artifact_id == artifact_id and self._engine_config == config:
                    return self.snapshot()
                raise DiffusionConflictError("another SD artifact is loaded; unload it before switching")
            self._state = "loading"
            self._last_error = None

        engine: Optional[SD15Engine] = None
        try:
            engine = self._engine_factory(config)
            engine.load(artifact.artifact.path)
        except Exception as exc:
            if engine is not None:
                try:
                    engine.unload()
                except Exception:
                    pass
            with self._lock:
                self._state = "error"
                self._last_error = str(exc)[:500]
            raise

        with self._lock:
            if self._closed:
                engine.unload()
                raise DiffusionConflictError("diffusion service is closed")
            self._engine = engine
            self._loaded_artifact_id = artifact_id
            self._engine_config = config
            self._state = "loaded"
            self._last_error = None
            return self.snapshot()

    def unload(self) -> Dict[str, Any]:
        with self._lock:
            self._ensure_open_locked()
            if self._state == "loading":
                raise DiffusionConflictError("diffusion engine is loading")
            engine = self._engine
            active_job_id = self._active_job_id
            if engine is None:
                self._state = "unloaded"
                return self.snapshot()
            self._state = "unloading"
            if active_job_id:
                job = self._jobs.get(active_job_id)
                if job and job.state not in self.TERMINAL_STATES:
                    job.cancel_requested = True
        if active_job_id:
            engine.cancel()
            deadline = time.monotonic() + DIFFUSION_UNLOAD_WAIT_SECONDS
            with self._job_finished:
                while self._active_job_id == active_job_id:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # Keep the live pipeline intact.  Unloading it while
                        # the executor is still inside Diffusers is unsafe.
                        self._state = "loaded"
                        self._last_error = (
                            "SD15 active job did not stop before unload timeout"
                        )
                        raise DiffusionConflictError(
                            "SD15 active job did not stop before unload timeout"
                        )
                    self._job_finished.wait(timeout=min(remaining, 0.25))
        try:
            engine.unload()
        finally:
            with self._lock:
                self._engine = None
                self._loaded_artifact_id = None
                self._engine_config = None
                self._state = "unloaded"
        return self.snapshot()

    def submit_generation(
        self,
        request: SD15GenerationRequest,
        *,
        owner_scope: str = 'local',
    ) -> Dict[str, Any]:
        request.validate()
        normalized_owner = owner_scope.strip()[:128] or 'local'
        with self._lock:
            self._ensure_open_locked()
            if self._engine is None or not self._engine.is_loaded:
                raise DiffusionConflictError("SD15 engine is not loaded")
            if self._state != "loaded" or self._active_job_id is not None:
                raise DiffusionConflictError("another SD15 generation is active")
            artifact_id = self._loaded_artifact_id
            if not artifact_id:
                raise DiffusionServiceError("loaded SD artifact identity is missing")
            job = DiffusionJob(
                job_id=f"sdjob_{uuid.uuid4().hex}",
                artifact_id=artifact_id,
                request=request,
                owner_scope=normalized_owner,
                created_at=self._clock(),
                progress_total=request.steps,
            )
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            self._last_error = None
            self._trim_jobs_locked()
            try:
                self._executor.submit(self._run_job, job.job_id)
            except Exception:
                self._jobs.pop(job.job_id, None)
                self._active_job_id = None
                raise
            return job.snapshot()

    def put_input_blob(
        self,
        data: bytes,
        *,
        purpose: str,
        owner_scope: str = 'local',
    ) -> Dict[str, Any]:
        normalized_owner = owner_scope.strip()[:128] or 'local'
        with self._lock:
            self._ensure_open_locked()
        return self._blob_store.put_upload(
            data,
            purpose=purpose,
            owner_scope=normalized_owner,
        ).descriptor()

    def validate_edit(
        self,
        request: SD15EditRequest,
        *,
        owner_scope: str = 'local',
    ) -> Dict[str, Any]:
        request.validate()
        source = self._blob_store.get(request.source_blob_id)
        if source.purpose not in {'input_image', 'output'}:
            raise DiffusionInputError(
                'source_blob_id must reference an input_image or output blob'
            )
        normalized_owner = owner_scope.strip()[:128] or 'local'
        if source.owner_scope != normalized_owner:
            raise DiffusionNotFoundError(f'image blob not found: {request.source_blob_id}')
        mask = None
        if request.mask_blob_id:
            mask = self._blob_store.get(request.mask_blob_id)
            if mask.purpose != 'mask':
                raise DiffusionInputError('mask_blob_id must reference a mask blob')
            if mask.owner_scope != normalized_owner:
                raise DiffusionNotFoundError(f'image blob not found: {request.mask_blob_id}')
            if (source.width, source.height) != (mask.width, mask.height):
                raise DiffusionInputError('mask dimensions must match the source image')
        adapter = None
        if request.mode == 'reference':
            with self._lock:
                adapter = self._get_artifact_locked(request.edit_adapter_id or '')
            if (
                adapter.artifact.artifact_kind != 'sd15_ip_adapter'
                or not adapter.artifact.loadable
            ):
                raise DiffusionInputError(
                    'edit_adapter_id must reference a complete SD1.5 IP-Adapter directory'
                )
        elif request.mode == 'inpaint':
            if not (request.edit_adapter_id or '').strip():
                raise DiffusionInputError(
                    'edit_adapter_id is required for inpaint mode'
                )
            with self._lock:
                adapter = self._get_artifact_locked(request.edit_adapter_id or '')
            if (
                adapter.artifact.artifact_kind != 'sd15_inpaint_pipeline'
                or not adapter.artifact.loadable
            ):
                raise DiffusionInputError(
                    'edit_adapter_id must reference a complete SD1.5 inpaint pipeline directory'
                )
        elif request.mode == 'instruction':
            with self._lock:
                adapter = self._get_artifact_locked(request.edit_adapter_id or '')
            if (
                adapter.artifact.artifact_kind != 'sd15_instruction_pipeline'
                or not adapter.artifact.loadable
            ):
                raise DiffusionInputError(
                    'edit_adapter_id must reference a complete InstructPix2Pix pipeline directory'
                )
        return {
            'request': request.snapshot(),
            'source_blob': source.descriptor(),
            'mask_blob': mask.descriptor() if mask else None,
            'edit_adapter': (
                adapter.snapshot(include_path=False) if adapter else None
            ),
        }

    def submit_edit(
        self,
        request: SD15EditRequest,
        *,
        owner_scope: str = 'local',
    ) -> Dict[str, Any]:
        self.validate_edit(request, owner_scope=owner_scope)
        if request.mode not in {'img2img', 'reference', 'inpaint', 'instruction'}:
            raise DiffusionUnsupportedError(
                f'{request.mode} edit executor is not installed yet'
            )
        normalized_owner = owner_scope.strip()[:128] or 'local'
        with self._lock:
            self._ensure_open_locked()
            if self._engine is None or not self._engine.is_loaded:
                raise DiffusionConflictError('SD15 engine is not loaded')
            if self._state != 'loaded' or self._active_job_id is not None:
                raise DiffusionConflictError('another SD15 edit is active')
            if request.mode in {'reference', 'inpaint', 'instruction'} and self._engine_config is not None:
                if (
                    self._engine_config.quantization != 'none'
                    or self._engine_config.enable_qkv_fusion
                ):
                    raise DiffusionUnsupportedError(
                        'IP-Adapter, inpaint, and instruction editing require a validated non-quantized, non-QKV SD15 profile'
                    )
                if (
                    request.mode in {'inpaint', 'instruction'}
                    and self._engine_config.device.startswith('cuda')
                    and not self._engine_config.enable_model_cpu_offload
                ):
                    raise DiffusionUnsupportedError(
                        'inpaint and instruction editing currently require the balanced CPU-offload profile on CUDA'
                    )
            artifact_id = self._loaded_artifact_id
            if not artifact_id:
                raise DiffusionServiceError('loaded SD artifact identity is missing')

        source_ids = [request.source_blob_id]
        if request.mask_blob_id:
            source_ids.append(request.mask_blob_id)
        leased_ids: list[str] = []
        try:
            for blob_id in source_ids:
                blob = self._blob_store.acquire_lease(blob_id)
                leased_ids.append(blob_id)
                if blob.owner_scope != normalized_owner:
                    raise DiffusionNotFoundError(f'image blob not found: {blob_id}')
            with self._lock:
                if self._active_job_id is not None:
                    raise DiffusionConflictError('another SD15 edit is active')
                job = DiffusionJob(
                    job_id=f'sdedit_{uuid.uuid4().hex}',
                    artifact_id=artifact_id,
                    request=request,
                    kind='edit',
                    source_blob_ids=tuple(source_ids),
                    owner_scope=normalized_owner,
                    created_at=self._clock(),
                    progress_total=request.denoising_steps,
                )
                self._jobs[job.job_id] = job
                self._active_job_id = job.job_id
                self._last_error = None
                self._trim_jobs_locked()
                try:
                    self._executor.submit(self._run_job, job.job_id)
                except Exception:
                    self._jobs.pop(job.job_id, None)
                    self._active_job_id = None
                    raise
                return job.snapshot()
        except Exception:
            for blob_id in reversed(leased_ids):
                try:
                    self._blob_store.release_lease(blob_id)
                except Exception:
                    pass
            raise

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            engine = self._engine
            if job is None:
                return
            if job.cancel_requested or engine is None:
                job.state = "cancelled" if job.cancel_requested else "failed"
                job.error = None if job.cancel_requested else "SD15 engine disappeared before generation"
                job.error_code = (
                    "DIFFUSION_CANCELLED"
                    if job.cancel_requested
                    else "DIFFUSION_EXECUTION_FAILED"
                )
                job.completed_at = self._clock()
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._job_finished.notify_all()
                early_release_ids = job.source_blob_ids if job.kind == 'edit' else ()
            else:
                early_release_ids = ()
        if early_release_ids:
            for blob_id in early_release_ids:
                try:
                    self._blob_store.release_lease(blob_id)
                except (DiffusionNotFoundError, DiffusionConflictError):
                    pass
        if job.cancel_requested or engine is None:
            return
        with self._lock:
            job.state = "running"
            job.started_at = self._clock()

        def _progress(step: int, total: int) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None or current.cancel_requested:
                    raise GenerationCancelled("SD15 generation was cancelled")
                current.progress_step = max(current.progress_step, int(step))
                current.progress_total = int(total)

        try:
            if job.kind == 'edit':
                source_blob = self._blob_store.get(job.request.source_blob_id)
                adapter_artifact = None
                if job.request.mode in {'reference', 'inpaint', 'instruction'}:
                    with self._lock:
                        registered_adapter = self._get_artifact_locked(
                            job.request.edit_adapter_id or ''
                        )
                    adapter_artifact = registered_adapter.artifact
                from PIL import Image

                with Image.open(io.BytesIO(source_blob.data)) as source_image:
                    source_image.load()
                    if job.request.mask_blob_id:
                        mask_blob = self._blob_store.get(job.request.mask_blob_id)
                        with Image.open(io.BytesIO(mask_blob.data)) as mask_image:
                            mask_image.load()
                            result = engine.edit(
                                job.request,
                                image=source_image,
                                mask=mask_image,
                                adapter=adapter_artifact,
                                callback=_progress,
                            )
                    else:
                        result = engine.edit(
                            job.request,
                            image=source_image,
                            adapter=adapter_artifact,
                            callback=_progress,
                        )
            else:
                result = engine.generate(job.request, callback=_progress)
            with self._lock:
                cancelled = job.cancel_requested
            if cancelled:
                raise GenerationCancelled("SD15 generation was cancelled")
            png_data = self._blob_store.encode_png(result.image)
            with self._lock:
                if job.cancel_requested:
                    raise GenerationCancelled("SD15 generation was cancelled")
                metadata = {
                    'job_id': job_id,
                    'artifact_id': job.artifact_id,
                    'prompt': job.request.prompt,
                    'negative_prompt': job.request.negative_prompt,
                    'seed': result.seed,
                    'width': job.request.width,
                    'height': job.request.height,
                }
                if job.kind == 'edit':
                    source_blob = self._blob_store.get(job.request.source_blob_id)
                    metadata.update({
                        'edit_mode': job.request.mode,
                        'strength': (
                            job.request.strength
                            if job.request.mode in {'img2img', 'inpaint'}
                            else None
                        ),
                        'source_blob_id': source_blob.blob_id,
                        'source_sha256': source_blob.sha256,
                        'instruction': job.request.instruction,
                        'edit_adapter_id': job.request.edit_adapter_id,
                        'conditioning_scale': job.request.conditioning_scale,
                        'image_guidance_scale': job.request.image_guidance_scale,
                        'ip_adapter_scale': job.request.ip_adapter_scale,
                    })
                    if job.request.mask_blob_id:
                        mask_blob = self._blob_store.get(job.request.mask_blob_id)
                        metadata.update({
                            'mask_blob_id': mask_blob.blob_id,
                            'mask_sha256': mask_blob.sha256,
                        })
                blob = self._blob_store.put_png(
                    png_data,
                    purpose='output',
                    owner_scope=job.owner_scope,
                    width=job.request.width,
                    height=job.request.height,
                    parent_blob_ids=job.source_blob_ids,
                    metadata=metadata,
                )
                job.state = "completed"
                job.completed_at = self._clock()
                job.progress_step = job.progress_total
                job.blob = blob.descriptor()
                job.metrics = {
                    "elapsed_seconds": result.elapsed_seconds,
                    **dict(result.metadata),
                }
                job.error = None
                job.error_code = None
        except GenerationCancelled:
            with self._lock:
                job.state = "cancelled"
                job.completed_at = self._clock()
                job.error = None
                job.error_code = "DIFFUSION_CANCELLED"
        except Exception as exc:
            with self._lock:
                job.state = "failed"
                job.completed_at = self._clock()
                job.error = str(exc)[:500]
                job.error_code = getattr(exc, "code", "DIFFUSION_EXECUTION_FAILED")
                self._last_error = job.error
        finally:
            if job.kind == 'edit':
                for blob_id in job.source_blob_ids:
                    try:
                        self._blob_store.release_lease(blob_id)
                    except (DiffusionNotFoundError, DiffusionConflictError):
                        pass
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._job_finished.notify_all()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DiffusionNotFoundError(f"diffusion job not found: {job_id}")
            return job.snapshot()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise DiffusionNotFoundError(f"diffusion job not found: {job_id}")
            if job.state in self.TERMINAL_STATES:
                return {"accepted": False, "job": job.snapshot()}
            job.cancel_requested = True
            engine = self._engine if self._active_job_id == job_id else None
        if engine is not None:
            engine.cancel()
        return {"accepted": True, "job": self.get_job(job_id)}

    def get_blob(self, blob_id: str) -> ImageBlob:
        return self._blob_store.get(blob_id)

    def delete_blob(self, blob_id: str) -> bool:
        return self._blob_store.delete(blob_id)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            engine = self._engine
            loaded_artifact = (
                self._artifacts.get(self._loaded_artifact_id or "")
                if self._loaded_artifact_id
                else None
            )
            active_job = self._jobs.get(self._active_job_id or "")
            config = self._engine_config
            dependencies = {
                name: importlib.util.find_spec(name) is not None
                for name in (
                    "torch",
                    "diffusers",
                    "transformers",
                    "accelerate",
                    "safetensors",
                    "bitsandbytes",
                    "PIL",
                )
            }
            return {
                "state": self._state,
                "loaded": bool(engine is not None and engine.is_loaded),
                "loaded_artifact": (
                    loaded_artifact.snapshot(include_path=False)
                    if loaded_artifact
                    else None
                ),
                "engine_config": (
                    {
                        "device": config.device,
                        "dtype": config.dtype,
                        "quantization": config.quantization,
                        "attention_slicing": config.enable_attention_slicing,
                        "vae_slicing": config.enable_vae_slicing,
                        "model_cpu_offload": config.enable_model_cpu_offload,
                        "qkv_fusion": config.enable_qkv_fusion,
                        "torch_compile": config.enable_torch_compile,
                    }
                    if config
                    else None
                ),
                "capabilities": engine.capabilities if engine else None,
                "active_job": active_job.snapshot() if active_job else None,
                "registered_artifacts": len(self._artifacts),
                "jobs": len(self._jobs),
                "blob_store": self._blob_store.snapshot(),
                "dependencies": dependencies,
                **sd_runtime_diagnostics(dependencies),
                "last_error": self._last_error,
            }

    def _trim_jobs_locked(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        terminal = sorted(
            (
                job
                for job in self._jobs.values()
                if job.state in self.TERMINAL_STATES
            ),
            key=lambda item: (item.completed_at or item.created_at, item.job_id),
        )
        while len(self._jobs) > self._max_jobs and terminal:
            job = terminal.pop(0)
            self._jobs.pop(job.job_id, None)

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise DiffusionConflictError("diffusion service is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active_job_id = self._active_job_id
            engine = self._engine
            if active_job_id:
                job = self._jobs.get(active_job_id)
                if job and job.state not in self.TERMINAL_STATES:
                    job.cancel_requested = True
        if active_job_id and engine is not None:
            engine.cancel()
        self._asset_manager.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if engine is not None:
            try:
                engine.unload()
            except Exception:
                pass
        self._blob_store.clear()
        with self._lock:
            for job in self._jobs.values():
                if job.state not in self.TERMINAL_STATES:
                    job.state = "cancelled"
                    job.cancel_requested = True
                    job.completed_at = self._clock()
                    job.error = None
                    job.error_code = "DIFFUSION_CANCELLED"
            self._engine = None
            self._loaded_artifact_id = None
            self._engine_config = None
            self._active_job_id = None
            self._state = "closed"


__all__ = [
    "DIFFUSION_MAX_IMAGE_PIXELS",
    "DIFFUSION_MAX_UPLOAD_BYTES",
    "SD15_LOAD_PROFILES",
    "build_sd15_engine_config",
    "build_sd15_generation_request",
    "DiffusionConflictError",
    "DiffusionBlobInUseError",
    "DiffusionBlobReferencedError",
    "DiffusionJob",
    "DiffusionNotFoundError",
    "DiffusionService",
    "DiffusionServiceError",
    "ImageBlob",
    "MemoryImageBlobStore",
    "RegisteredDiffusionArtifact",
]
