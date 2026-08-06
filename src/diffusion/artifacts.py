"""Local Stable Diffusion artifact inspection.

Inspection is intentionally conservative.  A ``.safetensors`` suffix only
identifies a tensor container; it does not prove that the file is a complete
Stable Diffusion checkpoint.  The inspector reads directory metadata or the
safetensors header and never unpickles a ``.ckpt`` file.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


_PIPELINE_CLASSES = {
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionControlNetPipeline",
}
_CONTROLNET_MARKERS = (
    "control_model",
    "controlnet_cond_embedding",
    "input_hint_block",
    "zero_convs",
)
_FULL_MODEL_MARKERS = (
    "model.diffusion_model.",
    "first_stage_model.",
    "cond_stage_model.",
    "unet.",
    "vae.",
    "text_encoder.",
)
_IP_ADAPTER_WEIGHT_NAMES = (
    "ip-adapter_sd15.safetensors",
    "ip_adapter_sd15.safetensors",
)


@dataclass(frozen=True)
class DiffusionArtifact:
    """Stable Diffusion artifact metadata safe to expose to the API."""

    path: str
    artifact_kind: str
    pipeline_family: str = "stable_diffusion_1"
    precision: str = "unknown"
    sha256: str = ""
    size_bytes: int = 0
    loadable: bool = False
    missing_components: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tensor_keys: List[str] = field(default_factory=list)

    def to_dict(self, *, include_path: bool = False) -> Dict[str, Any]:
        value = asdict(self)
        if not include_path:
            value.pop("path", None)
        # Tensor keys are useful for local diagnostics but needlessly noisy in
        # manifests and should not leave the process unless explicitly asked.
        value.pop("tensor_keys", None)
        return value


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256_directory(path: Path) -> str:
    """Hash relative filenames and bytes without exposing local absolute paths."""

    digest = hashlib.sha256()
    for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(struct.pack("<Q", len(relative)))
        digest.update(relative)
        with item.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_safetensors_header(path: Path) -> Optional[Dict[str, Any]]:
    """Read only the safetensors header; never materialize tensor data."""

    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                return None
            header_size = struct.unpack("<Q", raw_size)[0]
            # A corrupted header must not cause an unbounded allocation.
            if header_size <= 0 or header_size > 128 * 1024 * 1024:
                return None
            raw_header = handle.read(header_size)
    except OSError:
        return None
    if len(raw_header) != header_size:
        return None
    try:
        value = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _header_keys(header: Dict[str, Any]) -> List[str]:
    return sorted(str(key) for key in header if key != "__metadata__")


def _precision_from_header(header: Dict[str, Any]) -> str:
    dtypes: Set[str] = set()
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        dtype = value.get("dtype")
        if dtype:
            dtypes.add(str(dtype).upper())
    if not dtypes:
        return "unknown"
    if dtypes <= {"F16"}:
        return "fp16"
    if dtypes <= {"F32"}:
        return "fp32"
    if dtypes <= {"BF16"}:
        return "bf16"
    return "mixed"


def _kind_from_keys(keys: Iterable[str]) -> str:
    normalized = [key.lower() for key in keys]
    if (
        any(key.startswith("image_proj.") for key in normalized)
        and any(key.startswith("ip_adapter.") for key in normalized)
    ):
        return "sd15_ip_adapter"
    if any(marker in key for key in normalized for marker in _CONTROLNET_MARKERS):
        return "controlnet"
    if any(marker in key for key in normalized for marker in _FULL_MODEL_MARKERS):
        return "sd15_checkpoint"
    if any("lora" in key or "lora_te" in key for key in normalized):
        return "lora"
    if any("vae" in key and "decoder" in key for key in normalized):
        return "vae"
    return "unknown"


def resolve_sd15_ip_adapter_layout(path: str | Path) -> Dict[str, str]:
    """Resolve the supported offline SD1.5 IP-Adapter directory layout."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("SD1.5 IP-Adapter must be a local directory")
    for subfolder in ("models", ""):
        adapter_dir = root / subfolder if subfolder else root
        image_encoder = adapter_dir / "image_encoder"
        encoder_weights = tuple(image_encoder.glob("*.safetensors"))
        for weight_name in _IP_ADAPTER_WEIGHT_NAMES:
            weight_path = adapter_dir / weight_name
            if not weight_path.is_file():
                continue
            header = _read_safetensors_header(weight_path)
            keys = _header_keys(header or {})
            if _kind_from_keys(keys) != "sd15_ip_adapter":
                raise ValueError("IP-Adapter Safetensors lacks image_proj/ip_adapter weights")
            if not (image_encoder / "config.json").is_file() or not encoder_weights:
                raise ValueError("IP-Adapter directory lacks a Safetensors CLIP image_encoder")
            return {
                "root": str(root),
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": (
                    f"{subfolder}/image_encoder" if subfolder else "image_encoder"
                ),
            }
    raise ValueError(
        "directory lacks ip-adapter_sd15.safetensors and its image_encoder"
    )


class DiffusionArtifactInspector:
    """Fail-closed inspector for local SD 1.x artifacts."""

    def inspect(self, path: str, *, compute_hash: bool = False) -> DiffusionArtifact:
        resolved = Path(path).expanduser()
        display_path = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if not resolved.exists():
            return DiffusionArtifact(
                path=display_path,
                artifact_kind="unknown",
                loadable=False,
                warnings=["路径不存在"],
            )
        if resolved.is_dir():
            return self._inspect_directory(resolved, compute_hash=compute_hash)
        if resolved.is_file():
            return self._inspect_file(resolved, compute_hash=compute_hash)
        return DiffusionArtifact(
            path=display_path,
            artifact_kind="unknown",
            loadable=False,
            warnings=["路径不是普通文件或目录"],
        )

    def _inspect_directory(self, path: Path, *, compute_hash: bool) -> DiffusionArtifact:
        model_index = _read_json(path / "model_index.json")
        size_bytes = _directory_size(path)
        sha256 = _sha256_directory(path) if compute_hash else ""
        warnings: List[str] = []
        missing: List[str] = []

        if model_index is None:
            adapter_weights = [
                candidate
                for subfolder in (path / "models", path)
                for name in _IP_ADAPTER_WEIGHT_NAMES
                if (candidate := subfolder / name).is_file()
            ]
            if adapter_weights:
                weight = adapter_weights[0]
                header = _read_safetensors_header(weight) or {}
                kind = _kind_from_keys(_header_keys(header))
                try:
                    resolve_sd15_ip_adapter_layout(path)
                except ValueError as exc:
                    missing.append("image_encoder")
                    warnings.append(str(exc))
                return DiffusionArtifact(
                    path=str(path.resolve()),
                    artifact_kind=(
                        "sd15_ip_adapter" if kind == "sd15_ip_adapter" else "unknown"
                    ),
                    precision=_precision_from_header(header),
                    sha256=sha256,
                    size_bytes=size_bytes,
                    loadable=kind == "sd15_ip_adapter" and not missing,
                    missing_components=missing,
                    warnings=warnings,
                )
            return DiffusionArtifact(
                path=str(path.resolve()),
                artifact_kind="unknown",
                size_bytes=size_bytes,
                sha256=sha256,
                warnings=["目录缺少有效 model_index.json，不能确认是 Diffusers pipeline"],
            )

        class_name = str(model_index.get("_class_name", ""))
        if class_name not in _PIPELINE_CLASSES:
            return DiffusionArtifact(
                path=str(path.resolve()),
                artifact_kind="unknown",
                size_bytes=size_bytes,
                sha256=sha256,
                warnings=[f"不支持的 pipeline class: {class_name or 'unknown'}"],
            )

        required = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")
        for component in required:
            if not (path / component).exists():
                missing.append(component)
        kind = "sd15_pipeline" if not missing else "unknown"
        if missing:
            warnings.append("Diffusers pipeline 缺少必要组件: " + ", ".join(missing))
        if class_name == "StableDiffusionControlNetPipeline" and not (path / "controlnet").exists():
            missing.append("controlnet")
            kind = "unknown"
            warnings.append("ControlNet pipeline 缺少 controlnet 组件")

        return DiffusionArtifact(
            path=str(path.resolve()),
            artifact_kind=kind,
            precision=self._directory_precision(path),
            sha256=sha256,
            size_bytes=size_bytes,
            loadable=kind == "sd15_pipeline",
            missing_components=missing,
            warnings=warnings,
        )

    @staticmethod
    def _directory_precision(path: Path) -> str:
        for candidate in (path / "unet", path / "vae"):
            if not candidate.is_dir():
                continue
            for file_path in candidate.glob("*.safetensors"):
                header = _read_safetensors_header(file_path)
                if header:
                    return _precision_from_header(header)
        return "unknown"

    def _inspect_file(self, path: Path, *, compute_hash: bool) -> DiffusionArtifact:
        suffix = path.suffix.lower()
        size_bytes = path.stat().st_size
        sha256 = _sha256_file(path) if compute_hash else ""
        if suffix == ".ckpt":
            return DiffusionArtifact(
                path=str(path.resolve()),
                artifact_kind="unknown",
                size_bytes=size_bytes,
                sha256=sha256,
                warnings=["不在探测阶段反序列化 .ckpt；请转换为 Safetensors 或 Diffusers 目录"],
            )
        if suffix != ".safetensors":
            return DiffusionArtifact(
                path=str(path.resolve()),
                artifact_kind="unknown",
                size_bytes=size_bytes,
                sha256=sha256,
                warnings=["不是支持的 SD 单文件格式"],
            )

        header = _read_safetensors_header(path)
        if header is None:
            return DiffusionArtifact(
                path=str(path.resolve()),
                artifact_kind="unknown",
                size_bytes=size_bytes,
                sha256=sha256,
                warnings=["无法读取 Safetensors header，文件可能损坏或不是 Safetensors"],
            )
        keys = _header_keys(header)
        kind = _kind_from_keys(keys)
        warnings: List[str] = []
        if kind == "unknown":
            warnings.append("这是 Safetensors 容器，但无法确认它是完整 SD 1.5 checkpoint")
        if kind == "controlnet":
            warnings.append("ControlNet 是辅助组件，必须配合完整 SD 1.5 base 和输入图片")
        if kind == "sd15_ip_adapter":
            warnings.append(
                "IP-Adapter 是参考图条件组件；单文件缺少 CLIP image_encoder，不能直接加载"
            )
        return DiffusionArtifact(
            path=str(path.resolve()),
            artifact_kind=kind,
            precision=_precision_from_header(header),
            sha256=sha256,
            size_bytes=size_bytes,
            loadable=kind == "sd15_checkpoint",
            warnings=warnings,
            tensor_keys=keys,
        )


__all__ = [
    "DiffusionArtifact",
    "DiffusionArtifactInspector",
    "resolve_sd15_ip_adapter_layout",
]
