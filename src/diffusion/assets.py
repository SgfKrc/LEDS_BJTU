"""Pinned, external Stable Diffusion asset acquisition and verification.

Model weights stay outside Git and application installers.  This module owns a
small allow-listed catalog, resumable Hugging Face downloads, integrity checks,
and manifests for offline transfer.  It never installs Python dependencies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlparse

from .artifacts import DiffusionArtifactInspector


MANIFEST_NAME = ".qlh-sd-asset.json"
LOCAL_PROXY_FALLBACK = "http://127.0.0.1:7897"
MIN_FREE_SPACE_RESERVE = 1024 * 1024 * 1024


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiffusionAssetFile:
    path: str
    size_bytes: int
    source_repo: str
    source_revision: str
    sha256: str = ""


@dataclass(frozen=True)
class DiffusionAssetSpec:
    asset_id: str
    artifact_id: str
    name: str
    repo_id: str
    revision: str
    local_dir: str
    license_id: str
    model_card_url: str
    preset_id: str
    files: tuple[DiffusionAssetFile, ...]
    artifact_kind: str = "sd15_pipeline"
    safety_checker_required: bool = True
    composed: bool = False
    notes: tuple[str, ...] = ()

    @property
    def download_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    def target_path(self, root: Optional[Path] = None) -> Path:
        return (root or _app_root()) / self.local_dir


ORIGINAL_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ORIGINAL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
RETRO_REPO = "Aleksandra11/90style_anime_face_model"
RETRO_REVISION = "aa8a082c6a12d66ed995cca1ccb491bb171b9713"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"


def _file(
    path: str,
    size_bytes: int,
    *,
    sha256: str = "",
    repo: str = RETRO_REPO,
    revision: str = RETRO_REVISION,
) -> DiffusionAssetFile:
    return DiffusionAssetFile(path, size_bytes, repo, revision, sha256)


_ORIGINAL_FILES = (
    _file("README.md", 14461, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("model_index.json", 541, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("feature_extractor/preprocessor_config.json", 342, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("scheduler/scheduler_config.json", 308, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("text_encoder/config.json", 617, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file(
        "text_encoder/model.fp16.safetensors",
        246144864,
        sha256="77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e",
        repo=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
    ),
    _file("tokenizer/merges.txt", 524619, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("tokenizer/special_tokens_map.json", 472, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("tokenizer/tokenizer_config.json", 806, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("tokenizer/vocab.json", 1059962, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file("unet/config.json", 743, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file(
        "unet/diffusion_pytorch_model.fp16.safetensors",
        1719125304,
        sha256="c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21",
        repo=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
    ),
    _file("vae/config.json", 547, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file(
        "vae/diffusion_pytorch_model.fp16.safetensors",
        167335342,
        sha256="4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c",
        repo=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
    ),
    _file("safety_checker/config.json", 4723, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file(
        "safety_checker/model.fp16.safetensors",
        608018440,
        sha256="08902f19b1cfebd7c989f152fc0507bef6898c706a91d666509383122324b511",
        repo=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
    ),
)

_RETRO_FILES = (
    _file("README.md", 1320),
    _file("model_index.json", 362),
    _file("feature_extractor/preprocessor_config.json", 466),
    _file("scheduler/scheduler_config.json", 379),
    _file("text_encoder/config.json", 631),
    _file(
        "text_encoder/model.safetensors",
        492265168,
        sha256="658bdd290f65881ab77d8b707244f83fe7a666f5cc91cefb09919ca851d07ea5",
    ),
    _file("tokenizer/merges.txt", 524619),
    _file("tokenizer/special_tokens_map.json", 472),
    _file("tokenizer/tokenizer_config.json", 735),
    _file("tokenizer/vocab.json", 1059962),
    _file("unet/config.json", 1855),
    _file(
        "unet/diffusion_pytorch_model.safetensors",
        3438167536,
        sha256="7b641bf17b06365b03f51581dfe2843afbf12ecd06e2cfa75db4a1658254c010",
    ),
    _file("vae/config.json", 928),
    _file(
        "vae/diffusion_pytorch_model.safetensors",
        334643268,
        sha256="b4d2b5932bb4151e54e694fd31ccf51fca908223c9485bd56cd0e1d83ad94c49",
    ),
    _file("safety_checker/config.json", 4723, repo=ORIGINAL_REPO, revision=ORIGINAL_REVISION),
    _file(
        "safety_checker/model.fp16.safetensors",
        608018440,
        sha256="08902f19b1cfebd7c989f152fc0507bef6898c706a91d666509383122324b511",
        repo=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
    ),
)

_IP_ADAPTER_FILES = (
    _file(
        "models/image_encoder/config.json",
        560,
        sha256="625d37b31afbf2f0792a87846b3654ee23f20568409e35b78a1f795b04e1a7a1",
        repo=IP_ADAPTER_REPO,
        revision=IP_ADAPTER_REVISION,
    ),
    _file(
        "models/image_encoder/model.safetensors",
        2528373448,
        sha256="6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
        repo=IP_ADAPTER_REPO,
        revision=IP_ADAPTER_REVISION,
    ),
    _file(
        "models/ip-adapter_sd15.safetensors",
        44642768,
        sha256="289b45f16d043d0bf542e45831f971dcdaabe18b656f11e86d9dfba7e9ee3369",
        repo=IP_ADAPTER_REPO,
        revision=IP_ADAPTER_REVISION,
    ),
)


ASSET_CATALOG: Dict[str, DiffusionAssetSpec] = {
    "sd15_original_v1": DiffusionAssetSpec(
        asset_id="sd15_original_v1",
        artifact_id="sd15_original_v1",
        name="Stable Diffusion 1.5 Original",
        repo_id=ORIGINAL_REPO,
        revision=ORIGINAL_REVISION,
        local_dir="models/sd15-original-v1",
        license_id="creativeml-openrail-m",
        model_card_url=f"https://huggingface.co/{ORIGINAL_REPO}",
        preset_id="sd15_original_v1",
        files=_ORIGINAL_FILES,
    ),
    "sd15_90s_retrovers_v1": DiffusionAssetSpec(
        asset_id="sd15_90s_retrovers_v1",
        artifact_id="sd15_90s_retrovers_v1",
        name="90style Anime Face (retrovers)",
        repo_id=RETRO_REPO,
        revision=RETRO_REVISION,
        local_dir="models/sd15-90s-retrovers-v1",
        license_id="openrail",
        model_card_url=f"https://huggingface.co/{RETRO_REPO}",
        preset_id="sd15_retrovers_space_courier_v1",
        files=_RETRO_FILES,
        composed=True,
        notes=(
            "DreamBooth repository has no safety checker; QLH composes the pinned SD 1.5 safety checker.",
            "Community validation is limited; complete the ten-seed and two-reviewer quality gate before demos.",
        ),
    ),
    "sd15_ip_adapter_v1": DiffusionAssetSpec(
        asset_id="sd15_ip_adapter_v1",
        artifact_id="sd15_ip_adapter_v1",
        name="IP-Adapter for Stable Diffusion 1.5",
        repo_id=IP_ADAPTER_REPO,
        revision=IP_ADAPTER_REVISION,
        local_dir="models/sd15-ip-adapter-v1",
        license_id="apache-2.0",
        model_card_url=f"https://huggingface.co/{IP_ADAPTER_REPO}",
        preset_id="",
        files=_IP_ADAPTER_FILES,
        artifact_kind="sd15_ip_adapter",
        safety_checker_required=False,
        notes=(
            "Reference-image conditioning component; requires a separately loaded SD 1.5 base pipeline.",
            "The first GPU gate only enables non-quantized, non-QKV SD15 profiles.",
        ),
    ),
}


def get_asset_spec(asset_id: str) -> DiffusionAssetSpec:
    try:
        return ASSET_CATALOG[asset_id]
    except KeyError as exc:
        raise KeyError(f"unknown diffusion asset: {asset_id}") from exc


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_proxy(proxy_url: str) -> str:
    value = proxy_url.strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ValueError("diffusion download proxy must be a loopback HTTP(S) URL")
    if not parsed.port:
        raise ValueError("diffusion download proxy must include a port")
    return value


def _present_bytes(spec: DiffusionAssetSpec, target: Path) -> int:
    total = 0
    for item in spec.files:
        path = target / item.path
        if path.is_file():
            total += min(path.stat().st_size, item.size_bytes)
    return total


def _compose_retro_model_index(target: Path) -> None:
    model_index_path = target / "model_index.json"
    source_copy = target / ".qlh-source-model-index.json"
    source_bytes = model_index_path.read_bytes()
    if not source_copy.exists():
        source_copy.write_bytes(source_bytes)
    value = json.loads(source_bytes.decode("utf-8"))
    value["requires_safety_checker"] = True
    value["feature_extractor"] = ["transformers", "CLIPImageProcessor"]
    value["safety_checker"] = ["stable_diffusion", "StableDiffusionSafetyChecker"]
    model_index_path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_asset_directory(
    path: Path | str,
    asset_id: str,
    *,
    full_hash: bool = True,
) -> Dict[str, Any]:
    spec = get_asset_spec(asset_id)
    root = Path(path).expanduser().resolve()
    missing: list[str] = []
    size_mismatches: list[Dict[str, Any]] = []
    hash_mismatches: list[Dict[str, str]] = []
    composition_errors: list[str] = []
    file_records: list[Dict[str, Any]] = []
    for item in spec.files:
        candidate = root / item.path
        if not candidate.is_file():
            missing.append(item.path)
            continue
        size = candidate.stat().st_size
        # The composed retro model_index is intentionally rewritten locally.
        derived_index = spec.composed and item.path == "model_index.json"
        if size != item.size_bytes and not derived_index:
            size_mismatches.append(
                {"path": item.path, "expected": item.size_bytes, "actual": size}
            )
        actual_hash = _sha256(candidate) if full_hash else ""
        if item.sha256 and actual_hash and actual_hash != item.sha256:
            hash_mismatches.append(
                {"path": item.path, "expected": item.sha256, "actual": actual_hash}
            )
        record = {
                "path": item.path,
                "size_bytes": size,
                "sha256": actual_hash,
                "source_repo": item.source_repo,
                "source_revision": item.source_revision,
            }
        if derived_index:
            record["derived"] = True
        file_records.append(record)

    if spec.composed:
        try:
            model_index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            model_index = {}
        if model_index.get("requires_safety_checker") is not True:
            composition_errors.append("model_index.json does not require the safety checker")
        if model_index.get("safety_checker") != [
            "stable_diffusion",
            "StableDiffusionSafetyChecker",
        ]:
            composition_errors.append("model_index.json does not bind the pinned safety checker")
        if model_index.get("feature_extractor") != [
            "transformers",
            "CLIPImageProcessor",
        ]:
            composition_errors.append("model_index.json does not bind CLIPImageProcessor")

    artifact = DiffusionArtifactInspector().inspect(str(root), compute_hash=False)
    valid = (
        not missing
        and not size_mismatches
        and not hash_mismatches
        and not composition_errors
        and artifact.artifact_kind == spec.artifact_kind
        and artifact.loadable
    )
    return {
        "asset_id": asset_id,
        "valid": valid,
        "path": str(root),
        "artifact_sha256": _artifact_set_sha256(file_records),
        "artifact": artifact.to_dict(include_path=False),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "composition_errors": composition_errors,
        "files": file_records,
        "integrity_scope": (
            "pinned large-file hashes plus locally recorded complete manifest; not a publisher signature"
        ),
    }


def _artifact_set_sha256(files: Iterable[Dict[str, Any]]) -> str:
    records = [
        {
            "path": str(item.get("path", "")),
            "size_bytes": int(item.get("size_bytes", 0)),
            "sha256": str(item.get("sha256", "")),
            "source_repo": str(item.get("source_repo", "")),
            "source_revision": str(item.get("source_revision", "")),
        }
        for item in files
    ]
    payload = json.dumps(
        sorted(records, key=lambda item: item["path"]),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(path: Path, spec: DiffusionAssetSpec, report: Dict[str, Any]) -> None:
    artifact_sha256 = _artifact_set_sha256(report["files"])
    payload = {
        "schema_version": 2,
        "artifact_sha256": artifact_sha256,
        "asset": {
            "asset_id": spec.asset_id,
            "artifact_id": spec.artifact_id,
            "name": spec.name,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license_id": spec.license_id,
            "model_card_url": spec.model_card_url,
            "preset_id": spec.preset_id,
            "artifact_kind": spec.artifact_kind,
            "safety_checker_required": spec.safety_checker_required,
            "composed": spec.composed,
        },
        "verified_at": time.time(),
        "integrity_scope": report["integrity_scope"],
        "files": report["files"],
    }
    temporary = path / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path / MANIFEST_NAME)


@dataclass
class DiffusionAssetJob:
    asset_id: str
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    proxy_fallback_used: bool = False


class DiffusionAssetManager:
    """Own at most one background model download per process."""

    TERMINAL_STATES = {"completed", "failed"}

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        on_ready: Optional[Callable[[DiffusionAssetSpec, Path], None]] = None,
        download_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._root = (root or _app_root()).resolve()
        self._on_ready = on_ready
        self._download_fn = download_fn
        self._jobs: Dict[str, DiffusionAssetJob] = {}
        self._active_asset_id: Optional[str] = None
        self._notified_ready: set[tuple[str, str]] = set()
        self._closed = False
        self._lock = threading.RLock()

    def _notify_ready(self, spec: DiffusionAssetSpec, path: Path) -> None:
        if not self._on_ready:
            return
        resolved_path = path.resolve()
        key = (spec.asset_id, str(resolved_path))
        with self._lock:
            if key in self._notified_ready:
                return
            self._notified_ready.add(key)
        try:
            self._on_ready(spec, resolved_path)
        except Exception:
            with self._lock:
                self._notified_ready.discard(key)
            raise

    def _installed(self, spec: DiffusionAssetSpec) -> bool:
        report = verify_asset_directory(
            spec.target_path(self._root),
            spec.asset_id,
            full_hash=False,
        )
        return bool(report["valid"])

    def catalog(self) -> list[Dict[str, Any]]:
        result = []
        for spec in ASSET_CATALOG.values():
            target = spec.target_path(self._root)
            job = self._jobs.get(spec.asset_id)
            result.append(
                {
                    "asset_id": spec.asset_id,
                    "artifact_id": spec.artifact_id,
                    "name": spec.name,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "license_id": spec.license_id,
                    "model_card_url": spec.model_card_url,
                    "preset_id": spec.preset_id,
                    "artifact_kind": spec.artifact_kind,
                    "download_bytes": spec.download_bytes,
                    "present_bytes": _present_bytes(spec, target),
                    "installed": self._installed(spec),
                    "safety_checker_required": spec.safety_checker_required,
                    "composed": spec.composed,
                    "notes": list(spec.notes),
                    "job": self._job_snapshot(job, spec) if job else None,
                }
            )
        return result

    def status(self, asset_id: str) -> Dict[str, Any]:
        spec = get_asset_spec(asset_id)
        with self._lock:
            job = self._jobs.get(asset_id)
        if job is None:
            state = "completed" if self._installed(spec) else "not_started"
            return {
                "asset_id": asset_id,
                "state": state,
                "download_bytes": spec.download_bytes,
                "present_bytes": _present_bytes(spec, spec.target_path(self._root)),
                "installed": state == "completed",
            }
        return self._job_snapshot(job, spec)

    def _job_snapshot(self, job: DiffusionAssetJob, spec: DiffusionAssetSpec) -> Dict[str, Any]:
        target = spec.target_path(self._root)
        present = _present_bytes(spec, target)
        return {
            **asdict(job),
            "download_bytes": spec.download_bytes,
            "present_bytes": present,
            "progress_percent": min(100, round(present * 100 / max(1, spec.download_bytes))),
            "installed": job.state == "completed" and self._installed(spec),
        }

    def start_download(
        self,
        asset_id: str,
        *,
        license_accepted: bool,
        proxy_fallback: str = LOCAL_PROXY_FALLBACK,
    ) -> Dict[str, Any]:
        spec = get_asset_spec(asset_id)
        if not license_accepted:
            raise ValueError("model license acceptance is required before download")
        validated_proxy = _validate_proxy(proxy_fallback) if proxy_fallback else ""
        with self._lock:
            if self._closed:
                raise RuntimeError("diffusion asset manager is closed")
            if self._active_asset_id and self._active_asset_id != asset_id:
                raise RuntimeError(f"another diffusion asset download is active: {self._active_asset_id}")
            existing = self._jobs.get(asset_id)
            if existing and existing.state not in self.TERMINAL_STATES:
                return self._job_snapshot(existing, spec)
            if self._installed(spec):
                job = DiffusionAssetJob(
                    asset_id=asset_id,
                    state="completed",
                    started_at=time.time(),
                    completed_at=time.time(),
                )
                self._jobs[asset_id] = job
                self._notify_ready(spec, spec.target_path(self._root))
                return self._job_snapshot(job, spec)
            job = DiffusionAssetJob(asset_id=asset_id)
            self._jobs[asset_id] = job
            self._active_asset_id = asset_id
            thread = threading.Thread(
                target=self._run_download,
                args=(spec, job, validated_proxy),
                name=f"sd-asset-{asset_id}",
                daemon=True,
            )
            thread.start()
            return self._job_snapshot(job, spec)

    def _run_download(
        self,
        spec: DiffusionAssetSpec,
        job: DiffusionAssetJob,
        proxy_fallback: str,
    ) -> None:
        target = spec.target_path(self._root)
        try:
            job.state = "downloading"
            job.started_at = time.time()
            target.mkdir(parents=True, exist_ok=True)
            remaining = max(0, spec.download_bytes - _present_bytes(spec, target))
            free = shutil.disk_usage(target.parent).free
            if free < remaining + MIN_FREE_SPACE_RESERVE:
                raise RuntimeError(
                    f"insufficient disk space: need {remaining + MIN_FREE_SPACE_RESERVE} bytes, have {free}"
                )
            try:
                self._download_groups(spec, target, proxy_url="")
            except Exception:
                if not proxy_fallback:
                    raise
                job.proxy_fallback_used = True
                self._download_groups(spec, target, proxy_url=proxy_fallback)
            if spec.composed:
                _compose_retro_model_index(target)
            job.state = "verifying"
            report = verify_asset_directory(target, spec.asset_id, full_hash=True)
            if not report["valid"]:
                raise RuntimeError(
                    "downloaded diffusion asset failed verification: "
                    + json.dumps(
                        {
                            "missing": report["missing"],
                            "size_mismatches": report["size_mismatches"],
                            "hash_mismatches": report["hash_mismatches"],
                            "composition_errors": report["composition_errors"],
                        },
                        ensure_ascii=True,
                    )
                )
            _write_manifest(target, spec, report)
            self._notify_ready(spec, target)
            job.state = "completed"
            job.completed_at = time.time()
        except Exception as exc:
            job.state = "failed"
            job.error = str(exc)[:1000]
            job.completed_at = time.time()
        finally:
            with self._lock:
                if self._active_asset_id == spec.asset_id:
                    self._active_asset_id = None

    def _download_groups(
        self,
        spec: DiffusionAssetSpec,
        target: Path,
        *,
        proxy_url: str,
    ) -> None:
        if self._download_fn is not None:
            self._download_fn(spec=spec, target=target, proxy_url=proxy_url)
            return
        if importlib.util.find_spec("huggingface_hub") is None:
            raise RuntimeError(
                "huggingface_hub is missing from the optional SD environment; install packaging/requirements-sd15.txt"
            )
        groups: Dict[tuple[str, str], list[str]] = {}
        for item in spec.files:
            groups.setdefault((item.source_repo, item.source_revision), []).append(item.path)

        proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
        transfer_keys = ("HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER")
        managed_keys = proxy_keys + transfer_keys
        previous = {key: os.environ.get(key) for key in managed_keys}
        try:
            # hf-xet uses a separate transfer path that does not reliably honor
            # the local HTTP proxy on Windows. Standard Hub HTTP keeps large
            # safetensors resumable through the same proxy as metadata files.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            from huggingface_hub import constants as hub_constants
            from huggingface_hub import snapshot_download

            hub_constants.HF_HUB_DISABLE_XET = True
            if proxy_url:
                for key in proxy_keys:
                    os.environ[key] = proxy_url
            for (repo_id, revision), paths in groups.items():
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(target),
                    allow_patterns=paths,
                    etag_timeout=15,
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def import_asset(
        self,
        asset_id: str,
        path: str,
        *,
        license_accepted: bool,
    ) -> Dict[str, Any]:
        if not license_accepted:
            raise ValueError("model license acceptance is required before import")
        spec = get_asset_spec(asset_id)
        root = Path(path).expanduser().resolve()
        report = verify_asset_directory(root, asset_id, full_hash=True)
        if not report["valid"]:
            raise ValueError("offline diffusion asset package failed verification")
        _write_manifest(root, spec, report)
        self._notify_ready(spec, root)
        return report

    def discover_installed(self) -> None:
        if not self._on_ready:
            return
        for spec in ASSET_CATALOG.values():
            target = spec.target_path(self._root)
            if self._installed(spec):
                self._notify_ready(spec, target)

    def close(self) -> None:
        with self._lock:
            self._closed = True


__all__ = [
    "ASSET_CATALOG",
    "DiffusionAssetFile",
    "DiffusionAssetManager",
    "DiffusionAssetSpec",
    "IP_ADAPTER_REPO",
    "IP_ADAPTER_REVISION",
    "LOCAL_PROXY_FALLBACK",
    "MANIFEST_NAME",
    "get_asset_spec",
    "verify_asset_directory",
]
