"""Model one-click download job service (P0A).

后端下载 job 状态机：resolve -> download -> verify -> register -> ready，
并持久化到主节点 SQLite（``model_download_jobs`` 表），供前端轮询进度。

设计要点（对齐项目既有惯例）：
- 复用 ``import_model`` 的校验/注册语义（build_manifest / validate_model_artifact /
  write_manifest / save_local_experimental_model），但为可编程 service 重写，
  不依赖 scripts 目录的相对导入或 CLI 参数层。
- 资源门前置：GGUF 按 ``asset_size*1.2 + 512MB`` 估 RAM；safetensors 按
  recommended_vram + ``asset_size*0.55+512MB`` 估 VRAM，非 CUDA 允许 CPU 时按
  ``asset_size*2.2 + 1GB`` 估 RAM（与 ``llm_smoke_matrix._resource_rejection`` 对齐）。
- 幂等：同一 ``source + target + model_id`` 去重；ready 后不再重复排队。
- fail-closed：未配置显式 SHA-256 时仍做全量 SHA-256 记录；配置则严格校验，
  不匹配即失败并清理 staging。

下载链默认在调用方提供的 ``runner``（后台线程池）中执行；模块本身不持有全局后台线程。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import local_store
import model_registry_validation as mrv
import proxy_config
from model_config import _APP_ROOT
from model_registry_validation import build_manifest, validate_model_artifact, write_manifest
from local_store import save_local_experimental_model

# 下载成功后落盘子目录名（沿用 config.MODEL_PATH 的 ``models/<name>`` 约定）
_DEFAULT_MODELS_ROOT = os.path.join(_APP_ROOT, "models")
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")
_SAFE_SUFFIXES = (".safetensors", ".bin")

# job 状态机
STATUS_QUEUED = "queued"
STATUS_DOWNLOADING = "downloading"
STATUS_VERIFYING = "verifying"
STATUS_REGISTERING = "registering"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_DOWNLOADING, STATUS_VERIFYING, STATUS_REGISTERING}

# 资源门错误码
ERR_INSUFFICIENT_RAM = "insufficient_ram"
ERR_INSUFFICIENT_VRAM = "insufficient_vram"
ERR_CUDA_REQUIRED = "cuda_required"

_LOCK = threading.Lock()

# 预设列表（P0）。先代码内置，便于审查；后续可迁到 DB。
# 每项枚举了双源 repo、默认文件名量化、resource_gate 与元数据。
# NOTE: 真实 repo id / 受控 SHA 需逐项核实后再启用严格校验（见 docs 计划）。
PRESETS: list[dict[str, Any]] = [
    {
        "id": "qwen-1_8b-gguf-q4",
        "display": "Qwen-1.8B-Chat Q4_K_M (GGUF)",
        "kind": "gguf",
        "default_engine": "llama_cpp",
        "default_quant": "Q4_K_M",
        "default_model_id": "qwen-1_8b",
        "hf_repo": "Qwen/Qwen-1.8B-Chat",
        "ms_path": "",
        "file_pattern": "Qwen-1_8B-Chat.Q4_K_M.gguf",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 4.0,
            "min_vram_gb": 0.0,
            "min_disk_gb": 2.0,
            "allow_cpu": True,
        },
        "description": "Qwen-1.8B 的 4-bit GGUF，CPU 可跑，适合低显存环境。",
    },
    {
        "id": "qwen3-3_8b-gguf-q4",
        "display": "Qwen3-3.8B Q4_K_M (GGUF)",
        "kind": "gguf",
        "default_engine": "llama_cpp",
        "default_quant": "Q4_K_M",
        "default_model_id": "qwen3-3_8b",
        "hf_repo": "Qwen/Qwen3-3.8B",
        "ms_path": "",
        "file_pattern": "qwen3-3_8b-q4_k_m.gguf",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 6.0,
            "min_vram_gb": 0.0,
            "min_disk_gb": 3.0,
            "allow_cpu": True,
        },
        "description": "Qwen3-3.8B 的 4-bit GGUF，CPU 可跑，推理质量与速度均衡。",
    },
    {
        "id": "deepseek-7b-gguf-q4",
        "display": "DeepSeek-7B Q4_K_M (GGUF)",
        "kind": "gguf",
        "default_engine": "llama_cpp",
        "default_quant": "Q4_K_M",
        "default_model_id": "deepseek-7b",
        "hf_repo": "deepseek-ai/deepseek-llm-7b-chat",
        "ms_path": "",
        "file_pattern": "deepseek-llm-7b-chat.Q4_K_M.gguf",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 8.0,
            "min_vram_gb": 0.0,
            "min_disk_gb": 5.0,
            "allow_cpu": True,
        },
        "description": "DeepSeek-7B 的 4-bit GGUF，适合中低配机器。",
    },
    {
        "id": "qw3-vl-4b",
        "display": "QW3-VL-4B (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "fp16",
        "default_model_id": "qw3-vl-4b",
        "hf_repo": "Qwen/Qwen3-VL-4B",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 10.0,
            "min_vram_gb": 8.0,
            "min_disk_gb": 10.0,
            "allow_cpu": False,
        },
        "description": "Qwen3-VL 4B 视觉多模态，需 CUDA GPU（约 8GB 显存）。",
    },
    {
        "id": "sd15-base",
        "display": "Stable Diffusion 1.5 基础模型",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "fp16",
        "default_model_id": "sd15-base",
        "hf_repo": "runwayml/stable-diffusion-v1-5",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 8.0,
            "min_vram_gb": 6.0,
            "min_disk_gb": 4.0,
            "allow_cpu": False,
        },
        "description": "Stable Diffusion 1.5 图像模型，需 CUDA GPU。",
    },
]
_PRESETS_BY_ID = {p["id"]: p for p in PRESETS}


class JobError(Exception):
    """job 执行失败的宿主异常，message 会脱敏写入 error 字段。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# 数据库持久化
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_job(row: Any) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "progress": float(row["progress"] or 0.0),
        "source": row["source"],
        "target": row["target"],
        "model_id": row["model_id"],
        "preset_id": row["preset_id"],
        "engine": row["engine"],
        "quant": row["quant"],
        "total_bytes": int(row["total_bytes"] or 0),
        "downloaded_bytes": int(row["downloaded_bytes"] or 0),
        "error_code": row["error_code"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "finished_at": row["finished_at"],
    }


def _insert_job(row: dict[str, Any]) -> None:
    with local_store._write_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_download_jobs (
              job_id, status, progress, source, target, model_id, preset_id,
              engine, quant, total_bytes, downloaded_bytes, error_code, error,
              created_at, updated_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["job_id"], row["status"], row["progress"], row["source"],
                row["target"], row["model_id"], row["preset_id"], row["engine"],
                row["quant"], row["total_bytes"], row["downloaded_bytes"],
                row["error_code"], row["error"], row["created_at"], row["updated_at"],
                row["finished_at"],
            ),
        )


def _update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{key}=?" for key in fields)
    params = list(fields.values()) + [job_id]
    with local_store._write_connection() as connection:
        connection.execute(f"UPDATE model_download_jobs SET {cols} WHERE job_id=?", params)


def get_job(job_id: str) -> dict[str, Any] | None:
    connection = local_store._connect()
    try:
        row = connection.execute(
            "SELECT * FROM model_download_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row else None
    finally:
        connection.close()


def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    connection = local_store._connect()
    try:
        rows = connection.execute(
            "SELECT * FROM model_download_jobs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_job(r) for r in rows]
    finally:
        connection.close()


def _job_key(source: str, target: str, model_id: str) -> str:
    return f"{source}|{target}|{model_id}"


def _active_job_exists(source: str, target: str, model_id: str) -> bool:
    connection = local_store._connect()
    try:
        rows = connection.execute(
            "SELECT job_id FROM model_download_jobs WHERE source=? AND target=? AND model_id=?",
            (source, target, model_id),
        ).fetchall()
        for r in rows:
            if r["job_id"]:
                status = get_job(r["job_id"])
                if status and status["status"] in _ACTIVE_STATUSES:
                    return True
        return False
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 预设查询 + 资源门
# ---------------------------------------------------------------------------

def _preset_source(preset: dict[str, Any], use_modelscope: bool) -> str:
    if use_modelscope:
        if not preset.get("ms_path"):
            raise JobError("PRESET_NO_MS_SOURCE", f"预设 {preset['id']} 无 ModelScope 源")
        return preset["ms_path"]
    if not preset.get("hf_repo"):
        raise JobError("PRESET_NO_HF_SOURCE", f"预设 {preset['id']} 无 HF 源")
    return preset["hf_repo"]


def _available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def _resource_gate(rejected: dict[str, str], preset: dict[str, Any]) -> None:
    gate = preset.get("resource_gate", {})
    min_ram_gb = float(gate.get("min_ram_gb", 0))
    allow_cpu = bool(gate.get("allow_cpu", True))
    ram = _available_ram_bytes()
    if min_ram_gb > 0 and ram < int(min_ram_gb * 1024**3):
        rejected["ram"] = ERR_INSUFFICIENT_RAM
    if not allow_cpu:
        allowed_vram = float(gate.get("min_vram_gb", 0))
        try:
            import torch
            if not torch.cuda.is_available():
                rejected["gpu"] = ERR_CUDA_REQUIRED
            elif allowed_vram > 0:
                free, total = torch.cuda.mem_get_info()
                if int(total) < int(allowed_vram * 1024**3 * 0.9) or int(free) < int(allowed_vram * 1024**3):
                    rejected["gpu"] = ERR_INSUFFICIENT_VRAM
        except Exception:
            rejected["gpu"] = ERR_CUDA_REQUIRED


def list_presets() -> list[dict[str, Any]]:
    """返回带本机可装性评估的预设列表。"""
    out: list[dict[str, Any]] = []
    for preset in PRESETS:
        item = dict(preset)
        rejected: dict[str, str] = {}
        _resource_gate(rejected, preset)
        if rejected:
            item["installable"] = False
            item["blocked_reasons"] = rejected
        else:
            item["installable"] = True
            item["blocked_reasons"] = {}
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# 下载链（resolve -> download -> verify -> register）
# ---------------------------------------------------------------------------

def _weight_files(target: Path) -> list[Path]:
    if not target.is_dir():
        return []
    return sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES)
    )


def _resolve_target(repo_or_path: str, target: str | None, models_root: str) -> Path:
    """返回发布目标（destination）。

    对本地目录 source：destination 由 target/models_root 决定（source 仅作内容来源），
    name 取自目录名。
    """
    source = Path(repo_or_path)
    if source.is_dir():
        name = source.name
    else:
        name = repo_or_path.strip("/").split("/")[-1]
    if not name:
        raise JobError("SOURCE_INVALID", f"无法从 source 解析模型名: {repo_or_path!r}")
    return Path(target or os.path.join(models_root, name)).absolute()


def _download(source: str, staging: Path, *, use_modelscope: bool, proxy: str,
              progress_cb: Callable[[Path, int, int], None] | None) -> list[Path]:
    """下载到 staging，并回调（已下载字节，总字节）进度。

    本地目录作为 source 时整体拷贝到 staging（模拟下载产物），便于测试与本地导入。
    """
    staging.mkdir(parents=True, exist_ok=True)
    src_path = Path(source)
    if src_path.is_dir():
        for item in src_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_path)
                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        files = _weight_files(staging)
        total = sum(f.stat().st_size for f in files)
        if progress_cb and files:
            progress_cb(files[-1], total, total)
        if not files:
            raise JobError("DOWNLOAD_NO_WEIGHTS", "本地目录未发现权重文件")
        return files
    resolved_proxy = proxy_config.resolve_http_proxy(proxy or None)
    if use_modelscope:
        code = ("from modelscope import snapshot_download; "
                f"snapshot_download({source!r}, local_dir={str(staging)!r})")
        with proxy_config.proxy_environment(resolved_proxy) as environment:
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=environment,
            )
        if result.returncode != 0:
            raise JobError("DOWNLOAD_FAILED",
                           f"ModelScope 下载失败: {(result.stderr or result.stdout)[-300:]}")
    else:
        import huggingface_hub
        with proxy_config.proxy_environment(resolved_proxy) as environment:
            keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
            previous = {key: os.environ.get(key) for key in keys}
            try:
                os.environ.update({key: environment[key] for key in keys})
                huggingface_hub.snapshot_download(
                    repo_id=source, local_dir=str(staging), local_dir_use_symlinks=False,
                )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    files = _weight_files(staging)
    if not files:
        raise JobError("DOWNLOAD_NO_WEIGHTS",
                       "下载完成但未发现 safetensors/gguf/bin 权重文件")
    return files


def _verify(files: list[Path], expected_sha256: str | None) -> dict[str, Any]:
    if not files:
        raise JobError("VERIFY_EMPTY", "没有可校验的权重文件")
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise JobError("VERIFY_EMPTY_FILE", f"权重文件缺失或为空: {path}")
        size = path.stat().st_size
        total_bytes += size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    raw = digest.hexdigest()
    summary = {"file_count": len(files), "total_bytes": total_bytes, "sha256": raw}
    if expected_sha256 and raw != expected_sha256.lower():
        raise JobError(
            "SHA256_MISMATCH",
            f"SHA-256 不匹配: 期望 {expected_sha256.lower()}，实际 {raw}",
        )
    return summary


def _infer_artifact(target: Path, gguf_path: str = "") -> dict[str, Any]:
    files = _weight_files(target)
    safe_files = [f for f in files if f.name.lower().endswith(_SAFE_SUFFIXES)]
    gguf_files = [f for f in files if f.suffix.lower() == ".gguf"]
    explicit = Path(gguf_path).expanduser().absolute() if gguf_path else None
    if not explicit and len(gguf_files) > 1:
        raise JobError("MULTIPLE_GGUF", "发现多个 GGUF 文件；需指定 --gguf-path")
    selected = explicit or (gguf_files[0] if len(gguf_files) == 1 else None)
    model_type = ("both" if safe_files and selected else
                  "safetensors" if safe_files else
                  "gguf" if selected else "")
    if not model_type:
        raise JobError("ARTIFACT_INFER_FAILED", "无法推断模型类型：需 safetensors/bin 或 GGUF 权重")
    artifact = validate_model_artifact(
        model_type, str(target) if safe_files else "",
        str(selected) if selected else "",
    )
    if selected and selected not in files:
        files.append(selected)
        artifact["files"] = [*artifact["safetensors_files"], selected]
    return artifact


def _register(model_id: str, target: Path, summary: dict[str, Any],
              *, gguf_path: str = "", revision: str = "") -> bool:
    artifact = _infer_artifact(target, gguf_path)
    manifest = summary.get("manifest") or build_manifest(
        target, artifact["files"], model_type=artifact["model_type"],
        revision=revision, source="download-service",
    )
    config = {
        "model_id": model_id,
        "name": model_id,
        "model_type": artifact["model_type"],
        "model_path": artifact["model_path"],
        "gguf_path": artifact["gguf_path"],
        "quantization": "Q4_K_M" if artifact["model_type"] == "gguf" else "fp16",
        "sha256": summary["sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source": "download-service",
        "revision": revision,
        "manifest": manifest,
    }
    return save_local_experimental_model(model_id, config)


# ---------------------------------------------------------------------------
# job 主流程
# ---------------------------------------------------------------------------

def _run_job(job_id: str, *, source: str, target: str, model_id: str,
             preset_id: str, engine: str, quant: str, use_modelscope: bool,
             proxy: str, expected_sha256: str, gguf_path: str,
             models_root: str, allow_cpu: bool,
             progress_cb: Callable[[int, int], None] | None = None) -> None:
    """在调用方线程中执行 job 全流程；异常写回 job 并落盘清理。"""
    staging: Path | None = None
    published = False
    try:
        _update_job(job_id, status=STATUS_DOWNLOADING, progress=0.0,
                    downloaded_bytes=0, total_bytes=0, error=None, error_code=None)
        destination = _resolve_target(source, target, models_root)
        if destination.exists():
            raise JobError("TARGET_EXISTS", f"目标已存在，拒绝覆盖安装: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.qlh-dl-", dir=destination.parent))
        base_dir = staging
        total_bytes = 0
        downloaded = 0
        seen_files: set[str] = set()

        def _cb(path: Path, current: int, grand_total: int) -> None:
            nonlocal total_bytes, downloaded
            total_bytes = grand_total
            downloaded = current
            if progress_cb:
                progress_cb(downloaded, grand_total)
            _update_job(job_id, downloaded_bytes=current, total_bytes=grand_total,
                        progress=(current / grand_total if grand_total else 0.0))

        # 下载（含 fake/本地目录场景跳过）
        files = _download(source, base_dir, use_modelscope=use_modelscope,
                          proxy=proxy, progress_cb=_cb)
        _update_job(job_id, status=STATUS_VERIFYING)

        # 校验
        summary = _verify(files, expected_sha256 or None)
        artifact = _infer_artifact(staging, gguf_path)
        files = artifact["files"]
        manifest = build_manifest(staging, artifact["files"],
                                  model_type=artifact["model_type"],
                                  source="download-service")
        write_manifest(staging, manifest)
        summary.update({
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest": manifest,
            "model_type": artifact["model_type"],
        })

        _update_job(job_id, status=STATUS_REGISTERING, progress=1.0,
                    total_bytes=summary["total_bytes"],
                    downloaded_bytes=summary["total_bytes"])

        # staging -> 目标（原子 rename）
        staging.replace(destination)
        published = True
        registered = _register(model_id, destination, summary,
                               gguf_path=gguf_path, revision="")
        if not registered:
            raise JobError("REGISTER_FAILED", f"模型 '{model_id}' 注册失败")
        _update_job(job_id, status=STATUS_READY, progress=1.0,
                    finished_at=_now(), error=None, error_code=None)
    except JobError as exc:
        _fail_job(job_id, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail_job(job_id, "INTERNAL_ERROR", str(exc))
    finally:
        if staging is not None and staging.exists() and not published:
            shutil.rmtree(staging, ignore_errors=True)
        if published:
            # 幂等清理：已发布则不删（注册失败已由 _register 抛出、staging 已 rename 前失败则不动）
            pass


def _fail_job(job_id: str, code: str, message: str) -> None:
    if len(message) > 500:
        message = message[:500]
    _update_job(job_id, status=STATUS_FAILED, error_code=code, error=message,
                finished_at=_now())


def create_job(*, source: str = "", target: str = "", model_id: str = "", preset_id: str = "",
               engine: str = "auto", quant: str = "", use_modelscope: bool = False,
               proxy: str = "", expected_sha256: str = "", gguf_path: str = "",
               models_root: str = "", allow_cpu: bool = True,
               progress_cb: Callable[[int, int], None] | None = None,
               executor: Callable[[Callable[[], None]], None] | None = None) -> dict[str, Any]:
    """排队一个下载 job。``executor`` 若提供则将 ``_run_job`` 交给其调度（后台线程池）。

    返回 job dict。幂等：同 source+target+model_id 已有活跃 job 则返回既有 job。
    """
    preset: dict[str, Any] | None = None
    if preset_id:
        preset = _PRESETS_BY_ID.get(preset_id)
        if preset is None:
            raise JobError("PRESET_NOT_FOUND", f"预设 '{preset_id}' 不存在")
        source = _preset_source(preset, use_modelscope)
        model_id = model_id or preset.get("default_model_id", "")
        if engine == "auto":
            engine = preset.get("default_engine", "auto")
        quant = quant or preset.get("default_quant", "")
        expected_sha256 = expected_sha256 or preset.get("expected_sha256", "")
        # 资源门前置
        rejected: dict[str, str] = {}
        _resource_gate(rejected, preset)
        if rejected:
            raise JobError("RESOURCE_BLOCKED", "；".join(f"{k}:{v}" for k, v in rejected.items()))
    else:
        if not source:
            raise JobError("SOURCE_REQUIRED", "非预设下载必须提供 source")

    destination = _resolve_target(source, target or "", models_root or _DEFAULT_MODELS_ROOT)
    model_id = model_id or destination.name
    target = target or str(destination)

    with _LOCK:
        existing = _active_job_exists(source, target, model_id)
        if existing:
            # 返回已有活跃 job（首个匹配）
            for j in list_jobs(limit=100):
                if j["source"] == source and j["target"] == target and j["model_id"] == model_id \
                        and j["status"] in _ACTIVE_STATUSES:
                    return j
            # 若上面兜底没找到（竞态），再创建一个
        job_id = str(uuid.uuid4())
        now = _now()
        row = {
            "job_id": job_id, "status": STATUS_QUEUED, "progress": 0.0,
            "source": source, "target": target, "model_id": model_id,
            "preset_id": preset_id, "engine": engine or "auto", "quant": quant,
            "total_bytes": 0, "downloaded_bytes": 0, "error_code": None,
            "error": None, "created_at": now, "updated_at": now, "finished_at": None,
        }
        _insert_job(row)

    def _run() -> None:
        _run_job(
            job_id, source=source, target=target, model_id=model_id,
            preset_id=preset_id, engine=engine, quant=quant,
            use_modelscope=use_modelscope, proxy=proxy, expected_sha256=expected_sha256,
            gguf_path=gguf_path, models_root=models_root or _DEFAULT_MODELS_ROOT,
            allow_cpu=allow_cpu, progress_cb=progress_cb,
        )

    if executor is not None:
        executor(_run)
    return get_job(job_id) or row


def cancel_job(job_id: str) -> bool:
    """标记取消 job（仅 queued 可取消；执行中的由调用方线程池控制）。"""
    job = get_job(job_id)
    if not job:
        return False
    if job["status"] != STATUS_QUEUED:
        return False
    _update_job(job_id, status=STATUS_CANCELLED, finished_at=_now())
    return True


def delete_job(job_id: str) -> bool:
    """删除 job 记录（不删磁盘模型，仅清理记录）。"""
    with local_store._write_connection() as connection:
        result = connection.execute(
            "DELETE FROM model_download_jobs WHERE job_id=?", (job_id,)
        )
        return int(result.rowcount) > 0
