"""Cross-PC llama.cpp RPC probe with an SSH-managed Windows worker.

The remote process is a backend only.  It never receives ``--model``; the
local host loads the same pinned GGUF and sends only the tensors selected by
llama.cpp.  This is an acceptance probe, not a production RPC supervisor.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import ntpath
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llama_rpc_contract import RpcShardLeaseBook
from src.llama_rpc_planner import (
    RpcNodeProfile,
    RpcSplitDecision,
    plan_rpc_split,
    plan_to_tensor_split,
)

from scripts.llama_rpc_sim import (
    DEFAULT_RUNTIME,
    _command_display,
    _display_path,
    _process_metrics,
    _start,
    _stop,
    _wait_http,
    parse_backend_evidence,
)


#: 默认本地模型：**现行**模型（历史默认是已退役的 `Qwen-1_8B-Chat.Q4_K_M.gguf`，见 R-R6）。
DEFAULT_LOCAL_MODEL = ROOT / "models" / "qwen3-0.6b-q8_0.gguf"
DEFAULT_REMOTE_ROOT = r"C:\Users\surface\Documents\LEDS_BJTU"
DEFAULT_REMOTE_RUNTIME = DEFAULT_REMOTE_ROOT + r"\runtime\llama-cpp\b10964"
DEFAULT_REMOTE_LOG_DIR = DEFAULT_REMOTE_ROOT + r"\local_docs\rpc_probe"

# Keep these values aligned with ggml/src/ggml-rpc/ggml-rpc.cpp.  The probe
# deliberately uses zero connection capabilities so it stays on the TCP
# transport and does not attempt RDMA negotiation.
RPC_PROTO_MAJOR_VERSION = 5
RPC_PROTO_MINOR_VERSION = 0
RPC_CMD_HELLO = 14
RPC_CMD_DEVICE_COUNT = 15
RPC_CONN_CAPS_SIZE = 24
RPC_HELLO_RESPONSE_SIZE = 28
RPC_READY_TIMEOUT_SECONDS = 30.0


def default_remote_model(local_model: str | Path | None = None) -> str:
    """远端模型路径 = `<remote_root>\\models\\<**本地模型文件名**>`（★ R-R6 复盘，2026-09-25）。

    旧行为**写死** `...\\models\\Qwen-1_8B-Chat.Q4_K_M.gguf`（已退役模型的遗留路径）⇒ 同步任何
    别的模型都会**覆盖那一个文件**，于是出现「文件名说 1.8B、内容是 2b」的名实不符，并且
    **多个模型在远端无法共存**（每次同步互相顶掉）。改为跟随本地文件名后：文件名即内容，
    0.6b 与 2b 各自独立存在。
    ⚠️ 仍可用显式 `--remote-model` 覆盖（例如刻意复用某个固定远端路径）。
    """
    name = Path(local_model).name if local_model else Path(DEFAULT_LOCAL_MODEL).name
    return DEFAULT_REMOTE_ROOT + "\\models\\" + name


@dataclass(frozen=True)
class PcRpcPlan:
    model: Path
    runtime_dir: Path
    remote_user: str
    remote_host: str
    remote_runtime_dir: str
    remote_model: str
    remote_log_dir: str
    rpc_port: int
    http_port: int
    gpu_layers: int | None
    ctx_size: int
    max_tokens: int
    worker_budget_mib: float
    timeout_seconds: float
    remote_threads: int = 8
    ssh_tunnel: bool = False
    total_layers: int = 25
    auto_split: bool = True
    #: 关闭权重 repack（llama.cpp 的 -nr/--no-repack）。跨路径数值对齐用：本机 CPU 的
    #: CPU_REPACK 与 RPC 路径（权重在远端 buffer，无法 repack）浮点结果不同，见文档 §11/§13。
    no_repack: bool = False

    @property
    def target(self) -> str:
        return f"{self.remote_user}@{self.remote_host}"

    @property
    def worker_command(self) -> list[str]:
        return [
            self.remote_runtime_dir + r"\ggml-rpc-server.exe",
            "--host",
            "127.0.0.1" if self.ssh_tunnel else self.remote_host,
            "--port",
            str(self.rpc_port),
            "--threads",
            str(self.remote_threads),
            "--device",
            "CPU",
        ]

    @property
    def host_command(self) -> list[str]:
        rpc_host = "127.0.0.1" if self.ssh_tunnel else self.remote_host
        selected_gpu_layers = self.gpu_layers if self.gpu_layers is not None else 0
        command = [
            str(self.runtime_dir / "llama-server.exe"),
            "--model",
            str(self.model),
            "--rpc",
            f"{rpc_host}:{self.rpc_port}",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.http_port),
            "--device",
            "RPC0",
            "--gpu-layers",
            str(selected_gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--n-predict",
            str(self.max_tokens),
            "--threads",
            "2",
            "--no-warmup",
            "--no-webui",
            "--log-verbosity",
            "4",
        ]
        if self.no_repack:
            # 跨路径数值对齐：关闭 CPU_REPACK（与 Python 引擎的 align_numerics 等价）
            command.append("--no-repack")
        return command


def build_plan(
    model: str | Path = DEFAULT_LOCAL_MODEL,
    runtime_dir: str | Path = DEFAULT_RUNTIME,
    *,
    remote_user: str = "surface",
    remote_host: str = "100.100.52.106",
    remote_runtime_dir: str = DEFAULT_REMOTE_RUNTIME,
    remote_model: str | None = None,   # None ⇒ 由 default_remote_model(model) 推导（★ R-R6）
    remote_log_dir: str = DEFAULT_REMOTE_LOG_DIR,
    rpc_port: int = 50163,
    http_port: int = 18093,
    gpu_layers: int | None = None,
    ctx_size: int = 128,
    max_tokens: int = 2,
    worker_budget_mib: float = 512,
    timeout_seconds: float = 120,
    remote_threads: int = 8,
    ssh_tunnel: bool = False,
    total_layers: int = 25,
    auto_split: bool | None = None,
    no_repack: bool = False,
) -> PcRpcPlan:
    if auto_split is None:
        auto_split = gpu_layers is None
    return PcRpcPlan(
        model=Path(model).expanduser().resolve(),
        runtime_dir=Path(runtime_dir).expanduser().resolve(),
        remote_user=remote_user,
        remote_host=remote_host,
        remote_runtime_dir=remote_runtime_dir,
        remote_model=remote_model or default_remote_model(model),
        remote_log_dir=remote_log_dir,
        rpc_port=rpc_port,
        http_port=http_port,
        gpu_layers=gpu_layers,
        ctx_size=ctx_size,
        max_tokens=max_tokens,
        worker_budget_mib=worker_budget_mib,
        timeout_seconds=timeout_seconds,
        remote_threads=remote_threads,
        ssh_tunnel=ssh_tunnel,
        total_layers=max(0, int(total_layers)),
        auto_split=bool(auto_split),
        no_repack=bool(no_repack),
    )


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ssh(plan: PcRpcPlan, script: str, *, timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(_ssh_command(plan, script), text=True, capture_output=True, timeout=timeout, check=False)


def _ssh_command(plan: PcRpcPlan, script: str) -> list[str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        plan.target,
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded,
    ]


def _remote_hash(plan: PcRpcPlan) -> str:
    script = (
        f"$h=Get-FileHash -Algorithm SHA256 -LiteralPath {_ps_quote(plan.remote_model)}; "
        "if ($null -eq $h) { exit 2 }; Write-Output $h.Hash"
    )
    result = _ssh(plan, script)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "remote model hash failed")
    for line in result.stdout.splitlines():
        value = line.strip().upper()
        if len(value) == 64 and all(char in "0123456789ABCDEF" for char in value):
            return value
    raise RuntimeError("remote model hash was not returned")


def _local_profile() -> dict[str, Any]:
    """单机自环（``--engine-local-worker``）用的本机节点画像，字段与 DeviceProfiler 同形。

    走 SSH 的 ``_remote_profile`` 在自环场景里没有意义（对端就是本机），这里直接采集本机；
    ``psutil`` 缺失时退化为保守常量。
    """
    cores = os.cpu_count() or 4
    total_gb, avail_gb = 16.0, 8.0
    try:
        import psutil  # 可选依赖

        memory = psutil.virtual_memory()
        total_gb = memory.total / 2 ** 30
        avail_gb = memory.available / 2 ** 30
    except Exception:  # noqa: BLE001 - 画像缺失不应阻断自环
        pass
    return {
        "node_id": socket.gethostname(),
        "source": "local-self-loop",
        "cpu_cores": cores,
        "cpu_freq_mhz": 2000,
        "cpu_load_percent": 5,
        "ram_total_gb": round(total_gb, 1),
        "ram_available_gb": round(avail_gb, 1),
    }


def _remote_profile(plan: PcRpcPlan) -> dict[str, Any]:
    """Read the same CPU/RAM fields used by DeviceProfiler from the worker."""
    script = r"""
$cpu=Get-CimInstance Win32_Processor
$os=Get-CimInstance Win32_OperatingSystem
$load=($cpu | Measure-Object -Property LoadPercentage -Average).Average
if ($null -eq $load) { $load=0 }
$freq=($cpu | Measure-Object -Property MaxClockSpeed -Maximum).Maximum
@{
  node_id = $env:COMPUTERNAME
  cpu_cores = [int](($cpu | Measure-Object -Property NumberOfCores -Sum).Sum)
  logical_cores = [int](($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)
  cpu_freq_mhz = [double]$freq
  cpu_load_percent = [double]$load
  ram_total_gb = [math]::Round([double]$os.TotalVisibleMemorySize / 1MB, 2)
  ram_available_gb = [math]::Round([double]$os.FreePhysicalMemory / 1MB, 2)
} | ConvertTo-Json -Compress
"""
    result = _ssh(plan, script, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "remote device profile failed")
    for line in reversed(result.stdout.splitlines()):
        try:
            profile = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(profile, dict):
            return profile
    raise RuntimeError("remote device profile was not returned")


def _split_decision(plan: PcRpcPlan, profile: dict[str, Any] | None = None) -> RpcSplitDecision:
    model_size_mib = 0.0
    try:
        model_size_mib = plan.model.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        pass
    if profile is None:
        profile = {}
    decision = plan_rpc_split(
        plan.total_layers,
        model_size_mib,
        profile,
        worker_budget_mib=plan.worker_budget_mib,
    )
    if not plan.auto_split:
        selected = max(0, min(plan.total_layers - 1, int(plan.gpu_layers or 0)))
        return replace(
            decision,
            admitted=selected > 0,
            strategy="manual",
            rpc_layers=selected,
            local_layers=max(0, plan.total_layers - selected),
            reason="manual_gpu_layers",
        )
    return decision


def _resolved_plan(plan: PcRpcPlan, decision: RpcSplitDecision) -> PcRpcPlan:
    return replace(plan, gpu_layers=decision.rpc_layers, auto_split=False)


def _lease_decision_payload(decision: Any) -> dict[str, Any]:
    payload = {
        "accepted": bool(decision.accepted),
        "reason": str(decision.reason),
        "result_digest": str(decision.result_digest or ""),
    }
    lease = decision.lease
    if lease is not None:
        payload["lease"] = {
            "shard_id": lease.shard_id,
            "worker_id": lease.worker_id,
            "lease_id": lease.lease_id,
            "epoch": lease.epoch,
            "attempt": lease.attempt,
            "status": lease.status,
            "lease_expires_at": lease.lease_expires_at,
            "lease_ttl_seconds": lease.lease_ttl_seconds,
        }
    return payload


def _local_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass
class RemoteWorkerHandle:
    process: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path
    pid: int | None = None
    rpc_ready: dict[str, Any] | None = None


@dataclass
class SshTunnelHandle:
    process: subprocess.Popen[str]
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class RemoteAssetSyncPlan:
    """Non-destructive transfer plan for the remote model identity file."""

    source: Path
    remote_target: str
    remote_path: str
    plan_sha256: str
    source_sha256: str
    source_size_bytes: int
    temporary_path: str
    backup_dir: str
    scp_command: tuple[str, ...]
    prepare_script: str
    commit_script: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "remote_target": self.remote_target,
            "remote_path": self.remote_path,
            "plan_sha256": self.plan_sha256,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "temporary_path": self.temporary_path,
            "backup_dir": self.backup_dir,
            "scp_command": list(self.scp_command),
            "prepare_script": self.prepare_script,
            "commit_script": self.commit_script,
        }

    def as_pipeline_artifact_availability(
        self, node_id: str, total_layers: int, *, sync_status: str,
    ) -> Any:
        """Expose a verified full-GGUF sync result to the reshard contract.

        The caller may use this only after ``sync_remote_model`` reports
        ``already_current`` or ``applied``.  The source file is a whole-model
        GGUF, so it covers every decoder layer in the declared topology.
        """
        if sync_status not in {"already_current", "applied"}:
            raise ValueError(
                "pipeline artifact availability requires a successful sync result",
            )
        from src.pipeline_reshard import PipelineArtifactAvailability

        return PipelineArtifactAvailability(
            node_id=node_id,
            model_sha256=self.source_sha256,
            artifact_kind="gguf",
            layer_range=(0, int(total_layers)),
            artifact_sha256=self.source_sha256,
            verified=True,
            has_embedding=True,
            has_lm_head=True,
        )


def _validate_remote_asset_path(remote_path: str) -> str:
    normalized = str(remote_path or "").replace("/", "\\")
    drive, tail = ntpath.splitdrive(normalized)
    if not drive or not tail.startswith("\\"):
        raise ValueError("remote model path must be an absolute Windows path")
    components = tail.strip("\\").split("\\")
    if not components or any(part in {"", ".", ".."} for part in components):
        raise ValueError("remote model path contains an unsafe component")
    if any(char in normalized for char in ("\x00", "\r", "\n")):
        raise ValueError("remote model path contains control characters")
    return ntpath.normpath(normalized)


def build_remote_asset_sync_plan(
    source: str | Path,
    remote_target: str,
    remote_path: str,
) -> RemoteAssetSyncPlan:
    """Build a reviewable, non-destructive remote GGUF transfer plan."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"local model not found: {source_path}")
    target = _validate_remote_asset_path(remote_path)
    if not remote_target or any(char in remote_target for char in ("\x00", "\r", "\n")):
        raise ValueError("remote target contains unsafe characters")

    source_sha256 = _local_hash(source_path)
    source_size = source_path.stat().st_size
    plan_sha256 = hashlib.sha256(
        f"rpc-model-sync-v1\0{remote_target}\0{target}\0{source_size}\0{source_sha256}".encode("utf-8")
    ).hexdigest().upper()
    parent, name = ntpath.split(target)
    temporary_path = ntpath.join(parent, f".{name}.qlh-{source_sha256[:16]}.part")
    backup_dir = ntpath.join(parent, "_to_delete", "remote-models")
    temporary_quote = _ps_quote(temporary_path)
    target_quote = _ps_quote(target)
    backup_quote = _ps_quote(backup_dir)
    expected_quote = _ps_quote(source_sha256)
    scp_target = f"{remote_target}:{temporary_path.replace('\\', '/')}"
    prepare_script = f"""
$ErrorActionPreference = 'Stop'
$temporary = {temporary_quote}
$backup = {backup_quote}
New-Item -ItemType Directory -Force -Path $backup | Out-Null
if (Test-Path -LiteralPath $temporary) {{
  $stale = Join-Path $backup ('.part-' + [guid]::NewGuid().ToString('N'))
  Move-Item -LiteralPath $temporary -Destination $stale
}}
Write-Output 'prepared'
""".strip()
    commit_script = f"""
$ErrorActionPreference = 'Stop'
$temporary = {temporary_quote}
$target = {target_quote}
$backup = {backup_quote}
$expected = {expected_quote}
if (!(Test-Path -LiteralPath $temporary)) {{ throw 'staged model file is missing' }}
$staged = Get-Item -LiteralPath $temporary
if ([int64]$staged.Length -ne {source_size}) {{ throw 'staged model size mismatch' }}
$stagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporary).Hash.ToUpperInvariant()
if ($stagedHash -ne $expected) {{ throw 'staged model SHA256 mismatch' }}
if (Test-Path -LiteralPath $target) {{
  $currentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToUpperInvariant()
  if ($currentHash -eq $expected) {{
    Move-Item -LiteralPath $temporary -Destination (Join-Path $backup ('.redundant-' + [guid]::NewGuid().ToString('N')))
    @{{ status = 'already_current'; sha256 = $currentHash }} | ConvertTo-Json -Compress
    exit 0
  }}
  $backupPath = Join-Path $backup ((Split-Path -Leaf $target) + '.replaced-' + [guid]::NewGuid().ToString('N'))
  [System.IO.File]::Replace($temporary, $target, $backupPath, $true)
  $finalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToUpperInvariant()
  if ($finalHash -ne $expected) {{ throw 'remote model SHA256 mismatch after replacement' }}
  @{{ status = 'applied'; sha256 = $finalHash }} | ConvertTo-Json -Compress
  exit 0
}}
Move-Item -LiteralPath $temporary -Destination $target
$finalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToUpperInvariant()
if ($finalHash -ne $expected) {{ throw 'remote model SHA256 mismatch after replacement' }}
@{{ status = 'applied'; sha256 = $finalHash }} | ConvertTo-Json -Compress
""".strip()
    return RemoteAssetSyncPlan(
        source=source_path,
        remote_target=remote_target,
        remote_path=target,
        plan_sha256=plan_sha256,
        source_sha256=source_sha256,
        source_size_bytes=source_size,
        temporary_path=temporary_path,
        backup_dir=backup_dir,
        scp_command=(
            "scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            str(source_path), scp_target,
        ),
        prepare_script=prepare_script,
        commit_script=commit_script,
    )


def _remote_asset_state(plan: PcRpcPlan, sync_plan: RemoteAssetSyncPlan) -> dict[str, Any]:
    script = f"""
$path = {_ps_quote(sync_plan.remote_path)}
if (!(Test-Path -LiteralPath $path)) {{ @{{ status = 'missing' }} | ConvertTo-Json -Compress; exit 0 }}
$item = Get-Item -LiteralPath $path
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToUpperInvariant()
@{{ status = 'present'; size_bytes = [int64]$item.Length; sha256 = $hash }} | ConvertTo-Json -Compress
""".strip()
    result = _ssh(plan, script, timeout=max(20.0, min(plan.timeout_seconds, 60.0)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "remote model state query failed")
    for line in reversed(result.stdout.splitlines()):
        try:
            state = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(state, dict) and state.get("status") in {"missing", "present"}:
            return state
    raise RuntimeError("remote model state was not returned")


def sync_remote_model(
    plan: PcRpcPlan,
    *,
    apply: bool = False,
    confirmation_sha256: str = "",
    transfer_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Inspect or explicitly apply a verified, non-destructive model sync."""
    sync_plan = build_remote_asset_sync_plan(plan.model, plan.target, plan.remote_model)
    state = _remote_asset_state(plan, sync_plan)
    current = (
        state.get("status") == "present"
        and int(state.get("size_bytes", -1)) == sync_plan.source_size_bytes
        and str(state.get("sha256", "")).upper() == sync_plan.source_sha256
    )
    report: dict[str, Any] = {
        "status": "already_current" if current else ("apply_required" if apply else "dry_run"),
        "remote_state": state,
        "current": current,
        "plan": sync_plan.to_dict(),
    }
    if current or not apply:
        return report
    if confirmation_sha256.strip().upper() != sync_plan.plan_sha256:
        raise ValueError(
            "remote model sync requires a prior dry-run and matching "
            "--confirm-sync-plan-sha256"
        )

    prepared = _ssh(plan, sync_plan.prepare_script, timeout=max(20.0, plan.timeout_seconds))
    if prepared.returncode != 0:
        raise RuntimeError(prepared.stderr.strip() or "remote model sync prepare failed")
    uploaded = subprocess.run(
        list(sync_plan.scp_command),
        text=True,
        capture_output=True,
        timeout=max(60.0, float(transfer_timeout_seconds)),
        check=False,
    )
    if uploaded.returncode != 0:
        raise RuntimeError(uploaded.stderr.strip() or "remote model upload failed")
    committed = _ssh(plan, sync_plan.commit_script, timeout=max(20.0, plan.timeout_seconds))
    if committed.returncode != 0:
        raise RuntimeError(committed.stderr.strip() or "remote model sync commit failed")
    result = None
    for line in reversed(committed.stdout.splitlines()):
        try:
            result = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            break
    report["status"] = "applied"
    report["commit"] = result or {"status": "applied"}
    return report


def _ssh_tunnel_command(plan: PcRpcPlan) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        f"{plan.rpc_port}:127.0.0.1:{plan.rpc_port}",
        plan.target,
    ]


def _tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class RpcProtocolError(RuntimeError):
    """The endpoint accepted TCP but did not speak the expected ggml RPC."""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except (ConnectionError, TimeoutError) as exc:
            raise RpcProtocolError(
                f"RPC response truncated ({size - remaining}/{size} bytes): {exc}"
            ) from exc
        if not chunk:
            raise RpcProtocolError(
                f"RPC response truncated ({size - remaining}/{size} bytes)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _rpc_frame(sock: socket.socket, command: int, payload: bytes) -> None:
    sock.sendall(bytes((command,)))
    sock.sendall(struct.pack("<Q", len(payload)))
    if payload:
        sock.sendall(payload)


def _rpc_response(sock: socket.socket, expected_size: int) -> bytes:
    raw_size = _recv_exact(sock, 8)
    response_size = struct.unpack("<Q", raw_size)[0]
    # A non-RPC peer can leave arbitrary bytes in the size slot.  Treat an
    # impossible frame length as a truncated/malformed response instead of
    # waiting for a nonsensical payload or reporting a misleading shape error.
    if response_size > 64 * 1024 * 1024:
        raise RpcProtocolError(
            f"RPC response truncated (invalid frame size {response_size})"
        )
    if response_size != expected_size:
        raise RpcProtocolError(
            f"RPC response size mismatch ({response_size} != {expected_size})"
        )
    return _recv_exact(sock, expected_size)


def _rpc_protocol_probe(host: str, port: int, *, timeout: float = 1.0) -> dict[str, Any]:
    """Perform the real ggml RPC HELLO and DEVICE_COUNT exchange.

    A listening socket is not sufficient: ggml-rpc-server only becomes usable
    after this protocol exchange succeeds and reports at least one backend
    device.  Protocol-shape errors are fatal; connection/timeouts are left to
    ``_wait_rpc_ready`` so startup can tolerate a slow worker.
    """
    sock = socket.create_connection((host, port), timeout=max(0.1, timeout))
    try:
        sock.settimeout(max(0.1, timeout))
        _rpc_frame(sock, RPC_CMD_HELLO, bytes(RPC_CONN_CAPS_SIZE))
        hello = _rpc_response(sock, RPC_HELLO_RESPONSE_SIZE)
        major, minor, patch, _padding = struct.unpack("<BBBB", hello[:4])
        if major != RPC_PROTO_MAJOR_VERSION or minor > RPC_PROTO_MINOR_VERSION:
            raise RpcProtocolError(
                "RPC protocol mismatch: "
                f"server={major}.{minor}.{patch}, "
                f"client={RPC_PROTO_MAJOR_VERSION}.{RPC_PROTO_MINOR_VERSION}.x"
            )

        _rpc_frame(sock, RPC_CMD_DEVICE_COUNT, b"")
        device_response = _rpc_response(sock, 4)
        device_count = struct.unpack("<I", device_response)[0]
        if device_count < 1:
            raise RpcProtocolError("RPC server reported zero devices")
        return {
            "ok": True,
            "transport": "tcp",
            "protocol_version": f"{major}.{minor}.{patch}",
            "device_count": device_count,
        }
    except (ConnectionError, TimeoutError) as exc:
        raise RpcProtocolError(
            f"RPC response truncated or peer disconnected during handshake: {exc}"
        ) from exc
    finally:
        sock.close()


def _wait_rpc_ready(
    host: str,
    port: int,
    timeout: float,
    *,
    process: subprocess.Popen[str] | None = None,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    """Wait for protocol readiness while distinguishing startup from failure."""
    deadline = time.monotonic() + max(0.1, timeout)
    last_error = "connection refused"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "RPC worker process exited before protocol ready "
                f"(rc={process.returncode})"
            )
        remaining = max(0.1, deadline - time.monotonic())
        try:
            return _rpc_protocol_probe(host, port, timeout=min(1.0, remaining))
        except RpcProtocolError:
            raise
        except (OSError, TimeoutError) as exc:
            last_error = str(exc) or exc.__class__.__name__
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(
        f"RPC endpoint {host}:{port} did not pass protocol ready within "
        f"{timeout:.1f}s: {last_error}"
    )


def _start_ssh_tunnel(plan: PcRpcPlan, run_id: str, temp_dir: Path) -> SshTunnelHandle:
    stdout_path = temp_dir / f"ssh-tunnel-{run_id}.stdout.log"
    stderr_path = temp_dir / f"ssh-tunnel-{run_id}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _ssh_tunnel_command(plan),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    handle = SshTunnelHandle(process, stdout_path, stderr_path)
    deadline = time.monotonic() + 12
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error = stderr_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"ssh tunnel exited during start: {error[-2000:]}")
            if _tcp_ready("127.0.0.1", plan.rpc_port):
                return handle
            time.sleep(0.25)
        raise RuntimeError("ssh tunnel did not expose the local RPC port")
    except Exception:
        _stop_ssh_tunnel(handle)
        raise


def _start_remote_worker(
    plan: PcRpcPlan,
    run_id: str,
    temp_dir: Path,
    *,
    ready_host: str | None = None,
    ready_timeout: float = RPC_READY_TIMEOUT_SECONDS,
) -> RemoteWorkerHandle:
    _remote_clear_worker_port(plan)
    stdout_path = temp_dir / f"remote-worker-{run_id}.stdout.log"
    stderr_path = temp_dir / f"remote-worker-{run_id}.stderr.log"
    exe = plan.remote_runtime_dir + r"\ggml-rpc-server.exe"
    script = f"""
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath {_ps_quote(plan.remote_runtime_dir)}
& {_ps_quote(exe)} --host {_ps_quote('127.0.0.1' if plan.ssh_tunnel else plan.remote_host)} --port {plan.rpc_port} --threads {plan.remote_threads} --device CPU
"""
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _ssh_command(plan, script),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    handle = RemoteWorkerHandle(process, stdout_path, stderr_path)
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error = stderr_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"remote worker exited during start: {error[-2000:]}")
            metrics = _remote_find_worker(plan)
            if metrics.get("pid"):
                handle.pid = int(metrics["pid"])
                if ready_host is not None:
                    handle.rpc_ready = _wait_rpc_ready(
                        ready_host,
                        plan.rpc_port,
                        ready_timeout,
                        process=process,
                    )
                return handle
            time.sleep(0.25)
        raise RuntimeError("remote worker did not appear in the process table")
    except Exception:
        _stop_remote_worker(plan, handle)
        raise


def _remote_find_worker(plan: PcRpcPlan) -> dict[str, Any]:
    script = f"""
$w=Get-CimInstance Win32_Process -Filter \"Name = 'ggml-rpc-server.exe'\" |
  Where-Object {{ $_.CommandLine -match '--port\\s+{plan.rpc_port}(\\s|$)' }} |
  Select-Object -First 1
if ($null -eq $w) {{ @{{ pid = $null }} | ConvertTo-Json -Compress }}
else {{
  $p=Get-Process -Id $w.ProcessId -ErrorAction SilentlyContinue
  if ($null -eq $p) {{ @{{ pid = $null }} | ConvertTo-Json -Compress }}
  else {{ @{{ pid = $p.Id; rss_bytes = $p.WorkingSet64; private_bytes = $p.PrivateMemorySize64; cpu_seconds = $p.CPU }} | ConvertTo-Json -Compress }}
}}
"""
    result = _ssh(plan, script)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


def _remote_clear_worker_port(plan: PcRpcPlan) -> None:
    """Stop only stale workers owned by this probe's configured RPC port."""
    script = f"""
$workers=Get-CimInstance Win32_Process -Filter \"Name = 'ggml-rpc-server.exe'\" |
  Where-Object {{ $_.CommandLine -match '--port\\s+{plan.rpc_port}(\\s|$)' }}
$workers | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}
"""
    _ssh(plan, script, timeout=15)


def _remote_process_metrics(plan: PcRpcPlan, pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"pid": None, "query_error": "remote worker pid unavailable"}
    script = f"""
$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue
if ($null -eq $p) {{ @{{ pid = {int(pid)}; exited = $true }} | ConvertTo-Json -Compress }}
else {{ @{{ pid = $p.Id; rss_bytes = $p.WorkingSet64; private_bytes = $p.PrivateMemorySize64; cpu_seconds = $p.CPU }} | ConvertTo-Json -Compress }}
"""
    result = _ssh(plan, script)
    if result.returncode != 0 or not result.stdout.strip():
        return {"pid": pid, "query_error": result.stderr.strip() or "no response"}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"pid": pid, "query_error": result.stdout.strip()}


def _stop_remote_worker(plan: PcRpcPlan, worker: RemoteWorkerHandle | None) -> None:
    if worker is None:
        return
    if worker.pid is not None:
        _ssh(plan, f"Stop-Process -Id {int(worker.pid)} -Force -ErrorAction SilentlyContinue", timeout=15)
    else:
        _remote_clear_worker_port(plan)
    if worker.process.poll() is None:
        worker.process.terminate()
        try:
            worker.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait(timeout=5)


def _stop_ssh_tunnel(tunnel: SshTunnelHandle | None) -> None:
    if tunnel is None or tunnel.process.poll() is not None:
        return
    tunnel.process.terminate()
    try:
        tunnel.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tunnel.process.kill()
        tunnel.process.wait(timeout=5)


def _chat_with_timeout(url: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "Reply with one short word: okay"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "status": response.status,
                "content": str(content).strip(),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            }
    except HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": "http_error"}
    except (OSError, URLError, TimeoutError):
        return {"ok": False, "status": None, "error": "connection_or_timeout"}


def plan_report(
    plan: PcRpcPlan,
    *,
    split_decision: RpcSplitDecision | None = None,
    remote_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_decision = split_decision or _split_decision(plan, remote_profile)
    return {
        "status": "dry_run",
        "local_model": _display_path(plan.model),
        "remote_model": plan.remote_model,
        "remote_target": plan.target,
        "rpc_endpoint": f"{'127.0.0.1' if plan.ssh_tunnel else plan.remote_host}:{plan.rpc_port}",
        "loopback": plan.ssh_tunnel,
        "ssh_tunnel": plan.ssh_tunnel,
        "ssh_tunnel_command": _command_display(_ssh_tunnel_command(plan)) if plan.ssh_tunnel else None,
        "worker_backend": "CPU",
        "remote_threads": plan.remote_threads,
        "host_command": _command_display(_resolved_plan(plan, split_decision).host_command),
        "remote_worker_command": plan.worker_command,
        "remote_worker_receives_model": False,
        "model_identity_required": True,
        "configured_worker_budget_mib": plan.worker_budget_mib,
        "split_decision": split_decision.to_dict(),
        "automatic_split": plan.auto_split,
        "torch_imported": False,
    }


ENGINE_SMOKE = ROOT / "scripts" / "llama_rpc_engine_smoke.py"


def _run_engine_python(
    plan: PcRpcPlan,
    decision: RpcSplitDecision,
    *,
    python_exe: str,
    local_worker: bool,
    run_id: str,
    timeout: float | None = None,
    align_numerics: bool = False,
) -> dict[str, Any]:
    """用 ``LlamaCppEngine`` 作为执行端（替代 ``llama-server.exe --rpc``）。

    复用本脚本已有的远端 worker / SSH tunnel 生命周期，把 endpoint 与**planner 的层段决策
    换算出的 tensor_split** 交给引擎；所有实际加载与生成都在 ``llama_rpc_engine_smoke.py``
    里完成（需要带 GGML_RPC 的解释器，因此走子进程）。
    """
    split = plan_to_tensor_split(decision)
    result: dict[str, Any] = {
        "executor": "LlamaCppEngine(rpc device injection)",
        "python": python_exe,
        "local_worker": local_worker,
        "decision": {
            "rpc_layers": decision.rpc_layers,
            "local_layers": decision.local_layers,
            "total_layers": decision.total_layers,
            "reason": decision.reason,
        },
        "tensor_split": split,
        "endpoint": None,
    }
    if not Path(python_exe).is_file():
        result.update({"ok": False, "error": f"engine interpreter not found: {python_exe}"})
        return result
    if split is None:
        result.update({"ok": False, "error": "planner 未把任何层分给远端（rpc_layers=0）"})
        return result

    remote: RemoteWorkerHandle | None = None
    tunnel: SshTunnelHandle | None = None
    with tempfile.TemporaryDirectory(prefix="qlh-pc-rpc-engine-") as temp_dir:
        temp = Path(temp_dir)
        report_path = temp / "engine-report.json"
        try:
            if local_worker:
                # 不依赖 SSH：worker 由引擎在本机起（单机自环验证）
                endpoint = None
                result["endpoint"] = f"127.0.0.1:{plan.rpc_port} (engine-managed)"
            else:
                if plan.ssh_tunnel:
                    tunnel = _start_ssh_tunnel(plan, run_id, temp)
                remote = _start_remote_worker(
                    plan,
                    run_id,
                    temp,
                    ready_host="127.0.0.1" if plan.ssh_tunnel else plan.remote_host,
                    ready_timeout=plan.timeout_seconds,
                )
                host = "127.0.0.1" if plan.ssh_tunnel else plan.remote_host
                endpoint = f"{host}:{plan.rpc_port}"
                result["endpoint"] = endpoint
                result["remote_worker"] = {"pid": remote.pid, "stderr": str(remote.stderr_path)}
                result["rpc_ready"] = remote.rpc_ready

            command = [
                python_exe,
                str(ENGINE_SMOKE),
                "--model", str(plan.model),
                "--port", str(plan.rpc_port),
                "--threads", str(plan.remote_threads),
                "--n-predict", str(plan.max_tokens),
                "--split", ",".join(f"{value:.6f}" for value in split),
                "--report-json", str(report_path),
            ]
            if endpoint is None:
                # worker 必须与**被注入的那个 llama.dll** 协议一致：优先用 b_rpc 那份
                # （与 RPC 版 llama-cpp-python 同源码树编），否则回退 runtime_dir 里的官方构建
                candidate = ROOT / "runtime/llama-cpp/b_rpc/ggml-rpc-server.exe"
                worker_exe = candidate if candidate.is_file() else plan.runtime_dir / "ggml-rpc-server.exe"
                result["worker_exe"] = str(worker_exe)
                command += ["--worker-exe", str(worker_exe)]
            else:
                command += ["--rpc-servers", endpoint]
            if align_numerics:
                # 与 CLI 路径的 --no-repack 等价：关闭 CPU_REPACK，跨路径逐比特一致
                command.append("--align-numerics")

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or max(120.0, plan.timeout_seconds),
            )
            result["command"] = _command_display(command)
            result["returncode"] = completed.returncode
            result["stdout_tail"] = (completed.stdout or "")[-4000:]
            result["stderr_tail"] = (completed.stderr or "")[-2000:]
            if report_path.is_file():
                result["engine_report"] = json.loads(report_path.read_text(encoding="utf-8"))
            result["ok"] = bool(
                completed.returncode == 0
                and result.get("engine_report", {}).get("ok")
            )
            if not result["ok"]:
                result["error"] = "engine 路径未完成（见 stdout/stderr tail）"
        except subprocess.TimeoutExpired as exc:
            result.update({"ok": False, "error": f"engine 路径超时: {exc}"})
        finally:
            if remote is not None:
                try:
                    _stop_remote_worker(plan, remote)
                except Exception as exc:  # noqa: BLE001 - 清理不应掩盖主结果
                    result.setdefault("cleanup_warnings", []).append(f"remote worker: {exc}")
            if tunnel is not None:
                try:
                    _stop_ssh_tunnel(tunnel)
                except Exception as exc:  # noqa: BLE001
                    result.setdefault("cleanup_warnings", []).append(f"ssh tunnel: {exc}")
    return result


def run_probe(plan: PcRpcPlan, *, check_fallback: bool = True, engine: str = "cli",
              engine_python: str | None = None, engine_local_worker: bool = False,
              align_numerics: bool = False) -> dict[str, Any]:
    if not plan.model.is_file():
        raise FileNotFoundError(f"local model not found: {plan.model}")
    for name in ("llama-server.exe", "ggml-rpc-server.exe"):
        if not (plan.runtime_dir / name).is_file():
            raise FileNotFoundError(f"local runtime asset not found: {plan.runtime_dir / name}")

    local_hash = _local_hash(plan.model)
    self_loop = engine == "python" and engine_local_worker
    if self_loop:
        # 单机自环：worker 由引擎在本机起，跳过一切 SSH 采集
        remote_hash = local_hash
        remote_profile = _local_profile()
    else:
        remote_hash = _remote_hash(plan)
        remote_profile = None
    requested_plan = plan
    report = plan_report(plan)
    report.update(
        {
            "local_model_sha256": local_hash,
            "remote_model_sha256": remote_hash,
            "model_identity_match": local_hash == remote_hash,
            "self_loop": self_loop,
        }
    )
    if local_hash != remote_hash:
        report["status"] = "model_identity_mismatch"
        return report

    if not self_loop:
        try:
            remote_profile = _remote_profile(plan)
        except RuntimeError as exc:
            report.update({"status": "split_profile_unavailable", "split_profile_error": str(exc)})
            return report
    decision = _split_decision(plan, remote_profile)
    report = plan_report(
        requested_plan,
        split_decision=decision,
        remote_profile=remote_profile,
    )
    report.update(
        {
            "local_model_sha256": local_hash,
            "remote_model_sha256": remote_hash,
            "model_identity_match": True,
            "remote_device_profile": remote_profile,
            "self_loop": self_loop,
        }
    )
    if requested_plan.auto_split and not decision.admitted:
        report["status"] = "split_not_admitted"
        return report
    plan = _resolved_plan(plan, decision)

    lease_book = RpcShardLeaseBook()
    remote_lease = lease_book.assign(
        "llama-rpc-offload",
        "local-self-loop" if self_loop else plan.target,
        local_hash,
        {"backend": "RPC0", "gpu_layers": plan.gpu_layers, "total_layers": plan.total_layers},
        lease_seconds=max(30.0, plan.timeout_seconds + 60.0),
    )
    report["lease"] = {
        "schema": "llama-rpc-lease-v1",
        "initial": {
            "shard_id": remote_lease.shard_id,
            "worker_id": remote_lease.worker_id,
            "lease_id": remote_lease.lease_id,
            "epoch": remote_lease.epoch,
            "attempt": remote_lease.attempt,
            "lease_expires_at": remote_lease.lease_expires_at,
            "lease_ttl_seconds": remote_lease.lease_ttl_seconds,
            "allocation": dict(remote_lease.allocation),
        },
    }

    run_id = f"{int(time.time())}-{plan.rpc_port}"

    if engine == "python":
        # 执行端换成 LlamaCppEngine：复用上面的 decision 与 lease 记录
        report["engine"] = "python"
        report["engine_result"] = _run_engine_python(
            plan,
            decision,
            python_exe=engine_python or str(ROOT / ".venv-llama-rpc/Scripts/python.exe"),
            local_worker=engine_local_worker,
            run_id=run_id,
            timeout=plan.timeout_seconds,
            align_numerics=align_numerics,
        )
        report["status"] = "passed" if report["engine_result"].get("ok") else "engine_failed"
        return report

    remote: RemoteWorkerHandle | None = None
    tunnel: SshTunnelHandle | None = None
    host: subprocess.Popen[str] | None = None
    remote_stopped = False
    tunnel_stopped = False
    with tempfile.TemporaryDirectory(prefix="qlh-pc-rpc-") as temp_dir:
        temp = Path(temp_dir)
        host_err_path = temp / "host.stderr.log"
        try:
            if plan.ssh_tunnel:
                tunnel = _start_ssh_tunnel(plan, run_id, temp)
                report["ssh_tunnel"] = {
                    "stdout": str(tunnel.stdout_path),
                    "stderr": str(tunnel.stderr_path),
                }
            remote = _start_remote_worker(
                plan,
                run_id,
                temp,
                ready_host="127.0.0.1" if plan.ssh_tunnel else plan.remote_host,
                ready_timeout=plan.timeout_seconds,
            )
            report["remote_worker"] = {
                "pid": remote.pid,
                "stdout": str(remote.stdout_path),
                "stderr": str(remote.stderr_path),
                "rpc_ready": remote.rpc_ready,
            }

            host_out = (temp / "host.stdout.log").open("w", encoding="utf-8")
            host_err = host_err_path.open("w", encoding="utf-8")
            try:
                host = _start(plan.host_command, plan.runtime_dir, host_out, host_err)
                ready = _wait_http(f"http://127.0.0.1:{plan.http_port}/health", plan.timeout_seconds)
                report["host"] = _process_metrics(host)
                report["ready"] = ready
                report["remote_worker_before_request"] = _remote_process_metrics(plan, remote.pid)
                report["lease"]["renew_before_request"] = _lease_decision_payload(
                    lease_book.renew(remote_lease.lease_id, remote_lease.epoch)
                )
                if ready:
                    report["distributed_response"] = _chat_with_timeout(
                        f"http://127.0.0.1:{plan.http_port}/v1/chat/completions",
                        plan.max_tokens,
                        min(60, plan.timeout_seconds),
                    )
                    time.sleep(1)
                    report["host_after_request"] = _process_metrics(host)
                    report["remote_worker_after_request"] = _remote_process_metrics(plan, remote.pid)
                    if report["distributed_response"].get("ok"):
                        result = report["distributed_response"].get("content", "")
                        report["lease"]["remote_commit"] = _lease_decision_payload(
                            lease_book.commit(remote_lease.lease_id, remote_lease.epoch, result)
                        )
                else:
                    report["distributed_response"] = {"ok": False, "error": "host_not_ready"}

                report["remote_worker_before_stop"] = _remote_process_metrics(plan, remote.pid)

                _stop_remote_worker(plan, remote)
                remote_stopped = True
                _stop_ssh_tunnel(tunnel)
                tunnel_stopped = True
                report["remote_worker_disconnect"] = _chat_with_timeout(
                    f"http://127.0.0.1:{plan.http_port}/v1/chat/completions",
                    plan.max_tokens,
                    8,
                )
                report["disconnect_detected"] = not report["remote_worker_disconnect"].get("ok", False)
                if report["disconnect_detected"]:
                    fallback_lease = lease_book.reassign(
                        remote_lease.shard_id,
                        "local-cpu-fallback",
                        local_hash,
                        {"backend": "local", "gpu_layers": 0, "total_layers": plan.total_layers},
                    )
                    stale = lease_book.commit(
                        remote_lease.lease_id,
                        remote_lease.epoch,
                        "stale remote result after worker loss",
                    )
                    report["lease"]["reassignment"] = {
                        "worker_id": fallback_lease.worker_id,
                        "lease_id": fallback_lease.lease_id,
                        "epoch": fallback_lease.epoch,
                        "attempt": fallback_lease.attempt,
                        "stale_remote_commit": _lease_decision_payload(stale),
                    }
                else:
                    fallback_lease = None
            finally:
                _stop(host)
                host_out.close()
                host_err.close()

            if check_fallback:
                fallback_port = plan.http_port + 1
                fallback_command = [
                    str(plan.runtime_dir / "llama-server.exe"),
                    "--model",
                    str(plan.model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(fallback_port),
                    "--device",
                    "none",
                    "--gpu-layers",
                    "0",
                    "--ctx-size",
                    str(plan.ctx_size),
                    "--n-predict",
                    str(plan.max_tokens),
                    "--threads",
                    "2",
                    "--no-warmup",
                    "--no-webui",
                    "--log-verbosity",
                    "2",
                ]
                fallback_out = (temp / "fallback.stdout.log").open("w", encoding="utf-8")
                fallback_err = (temp / "fallback.stderr.log").open("w", encoding="utf-8")
                fallback: subprocess.Popen[str] | None = None
                try:
                    fallback = _start(fallback_command, plan.runtime_dir, fallback_out, fallback_err)
                    fallback_ready = _wait_http(f"http://127.0.0.1:{fallback_port}/health", plan.timeout_seconds)
                    report["fallback"] = {
                        "ready": fallback_ready,
                        "response": _chat_with_timeout(
                            f"http://127.0.0.1:{fallback_port}/v1/chat/completions",
                            plan.max_tokens,
                            min(30, plan.timeout_seconds),
                        )
                        if fallback_ready
                        else {"ok": False, "error": "fallback_not_ready"},
                        "process": _process_metrics(fallback),
                        "command": _command_display(fallback_command),
                    }
                    if fallback_lease is not None and report["fallback"]["response"].get("ok"):
                        report["lease"]["fallback_commit"] = _lease_decision_payload(
                            lease_book.commit(
                                fallback_lease.lease_id,
                                fallback_lease.epoch,
                                report["fallback"]["response"].get("content", ""),
                            )
                        )
                finally:
                    _stop(fallback)
                    fallback_out.close()
                    fallback_err.close()

            report["output_match"] = bool(
                report.get("distributed_response", {}).get("ok")
                and report.get("fallback", {}).get("response", {}).get("ok")
                and report["distributed_response"].get("content", "").strip()
                == report["fallback"]["response"].get("content", "").strip()
            )
            server_log = host_err_path.read_text(encoding="utf-8", errors="replace")
            worker_log = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (remote.stdout_path, remote.stderr_path)
                if path.is_file()
            )
            report["backend_evidence"] = parse_backend_evidence(
                server_log,
                worker_log,
                worker_budget_mib=plan.worker_budget_mib,
            )
            report["logs"] = {
                "host_stderr_tail": server_log[-12000:],
                "remote_worker_tail": worker_log[-12000:],
            }
            report["host_exit_code"] = host.returncode if host is not None else None
            report["remote_worker_exit_code"] = remote.process.returncode if remote is not None else None
        finally:
            if not tunnel_stopped:
                _stop_ssh_tunnel(tunnel)
            if not remote_stopped:
                _stop_remote_worker(plan, remote)

    report["status"] = (
        "passed"
        if report.get("model_identity_match")
        and report.get("ready")
        and report.get("distributed_response", {}).get("ok")
        and report.get("backend_evidence", {}).get("partial_backend_residency_proven")
        and report.get("disconnect_detected")
        and report.get("lease", {}).get("reassignment", {}).get("stale_remote_commit", {}).get("accepted") is False
        and report.get("lease", {}).get("fallback_commit", {}).get("accepted", not check_fallback)
        and (not check_fallback or report.get("output_match"))
        else "failed_acceptance"
    )
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--model", default=str(DEFAULT_LOCAL_MODEL))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--remote-user", default="surface")
    parser.add_argument("--remote-host", default="100.100.52.106")
    parser.add_argument("--remote-runtime-dir", default=DEFAULT_REMOTE_RUNTIME)
    parser.add_argument("--remote-model", default=None,
                        help="远端模型路径；缺省 ⇒ 按**本地模型文件名**推导"
                             "（`default_remote_model()`，★ R-R6）")
    parser.add_argument("--remote-log-dir", default=DEFAULT_REMOTE_LOG_DIR)
    parser.add_argument("--rpc-port", type=int, default=50163)
    parser.add_argument("--http-port", type=int, default=18093)
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="manual RPC layer count; omit to let the device-profile planner decide",
    )
    parser.add_argument("--total-layers", type=int, default=25)
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--worker-budget-mib", type=float, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--remote-threads", type=int, default=8)
    sync_mode = parser.add_mutually_exclusive_group()
    sync_mode.add_argument(
        "--sync-remote-model",
        action="store_true",
        help="query remote GGUF state and print a verified sync plan; do not write",
    )
    sync_mode.add_argument(
        "--apply-remote-model-sync",
        action="store_true",
        help="upload and atomically replace the remote GGUF after SHA256 verification",
    )
    parser.add_argument(
        "--confirm-sync-plan-sha256",
        default="",
        help="plan digest emitted by a prior --sync-remote-model dry-run",
    )
    parser.add_argument(
        "--asset-sync-timeout-seconds",
        type=float,
        default=3600,
        help="timeout for the large model transfer only (default: 3600)",
    )
    parser.add_argument(
        "--ssh-tunnel",
        action="store_true",
        help="bind the remote worker to loopback and forward RPC over an SSH local port",
    )
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--engine",
        choices=("cli", "python"),
        default="cli",
        help="执行端：cli=llama-server.exe --rpc（原路径）；python=LlamaCppEngine 注入 RPC device",
    )
    parser.add_argument(
        "--engine-python",
        default=str(ROOT / ".venv-llama-rpc/Scripts/python.exe"),
        help="--engine python 用的解释器（需带 GGML_RPC 的 llama-cpp-python）",
    )
    parser.add_argument(
        "--engine-local-worker",
        action="store_true",
        help="--engine python 时由引擎在本机起 worker（不依赖 SSH，用于单机自环）",
    )
    parser.add_argument(
        "--align-numerics",
        action="store_true",
        help="关闭权重 repack（CPU_REPACK）：cli 模式加 --no-repack，python 模式注入 "
             "use_extra_bufts=False，使本机与 RPC 路径逐比特一致（见文档 §11/§13）",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    plan = build_plan(
        args.model,
        args.runtime_dir,
        remote_user=args.remote_user,
        remote_host=args.remote_host,
        remote_runtime_dir=args.remote_runtime_dir,
        remote_model=args.remote_model,
        remote_log_dir=args.remote_log_dir,
        rpc_port=args.rpc_port,
        http_port=args.http_port,
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        max_tokens=args.max_tokens,
        worker_budget_mib=args.worker_budget_mib,
        timeout_seconds=args.timeout_seconds,
        remote_threads=args.remote_threads,
        ssh_tunnel=args.ssh_tunnel,
        total_layers=args.total_layers,
        no_repack=args.align_numerics,
    )
    try:
        if args.sync_remote_model or args.apply_remote_model_sync:
            report = sync_remote_model(
                plan,
                apply=args.apply_remote_model_sync,
                confirmation_sha256=args.confirm_sync_plan_sha256,
                transfer_timeout_seconds=args.asset_sync_timeout_seconds,
            )
        else:
            report = plan_report(plan) if not args.run else run_probe(
                plan,
                check_fallback=not args.no_fallback,
                engine=args.engine,
                engine_python=args.engine_python,
                engine_local_worker=args.engine_local_worker,
                align_numerics=args.align_numerics,
            )
    except (FileNotFoundError, OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        report = {"status": "invalid", "error": str(exc)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.json or not args.run:
        print(encoded)
    else:
        print(f"status={report.get('status')} model_identity_match={report.get('model_identity_match')}")
    return 0 if report.get("status") in {"dry_run", "passed", "already_current", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
