"""Reproducible same-host llama.cpp RPC sharding probe.

The probe deliberately uses the native llama.cpp executables.  It is a
measurement tool, not a production RPC launcher: RPC remains loopback-only
and the run must prove that the host allocated both CPU and RPC buffers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import psutil
except ImportError:  # pragma: no cover - the edge environment includes psutil
    psutil = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "runtime" / "llama-cpp" / "b10964"
DEFAULT_MODEL = ROOT / "models" / "distilqwen25-ds3-0324-7b-q4_k_m.gguf"


@dataclass(frozen=True)
class ProbePlan:
    model: Path
    runtime_dir: Path
    rpc_port: int
    http_port: int
    gpu_layers: int
    ctx_size: int
    max_tokens: int
    worker_budget_mib: float

    @property
    def worker_command(self) -> list[str]:
        return [
            str(self.runtime_dir / "ggml-rpc-server.exe"),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.rpc_port),
            "--device",
            "CUDA0",
        ]

    @property
    def host_command(self) -> list[str]:
        return [
            str(self.runtime_dir / "llama-server.exe"),
            "--model",
            str(self.model),
            "--rpc",
            f"127.0.0.1:{self.rpc_port}",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.http_port),
            "--device",
            "RPC0",
            "--gpu-layers",
            str(self.gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--n-predict",
            str(self.max_tokens),
            "--threads",
            "2",
            "--no-webui",
            "--log-verbosity",
            "4",
        ]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_plan(
    model: str | Path = DEFAULT_MODEL,
    runtime_dir: str | Path = DEFAULT_RUNTIME,
    *,
    rpc_port: int = 50162,
    http_port: int = 18090,
    gpu_layers: int = 12,
    ctx_size: int = 256,
    max_tokens: int = 2,
    worker_budget_mib: float = 2048,
) -> ProbePlan:
    return ProbePlan(
        model=Path(model).expanduser().resolve(),
        runtime_dir=Path(runtime_dir).expanduser().resolve(),
        rpc_port=rpc_port,
        http_port=http_port,
        gpu_layers=gpu_layers,
        ctx_size=ctx_size,
        max_tokens=max_tokens,
        worker_budget_mib=worker_budget_mib,
    )


def _command_display(command: Iterable[str]) -> list[str]:
    values = list(command)
    if values and Path(values[0]).is_absolute():
        values[0] = _display_path(Path(values[0]))
    return values


def plan_report(plan: ProbePlan) -> dict[str, Any]:
    required = [plan.runtime_dir / "ggml-rpc-server.exe", plan.runtime_dir / "llama-server.exe"]
    return {
        "status": "dry_run",
        "model": _display_path(plan.model),
        "model_bytes": plan.model.stat().st_size if plan.model.is_file() else None,
        "runtime_dir": _display_path(plan.runtime_dir),
        "assets_present": all(path.is_file() for path in required) and plan.model.is_file(),
        # This script intentionally has no torch import; do not inspect the
        # caller's module table because pytest may have loaded torch elsewhere.
        "torch_imported": False,
        "worker_command": _command_display(plan.worker_command),
        "host_command": _command_display(plan.host_command),
        "sharding_contract": {
            "host_device": "RPC0",
            "worker_device": "CUDA0",
            "requires_cpu_and_rpc_buffers": True,
            "reject_full_model_copy": True,
            "loopback_only": True,
            "configured_worker_budget_mib": plan.worker_budget_mib,
        },
    }


_SIZE_RE = re.compile(
    r"(?P<label>CPU_Mapped|CPU_REPACK|RPC0[^\n]*?)\s+model buffer size\s*=\s*(?P<value>[0-9.]+)\s*(?P<unit>MiB|GiB|KiB)",
    re.IGNORECASE,
)
_OFFLOAD_RE = re.compile(r"offloaded\s+(?P<loaded>\d+)\s*/\s*(?P<total>\d+)\s+layers", re.IGNORECASE)
_KV_RE = re.compile(r"RPC0[^\n]*KV buffer size\s*=\s*(?P<value>[0-9.]+)\s*MiB", re.IGNORECASE)
_COMPUTE_RE = re.compile(r"RPC0[^\n]*compute buffer size\s*=\s*(?P<value>[0-9.]+)\s*MiB", re.IGNORECASE)


def _to_mib(value: str, unit: str) -> float:
    factor = {"kib": 1 / 1024, "mib": 1, "gib": 1024}[unit.lower()]
    return round(float(value) * factor, 3)


def parse_backend_evidence(
    server_log: str,
    worker_log: str = "",
    *,
    worker_budget_mib: float | None = 2048,
) -> dict[str, Any]:
    buffers: dict[str, float] = {}
    for match in _SIZE_RE.finditer(server_log):
        label = match.group("label").strip()
        if label.upper().startswith("RPC0"):
            label = "RPC0"
        buffers[label] = _to_mib(match.group("value"), match.group("unit"))

    offload = _OFFLOAD_RE.search(server_log)
    kv = _KV_RE.search(server_log)
    compute = _COMPUTE_RE.search(server_log)
    worker_model_path_seen = bool(re.search(r"\.gguf|loading model|model buffer", worker_log, re.IGNORECASE))
    loaded = int(offload.group("loaded")) if offload else None
    total = int(offload.group("total")) if offload else None
    cpu_repack_mib = buffers.get("CPU_REPACK", 0)
    rpc_model_mib = buffers.get("RPC0", 0)
    rpc_aux_mib = (float(kv.group("value")) if kv else 0) + (float(compute.group("value")) if compute else 0)
    remote_resident_mib = round(rpc_model_mib + rpc_aux_mib, 3)
    distributed_resident_mib = round(cpu_repack_mib + remote_resident_mib, 3)
    capacity_merge = bool(
        worker_budget_mib is not None
        and worker_budget_mib > 0
        and remote_resident_mib <= worker_budget_mib
        and distributed_resident_mib > worker_budget_mib
    )
    partial = bool(
        loaded is not None
        and total is not None
        and 0 < loaded < total
        and buffers.get("RPC0", 0) > 0
        and any(key.startswith("CPU") and value > 0 for key, value in buffers.items())
        and not worker_model_path_seen
    )
    return {
        "offloaded_layers": loaded,
        "total_layers": total,
        "model_buffers_mib": buffers,
        "rpc_kv_buffer_mib": float(kv.group("value")) if kv else None,
        "rpc_compute_buffer_mib": float(compute.group("value")) if compute else None,
        "remote_resident_mib": remote_resident_mib,
        "distributed_resident_mib": distributed_resident_mib,
        "configured_worker_budget_mib": worker_budget_mib,
        "worker_model_path_seen": worker_model_path_seen,
        "partial_backend_residency_proven": partial,
        "capacity_merge_proven": capacity_merge,
        "physical_single_process_oom_proven": False,
        "capacity_note": "容量门使用可复现的 worker budget；不冒充物理单进程 OOM 证据。",
    }


def _read_log(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _process_metrics(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    if process is None or psutil is None:
        return {}
    try:
        info = psutil.Process(process.pid).memory_info()
        result: dict[str, Any] = {
            "pid": process.pid,
            "rss_bytes": int(info.rss),
            "rss_mib": round(info.rss / 2**20, 2),
        }
        try:
            full = psutil.Process(process.pid).memory_full_info()
            for field in ("uss", "pss"):
                if hasattr(full, field):
                    value = int(getattr(full, field))
                    result[f"{field}_bytes"] = value
                    result[f"{field}_mib"] = round(value / 2**20, 2)
        except (OSError, psutil.Error):
            pass
        return result
    except (OSError, psutil.Error):
        return {"pid": process.pid, "exited": process.poll() is not None}


def _wait_http(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (OSError, HTTPError, URLError):
            time.sleep(0.2)
    return False


def _chat(url: str, max_tokens: int) -> dict[str, Any]:
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
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "status": response.status,
                "content": str(content).strip(),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            }
    except HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "error": "http_error",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    except (OSError, URLError, TimeoutError):
        return {
            "ok": False,
            "status": None,
            "error": "connection_error",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }


def _stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _start(command: list[str], cwd: Path, stdout: Any, stderr: Any) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PATH"] = str(cwd) + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )


def run_probe(plan: ProbePlan, *, check_fallback: bool = True) -> dict[str, Any]:
    if not plan.model.is_file():
        raise FileNotFoundError(f"model not found: {plan.model}")
    for name in ("ggml-rpc-server.exe", "llama-server.exe"):
        if not (plan.runtime_dir / name).is_file():
            raise FileNotFoundError(f"runtime asset not found: {plan.runtime_dir / name}")

    report: dict[str, Any] = plan_report(plan)
    report["status"] = "running"
    report["model"] = _display_path(plan.model)
    report["model_bytes"] = plan.model.stat().st_size
    with tempfile.TemporaryDirectory(prefix="qlh-llama-rpc-") as temp_dir:
        temp = Path(temp_dir)
        worker_out = (temp / "worker.out.log").open("w", encoding="utf-8")
        worker_err = (temp / "worker.err.log").open("w", encoding="utf-8")
        host_out = (temp / "host.out.log").open("w", encoding="utf-8")
        host_err = (temp / "host.err.log").open("w", encoding="utf-8")
        worker: subprocess.Popen[str] | None = None
        host: subprocess.Popen[str] | None = None
        try:
            worker = _start(plan.worker_command, plan.runtime_dir, worker_out, worker_err)
            time.sleep(1.5)
            host = _start(plan.host_command, plan.runtime_dir, host_out, host_err)
            ready = _wait_http(f"http://127.0.0.1:{plan.http_port}/health", 90)
            report["ready"] = ready
            report["worker"] = _process_metrics(worker)
            report["host"] = _process_metrics(host)
            if not ready:
                report["status"] = "failed_to_start"
                return report

            report["distributed_response"] = _chat(
                f"http://127.0.0.1:{plan.http_port}/v1/chat/completions", plan.max_tokens
            )
            time.sleep(1)
            report["worker_after_request"] = _process_metrics(worker)
            report["host_after_request"] = _process_metrics(host)

            _stop(worker)
            disconnected = _chat(
                f"http://127.0.0.1:{plan.http_port}/v1/chat/completions", plan.max_tokens
            )
            report["after_worker_disconnect"] = disconnected
            report["disconnect_detected"] = not bool(disconnected.get("ok"))
            _stop(host)

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
                    "--gpu-layers",
                    "0",
                    "--ctx-size",
                    str(min(plan.ctx_size, 128)),
                    "--n-predict",
                    str(plan.max_tokens),
                    "--threads",
                    "2",
                    "--no-webui",
                    "--log-verbosity",
                    "2",
                ]
                fallback_out = (temp / "fallback.out.log").open("w", encoding="utf-8")
                fallback_err = (temp / "fallback.err.log").open("w", encoding="utf-8")
                fallback: subprocess.Popen[str] | None = None
                try:
                    fallback = _start(fallback_command, plan.runtime_dir, fallback_out, fallback_err)
                    fallback_ready = _wait_http(f"http://127.0.0.1:{fallback_port}/health", 90)
                    report["fallback"] = {
                        "ready": fallback_ready,
                        "response": _chat(
                            f"http://127.0.0.1:{fallback_port}/v1/chat/completions", plan.max_tokens
                        )
                        if fallback_ready
                        else {"ok": False, "error": "fallback_not_ready"},
                        "process": _process_metrics(fallback),
                        "command": _command_display(fallback_command),
                    }
                finally:
                    _stop(fallback)
                    fallback_out.close()
                    fallback_err.close()

            fallback_response = report.get("fallback", {}).get("response", {})
            distributed_response = report["distributed_response"]
            report["output_match"] = bool(
                distributed_response.get("ok")
                and fallback_response.get("ok")
                and distributed_response.get("content", "").strip()
                == fallback_response.get("content", "").strip()
            )

            server_log = _read_log(host_err.name)
            worker_log = _read_log(worker_err.name)
            report["backend_evidence"] = parse_backend_evidence(
                server_log,
                worker_log,
                worker_budget_mib=plan.worker_budget_mib,
            )
            report["status"] = (
                "passed"
                if report["distributed_response"].get("ok")
                and report["backend_evidence"]["partial_backend_residency_proven"]
                and report["backend_evidence"]["capacity_merge_proven"]
                and report.get("disconnect_detected", False)
                and (not check_fallback or report.get("output_match", False))
                and (
                    not check_fallback
                    or report.get("fallback", {}).get("response", {}).get("ok", False)
                )
                else "failed_acceptance"
            )
        finally:
            _stop(host)
            _stop(worker)
            for handle in (worker_out, worker_err, host_out, host_err):
                handle.close()
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="only validate assets and print the command plan")
    mode.add_argument("--run", action="store_true", help="start the native loopback RPC probe")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="GGUF model path")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME), help="llama.cpp executable directory")
    parser.add_argument("--rpc-port", type=int, default=50162)
    parser.add_argument("--http-port", type=int, default=18090)
    parser.add_argument("--gpu-layers", type=int, default=12)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--worker-budget-mib", type=float, default=2048)
    parser.add_argument("--no-fallback", action="store_true", help="skip the local CPU fallback probe")
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a short summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    plan = build_plan(
        args.model,
        args.runtime_dir,
        rpc_port=args.rpc_port,
        http_port=args.http_port,
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size,
        max_tokens=args.max_tokens,
        worker_budget_mib=args.worker_budget_mib,
    )
    try:
        report = plan_report(plan) if not args.run else run_probe(plan, check_fallback=not args.no_fallback)
    except (FileNotFoundError, OSError, ValueError) as exc:
        report = {"status": "invalid", "error": str(exc)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    if args.json or args.dry_run or not args.run:
        print(encoded)
    else:
        print(f"status={report.get('status')} partial_backend_residency={report.get('backend_evidence', {}).get('partial_backend_residency_proven')}")
    return 0 if report.get("status") in {"dry_run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
