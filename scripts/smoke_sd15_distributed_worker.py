"""Run a real Diffusers SD 1.5 Stage over the v3 TCP and HTTP bridge.

The default mode owns a coordinator and launches an isolated worker process.
The worker loads an installed local artifact, executes one image_generate Stage,
and publishes the PNG through the authenticated diffusion data plane.  Hub
access is forced offline for the child process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_ARTIFACT_ID = "sd15_original_v1"
DEFAULT_MODEL_PATH = ROOT / "models" / "sd15-original-v1"
DEFAULT_OUTPUT_DIR = ROOT / "build" / "sd15-distributed-worker-smoke"
DEFAULT_NODE_ID = "sd15_real_worker"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DistributedSD15SmokeError(RuntimeError):
    """A bounded, user-facing failure in the distributed SD smoke gate."""


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _bounded_steps(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 50:
        raise argparse.ArgumentTypeError("steps must be between 1 and 50")
    return parsed


def _bounded_dimension(value: str) -> int:
    parsed = int(value)
    if parsed < 64 or parsed > 768 or parsed % 8:
        raise argparse.ArgumentTypeError(
            "image dimensions must be multiples of 8 between 64 and 768"
        )
    return parsed


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributedSD15SmokeError(
            f"could not read JSON file: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise DistributedSD15SmokeError(f"JSON root must be an object: {path.name}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_report_safe(
    report: Mapping[str, Any],
    *,
    model_path: Path,
    cluster_secret: str,
) -> None:
    resolved_model = str(model_path.resolve()).casefold()

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in {"grant", "lease_id"}:
                    raise DistributedSD15SmokeError(
                        "smoke report contains a forbidden local path or credential"
                    )
                inspect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
            return
        if isinstance(value, str) and (
            (resolved_model and resolved_model in value.casefold())
            or (cluster_secret and cluster_secret in value)
        ):
            raise DistributedSD15SmokeError(
                "smoke report contains a forbidden local path or credential"
            )

    inspect(report)


def _tail(path: Path, *, max_chars: int = 4000) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return value[-max_chars:]


def _wait_for_file(
    path: Path,
    *,
    timeout_seconds: float,
    process: subprocess.Popen[Any] | None = None,
    worker_log: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process is not None and process.poll() is not None:
            detail = _tail(worker_log) if worker_log is not None else ""
            raise DistributedSD15SmokeError(
                f"worker exited before becoming ready (exit={process.returncode})"
                + (f":\n{detail}" if detail else "")
            )
        time.sleep(0.1)
    raise DistributedSD15SmokeError(f"timed out waiting for {path.name}")


def _wait_for_http(url: str, *, timeout_seconds: float = 10.0) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise DistributedSD15SmokeError("worker data-plane HTTP server did not start")


def _offline_worker_environment(cluster_secret: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DIFFUSERS_VERBOSITY": "error",
        "QLH_CLUSTER_SECRET": cluster_secret,
        "QLH_DIFFUSION_WORKER_EXPERIMENTAL_ENABLED": "true",
        "PYTHONUTF8": "1",
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
    })
    return env


def _artifact_manifest_digest(model_path: Path, artifact_id: str) -> str:
    manifest_path = model_path / ".qlh-sd-asset.json"
    manifest = _read_json(manifest_path)
    asset = manifest.get("asset")
    if not isinstance(asset, Mapping) or asset.get("artifact_id") != artifact_id:
        raise DistributedSD15SmokeError(
            "installed SD asset manifest does not match the requested artifact"
        )
    digest = str(manifest.get("artifact_sha256", "")).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DistributedSD15SmokeError(
            "installed SD asset manifest has no valid artifact SHA-256"
        )
    return digest


def _ensure_worker_artifact(service: Any, model_path: Path, artifact_id: str) -> None:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise DistributedSD15SmokeError("SD model path is not a local directory")
    digest = _artifact_manifest_digest(resolved, artifact_id)
    for registered in service.list_artifacts(include_path=True):
        if registered.get("artifact_id") != artifact_id:
            continue
        artifact = registered.get("artifact") or {}
        registered_path = Path(str(artifact.get("path", ""))).resolve()
        if registered_path != resolved:
            raise DistributedSD15SmokeError(
                "artifact ID is already registered to a different local path"
            )
        if str(artifact.get("sha256", "")).lower() != digest:
            raise DistributedSD15SmokeError(
                "registered artifact identity does not match its installed manifest"
            )
        return
    service.register_artifact(
        str(resolved),
        artifact_id=artifact_id,
        compute_hash=False,
        _trusted_sha256=digest,
    )


def _worker_versions() -> dict[str, Any]:
    import diffusers
    import torch
    import transformers

    device_name = ""
    total_memory = 0
    if torch.cuda.is_available():
        device_name = str(torch.cuda.get_device_name(0))
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "diffusers": str(diffusers.__version__),
        "transformers": str(transformers.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": device_name,
        "cuda_total_memory_bytes": total_memory,
    }


def _run_worker(args: argparse.Namespace) -> int:
    import uvicorn
    from fastapi import FastAPI

    import scheduler as scheduler_module
    from diffusion.data_plane import DiffusionDataPlaneRuntime, router
    from diffusion.service import DiffusionService, build_sd15_engine_config
    from diffusion.worker_runtime import DiffusionWorkerRuntime
    from scheduler import Scheduler
    from tcp_comm import TCPClient

    required = {
        "state_dir": args.state_dir,
        "ready_file": args.ready_file,
        "completion_file": args.completion_file,
        "tcp_port": args.tcp_port,
        "http_port": args.http_port,
    }
    missing = [name for name, value in required.items() if value in {None, ""}]
    if missing:
        raise DistributedSD15SmokeError(
            f"worker mode is missing arguments: {', '.join(sorted(missing))}"
        )
    cluster_secret = os.environ.get("QLH_CLUSTER_SECRET", "")
    if len(cluster_secret.encode("utf-8")) < 32:
        raise DistributedSD15SmokeError("worker cluster secret is missing or too short")

    scheduler_module.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED = True
    state_dir = Path(args.state_dir).resolve()
    ready_file = Path(args.ready_file).resolve()
    completion_file = Path(args.completion_file).resolve()
    service = DiffusionService()
    data_plane = None
    http_server = None
    http_thread = None
    scheduler = None
    client = None
    try:
        _ensure_worker_artifact(service, Path(args.model_path), args.artifact_id)
        config = build_sd15_engine_config(args.profile)
        service.load(args.artifact_id, config)
        versions = _worker_versions()
        if not versions["cuda_available"]:
            raise DistributedSD15SmokeError("real SD worker requires CUDA")

        data_plane = DiffusionDataPlaneRuntime.create(
            state_dir=state_dir,
            cluster_secret=cluster_secret,
        )
        http_app = FastAPI()
        http_app.state.diffusion_data_plane = data_plane
        http_app.include_router(router)
        http_server = uvicorn.Server(uvicorn.Config(
            http_app,
            host="127.0.0.1",
            port=int(args.http_port),
            log_level="warning",
        ))
        http_thread = threading.Thread(target=http_server.run, daemon=True)
        http_thread.start()
        _wait_for_http(
            f"http://127.0.0.1:{args.http_port}"
            "/internal/v1/diffusion/data-plane/status",
        )

        runtime = DiffusionWorkerRuntime(
            service=service,
            data_plane=data_plane,
            node_id=args.node_id,
            data_plane_base_url=f"http://127.0.0.1:{args.http_port}",
        )
        capabilities = runtime.capabilities()
        if capabilities is None:
            raise DistributedSD15SmokeError(
                "loaded SD artifact did not produce worker capabilities"
            )
        scheduler = Scheduler()
        scheduler._role_override = "client"
        scheduler.get_effective_node_id = lambda: args.node_id
        if not scheduler.configure_diffusion_worker(
            capabilities=capabilities,
            executor=runtime.execute,
        ) and scheduler._diffusion_worker_adapter is None:
            raise DistributedSD15SmokeError("could not install diffusion worker adapter")

        client = TCPClient(
            server_host="127.0.0.1",
            server_port=int(args.tcp_port),
            client_id=args.node_id,
            role="client",
            node_type="pc",
        )
        scheduler._tcp_client = client
        if not client.connect(
            on_message=lambda outer: scheduler._on_tcp_message("master", outer),
        ):
            raise DistributedSD15SmokeError("worker could not connect to coordinator TCP")
        if not scheduler._send_diffusion_worker_hello(client):
            raise DistributedSD15SmokeError("worker could not send v3 hello")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = scheduler.get_diffusion_worker_protocol_status()
            if status.get("control_plane_connected"):
                break
            time.sleep(0.05)
        else:
            raise DistributedSD15SmokeError("worker v3 hello was not acknowledged")

        _write_json(ready_file, {
            "schema_version": 1,
            "node_id": args.node_id,
            "artifact_manifest": runtime.artifact_manifest(),
            "versions": versions,
            "profile": args.profile,
            "offline": True,
        })
        deadline = time.monotonic() + float(args.timeout_seconds)
        while time.monotonic() < deadline:
            if completion_file.is_file():
                return 0
            time.sleep(0.1)
        raise DistributedSD15SmokeError("coordinator did not finish before timeout")
    finally:
        if client is not None:
            client.disconnect()
        if scheduler is not None:
            scheduler.clear_diffusion_worker()
        if http_server is not None:
            http_server.should_exit = True
        if http_thread is not None:
            http_thread.join(timeout=5.0)
        if data_plane is not None:
            data_plane.close()
        service.close()


def _worker_command(
    args: argparse.Namespace,
    *,
    tcp_port: int,
    http_port: int,
    worker_state: Path,
    ready_file: Path,
    completion_file: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--tcp-port", str(tcp_port),
        "--http-port", str(http_port),
        "--state-dir", str(worker_state),
        "--ready-file", str(ready_file),
        "--completion-file", str(completion_file),
        "--node-id", args.node_id,
        "--artifact-id", args.artifact_id,
        "--model-path", str(Path(args.model_path).resolve()),
        "--profile", args.profile,
        "--timeout-seconds", str(args.timeout_seconds),
    ]


def _run_coordinator(args: argparse.Namespace) -> dict[str, Any]:
    import config as config_module
    import scheduler as scheduler_module
    from diffusion.coordinator_runtime import DiffusionCoordinatorRuntime
    from diffusion.data_plane import DiffusionDataPlaneRuntime
    from diffusion.presets import get_preset
    from scheduler import NodeInfo, NodeRole, Scheduler
    from task_graph import StageSpec, TaskGraphCoordinator
    from tcp_comm import TCPServer

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_log = output_dir / "worker.log"
    output_png = output_dir / "distributed-output.png"
    report_path = output_dir / "smoke-result.json"
    for stale_result in (output_png, report_path):
        try:
            stale_result.unlink()
        except FileNotFoundError:
            pass
    cluster_secret = secrets.token_urlsafe(32)
    previous_cluster_secret = config_module.CLUSTER_SECRET
    config_module.CLUSTER_SECRET = cluster_secret
    tcp_port = _free_port()
    http_port = _free_port()
    workflow_id = f"wf_sdsmoke_{uuid.uuid4().hex[:16]}"
    started_at = time.time()
    started_monotonic = time.monotonic()

    scheduler_module.RUN_MODE = "distributed"
    scheduler_module.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED = True
    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.get_effective_node_id = lambda: "sd15_smoke_master"
    scheduler.init_nodes()
    scheduler.nodes[args.node_id] = NodeInfo(
        node_id=args.node_id,
        role=NodeRole.CLIENT,
        node_type="pc",
    )
    server = TCPServer(host="127.0.0.1", port=tcp_port)
    scheduler._tcp_server = server
    coordinator = TaskGraphCoordinator(max_records=10)
    process = None

    with tempfile.TemporaryDirectory(
        prefix="sd15-distributed-",
        dir=str(output_dir),
    ) as temporary:
        work_dir = Path(temporary)
        worker_state = work_dir / "worker-state"
        coordinator_state = work_dir / "coordinator-state"
        ready_file = work_dir / "worker-ready.json"
        completion_file = work_dir / "coordinator-complete"
        worker_state.mkdir()
        coordinator_state.mkdir()
        data_plane = None
        try:
            data_plane = DiffusionDataPlaneRuntime.create(
                state_dir=coordinator_state,
                cluster_secret=cluster_secret,
            )
            coordinator_runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
            if not scheduler.configure_diffusion_coordinator(
                result_ingestor=coordinator_runtime.ingest_result,
                dispatch_enabled=True,
            ):
                raise DistributedSD15SmokeError(
                    "could not configure the diffusion coordinator"
                )
            server.start(
                on_message=scheduler._on_tcp_message,
                on_disconnect=scheduler._on_tcp_disconnect,
            )
            with worker_log.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    _worker_command(
                        args,
                        tcp_port=tcp_port,
                        http_port=http_port,
                        worker_state=worker_state,
                        ready_file=ready_file,
                        completion_file=completion_file,
                    ),
                    cwd=str(ROOT),
                    env=_offline_worker_environment(cluster_secret),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                _wait_for_file(
                    ready_file,
                    timeout_seconds=float(args.load_timeout_seconds),
                    process=process,
                    worker_log=worker_log,
                )
                worker_ready = _read_json(ready_file)
                deadline = time.monotonic() + 15.0
                provider = None
                while time.monotonic() < deadline:
                    providers = scheduler.remote_diffusion_providers()
                    if providers:
                        provider = providers[0]
                        break
                    if process.poll() is not None:
                        raise DistributedSD15SmokeError(
                            "worker exited before Provider admission:\n"
                            + _tail(worker_log)
                        )
                    time.sleep(0.05)
                if provider is None:
                    raise DistributedSD15SmokeError(
                        "worker v3 hello did not produce a remote Provider"
                    )
                manifests = provider.artifact_manifests()
                if len(manifests) != 1:
                    raise DistributedSD15SmokeError(
                        "worker must advertise exactly one SD artifact manifest"
                    )
                manifest = manifests[0]
                if manifest != worker_ready.get("artifact_manifest"):
                    raise DistributedSD15SmokeError(
                        "coordinator and worker disagree on artifact identity"
                    )
                coordinator.register_provider(provider)
                preset = get_preset(args.preset)
                root_input = {
                    "prompt": preset.prompt,
                    "negative_prompt": preset.negative_prompt,
                    "seed": int(args.seed),
                    "width": int(args.width),
                    "height": int(args.height),
                    "steps": int(args.steps),
                    "guidance_scale": float(preset.guidance_scale),
                    "scheduler": str(preset.scheduler),
                    "artifact_manifest_sha256": manifest["sha256"],
                }
                output, workflow = coordinator.run(
                    stages=[StageSpec(
                        "image_generate",
                        "image_generate",
                        provider=provider.provider_id,
                        lease_timeout_seconds=float(args.timeout_seconds),
                    )],
                    final_stage_id="image_generate",
                    root_input=root_input,
                    runtime_context={"diffusion_artifact_manifest": manifest},
                    workflow_id=workflow_id,
                )
                descriptor = output.get("image")
                metrics = output.get("metrics")
                if not isinstance(descriptor, Mapping) or not isinstance(metrics, Mapping):
                    raise DistributedSD15SmokeError(
                        "completed Stage did not return image metrics"
                    )
                coordinator.commit_result(workflow["workflow_id"])
                committed_workflow = coordinator.get(workflow["workflow_id"])
                local_descriptor, image_data = coordinator_runtime.read_result(
                    workflow_id=workflow_id,
                    blob_id=str(descriptor["blob_id"]),
                )
                if not image_data.startswith(PNG_SIGNATURE):
                    raise DistributedSD15SmokeError("coordinator CAS output is not PNG")
                output_png.write_bytes(image_data)
                completion_file.write_text("complete\n", encoding="ascii")
                worker_exit = process.wait(timeout=30.0)
                if worker_exit != 0:
                    raise DistributedSD15SmokeError(
                        f"worker cleanup failed (exit={worker_exit}):\n"
                        + _tail(worker_log)
                    )

            elapsed = time.monotonic() - started_monotonic
            report = {
                "schema_version": 1,
                "status": "passed",
                "real_diffusers_worker": True,
                "offline": True,
                "transport": {
                    "control": "tcp_length_prefixed_v3",
                    "data": "authenticated_http_chunked",
                    "worker_process_isolated": True,
                },
                "workflow": {
                    "workflow_id": workflow_id,
                    "state": committed_workflow.get("state"),
                    "provider_id": provider.provider_id,
                    "worker_node_id": provider.node_id,
                    "distributed": True,
                },
                "artifact": {
                    "artifact_id": manifest.get("artifact_id"),
                    "manifest_sha256": manifest.get("sha256"),
                    "component_sha256": manifest.get("components", [{}])[0].get(
                        "sha256"
                    ),
                },
                "request": {
                    "preset_id": args.preset,
                    "seed": int(args.seed),
                    "width": int(args.width),
                    "height": int(args.height),
                    "steps": int(args.steps),
                    "scheduler": str(preset.scheduler),
                    "profile": args.profile,
                },
                "result": {
                    "output_file": output_png.name,
                    "sha256": _sha256_bytes(image_data),
                    "size_bytes": len(image_data),
                    "content_type": local_descriptor.get("content_type"),
                    "width": local_descriptor.get("width"),
                    "height": local_descriptor.get("height"),
                    "metrics": dict(metrics),
                    "wall_seconds": round(elapsed, 3),
                },
                "worker_environment": worker_ready.get("versions", {}),
                "started_at_unix": started_at,
            }
            _assert_report_safe(
                report,
                model_path=Path(args.model_path),
                cluster_secret=cluster_secret,
            )
            _write_json(report_path, report)
            return report
        finally:
            try:
                completion_file.write_text("stop\n", encoding="ascii")
            except OSError:
                pass
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
            coordinator.close()
            scheduler.clear_diffusion_coordinator()
            if data_plane is not None:
                data_plane.close()
            server.stop()
            config_module.CLUSTER_SECRET = previous_cluster_secret


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real SD 1.5 image_generate Stage through the isolated "
            "v3 TCP and authenticated HTTP worker bridge"
        ),
    )
    parser.add_argument("--artifact-id", default=DEFAULT_ARTIFACT_ID)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--preset", default="sd15_original_v1")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--seed", type=int, default=19950101)
    parser.add_argument("--steps", type=_bounded_steps, default=4)
    parser.add_argument("--width", type=_bounded_dimension, default=512)
    parser.add_argument("--height", type=_bounded_dimension, default=512)
    parser.add_argument("--node-id", default=DEFAULT_NODE_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--load-timeout-seconds",
        type=_positive_float,
        default=180.0,
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=180.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tcp-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--http-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--state-dir", help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", help=argparse.SUPPRESS)
    parser.add_argument("--completion-file", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker:
            return _run_worker(args)
        report = _run_coordinator(args)
    except Exception as exc:
        if args.worker:
            print(f"worker_error: {exc}", file=sys.stderr)
        else:
            payload = {
                "schema_version": 1,
                "status": "failed",
                "error": str(exc)[:1000],
            }
            print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        result = report["result"]
        print(
            "SD distributed worker smoke passed: "
            f"{result['output_file']} ({result['size_bytes']} bytes, "
            f"{result['wall_seconds']}s wall)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
