"""QLH defense demo launcher.

The default fixture mode starts only the product frontend and is safe to run
without a model, network, or worker. Live mode must be explicitly selected and
starts the local API plus the frontend on loopback addresses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demo_ownership import OwnershipLedger


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = ROOT / "frontend_cybergothic"
BUILD_ROOT = ROOT / "build" / "defense-demo"
DEFAULT_REPORT = BUILD_ROOT / "latest.json"
SCENARIO_MANIFEST = Path(__file__).with_name("scenarios.json")
FAILURE_WORKER = Path(__file__).with_name("failure_worker.py")
FAILURE_WORKER_EXIT_CODE = 23
STARTUP_BUDGET_SECONDS = 120.0
SCENARIO_SOURCE_PATHS = {
    "fixture-data": Path("src/data/fixtures.ts"),
    "chat-pane": Path("src/components/ChatPane.tsx"),
    "image-studio": Path("src/pages/ImageStudioPage.tsx"),
    "topology-snapshot": Path("src/data/defense-topology.json"),
    "cluster-page": Path("src/components/DefenseTopologySnapshot.tsx"),
}

_TOPOLOGY_ALIAS = re.compile(r"^node-[a-z0-9]+(?:-[a-z0-9]+)*$")
_IPV4_TEXT = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


@dataclass(frozen=True)
class DemoConfig:
    mode: str = "fixtures"
    scenario: str | None = None
    api_port: int = 8000
    frontend_port: int = 5174
    startup_timeout: float = 30.0
    duration: float = 0.0
    open_browser: bool = False
    skip_frontend: bool = False
    report_path: Path = DEFAULT_REPORT


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    log_path: Path


@dataclass
class DemoRun:
    config: DemoConfig
    frontend_url: str | None = None
    processes: list[ManagedProcess] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    log_paths: list[Path] = field(default_factory=list)
    failure_injection: dict[str, Any] | None = None
    ownership: OwnershipLedger | None = None

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"name": name, "ok": ok, "detail": detail})

    def stop(self) -> None:
        try:
            for managed in reversed(self.processes):
                process = managed.process
                if process.poll() is not None:
                    continue
                if os.name == "nt":
                    # npm/python on Windows can be launcher shims. Kill the tree
                    # so a child server cannot survive the demo process.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    continue
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
        finally:
            self.processes.clear()
            if self.ownership is not None:
                self.ownership.close()


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1 到 65535 之间")
    return port


def _python_command() -> str:
    executable = "python.exe" if os.name == "nt" else "python"
    candidates = (
        ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv-test" / ("Scripts" if os.name == "nt" else "bin") / executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _live_python_command() -> str:
    executable = "python.exe" if os.name == "nt" else "python"
    candidates = (
        ROOT / ".venv-packaging-cuda" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv-test" / ("Scripts" if os.name == "nt" else "bin") / executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _loopback_url(port: int, path: str = "") -> str:
    return f"http://127.0.0.1:{port}{path}"


def _cluster_port(config: DemoConfig) -> int:
    port = config.api_port + 1
    if port == config.frontend_port:
        port += 1
    return port


def _public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _http_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def _http_ready(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 400:
            raise RuntimeError(f"HTTP 状态异常：{response.status}")


def _scenario_sources() -> dict[str, Path]:
    return {source_id: FRONTEND_ROOT / relative_path for source_id, relative_path in SCENARIO_SOURCE_PATHS.items()}


def _load_fixture_scenarios() -> list[dict[str, Any]]:
    """Load and validate the offline dialog/image rehearsal contract."""
    try:
        payload = json.loads(SCENARIO_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"预置演示剧本不可读取：{type(exc).__name__}") from exc
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError("预置演示剧本为空")
    for item in scenarios:
        if not isinstance(item, dict) or item.get("kind") not in {"dialog", "image", "topology"}:
            raise RuntimeError("预置演示剧本包含未知场景")
        if not isinstance(item.get("route"), str) or not item["route"].startswith("#/"):
            raise RuntimeError("预置演示剧本路由无效")
        markers = item.get("markers")
        if not isinstance(markers, list) or not markers:
            raise RuntimeError("预置演示剧本缺少固定成功标记")
        for marker in markers:
            if not isinstance(marker, dict) or marker.get("source") not in SCENARIO_SOURCE_PATHS:
                raise RuntimeError("预置演示剧本标记来源无效")
            if not isinstance(marker.get("value"), str) or not marker["value"]:
                raise RuntimeError("预置演示剧本标记无效")
    return scenarios


def _validate_topology_snapshot(payload: object) -> dict[str, Any]:
    """Fail closed when the DEF-P4 fixture stops being redacted or contiguous."""
    if not isinstance(payload, dict) or payload.get("schema") != "qlh.defense_topology.v1":
        raise RuntimeError("拓扑快照 schema 无效")
    if payload.get("source") != "fixture":
        raise RuntimeError("拓扑快照不得冒充实时来源")

    guard = payload.get("claim_guard")
    required_guard = {
        "redacted": True,
        "live_cluster": False,
        "real_model_loaded": False,
        "physical_dual_host": False,
    }
    if not isinstance(guard, dict) or any(guard.get(key) is not value for key, value in required_guard.items()):
        raise RuntimeError("拓扑快照声明边界无效")

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if _IPV4_TEXT.search(serialized):
        raise RuntimeError("拓扑快照包含未脱敏 IPv4 地址")

    readiness = payload.get("readiness")
    checks = readiness.get("checks") if isinstance(readiness, dict) else None
    if readiness is None or readiness.get("state") != "ready" or not isinstance(checks, list) or not checks:
        raise RuntimeError("拓扑快照 readiness 无效")
    if any(not isinstance(item, dict) or item.get("state") != "ready" for item in checks):
        raise RuntimeError("拓扑快照包含未通过的 readiness 检查")

    nodes = payload.get("nodes")
    allowed_node_keys = {"alias", "role", "state", "heartbeat", "summary"}
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise RuntimeError("拓扑快照节点不足")
    aliases: set[str] = set()
    node_states: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or set(node) != allowed_node_keys:
            raise RuntimeError("拓扑快照节点字段未通过脱敏白名单")
        alias = node.get("alias")
        state = node.get("state")
        if not isinstance(alias, str) or not _TOPOLOGY_ALIAS.fullmatch(alias) or alias in aliases:
            raise RuntimeError("拓扑快照节点别名无效")
        if state not in {"ready", "degraded", "offline"}:
            raise RuntimeError("拓扑快照节点状态无效")
        aliases.add(alias)
        node_states[alias] = state

    layer_plan = payload.get("layer_plan")
    assignments = layer_plan.get("assignments") if isinstance(layer_plan, dict) else None
    total_layers = layer_plan.get("total_layers") if isinstance(layer_plan, dict) else None
    if isinstance(total_layers, bool) or not isinstance(total_layers, int) or total_layers < 1:
        raise RuntimeError("拓扑快照总层数无效")
    if not isinstance(assignments, list) or not assignments:
        raise RuntimeError("拓扑快照缺少层段分配")

    allowed_assignment_keys = {
        "node_alias", "start_layer", "end_layer_exclusive", "layer_count", "capabilities",
    }
    cursor = 0
    assigned_aliases: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != allowed_assignment_keys:
            raise RuntimeError("拓扑快照层段字段无效")
        alias = assignment.get("node_alias")
        start = assignment.get("start_layer")
        end = assignment.get("end_layer_exclusive")
        count = assignment.get("layer_count")
        if alias not in aliases or node_states.get(str(alias)) != "ready" or alias in assigned_aliases:
            raise RuntimeError("拓扑快照层段节点无效")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, count)):
            raise RuntimeError("拓扑快照层段边界无效")
        if start != cursor or end <= start or count != end - start or end > total_layers:
            raise RuntimeError("拓扑快照层段必须连续、无重叠并匹配层数")
        capabilities = assignment.get("capabilities")
        if not isinstance(capabilities, list) or any(not isinstance(item, str) or not item for item in capabilities):
            raise RuntimeError("拓扑快照层段能力无效")
        assigned_aliases.add(str(alias))
        cursor = end
    if cursor != total_layers:
        raise RuntimeError("拓扑快照层段未覆盖完整模型投影")

    return {
        "snapshot_id": str(payload.get("snapshot_id", "unknown")),
        "node_count": len(nodes),
        "ready_node_count": sum(state == "ready" for state in node_states.values()),
        "readiness_check_count": len(checks),
        "total_layers": total_layers,
    }


def verify_fixture_scenarios(run: DemoRun) -> list[dict[str, Any]]:
    """Verify fixture content without invoking a model or a write API."""
    scenarios = _load_fixture_scenarios()
    sources: dict[str, str] = {}
    for source_id, source_path in _scenario_sources().items():
        try:
            sources[source_id] = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"演示数据源不可读取：{source_id}:{type(exc).__name__}") from exc
    for item in scenarios:
        missing = [
            marker["value"]
            for marker in item["markers"]
            if marker["value"] not in sources[marker["source"]]
        ]
        if missing:
            raise RuntimeError(f"预置{item['kind']}剧本标记缺失：{','.join(missing)}")
        if item["kind"] == "topology":
            try:
                topology = json.loads(sources["topology-snapshot"])
            except ValueError as exc:
                raise RuntimeError("拓扑快照 JSON 无效") from exc
            summary = _validate_topology_snapshot(topology)
            detail = (
                f"route={item['route']}; snapshot={summary['snapshot_id']}; "
                f"nodes={summary['node_count']}; layers={summary['total_layers']}; "
                "claim=fixture/redacted/not-live"
            )
        else:
            detail = f"route={item['route']}; marker={item['markers'][0]['value']}"
        run.record(
            f"fixture-{item['kind']}",
            True,
            detail,
        )
    return scenarios


def wait_for_health(url: str, timeout: float, *, sleep_seconds: float = 0.25) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "unavailable"
    while time.monotonic() < deadline:
        try:
            payload = _http_json(url, min(2.0, max(0.2, deadline - time.monotonic())))
            if payload.get("status") == "ok":
                return payload
            last_error = "health response status is not ok"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(sleep_seconds)
    raise RuntimeError(f"后端健康检查超时：{last_error}")


def wait_for_ready(url: str, timeout: float, *, sleep_seconds: float = 0.25) -> None:
    deadline = time.monotonic() + timeout
    last_error = "unavailable"
    while time.monotonic() < deadline:
        try:
            _http_ready(url, min(2.0, max(0.2, deadline - time.monotonic())))
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(sleep_seconds)
    raise RuntimeError(f"前端就绪检查超时：{last_error}")


def _start_process(
    run: DemoRun,
    name: str,
    command: list[str],
    env: dict[str, str],
    log_name: str,
    *,
    cwd: Path | None = None,
) -> ManagedProcess:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = BUILD_ROOT / log_name
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd or (ROOT if name == "backend" else FRONTEND_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except Exception:
        handle.close()
        raise
    handle.close()
    managed = ManagedProcess(name=name, process=process, log_path=log_path)
    run.processes.append(managed)
    run.log_paths.append(log_path)
    if run.ownership is not None:
        try:
            run.ownership.register(name, process.pid)
        except Exception:
            if process.poll() is None:
                process.terminate()
            raise
    return managed


def _startup_timeout(config: DemoConfig, started_at: float) -> float:
    remaining = STARTUP_BUDGET_SECONDS - (time.monotonic() - started_at)
    if remaining <= 0:
        raise RuntimeError("启动超过 2 分钟门")
    return min(config.startup_timeout, remaining)


def _summarize_failure_workflow(snapshot: dict[str, Any], worker_exit_code: int) -> dict[str, Any]:
    """Build the public DEF-P2 evidence without retaining inputs or outputs."""
    stages = snapshot.get("stages", [])
    stage = stages[0] if isinstance(stages, list) and stages else {}
    attempts = stage.get("attempts", []) if isinstance(stage, dict) else []
    public_attempts = [
        {
            "provider_id": str(attempt.get("provider", "")),
            "provider_kind": str(attempt.get("provider_kind", "")),
            "node_id": str(attempt.get("provider_node_id", "")),
            "state": str(attempt.get("state", "")),
            "lease_epoch": int(attempt.get("lease_epoch", 0) or 0),
        }
        for attempt in attempts
        if isinstance(attempt, dict)
    ]
    return {
        "schema": "qlh.defense_failure.v1",
        "execution_environment": {
            "kind": "loopback_dual_process",
            "process_count": 2,
            "transport": "tcp_loopback",
            "external_network": False,
            "real_model_loaded": False,
            "physical_nodes": False,
        },
        "injection": {
            "kind": "worker_exit_after_stage_accept",
            "worker_exit_code": int(worker_exit_code),
        },
        "workflow": {
            "state": str(snapshot.get("state", "")),
            "stage_count": int(snapshot.get("stage_count", 0) or 0),
            "attempt_count": int(snapshot.get("attempt_count", 0) or 0),
            "retry_count": int(snapshot.get("retry_count", 0) or 0),
            "stage": {
                "state": str(stage.get("state", "")),
                "requested_provider_id": str(stage.get("requested_provider", "")),
                "selected_provider_id": str(stage.get("selected_provider", "")),
                "last_retry_error_code": str(stage.get("last_retry_error_code", "")),
                "output_available": bool(stage.get("output_available", False)),
                "attempts": public_attempts,
            },
        },
    }


def _validate_failure_evidence(evidence: dict[str, Any]) -> None:
    workflow = evidence["workflow"]
    stage = workflow["stage"]
    attempts = stage["attempts"]
    providers = [item["provider_id"] for item in attempts]
    if evidence["injection"]["worker_exit_code"] != FAILURE_WORKER_EXIT_CODE:
        raise RuntimeError("故障 worker 未按固定退出码退出")
    if workflow["state"] != "completed" or stage["state"] != "completed":
        raise RuntimeError("故障注入工作流未完成")
    if workflow["retry_count"] != 1 or len(attempts) != 2:
        raise RuntimeError("故障注入未形成一次且仅一次重派")
    if stage["last_retry_error_code"] != "remote_worker_disconnected":
        raise RuntimeError("故障注入未记录 worker 断连原因")
    if providers != ["remote_demo-worker-primary", "demo-local-recovery"]:
        raise RuntimeError("故障注入 provider 重派顺序异常")
    if stage["selected_provider_id"] != "demo-local-recovery" or not stage["output_available"]:
        raise RuntimeError("故障注入恢复 provider 未产生完成结果")


def run_failure_injection(run: DemoRun, timeout: float) -> None:
    """Run one real loopback worker exit and recover through TaskGraph fallback."""
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import config as runtime_config
    from task_graph import StageSpec, TaskGraphCoordinator
    from task_provider import DeterministicFakeProvider, ModelIdentity, ProviderRegistry
    from task_worker_adapter import RemoteFullWorkerProvider, TaskWorkerControlPlane
    from tcp_comm import MessageType, TCPServer

    secret = secrets.token_hex(24)
    previous_secret = runtime_config.CLUSTER_SECRET
    runtime_config.CLUSTER_SECRET = secret
    worker_id = "demo-worker-primary"
    control = TaskWorkerControlPlane(health_timeout_seconds=max(5.0, timeout))
    hello_received = threading.Event()
    disconnected = threading.Event()
    provider_holder: dict[str, Any] = {}
    server = TCPServer(host="127.0.0.1", port=0)
    coordinator = None

    def on_server_message(client_id: str, outer: dict[str, Any]) -> None:
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer.get("data", {})
        if inner.get("message_type") == "hello":
            acknowledgement = control.receive_on_coordinator(
                client_id,
                inner,
                coordinator_node_id="demo-master",
            )
            server.send_to_client(client_id, acknowledgement.snapshot(), MessageType.TASK_WORKER)
            hello_received.set()
            return
        provider_holder["provider"].handle_message(inner)

    def on_disconnect(client_id: str) -> None:
        control.disconnect_worker(client_id)
        provider = provider_holder.get("provider")
        if provider is not None:
            provider.notify_disconnect()
        disconnected.set()

    remote_provider = RemoteFullWorkerProvider(
        node_id=worker_id,
        peer_snapshot=lambda: control.worker_snapshot(worker_id),
        send_message=lambda message: server.send_to_client(
            worker_id,
            message.snapshot(),
            MessageType.TASK_WORKER,
        ),
    )
    provider_holder["provider"] = remote_provider
    fallback_provider = DeterministicFakeProvider(
        "demo-local-recovery",
        node_id="demo-master",
        supported_stage_types=("full_inference",),
        output_factory=lambda _request, _cancel: {"content": "fixture-recovered"},
    )
    identity = ModelIdentity(
        model_id="defense-fixture",
        engine="pytorch",
        format="safetensors",
        revision="defense-fixture-v1",
        sha256="d" * 64,
    )

    try:
        server.start(on_message=on_server_message, on_disconnect=on_disconnect)
        run.record("start-coordinator", True, "transport=tcp-loopback; process=coordinator")
        worker_env = os.environ.copy()
        worker_env.update({
            "HF_HUB_OFFLINE": "1",
            "PYTHONUTF8": "1",
            "QLH_CLUSTER_SECRET": secret,
            "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "true",
            "TRANSFORMERS_OFFLINE": "1",
        })
        worker = _start_process(
            run,
            "failure-worker",
            [
                _python_command(),
                str(FAILURE_WORKER),
                "--port",
                str(server.port),
                "--node-id",
                worker_id,
                "--model-id",
                identity.model_id,
                "--model-sha256",
                identity.sha256,
            ],
            worker_env,
            "failure-worker.log",
            cwd=ROOT,
        )
        if not hello_received.wait(timeout):
            raise RuntimeError("故障 worker 协议握手超时")
        run.record("worker-ready", True, "node=demo-worker-primary; protocol=v2")

        registry = ProviderRegistry()
        registry.register(remote_provider)
        registry.register(fallback_provider)
        coordinator = TaskGraphCoordinator(provider_registry=registry)
        output, snapshot = coordinator.run(
            stages=[StageSpec(
                "answer",
                "full_inference",
                provider=remote_provider.provider_id,
                fallback_providers=(fallback_provider.provider_id,),
                pure=True,
                lease_timeout_seconds=5.0,
            )],
            final_stage_id="answer",
            root_input={},
            model_identity=identity,
            template="defense_failure_injection_v1",
            workflow_id="wf_defensefailure01",
        )
        if output.get("content") != "fixture-recovered":
            raise RuntimeError("恢复 provider 输出标记异常")
        snapshot = coordinator.commit_result(snapshot["workflow_id"])
        try:
            exit_code = worker.process.wait(timeout=min(5.0, timeout))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("故障 worker 未退出") from exc
        if not disconnected.wait(min(2.0, timeout)):
            raise RuntimeError("未观察到故障 worker 断连")

        evidence = _summarize_failure_workflow(snapshot, exit_code)
        run.failure_injection = evidence
        _validate_failure_evidence(evidence)
        run.record("worker-exit", True, f"exit_code={exit_code}; reason=remote_worker_disconnected")
        run.record(
            "stage-reassigned",
            True,
            "from=remote_demo-worker-primary; to=demo-local-recovery; retry_count=1",
        )
        run.record("workflow-complete", True, "state=completed; attempts=2; output=redacted")
    finally:
        if coordinator is not None:
            coordinator.close()
        else:
            remote_provider.close()
            fallback_provider.close()
        server.stop()
        runtime_config.CLUSTER_SECRET = previous_secret


def _validate_layout(config: DemoConfig) -> list[str]:
    errors: list[str] = []
    if config.mode == "live" and not (ROOT / "src" / "api_server.py").is_file():
        errors.append("缺少后端入口：src/api_server.py")
    if config.mode == "failure" and not FAILURE_WORKER.is_file():
        errors.append("缺少故障注入 worker：scripts/demo/failure_worker.py")
    uses_frontend = config.mode in {"fixtures", "live"} and not config.skip_frontend
    if uses_frontend and not FRONTEND_ROOT.is_dir():
        errors.append("缺少现行前端目录：frontend_cybergothic")
    if uses_frontend and not (FRONTEND_ROOT / "package.json").is_file():
        errors.append("缺少现行前端配置：frontend_cybergothic/package.json")
    if config.mode == "live" and not shutil.which(_live_python_command()):
        errors.append("live 模式 Python 运行时不可用")
    if uses_frontend and not shutil.which(_npm_command()):
        errors.append("前端 npm 运行时不可用")
    if uses_frontend and not (FRONTEND_ROOT / "node_modules" / "vite" / "package.json").is_file():
        errors.append("现行前端依赖未安装：请在 frontend_cybergothic 目录执行 npm ci")
    if config.mode == "live" and config.api_port == config.frontend_port:
        errors.append("后端与前端端口必须不同")
    if config.scenario is not None and config.mode != "fixtures":
        errors.append("预置场景只能用于 fixtures 模式")
    if config.mode == "live" and _cluster_port(config) > 65535:
        errors.append("后端端口未为本地集群端口保留可用范围")
    return errors


def run_demo(config: DemoConfig) -> int:
    uses_frontend = config.mode in {"fixtures", "live"}
    run = DemoRun(
        config,
        frontend_url=(
            _loopback_url(config.frontend_port, "/?fixtures=1" if config.mode == "fixtures" else "")
            if uses_frontend else None
        ),
    )
    started_at = time.monotonic()
    errors = _validate_layout(config)
    if errors:
        for error in errors:
            run.record("preflight", False, error)
        _write_report(run)
        for error in errors:
            print(f"[QLH-DEMO] ERROR {error}", file=sys.stderr)
        return 2

    owned_ports: list[int] = []
    if not config.skip_frontend and uses_frontend:
        owned_ports.append(config.frontend_port)
    if config.mode == "live":
        owned_ports.extend((config.api_port, _cluster_port(config)))
    run.ownership = OwnershipLedger("defense_demo", ports=owned_ports)

    try:
        run.record("preflight", True, f"mode={config.mode}")
        if config.mode == "failure":
            run_failure_injection(run, _startup_timeout(config, started_at))
            startup_seconds = time.monotonic() - started_at
            if startup_seconds > STARTUP_BUDGET_SECONDS:
                raise RuntimeError(f"启动超过 2 分钟门：{startup_seconds:.1f}s")
            run.record(
                "startup-budget",
                True,
                f"elapsed_seconds={startup_seconds:.3f}; budget_seconds=120",
            )
            _write_report(run)
            print("[QLH-DEMO] OK mode=failure")
            print("[QLH-DEMO] RECOVERY worker-exit -> reassigned -> completed")
            return 0

        if config.mode == "live":
            backend_env = os.environ.copy()
            backend_env["QLH_API_PORT"] = str(config.api_port)
            backend_env["QLH_BACKEND_HOST"] = "127.0.0.1"
            backend_env["QLH_SERVER_IP"] = "127.0.0.1"
            backend_env["QLH_SERVER_PORT"] = str(_cluster_port(config))
            _start_process(
                run,
                "backend",
                [_live_python_command(), "-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1", "--port", str(config.api_port)],
                backend_env,
                "backend.log",
            )
            wait_for_health(_loopback_url(config.api_port, "/api/health"), _startup_timeout(config, started_at))
            run.record("start-backend", True, "health=ok")

        if not config.skip_frontend:
            frontend_env = os.environ.copy()
            if config.mode == "live":
                frontend_env["QLH_VITE_API_TARGET"] = _loopback_url(config.api_port)
            _start_process(
                run,
                "frontend",
                [_npm_command(), "run", "dev", "--", "--host", "127.0.0.1", "--port", str(config.frontend_port)],
                frontend_env,
                "frontend.log",
            )
            wait_for_ready(_loopback_url(config.frontend_port), _startup_timeout(config, started_at))
            run.record("start-frontend", True, "vite=ready")

        if config.mode == "fixtures":
            scenarios = verify_fixture_scenarios(run)
            if config.scenario is not None:
                selected = next((item for item in scenarios if item["kind"] == config.scenario), None)
                if selected is None:
                    raise RuntimeError(f"预置演示场景不存在：{config.scenario}")
                run.frontend_url = _loopback_url(config.frontend_port, f"/{selected['route']}")

        frontend_url = run.frontend_url or _loopback_url(config.frontend_port)
        run.record("demo-url", True, frontend_url)
        startup_seconds = time.monotonic() - started_at
        if startup_seconds > STARTUP_BUDGET_SECONDS:
            raise RuntimeError(f"启动超过 2 分钟门：{startup_seconds:.1f}s")
        run.record("startup-budget", True, f"elapsed_seconds={startup_seconds:.3f}; budget_seconds=120")
        _write_report(run)
        print(f"[QLH-DEMO] OK mode={config.mode}")
        print(f"[QLH-DEMO] URL {frontend_url}")
        if config.open_browser:
            webbrowser.open(frontend_url)
        if config.duration > 0:
            time.sleep(config.duration)
        elif run.processes:
            print("[QLH-DEMO] Press Ctrl+C to stop.")
            while all(item.process.poll() is None for item in run.processes):
                time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        run.record("stopped", True, "interrupted")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.record("runtime", False, type(exc).__name__)
        print(f"[QLH-DEMO] ERROR {exc}", file=sys.stderr)
        return 1
    finally:
        _write_report(run)
        run.stop()


def _write_report(run: DemoRun) -> None:
    payload = {
        "schema": "qlh.defense_demo.v1",
        "mode": run.config.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frontend_url": run.frontend_url,
        "api_url": _loopback_url(run.config.api_port, "/api/health") if run.config.mode == "live" else None,
        "steps": run.steps,
        "logs": [_public_path(path) for path in run.log_paths],
    }
    if run.failure_injection is not None:
        payload["failure_injection"] = run.failure_injection
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    targets = {DEFAULT_REPORT, run.config.report_path}
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH defense demo launcher")
    parser.add_argument("--mode", choices=("fixtures", "live", "failure"), default="fixtures")
    parser.add_argument("--scenario", choices=("dialog", "image", "topology"), help="open a fixture scenario directly")
    parser.add_argument("--api-port", type=_port, default=8000)
    parser.add_argument("--frontend-port", type=_port, default=5174)
    parser.add_argument("--timeout", type=float, default=30.0, dest="startup_timeout")
    parser.add_argument("--duration", type=float, default=0.0, help="stop automatically after N seconds")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="open the local demo URL")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DemoConfig(
        mode=args.mode,
        scenario=args.scenario,
        api_port=args.api_port,
        frontend_port=args.frontend_port,
        startup_timeout=args.startup_timeout,
        duration=args.duration,
        open_browser=args.open_browser,
        skip_frontend=args.skip_frontend,
        report_path=args.report,
    )
    return run_demo(config)


if __name__ == "__main__":
    raise SystemExit(main())
