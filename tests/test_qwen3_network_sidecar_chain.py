"""QW3.17：真实 network sidecar 进程链（2/3 节点 CPU 真实 torch artifact）。

架构：主进程提供 source artifact → `Qwen3NetworkHandoffTransport`
（target_execution=True）逐段 `transfer_and_consume` 到目标 helper 进程 →
helper 进程内真实 `Qwen3NetworkSidecarExecutor`（隔离 sidecar session 在
`.venv-qwen3-sidecar` 中真实加载层段）执行并返回 path-free
output_reference。全程 `full_model_materialized=false`、禁止整模回退、
控制面不返回路径/ticket/tensor。

前置：`models/qwen3-4b`（真实 Safetensors）+ `.venv-qwen3-sidecar`
（transformers 4.57.6 / torch 2.13.0+cu126）。两者缺失时整文件 skip
（真机工件门，不虚构证据）。RAM 准入：2 节点 ≥ 3GB、3 节点 ≥ 5GB，
不足 skip（与项目资源门惯例一致，结论登记后置）。
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_pipeline_network import (  # noqa: E402
    Qwen3NetworkHandoffTransport,
    Qwen3NetworkTarget,
)
from qwen3_pipeline_peer_auth import Qwen3PeerRequestSigner  # noqa: E402
from qwen3_pipeline_control import Qwen3LoopbackNetworkControlClient  # noqa: E402
from qwen3_pipeline_transaction import build_qwen3_dry_run_contract  # noqa: E402
from qwen3_pipeline_transfer import (  # noqa: E402
    QWEN3_TRANSFER_PREFIX,
    default_transfer_request,
)

SECRET = "qwen3-network-contract-secret-value!!"
MODEL_PATH = ROOT / "models" / "qwen3-4b"
SIDECAR_PYTHON = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
TOTAL_LAYERS = 36

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_dir() or not SIDECAR_PYTHON.is_file(),
    reason="QW3.17 需要真实 models/qwen3-4b 工件与 .venv-qwen3-sidecar（真机工件门）",
)


def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / (2**30)


def _real_contract(*, segment_count, generation):
    """node_local_sidecar 合同：每段 1 层（[0,1]/[1,2]/[2,3]），36 层模型。"""
    nodes = ["node-a", "node-b", "node-c"][:segment_count]
    segments = []
    for index, node_id in enumerate(nodes):
        segments.append({
            "node_id": node_id,
            "layer_range": [index, index + 1],
            "has_embedding": index == 0,
            "has_lm_head": index == segment_count - 1,
            "required_bytes": 100,
            "assignment_manifest_sha256": f"{index + 1}" * 64,
            "execution_device": "cpu",
            "dtype": "float32",
        })
    return build_qwen3_dry_run_contract(
        config_id=f"cfg-real-{generation}",
        plan_id=f"plan-real-{generation}",
        generation=generation,
        model_id="qwen3-4b",
        model_sha256=("2c54d5a09e7e92d4f5126b92a5a457448c9593e6" + "0" * 24),
        total_layers=segment_count,  # 合同层数=段覆盖；真实加载校验用 config.json（36 层）
        hidden_size=4,
        execution_mode="node_local_sidecar",
        segments=segments,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn_real_node(tmp_path, node_id, port, allowed_peers, *,
                     layer_range, has_embedding=False, has_lm_head=False):
    """启动带真实 sidecar executor 的 helper 进程（隔离 sidecar 真实加载）。"""
    helper = Path(__file__).parent / "helpers" / "qwen3_network_node.py"
    state_dir = tmp_path / node_id
    command = [
        sys.executable,
        str(helper),
        "--node-id", node_id,
        "--port", str(port),
        "--state-dir", str(state_dir),
        "--secret", SECRET,
        "--sidecar-model-path", str(MODEL_PATH),
        "--sidecar-python", str(SIDECAR_PYTHON),
        "--sidecar-layer-range", str(layer_range[0]), str(layer_range[1]),
        "--sidecar-total-layers", str(TOTAL_LAYERS),
    ]
    if has_embedding:
        command.append("--sidecar-has-embedding")
    if has_lm_head:
        command.append("--sidecar-has-lm-head")
    for peer in allowed_peers:
        command.extend(["--allowed-peer", peer])
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}{QWEN3_TRANSFER_PREFIX}/status"
    deadline = time.monotonic() + 120  # 真实 sidecar 加载权重慢
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Qwen3 真实节点 {node_id} 退出 rc={process.returncode}（状态目录 {state_dir}）"
            )
        try:
            response = default_transfer_request("GET", url, {}, None)
            if response.status_code == 200:
                return process, state_dir
        except Exception:
            pass
        time.sleep(0.2)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError(f"Qwen3 真实节点 {node_id} 未就绪（120s 超时）")


def _stop_node(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _targets(tmp_path, contract, ports, roots):
    targets = {}
    for index in range(1, len(contract["segments"])):
        node_id = contract["segments"][index]["node_id"]
        signer_peer = "node-a" if index == 1 else contract["segments"][index - 1]["node_id"]
        targets[node_id] = Qwen3NetworkTarget(
            node_id=node_id,
            base_url=f"http://127.0.0.1:{ports[node_id]}",
            coordinator=Qwen3LoopbackNetworkControlClient(
                node_id=node_id,
                base_url=f"http://127.0.0.1:{ports[node_id]}",
                artifact_root=roots[node_id] / "qwen3" / "network_artifacts",
                signer=Qwen3PeerRequestSigner(SECRET, peer_node_id=signer_peer),
            ),
        )
    return targets



def _make_source_tensor(root: Path, name: str, sequence_length: int) -> Path:
    """用 sidecar python 生成合法 hidden-handoff tensor（torch 序列化）。

    worker 的 execute 用 torch.load 读取 input_ref；纯字节文件会被拒绝。
    hidden_size 取 qwen3-4b 的 2560，shape [1, seq, 2560]。
    """
    path = root / name
    # worker 要求 dict 且恰好含 input_ids 或 hidden_states 之一（prefill 用）
    code = (
        "import torch; "
        f"torch.save({{'hidden_states': torch.zeros(1, {sequence_length}, 2560)}}, r'{path}')"
    )
    subprocess.run(
        [str(SIDECAR_PYTHON), "-c", code],
        cwd=str(ROOT), check=True, capture_output=True,
    )
    return path


def _run_real_chain(tmp_path, *, segment_count, generation):
    """spawn 真实节点并执行 prefill → decode → release；返回执行报告。"""
    contract = _real_contract(segment_count=segment_count, generation=generation)
    root = tmp_path / "real-chain"
    root.mkdir()
    node_ids = [segment["node_id"] for segment in contract["segments"][1:]]
    ports = {node_id: _free_port() for node_id in node_ids}
    processes = []
    roots = {}
    try:
        for index, node_id in enumerate(node_ids, start=1):
            segment = contract["segments"][index]
            # 段 1 的注册表需含段 2 节点：transfer_registered_output 的
            # output lease 以 target 身份请求 source（for_peer(target signer)）
            allowed = ["node-a"]
            if index + 1 < len(contract["segments"]):
                allowed.append(contract["segments"][index + 1]["node_id"])
            if index > 1:
                allowed.append(contract["segments"][index - 1]["node_id"])
            process, state_dir = _spawn_real_node(
                tmp_path, node_id, ports[node_id], allowed,
                layer_range=segment["layer_range"],
                has_embedding=segment["has_embedding"],
                has_lm_head=segment["has_lm_head"],
            )
            processes.append(process)
            roots[node_id] = state_dir
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets=_targets(tmp_path, contract, ports, roots),
            peer_signers={
                segment["node_id"]: Qwen3PeerRequestSigner(SECRET, peer_node_id=segment["node_id"])
                for segment in contract["segments"]
            },
            chunk_bytes=64 * 1024,
            target_execution=True,
        )
        transport.activate(contract)
        source = _make_source_tensor(root, "input.pt", sequence_length=4)

        # execute_target_chain 不自动激活 phase（multisidecar 的 _execute
        # 会在每阶段前调用 begin_phase）——直接调用必须先 begin_phase
        transport.begin_phase("prefill", int(generation))
        prefill = transport.execute_target_chain(
            source_path=source, phase="prefill", generation=int(generation),
            batch_size=1, sequence_length=4,
        )
        transport.finish_phase("prefill", int(generation))
        transport.begin_phase("decode", int(generation) + 1)
        # decode 输入是增量新 token（[1,1]）；KV 总长 = prefill 4 + 新 1 = 5
        decode_source = _make_source_tensor(root, "decode_input.pt", sequence_length=1)
        decode = transport.execute_target_chain(
            source_path=decode_source, phase="decode", generation=int(generation) + 1,
            batch_size=1, sequence_length=5,
        )
        return {"prefill": prefill, "decode": decode}
    finally:
        for process in reversed(processes):
            _stop_node(process)


def test_real_sidecar_two_node_chain_prefill_decode(tmp_path):
    """2 节点（node-b [0,1]+embedding）：真实工件 prefill→decode→release。"""
    if _available_ram_gb() < 3.0:
        pytest.skip("可用 RAM < 3GB，2 节点真实链后置（资源门）")
    result = _run_real_chain(tmp_path, segment_count=2, generation=80)
    executions = [e["execution"] for e in result["prefill"]["executions"]]
    assert len(executions) == 1  # 2 节点链：1 个远端目标段执行
    assert executions[0]["input_consumed"] is True
    # full_model_materialized 是执行结果顶层证据（不在 execution 子结构内）
    assert result["prefill"]["full_model_materialized"] is False
    # decode 使用同一目标段，KV 契约存在
    assert result["decode"]["executions"][0]["kv_contract"]["present"] is True
    assert result["decode"]["full_model_materialized"] is False
    # 2 节点链无段间转发：无输出被中途释放（末段输出保留在目标端）
    assert result["prefill"]["completed"] is True
    assert result["prefill"]["released_output_ids"] == []


def test_real_sidecar_three_node_chain_prefill_decode(tmp_path):
    """3 节点（node-b [0,1]+emb、node-c [1,2]）：output reference 级联。"""
    if _available_ram_gb() < 5.0:
        pytest.skip("可用 RAM < 5GB，3 节点真实链后置（资源门）")
    result = _run_real_chain(tmp_path, segment_count=3, generation=81)
    prefill = result["prefill"]
    assert len(prefill["executions"]) == 2
    # 第二段的 input 来自第一段 output reference（path-free 级联）
    assert prefill["executions"][1]["input_reference"]["artifact_id"]
    for execution in prefill["executions"]:
        assert execution["execution"]["input_consumed"] is True
    assert prefill["full_model_materialized"] is False
    decode = result["decode"]
    assert len(decode["executions"]) == 2
    assert decode["executions"][0]["kv_contract"]["present"] is True
    assert decode["completed"] is True
    # 3 节点链：段 1 输出在转发给段 2 前被释放一次（prefill 与 decode 各一次）
    assert len(decode["released_output_ids"]) == 1


def test_real_sidecar_consume_retry_is_idempotent(tmp_path):
    """重复 consume（同合同同参数）返回缓存结果，不重复物化。"""
    if _available_ram_gb() < 3.0:
        pytest.skip("可用 RAM < 3GB，真实链后置（资源门）")
    contract = _real_contract(segment_count=2, generation=82)
    root = tmp_path / "idem-chain"
    root.mkdir()
    node_b = contract["segments"][1]
    port = _free_port()
    process, state_dir = _spawn_real_node(
        tmp_path, "node-b", port, ["node-a"],
        layer_range=node_b["layer_range"],
    )
    try:
        from qwen3_pipeline_control import (
            Qwen3LoopbackNetworkControlClient,
        )
        from qwen3_pipeline_network import (
            Qwen3NetworkHandoffTransport,
            Qwen3NetworkTarget,
        )
        target = Qwen3NetworkTarget(
            node_id="node-b",
            base_url=f"http://127.0.0.1:{port}",
            coordinator=Qwen3LoopbackNetworkControlClient(
                node_id="node-b",
                base_url=f"http://127.0.0.1:{port}",
                artifact_root=state_dir / "qwen3" / "network_artifacts",
                signer=Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
            ),
        )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets={"node-b": target},
            peer_signers={
                "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
                "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
            },
            chunk_bytes=64 * 1024,
            target_execution=True,
        )
        transport.activate(contract)
        source = _make_source_tensor(root, "input.pt", sequence_length=4)
        transport.begin_phase("prefill", 82)
        first = transport.transfer_and_consume(
            source_path=source,
            chain_id=contract["contract_sha256"],
            generation=82,
            phase="prefill",
            from_segment=0,
            to_segment=1,
            source_node_id="node-a",
            target_node_id="node-b",
            batch_size=1,
            sequence_length=4,
            dtype="float32",
            device="cpu",
            has_next_segment=False,
        )
        second = transport.transfer_and_consume(
            source_path=source,
            chain_id=contract["contract_sha256"],
            generation=82,
            phase="prefill",
            from_segment=0,
            to_segment=1,
            source_node_id="node-a",
            target_node_id="node-b",
            batch_size=1,
            sequence_length=4,
            dtype="float32",
            device="cpu",
            has_next_segment=False,
        )
        # 幂等：第二次返回缓存结果，输出 reference 一致
        assert first["consume"]["full_model_materialized"] is False
        assert (
            second["consume"].get("output_reference")
            == first["consume"].get("output_reference")
        )
    finally:
        _stop_node(process)


def test_real_sidecar_revoke_interrupts_and_cleans_artifacts(tmp_path):
    """revoke 中断：传输取消后 executor 清理 consume 工件（无残留 lease）。"""
    if _available_ram_gb() < 3.0:
        pytest.skip("可用 RAM < 3GB，真实链后置（资源门）")
    contract = _real_contract(segment_count=2, generation=83)
    root = tmp_path / "revoke-chain"
    root.mkdir()
    node_b = contract["segments"][1]
    port = _free_port()
    process, state_dir = _spawn_real_node(
        tmp_path, "node-b", port, ["node-a"],
        layer_range=node_b["layer_range"],
    )
    try:
        from qwen3_pipeline_control import (
            Qwen3LoopbackNetworkControlClient,
        )
        from qwen3_pipeline_network import (
            Qwen3NetworkHandoffTransport,
            Qwen3NetworkTarget,
        )
        target = Qwen3NetworkTarget(
            node_id="node-b",
            base_url=f"http://127.0.0.1:{port}",
            coordinator=Qwen3LoopbackNetworkControlClient(
                node_id="node-b",
                base_url=f"http://127.0.0.1:{port}",
                artifact_root=state_dir / "qwen3" / "network_artifacts",
                signer=Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
            ),
        )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets={"node-b": target},
            peer_signers={
                "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
                "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
            },
            chunk_bytes=64 * 1024,
            target_execution=True,
        )
        transport.activate(contract)
        source = _make_source_tensor(root, "input.pt", sequence_length=4)
        transport.begin_phase("prefill", 83)
        reference = transport.transfer_reference(
            source_path=source,
            chain_id=contract["contract_sha256"],
            generation=83,
            phase="prefill",
            from_segment=0,
            to_segment=1,
            source_node_id="node-a",
            target_node_id="node-b",
        )
        # 传输已提交但未 consume → revoke
        target.coordinator.cancel_transfer(reference["artifact_id"])
        # 取消后目标端不得遗留 consume 工件
        leftovers = list(state_dir.rglob("qwen3-consume-*"))
        assert leftovers == []
        # 再次 consume 同一 transfer 必须 fail-closed（已撤销）
        from qwen3_pipeline_network import Qwen3NetworkError
        with pytest.raises(Qwen3NetworkError):
            transport.consume_target(
                target_node_id="node-b",
                transfer_id=reference["artifact_id"],
                phase="prefill",
                generation=83,
                batch_size=1,
                sequence_length=4,
                dtype="float32",
                device="cpu",
                has_next_segment=False,
            )
    finally:
        _stop_node(process)
