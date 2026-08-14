from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scheduler as scheduler_module  # noqa: E402
import scheduler_svc_http  # noqa: E402
from qwen3_pipeline_contract import build_kv_contract  # noqa: E402
from qwen3_pipeline_control import (  # noqa: E402
    Qwen3LoopbackNetworkControlClient,
    router as network_control_router,
)
from qwen3_pipeline_data_plane import (  # noqa: E402
    Qwen3ArtifactTransferRuntime,
    router as transfer_router,
)
from qwen3_pipeline_multisidecar import (  # noqa: E402
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
)
from qwen3_pipeline_network import (  # noqa: E402
    Qwen3NetworkError,
    Qwen3NetworkHandoffTransport,
    Qwen3NetworkTarget,
    Qwen3NetworkTransferCoordinator,
    validate_qwen3_artifact_reference,
)
from qwen3_pipeline_peer_auth import (  # noqa: E402
    QWEN3_PEER_PROOF_HEADER,
    Qwen3PeerAuthError,
    Qwen3PeerAuthMiddleware,
    Qwen3PeerRequestSigner,
    Qwen3PeerRequestVerifier,
)
from qwen3_pipeline_transaction import build_qwen3_dry_run_contract  # noqa: E402
from qwen3_pipeline_transfer import (  # noqa: E402
    QWEN3_TRANSFER_PREFIX,
    Qwen3ArtifactTransferClient,
    TransferResponse,
)
from qwen3_pipeline_transfer import default_transfer_request  # noqa: E402


SECRET = "qwen3-network-contract-secret-value!!"


def _contract(*, segment_count=3, generation=5):
    nodes = ["node-a", "node-b", "node-c"][:segment_count]
    segments = []
    for index, node_id in enumerate(nodes):
        segments.append({
            "node_id": node_id,
            "layer_range": [index * 2, (index + 1) * 2],
            "has_embedding": index == 0,
            "has_lm_head": index == segment_count - 1,
            "required_bytes": 100,
            "assignment_manifest_sha256": f"{index + 1}" * 64,
            "execution_device": "cpu",
            "dtype": "float32",
        })
    return build_qwen3_dry_run_contract(
        config_id=f"cfg-{generation}",
        plan_id=f"plan-{generation}",
        generation=generation,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        total_layers=segment_count * 2,
        hidden_size=4,
        execution_mode="node_local_sidecar",
        segments=segments,
    )


def _requester(client, *, calls=None, disconnect_after_patch=None):
    state = disconnect_after_patch

    def request(method, url, headers, body):
        parsed = urlsplit(url)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        response = client.request(
            method,
            target,
            headers=dict(headers),
            content=body,
        )
        if calls is not None:
            calls.append({
                "method": method,
                "target": target,
                "body_bytes": 0 if body is None else len(body),
                "status": response.status_code,
            })
        if method == "PATCH" and state is not None and state["pending"]:
            state["pending"] = False
            raise OSError("simulated disconnect after persisted PATCH")
        return TransferResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    return request


def _node(tmp_path, node_id, allowed_peers, *, now=None):
    clock = (lambda: now[0]) if now is not None else None
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / node_id,
        cluster_secret=SECRET,
        clock=clock,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id=node_id,
        runtime=runtime,
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in allowed_peers,
        **({"clock": clock} if clock is not None else {}),
    )
    app = FastAPI()
    app.state.qwen3_artifact_transfer = runtime
    app.add_middleware(Qwen3PeerAuthMiddleware, verifier=verifier)
    app.include_router(transfer_router)
    return runtime, coordinator, TestClient(app)


class _Session:
    def __init__(self, index, layer_range, *, fail_phase=None):
        self.index = index
        self.layer_range = layer_range
        self.fail_phase = fail_phase
        self.calls = []
        self.input_paths = []

    def prepare(self):
        self.calls.append("prepare")
        return {"status": "prepared", "gate_passed": True}

    def commit(self):
        self.calls.append("commit")
        return {
            "status": "committed",
            "gate_passed": True,
            "full_model_materialized": False,
        }

    def execute(self, **request):
        phase = request["phase"]
        self.calls.append(phase)
        self.input_paths.append(Path(request["input_ref"]).resolve())
        if self.fail_phase == phase:
            raise RuntimeError(f"{phase} failed on segment {self.index}")
        incoming = Path(request["input_ref"]).read_bytes()
        data = f"{phase}:{self.index}:".encode("ascii") + incoming
        output = Path(request["output_ref"])
        output.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        sequence = int(request["sequence_length"])
        batch = int(request["batch_size"])
        generation = int(request["generation"])
        dtype = str(request["dtype"])
        device = str(request["device"])
        return {
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
                "artifact_bytes": len(data),
                "artifact_sha256": digest,
            },
            "kv_contract": build_kv_contract(
                chain_id=request["chain_id"],
                segment_index=self.index,
                layer_range=self.layer_range,
                sequence_length=sequence,
                batch_size=batch,
                dtype=dtype,
                device=device,
                phase=phase,
                generation=generation,
            ),
            "hidden_handoff": (
                {
                    "schema_version": 1,
                    "chain_id": request["chain_id"],
                    "from_segment": self.index,
                    "to_segment": self.index + 1,
                    "shape": [batch, sequence, 4],
                    "batch_size": batch,
                    "sequence_length": sequence,
                    "hidden_size": 4,
                    "dtype": dtype,
                    "device": device,
                }
                if request["has_next_segment"]
                else None
            ),
        }

    def release(self):
        self.calls.append("release")
        return {"status": "released"}

    def abort(self):
        self.calls.append("abort")
        return {"status": "aborted"}


def _network_chain(tmp_path, *, segment_count=3, fail_index=None, disconnect=False):
    contract = _contract(segment_count=segment_count)
    root = tmp_path / "chain"
    root.mkdir()
    allowed = {"node-a", "node-b"}
    targets = {}
    calls = []
    disconnect_state = {"pending": bool(disconnect)}
    for index, node_id in enumerate(["node-b", "node-c"][:segment_count - 1], start=1):
        _runtime, coordinator, client = _node(root, node_id, allowed)
        targets[node_id] = Qwen3NetworkTarget(
            node_id=node_id,
            base_url=f"http://127.0.0.1:{9800 + index}",
            coordinator=coordinator,
            requester=_requester(
                client,
                calls=calls,
                disconnect_after_patch=(
                    disconnect_state if disconnect and node_id == "node-b" else None
                ),
            ),
        )
    transport = Qwen3NetworkHandoffTransport(
        artifact_root=root,
        targets=targets,
        peer_signers={
            node_id: Qwen3PeerRequestSigner(
                SECRET,
                peer_node_id=node_id,
            )
            for node_id in ["node-a", "node-b"][:segment_count - 1]
        },
        chunk_bytes=7,
    )
    sessions = []

    def factory(frame):
        index = int(frame["segment_index"])
        session = _Session(
            index,
            list(frame["layer_range"]),
            fail_phase="prefill" if fail_index == index else None,
        )
        sessions.append(session)
        return session

    chain = Qwen3PipelineMultiSidecar.from_contract(
        contract=contract,
        artifact_root=root,
        session_factory=factory,
        handoff_transport=transport,
    )
    source = root / "input.pt"
    source.write_bytes(bytes(range(40)))
    return chain, sessions, source, root, targets, calls, disconnect_state


def test_peer_proof_binds_request_ticket_tcp_registry_and_nonce():
    now = [1000.0]
    allowed = {"node-a"}
    signer = Qwen3PeerRequestSigner(
        SECRET,
        peer_node_id="node-a",
        clock=lambda: now[0],
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in allowed,
        clock=lambda: now[0],
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/qtx_{'1' * 32}"
    proof = signer.proof("GET", path, "ticket", nonce="a" * 32)

    assert verifier.verify(
        proof,
        method="GET",
        path=path,
        transfer_ticket="ticket",
    ) == "node-a"
    with pytest.raises(Qwen3PeerAuthError) as replay:
        verifier.verify(proof, method="GET", path=path, transfer_ticket="ticket")
    assert replay.value.reason_code == "qwen3_peer_proof_replay"

    scoped = signer.proof("PATCH", path, "ticket", nonce="b" * 32)
    with pytest.raises(Qwen3PeerAuthError) as wrong_method:
        verifier.verify(scoped, method="GET", path=path, transfer_ticket="ticket")
    assert wrong_method.value.reason_code == "qwen3_peer_proof_scope"

    now[0] = 1031.0
    stale = signer.proof("GET", path, "ticket", nonce="c" * 32)
    now[0] = 1062.0
    with pytest.raises(Qwen3PeerAuthError) as expired:
        verifier.verify(stale, method="GET", path=path, transfer_ticket="ticket")
    assert expired.value.reason_code == "qwen3_peer_proof_expired"

    now[0] = 1070.0
    denied = signer.proof("GET", path, "ticket", nonce="d" * 32)
    allowed.clear()
    with pytest.raises(Qwen3PeerAuthError) as not_registered:
        verifier.verify(denied, method="GET", path=path, transfer_ticket="ticket")
    assert not_registered.value.reason_code == "qwen3_peer_not_authenticated"


def test_peer_auth_middleware_rejects_missing_proof_and_replay(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    contract = _contract(segment_count=2)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"proof-bound"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/{plan['transfer_id']}"
    authorization = {"Authorization": f"Bearer {plan['ticket']}"}
    missing = http.get(path, headers=authorization)
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "qwen3_peer_proof_missing"

    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")
    proof = signer.proof("GET", path, plan["ticket"])
    headers = {**authorization, QWEN3_PEER_PROOF_HEADER: proof}
    accepted = http.get(path, headers=headers)
    replay = http.get(path, headers=headers)
    assert accepted.status_code == 200
    assert replay.status_code == 401
    assert replay.json()["detail"]["code"] == "qwen3_peer_proof_replay"


def test_active_coordinator_fences_topology_generation_and_release(tmp_path):
    runtime, coordinator, _http = _node(tmp_path, "node-b", {"node-a"})
    contract = _contract(segment_count=2, generation=5)
    assert coordinator.activate(contract)["active"] is True
    data = b"contract-bound"
    fields = {
        "base_url": "http://127.0.0.1:9876",
        "chain_id": contract["contract_sha256"],
        "generation": 5,
        "phase": "prefill",
        "from_segment": 0,
        "to_segment": 1,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    with pytest.raises(Qwen3NetworkError) as wrong_peer:
        coordinator.begin_receive(source_peer_id="node-c", **fields)
    assert wrong_peer.value.reason_code == "qwen3_network_contract_mismatch"
    with pytest.raises(Qwen3NetworkError) as stale_decode:
        coordinator.begin_receive(
            source_peer_id="node-a",
            **{**fields, "phase": "decode", "generation": 5},
        )
    assert stale_decode.value.reason_code == "qwen3_network_contract_mismatch"

    with pytest.raises(Qwen3NetworkError) as before_phase:
        coordinator.begin_receive(source_peer_id="node-a", **fields)
    assert before_phase.value.reason_code == "qwen3_network_contract_mismatch"
    coordinator.begin_phase("prefill", 5)
    plan = coordinator.begin_receive(source_peer_id="node-a", **fields)
    assert runtime.receiver.root.joinpath(f".{plan['transfer_id']}.part").is_file()
    cleanup = coordinator.release()
    assert cleanup["cleanup_complete"] is True
    assert not runtime.receiver.root.joinpath(f".{plan['transfer_id']}.part").exists()
    with pytest.raises(Qwen3NetworkError) as fenced:
        coordinator.activate(contract)
    assert fenced.value.reason_code == "qwen3_network_generation_stale"
    assert coordinator.activate(_contract(segment_count=2, generation=6))["generation"] == 6


@pytest.mark.parametrize("segment_count", [2, 3])
def test_network_chain_runs_prefill_decode_with_path_free_references_and_cleanup(
    tmp_path,
    segment_count,
):
    chain, sessions, source, root, targets, _calls, _disconnect = _network_chain(
        tmp_path,
        segment_count=segment_count,
    )
    chain.prepare()
    chain.commit()
    prefill = chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    decode = chain.decode(input_ref=source, batch_size=1, sequence_length=4)

    assert prefill["handoff_modes"] == ["network"]
    assert decode["handoff_reference_count"] == 2 * (segment_count - 1)
    for phase in ("prefill", "decode"):
        references = chain.handoff_references(phase)
        assert len(references) == segment_count - 1
        for reference in references:
            assert validate_qwen3_artifact_reference(reference) == reference
            serialized = json.dumps(reference)
            assert "path" not in serialized.lower()
            assert "ticket" not in serialized.lower()
            assert reference["mode"] == "network"
    for index in range(1, segment_count):
        expected_root = targets[["node-b", "node-c"][index - 1]].coordinator.runtime.receiver.root
        assert any(path.is_relative_to(expected_root) for path in sessions[index].input_paths)
    assert chain.snapshot["full_model_materialized"] is False

    released = chain.release()
    assert released["phase"] == "released"
    assert released["cleanup_complete"] is True
    assert not list(root.glob("**/qtx_*.pt"))
    assert not list(root.glob("**/*.part"))


def test_network_chain_resumes_after_lost_ack_without_duplicate_bytes(tmp_path):
    chain, _sessions, source, _root, _targets, calls, disconnect = _network_chain(
        tmp_path,
        segment_count=2,
        disconnect=True,
    )
    chain.prepare()
    chain.commit()
    result = chain.prefill(input_ref=source, batch_size=1, sequence_length=3)

    assert result["phase"] == "prefilled"
    assert disconnect["pending"] is False
    gets = [call for call in calls if call["method"] == "GET"]
    patches = [call for call in calls if call["method"] == "PATCH"]
    assert len(gets) >= 2
    assert patches[0]["status"] == 200
    assert sum(call["body_bytes"] for call in patches) == len(
        Path(chain.artifact_refs("prefill")[0]).read_bytes()
    )
    chain.cancel()


def test_downstream_failure_cancels_network_artifacts_and_all_sidecars(tmp_path):
    chain, sessions, source, root, targets, _calls, _disconnect = _network_chain(
        tmp_path,
        segment_count=2,
        fail_index=1,
    )
    chain.prepare()
    chain.commit()
    with pytest.raises(Qwen3MultiSidecarError):
        chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    assert chain.phase == "aborted"
    assert all("abort" in session.calls for session in sessions)
    assert targets["node-b"].coordinator.snapshot()["active"] is False
    assert not list(root.glob("**/qtx_*.pt"))
    assert not list(root.glob("**/*.part"))


def test_runtime_startup_reconciles_only_scoped_orphan_files(tmp_path):
    root = tmp_path / "state" / "qwen3" / "network_artifacts"
    root.mkdir(parents=True)
    temporary = root / f".qtx_{'1' * 32}.part"
    committed = root / f"qtx_{'2' * 32}.pt"
    unrelated = root / "user-data.pt"
    temporary.write_bytes(b"partial")
    committed.write_bytes(b"committed")
    unrelated.write_bytes(b"keep")

    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / "state",
        cluster_secret=SECRET,
    )

    assert runtime.receiver.root == root.resolve()
    assert not temporary.exists()
    assert not committed.exists()
    assert unrelated.read_bytes() == b"keep"


def test_scheduler_service_wires_data_plane_to_live_tcp_peer_predicate(
    monkeypatch,
    tmp_path,
):
    class _TcpServer:
        _running = True

        def __init__(self):
            self.allowed = {"node-a"}

        def is_authenticated_loopback_client(self, peer):
            return peer in self.allowed

    sched = scheduler_module.Scheduler()
    server = _TcpServer()
    sched._tcp_server = server
    monkeypatch.setattr(sched, "_effective_role", lambda: "master")
    monkeypatch.setattr(sched, "_qwen3_cluster_secret", lambda: SECRET)
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path,
        cluster_secret=SECRET,
    )
    configured = sched.configure_qwen3_artifact_transfer(runtime)
    assert configured["production_admitted"] is False
    http = TestClient(scheduler_svc_http.build_scheduler_app(sched))
    data = b"scheduler-wired"
    plan = runtime.begin_receive(
        base_url="http://127.0.0.1:9876",
        peer_node_id="node-a",
        chain_id="a" * 64,
        generation=1,
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/{plan['transfer_id']}"
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")

    def headers():
        return {
            "Authorization": f"Bearer {plan['ticket']}",
            **signer.headers("GET", path, plan["ticket"]),
        }

    assert http.get(path, headers=headers()).status_code == 200
    server.allowed.clear()
    denied = http.get(path, headers=headers())
    assert denied.status_code == 401
    assert denied.json()["detail"]["code"] == "qwen3_peer_not_authenticated"


def test_scheduler_service_registers_authenticated_network_control_router(
    monkeypatch,
    tmp_path,
):
    class _TcpServer:
        _running = True

        def is_authenticated_loopback_client(self, peer):
            return peer == "node-a"

    sched = scheduler_module.Scheduler()
    sched._tcp_server = _TcpServer()
    monkeypatch.setattr(sched, "_effective_role", lambda: "master")
    monkeypatch.setattr(sched, "_qwen3_cluster_secret", lambda: SECRET)
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path,
        cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b",
        runtime=runtime,
    )
    configured = sched.configure_qwen3_artifact_transfer(
        runtime,
        network_coordinator=coordinator,
    )
    assert configured["network_control"] is True
    http = TestClient(scheduler_svc_http.build_scheduler_app(sched))
    response = http.post(
        "/internal/v1/qwen3/network-control/activate",
        json={"contract": _contract(segment_count=2)},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "qwen3_peer_proof_missing"


def test_loopback_process_control_client_fences_control_and_data_plane(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    http.app.state.qwen3_network_transfer_coordinator = coordinator
    http.app.include_router(network_control_router)
    contract = _contract(segment_count=2, generation=8)
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")

    def request(method, url, headers, body):
        parsed = urlsplit(url)
        return TransferResponse(
            status_code=(response := http.request(
                method,
                parsed.path + (f"?{parsed.query}" if parsed.query else ""),
                headers=dict(headers),
                content=body,
            )).status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    remote = Qwen3LoopbackNetworkControlClient(
        node_id="node-b",
        base_url="http://127.0.0.1:9876",
        artifact_root=runtime.receiver.root,
        signer=signer,
        requester=request,
    )
    activated = remote.activate(contract)
    assert activated["active"] is True
    remote.begin_phase("prefill", contract["generation"])
    source = tmp_path / "control-source.pt"
    source.write_bytes(b"control-process-bound")
    plan = remote.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ttl_seconds=60,
    )
    transfer = Qwen3ArtifactTransferClient(
        tmp_path,
        request,
        chunk_bytes=5,
        peer_proof_headers=signer.headers,
    )
    assert transfer.upload(source=source, plan=plan)["status"] == "committed"
    resolved = remote.resolve(plan["transfer_id"])
    assert resolved.path.read_bytes() == source.read_bytes()
    resolved.path.unlink()
    with pytest.raises(Qwen3NetworkError) as missing:
        remote.resolve(plan["transfer_id"])
    assert missing.value.reason_code == "qwen3_network_artifact_missing"
    remote.finish_phase("prefill", contract["generation"])
    released = remote.release()
    assert released["cleanup_complete"] is True
    assert not resolved.path.exists()


def test_loopback_process_control_rejects_body_rebinding(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    http.app.state.qwen3_network_transfer_coordinator = coordinator
    http.app.include_router(network_control_router)
    contract = _contract(segment_count=2, generation=9)
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")
    first = json.dumps({"contract": contract}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(first).hexdigest()
    path = "/internal/v1/qwen3/network-control/activate"
    headers = {
        "Authorization": f"Bearer {digest}",
        "Content-Type": "application/json",
        "Content-Length": str(len(first)),
        **signer.headers("POST", path, digest),
    }
    changed = first.replace(b"cfg-9", b"cfg-x")
    response = http.post(path, headers=headers, content=changed)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "qwen3_network_control_body_binding_mismatch"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_network_node(tmp_path, node_id, port, allowed_peers):
    helper = Path(__file__).parent / "helpers" / "qwen3_network_node.py"
    state_dir = tmp_path / node_id
    command = [
        sys.executable,
        str(helper),
        "--node-id", node_id,
        "--port", str(port),
        "--state-dir", str(state_dir),
        "--secret", SECRET,
    ]
    for peer in allowed_peers:
        command.extend(["--allowed-peer", peer])
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}{QWEN3_TRANSFER_PREFIX}/status"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qwen3 test node {node_id} exited with {process.returncode}")
        try:
            response = default_transfer_request("GET", url, {}, None)
            if response.status_code == 200:
                return process, state_dir
        except Exception:
            pass
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(f"Qwen3 test node {node_id} did not become ready")


def _stop_network_node(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_same_machine_three_process_network_chain_prefill_decode(tmp_path):
    contract = _contract(segment_count=3, generation=12)
    root = tmp_path / "chain"
    root.mkdir()
    ports = {"node-b": _free_port(), "node-c": _free_port()}
    processes = []
    targets = {}
    try:
        for node_id, allowed in (("node-b", ["node-a"]), ("node-c", ["node-b"])):
            process, state_dir = _spawn_network_node(
                root, node_id, ports[node_id], allowed,
            )
            processes.append(process)
            targets[node_id] = Qwen3NetworkTarget(
                node_id=node_id,
                base_url=f"http://127.0.0.1:{ports[node_id]}",
                coordinator=Qwen3LoopbackNetworkControlClient(
                    node_id=node_id,
                    base_url=f"http://127.0.0.1:{ports[node_id]}",
                    artifact_root=state_dir / "qwen3" / "network_artifacts",
                    signer=Qwen3PeerRequestSigner(
                        SECRET,
                        peer_node_id="node-a" if node_id == "node-b" else "node-b",
                    ),
                ),
            )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets=targets,
            peer_signers={
                "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
                "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
            },
            chunk_bytes=7,
        )
        sessions = []

        def factory(frame):
            session = _Session(int(frame["segment_index"]), list(frame["layer_range"]))
            sessions.append(session)
            return session

        chain = Qwen3PipelineMultiSidecar.from_contract(
            contract=contract,
            artifact_root=root,
            session_factory=factory,
            handoff_transport=transport,
        )
        source = root / "input.pt"
        source.write_bytes(bytes(range(40)))
        chain.prepare()
        chain.commit()
        assert chain.prefill(input_ref=source, batch_size=1, sequence_length=3)["phase"] == "prefilled"
        assert chain.decode(input_ref=source, batch_size=1, sequence_length=4)["phase"] == "decoded"
        assert chain.handoff_references("prefill")[0]["mode"] == "network"
        assert chain.handoff_references("decode")[1]["target_node_id"] == "node-c"
        chain.release()
        assert not list(root.glob("**/qtx_*.pt"))
        assert not list(root.glob("**/*.part"))
    finally:
        for process in reversed(processes):
            _stop_network_node(process)


def test_same_machine_network_node_restart_fences_stale_contract_and_reconciles_files(tmp_path):
    contract = _contract(segment_count=2, generation=14)
    port = _free_port()
    process, state_dir = _spawn_network_node(tmp_path / "restart", "node-b", port, ["node-a"])
    try:
        artifact_root = state_dir / "qwen3" / "network_artifacts"
        remote = Qwen3LoopbackNetworkControlClient(
            node_id="node-b",
            base_url=f"http://127.0.0.1:{port}",
            artifact_root=artifact_root,
            signer=Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
        )
        remote.activate(contract)
        artifact_root.mkdir(parents=True, exist_ok=True)
        orphan_part = artifact_root / f".qtx_{'a' * 32}.part"
        orphan_final = artifact_root / f"qtx_{'b' * 32}.pt"
        unrelated = artifact_root / "keep.pt"
        orphan_part.write_bytes(b"orphan")
        orphan_final.write_bytes(b"orphan")
        unrelated.write_bytes(b"keep")
        _stop_network_node(process)
        process, _ = _spawn_network_node(tmp_path / "restart", "node-b", port, ["node-a"])
        assert not orphan_part.exists()
        assert not orphan_final.exists()
        assert unrelated.read_bytes() == b"keep"
        with pytest.raises(Qwen3NetworkError) as stale:
            remote.begin_phase("prefill", contract["generation"])
        assert stale.value.reason_code == "qwen3_network_contract_inactive"
    finally:
        _stop_network_node(process)


def test_peer_registration_epoch_fences_reconnect_proof_and_transfer_ticket():
    epochs = {"node-a": 1}
    signer = Qwen3PeerRequestSigner(
        SECRET, peer_node_id="node-a", peer_epoch_provider=lambda: epochs["node-a"],
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in epochs,
        is_authenticated_peer_epoch=lambda peer, epoch: epochs.get(peer) == epoch,
    )
    ticket = "digest"
    proof = signer.proof("GET", "/internal/v1/qwen3/artifact-transfer/status", ticket)
    assert verifier.verify(
        proof,
        method="GET",
        path="/internal/v1/qwen3/artifact-transfer/status",
        transfer_ticket=ticket,
    ) == "node-a"
    epochs["node-a"] = 2
    fresh = signer.proof("GET", "/internal/v1/qwen3/artifact-transfer/status", ticket)
    with pytest.raises(Qwen3PeerAuthError) as stale:
        verifier.verify(
            proof,
            method="GET",
            path="/internal/v1/qwen3/artifact-transfer/status",
            transfer_ticket=ticket,
        )
    assert stale.value.reason_code == "qwen3_peer_epoch_mismatch"
    assert verifier.verify(
        fresh,
        method="GET",
        path="/internal/v1/qwen3/artifact-transfer/status",
        transfer_ticket=ticket,
    ) == "node-a"


def test_target_consume_returns_path_free_execution_contract(tmp_path):
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path, cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b", runtime=runtime,
    )
    contract = _contract(segment_count=2, generation=18)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"target-local-artifact"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    transfer_id = plan["transfer_id"]
    runtime.receiver.write(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
        offset=0,
        data=data,
    )
    runtime.receiver.commit(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
    )
    coordinator.resolve(transfer_id)
    result = coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=False,
    )
    encoded = json.dumps(result, ensure_ascii=True)
    assert "path" not in encoded.lower()
    assert result["execution"]["artifact_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["hidden_handoff"]["has_next_segment"] is False
