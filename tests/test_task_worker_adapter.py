import copy
import sys
import os
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tcp_comm as tcp_comm_mod
from task_worker_adapter import TaskWorkerControlPlane
from task_worker_protocol import (
    WorkerProtocolError,
    build_message,
    canonical_sha256,
    stage_input_sha256,
)
from task_provider import (
    ModelIdentity,
    ProviderExecutionError,
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)
from task_worker_adapter import RemoteFullWorkerProvider, remote_provider_id
from tcp_comm import MessageType, TCPClient, TCPServer


@pytest.fixture(autouse=True)
def _enable_task_worker_experiment(monkeypatch):
    import scheduler as scheduler_mod

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _capabilities():
    return {
        "stage_types": ["full_inference", "aggregate"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": "qwen-1_8b",
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "local-rev1",
            "sha256": "1" * 64,
        }],
        "max_concurrency": 1,
    }


def _admitted_control_plane():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    worker.receive_on_worker(ack.snapshot())
    return coordinator, worker


def _model_identity():
    return ModelIdentity(**_capabilities()["models"][0])


def _remote_request(provider_id):
    return StageRequest(
        workflow_id="wf_remote001",
        request_id="request-remote-01",
        stage_id="candidate_a",
        stage_type="full_inference",
        provider_id=provider_id,
        dependencies={},
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
    )


def _response_identity(offer):
    return {
        "workflow_id": offer["workflow_id"],
        "stage_id": offer["stage_id"],
        "attempt_id": offer["attempt_id"],
        "lease_id": offer["lease_id"],
        "lease_epoch": offer["lease_epoch"],
        "provider_id": offer["provider_id"],
    }


def _v1_hello(node_id="worker_01"):
    return build_message(
        "hello",
        {
            "node_id": node_id,
            "worker_kind": "pc_full_worker",
            "min_version": 1,
            "max_version": 1,
            "capabilities": _capabilities(),
        },
        message_id="msg_v1hello01",
        sent_at_ms=1000,
        version=1,
    )


def _stage_offer():
    root_input = {"message": "hello"}
    dependencies = {}
    return build_message(
        "stage_offer",
        {
            "workflow_id": "wf_adapter001",
            "request_id": "request-adapter-01",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_adapter001",
            "lease_id": "lease_adapter001",
            "lease_epoch": 1,
            "lease_expires_at_ms": 2000,
            "provider_id": "worker_01",
            "root_input": root_input,
            "dependencies": dependencies,
            "input_sha256": stage_input_sha256(root_input, dependencies),
            "model_identity": _capabilities()["models"][0],
        },
        message_id="msg_offer001",
        sent_at_ms=1000,
        version=2,
    )


def test_v2_hello_negotiates_and_reports_control_plane_only():
    worker = TaskWorkerControlPlane()
    coordinator = TaskWorkerControlPlane()

    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=1000,
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(),
        coordinator_node_id="master", sent_at_ms=1001,
    )
    worker.receive_on_worker(ack.snapshot())

    master_status = coordinator.status(role="master")
    worker_status = worker.status(role="client")
    assert ack.payload == {
        "coordinator_node_id": "master",
        "accepted": True,
        "selected_version": 2,
        "reason_code": "",
    }
    assert master_status["control_plane_connected"] is True
    assert master_status["connected_worker_count"] == 1
    assert master_status["workers"][0]["capabilities"] == _capabilities()
    assert master_status["adapter_connected"] is False
    assert master_status["task_dispatch_enabled"] is False
    assert worker_status["coordinator"]["selected_version"] == 2


def test_v1_and_transport_identity_mismatch_are_stably_rejected():
    coordinator = TaskWorkerControlPlane()

    old_ack = coordinator.receive_on_coordinator(
        "worker_01", _v1_hello().snapshot(), coordinator_node_id="master",
    )
    mismatch_ack = coordinator.receive_on_coordinator(
        "different_worker", _v1_hello("worker_01").snapshot(),
        coordinator_node_id="master",
    )

    assert old_ack.payload["accepted"] is False
    assert old_ack.payload["reason_code"] == "protocol_v2_required"
    assert mismatch_ack.payload["accepted"] is False
    assert mismatch_ack.payload["reason_code"] == "node_identity_mismatch"


def test_stage_messages_are_fenced_before_provider_dispatch():
    coordinator = TaskWorkerControlPlane()

    with pytest.raises(WorkerProtocolError) as captured:
        coordinator.receive_on_coordinator(
            "worker_01", _stage_offer().snapshot(), coordinator_node_id="master",
        )

    assert captured.value.code == "control_plane_only"
    assert coordinator.status(role="master")["rejected_message_count"] == 1


def test_duplicate_hello_is_idempotent_but_message_id_conflict_is_rejected():
    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(), sent_at_ms=1000,
    )
    assert hello is not None

    first = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
        sent_at_ms=1001,
    )
    duplicate = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
        sent_at_ms=2000,
    )
    assert duplicate.snapshot() == first.snapshot()

    conflicting = copy.deepcopy(hello.snapshot())
    conflicting["payload"]["capabilities"]["max_concurrency"] = 2
    with pytest.raises(WorkerProtocolError) as captured:
        coordinator.receive_on_coordinator(
            "worker_01", conflicting, coordinator_node_id="master",
        )
    assert captured.value.code == "message_id_conflict"


def test_disconnect_clears_health_and_pending_hello_fence():
    worker = TaskWorkerControlPlane()
    first = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert first is not None
    assert worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    ) is None

    worker.disconnect_coordinator()
    assert worker.status(role="client")["control_plane_connected"] is False
    assert worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    ) is not None


def test_remote_provider_requires_exact_model_and_returns_fenced_identity():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_remote001",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_remote001",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    deadline = time.time() + 2
    while not sent and time.time() < deadline:
        time.sleep(0.01)
    assert sent and sent[0].message_type == "stage_offer"
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_acceptremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "remote answer", "usage": {"total_tokens": 2}}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {
                "usage": {"total_tokens": 2},
                "usage_estimated": False,
                "model": "qwen-1_8b",
            },
        },
        message_id="msg_resultremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    assert finished.wait(2)
    assert "error" not in outcome
    result = outcome["result"]
    assert result.output == output
    assert result.provider_id == provider.provider_id
    assert result.attempt_id == attempt.attempt_id
    assert result.lease_epoch == 1
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0

    wrong_model_request = StageRequest(
        **{
            **request.__dict__,
            "model_identity": ModelIdentity(
                model_id="other-model",
                engine="pytorch",
                format="safetensors",
                revision="local-other",
                sha256="2" * 64,
            ),
        }
    )
    with pytest.raises(ProviderUnavailable) as mismatch:
        provider.reserve(wrong_model_request)
    assert mismatch.value.code == "model_identity_mismatch"


def test_remote_provider_rejects_wrong_epoch_without_waking_attempt():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_remote002",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_remote002",
        lease_epoch=2,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    deadline = time.time() + 2
    while not sent and time.time() < deadline:
        time.sleep(0.01)
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_acceptremote02",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    output = {"content": "wrong epoch"}
    with pytest.raises(WorkerProtocolError) as wrong_epoch:
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "lease_epoch": 1,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_resultremote02",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
    assert wrong_epoch.value.code == "attempt_identity_mismatch"
    assert not finished.wait(0.1)

    correct = {"content": "correct epoch"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": correct,
            "output_sha256": canonical_sha256(correct),
            "metadata": {},
        },
        message_id="msg_resultremote03",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["result"].output == correct
    provider.release(reservation.reservation_id)


def test_remote_provider_disconnect_unblocks_pending_attempt():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_disconnect01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_disconnect01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    deadline = time.time() + 2
    while not sent and time.time() < deadline:
        time.sleep(0.01)
    assert sent

    provider.notify_disconnect()

    assert finished.wait(2)
    assert outcome["error"].code == "remote_worker_disconnected"
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0


def test_task_graph_commits_remote_result_through_existing_winner_gate():
    from task_graph import StageSpec, TaskGraphCoordinator

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}

    def respond(message):
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_graphaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        output = {"content": "remote graph result"}
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_graphresult01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=respond,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)

    output, workflow = coordinator.run(
        stages=[StageSpec(
            "candidate_a",
            "full_inference",
            provider=provider.provider_id,
            fallback_providers=(),
            pure=False,
        )],
        final_stage_id="candidate_a",
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
        workflow_id="wf_remotegraph01",
    )

    assert output == {"content": "remote graph result"}
    stage = workflow["stages"][0]
    assert stage["winner_attempt_id"]
    assert stage["attempts"][0]["provider_kind"] == (
        "remote_full_worker"
    )
    assert stage["attempts"][0]["provider_node_id"] == "worker_01"
    assert stage["fallback_providers"] == []
    coordinator.close()


def test_hello_round_trips_over_authenticated_loopback_tcp(monkeypatch):
    import config as cfg

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "n2-control-plane-test-secret")
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "")
    )
    monkeypatch.setattr(
        TCPClient,
        "_heartbeat_loop",
        lambda self, connection_generation=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

    coordinator = TaskWorkerControlPlane()
    worker = TaskWorkerControlPlane()
    completed = threading.Event()
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, message):
        if message.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
        elif message.get("type") == MessageType.TASK_WORKER.value:
            ack = coordinator.receive_on_coordinator(
                client_id,
                message["data"],
                coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )

    def on_client_message(message):
        if message.get("type") == MessageType.TASK_WORKER.value:
            worker.receive_on_worker(message["data"])
            completed.set()

    client = TCPClient(
        server_host="127.0.0.1",
        server_port=port,
        client_id="worker_01",
        role="client",
    )
    try:
        server.start(on_message=on_server_message)
        assert client.connect(on_message=on_client_message) is True
        hello = worker.begin_worker_hello(
            node_id="worker_01", capabilities=_capabilities(),
        )
        assert hello is not None
        client.send_data(hello.snapshot(), MessageType.TASK_WORKER)
        assert completed.wait(3.0)
        assert coordinator.status(role="master")["connected_worker_count"] == 1
        assert worker.status(role="client")["control_plane_connected"] is True
    finally:
        client.disconnect()
        server.stop()


def test_single_stage_round_trips_over_authenticated_loopback_tcp(monkeypatch):
    import config as cfg

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "n2-stage-loopback-test-secret")
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: "")
    )
    monkeypatch.setattr(
        TCPClient,
        "_heartbeat_loop",
        lambda self, connection_generation=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")

    coordinator_control = TaskWorkerControlPlane()
    worker_control = TaskWorkerControlPlane()
    hello_completed = threading.Event()
    provider_holder = {}
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, outer):
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello":
            ack = coordinator_control.receive_on_coordinator(
                client_id, inner, coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )
        else:
            provider_holder["provider"].handle_message(inner)

    client = TCPClient(
        server_host="127.0.0.1",
        server_port=port,
        client_id="worker_01",
        role="client",
    )

    def on_client_message(outer):
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello_ack":
            worker_control.receive_on_worker(inner)
            hello_completed.set()
            return
        offer = inner["payload"]
        identity = _response_identity(offer)
        accepted = build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_tcpaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        )
        client.send_data(accepted.snapshot(), MessageType.TASK_WORKER)
        output = {"content": "tcp remote result"}
        result = build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id="msg_tcpresult01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        )
        client.send_data(result.snapshot(), MessageType.TASK_WORKER)

    try:
        server.start(on_message=on_server_message)
        assert client.connect(on_message=on_client_message)
        hello = worker_control.begin_worker_hello(
            node_id="worker_01", capabilities=_capabilities(),
        )
        assert hello is not None
        client.send_data(hello.snapshot(), MessageType.TASK_WORKER)
        assert hello_completed.wait(2)

        provider = RemoteFullWorkerProvider(
            node_id="worker_01",
            peer_snapshot=lambda: coordinator_control.worker_snapshot(
                "worker_01"
            ),
            send_message=lambda message: server.send_to_client(
                "worker_01", message.snapshot(), MessageType.TASK_WORKER,
            ),
        )
        provider_holder["provider"] = provider
        request = _remote_request(provider.provider_id)
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_tcpstage01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_tcpstage01",
            lease_epoch=1,
            lease_expires_at=time.time() + 5,
        )

        result = provider.execute(
            attempt, reservation, threading.Event(),
        )

        assert result.output == {"content": "tcp remote result"}
        assert result.attempt_id == "att_tcpstage01"
        provider.release(reservation.reservation_id)
    finally:
        client.disconnect()
        server.stop()


def test_scheduler_binds_full_worker_hello_to_registered_pc_client(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, node_id, data, message_type: sent.append(
            (node_id, data, message_type)
        ),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    assert sent[0][0] == "worker_01"
    assert sent[0][2] == MessageType.TASK_WORKER
    assert sent[0][1]["message_type"] == "hello_ack"
    assert sent[0][1]["payload"]["accepted"] is True
    assert scheduler.get_task_worker_protocol_status()[
        "connected_worker_count"
    ] == 1


def test_scheduler_experimental_gate_reports_physical_validation_pending(
    monkeypatch,
):
    import scheduler as scheduler_mod
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True,
    )
    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["worker_01"] = NodeInfo(
        node_id="worker_01",
        role=NodeRole.CLIENT,
        node_type="pc",
        state=NodeState.ONLINE,
    )
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, *args: None,
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    scheduler._handle_task_worker_message(
        "worker_01", {"data": hello.snapshot()},
    )

    status = scheduler.get_task_worker_protocol_status()
    assert status["experiment_enabled"] is True
    assert status["experimental_dispatch_enabled"] is True
    assert status["adapter_connected"] is False
    assert status["task_dispatch_enabled"] is False
    assert status["admission_state"] == (
        "n2_4_experimental_physical_validation_pending"
    )


def test_scheduler_rejects_android_client_full_worker_claim(monkeypatch):
    from scheduler import NodeInfo, NodeRole, NodeState, Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.nodes["android_01"] = NodeInfo(
        node_id="android_01",
        role=NodeRole.CLIENT,
        node_type="android",
        state=NodeState.ONLINE,
    )
    sent = []
    scheduler._tcp_server = type("Server", (), {
        "send_to_client": lambda self, *args: sent.append(args),
    })()
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "master")
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(
        node_id="android_01", capabilities=_capabilities(),
    )
    assert hello is not None

    scheduler._handle_task_worker_message(
        "android_01", {"data": hello.snapshot()},
    )

    assert sent == []
    status = scheduler.get_task_worker_protocol_status()
    assert status["connected_worker_count"] == 0
    assert status["rejected_message_count"] == 1


def test_scheduler_does_not_advertise_a_layer_partition_as_full_model(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    partition_manager = SimpleNamespace(
        is_loaded=True,
        layer_range=(0, 12),
    )
    fake_api = SimpleNamespace(
        model_loaded=True,
        model_manager=SimpleNamespace(_instance=partition_manager),
        _active_task_graph_model_identity=lambda: pytest.fail(
            "a layer partition must not compute or advertise full model identity"
        ),
    )
    monkeypatch.setitem(sys.modules, "api_server", fake_api)

    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["models"] == []
    assert capabilities["stage_types"] == ["full_inference", "aggregate"]


def test_scheduler_worker_gate_blocks_hello_and_rejects_stale_offer(
    monkeypatch,
):
    import scheduler as scheduler_mod
    from scheduler import Scheduler

    monkeypatch.setattr(
        scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", False,
    )
    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    sent = []

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))

    scheduler._tcp_client = Client()
    assert scheduler._send_task_worker_hello() is False
    assert sent == []

    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=lambda *args: pytest.fail(
            "a disabled worker must not execute a Stage"
        ),
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_gateworker01",
            "request_id": "request-gate-worker",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_gateworker01",
            "lease_id": "lease_gateworker01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_gateworkeroffer01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert _wait_until(lambda: bool(sent))
    response = sent[-1][0]
    assert response["message_type"] == "stage_accept"
    assert response["payload"]["accepted"] is False
    assert response["payload"]["reason_code"] == "worker_experiment_disabled"
    assert response["payload"]["retryable"] is True
    assert scheduler._task_worker_active_attempts == {}


def test_scheduler_worker_executes_one_admitted_stage_and_returns_result(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator_control = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator_control.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())

    sent = []
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data.get("message_type") in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()
    fake_api = SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=lambda request, cancel_event: {
            "content": f"remote:{request.stage_id}",
            "usage": {"total_tokens": 3},
            "model": "qwen-1_8b",
        },
    )
    monkeypatch.setitem(sys.modules, "api_server", fake_api)
    root_input = {"message": "hello", "messages": []}
    dependencies = {}
    now_ms = int(time.time() * 1000)
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workerexec01",
            "request_id": "request-worker-exec",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workerexec01",
            "lease_id": "lease_workerexec01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": dependencies,
            "input_sha256": stage_input_sha256(root_input, dependencies),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workerexec01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message(
        "master", {"data": offer.snapshot()},
    )

    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[0][0]["payload"]["accepted"] is True
    assert sent[1][0]["payload"]["output"]["content"] == (
        "remote:candidate_a"
    )
    assert sent[1][0]["payload"]["attempt_id"] == "att_workerexec01"


def test_scheduler_worker_rechecks_model_identity_when_offer_arrives(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    initial_capabilities = _capabilities()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=initial_capabilities,
    )
    assert hello is not None
    coordinator = TaskWorkerControlPlane()
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    changed_capabilities = _capabilities()
    changed_capabilities["models"] = []
    monkeypatch.setattr(
        scheduler, "_task_worker_capabilities", lambda: changed_capabilities,
    )
    sent = []
    rejected = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            rejected.set()

    scheduler._tcp_client = Client()
    root_input = {"message": "hello", "messages": []}
    now_ms = int(time.time() * 1000)
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_modelcheck01",
            "request_id": "request-model-check",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_modelcheck01",
            "lease_id": "lease_modelcheck01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_modelcheck01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message(
        "master", {"data": offer.snapshot()},
    )

    assert rejected.wait(2)
    assert len(sent) == 1
    assert sent[0][0]["message_type"] == "stage_accept"
    assert sent[0][0]["payload"]["accepted"] is False
    assert sent[0][0]["payload"]["reason_code"] == (
        "model_identity_mismatch"
    )


def test_remote_provider_renews_lease_and_finishes_after_original_deadline():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    original_deadline = time.time() + 0.2
    attempt = StageAttempt(
        attempt_id="att_renewremote01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_renewremote01",
        lease_epoch=1,
        lease_expires_at=original_deadline,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            outcome["result"] = provider.execute(
                attempt, reservation, threading.Event(),
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    deadline = time.time() + 2
    while not sent and time.time() < deadline:
        time.sleep(0.01)
    offer = sent[0].payload
    identity = _response_identity(offer)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_renewaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    renewed_deadline = time.time() + 1.0
    assert provider.renew_lease(
        attempt.attempt_id,
        attempt.lease_id,
        attempt.lease_epoch,
        renewed_deadline,
    ) is True
    assert _wait_until(
        lambda: any(message.message_type == "lease_renew" for message in sent)
    )
    assert sent[-1].message_type == "lease_renew"
    time.sleep(max(0.0, original_deadline - time.time()) + 0.05)
    assert not finished.is_set()

    output = {"content": "after renewal"}
    provider.handle_message(build_message(
        "stage_result",
        {
            **identity,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "metadata": {},
        },
        message_id="msg_renewresult01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["result"].output == output
    provider.release(reservation.reservation_id)


def test_remote_provider_sends_cancel_and_accepts_cancel_ack():
    coordinator_control, _worker_control = _admitted_control_plane()
    sent = []
    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=sent.append,
    )
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_cancelremote01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_cancelremote01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5,
    )
    outcome = {}
    finished = threading.Event()

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=execute, daemon=True).start()
    deadline = time.time() + 2
    while not sent and time.time() < deadline:
        time.sleep(0.01)
    identity = _response_identity(sent[0].payload)
    provider.handle_message(build_message(
        "stage_accept",
        {**identity, "accepted": True, "reason_code": "", "retryable": False},
        message_id="msg_cancelaccept01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())

    provider.cancel(attempt.attempt_id)
    assert _wait_until(
        lambda: any(message.message_type == "stage_cancel" for message in sent)
    )
    assert sent[-1].message_type == "stage_cancel"
    provider.handle_message(build_message(
        "stage_cancelled",
        {**identity, "reason_code": "coordinator_cancelled"},
        message_id="msg_cancelledremote01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    ).snapshot())
    assert finished.wait(2)
    assert outcome["error"].code == "provider_cancelled"
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0


def test_remote_cancel_and_renew_do_not_block_on_network_send():
    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    outbound_block = threading.Event()
    offer_sent = threading.Event()
    sent_types = []

    def send(message):
        sent_types.append(message.message_type)
        if message.message_type == "stage_offer":
            offer_sent.set()
            identity = _response_identity(message.payload)
            provider_holder["provider"].handle_message(build_message(
                "stage_accept",
                {
                    **identity,
                    "accepted": True,
                    "reason_code": "",
                    "retryable": False,
                },
                message_id="msg_nonblockingaccept01",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            return
        outbound_block.wait(1.0)

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    provider_holder["provider"] = provider
    request = _remote_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_nonblocking01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_nonblocking01",
        lease_epoch=1,
        lease_expires_at=time.time() + 2.0,
    )
    outcome = {}

    def execute():
        try:
            provider.execute(attempt, reservation, threading.Event())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    assert offer_sent.wait(2)

    started = time.perf_counter()
    assert provider.renew_lease(
        attempt.attempt_id,
        attempt.lease_id,
        attempt.lease_epoch,
        time.time() + 3.0,
    ) is True
    renew_elapsed = time.perf_counter() - started
    assert _wait_until(lambda: "lease_renew" in sent_types)

    started = time.perf_counter()
    provider.cancel(attempt.attempt_id)
    cancel_elapsed = time.perf_counter() - started

    assert renew_elapsed < 0.1
    assert cancel_elapsed < 0.1
    thread.join(2)
    assert not thread.is_alive()
    assert outcome["error"].code == "provider_cancelled"
    outbound_block.set()
    assert _wait_until(lambda: "stage_cancel" in sent_types)
    provider.release(reservation.reservation_id)
    provider.close()


def test_task_graph_auto_renews_remote_lease_before_result():
    from task_graph import StageSpec, TaskGraphCoordinator

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    sent_types = []

    def send(message):
        sent_types.append(message.message_type)
        if message.message_type != "stage_offer":
            return
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_autorenewaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

        def finish_later():
            time.sleep(0.35)
            output = {"content": "renewed graph result"}
            provider.handle_message(build_message(
                "stage_result",
                {
                    **identity,
                    "output": output,
                    "output_sha256": canonical_sha256(output),
                    "metadata": {},
                },
                message_id="msg_autorenewresult01",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())

        threading.Thread(target=finish_later, daemon=True).start()

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=send,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)

    output, workflow = coordinator.run(
        stages=[StageSpec(
            "candidate_a",
            "full_inference",
            provider=provider.provider_id,
            fallback_providers=(),
            pure=False,
            lease_timeout_seconds=0.15,
        )],
        final_stage_id="candidate_a",
        root_input={"message": "hello", "messages": []},
        model_identity=_model_identity(),
        workflow_id="wf_autorenewremote01",
    )

    assert output == {"content": "renewed graph result"}
    assert sent_types.count("lease_renew") >= 2
    assert workflow["stages"][0]["attempts"][0]["state"] == "completed"
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_task_graph_disconnect_and_accept_timeout_release_remote_slot():
    from task_graph import StageSpec, TaskGraphCoordinator, WorkflowExecutionError

    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}
    offered = threading.Event()

    def accept_only(message):
        if message.message_type != "stage_offer":
            return
        identity = _response_identity(message.payload)
        provider_holder["provider"].handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id="msg_disconnectaccept01",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        offered.set()

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=accept_only,
    )
    provider_holder["provider"] = provider
    coordinator = TaskGraphCoordinator()
    coordinator.register_provider(provider)
    errors = []

    def run_disconnected():
        try:
            coordinator.run(
                stages=[StageSpec(
                    "candidate_a",
                    "full_inference",
                    provider=provider.provider_id,
                    fallback_providers=(),
                    pure=False,
                    lease_timeout_seconds=1.0,
                )],
                final_stage_id="candidate_a",
                root_input={"message": "hello"},
                model_identity=_model_identity(),
                workflow_id="wf_disconnectremote01",
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_disconnected, daemon=True)
    thread.start()
    assert offered.wait(2)
    provider.notify_disconnect()
    thread.join(2)
    assert not thread.is_alive()
    assert isinstance(errors[0], WorkflowExecutionError)
    assert provider.inspect().active_reservations == 0
    coordinator.close()

    timeout_control, _ = _admitted_control_plane()
    timeout_sent = []
    timeout_provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: timeout_control.worker_snapshot("worker_01"),
        send_message=timeout_sent.append,
    )
    timeout_coordinator = TaskGraphCoordinator()
    timeout_coordinator.register_provider(timeout_provider)
    with pytest.raises(WorkflowExecutionError):
        timeout_coordinator.run(
            stages=[StageSpec(
                "candidate_a",
                "full_inference",
                provider=timeout_provider.provider_id,
                fallback_providers=(),
                pure=False,
                accept_timeout_seconds=0.1,
                lease_timeout_seconds=1.0,
            )],
            final_stage_id="candidate_a",
            root_input={"message": "hello"},
            model_identity=_model_identity(),
            workflow_id="wf_accepttimeout01",
        )
    assert _wait_until(
        lambda: any(
            message.message_type == "stage_cancel"
            for message in timeout_sent
        )
    )
    assert [message.message_type for message in timeout_sent] == [
        "stage_offer", "stage_cancel",
    ]
    assert timeout_provider.inspect().active_reservations == 0
    timeout_coordinator.close()


def test_scheduler_worker_replays_duplicate_offer_without_second_execution(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    completed = threading.Event()
    calls = []

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if len(sent) >= 2 and data["message_type"] == "stage_result":
                completed.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        calls.append(request.stage_id)
        return {"content": "once"}

    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute,
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_replayoffer01",
            "request_id": "request-replay-offer",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_replayoffer01",
            "lease_id": "lease_replayoffer01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_replayoffer01",
        sent_at_ms=now_ms,
        version=2,
    )

    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert completed.wait(2)
    first_responses = [item[0] for item in sent]
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert calls == ["candidate_a"]
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result", "stage_accept", "stage_result",
    ]
    assert sent[2][0] == first_responses[0]
    assert sent[3][0] == first_responses[1]


def test_scheduler_worker_lease_renew_extends_active_execution(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    completed = threading.Event()
    release_execution = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        while not release_execution.wait(0.01):
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
        return {"content": "renewed worker result"}

    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute,
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workerrenew01",
            "request_id": "request-worker-renew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workerrenew01",
            "lease_id": "lease_workerrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 250,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workerrenewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)

    renew_now_ms = int(time.time() * 1000)
    renewal = build_message(
        "lease_renew",
        {
            "workflow_id": "wf_workerrenew01",
            "stage_id": "candidate_a",
            "attempt_id": "att_workerrenew01",
            "lease_id": "lease_workerrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": renew_now_ms + 1000,
        },
        message_id="msg_workerrenew01",
        sent_at_ms=renew_now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message(
        "master", {"data": renewal.snapshot()},
    )
    time.sleep(max(0.0, (now_ms + 300) / 1000.0 - time.time()))
    assert not completed.is_set()

    release_execution.set()
    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[-1][0]["payload"]["output"]["content"] == (
        "renewed worker result"
    )


def test_scheduler_worker_cancel_ack_is_replayed_without_stage_error(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    cancelled = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] == "stage_cancelled":
                cancelled.set()

    scheduler._tcp_client = Client()

    def execute(request, cancel_event):
        assert cancel_event.wait(2)
        return {"content": "must not be committed"}

    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute,
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_workercancel01",
            "request_id": "request-worker-cancel",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_workercancel01",
            "lease_id": "lease_workercancel01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_workercanceloffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)

    cancel = build_message(
        "stage_cancel",
        {
            "workflow_id": "wf_workercancel01",
            "stage_id": "candidate_a",
            "attempt_id": "att_workercancel01",
            "lease_id": "lease_workercancel01",
            "lease_epoch": 1,
            "reason_code": "coordinator_cancelled",
        },
        message_id="msg_workercancel01",
        sent_at_ms=int(time.time() * 1000),
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": cancel.snapshot()})
    assert cancelled.wait(2)
    deadline = time.time() + 2
    while scheduler._task_worker_active_attempts and time.time() < deadline:
        time.sleep(0.01)
    scheduler._handle_task_worker_message("master", {"data": cancel.snapshot()})

    message_types = [item[0]["message_type"] for item in sent]
    assert message_types == [
        "stage_accept", "stage_cancelled", "stage_cancelled",
    ]
    assert "stage_error" not in message_types
    assert sent[1][0] == sent[2][0]


def test_scheduler_worker_replays_cached_result_after_send_failure(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    first_sent = []
    result_failed = threading.Event()
    calls = []

    class FailingResultClient:
        is_registered = True

        def send_data(self, data, message_type):
            first_sent.append((data, message_type))
            if data["message_type"] == "stage_result":
                result_failed.set()
                raise ConnectionError("result transport failed")

    scheduler._tcp_client = FailingResultClient()

    def execute(request, cancel_event):
        calls.append(request.stage_id)
        return {"content": "cached result"}

    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute,
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_resultretry01",
            "request_id": "request-result-retry",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_resultretry01",
            "lease_id": "lease_resultretry01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 5000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_resultretryoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert result_failed.wait(2)
    deadline = time.time() + 2
    while scheduler._task_worker_active_attempts and time.time() < deadline:
        time.sleep(0.01)

    replayed = []

    class ReconnectedClient:
        is_registered = True

        def send_data(self, data, message_type):
            replayed.append((data, message_type))

    scheduler._tcp_client = ReconnectedClient()
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert calls == ["candidate_a"]
    assert [item[0]["message_type"] for item in replayed] == [
        "stage_accept", "stage_result",
    ]
    assert replayed[-1][0]["payload"]["output"] == {
        "content": "cached result"
    }


def test_scheduler_worker_does_not_resurrect_an_expired_lease(monkeypatch):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    accepted = threading.Event()
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] == "stage_accept":
                accepted.set()
            if data["message_type"] == "stage_error":
                completed.set()

    scheduler._tcp_client = Client()
    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=lambda request, cancel_event: (
            time.sleep(0.3) or {"content": "too late"}
        ),
    ))
    now_ms = int(time.time() * 1000)
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_expiredrenew01",
            "request_id": "request-expired-renew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_expiredrenew01",
            "lease_id": "lease_expiredrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 100,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_expiredrenewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})
    assert accepted.wait(2)
    time.sleep(0.15)
    renew_now_ms = int(time.time() * 1000)
    renewal = build_message(
        "lease_renew",
        {
            "workflow_id": "wf_expiredrenew01",
            "stage_id": "candidate_a",
            "attempt_id": "att_expiredrenew01",
            "lease_id": "lease_expiredrenew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": renew_now_ms + 1000,
        },
        message_id="msg_expiredrenew01",
        sent_at_ms=renew_now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message(
        "master", {"data": renewal.snapshot()},
    )

    assert completed.wait(2)
    assert sent[-1][0]["message_type"] == "stage_error"
    assert sent[-1][0]["payload"]["error_code"] == "lease_expired"


def test_scheduler_worker_lease_uses_duration_across_wall_clock_skew(
    monkeypatch,
):
    from scheduler import Scheduler

    scheduler = Scheduler()
    scheduler._role_override = "client"
    monkeypatch.setattr(scheduler, "get_effective_node_id", lambda: "worker_01")
    monkeypatch.setattr(scheduler, "_task_worker_capabilities", _capabilities)
    coordinator = TaskWorkerControlPlane()
    hello = scheduler._task_worker_control.begin_worker_hello(
        node_id="worker_01", capabilities=_capabilities(),
    )
    assert hello is not None
    ack = coordinator.receive_on_coordinator(
        "worker_01", hello.snapshot(), coordinator_node_id="master",
    )
    scheduler._task_worker_control.receive_on_worker(ack.snapshot())
    sent = []
    completed = threading.Event()

    class Client:
        is_registered = True

        def send_data(self, data, message_type):
            sent.append((data, message_type))
            if data["message_type"] in {"stage_result", "stage_error"}:
                completed.set()

    scheduler._tcp_client = Client()
    monkeypatch.setitem(sys.modules, "api_server", SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=lambda request, cancel_event: {
            "content": "clock independent"
        },
    ))
    now_ms = 1000
    root_input = {"message": "hello"}
    offer = build_message(
        "stage_offer",
        {
            "workflow_id": "wf_clockskew01",
            "request_id": "request-clock-skew",
            "stage_id": "candidate_a",
            "stage_type": "full_inference",
            "attempt_id": "att_clockskew01",
            "lease_id": "lease_clockskew01",
            "lease_epoch": 1,
            "lease_expires_at_ms": now_ms + 1000,
            "provider_id": remote_provider_id("worker_01"),
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}),
            "model_identity": _model_identity().snapshot(),
        },
        message_id="msg_clockskewoffer01",
        sent_at_ms=now_ms,
        version=2,
    )
    scheduler._handle_task_worker_message("master", {"data": offer.snapshot()})

    assert completed.wait(2)
    assert [item[0]["message_type"] for item in sent] == [
        "stage_accept", "stage_result",
    ]
    assert sent[-1][0]["payload"]["output"] == {
        "content": "clock independent"
    }


def test_full_worker_stage_round_trips_through_independent_process(
    monkeypatch,
):
    import config as cfg
    from task_graph import StageSpec, TaskGraphCoordinator

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    secret = "n2-independent-process-secret"
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", secret)
    control = TaskWorkerControlPlane()
    identity = ModelIdentity(
        model_id="qwen-1_8b",
        engine="pytorch",
        format="safetensors",
        revision="process-test",
        sha256="b" * 64,
    )
    hello_received = threading.Event()
    provider_holder = {}
    server = TCPServer(host="127.0.0.1", port=port)

    def on_server_message(client_id, outer):
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer["data"]
        if inner["message_type"] == "hello":
            ack = control.receive_on_coordinator(
                client_id,
                inner,
                coordinator_node_id="master",
            )
            server.send_to_client(
                client_id, ack.snapshot(), MessageType.TASK_WORKER,
            )
            hello_received.set()
            return
        provider_holder["provider"].handle_message(inner)

    provider = RemoteFullWorkerProvider(
        node_id="worker_process_01",
        peer_snapshot=lambda: control.worker_snapshot("worker_process_01"),
        send_message=lambda message: server.send_to_client(
            "worker_process_01",
            message.snapshot(),
            MessageType.TASK_WORKER,
        ),
    )
    provider_holder["provider"] = provider
    helper = os.path.join(
        os.path.dirname(__file__), "helpers", "task_worker_process.py",
    )
    env = dict(os.environ)
    env.update({
        "QLH_CLUSTER_SECRET": secret,
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "true",
        "PYTHONUTF8": "1",
    })
    process = None
    coordinator = TaskGraphCoordinator()
    try:
        server.start(on_message=on_server_message)
        process = subprocess.Popen(
            [
                sys.executable,
                helper,
                "--port", str(port),
                "--node-id", "worker_process_01",
                "--model-id", identity.model_id,
                "--model-sha256", identity.sha256,
            ],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert hello_received.wait(8.0)
        coordinator.register_provider(provider)
        output, workflow = coordinator.run(
            stages=[StageSpec(
                "candidate_a",
                "full_inference",
                provider=provider.provider_id,
                fallback_providers=(),
                pure=False,
                lease_timeout_seconds=5.0,
            )],
            final_stage_id="candidate_a",
            root_input={"message": "independent process"},
            model_identity=identity,
            workflow_id="wf_processworker01",
        )

        assert output["content"] == "process:candidate_a"
        assert output["worker_pid"] != os.getpid()
        assert output["worker_node_id"] == "worker_process_01"
        assert workflow["stages"][0]["attempts"][0][
            "provider_node_id"
        ] == "worker_process_01"
        assert process.wait(timeout=10) == 0
        assert provider.inspect().active_reservations == 0
    finally:
        coordinator.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server.stop()


def test_remote_worker_releases_slot_across_success_and_failure_cycles():
    coordinator_control, _worker_control = _admitted_control_plane()
    provider_holder = {}

    def respond(message):
        if message.message_type != "stage_offer":
            return
        offer = message.payload
        identity = _response_identity(offer)
        provider = provider_holder["provider"]
        suffix = offer["attempt_id"].removeprefix("att_stress")
        provider.handle_message(build_message(
            "stage_accept",
            {
                **identity,
                "accepted": True,
                "reason_code": "",
                "retryable": False,
            },
            message_id=f"msg_stressaccept_{suffix}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())
        if int(suffix) % 5 == 0:
            provider.handle_message(build_message(
                "stage_error",
                {
                    **identity,
                    "error_code": "remote_worker_disconnected",
                    "retryable": True,
                },
                message_id=f"msg_stresserror_{suffix}",
                sent_at_ms=int(time.time() * 1000),
                version=2,
            ).snapshot())
            return
        output = {"content": f"stress:{suffix}"}
        provider.handle_message(build_message(
            "stage_result",
            {
                **identity,
                "output": output,
                "output_sha256": canonical_sha256(output),
                "metadata": {},
            },
            message_id=f"msg_stressresult_{suffix}",
            sent_at_ms=int(time.time() * 1000),
            version=2,
        ).snapshot())

    provider = RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: coordinator_control.worker_snapshot("worker_01"),
        send_message=respond,
    )
    provider_holder["provider"] = provider

    for index in range(1, 31):
        suffix = f"{index:08d}"
        request = StageRequest(
            workflow_id=f"wf_stress{suffix}",
            request_id=f"request-stress-{suffix}",
            stage_id="candidate_a",
            stage_type="full_inference",
            provider_id=provider.provider_id,
            dependencies={},
            root_input={"message": "stress"},
            model_identity=_model_identity(),
        )
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id=f"att_stress{suffix}",
            request=request,
            provider_id=provider.provider_id,
            lease_id=f"lease_stress{suffix}",
            lease_epoch=1,
            lease_expires_at=time.time() + 2,
        )
        try:
            if index % 5 == 0:
                with pytest.raises(ProviderExecutionError) as captured:
                    provider.execute(
                        attempt, reservation, threading.Event(),
                    )
                assert captured.value.code == "remote_worker_disconnected"
            else:
                result = provider.execute(
                    attempt, reservation, threading.Event(),
                )
                assert result.output == {"content": f"stress:{suffix}"}
        finally:
            provider.release(reservation.reservation_id)
        assert provider.inspect().active_reservations == 0
