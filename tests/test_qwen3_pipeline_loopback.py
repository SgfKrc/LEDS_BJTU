from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, "src")

import tcp_comm as tcp_comm_mod  # noqa: E402
import api_server  # noqa: E402
import pipeline_assignment_manifest  # noqa: E402
from qwen3_pipeline_loopback import (  # noqa: E402
    Qwen3LoopbackError,
    Qwen3PipelineLoopbackWorker,
    build_safetensors_header_probe,
    fetch_assignment_probe,
    sign_loopback_message,
    verify_loopback_message,
)
from qwen3_pipeline_transaction import (  # noqa: E402
    Qwen3PipelineDryRunTransaction,
    build_qwen3_dry_run_contract,
)
from scheduler import Scheduler  # noqa: E402
from tcp_comm import MessageType, TCPClient, TCPServer  # noqa: E402


SECRET = "qwen3-loopback-test-secret"


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"


def _probe(payload: bytes, relative_path: str = "model.safetensors") -> dict:
    header_length = int.from_bytes(payload[:8], "little")
    length = 8 + header_length
    return {
        "relative_path": relative_path,
        "file_size": len(payload),
        "offset": 0,
        "length": length,
        "sha256": hashlib.sha256(payload[:length]).hexdigest(),
    }


def _assignment_manifest(
    node_id: str,
    layer_range: tuple[int, int] | list[int],
    *,
    has_embedding: bool,
    has_lm_head: bool,
) -> dict:
    manifest = {
        "schema_version": 1,
        "manifest_kind": "pytorch_pipeline_assignment",
        "model_id": "qwen3-4b",
        "model_sha256": "a" * 64,
        "model_type": "qwen3",
        "total_layers": 4,
        "config_id": "cfg-loopback",
        "plan_id": "plan-loopback",
        "node_id": node_id,
        "layer_range": list(layer_range),
        "has_embedding": has_embedding,
        "has_lm_head": has_lm_head,
        "files": [{"path": "model.safetensors", "kind": "weights"}],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return manifest


def _contract(
    payload: bytes,
    node_ids=("worker-a", "worker-b"),
    *,
    execution_mode: str = "metadata_only",
) -> dict:
    probe = _probe(payload)
    segments = []
    layer_ranges = [(0, 2), (2, 4)] if len(node_ids) == 2 else [(0, 1), (1, 3), (3, 4)]
    for index, (node_id, layer_range) in enumerate(zip(node_ids, layer_ranges)):
        manifest = _assignment_manifest(
            node_id,
            layer_range,
            has_embedding=index == 0,
            has_lm_head=index == len(node_ids) - 1,
        )
        segments.append({
            "node_id": node_id,
            "layer_range": list(layer_range),
            "has_embedding": index == 0,
            "has_lm_head": index == len(node_ids) - 1,
            "required_bytes": 32,
            "assignment_manifest_sha256": manifest["manifest_sha256"],
            "execution_device": "cpu",
            "dtype": "float32",
            "assignment_probe": dict(probe),
        })
    return build_qwen3_dry_run_contract(
        config_id="cfg-loopback",
        plan_id="plan-loopback",
        generation=9,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        total_layers=4,
        hidden_size=16,
        segments=segments,
        execution_mode=execution_mode,
    )


class _RangeServer:
    def __init__(self, payload: bytes, *, mode: str = "normal") -> None:
        self.payload = payload
        self.mode = mode
        self.ranges: list[str | None] = []
        self.manifest_requests: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path.startswith("/api/models/pipeline-assignment/"):
                    owner.manifest_requests.append(self.path)
                    if owner.mode == "manifest_403":
                        self.send_error(403)
                        return
                    query = parse_qs(parsed.query)
                    layer_range = (
                        int(query["start_layer"][0]),
                        int(query["end_layer"][0]),
                    )
                    manifest = _assignment_manifest(
                            query["node_id"][0],
                            layer_range,
                            has_embedding=bool(int(query["has_embedding"][0])),
                            has_lm_head=bool(int(query["has_lm_head"][0])),
                        )
                    if owner.mode == "manifest_tamper":
                        manifest["total_layers"] = 5
                    body = json.dumps(
                        manifest,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                value = self.headers.get("Range")
                owner.ranges.append(value)
                if owner.mode in {"401", "403", "416"}:
                    self.send_error(int(owner.mode))
                    return
                if owner.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://example.invalid/model")
                    self.end_headers()
                    return
                if owner.mode == "timeout":
                    time.sleep(0.15)
                start, end = 0, len(owner.payload) - 1
                if value and value.startswith("bytes="):
                    raw_start, raw_end = value[6:].split("-", 1)
                    start = int(raw_start)
                    if raw_end:
                        end = int(raw_end)
                body = owner.payload[start:end + 1]
                self.send_response(206)
                range_start = start + 1 if owner.mode == "wrong_range" else start
                self.send_header(
                    "Content-Range",
                    f"bytes {range_start}-{end}/{len(owner.payload)}",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if owner.mode == "truncate_once" and len(owner.ranges) == 1:
                    body = body[: max(1, len(body) // 2)]
                    self.close_connection = True
                elif owner.mode == "truncate_always":
                    body = body[:1]
                    self.close_connection = True
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _unsigned(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "transport_auth"}


def test_safetensors_probe_reads_header_only(tmp_path: Path):
    payload = _safetensors_bytes()
    model = tmp_path / "model.safetensors"
    model.write_bytes(payload)

    probe = build_safetensors_header_probe(model, relative_path=model.name)

    assert probe == _probe(payload)
    assert probe["length"] < probe["file_size"]


def test_real_qwen3_shard_header_only_range_loopback():
    shard = Path("models/qwen3-4b/model-00001-of-00003.safetensors")
    if not shard.is_file():
        pytest.skip("local Qwen3-4B Safetensors shard is unavailable")
    probe = build_safetensors_header_probe(shard, relative_path=shard.name)
    requests: list[tuple[int, int]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = self.headers.get("Range", "")
            raw_start, raw_end = value[6:].split("-", 1)
            start, end = int(raw_start), int(raw_end)
            requests.append((start, end))
            with shard.open("rb") as handle:
                handle.seek(start)
                body = handle.read(end - start + 1)
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{shard.stat().st_size}",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = fetch_assignment_probe(
            f"http://127.0.0.1:{server.server_port}",
            "qwen3-4b",
            probe,
            timeout_seconds=3,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert report["sha256"] == probe["sha256"]
    assert report["bytes_received"] == probe["length"]
    assert requests == [(0, probe["length"] - 1)]
    assert probe["length"] < shard.stat().st_size // 100


def test_hmac_binds_payload_peer_time_and_nonce():
    message = {
        "phase": "prepare", "contract_sha256": "a" * 64,
        "generation": 1, "node_id": "worker-a",
    }
    signed = sign_loopback_message(
        message, peer_node_id="worker-a", secret=SECRET,
        now=1000, nonce="1" * 32,
    )
    assert verify_loopback_message(
        signed, authenticated_peer_id="worker-a", secret=SECRET, now=1001,
    )[0] == "1" * 32

    tampered = copy.deepcopy(signed)
    tampered["generation"] = 2
    with pytest.raises(Qwen3LoopbackError) as exc:
        verify_loopback_message(
            tampered, authenticated_peer_id="worker-a", secret=SECRET, now=1001,
        )
    assert exc.value.reason_code == "qwen3_loopback_auth_mismatch"
    with pytest.raises(Qwen3LoopbackError) as exc:
        verify_loopback_message(
            signed, authenticated_peer_id="worker-b", secret=SECRET, now=1001,
        )
    assert exc.value.reason_code == "qwen3_loopback_peer_mismatch"
    with pytest.raises(Qwen3LoopbackError) as exc:
        verify_loopback_message(
            signed, authenticated_peer_id="worker-a", secret=SECRET, now=1400,
        )
    assert exc.value.reason_code == "qwen3_loopback_auth_stale"


def test_range_probe_resumes_after_truncated_response():
    payload = _safetensors_bytes()
    with _RangeServer(payload, mode="truncate_once") as server:
        report = fetch_assignment_probe(
            server.base_url, "qwen3-4b", _probe(payload), timeout_seconds=1,
        )

    assert report["sha256"] == _probe(payload)["sha256"]
    assert report["attempts"] == 2
    assert server.ranges[0] == f"bytes=0-{_probe(payload)['length'] - 1}"
    assert server.ranges[1].startswith("bytes=")
    assert server.ranges[1] != server.ranges[0]


@pytest.mark.parametrize(
    ("mode", "reason_code"),
    [
        ("401", "qwen3_range_auth_required"),
        ("403", "qwen3_range_forbidden"),
        ("416", "qwen3_range_unsatisfiable"),
        ("redirect", "qwen3_range_http_error"),
        ("wrong_range", "qwen3_range_header_mismatch"),
        ("truncate_always", "qwen3_range_truncated"),
        ("timeout", "qwen3_range_timeout"),
    ],
)
def test_range_probe_fails_closed_for_http_faults(mode, reason_code):
    payload = _safetensors_bytes()
    with _RangeServer(payload, mode=mode) as server:
        with pytest.raises(Qwen3LoopbackError) as exc:
            fetch_assignment_probe(
                server.base_url,
                "qwen3-4b",
                _probe(payload),
                timeout_seconds=0.03 if mode == "timeout" else 1,
            )
    assert exc.value.reason_code == reason_code


def test_range_probe_rejects_sha_tamper_and_non_loopback_url():
    payload = _safetensors_bytes()
    changed = dict(_probe(payload), sha256="0" * 64)
    with _RangeServer(payload) as server:
        with pytest.raises(Qwen3LoopbackError) as exc:
            fetch_assignment_probe(server.base_url, "qwen3-4b", changed)
    assert exc.value.reason_code == "qwen3_range_sha256_mismatch"
    with pytest.raises(Qwen3LoopbackError) as exc:
        fetch_assignment_probe("http://192.0.2.1:8000", "qwen3-4b", _probe(payload))
    assert exc.value.reason_code == "qwen3_loopback_url_rejected"


def test_worker_is_idempotent_and_rejects_changed_nonce_replay():
    payload = _safetensors_bytes()
    with _RangeServer(payload) as server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload), network_dispatch=True,
        )
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-a", secret=SECRET, base_url=server.base_url,
            available_bytes=1024,
        )
        message = tx._message("worker-a", "prepare")
        message["assignment_base_url"] = server.base_url
        signed = sign_loopback_message(
            message, peer_node_id="worker-a", secret=SECRET,
            now=1000, nonce="2" * 32,
        )
        first = worker.handle(signed, now=1001)
        assert worker.handle(signed, now=1001) == first
        assert len(server.ranges) == 1

        changed = dict(message, retry_count=1)
        changed_signed = sign_loopback_message(
            changed, peer_node_id="worker-a", secret=SECRET,
            now=1000, nonce="2" * 32,
        )
        with pytest.raises(Qwen3LoopbackError) as exc:
            worker.handle(changed_signed, now=1001)
    assert exc.value.reason_code == "qwen3_loopback_replay_mismatch"


def test_loopback_worker_connects_node_local_sidecar_lifecycle():
    payload = _safetensors_bytes()

    class Session:
        def __init__(self, _message):
            self.calls: list[str] = []

        def prepare(self):
            self.calls.append("prepare")
            return {"status": "prepared", "gate_passed": True, "segment_materialized": False}

        def commit(self):
            self.calls.append("commit")
            return {"status": "committed", "gate_passed": True, "segment_materialized": True}

        def release(self):
            self.calls.append("release")
            return {"status": "released", "gate_passed": True, "cleanup_complete": True}

        def abort(self):
            self.calls.append("abort")

    sessions: list[Session] = []

    def factory(message):
        session = Session(message)
        sessions.append(session)
        return session

    with _RangeServer(payload) as server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload, execution_mode="node_local_sidecar"),
            network_dispatch=True,
        )
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-a", secret=SECRET, base_url=server.base_url,
            available_bytes=1024, sidecar_session_factory=factory,
        )
        prepare = tx._message("worker-a", "prepare")
        prepare["assignment_base_url"] = server.base_url
        prepare_ack = worker.handle(sign_loopback_message(
            prepare, peer_node_id="worker-a", secret=SECRET,
        ))
        commit = tx._message("worker-a", "commit")
        commit["assignment_base_url"] = server.base_url
        commit_ack = worker.handle(sign_loopback_message(
            commit, peer_node_id="worker-a", secret=SECRET,
        ))
        assert _unsigned(commit_ack)["segment_materialized"] is True
        release = tx._message("worker-a", "release")
        release["assignment_base_url"] = server.base_url
        release["release"] = True
        release_ack = worker.handle(sign_loopback_message(
            release, peer_node_id="worker-a", secret=SECRET,
        ))

    assert _unsigned(prepare_ack)["segment_materialized"] is False
    assert _unsigned(release_ack)["status"] == "released"
    assert sessions[0].calls == ["prepare", "commit", "release"]


def test_worker_rejects_signed_assignment_base_url_change():
    payload = _safetensors_bytes()
    with _RangeServer(payload) as server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload), network_dispatch=True,
        )
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-a", secret=SECRET, base_url=server.base_url,
            available_bytes=1024,
        )
        message = tx._message("worker-a", "prepare")
        message["assignment_base_url"] = "http://127.0.0.1:9"
        with pytest.raises(Qwen3LoopbackError) as exc:
            worker.handle(sign_loopback_message(
                message, peer_node_id="worker-a", secret=SECRET,
            ))

    assert exc.value.reason_code == "qwen3_loopback_contract_mismatch"
    assert server.manifest_requests == []
    assert server.ranges == []


def test_worker_accepts_authenticated_stateless_release_after_early_disconnect():
    payload = _safetensors_bytes()
    with _RangeServer(payload) as server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload), network_dispatch=True,
        )
        tx.abort("fixture_disconnect")
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-b", secret=SECRET, base_url=server.base_url,
            available_bytes=0,
        )
        release = next(
            item for item in tx.release_messages()
            if item["node_id"] == "worker-b"
        )
        release["assignment_base_url"] = server.base_url
        ack = worker.handle(sign_loopback_message(
            release, peer_node_id="worker-b", secret=SECRET,
        ))

    verify_loopback_message(
        ack, authenticated_peer_id="worker-b", secret=SECRET,
    )
    assert _unsigned(ack)["status"] == "released"
    assert server.ranges == []


@pytest.mark.parametrize(
    ("mode", "reason_code"),
    [
        ("manifest_403", "qwen3_manifest_forbidden"),
        ("manifest_tamper", "qwen3_manifest_digest_mismatch"),
    ],
)
def test_worker_rejects_manifest_fault_before_range_read(mode, reason_code):
    payload = _safetensors_bytes()
    with _RangeServer(payload, mode=mode) as server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload), network_dispatch=True,
        )
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-a", secret=SECRET, base_url=server.base_url,
            available_bytes=1024,
        )
        message = tx._message("worker-a", "prepare")
        message["assignment_base_url"] = server.base_url
        with pytest.raises(Qwen3LoopbackError) as exc:
            worker.handle(sign_loopback_message(
                message, peer_node_id="worker-a", secret=SECRET,
            ))
    assert exc.value.reason_code == reason_code
    assert server.ranges == []


def test_assignment_api_allows_only_active_qwen3_loopback_segment(monkeypatch, tmp_path):
    payload = _safetensors_bytes()
    contract = _contract(payload)
    tx = Qwen3PipelineDryRunTransaction(contract, network_dispatch=True)
    segment = tx.contract["segments"][0]
    manifest = _assignment_manifest(
        "worker-a", (0, 2), has_embedding=True, has_lm_head=False,
    )
    monkeypatch.setattr(api_server, "_require_trusted_model_peer", lambda _request: None)
    monkeypatch.setattr(api_server, "_active_pytorch_model", lambda: {
        "model_id": "qwen3-4b", "model_path": str(tmp_path),
        "model_sha256": "a" * 64, "total_layers": 4,
    })
    monkeypatch.setattr(api_server.scheduler, "_pipeline_load_transaction", None)
    monkeypatch.setattr(api_server.scheduler, "_qwen3_pipeline_dry_run", tx)
    monkeypatch.setattr(
        pipeline_assignment_manifest,
        "build_assignment_manifest",
        lambda *_args, **_kwargs: dict(manifest),
    )
    client = TestClient(api_server.app)
    query = (
        "config_id=cfg-loopback&plan_id=plan-loopback&node_id=worker-a"
        "&start_layer=0&end_layer=2&total_layers=4"
        "&has_embedding=1&has_lm_head=0"
    )

    accepted = client.get(
        f"/api/models/pipeline-assignment/qwen3-4b?{query}",
    )
    wrong_node = client.get(
        f"/api/models/pipeline-assignment/qwen3-4b?{query.replace('worker-a', 'worker-x')}",
    )
    wrong_range = client.get(
        f"/api/models/pipeline-assignment/qwen3-4b?{query.replace('end_layer=2', 'end_layer=3')}",
    )

    assert accepted.status_code == 200
    assert accepted.json()["manifest_sha256"] == segment["assignment_manifest_sha256"]
    assert wrong_node.status_code == 409
    assert wrong_range.status_code == 409


def test_assignment_api_rejects_malformed_active_segment(monkeypatch, tmp_path):
    payload = _safetensors_bytes()
    contract = _contract(payload)
    contract["segments"][0]["layer_range"] = [0]
    tx = type("MalformedTransaction", (), {
        "contract": contract,
        "network_dispatch": True,
        "phase": "preparing",
    })()
    monkeypatch.setattr(api_server, "_require_trusted_model_peer", lambda _request: None)
    monkeypatch.setattr(api_server, "_active_pytorch_model", lambda: {
        "model_id": "qwen3-4b", "model_path": str(tmp_path),
        "model_sha256": "a" * 64, "total_layers": 4,
    })
    monkeypatch.setattr(api_server.scheduler, "_pipeline_load_transaction", None)
    monkeypatch.setattr(api_server.scheduler, "_qwen3_pipeline_dry_run", tx)

    response = TestClient(api_server.app).get(
        "/api/models/pipeline-assignment/qwen3-4b"
        "?config_id=cfg-loopback&plan_id=plan-loopback&node_id=worker-a"
        "&start_layer=0&end_layer=2&total_layers=4"
        "&has_embedding=1&has_lm_head=0",
    )

    assert response.status_code == 409


def test_scheduler_loopback_reaches_ready_and_releases(monkeypatch):
    payload = _safetensors_bytes()
    sent: list[tuple[str, dict]] = []

    class Server:
        _running = True

        @staticmethod
        def is_authenticated_loopback_client(node_id):
            return node_id in {"worker-a", "worker-b"}

        @staticmethod
        def send_qwen3_pipeline_dry_run(node_id, message):
            sent.append((node_id, message))

    with _RangeServer(payload) as http_server:
        scheduler = Scheduler()
        scheduler._tcp_server = Server()
        monkeypatch.setattr(scheduler, "_qwen3_cluster_secret", lambda: SECRET)
        started = scheduler.begin_qwen3_pipeline_loopback(
            _contract(payload), assignment_base_url=http_server.base_url,
        )
        assert started["transaction"]["network_dispatch"] is True
        workers = {
            node_id: Qwen3PipelineLoopbackWorker(
                node_id=node_id, secret=SECRET, base_url=http_server.base_url,
                available_bytes=1024,
            )
            for node_id in ("worker-a", "worker-b")
        }

        def drain(expected_phase):
            pending = [item for item in sent if item[1]["phase"] == expected_phase]
            sent[:] = [item for item in sent if item[1]["phase"] != expected_phase]
            assert len(pending) == 2
            for node_id, message in pending:
                ack = workers[node_id].handle(message)
                scheduler._on_tcp_message(node_id, {
                    "type": MessageType.QWEN3_PIPELINE_DRY_RUN_ACK.value,
                    "data": ack,
                })

        drain("prepare")
        drain("commit")
        assert scheduler.get_qwen3_pipeline_dry_run_status()["phase"] == "ready"
        scheduler.release_qwen3_pipeline_loopback()
        drain("release")
        status = scheduler.get_qwen3_pipeline_dry_run_status()
        assert status["phase"] == "released"
        assert status["weight_materialization"] is False
        assert scheduler._qwen3_loopback_base_url == ""


def test_scheduler_refuses_unconfirmed_or_non_loopback_peer(monkeypatch):
    payload = _safetensors_bytes()

    class Server:
        _running = True

        @staticmethod
        def is_authenticated_loopback_client(_node_id):
            return False

    with _RangeServer(payload) as http_server:
        scheduler = Scheduler()
        scheduler._tcp_server = Server()
        monkeypatch.setattr(scheduler, "_qwen3_cluster_secret", lambda: SECRET)
        with pytest.raises(Qwen3LoopbackError) as exc:
            scheduler.begin_qwen3_pipeline_loopback(
                _contract(payload), assignment_base_url=http_server.base_url,
            )
    assert exc.value.reason_code == "qwen3_loopback_peer_rejected"
    assert scheduler.get_qwen3_pipeline_dry_run_status()["phase"] == "idle"


def test_network_ready_disconnect_aborts_for_release_cleanup():
    tx = Qwen3PipelineDryRunTransaction(
        _contract(_safetensors_bytes()), network_dispatch=True,
    )
    tx.phase = "ready"

    result = tx.disconnect("worker-a")

    assert result["phase"] == "aborted"
    assert len(result["outbound"]) == 2
    assert all(item["release"] is True for item in result["outbound"])


def test_scheduler_ready_disconnect_dispatches_release_to_remaining_peer(monkeypatch):
    payload = _safetensors_bytes()
    connected = {"worker-b"}
    sent: list[tuple[str, str]] = []

    class Server:
        _running = True

        @staticmethod
        def is_authenticated_loopback_client(node_id):
            return node_id in connected

        @staticmethod
        def send_qwen3_pipeline_dry_run(node_id, message):
            sent.append((node_id, message["phase"]))

    scheduler = Scheduler()
    scheduler._tcp_server = Server()
    monkeypatch.setattr(scheduler, "_qwen3_cluster_secret", lambda: SECRET)
    tx = Qwen3PipelineDryRunTransaction(
        _contract(payload), network_dispatch=True,
    )
    tx.phase = "ready"
    scheduler._qwen3_pipeline_dry_run = tx
    scheduler._qwen3_loopback_base_url = "http://127.0.0.1:12345"

    scheduler._on_tcp_disconnect("worker-a")

    assert scheduler.get_qwen3_pipeline_dry_run_status()["phase"] == "aborted"
    assert sent == [("worker-b", "release")]


def test_loopback_capacity_uses_matching_cpu_or_cuda_pool(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: type("Memory", (), {"available": 1234})(),
    )
    assert scheduler._qwen3_loopback_available_bytes({
        "execution_device": "cpu",
    }) == 1234

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return (5678, 9999)

    import torch
    monkeypatch.setattr(torch, "cuda", Cuda())
    assert scheduler._qwen3_loopback_available_bytes({
        "execution_device": "cuda",
    }) == 5678


def test_partial_dispatch_aborts_and_releases_connected_peer(monkeypatch):
    payload = _safetensors_bytes()
    sent: list[tuple[str, str]] = []

    class Server:
        _running = True

        @staticmethod
        def is_authenticated_loopback_client(_node_id):
            return True

        @staticmethod
        def send_qwen3_pipeline_dry_run(node_id, message):
            if node_id == "worker-b" and message["phase"] == "prepare":
                raise ConnectionError("fixture disconnect")
            sent.append((node_id, message["phase"]))

    with _RangeServer(payload) as http_server:
        scheduler = Scheduler()
        scheduler._tcp_server = Server()
        monkeypatch.setattr(scheduler, "_qwen3_cluster_secret", lambda: SECRET)
        with pytest.raises(ConnectionError):
            scheduler.begin_qwen3_pipeline_loopback(
                _contract(payload), assignment_base_url=http_server.base_url,
            )

    status = scheduler.get_qwen3_pipeline_dry_run_status()
    assert status["phase"] == "aborted"
    assert status["reason_code"] == "qwen3_loopback_dispatch_failed"
    assert ("worker-a", "release") in sent
    assert ("worker-b", "release") in sent


def test_release_retry_sends_only_unreleased_reconnected_peer(monkeypatch):
    payload = _safetensors_bytes()
    connected = {"worker-a"}
    sent: list[tuple[str, str]] = []

    class Server:
        _running = True

        @staticmethod
        def is_authenticated_loopback_client(node_id):
            return node_id in connected

        @staticmethod
        def send_qwen3_pipeline_dry_run(node_id, message):
            sent.append((node_id, message["phase"]))

    scheduler = Scheduler()
    scheduler._tcp_server = Server()
    monkeypatch.setattr(scheduler, "_qwen3_cluster_secret", lambda: SECRET)
    tx = Qwen3PipelineDryRunTransaction(
        _contract(payload), network_dispatch=True,
    )
    tx.abort("fixture_abort")
    scheduler._qwen3_pipeline_dry_run = tx
    scheduler._qwen3_loopback_base_url = "http://127.0.0.1:12345"

    first = scheduler.retry_qwen3_pipeline_loopback_release()
    assert [(node_id, message["phase"]) for node_id, message in [
        (item["transport_auth"]["peer_node_id"], item)
        for item in first["outbound"]
    ]] == [("worker-a", "release")]
    tx.release_ack("worker-a", {
        "node_id": "worker-a", "config_id": "cfg-loopback",
        "plan_id": "plan-loopback", "generation": 9,
        "contract_sha256": tx.contract["contract_sha256"],
        "phase": "release", "status": "released", "release": True,
    })
    connected.add("worker-b")
    second = scheduler.retry_qwen3_pipeline_loopback_release()
    assert len(second["outbound"]) == 1
    assert second["outbound"][0]["transport_auth"]["peer_node_id"] == "worker-b"


def test_real_tcp_hmac_loopback_round_trip(monkeypatch):
    import config as cfg

    payload = _safetensors_bytes()
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", SECRET)
    monkeypatch.setattr(
        TCPClient, "_compute_local_model_sha256", staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        TCPClient,
        "_heartbeat_loop",
        lambda self, connection_generation=None, connection_sock=None: None,
    )
    monkeypatch.setattr(tcp_comm_mod, "detect_network_type", lambda: "ethernet")
    monkeypatch.setattr(tcp_comm_mod, "detect_lan_ip", lambda: "127.0.0.1")
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe_socket.bind(("127.0.0.1", 0))
    port = probe_socket.getsockname()[1]
    probe_socket.close()
    acknowledgements: list[dict] = []
    ack_event = threading.Event()

    with _RangeServer(payload) as http_server:
        tx = Qwen3PipelineDryRunTransaction(
            _contract(payload, node_ids=("worker-a", "worker-b")),
            network_dispatch=True,
        )
        # One real TCP peer exercises the first segment. The second segment is
        # deliberately left to the scheduler matrix above.
        worker = Qwen3PipelineLoopbackWorker(
            node_id="worker-a", secret=SECRET, base_url=http_server.base_url,
            available_bytes=1024,
        )
        server = TCPServer(host="127.0.0.1", port=port)

        def on_server_message(client_id, message):
            if message.get("type") == MessageType.REGISTER.value:
                server.confirm_registration(client_id)
            elif message.get("type") == MessageType.QWEN3_PIPELINE_DRY_RUN_ACK.value:
                acknowledgements.append(message["data"])
                ack_event.set()

        server.start(on_message=on_server_message)
        client = TCPClient(
            server_host="127.0.0.1", server_port=port,
            client_id="worker-a", role="client",
            advertise_host="127.0.0.1", advertise_port=port,
        )

        def on_client_message(message):
            if message.get("type") != MessageType.QWEN3_PIPELINE_DRY_RUN.value:
                return
            ack = worker.handle(message["data"])
            client.send_data(ack, MessageType.QWEN3_PIPELINE_DRY_RUN_ACK)

        try:
            assert client.connect(on_message=on_client_message) is True
            assert server.is_authenticated_loopback_client("worker-a") is True
            prepare = tx._message("worker-a", "prepare")
            prepare["assignment_base_url"] = http_server.base_url
            server.send_qwen3_pipeline_dry_run(
                "worker-a",
                sign_loopback_message(
                    prepare, peer_node_id="worker-a", secret=SECRET,
                ),
            )
            assert ack_event.wait(3), "Qwen3 prepare ACK timed out"
            ack = acknowledgements.pop()
            verify_loopback_message(
                ack, authenticated_peer_id="worker-a", secret=SECRET,
            )
            assert _unsigned(ack)["assignment_probe"]["sha256"] == _probe(payload)["sha256"]
            assert http_server.ranges == [f"bytes=0-{_probe(payload)['length'] - 1}"]
        finally:
            client.disconnect()
            server.stop()
