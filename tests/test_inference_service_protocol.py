"""inference-svc 契约测试（微服务架构改造计划 §1.1 / §4.1）。

锁定 /v1/* 契约：端点存在性、协议字段校验、错误码、SSE 事件格式
（对齐 api_server /api/chat/stream）、KV 生命周期、张量 roundtrip。
全部测试不加载真实模型（FakeEngineHost 注入；KVHost 用真实实现，
PagedKVCache 为纯 torch 轻量对象）。

并行共存：本测试只 import inference_service 包，不触碰 api_server。
"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_service.engine_host import EngineHost
from inference_service.kv_host import KVHost
from inference_service.protocol import ChatRequest, LoadModelRequest
from inference_service.routes import router
from inference_service.tensor_transport import deserialize_tensor, serialize_tensor


class FakeEngineHost(EngineHost):
    """轻量假宿主：只覆盖对话/加载方法，取消注册表等继承真实实现。"""

    def load_model(self, engine=None, quant_type=None, use_compile=False, model_id=None):
        return {"success": True, "engine": engine or "pytorch", "model_id": model_id}

    def unload_model(self):
        return {"success": True, "message": "模型已卸载"}

    def switch_model(self, model_id, engine=None):
        return {"success": True, "model_id": model_id, "engine": engine}

    def current_model(self):
        return {"loaded": True, "engine": "pytorch", "model_id": "qwen-1.8b"}

    def chat_full(self, req):
        return {
            "response": "你好，我是 QLH。",
            "followups": [],
            "metrics": {"tokens_per_second": 42.0},
        }

    def chat_stream_events(self, req, cancel_event):
        for chunk in ("你", "好"):
            if cancel_event is not None and cancel_event.is_set():
                break
            yield {"token": chunk}
        yield {
            "done": True,
            "response": "你好",
            "followups": [],
            "metrics": {},
        }

    def forward_layers(self, layer_range, hidden, past_key_values=None, **kw):
        return hidden  # identity：验证张量 roundtrip


def make_app(engine_host=None, kv_host=None) -> FastAPI:
    app = FastAPI()
    app.state.engine_host = engine_host if engine_host is not None else FakeEngineHost()
    app.state.kv_host = kv_host if kv_host is not None else KVHost()
    app.include_router(router)
    return app


@pytest.fixture()
def client():
    return TestClient(make_app())


# ----------------------------------------------------------------------
# 1. 契约：端点存在性（§4.1 全清单）
# ----------------------------------------------------------------------
EXPECTED_ENDPOINTS = [
    ("GET", "/v1/health"),
    ("GET", "/v1/ready"),
    ("GET", "/v1/status"),
    ("POST", "/v1/models/load"),
    ("POST", "/v1/models/unload"),
    ("POST", "/v1/models/switch"),
    ("GET", "/v1/models/current"),
    ("POST", "/v1/chat"),
    ("POST", "/v1/chat/stream"),
    ("POST", "/v1/chat/cancel"),
    ("POST", "/v1/speculative/run"),
    ("POST", "/v1/layers/load"),
    ("POST", "/v1/layers/unload"),
    ("POST", "/v1/layers/forward"),
    ("POST", "/v1/layers/embedding"),
    ("POST", "/v1/layers/lm_head"),
    ("POST", "/v1/kv/init"),
    ("POST", "/v1/kv/free"),
]


def test_contract_endpoint_existence():
    app = make_app()
    paths = {r.path for r in app.routes}
    for method, path in EXPECTED_ENDPOINTS:
        assert path in paths, f"缺少端点 {method} {path}"


# ----------------------------------------------------------------------
# 2. 协议字段校验（pydantic）
# ----------------------------------------------------------------------
def test_protocol_chat_request_validation():
    with pytest.raises(ValidationError):
        ChatRequest(message="")  # min_length=1
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", max_new_tokens=0)  # ge=1
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", temperature=2.5)  # le=2.0
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", streaming_mode="bogus")  # pattern
    req = ChatRequest(message="hi")
    assert req.max_new_tokens == 1024
    assert req.streaming_mode == "full"


def test_protocol_load_model_request_defaults():
    req = LoadModelRequest()
    assert req.engine is None
    assert req.quant_type is None
    assert req.use_compile is False
    assert req.layer_range is None


def test_protocol_endpoint_422_on_missing_field(client):
    resp = client.post("/v1/chat", json={})
    assert resp.status_code == 422
    assert "detail" in resp.json()


# ----------------------------------------------------------------------
# 3. 存活/就绪/状态
# ----------------------------------------------------------------------
def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "inference-svc"


def test_ready(client):
    resp = client.get("/v1/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert {"ready", "model_loaded", "layers"} <= set(body.keys())
    assert isinstance(body["layers"], list)


def test_status_includes_kv_cache(client):
    resp = client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "kv_cache" in body
    assert body["kv_cache"]["task_count"] == 0


def test_uninitialized_host_returns_503():
    app = FastAPI()  # 不设置 app.state
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/v1/ready")
    assert resp.status_code == 503
    assert "detail" in resp.json()


# ----------------------------------------------------------------------
# 4. 对话 JSON / SSE 事件格式（对齐 api_server /api/chat/stream）
# ----------------------------------------------------------------------
def test_chat_json(client):
    resp = client.post("/v1/chat", json={"message": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "你好，我是 QLH。"
    assert "followups" in body
    assert "metrics" in body


def test_chat_stream_fast_event_format(client):
    resp = client.post(
        "/v1/chat/stream",
        json={"message": "你好", "streaming_mode": "fast"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # 纯 data 事件：data: {...}\n\n，无 event: 行
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
        else:
            assert line == "", f"SSE 中出现非 data 行: {line!r}"
    assert len(events) >= 2
    assert "token" in events[0]
    assert events[-1]["done"] is True
    assert "response" in events[-1]


def test_chat_stream_full_single_done(client):
    resp = client.post(
        "/v1/chat/stream",
        json={"message": "你好", "streaming_mode": "full"},
    )
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0]["done"] is True
    assert events[0]["response"] == "你好，我是 QLH。"


def test_chat_cancel_semantics(client):
    # 未知 generation_id → 404 JSON detail
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_nope"})
    assert resp.status_code == 404
    assert "detail" in resp.json()
    # 已注册 → 200
    host = client.app.state.engine_host
    gid, ev = host.register_generation("gen_test_1")
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_test_1"})
    assert resp.status_code == 200
    assert ev.is_set()
    host.unregister_generation(gid)


# ----------------------------------------------------------------------
# 5. KV 生命周期（真实 KVHost）
# ----------------------------------------------------------------------
def test_kv_init_reuse_free(client):
    r = client.post("/v1/kv/init", json={"task_id": "t1"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "t1"
    assert r.json()["reused"] is False

    r = client.post("/v1/kv/init", json={"task_id": "t1"})  # 幂等复用
    assert r.json()["reused"] is True

    r = client.post("/v1/kv/free", json={"task_id": "t1"})
    assert r.json()["freed"] is True

    r = client.post("/v1/kv/free", json={"task_id": "t1"})  # 已释放 → 404
    assert r.status_code == 404

    r = client.get("/v1/status")
    assert r.json()["kv_cache"]["task_count"] == 0


def test_kv_auto_task_id(client):
    r = client.post("/v1/kv/init", json={})
    assert r.status_code == 200
    assert r.json()["task_id"]


# ----------------------------------------------------------------------
# 6. 层段张量传输 roundtrip（tensor_transport）
# ----------------------------------------------------------------------
def test_tensor_transport_roundtrip():
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    blob = serialize_tensor(hidden)
    restored = deserialize_tensor(blob)
    torch.testing.assert_close(restored, hidden)


def test_layers_forward_roundtrip(client):
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")
    resp = client.post(
        "/v1/layers/forward",
        json={"layer_range": "0-12", "tensor_ref": tensor_ref},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "output_ref" in body
    out = deserialize_tensor(base64.b64decode(body["output_ref"]))
    torch.testing.assert_close(out, hidden)


def test_layers_forward_bad_tensor_ref(client):
    resp = client.post(
        "/v1/layers/forward",
        json={"layer_range": "0-12", "tensor_ref": "!!!not-base64!!!"},
    )
    assert resp.status_code in (400, 500)  # base64 失败或反序列化失败均被捕获
    assert "detail" in resp.json()


def test_layers_forward_unknown_kv_ref(client):
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")
    resp = client.post(
        "/v1/layers/forward",
        json={
            "layer_range": "0-12",
            "tensor_ref": tensor_ref,
            "past_key_values_ref": "task_missing",
        },
    )
    assert resp.status_code == 404


# ----------------------------------------------------------------------
# 7. 实验端点门控
# ----------------------------------------------------------------------
def test_speculative_run_not_implemented(client):
    resp = client.post(
        "/v1/speculative/run", json={"prompt": "你好"}
    )
    assert resp.status_code == 501
    assert "detail" in resp.json()


# ----------------------------------------------------------------------
# 8. 层段加载/卸载
# ----------------------------------------------------------------------
def test_layers_load_unload(client):
    resp = client.post(
        "/v1/layers/load",
        json={"layer_range": "0-12", "embed": True, "lm_head": False},
    )
    assert resp.status_code == 200
    r = client.get("/v1/status")
    assert "0-12" in r.json()["layers"]
    resp = client.post("/v1/layers/unload", json={"layer_range": "0-12"})
    assert resp.status_code == 200
    r = client.get("/v1/status")
    assert "0-12" not in r.json()["layers"]
