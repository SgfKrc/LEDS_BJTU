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
import threading

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


class FakeModel:
    """假模型宿主：有 chat / active_model_id，行为可断言。"""

    active_model_id = "qwen-1.8b"
    _engine_type = "llama_cpp"
    model_loaded = True
    is_loaded = True
    generation_config = {
        "max_new_tokens": 1024,
        "tier_max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens=None, temperature=None, top_p=None,
             **_kw):
        self.calls.append(
            (list(messages), max_tokens, temperature, top_p)
        )
        return {
            "content": "候选答案内容",
            "usage": {"total_tokens": 8},
            "tokens_per_second": 12.5,
            "model": "qwen-1.8b",
        }

    def load_layer_range(self, start_layer=0, end_layer=24,
                         has_embedding=False, has_lm_head=False):
        """层段加载桩：1.2 复制完成后接入真实实现。"""
        return None


class FakeEngineHost(EngineHost):
    """轻量假宿主：注入 FakeModel 替代真实 ModelHost（不触发 model_module）。"""

    def __init__(self):
        super().__init__()
        self._host = FakeModel()  # 替换真实 ModelHost

    def load_model(self, engine=None, quant_type=None, use_compile=False, model_id=None):
        return {"success": True, "engine": engine or "pytorch", "model_id": model_id}

    def unload_model(self):
        return {"success": True, "message": "模型已卸载"}

    def switch_model(self, model_id, engine=None):
        return {"success": True, "model_id": model_id, "engine": engine}

    def current_model(self):
        # 对齐真实 EngineHost.current_model 完整字段（10 字段 loaded 形状）
        return {
            "loaded": True,
            "model_id": "qwen-1.8b",
            "quant_type": "int4",
            "model_name": "Qwen-1.8B-Chat",
            "model_path": "models/qwen-1_8b-chat",
            "engine": "pytorch",
            "total_params": "N/A",
            "device": "cpu",
            "gpu_allocated_gb": 0,
            "gpu_reserved_gb": 0,
        }

    def chat_full(self, req, cancel_event=None):
        # 返回结构与真实 EngineHost.chat_full 一致（content 而非 response，
        # 见 engine_host.py:678）：routes 取 result["content"]
        return {
            "content": "你好，我是 QLH。",
            "thinking_content": None,
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
            "request_id": "-",
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
    # 真实 EngineHost.chat_full 返回 content（对齐 api_server）
    assert body["content"] == "你好，我是 QLH。"
    assert body["request_id"] == "-"
    assert body["generation_id"].startswith("gen_")
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
    # 薄实现 done 事件的 "-" 占位被 routes 覆盖为真实 request_id/generation_id
    assert events[-1]["request_id"] == "-"  # 无 X-QLH-Request-ID 头时默认 "-"
    assert events[-1]["generation_id"].startswith("gen_")


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
    # 真实 chat_full 返回 content（修复前取 response 恒为空）
    assert events[0]["response"] == "你好，我是 QLH。"
    assert events[0]["thinking_content"] is None
    assert events[0]["generation_id"].startswith("gen_")


def test_chat_cancel_semantics(client):
    # 格式无效 → 400（对齐 api_server generation_id 格式校验）
    resp = client.post("/v1/chat/cancel", json={"generation_id": "bad!"})
    assert resp.status_code == 400
    assert "detail" in resp.json()
    # 未知但格式合法 → 200 cancel_pending（对齐 api_server，不 404）
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_nope_12345678"})
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "cancel_pending",
        "generation_id": "gen_nope_12345678",
    }
    # 已注册 → 200 cancel_requested
    host = client.app.state.engine_host
    gid, ev = host.register_generation("gen_test_123456")
    resp = client.post("/v1/chat/cancel", json={"generation_id": "gen_test_123456"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancel_requested"
    assert ev.is_set()
    host.unregister_generation(gid)


def test_chat_cancel_during_stream(monkeypatch):
    """生成期间 cancel 可命中（threadpool/线程桥接后事件循环不被阻塞）。

    修复前：async 端点内同步迭代生成器，流式生成期间 /v1/chat/cancel
    得不到处理（事件循环被占），取消功能实际失效。"""
    import time as _time

    host = FakeEngineHost()
    app = make_app(engine_host=host)
    # TestClient 单 portal 不支持并发：流式与 cancel 各用独立 TestClient，
    # 共享同一 app 实例（app.state.engine_host 是同一个 host）
    stream_client = TestClient(app)
    cancel_client = TestClient(app)

    def _slow_stream(self, req, cancel_event):
        for i in range(50):
            if cancel_event is not None and cancel_event.is_set():
                yield {"token": "[cancelled]"}
                return
            yield {"token": f"t{i}"}
            _time.sleep(0.05)  # 模拟慢生成：期间事件循环必须仍可服务 cancel
        yield {"done": True, "response": "ok", "followups": [], "metrics": {}}

    # FakeEngineHost 自带 chat_stream_events 覆盖基类，patch 必须作用于子类
    monkeypatch.setattr(FakeEngineHost, "chat_stream_events", _slow_stream)

    tokens = []
    statuses = []
    cancelled_ok = []
    errors = []

    def _consumer():
        try:
            with stream_client.stream(
                "POST",
                "/v1/chat/stream",
                json={"message": "hi", "streaming_mode": "fast"},
            ) as resp:
                statuses.append(resp.status_code)
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        ev = json.loads(line[6:])
                        tokens.append(ev)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=_consumer, daemon=True)
    t.start()
    # 等生成开始（已注册 gid）：轮询而非固定 sleep（TestClient 启动有开销）
    gids = []
    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        with host._gen_lock:
            gids = list(host._generations.keys())
        if gids:
            break
        _time.sleep(0.02)
    assert len(gids) == 1, (
        f"流式期间应有 1 个已注册 generation，实际: {gids}；"
        f"consumer 线程存活: {t.is_alive()}；statuses: {statuses}；"
        f"tokens: {tokens}；errors: {errors}"
    )
    resp = cancel_client.post("/v1/chat/cancel", json={"generation_id": gids[0]})
    cancelled_ok.append(resp.status_code)
    t.join(timeout=10)
    assert not errors
    assert cancelled_ok == [200], f"cancel 应 200，实际 {cancelled_ok}"
    # 流被 cancel 中断：出现 [cancelled] 且没有正常 done
    assert any(ev.get("token") == "[cancelled]" for ev in tokens)
    assert not any(ev.get("done") for ev in tokens)
    # 注册表已清空（无泄漏）
    with host._gen_lock:
        assert host._generations == {}


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
def test_speculative_run_gated_off(client):
    """SPEC_ENABLED 未启用（默认）→ 404 门控（复制 api_server 语义）。"""
    resp = client.post(
        "/v1/speculative/run", json={"message": "你好"}
    )
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_speculative_run_enabled_path(client, monkeypatch):
    """SPEC_ENABLED=true 时启用路径：真实 SpeculativeRunRequest 全字段
    透传到 host.speculative_run（修复前协议缺字段 → AttributeError 500）。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "SPEC_ENABLED", True)

    captured = {}

    def _fake_speculative_run(self, req):
        captured["message"] = req.message
        captured["gamma"] = req.gamma
        captured["max_rounds"] = req.max_rounds
        captured["seed"] = req.seed
        captured["draft_hint"] = req.draft_hint
        captured["allow_external"] = req.allow_external
        captured["max_new_tokens"] = req.max_new_tokens
        return {
            "content": "草稿验证结果",
            "finish_reason": "stop",
            "metrics": {},
            "rounds": 1,
        }

    from inference_service.engine_host import EngineHost

    monkeypatch.setattr(EngineHost, "speculative_run", _fake_speculative_run)
    resp = client.post(
        "/v1/speculative/run",
        json={
            "message": "你好",
            "gamma": 4,
            "max_rounds": 2,
            "seed": 42,
            "draft_hint": "你好，",
            "allow_external": True,
            "max_new_tokens": 32,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "草稿验证结果"
    assert captured == {
        "message": "你好",
        "gamma": 4,
        "max_rounds": 2,
        "seed": 42,
        "draft_hint": "你好，",
        "allow_external": True,
        "max_new_tokens": 32,
    }


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


# ----------------------------------------------------------------------
# 9. 1.2a 数据面执行段：task-worker Stage（复制自 api_server
#    _execute_task_worker_stage；宿主适配 model_manager → self._host）
# ----------------------------------------------------------------------
from task_provider import StageRequest as ProviderStageRequest
from task_graph import TaskGraphError


def _stage(stage_type, stage_id, root_input, dependencies=None):
    return ProviderStageRequest(
        workflow_id="w1",
        request_id="r1",
        stage_id=stage_id,
        stage_type=stage_type,
        provider_id="local-full-model",
        dependencies=dependencies or {},
        root_input=root_input,
    )


def test_worker_stage_full_inference():
    host = FakeEngineHost()
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [{"role": "user", "content": "1+1=?"}],
         "task_options": {"candidate_max_tokens": 128, "temperature": 0.5}},
    )
    result = host.execute_task_worker_stage(req, threading.Event())
    assert result["content"] == "候选答案内容"
    assert result["tokens_per_second"] == 12.5
    assert result["model"] == "qwen-1.8b"
    # 系统提示注入 + 用户消息
    messages = host._host.calls[0][0]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "1+1=?"
    assert host._host.calls[0][1] == 128  # candidate_budget
    assert host._host.calls[0][2] == 0.5  # temperature


def test_worker_stage_aggregate():
    host = FakeEngineHost()
    req = _stage(
        "aggregate", "aggregate",
        {"message": "原始问题", "task_options": {"final_max_tokens": 256}},
        dependencies={
            "candidate_a": {"content": "候选一"},
            "candidate_b": {"content": "候选二"},
        },
    )
    result = host.execute_task_worker_stage(req, threading.Event())
    assert result["content"] == "候选答案内容"
    messages = host._host.calls[0][0]
    assert "候选一" in messages[-1]["content"]
    assert "候选二" in messages[-1]["content"]


def test_worker_stage_unsupported_type():
    host = FakeEngineHost()
    req = _stage("bogus", "x", {"task_options": {}})
    with pytest.raises(TaskGraphError):
        host.execute_task_worker_stage(req, threading.Event())


def test_worker_stage_invalid_options():
    host = FakeEngineHost()
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [], "task_options": {"temperature": "不是数字"}},
    )
    with pytest.raises(TaskGraphError):
        host.execute_task_worker_stage(req, threading.Event())


def test_worker_stage_cancel_stops_inference():
    host = FakeEngineHost()
    cancel = threading.Event()
    cancel.set()  # 已取消 → run_model 直接返回空
    req = _stage(
        "full_inference", "candidate_a",
        {"messages": [{"role": "user", "content": "hi"}],
         "task_options": {}},
    )
    result = host.execute_task_worker_stage(req, cancel)
    assert result["content"] == ""
    assert host._host.calls == []  # 未触发模型调用


def test_worker_stage_route_generation_no_leak(monkeypatch):
    """路由级：/v1/worker/stage 结束时 generation 注册表无泄漏。
    （修复前：req.request_id 默认空时 finally 用 header 默认值 "-" 注销，
    注册表条目永不释放，且该 generation 永远无法 cancel。）"""
    from inference_service.engine_host import EngineHost

    host = FakeEngineHost()
    app = make_app(engine_host=host)
    client = TestClient(app)
    captured = {}

    def _fake_execute(self, stage_request, provider_cancel_event):
        captured["cancel_event"] = provider_cancel_event
        captured["stage_type"] = stage_request.stage_type
        return {"success": True, "content": "ok"}

    monkeypatch.setattr(EngineHost, "execute_task_worker_stage", _fake_execute)
    resp = client.post(
        "/v1/worker/stage",
        json={
            "workflow_id": "wf-1",
            "request_id": "",  # 空 request_id：注册表用自动生成 gid
            "stage_id": "s1",
            "stage_type": "candidate_a",
            "provider_id": "local",
            "dependencies": {},
            "root_input": {"task_options": {}},
            "model_identity": None,
            "runtime_context": {},
        },
    )
    assert resp.status_code == 200
    assert captured["stage_type"] == "candidate_a"
    # 请求结束后注册表必须清空（无泄漏）
    with host._gen_lock:
        assert host._generations == {}
    # 且注册期间 cancel 可命中（用注册时的 gid）
    assert captured["cancel_event"] is not None


# ----------------------------------------------------------------------
# 10. 1.2a KV from_profile（复制自 api_server._init_kv_cache 自适应逻辑）
# ----------------------------------------------------------------------
def test_kv_from_profile_tier_sizing():
    kv = KVHost()
    edge = kv.from_profile(profile={"tier": "edge", "gpu": {}})
    assert edge["reused"] is False
    edge_pages = edge["max_pages"]

    kv2 = KVHost()
    workstation = kv2.from_profile(profile={"tier": "workstation", "gpu": {}})
    assert workstation["max_pages"] > edge_pages  # 工作站档位更大


def test_kv_from_profile_with_model_heads():
    kv = KVHost()
    r = kv.from_profile(profile={"tier": "laptop", "gpu": {}},
                        num_heads=16, head_dim=64)
    assert r["total_pages"] == 0  # 初始未分配页
    assert r["max_pages"] > 0


# ----------------------------------------------------------------------
# 11. 1.2a 辅助纯函数行为（复制自 api_server，等价性由一次性对比脚本验证）
# ----------------------------------------------------------------------
def test_aux_strip_thinking_tags():
    from inference_service.engine_host import _strip_native_thinking_tags as strip
    assert strip("答案<think>思考</think>") == "答案"
    assert strip("<think>仅思考</think>") == ""
    assert strip("a\n\n\n\nb") == "a\n\nb"
    assert strip("") == ""


def test_aux_parse_thinking_response():
    from inference_service.engine_host import _parse_thinking_response as parse
    answer, thinking = parse("【思考】推理过程\n【思考结束】最终答案")
    assert thinking == "推理过程"
    assert answer == "最终答案"
    answer, thinking = parse("无标记直接回答")
    assert answer == "无标记直接回答"
    assert thinking is None


# ----------------------------------------------------------------------
# 12. 1.2b 追问生成（复制自 api_server._generate_followups_llama +
#     _is_question / _fallback_followups）
# ----------------------------------------------------------------------
def test_followups_history_too_short():
    host = FakeEngineHost()
    assert host.generate_followups_llama([]) == []
    assert host.generate_followups_llama([{"role": "user", "content": "hi"}]) == []


def test_followups_cancel_event():
    host = FakeEngineHost()
    cancel = threading.Event()
    cancel.set()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    assert host.generate_followups_llama(history, cancel) == []


def test_followups_model_generates_valid_questions():
    host = FakeEngineHost()

    class ChatModel(FakeModel):
        def chat(self, messages, **_kw):
            return {"content": "1. 量化有哪些方法？\n2. INT8量化相比INT4有哪些优势？\n3. 陈述句不是问题"}

    host._host = ChatModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert "量化有哪些方法？" in questions
    assert "INT8量化相比INT4有哪些优势？" in questions
    assert "陈述句不是问题" not in questions
    assert len(questions) <= 3


def test_followups_fallback_on_garbage():
    host = FakeEngineHost()

    class GarbageModel(FakeModel):
        def chat(self, messages, **_kw):
            return {"content": "完全不相关的内容没有问号"}

    host._host = GarbageModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert len(questions) >= 2  # 关键词模板兜底
    assert any("量化" in q or "模型" in q for q in questions)


def test_followups_generation_failure_non_fatal():
    host = FakeEngineHost()

    class BrokenModel(FakeModel):
        def chat(self, messages, **_kw):
            raise RuntimeError("模型调用失败")

    host._host = BrokenModel()
    history = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩技术。"},
    ]
    questions = host.generate_followups_llama(history)
    assert len(questions) >= 2  # 异常非致命 → 模板兜底


def test_is_question_aux():
    from inference_service.engine_host import _is_question
    assert _is_question("量化有哪些方法？")
    assert _is_question("为什么？")
    assert not _is_question("量化很重要")  # 无问号
    assert not _is_question("有以下几点：")  # 陈述句式
    assert not _is_question("1. 首先")  # 列举开头
    assert not _is_question("")


# ----------------------------------------------------------------------
# 13. 1.2c 外部推理整请求路由（复制自 api_server._execute_external_chat）
# ----------------------------------------------------------------------
class FakeExternalClient:
    masked_base_url = "https://***"
    model_name = "gpt-4o-mini"

    def __init__(self):
        self.last = None

    def ensure_connected(self):
        pass

    def chat(self, history, **kw):
        self.last = (list(history), kw)
        return {
            "content": "外部回答",
            "usage": {"completion_tokens": 3},
            "tokens_per_second": 9.0,
            "model": "gpt-4o-mini",
        }


def test_external_chat_routes_and_persists(monkeypatch):
    import db
    import external_provider

    fake_client = FakeExternalClient()
    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: fake_client)
    monkeypatch.setattr(db, "get_save_history", lambda: False)  # 测试不落盘

    host = FakeEngineHost()
    host._host._db_available = True  # 避免走 local_store 落盘分支

    req = ChatRequest(message="你好", allow_external=True, client_node_type="pc")
    result = host.execute_external_chat(
        req,
        [{"role": "user", "content": "之前的问题"}],
        target_session_id="s1",
    )
    assert result["content"] == "外部回答"
    assert result["metrics"]["engine"] == "external_api"
    assert result["metrics"]["request_origin"] == "pc_http"
    assert result["metrics"]["completion_tokens"] == 3
    assert host._conversation_stats["rounds"] == 1
    assert host._conversation_stats["total_generated_tokens"] == 3
    # 请求历史 = 原历史 + 新消息
    assert fake_client.last[0][-1] == {"role": "user", "content": "你好"}
    assert fake_client.last[1]["allow_external"] is True


def test_external_chat_strips_thinking_when_disabled(monkeypatch):
    import db
    import external_provider

    class ThinkingClient(FakeExternalClient):
        def chat(self, history, **kw):
            return {"content": "<think>内部</think>外部回答"}

    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: ThinkingClient())
    monkeypatch.setattr(db, "get_save_history", lambda: False)

    host = FakeEngineHost()
    host._host._db_available = True
    req = ChatRequest(message="hi", allow_external=True, show_thinking=False)
    result = host.execute_external_chat(req, [], "s2")
    assert "内部" not in result["content"]


def test_chat_origin_aux():
    from inference_service.engine_host import _chat_origin
    assert _chat_origin(ChatRequest(message="x", client_node_type="android")) == "android_http"
    assert _chat_origin(ChatRequest(message="x", client_node_type="pc")) == "pc_http"
    assert _chat_origin(ChatRequest(message="x")) == "web_http"


def make_full_host(fake_model_cls=FakeModel):
    """真实 EngineHost + 假模型注入（测试 chat_full 复制实现本体）。"""
    host = EngineHost()
    host._host = fake_model_cls()
    host._host._db_available = True
    return host


# ----------------------------------------------------------------------
# 14.5 1.2d task_graph 执行段（复制自 api_server._execute_task_graph_chat
#      + _execute_task_graph_chat_with_slot + 5 个辅助函数）
# ----------------------------------------------------------------------
def test_task_graph_gate_off():
    """TASK_GRAPH_ENABLED 默认关闭 → chat_full(task_graph) 409 门控
    （对齐 api_server 语义；不加载模型、不创建 journal）。"""
    # 真实 EngineHost（FakeEngineHost 覆盖了基类 chat_full，分支不会触发）
    host = EngineHost()
    req = ChatRequest(message="hi", execution_mode="task_graph")
    with pytest.raises(Exception) as excinfo:
        host.chat_full(req)
    assert excinfo.value.status_code == 409
    assert "任务链实验未启用" in str(excinfo.value.detail)


def test_task_graph_chat_full_dispatches(monkeypatch):
    """execution_mode=task_graph → chat_full 分派到 execute_task_graph_chat
    （对齐 api_server._execute_requested_chat）。"""
    from inference_service.engine_host import EngineHost

    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    host = EngineHost()  # 真实 EngineHost：基类 chat_full 才有 task_graph 分支
    captured = {}

    def _fake_execute(self, req, cancel_event=None):
        captured["mode"] = req.execution_mode
        return {"content": "task-graph-ok", "thinking_content": None,
                "metrics": {"execution_mode": "task_graph"}, "followups": []}

    monkeypatch.setattr(EngineHost, "execute_task_graph_chat", _fake_execute)
    req = ChatRequest(message="hi", execution_mode="task_graph")
    result = host.chat_full(req)
    assert captured["mode"] == "task_graph"
    assert result["content"] == "task-graph-ok"


def test_task_graph_slot_exclusive(monkeypatch):
    """task_graph 执行槽互斥：占用中再来 → 429（对齐 api_server 语义）。"""
    from inference_service.engine_host import EngineHost

    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    host = EngineHost()
    captured = []

    def _fake_execute(self, req, cancel_event=None):
        captured.append("entered")
        return {"content": "ok", "thinking_content": None,
                "metrics": {}, "followups": []}

    monkeypatch.setattr(EngineHost, "execute_task_graph_chat_with_slot", _fake_execute)
    # gate 的 journal 检查需 available：伪造 coordinator
    class _FakeCoordinator:
        def journal_status(self):
            return {"available": True}

    monkeypatch.setattr(EngineHost, "_ensure_task_graph_coordinator",
                        lambda self: _FakeCoordinator())
    # 先占用槽位
    assert host._task_graph_execution_slot.acquire(blocking=False) is True
    req = ChatRequest(message="hi", execution_mode="task_graph")
    with pytest.raises(Exception) as excinfo:
        host.chat_full(req)
    assert excinfo.value.status_code == 429
    assert captured == []  # 未进入执行体
    host._task_graph_execution_slot.release()


def test_task_graph_coordinator_lazy_idempotent():
    """_ensure_task_graph_coordinator 惰性创建且幂等（单实例缓存）。"""
    host = FakeEngineHost()
    c1 = host._ensure_task_graph_coordinator()
    c2 = host._ensure_task_graph_coordinator()
    assert c1 is c2


def test_task_graph_active_identity_none_without_model():
    """未加载模型 → _active_task_graph_model_identity 返回 None。"""
    host = EngineHost()  # 真实 ModelHost：model_loaded=False
    assert host._active_task_graph_model_identity() is None


def test_task_graph_with_slot_real_execution(monkeypatch):
    """真实执行 execute_task_graph_chat_with_slot 主体（不 monkeypatch
    执行体）：本地模板分支全链路——会话切换/历史维护/run_template/
    持久化/metrics。修复前此路径 6 处 NameError 必崩。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(_cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", False)

    host = make_full_host()  # 真实 EngineHost + FakeModel（_db_available=True）
    host._host._db_available = False  # 走 local_store 分支
    host._host.full_chat_execution_lock = threading.RLock()  # FakeModel 缺锁
    host._active_session_id = "s1"
    host._session_histories["s1"] = []

    class FakeCoordinator:
        def __init__(self):
            self.providers = []

        def journal_status(self):
            return {"available": True}

        def has_provider(self, pid):
            return pid in self.providers

        def register_provider(self, provider):
            self.providers.append(provider.provider_id)

        def run_template(self, **kw):
            captured["run_template"] = kw
            return (
                {"content": "任务链答案", "thinking_content": None},
                {
                    "workflow_id": "wf-1",
                    "template": "dual_candidate",
                    "state": "completed",
                    "partial_result": False,
                    "stages": [],
                    "stage_count": 0,
                    "attempt_count": 0,
                    "duration_seconds": 0.5,
                },
            )

        def commit_result(self, workflow_id):
            return {"workflow_id": workflow_id, "stages": [],
                    "template": "dual_candidate", "state": "completed",
                    "partial_result": False, "stage_count": 0,
                    "attempt_count": 0, "duration_seconds": 0.5}

        def discard_result(self, workflow_id):
            return None

        def provider_status(self):
            return []

    fake_coord = FakeCoordinator()
    captured = {}
    monkeypatch.setattr(
        EngineHost, "_ensure_task_graph_coordinator",
        lambda self: fake_coord,
    )

    req = ChatRequest(message="hi", session_id="s1", execution_mode="task_graph")
    result = host.chat_full(req)

    assert result["content"] == "任务链答案"
    assert result["metrics"]["execution_mode"] == "task_graph"
    assert result["metrics"]["workflow_id"] == "wf-1"
    # run_template 收到正确入参（root_input 带消息与历史）
    assert captured["run_template"]["root_input"]["message"] == "hi"
    assert captured["run_template"]["session_id"] == "s1"
    # 历史已追加（user + assistant）
    assert len(host._session_histories["s1"]) == 2
    assert host._session_histories["s1"][0]["role"] == "user"
    assert host._session_histories["s1"][1]["role"] == "assistant"
    # 本地 provider 注册成功（self._dispatch_local_task_provider 引用有效）
    assert "local_full_model" in fake_coord.providers


def test_task_graph_auto_remote_identity_path(monkeypatch):
    """auto_remote 分支调用 _active_task_graph_model_identity（修复前
    1597 行裸 model_manager NameError）：无可用远端 → 本地降级执行。"""
    import config as _cfg

    monkeypatch.setattr(_cfg, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(_cfg, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)

    host = make_full_host()
    host._host._db_available = False
    host._host.full_chat_execution_lock = threading.RLock()
    host._active_session_id = "s1"
    host._session_histories["s1"] = []

    class FakeCoordinator:
        def __init__(self):
            self.providers = []

        def journal_status(self):
            return {"available": True}

        def has_provider(self, pid):
            return pid in self.providers

        def register_provider(self, provider):
            self.providers.append(provider.provider_id)

        def run_template(self, **kw):
            return (
                {"content": "远端不可用本地降级", "thinking_content": None},
                {
                    "workflow_id": "wf-2",
                    "template": "dual_candidate",
                    "state": "completed",
                    "partial_result": False,
                    "stages": [],
                    "stage_count": 0,
                    "attempt_count": 0,
                    "duration_seconds": 0.3,
                },
            )

        def commit_result(self, workflow_id):
            return {"workflow_id": workflow_id, "stages": [],
                    "template": "dual_candidate", "state": "completed",
                    "partial_result": False, "stage_count": 0,
                    "attempt_count": 0, "duration_seconds": 0.3}

        def discard_result(self, workflow_id):
            return None

        def provider_status(self):
            return []

    monkeypatch.setattr(
        EngineHost, "_ensure_task_graph_coordinator",
        lambda self: FakeCoordinator(),
    )
    # 不 stub _active_task_graph_model_identity：真实执行方法体
    # （FakeModel 有 _engine_type/active_model_id 但缺 _model_path →
    # 返回 None → model_identity_unavailable → 本地降级；修复前 1597
    # 行裸 model_manager 在此路径 NameError）

    req = ChatRequest(
        message="hi", session_id="s1", execution_mode="task_graph",
        task_graph_auto_remote=True,
    )
    result = host.chat_full(req)
    assert result["content"] == "远端不可用本地降级"
    # 无可用远端 → 降级标记
    assert result["metrics"]["fallback"] is True
    assert result["metrics"]["auto_remote_enabled"] is True
    assert result["metrics"]["auto_remote_providers"] == []
    assert result["metrics"]["fallback_reason"] == "model_identity_unavailable"


def test_chat_full_llama_cpp_path(monkeypatch):
    """llama.cpp 引擎整请求路径：历史维护 + metrics + followups 兜底。"""
    import db

    monkeypatch.setattr(db, "get_save_history", lambda: False)  # 不落盘

    host = make_full_host()
    host._active_session_id = "s1"
    host._session_histories["s1"] = [
        {"role": "user", "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"},
    ]

    req = ChatRequest(message="新问题", session_id="s1")
    result = host.chat_full(req)

    assert result["content"] == "候选答案内容"
    assert result["metrics"]["engine"] == "llama_cpp"
    assert result["metrics"]["execution_mode"] == "local_llama_cpp"
    assert result["metrics"]["request_origin"] == "web_http"
    # 历史已追加两轮
    assert host._session_histories["s1"][-2:] == [
        {"role": "user", "content": "新问题"},
        {"role": "assistant", "content": "候选答案内容"},
    ]
    # followups：模型无合格问句 → 模板兜底 ≥2
    assert len(result["followups"]) >= 2
    assert host._conversation_stats["rounds"] == 1


def test_chat_full_session_switch(monkeypatch):
    """session_id 切换 → _switch_session 加载目标会话。"""
    import db

    monkeypatch.setattr(db, "get_save_history", lambda: False)

    host = make_full_host()
    host._active_session_id = "s_old"
    host._session_histories["s_old"] = [{"role": "user", "content": "旧会话"}]

    req = ChatRequest(message="hello", session_id="s_new")
    result = host.chat_full(req)
    assert host._active_session_id == "s_new"
    assert result["content"] == "候选答案内容"
    # 新会话历史 = 本次对话
    assert host._session_histories["s_new"][-1]["role"] == "assistant"


def test_chat_full_first_message_auto_title(monkeypatch):
    """首条消息自动标题：截取前 30 字。"""
    import db

    monkeypatch.setattr(db, "get_save_history", lambda: False)
    monkeypatch.setattr(db, "update_session_title",
                        lambda sid, title: titles.append((sid, title)))
    titles = []

    host = make_full_host()
    long_msg = "这是一个非常非常非常非常非常非常非常非常非常非常长的首条消息啊"
    req = ChatRequest(message=long_msg, session_id="s_title")
    host.chat_full(req)
    assert titles and titles[0][0] == "s_title"
    assert titles[0][1].endswith("...")


def test_chat_full_external_fallback(monkeypatch):
    """外部路由失败（无本地模型）→ prefer_external 时 502。"""
    import config as _cfg
    import db
    import external_provider

    monkeypatch.setattr(db, "get_save_history", lambda: False)
    _cfg.EXTERNAL_ENABLED = True
    _cfg.EXTERNAL_BASE_URL = "https://example.com/v1"
    _cfg.EXTERNAL_DATA_SCOPE = "opt_in"

    class FailingClient(FakeExternalClient):
        def chat(self, history, **kw):
            raise RuntimeError("external down")

    monkeypatch.setattr(external_provider, "get_external_chat_client", lambda: FailingClient())

    host = make_full_host()
    host._host.model_loaded = False
    host._host.is_loaded = False

    from fastapi import HTTPException
    req = ChatRequest(message="hi", session_id="s1",
                      allow_external=True, prefer_external=True)
    with pytest.raises(HTTPException) as exc:
        host.chat_full(req)
    assert exc.value.status_code == 502


class FakeTokenizer:
    eos_token_id = 151643

    def __call__(self, prompt, return_tensors="pt"):
        import torch
        return {"input_ids": torch.zeros(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "fake prompt"

    def decode(self, ids, skip_special_tokens=True):
        return "PyTorch本地回答"


class FakePyTorchModel(FakeModel):
    """PyTorch 引擎：带 tokenizer + model.generate 的假实现。"""

    _engine_type = "pytorch"
    tokenizer = FakeTokenizer()

    def __init__(self):
        super().__init__()
        import torch
        self._model = torch.nn.Linear(4, 4)  # 仅占位
        self.device_str = "cpu"

    @property
    def model(self):
        return self

    def ensure_full_model(self):
        return None

    def get_device(self):
        return "cpu"

    def _merge_stop_sequences(self, seqs):
        return None

    def _get_generation_eos_token_ids(self, stop_sequences):
        return None

    def _build_stop_criteria(self, seqs, prompt_len, **kw):
        return None

    def _decode_generated_ids(self, ids, stop_sequences):
        return "PyTorch本地回答"

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.generate_called = True
        return input_ids  # 形状 [1, prompt_len] → generated_ids 为空


def test_chat_full_pytorch_path(monkeypatch):
    """PyTorch 引擎路径：generate + 解码 + 追问 + 持久化。"""
    import db

    monkeypatch.setattr(db, "get_save_history", lambda: False)

    host = make_full_host(FakePyTorchModel)
    host._active_session_id = "s1"
    host._session_histories["s1"] = [
        {"role": "user", "content": "什么是量化？"},
        {"role": "assistant", "content": "量化是模型压缩。"},
    ]

    req = ChatRequest(message="详细讲讲", session_id="s1")
    result = host.chat_full(req)

    assert result["content"] == "PyTorch本地回答"
    assert result["metrics"]["engine"] == "pytorch"
    assert result["metrics"]["execution_mode"] == "local_pytorch"
    assert host._conversation_stats["rounds"] == 1
    # followups 走 PyTorch 追问路径（模型无问句 → 兜底）
    assert len(result["followups"]) >= 2
    # 会话历史追加
    assert host._session_histories["s1"][-1]["role"] == "assistant"


# ----------------------------------------------------------------------
# 15. 1.3 服务入口：build_app + 角色感知门控
# ----------------------------------------------------------------------
def test_build_app_master_role():
    from inference_svc_main import build_app

    app = build_app("master")
    assert app.state.engine_host.role == "master"
    assert app.state.node_role == "master"
    with TestClient(app) as c:
        assert c.get("/v1/health").status_code == 200
        # master 角色放行 chat（无模型文件环境返回 500 模型未加载；有模型则 200）
        assert c.post("/v1/chat", json={"message": "你好"}).status_code in (200, 500, 507)


def test_build_app_client_role_gates_chat():
    from inference_svc_main import build_app

    app = build_app("client")
    assert app.state.engine_host.role == "client"
    with TestClient(app) as c:
        # client 角色：chat 端点 404，层段/KV 端点仍可用
        assert c.post("/v1/chat", json={"message": "你好"}).status_code == 404
        assert c.post("/v1/chat/stream", json={"message": "你好"}).status_code == 404
        assert c.get("/v1/health").status_code == 200
        assert c.post("/v1/kv/init", json={}).status_code == 200
        r = c.post("/v1/layers/load", json={"layer_range": "0-12"})
        assert r.status_code in (200, 500)  # FakeModel 桩可加载


def test_entry_module_no_heavy_imports():
    """1.3 验收：入口顶层不 import model_module/transformers/sklearn。"""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src'); import inference_svc_main; "
        "mods = set(sys.modules); "
        "heavy = [m for m in ('model_module', 'transformers', 'sklearn', 'pandas') "
        "         if any(k == m or k.startswith(m + '.') for k in mods)]; "
        "print('HEAVY:', heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "HEAVY: []" in result.stdout, f"顶层拉入了重依赖: {result.stdout} {result.stderr}"


# ----------------------------------------------------------------------
# 16. 1.4 InferenceClient（HTTP 客户端，真实 uvicorn + FakeEngineHost）
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_inference_svc():
    """起真实 inference-svc（随机端口），engine_host 换为 FakeEngineHost。"""
    import socket
    import threading

    import uvicorn

    from inference_svc_main import build_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    app = build_app("master")
    app.state.engine_host = FakeEngineHost()
    app.state.node_role = "master"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 等待就绪
    import time
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{port}/v1/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_inference_client_lifecycle(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    # 模型生命周期
    r = client.load_model(engine="pytorch")
    assert r["success"] is True
    assert client.model_loaded is True  # 状态缓存
    r = client.unload_model()
    assert r["success"] is True


def test_inference_client_chat(live_inference_svc, monkeypatch):
    from inference_client import InferenceClient

    # 验证 session_id 透传（修复前丢 session_id → 多轮历史语义漂移）
    captured = {}

    def _chat_full(self, req, cancel_event=None):
        captured["session_id"] = req.session_id
        return {
            "content": "你好，我是 QLH。",
            "thinking_content": None,
            "followups": [],
            "metrics": {"tokens_per_second": 42.0},
        }

    # FakeEngineHost 自带 chat_full 覆盖基类，patch 必须作用于子类
    monkeypatch.setattr(FakeEngineHost, "chat_full", _chat_full)
    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    result = client.chat(
        [{"role": "user", "content": "你好"}], session_id="sess-1"
    )
    assert captured["session_id"] == "sess-1"
    assert result["content"] == "你好，我是 QLH。"
    assert result["metrics"]["tokens_per_second"] == 42.0


def test_inference_client_chat_stream(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    events = list(client.chat_stream([{"role": "user", "content": "你好"}]))
    tokens = [e["token"] for e in events if "token" in e]
    assert "".join(tokens) == "你好"
    assert events[-1]["done"] is True


def test_inference_client_forward_layers_loopback(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    hidden = torch.randn(2, 2048, dtype=torch.float16)
    out = client.forward_layers("0-12", hidden)
    torch.testing.assert_close(out, hidden)  # FakeEngineHost identity


def test_inference_client_worker_stage(live_inference_svc):
    from inference_client import InferenceClient
    from task_provider import StageRequest as ProviderStageRequest

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    stage = ProviderStageRequest(
        workflow_id="w1", request_id="r1", stage_id="candidate_a",
        stage_type="full_inference", provider_id="local-full-model",
        dependencies={},
        root_input={"messages": [{"role": "user", "content": "1+1=?"}],
                    "task_options": {}},
    )
    result = client._execute_task_worker_stage(stage)
    assert result["content"] == "候选答案内容"


def test_inference_client_kv_and_cancel(live_inference_svc):
    from inference_client import InferenceClient

    client = InferenceClient(base_url=live_inference_svc, timeout=30)
    r = client.kv_init(task_id="t_remote")
    assert r["task_id"] == "t_remote"
    r = client.kv_free(task_id="t_remote")
    assert r["freed"] is True
    # 未知 generation cancel → 200 cancel_pending（对齐 api_server 语义，2026-08-05）
    r = client.cancel_generation("gen_nonexistent")
    assert r["status"] == "cancel_pending"


# ----------------------------------------------------------------------
# 17. 1.5 从节点 PeerClient（复制自 scheduler client 分支）
# ----------------------------------------------------------------------
class FakeTCPClient:
    def __init__(self):
        self.sent = []
        self._running = True
        self.device_info = {}
        self.server_host = "127.0.0.1"
        self.server_port = 8888
        self.is_registered = True
        self.sock = object()

    def send_data(self, payload, msg_type):
        self.sent.append((msg_type, dict(payload)))


def make_peer():
    from inference_service.peer import PeerClient

    peer = PeerClient(master_host="127.0.0.1", master_port=8888,
                      node_id="test_client")
    fake = FakeModel()
    fake.forward_layers_called = []

    def _forward_layers(input_ids=None, hidden_states=None, attention_mask=None,
                        position_ids=None, past_key_values=None, use_cache=True,
                        apply_lm_head=False, **_kw):
        fake.forward_layers_called.append(
            (input_ids, hidden_states, past_key_values, apply_lm_head))
        hs = hidden_states if hidden_states is not None else torch.zeros(1, 8, 128)
        pkv = (torch.zeros(1, 2, 8, 64), torch.zeros(1, 2, 8, 64))
        out = {"hidden_states": hs}
        if apply_lm_head:
            out["logits"] = torch.zeros(1, 8, 32000)
        if use_cache:
            out["past_key_values"] = pkv
        return out

    fake.forward_layers = _forward_layers
    fake.load_model = lambda **kw: {"success": True}
    fake.is_loaded = True
    fake._engine_type = "pytorch"
    fake.model = type("M", (), {"config": type("C", (), {"model_type": "qwen"})()})()
    peer._host._host = fake
    peer._client = FakeTCPClient()
    return peer


def test_peer_layer_config_new_format(monkeypatch):
    peer = make_peer()
    monkeypatch.setattr("model_sync.resolve_worker_model_path", lambda mid: "/fake/path")
    monkeypatch.setattr("model_sync.ensure_model_available", lambda mid: None)

    peer._handle_layer_config({
        "config_id": "cfg-1",
        "node_id": "test_client",
        "start_layer": 0, "end_layer": 12,
        "has_embedding": True, "has_lm_head": False,
        "model_id": "qwen-1.8b", "model_sha256": "abc123",
        "model_type": "qwen", "total_layers": 24,
    })
    ack = [p for t, p in peer._client.sent if t.value == "layer_config_ack"]
    assert ack and ack[0]["status"] == "ready"
    assert ack[0]["config_id"] == "cfg-1"
    assert peer._active_layer_config["config_id"] == "cfg-1"


def test_peer_layer_config_missing_contract(monkeypatch):
    peer = make_peer()
    peer._handle_layer_config({
        "config_id": "cfg-x", "node_id": "test_client",
        "start_layer": 0, "end_layer": 12,
        "model_id": "", "model_sha256": "", "model_type": "qwen",
        "total_layers": 24,
    })
    ack = [p for t, p in peer._client.sent if t.value == "layer_config_ack"]
    assert ack and ack[0]["status"] == "error"


def test_peer_layer_forward_prefill_and_decode():
    from tcp_comm import MessageType

    peer = make_peer()
    peer._active_layer_config = {
        "config_id": "cfg-1", "model_sha256": "abc123",
        "model_type": "qwen", "model_id": "qwen-1.8b",
    }
    # prefill step 0
    peer._handle_layer_forward({
        "task_id": "t1", "step": 0, "use_kv_cache": False,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
        "input_ids": [1, 2, 3], "apply_lm_head": False,
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" not in results[0]
    assert results[0]["step"] == 0
    assert "hidden_states" in results[0]  # base64 序列化
    # decode step 1（用 KV cache）
    peer._client.sent.clear()
    peer._handle_layer_forward({
        "task_id": "t1", "step": 1, "use_kv_cache": True,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
        "hidden_states": None, "apply_lm_head": True,
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" not in results[0]
    assert "logits" in results[0]


def test_peer_layer_forward_step_out_of_order():
    from tcp_comm import MessageType

    peer = make_peer()
    peer._active_layer_config = {
        "config_id": "cfg-1", "model_sha256": "abc123",
        "model_type": "qwen", "model_id": "qwen-1.8b",
    }
    # 直接 decode step 2（无 prefill）→ 越序错误
    peer._handle_layer_forward({
        "task_id": "t9", "step": 2, "use_kv_cache": True,
        "config_id": "cfg-1", "model_sha256": "abc123", "model_type": "qwen",
    })
    results = [p for t, p in peer._client.sent if t == MessageType.LAYER_RESULT]
    assert results and "error" in results[0]


def test_peer_pipeline_done_abort_cleanup():
    peer = make_peer()
    peer._kv_cache["t1"] = object()
    peer._local_pipeline_steps["t1"] = 3
    peer._handle_pipeline_done({"task_id": "t1"})
    assert "t1" not in peer._kv_cache
    assert "t1" not in peer._local_pipeline_steps

    peer._kv_cache["t2"] = object()
    peer._handle_pipeline_abort({"task_id": "t2"})
    assert "t2" not in peer._kv_cache
    assert "t2" in peer._local_pipeline_cancelled


def test_peer_client_no_heavy_imports():
    """1.5 验收：peer 路径不 import fastapi/scheduler/api_server。"""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import inference_service.peer; "
        "mods = set(sys.modules); "
        "bad = [m for m in ('fastapi', 'scheduler', 'api_server', 'uvicorn') "
        "       if any(k == m or k.startswith(m + '.') for k in mods)]; "
        "print('BAD:', bad)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert "BAD: []" in result.stdout, f"从节点入口拉入了重依赖: {result.stdout} {result.stderr}"


# ----------------------------------------------------------------------
# 18. 1.7 scheduler-svc 入口（QLH_MONOLITH 回退选择）
# ----------------------------------------------------------------------
def test_scheduler_svc_host_selection(monkeypatch):
    import scheduler_svc_main

    # 微服务模式：host=InferenceClient
    monkeypatch.setenv("QLH_MONOLITH", "0")
    sched = scheduler_svc_main.build_scheduler()
    from inference_client import InferenceClient

    assert isinstance(sched._host, InferenceClient)

    # 回退模式：host=进程内 model_host（一键回单进程）
    monkeypatch.setenv("QLH_MONOLITH", "1")
    sched2 = scheduler_svc_main.build_scheduler()
    from model_host import ModelHost

    assert isinstance(sched2._host, ModelHost)


# ----------------------------------------------------------------------
# 17.5 scheduler-svc HTTP 壳（§4.2 透传路径契约，阶段 2 起点实现）
# ----------------------------------------------------------------------
@pytest.fixture()
def sched_http_client():
    """真实 Scheduler 实例 + HTTP 壳应用（不启动 TCP，TestClient 直连）。"""
    import scheduler as sched_mod
    import scheduler_svc_http as http_mod

    sched = sched_mod.Scheduler()
    app = http_mod.build_scheduler_app(sched)
    yield TestClient(app), sched
    http_mod.reset_scheduler()  # 测试隔离：清空注入实例


def test_sched_http_status(sched_http_client):
    """GET /cluster/status：run_mode/node_role/node_id/max_nodes 字段。"""
    client, _ = sched_http_client
    r = client.get("/cluster/status")
    assert r.status_code == 200
    body = r.json()
    assert "run_mode" in body and "node_role" in body
    assert "node_id" in body and "max_nodes" in body


def test_sched_http_nodes(sched_http_client):
    """GET /cluster/nodes：nodes/count/online_count/offline_count。"""
    client, _ = sched_http_client
    r = client.get("/cluster/nodes")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"nodes", "count", "online_count", "offline_count"}
    assert isinstance(body["count"], int)


def test_sched_http_my_role(sched_http_client):
    client, _ = sched_http_client
    r = client.get("/cluster/my-role")
    assert r.status_code == 200
    body = r.json()
    assert "is_master" in body and "node_role" in body and "node_id" in body


def test_sched_http_queue_lifecycle(sched_http_client):
    """queue 域：detail/strategy/pause/resume/clear。"""
    client, _ = sched_http_client
    r = client.get("/cluster/queue")
    assert r.status_code == 200
    body = r.json()
    for key in ("paused", "strategy", "queue_size", "max_size", "q0", "q1", "q2"):
        assert key in body, f"queue 缺字段 {key}"

    r = client.post("/cluster/queue/strategy", json={"strategy": "mlfq"})
    assert r.status_code == 200
    assert r.json()["strategy"] == "mlfq"

    r = client.post("/cluster/queue/pause")
    assert r.status_code == 200 and r.json()["paused"] is True
    r = client.post("/cluster/queue/resume")
    assert r.status_code == 200 and r.json()["paused"] is False
    r = client.post("/cluster/queue/clear")
    assert r.status_code == 200 and "cleared" in r.json()


def test_sched_http_config_and_layers(sched_http_client):
    client, _ = sched_http_client
    r = client.get("/cluster/config")
    assert r.status_code == 200
    assert "run_mode" in r.json() and "max_nodes" in r.json()

    r = client.get("/cluster/layers")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and "assignments" in body

    r = client.get("/cluster/config/distributed-inference")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_sched_http_nodes_deregister_404(sched_http_client):
    """deregister 不存在节点 → 404（对齐 api_server 语义）。"""
    client, _ = sched_http_client
    r = client.post("/cluster/nodes/nope/deregister")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_sched_http_queue_task_cancel_404_semantics(sched_http_client):
    """cancel 不存在任务 → 200 + success:false（对齐 api_server.py:5715-5732）。"""
    client, _ = sched_http_client
    r = client.delete("/cluster/queue/task/nonexistent-task")
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is False
    assert "message" in body


# ----------------------------------------------------------------------
# 17.6 scheduler-svc device 域（画像 profile / auto-configure / select-gpu）
# ----------------------------------------------------------------------

FAKE_DEVICE_PROFILE = {
    "tier": "laptop",
    "tier_label": "笔记本",
    "tier_icon": "💻",
    "score_total": 85.5,
    "score_breakdown": {"gpu": 50.0, "ram": 24.0, "cpu": 11.5},
    "cpu": {
        "model_name": "Intel(R) Core(TM) i5-12400F",
        "physical_cores": 6,
        "logical_cores": 12,
        "freq_mhz": 2496.0,
        "freq_max_mhz": 4400.0,
        "architecture": "x86_64",
        "usage_percent": 3.2,
    },
    "ram": {"total_gb": 16.0, "available_gb": 8.5, "used_gb": 7.5, "percent_used": 46.9},
    "gpu": None,
    "gpus": [{
        "name": "NVIDIA GeForce RTX 3060",
        "vram_total_gb": 12.0,
        "vram_free_gb": 10.0,
        "cuda_available": True,
        "compute_capability": "8.6",
        "is_integrated": False,
        "gpu_type": "discrete",
        "driver_version": "566.36",
        "mps_available": False,
        "index": 0,
    }],
    "selected_gpu_index": 0,
    "disk": {"free_gb": 100.0, "total_gb": 512.0},
    "platform": {
        "os": "Windows",
        "os_version": "10.0.22631",
        "architecture": "AMD64",
        "machine": "AMD64",
        "hostname": "test-pc",
        "python_version": "3.12.10",
    },
    "recommendations": ["推荐使用 INT4 量化档位"],
    "warnings": [],
    "android_ready": False,
    "island": False,
}


def test_sched_http_device_profile(sched_http_client, monkeypatch):
    """GET /device/profile：真实 to_dict 字段 + TUI 兼容别名字段。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)
    r = client.get("/device/profile")
    assert r.status_code == 200
    body = r.json()
    # 真实字段（device_profiler.to_dict 形状）
    assert body["tier"] == "laptop"
    assert body["score_total"] == 85.5
    assert body["platform"]["hostname"] == "test-pc"
    assert body["cpu"]["model_name"] == "Intel(R) Core(TM) i5-12400F"
    assert body["gpus"][0]["gpu_type"] == "discrete"
    # 兼容字段（TUI tui_admin.py:1141-1151 消费）
    assert body["hostname"] == "test-pc"
    assert body["os"] == {"system": "Windows", "release": "10.0.22631"}
    assert body["cpu"]["model"] == "Intel(R) Core(TM) i5-12400F"
    assert body["cpu"]["brand"] == "Intel(R) Core(TM) i5-12400F"
    assert body["memory"]["total_gb"] == 16.0


def test_sched_http_device_profile_detect_failure(sched_http_client, monkeypatch):
    """GET /device/profile：检测失败 → 500。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", None)

    def _boom():
        raise RuntimeError("detect failed")

    monkeypatch.setattr(http_mod, "_detect_device_profile", _boom)
    r = client.get("/device/profile")
    assert r.status_code == 500
    assert "设备检测失败" in r.json()["detail"]


def test_sched_http_device_auto_configure(sched_http_client, monkeypatch):
    """POST /device/auto-configure：应用推荐配置。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)

    class FakeProfiler:
        def recommend_config(self):
            return {
                "quant_type": "int4",
                "page_size": 128,
                "max_pages": 256,
                "max_seq_len": 2048,
                "max_new_tokens": 512,
                "use_compile": False,
                "device": "cuda:0",
                "description": "笔记本档：INT4 量化 + 中等上下文",
            }

    monkeypatch.setattr("device_profiler.get_profile", lambda: FakeProfiler())
    r = client.post("/device/auto-configure")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "configured"
    assert body["tier"] == "laptop"
    assert body["score"] == 85.5
    assert body["applied_config"]["page_size"] == 128
    assert body["recommendations"] == ["推荐使用 INT4 量化档位"]
    assert body["warnings"] == []


def test_sched_http_device_select_gpu_not_ready(sched_http_client, monkeypatch):
    """POST /device/select-gpu：画像未就绪 → 400。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", None)
    r = client.post("/device/select-gpu", json={"gpu_index": 0})
    assert r.status_code == 400
    assert "未就绪" in r.json()["detail"]


def test_sched_http_device_select_gpu_out_of_range(sched_http_client, monkeypatch):
    """POST /device/select-gpu：序号越界 → 400。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)
    r = client.post("/device/select-gpu", json={"gpu_index": 5})
    assert r.status_code == 400
    assert "无效的 GPU 序号" in r.json()["detail"]
    assert "0-0" in r.json()["detail"]


def test_sched_http_device_select_gpu_ok(sched_http_client, monkeypatch):
    """POST /device/select-gpu：正常切换 → 200 switched。"""
    import scheduler_svc_http as http_mod

    client, _ = sched_http_client
    monkeypatch.setattr(http_mod, "_device_profile_cache", FAKE_DEVICE_PROFILE)

    class FakeProfiler:
        def __init__(self):
            self.selected = None

        def select_gpu(self, index):
            self.selected = index
            return True

        def to_dict(self):
            return FAKE_DEVICE_PROFILE

        def recommend_config(self):
            return {"device": "cuda:0", "description": "x"}

    fake = FakeProfiler()
    monkeypatch.setattr("device_profiler.get_profile", lambda: fake)
    r = client.post("/device/select-gpu", json={"gpu_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "switched"
    assert body["selected_gpu_index"] == 0
    assert body["selected_gpu"]["name"] == "NVIDIA GeForce RTX 3060"
    assert body["selected_gpu"]["gpu_type"] == "discrete"
    assert body["device"] == "cuda:0"
    assert fake.selected == 0


# ----------------------------------------------------------------------
# 17.7 /v1/models 注册表 / available / current 完整契约（推理面缺口修复）
# ----------------------------------------------------------------------

def test_v1_models_registry_shape(live_inference_svc):
    """GET /v1/models：内置模型 + 文件状态 payload + active_model_id。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models", timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "active_model_id" in body
    assert isinstance(body["models"], list) and len(body["models"]) >= 1
    m = body["models"][0]
    # payload 完整字段（对齐 api_server _model_api_payload）
    for key in (
        "model_id", "name", "is_builtin", "model_type", "is_experimental",
        "recommended_vram_gb", "max_context", "quant_types", "description",
        "huggingface_id", "location", "model_path", "gguf_path",
        "is_available", "unavailable_reason", "available_formats",
        "has_safetensors", "has_gguf", "expected_paths",
        "supported_engines", "preferred_engine", "default_quant_type",
        "requires_cuda",
    ):
        assert key in m, f"payload 缺字段 {key}"
    # 至少一个内置模型 is_builtin
    assert any(x["is_builtin"] for x in body["models"])


def test_v1_models_available_shape(live_inference_svc):
    """GET /v1/models/available：quant 选项 + available_engines。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models/available", timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "available_engines" in body
    assert "current" in body and "current_engine" in body
    for quant in body["models"]:
        for key in ("id", "name", "engine", "is_available", "memory_gb",
                    "speed_tok_s", "compile_support", "description"):
            assert key in quant, f"quant 缺字段 {key}"
    for engine in body["available_engines"]:
        for key in ("id", "name", "description", "model_size_gb", "requires_cuda"):
            assert key in engine, f"engine 缺字段 {key}"


def test_v1_models_current_loaded_shape(live_inference_svc):
    """GET /v1/models/current 已加载：完整 10 字段（FakeEngineHost 对齐新契约）。"""
    import requests
    r = requests.get(f"{live_inference_svc}/v1/models/current", timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    for key in (
        "model_id", "quant_type", "model_name", "model_path", "engine",
        "total_params", "device", "gpu_allocated_gb", "gpu_reserved_gb",
    ):
        assert key in body, f"current 缺字段 {key}"


def test_v1_models_current_unloaded_shape_real_host():
    """真实 EngineHost 未加载：{loaded:False, quant_type:None, model_id:None}
    （不触发模型加载，EngineHost.__init__ 仅 import model_host）。"""
    from inference_service.engine_host import EngineHost

    host = EngineHost()
    assert host.current_model() == {
        "loaded": False, "quant_type": None, "model_id": None,
    }
