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
        return {"loaded": True, "engine": "pytorch", "model_id": "qwen-1.8b"}

    def chat_full(self, req, cancel_event=None):
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
def test_speculative_run_gated_off(client):
    """SPEC_ENABLED 未启用（默认）→ 404 门控（复制 api_server 语义）。"""
    resp = client.post(
        "/v1/speculative/run", json={"prompt": "你好"}
    )
    assert resp.status_code == 404
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
# 14. 1.2c 完整聊天流程（复制自 api_server._execute_chat_full）
# ----------------------------------------------------------------------
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
