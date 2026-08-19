"""
T9 interactive 流式模式契约测试
==============================
覆盖 /api/chat/stream?streaming_mode=interactive 的 SSE 事件序列：
start → token* → done（含 generation_id/request_id/session_id/history_committed），
取消 → cancelled，失败 → error，以及路由偏好 metrics 与事务提交。
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from api_server import ChatGenerationCancelled


@pytest.fixture
def interactive_env(monkeypatch):
    """把调度/模型状态 mock 到 llama.cpp 假流式路径（run_pipeline_safe）。"""
    calls = {"run_pipeline_safe": None, "commits": [],
             "pipeline_stream": 0, "full_model_stream": 0, "forward": 0}

    def fake_run_pipeline_safe(message, **kwargs):
        calls["run_pipeline_safe"] = (message, kwargs)
        if calls.get("raise_cancelled"):
            raise ChatGenerationCancelled("gen_x")
        if calls.get("raise_error"):
            raise RuntimeError("engine exploded")
        return {
            "status": "ok",
            "response": "你好，世界！",
            "metrics": {"engine": "llama_cpp", "tokens_generated": 6},
            "error": None,
        }

    def fake_pipeline_stream(message, **kwargs):
        calls["pipeline_stream"] += 1
        return iter([
            {"token": "流"},
            {"token": "式"},
            {"done": True, "response": "流式",
             "metrics": {"engine": "pytorch_pipeline"}},
        ])

    def fake_full_model_stream(message, **kwargs):
        calls["full_model_stream"] += 1
        return iter([
            {"token": "单机"},
            {"done": True, "response": "单机",
             "metrics": {"engine": "pytorch"}},
        ])

    def fake_forward(message, **kwargs):
        calls["forward"] += 1
        return {"status": "ok", "content": "转发结果",
                "metrics": {"engine": "distributed_forward"}}

    def fake_chat(messages, **kwargs):
        calls["chat"] = (calls.get("chat") or 0) + 1
        return {
            "content": "full 本地回复",
            "tokens_per_second": 12.5,
            "usage": {"completion_tokens": 7},
            "followups": [],
        }

    fake_scheduler = SimpleNamespace(
        get_distributed_inference_enabled=lambda: calls.get(
            "distributed_enabled", False,
        ),
        _effective_role=lambda: calls.get("role", "master"),
        run_pipeline_safe=fake_run_pipeline_safe,
        run_pipeline_stream=fake_pipeline_stream,
        _run_full_model_inference_stream=fake_full_model_stream,
        forward_inference_to_master=fake_forward,
        record_task_complete=lambda success=True: None,
        get_effective_node_id=lambda: "master",
    )
    fake_model_manager = SimpleNamespace(
        is_loaded=True,
        _engine_type="llama_cpp",
        chat=fake_chat,
    )

    monkeypatch.setattr(api_server, "scheduler", fake_scheduler)
    monkeypatch.setattr(api_server, "model_manager", fake_model_manager)
    monkeypatch.setattr(
        api_server, "model_host",
        SimpleNamespace(model_loaded=True),
    )
    monkeypatch.setattr(api_server, "RUN_MODE", "local")

    def fake_commit(session_id, user_message, response_text, metrics):
        calls["commits"].append(
            (session_id, user_message, response_text, dict(metrics)),
        )
        return True

    monkeypatch.setattr(api_server, "_commit_interactive_history", fake_commit)
    monkeypatch.setattr(api_server, "_external_route_decision",
                        lambda req: SimpleNamespace(use_external=False))

    client = TestClient(api_server.app)
    return client, calls


def _sse_events(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


class TestInteractiveContract:
    """事件序列与字段契约。"""

    def test_start_then_tokens_then_done(self, interactive_env):
        client, calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "你好",
            "streaming_mode": "interactive",
            "session_id": "sess_t9",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _sse_events(response)
        # start → token → done
        assert events[0]["start"] is True
        assert events[0]["generation_id"].startswith("gen_")
        assert events[0]["session_id"] == "sess_t9"
        assert events[0]["routing_preference"] == "local_only"

        tokens = [e["token"] for e in events if "token" in e]
        assert tokens == ["你好，世界！"]

        done = events[-1]
        assert done["done"] is True
        assert done["response"] == "你好，世界！"
        assert done["generation_id"] == events[0]["generation_id"]
        assert done["session_id"] == "sess_t9"
        assert done["history_committed"] is True
        assert done["metrics"]["routing_preference"] == "local_only"
        assert done["metrics"]["distributed_requested"] is False

    def test_cancel_emits_cancelled_event(self, interactive_env):
        client, calls = interactive_env
        calls["raise_cancelled"] = True
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "generation_id": "gen_t9_cancel",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        cancelled = [e for e in events if e.get("cancelled")]
        assert len(cancelled) == 1
        assert cancelled[0]["generation_id"] == "gen_t9_cancel"
        assert "partial" in cancelled[0]
        assert not [e for e in events if e.get("done")]
        # 取消不提交历史
        assert calls["commits"] == []

    def test_error_emits_error_event(self, interactive_env):
        client, calls = interactive_env
        calls["raise_error"] = True
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "engine exploded" in error[0]["error"]
        assert calls["commits"] == []

    def test_commit_receives_user_and_response_together(self, interactive_env):
        client, calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "提交测试",
            "streaming_mode": "interactive",
            "session_id": "sess_commit",
        })
        assert response.status_code == 200
        assert len(calls["commits"]) == 1
        session_id, user_message, response_text, metrics = calls["commits"][0]
        assert session_id == "sess_commit"
        assert user_message == "提交测试"
        assert response_text == "你好，世界！"
        assert metrics["engine"] == "llama_cpp"

    def test_distributed_required_marks_requested(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, _engine_type="pytorch"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        done = _sse_events(response)[-1]
        assert done["metrics"]["distributed_requested"] is True
        assert done["metrics"]["distributed_used"] is True

    def test_invalid_routing_preference_rejected(self, interactive_env):
        client, _calls = interactive_env
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
            "routing_preference": "sideways",
        })
        assert response.status_code == 422

    def test_client_node_guides_to_master(self, interactive_env, monkeypatch):
        client, _calls = interactive_env
        # 从节点场景：转发条件成立
        fake_scheduler = SimpleNamespace(
            get_distributed_inference_enabled=lambda: True,
            _effective_role=lambda: "client",
            run_pipeline_safe=lambda **kw: {},
        )
        monkeypatch.setattr(api_server, "scheduler", fake_scheduler)
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        response = client.post("/api/chat/stream", json={
            "message": "hi",
            "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[0]["start"] is True
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "主节点" in error[0]["error"]


class TestRoutingPreference:
    """T9.5 请求级路由偏好：local_only / distributed_required / 回退 metrics。"""

    def _set_client_scene(self, env, monkeypatch):
        """从节点场景：分布式启用 + role=client。"""
        client, calls = env
        calls["distributed_enabled"] = True
        calls["role"] = "client"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        return client, calls

    def test_client_without_local_only_guides_to_master(self, interactive_env, monkeypatch):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "主节点" in error[0]["error"]
        assert calls["forward"] == 0

    def test_local_only_on_client_executes_locally(self, interactive_env, monkeypatch):
        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert events[-1]["response"] == "你好，世界！"
        assert calls["forward"] == 0  # 未转发主节点
        assert calls["run_pipeline_safe"] is not None  # 本地路径执行
        assert events[-1]["metrics"]["distributed_used"] is False

    def test_local_only_skips_pipeline_on_master(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, _engine_type="pytorch"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "local_only",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 0   # 流水线被跳过
        assert calls["full_model_stream"] == 1  # 单机 PyTorch 执行
        assert events[-1]["metrics"]["distributed_used"] is False

    def test_auto_uses_pipeline_when_available(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, _engine_type="pytorch"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 1
        assert events[-1]["metrics"]["distributed_used"] is True

    def test_distributed_required_fails_without_path(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]

    def test_distributed_required_allowed_with_path(self, interactive_env, monkeypatch):
        client, calls = interactive_env
        calls["distributed_enabled"] = True
        calls["role"] = "master"
        monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
        monkeypatch.setattr(
            api_server, "model_manager",
            SimpleNamespace(is_loaded=True, _engine_type="pytorch"),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_required",
        })
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["pipeline_stream"] == 1
        assert events[-1]["metrics"]["distributed_used"] is True

    def test_distributed_preferred_fallback_metrics(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用 → 本地回退
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "interactive",
            "routing_preference": "distributed_preferred",
        })
        events = _sse_events(response)
        done = events[-1]
        assert done["done"] is True
        assert done["metrics"]["distributed_used"] is False
        assert done["metrics"]["fallback"] is True
        assert "fallback_reason" in done["metrics"]

    def test_full_mode_local_only_executes_locally(self, interactive_env, monkeypatch):
        class _Lock:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        client, calls = self._set_client_scene(interactive_env, monkeypatch)
        monkeypatch.setattr(
            api_server, "model_host",
            SimpleNamespace(model_loaded=True,
                            full_chat_execution_lock=_Lock()),
        )
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "local_only",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        assert events[-1]["done"] is True
        assert calls["forward"] == 0

    def test_full_mode_distributed_required_rejected(self, interactive_env):
        client, _calls = interactive_env  # 分布式未启用
        response = client.post("/api/chat/stream", json={
            "message": "hi", "streaming_mode": "full",
            "routing_preference": "distributed_required",
        })
        assert response.status_code == 200
        events = _sse_events(response)
        error = [e for e in events if e.get("error")]
        assert len(error) == 1
        assert "distributed_required" in error[0]["error"]


class TestCommitFunction:
    """真实 _commit_interactive_history 的本地存储分支。"""

    def test_local_store_branch(self, monkeypatch):
        saved = []

        fake_store = SimpleNamespace(
            get_local_save_history=lambda: True,
            save_local_conversation_turn=(
                lambda sid, user, assistant, metrics=None, **kwargs:
                saved.append((sid, user, assistant, metrics)) or True
            ),
        )
        monkeypatch.setattr(api_server, "_local_store", fake_store)
        monkeypatch.setattr(
            api_server, "model_host",
            SimpleNamespace(),
        )
        committed = api_server._commit_interactive_history(
            "sess_l", "问", "答", {"engine": "llama_cpp"},
        )
        assert committed is True
        assert saved == [("sess_l", "问", "答", {"engine": "llama_cpp"})]
