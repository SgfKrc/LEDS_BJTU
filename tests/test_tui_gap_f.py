"""★ TUI 缺口 F：会话细粒度（2026-09-19）。

TUI 原已有 `/sessions`（列表）、`/new`、`/resume`、`/rename`、`/delete-session`、`/reset`；
**缺**：查看对话历史、会话详情、本地持久化状态、删单轮。

补的端点（`api_server.py`）：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/conversations?session_id=&limit=` | GET | 读对话历史 |
| `/api/conversations/sync-status` | GET | 本地持久化状态 |
| `/api/sessions/{session_id}` | GET | 会话元数据 |
| `/api/sessions/{session_id}/turns/{turn_index}` | DELETE | 删单轮（user + assistant） |
"""

from __future__ import annotations

from typing import Any

import tui_api
from tui_shared import API_PATHS, COMMAND_SPECS


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get(self, path: str, **kw: Any) -> dict:
        self.calls.append(("get", path, kw))
        return {"ok": True}

    def request(self, method: str, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("request", method, path, body, kw))
        return {"ok": True}


class TestFPaths:
    def test_new_paths(self):
        assert API_PATHS["session_info"] == "/sessions/{session_id}"
        assert API_PATHS["session_turn"] == "/sessions/{session_id}/turns/{turn_index}"
        assert API_PATHS["conversation_sync_status"] == "/conversations/sync-status"

    def test_existing_paths_intact(self):
        assert API_PATHS["conversations"] == "/conversations"
        assert API_PATHS["session_detail"] == "/sessions/{session_id}"
        assert API_PATHS["session_activate"] == "/sessions/{session_id}/activate"


class TestHistoryCalls:
    def test_get_conversation_query(self):
        api = FakeApi()
        tui_api.get_conversation(api, "s-1", 50)
        assert api.calls[0][0] == "get"
        assert api.calls[0][1] == "/conversations?session_id=s-1&limit=50"

    def test_get_conversation_defaults(self):
        api = FakeApi()
        tui_api.get_conversation(api)
        assert api.calls[0][1] == "/conversations?session_id=default&limit=200"

    def test_sync_status(self):
        api = FakeApi()
        tui_api.conversation_sync_status(api)
        assert api.calls[0] == ("get", "/conversations/sync-status", {})

    def test_session_info_quotes_id(self):
        api = FakeApi()
        tui_api.session_info(api, "a/b")
        assert api.calls[0][1] == "/sessions/a%2Fb"


class TestDeleteTurn:
    """删单轮是高影响操作（连带删两条消息），路径与索引必须正确。"""

    def test_uses_delete_with_index(self):
        api = FakeApi()
        tui_api.delete_turn(api, "s-1", 3)
        assert api.calls[0][0] == "request"
        assert api.calls[0][1] == "DELETE"
        assert api.calls[0][2] == "/sessions/s-1/turns/3"

    def test_index_coerced_to_int(self):
        api = FakeApi()
        tui_api.delete_turn(api, "s-1", "7")  # type: ignore[arg-type]
        assert api.calls[0][2].endswith("/turns/7")


class TestCommandSpec:
    def test_history_spec(self):
        spec = next((c for c in COMMAND_SPECS if c["name"] == "/history"), None)
        assert spec is not None, "命令表应包含 /history"
        for token in ("sync-status", "info", "drop-turn"):
            assert token in spec["args"], f"/history 用法应含 {token}"
        assert "确认" in spec["desc"], "删单轮应在说明中标注需确认"

    def test_no_duplicate_commands(self):
        names = [c["name"] for c in COMMAND_SPECS]
        assert len(names) == len(set(names))


class TestNoRegression:
    def test_prior_gaps_intact(self):
        for key in ("models_available", "db_health", "cluster_master_health",
                    "logs_list", "cluster_queue_task_cancel"):
            assert key in API_PATHS

    def test_new_functions_defined_once(self):
        import pathlib
        import re

        src = pathlib.Path(tui_api.__file__).read_text(encoding="utf-8")
        for fn in ("get_conversation", "conversation_sync_status", "session_info", "delete_turn"):
            assert len(re.findall(rf"^def {fn}\(", src, re.M)) == 1, f"{fn} 应只定义一次"
