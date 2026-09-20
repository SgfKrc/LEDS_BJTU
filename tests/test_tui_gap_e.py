"""★ TUI 缺口 E：集群高可用（2026-09-19）。

补的端点（`api_server.py`）：

| 端点 | 方法 | 危险 | 请求体 |
|---|---|---|---|
| `/api/cluster/master-health` | GET | 安全 | — |
| `/api/cluster/transfer-logs` | GET | 安全 | — |
| `/api/cluster/spare-master` | GET | 安全 | — |
| `/api/cluster/spare-master/logs` | GET | 安全 | — |
| `/api/cluster/spare-master` | POST | ⚠️ 中 | `{target_node_id}` |
| `/api/cluster/spare-master` | DELETE | ⚠️ 中 | — |
| `/api/cluster/transfer-master` | POST | ⚠️⚠️ **高** | `{target_node_id}`（转让后需重启） |
| `/api/cluster/reset-identity` | POST | ⚠️⚠️ **高** | `{confirm: "reset"}`（不可撤销） |

本测试锁定：路径常量、HTTP 方法/路径、**高危接口的请求体约束**
（`reset-identity` 必须带 `confirm="reset"`，否则后端 400）。
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

    def post(self, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("post", path, body, kw))
        return {"ok": True}

    def request(self, method: str, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("request", method, path, body, kw))
        return {"ok": True}


class TestEPaths:
    def test_all_six_paths(self):
        assert API_PATHS["cluster_master_health"] == "/cluster/master-health"
        assert API_PATHS["cluster_transfer_logs"] == "/cluster/transfer-logs"
        assert API_PATHS["cluster_spare_master"] == "/cluster/spare-master"
        assert API_PATHS["cluster_spare_master_logs"] == "/cluster/spare-master/logs"
        assert API_PATHS["cluster_transfer_master"] == "/cluster/transfer-master"
        assert API_PATHS["cluster_reset_identity"] == "/cluster/reset-identity"


class TestReadOnlyCalls:
    """只读四项。"""

    def test_master_health(self):
        api = FakeApi()
        tui_api.master_health(api)
        assert api.calls[0][0] == "get"
        assert api.calls[0][1] == "/cluster/master-health"

    def test_transfer_logs(self):
        api = FakeApi()
        tui_api.transfer_logs(api)
        assert api.calls[0][1] == "/cluster/transfer-logs"

    def test_get_spare_master(self):
        api = FakeApi()
        tui_api.get_spare_master(api)
        assert api.calls[0][1] == "/cluster/spare-master"

    def test_spare_master_logs(self):
        api = FakeApi()
        tui_api.spare_master_logs(api)
        assert api.calls[0][1] == "/cluster/spare-master/logs"


class TestWriteCalls:
    """写操作：方法/路径/请求体。"""

    def test_designate_spare_master_body(self):
        api = FakeApi()
        tui_api.designate_spare_master(api, "client-A")
        assert api.calls[0][0] == "post"
        assert api.calls[0][1] == "/cluster/spare-master"
        assert api.calls[0][2] == {"target_node_id": "client-A"}

    def test_clear_spare_master_uses_delete(self):
        api = FakeApi()
        tui_api.clear_spare_master(api)
        assert api.calls[0][0] == "request"
        assert api.calls[0][1] == "DELETE"
        assert api.calls[0][2] == "/cluster/spare-master"

    def test_transfer_master_body(self):
        api = FakeApi()
        tui_api.transfer_master(api, "client-B")
        assert api.calls[0][0] == "post"
        assert api.calls[0][1] == "/cluster/transfer-master"
        assert api.calls[0][2] == {"target_node_id": "client-B"}


class TestHighRiskResetIdentity:
    """⚠️ 高危：`reset-identity` 必须带 `confirm="reset"`，否则后端 400 拒绝。"""

    def test_sends_required_confirm_token(self):
        api = FakeApi()
        tui_api.reset_master_identity(api)
        assert api.calls[0][0] == "post"
        assert api.calls[0][1] == "/cluster/reset-identity"
        body = api.calls[0][2]
        assert isinstance(body, dict) and body.get("confirm") == "reset", (
            "后端要求 confirm == 'reset' 才允许重置身份（防误操作）"
        )


class TestCommandSpec:
    def test_ha_spec_exists_and_lists_high_risk(self):
        ha = next((c for c in COMMAND_SPECS if c["name"] == "/ha"), None)
        assert ha is not None, "命令表应包含 /ha"
        for token in ("health", "transfer-logs", "spare", "designate", "transfer",
                      "reset-identity"):
            assert token in ha["args"], f"/ha 用法应含 {token}"
        assert "高危" in ha["desc"], "/ha 说明应标注高危动作"


class TestNoRegression:
    def test_existing_logs_and_queue_still_declared(self):
        for key in ("logs_list", "logs_nodes_summary", "cluster_queue_task_cancel"):
            assert key in API_PATHS

    def test_ha_name_not_colliding(self):
        names = [c["name"] for c in COMMAND_SPECS]
        assert names.count("/ha") == 1
