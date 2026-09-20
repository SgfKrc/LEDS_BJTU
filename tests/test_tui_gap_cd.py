"""★ TUI 缺口 C + D（2026-09-19）。

* **C. 模型资产浏览**（只读）：
  `/api/models/available`（可选模型 + 引擎）、`/api/models/registry`（已注册实验模型）、
  `/api/models/downloadable`（可下载清单）、`/api/models/gguf`（本地 GGUF）。
* **D. 存储健康**（只读）：`/api/db/health`（SQLite）、`/api/storage/health`。

此前 TUI 只能看**已登记**资产（`/models/local-assets`）与下载任务，无法浏览上述清单。
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


class TestCPaths:
    def test_asset_paths(self):
        assert API_PATHS["models_available"] == "/models/available"
        assert API_PATHS["models_registry"] == "/models/registry"
        assert API_PATHS["models_downloadable"] == "/models/downloadable"
        assert API_PATHS["models_gguf"] == "/models/gguf"


class TestDPaths:
    def test_storage_paths(self):
        assert API_PATHS["db_health"] == "/db/health"
        assert API_PATHS["storage_health"] == "/storage/health"


class TestAssetCalls:
    """C：四个资产查询均走 GET，且路径正确。"""

    def test_available(self):
        api = FakeApi()
        tui_api.list_models_available(api)
        assert api.calls[0] == ("get", "/models/available", {})

    def test_registry(self):
        api = FakeApi()
        tui_api.list_model_registry(api)
        assert api.calls[0][1] == "/models/registry"

    def test_downloadable(self):
        api = FakeApi()
        tui_api.list_models_downloadable(api)
        assert api.calls[0][1] == "/models/downloadable"

    def test_gguf(self):
        api = FakeApi()
        tui_api.list_local_gguf(api)
        assert api.calls[0][1] == "/models/gguf"


class TestStorageCalls:
    """D：两个健康查询。"""

    def test_db_health(self):
        api = FakeApi()
        tui_api.db_health(api)
        assert api.calls[0] == ("get", "/db/health", {})

    def test_storage_health(self):
        api = FakeApi()
        tui_api.storage_health(api)
        assert api.calls[0][1] == "/storage/health"


class TestCommandSpecs:
    def test_assets_spec(self):
        spec = next((c for c in COMMAND_SPECS if c["name"] == "/assets"), None)
        assert spec is not None, "命令表应包含 /assets"
        for token in ("available", "registry", "downloadable", "gguf"):
            assert token in spec["args"], f"/assets 用法应含 {token}"

    def test_storage_spec(self):
        assert any(c["name"] == "/storage" for c in COMMAND_SPECS), "命令表应包含 /storage"

    def test_no_duplicate_commands(self):
        names = [c["name"] for c in COMMAND_SPECS]
        assert len(names) == len(set(names)), f"命令名重复: {sorted({n for n in names if names.count(n) > 1})}"


class TestNoRegression:
    """既有能力不受影响，且新函数定义唯一（前轮曾出现重复定义）。"""

    def test_prior_gaps_still_declared(self):
        for key in ("cluster_queue_task_cancel", "logs_list", "logs_nodes_summary",
                    "cluster_master_health", "cluster_transfer_master"):
            assert key in API_PATHS

    def test_new_functions_defined_once(self):
        import pathlib
        import re

        src = pathlib.Path(tui_api.__file__).read_text(encoding="utf-8")
        for fn in ("list_models_available", "list_model_registry", "list_models_downloadable",
                   "list_local_gguf", "db_health", "storage_health"):
            assert len(re.findall(rf"^def {fn}\(", src, re.M)) == 1, f"{fn} 应只定义一次"
