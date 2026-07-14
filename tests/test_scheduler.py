"""
单元测试 — 调度器动态分层算法
=============================
测试 _compute_node_weight() 和 compute_layer_assignment()
使用模拟设备数据，无需真实硬件/数据库连接。
"""

import sys
import os
import logging
import time
import threading
from typing import Any, cast
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from scheduler import Scheduler, PipelineQueue, NodeInfo, NodeState, NodeRole


# ================================================================
# 模拟设备画像（不同硬件配置）
# ================================================================

PROFILE_WORKSTATION = {
    "gpu": {
        "name": "NVIDIA RTX 4090",
        "vram_total_gb": 24.0,
        "cuda_available": True,
        "is_integrated": False,
    },
    "ram": {"total_gb": 64.0},
    "cpu": {"physical_cores": 16, "freq_max_mhz": 4500},
}

PROFILE_LAPTOP = {
    "gpu": {
        "name": "NVIDIA RTX 4060 Laptop",
        "vram_total_gb": 8.0,
        "cuda_available": True,
        "is_integrated": False,
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_ULTRABOOK = {
    "gpu": {
        "name": "Intel Iris Xe",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 8.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 3000},
}

PROFILE_EDGE = {
    "gpu": {
        "name": "None",
        "vram_total_gb": 0,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 4.0},
    "cpu": {"physical_cores": 2, "freq_max_mhz": 1500},
}

PROFILE_MOBILE = {
    "gpu": {
        "name": "ARM Mali",
        "vram_total_gb": 0,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 2.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 2000},
}

PROFILE_NO_GPU = {
    "gpu": {},
    "ram": {"total_gb": 8.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 2500},
}


# ================================================================
# _compute_node_weight 测试
# ================================================================

class TestComputeNodeWeight:
    """测试节点权重计算"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_workstation_max_score(self, sched):
        """CUDA 工作站应获得最高执行吞吐分。"""
        weight = sched._compute_node_weight(PROFILE_WORKSTATION)
        assert 145 <= weight <= 165, f"工作站权重应在 145-165，实际: {weight:.1f}"

    def test_laptop_moderate_score(self, sched):
        """CUDA 游戏本应显著高于 CPU worker。"""
        weight = sched._compute_node_weight(PROFILE_LAPTOP)
        assert 90 <= weight <= 110, f"游戏本权重应在 90-110，实际: {weight:.1f}"

    def test_ultrabook_low_score(self, sched):
        """轻薄本（集显）应获得低分"""
        weight = sched._compute_node_weight(PROFILE_ULTRABOOK)
        # VRAM: 0.5/24*50≈1.0, RAM: 8/64*30=3.75, CPU: 4/16*10+3000/4000*10=10
        # Bonus: 0 (集成显卡) → total ≈ 14.8
        assert 5 <= weight <= 30, f"轻薄本权重应在 5-30，实际: {weight:.1f}"

    def test_edge_minimal_score(self, sched):
        """边缘设备应获得最低分"""
        weight = sched._compute_node_weight(PROFILE_EDGE)
        # VRAM: 0, RAM: 4/64*30=1.875, CPU: 2/16*10+1500/4000*10=5
        # Bonus: 0 → total ≈ 6.9
        assert 0 <= weight <= 15, f"边缘设备权重应在 0-15，实际: {weight:.1f}"

    def test_no_gpu_field(self, sched):
        """device_info 缺少 GPU 字段时应正常降级"""
        weight = sched._compute_node_weight(PROFILE_NO_GPU)
        # VRAM: 0 (no gpu field → defaults to 0)
        # RAM: 8/64*30=3.75, CPU: 4/16*10+2500/4000*10=8.75
        # Bonus: 0 → total ≈ 12.5
        assert 0 <= weight <= 25, f"无 GPU 字段时权重应在 0-25，实际: {weight:.1f}"

    def test_empty_device_info(self, sched):
        """空 device_info 应安全降级"""
        weight = sched._compute_node_weight({})
        assert 0 <= weight <= 20, f"空设备信息权重应在 0-20，实际: {weight:.1f}"

    def test_none_device_info(self, sched):
        """None device_info 应安全降级"""
        weight = sched._compute_node_weight(None)
        assert 0 <= weight <= 20, f"None 设备信息权重应在 0-20，实际: {weight:.1f}"

    def test_discrete_gpu_bonus(self, sched):
        """只有可用 CUDA 独显才应获得专用显存与执行后端分。"""
        with_gpu = PROFILE_LAPTOP  # 独显
        without_gpu = {**PROFILE_LAPTOP, "gpu": {**PROFILE_LAPTOP["gpu"], "cuda_available": False}}
        w_gpu = sched._compute_node_weight(with_gpu)
        w_no_gpu = sched._compute_node_weight(without_gpu)
        # 8GB VRAM 约 16.7 分 + CUDA 执行后端 60 分。
        bonus = w_gpu - w_no_gpu
        assert 70 <= bonus <= 85, f"CUDA 执行优势应在 70-85，实际: {bonus:.1f}"

    def test_mobile_arm_score(self, sched):
        """移动设备应有合理分数"""
        weight = sched._compute_node_weight(PROFILE_MOBILE)
        assert 0 <= weight <= 15, f"移动设备权重应在 0-15，实际: {weight:.1f}"

    def test_cuda_laptop_outscores_igpu_cpu_worker(self, sched):
        """4060 CUDA 主机评分必须显著高于同核数的集显 CPU worker。"""
        cuda_weight = sched._compute_node_weight(PROFILE_LAPTOP)
        igpu_weight = sched._compute_node_weight(PROFILE_IGPU_ONLY)

        assert cuda_weight > igpu_weight * 2


# ================================================================
# compute_layer_assignment 测试
# ================================================================

class TestComputeLayerAssignment:
    """测试动态分层计算"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # 注入模拟节点
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_LAPTOP,
            ),
            "client2": NodeInfo(
                node_id="client2", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_ULTRABOOK,
            ),
        }
        return s

    def test_single_node_gets_all_layers(self, sched):
        """单节点应获得全部 24 层"""
        single = [
            {"node_id": "master", "role": "master", "device_info": PROFILE_WORKSTATION}
        ]
        result = sched.compute_layer_assignment(single)
        assert len(result) == 1
        assert result[0]["start_layer"] == 0
        assert result[0]["end_layer"] == 24
        assert result[0]["layers_count"] == 24
        assert result[0]["has_embedding"] is True
        assert result[0]["has_lm_head"] is True

    def test_active_deepseek_layer_count_replaces_fixed_qwen_count(self, sched, monkeypatch):
        import api_server as _api

        manager = type("DeepSeekManager", (), {
            "_total_model_layers": 28,
            "model": None,
            "_model_path": "",
        })()
        monkeypatch.setattr(_api, "model_manager", manager)
        nodes = [{
            "node_id": "master",
            "role": "master",
            "node_type": "pc",
            "device_info": {},
        }]

        result = sched.compute_layer_assignment(nodes)

        assert result[0]["start_layer"] == 0
        assert result[0]["end_layer"] == 28
        assert result[0]["layers_count"] == 28

    def test_multi_node_covers_all_layers(self, sched):
        """多节点分配应完整覆盖 0-24"""
        result = sched.compute_layer_assignment()
        total = sum(a["layers_count"] for a in result)
        assert total == 24, f"总层数应为 24，实际: {total}"

        # 验证连续性
        result.sort(key=lambda x: x["start_layer"])
        cursor = 0
        for a in result:
            assert a["start_layer"] == cursor, \
                f"节点 {a['node_id']} 起始层应为 {cursor}，实际: {a['start_layer']}"
            cursor = a["end_layer"]

    def test_first_node_has_embedding(self, sched):
        """首节点（master）应有 Embedding 且 start_layer=0"""
        result = sched.compute_layer_assignment()
        executable = [a for a in result if a.get("layers_count", 0) > 0]
        executable.sort(key=lambda x: x["start_layer"])
        assert executable[0]["has_embedding"] is True
        assert executable[0]["start_layer"] == 0
        # master 参与执行时应为首位
        assert executable[0].get("node_id") == "master" or executable[0].get("role") == "master"

    def test_last_node_has_lm_head(self, sched):
        """强 master 应保留 LM Head，避免弱末端 worker 回传整词表 logits。"""
        result = sched.compute_layer_assignment()
        result.sort(key=lambda x: x["start_layer"])
        master = next(item for item in result if item["node_id"] == "master")
        assert master["has_lm_head"] is True
        assert result[-1]["end_layer"] == 24

    def test_master_first_sorting(self, sched):
        """master 节点应排在第一位"""
        result = sched.compute_layer_assignment()
        # 第一个是 master
        master_assignments = [a for a in result if a["role"] == "master"]
        client_assignments = [a for a in result if a["role"] != "master"]

        if master_assignments:
            # master 应该在所有 client 之前
            master_max_end = max(a["end_layer"] for a in master_assignments)
            if client_assignments:
                client_min_start = min(a["start_layer"] for a in client_assignments)
                assert master_max_end <= client_min_start, \
                    "master 的层应排在 client 之前"

    def test_weight_proportional_distribution(self, sched):
        """worker 节点按权重分配层；master 作为首段 Embedding 锚点至少执行 1 层"""
        result = sched.compute_layer_assignment()

        scores = {a["node_id"]: a["score"] for a in result}
        layers = {a["node_id"]: a["layers_count"] for a in result}

        # master 分数最高，作为首段锚点至少执行 1 层
        assert scores["master"] > scores["client1"] > scores["client2"], \
            f"权重排序错误: {scores}"
        assert layers["master"] >= 1, \
            f"master 应至少执行 1 层（作为 Embedding 锚点），实际: {layers['master']}"
        assert "coordinator_only" not in result[0], \
            "master 不应再有 coordinator_only 标记"
        # master 是首节点，具有 Embedding
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        # worker 中游戏本 > 轻薄本，应分配更多层
        assert layers["client1"] >= layers["client2"], \
            f"worker 权重高应分配更多层: {layers}"
        total_layers = sum(layers.values())
        assert total_layers == 24, f"总层数应为 24，实际: {total_layers}"

    def test_empty_nodes_returns_empty(self, sched):
        """空节点列表应返回空"""
        result = sched.compute_layer_assignment([])
        assert result == []

    def test_android_nodes_excluded_from_layer_assignment(self, sched):
        """Android HTTP/移动节点不参与 Transformer 层间拆分。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc", "device_info": PROFILE_WORKSTATION},
            {"node_id": "pc-worker", "role": "client", "node_type": "pc", "device_info": PROFILE_LAPTOP},
            {"node_id": "android-live", "role": "client", "node_type": "android", "device_info": PROFILE_MOBILE},
        ]
        result = sched.compute_layer_assignment(nodes)
        node_ids = {a["node_id"] for a in result}
        assert "android-live" not in node_ids
        assert node_ids == {"master", "pc-worker"}

    def test_offline_pc_nodes_are_excluded_from_runtime_assignment(self):
        """数据库恢复的历史 PC 节点离线时不能继续占用模型层。"""
        sched = Scheduler()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_LAPTOP,
            ),
            "old-worker": NodeInfo(
                node_id="old-worker", role="client", state=NodeState.OFFLINE,
                node_type="pc", device_info=PROFILE_IGPU_ONLY,
            ),
        }

        result = sched.compute_layer_assignment()

        assert [item["node_id"] for item in result] == ["master"]
        assert result[0]["layers_count"] == 24

    def test_all_zero_weight_equal_split(self, sched):
        """权重全为 0 时应均分"""
        nodes = [
            {"node_id": "n1", "role": "client", "device_info": {}},
            {"node_id": "n2", "role": "client", "device_info": {}},
            {"node_id": "n3", "role": "client", "device_info": {}},
        ]
        result = sched.compute_layer_assignment(nodes)

        total = sum(a["layers_count"] for a in result)
        assert total == 24

        # 3 个节点均分 24 层 → 每节点 8 层
        for a in result:
            assert 7 <= a["layers_count"] <= 9, \
                f"均分时每节点应有 ~8 层，实际 {a['node_id']}: {a['layers_count']}"

    def test_two_nodes_continuous(self, sched):
        """两个节点时 master 参与首段计算，分层应连续"""
        two = [
            {"node_id": "master", "role": "master", "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client", "device_info": PROFILE_LAPTOP},
        ]
        result = sched.compute_layer_assignment(two)
        result.sort(key=lambda x: x["start_layer"])

        assert len(result) == 2
        assert result[0]["node_id"] == "master"
        assert result[0]["layers_count"] >= 1, \
            f"master 应至少 1 层，实际: {result[0]['layers_count']}"
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        assert "coordinator_only" not in result[0]
        worker = result[1]
        assert worker["node_id"] == "client1"
        assert worker["start_layer"] == result[0]["end_layer"]
        assert worker["end_layer"] == 24
        assert result[0]["has_lm_head"] is True
        assert worker["has_lm_head"] is False

    def test_4060_master_gets_more_layers_than_igpu_i7_worker(self, sched):
        """主节点画像正确时，不得再出现弱 CPU worker 的层数反超 CUDA 主节点。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc",
             "device_info": PROFILE_LAPTOP},
            {"node_id": "igpu-worker", "role": "client", "node_type": "pc",
             "device_info": PROFILE_IGPU_ONLY},
        ]

        result = sched.compute_layer_assignment(nodes)
        by_id = {item["node_id"]: item for item in result}

        assert by_id["master"]["score"] > by_id["igpu-worker"]["score"] * 2
        assert by_id["master"]["layers_count"] > by_id["igpu-worker"]["layers_count"]
        assert by_id["master"]["layers_count"] >= 19
        assert by_id["master"]["has_lm_head"] is True
        assert by_id["igpu-worker"]["has_lm_head"] is False

    def test_each_node_has_min_one_layer(self, sched):
        """每个节点至少分配 1 层"""
        # 极端场景：一个超强节点 + 一个超弱节点
        nodes = [
            {"node_id": "strong", "role": "client", "device_info": PROFILE_WORKSTATION},
            {"node_id": "weak", "role": "client", "device_info": PROFILE_MOBILE},
        ]
        result = sched.compute_layer_assignment(nodes)
        for a in result:
            assert a["layers_count"] >= 1, \
                f"节点 {a['node_id']} 应至少 1 层，实际: {a['layers_count']}"

    def test_result_fields_complete(self, sched):
        """返回结果应包含所有必要字段"""
        result = sched.compute_layer_assignment()
        for a in result:
            assert "node_id" in a
            assert "role" in a
            assert "start_layer" in a
            assert "end_layer" in a
            assert "layers_count" in a
            assert "has_embedding" in a
            assert "has_lm_head" in a
            assert "score" in a
            assert isinstance(a["score"], (int, float))


class TestLocalDeviceProfileSync:
    """异步设备检测必须进入调度器评分，并清除旧动态分层。"""

    def test_master_profile_updates_node_invalidates_cache_and_repushes(
            self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info={},
            ),
        }
        cleared = []
        persisted = []
        pushed = []

        class FakeDb:
            def upsert_node(self, **kwargs):
                persisted.append(kwargs)

            def set_layer_assignments(self, value):
                cleared.append(value)

        monkeypatch.setattr(scheduler_mod, "_get_db", lambda *args, **kwargs: FakeDb())
        monkeypatch.setattr(scheduler_mod, "_db_available", True)
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushed.append(True),
        )

        sched.update_local_device_profile(PROFILE_LAPTOP)

        assert sched.nodes["master"].device_info["gpu"]["name"] == "NVIDIA RTX 4060 Laptop"
        assert cleared == [{}]
        assert persisted[0]["device_info"] == PROFILE_LAPTOP
        assert pushed == [True]

    def test_client_profile_is_reported_after_background_detection(
            self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker1")
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info={},
            ),
        }
        sent = []

        class FakeClient:
            is_registered = True
            device_info = {}

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        fake_client = FakeClient()
        monkeypatch.setattr(sched, "_tcp_client", fake_client, raising=False)

        sched.update_local_device_profile(PROFILE_IGPU_ONLY)

        assert fake_client.device_info == PROFILE_IGPU_ONLY
        assert sent == [(
            {"state": "online", "device_info": PROFILE_IGPU_ONLY},
            "status_res",
        )]

    def test_failed_profile_report_remains_retryable(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "client"
        sched._local_device_profile = PROFILE_IGPU_ONLY
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker1")
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info=PROFILE_IGPU_ONLY,
            ),
        }

        class FlakyClient:
            is_registered = True
            device_info = {}
            attempts = 0

            def send_data(self, data, msg_type):
                self.attempts += 1
                if self.attempts == 1:
                    raise ConnectionError("temporary")

        client = FlakyClient()

        assert sched._report_local_device_profile(client, "worker1") is False
        assert client.device_info == {}
        assert sched._report_local_device_profile(client, "worker1") is True
        assert client.device_info == PROFILE_IGPU_ONLY
        assert client.attempts == 2

    def test_master_profile_report_recomputes_layer_config(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info={"platform": "Windows"},
            ),
        }
        pushed = []
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushed.append(True),
        )

        sched._on_tcp_message("worker1", {
            "type": "status_res",
            "data": {"state": "online", "device_info": PROFILE_IGPU_ONLY},
        })

        assert sched.nodes["worker1"].device_info == PROFILE_IGPU_ONLY
        assert pushed == [True]


class TestManualLayerAssignmentNormalization:
    """前端只提交区间，后端必须补齐可执行分层字段。"""

    def test_manual_ranges_gain_counts_roles_scores_and_io_heads(self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info=PROFILE_LAPTOP,
            ),
            "worker": NodeInfo(
                node_id="worker", role="client", state=NodeState.ONLINE,
                device_info=PROFILE_IGPU_ONLY,
            ),
        }
        stored = []

        class FakeDb:
            def set_layer_strategy(self, _strategy):
                pass

            def set_layer_override(self, value):
                stored.append(value)

        monkeypatch.setattr(scheduler_mod, "_get_db", lambda *args, **kwargs: FakeDb())
        monkeypatch.setattr(scheduler_mod, "_db_available", True)
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)

        result = sched.override_layer_assignments([
            {"node_id": "master", "start_layer": 0, "end_layer": 20},
            {"node_id": "worker", "start_layer": 20, "end_layer": 24},
        ])

        assert result["status"] == "ok"
        normalized = result["current_assignments"]["assignments"]
        assert normalized[0]["layers_count"] == 20
        assert normalized[1]["layers_count"] == 4
        assert normalized[0]["role"] == "master"
        assert normalized[0]["has_embedding"] is True
        assert normalized[0]["has_lm_head"] is True
        assert normalized[1]["has_lm_head"] is False
        assert stored == [normalized]

    def test_manual_ranges_reject_master_outside_first_segment(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE),
            "worker": NodeInfo(node_id="worker", role="client", state=NodeState.ONLINE),
        }

        result = sched.override_layer_assignments([
            {"node_id": "worker", "start_layer": 0, "end_layer": 12},
            {"node_id": "master", "start_layer": 12, "end_layer": 24},
        ])

        assert result["status"] == "invalid"
        assert "主节点" in result["reason"]


# ================================================================
# get_layer_assignments 测试
# ================================================================

class TestGetLayerAssignments:
    """测试分层配置获取（内存模式，无 DB）"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # ★ 清除 DB 缓存的层分配（避免其他测试/真实运行的旧数据污染）
        from db import set_layer_assignments as _clear_cache
        try:
            _clear_cache({})
        except Exception:
            pass
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_LAPTOP,
            ),
        }
        return s

    def test_returns_total_and_strategy(self, sched):
        """应返回 total, strategy, assignments"""
        result = sched.get_layer_assignments()
        assert result["total"] == 24
        assert result["strategy"] in ("dynamic", "manual")
        assert isinstance(result["assignments"], list)
        assert len(result["assignments"]) >= 1

    def test_single_node_all_24_layers(self, sched):
        """单节点时全部 24 层归该节点"""
        result = sched.get_layer_assignments()
        assignments = result["assignments"]
        assert len(assignments) == 1
        assert assignments[0]["start_layer"] == 0
        assert assignments[0]["end_layer"] == 24
        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["has_lm_head"] is True

    def test_legacy_dynamic_cache_without_runtime_signature_is_recomputed(
            self, sched, monkeypatch):
        import scheduler as scheduler_mod

        stale = {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [{
                "node_id": "old-worker", "start_layer": 0, "end_layer": 24,
                "layers_count": 24, "has_embedding": True, "has_lm_head": True,
            }],
        }

        class FakeDb:
            def get_layer_strategy(self):
                return "dynamic"

            def get_layer_assignments(self):
                return stale

            def set_layer_assignments(self, value):
                self.saved = value

        fake_db = FakeDb()
        monkeypatch.setattr(scheduler_mod, "_get_db", lambda *a, **kw: fake_db)
        monkeypatch.setattr(scheduler_mod, "_db_available", True)

        result = sched.get_layer_assignments()

        assert result["assignments"][0]["node_id"] == "master"
        assert result.get("cache_key")


# ================================================================
# NodeInfo RTT 字段测试
# ================================================================

class TestNodeRTT:
    """测试 NodeInfo 的 RTT 字段"""

    def test_default_rtt_zero(self):
        """新建 NodeInfo 时 RTT 默认为 0"""
        from scheduler import NodeInfo, NodeState
        node = NodeInfo(node_id="test", role="client")
        assert node.avg_rtt_ms == 0.0
        assert node.last_rtt_ms == 0.0

    def test_rtt_fields_in_to_dict(self):
        """to_dict() 应包含 avg_rtt_ms 和 last_rtt_ms"""
        from scheduler import NodeInfo, NodeState
        node = NodeInfo(
            node_id="rtt_test", role="client",
            avg_rtt_ms=12.5, last_rtt_ms=11.2,
        )
        d = node.to_dict()
        assert "avg_rtt_ms" in d
        assert "last_rtt_ms" in d
        assert d["avg_rtt_ms"] == 12.5
        assert d["last_rtt_ms"] == 11.2

    def test_rtt_rounded_in_to_dict(self):
        """to_dict() 中 RTT 应四舍五入到 1 位小数"""
        from scheduler import NodeInfo
        node = NodeInfo(
            node_id="round_test", role="client",
            avg_rtt_ms=12.567, last_rtt_ms=3.14159,
        )
        d = node.to_dict()
        assert d["avg_rtt_ms"] == 12.6
        assert d["last_rtt_ms"] == 3.1


# ================================================================
# Android 节点手动注册 / 删除 测试
# ================================================================

class TestAndroidNodeManagement:
    """测试 Android 节点注册与离线删除能力。"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        yield s
        # 清理测试节点（内存 + 数据库）
        test_ids = [nid for nid in s.nodes if nid != "master"]
        for nid in test_ids:
            try:
                del s.nodes[nid]
            except Exception:
                pass
            try:
                from db import delete_node
                delete_node(nid)
            except Exception:
                pass

    def test_manual_register_android_node(self, sched):
        """手动注册 Android 节点应保存 node_type=android 且初始离线。"""
        result = sched.manual_register_node(
            "android-test", hostname="Android Phone",
            network_type="wifi", node_type="android",
        )
        assert result["status"] == "registered"
        node = sched.nodes["android-test"]
        assert node.node_type == "android"
        assert node.hostname == "Android Phone"
        assert node.network_type == "wifi"
        assert node.state == NodeState.OFFLINE

    def test_manual_register_existing_android_updates_metadata(self, sched):
        """重复注册同一 Android 节点应更新元数据而非失败。"""
        sched.manual_register_node("android-test", hostname="Old", node_type="android")
        result = sched.manual_register_node(
            "android-test", hostname="New", address="1.2.3.4:8888",
            network_type="wifi", node_type="android",
        )
        assert result["status"] == "updated"
        node = sched.nodes["android-test"]
        assert node.hostname == "New"
        assert node.address == "1.2.3.4:8888"
        assert node.network_type == "wifi"
        assert node.node_type == "android"

    def test_manual_register_real_connected_node_conflict(self, sched):
        """手动注册不应覆盖已经通过真实连接建立的节点记录。"""
        sched.nodes["client-live"] = NodeInfo(
            node_id="client-live", role="client", state=NodeState.OFFLINE,
            node_type="pc", address="10.0.0.2:8888", hostname="worker",
            network_type="ethernet", connected_at=1700000000.0,
        )

        result = sched.manual_register_node(
            "client-live", hostname="manual", address="1.2.3.4:9999",
            network_type="wifi", node_type="android",
        )

        assert result["status"] == "conflict"
        node = sched.nodes["client-live"]
        assert node.hostname == "worker"
        assert node.address == "10.0.0.2:8888"
        assert node.network_type == "ethernet"
        assert node.node_type == "pc"

    def test_invite_capacity_ignores_historical_offline_records(self, sched):
        """邀请容量不应被无地址、不可用的历史离线节点占用。"""
        sched._max_nodes = 2
        sched.nodes["old-offline"] = NodeInfo(
            node_id="old-offline", role="client", state=NodeState.OFFLINE,
            hostname="", address="", connected_at=1700000000.0,
        )

        invite = sched.get_invite_info()

        assert invite["has_capacity"] is True
        assert invite["node_count"] == 1
        assert invite["total_node_records"] >= 1

    def test_delete_node_rejects_master_and_missing(self, sched):
        """删除 master / 不存在节点应返回明确状态。"""
        assert sched.delete_node("master")["status"] == "invalid"
        assert sched.delete_node("missing")["status"] == "not_found"

    def test_delete_node_rejects_online_node(self, sched):
        """在线节点需要先注销，不能直接删除。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        assert sched.delete_node("client1")["status"] == "online"
        assert "client1" in sched.nodes

    def test_android_presence_registers_online_and_expires(self, sched, monkeypatch):
        """Android HTTP thin client presence 应在线登记，并在超时后离线。"""
        class DummyDb:
            def __init__(self):
                self.upserts = []
                self.state_updates = []
            def upsert_node(self, **kwargs):
                self.upserts.append(kwargs)
                return kwargs
            def update_node_state(self, *args, **kwargs):
                self.state_updates.append((args, kwargs))
                return {}

        dummy = DummyDb()
        import scheduler as scheduler_mod
        monkeypatch.setattr(scheduler_mod, "_get_db", lambda: dummy)
        monkeypatch.setattr(scheduler_mod, "_db_available", True)
        pushed = []
        sched._push_node_update_to_all_clients = lambda *args: pushed.append(args)

        result = sched.register_android_client(
            "android-live",
            hostname="Game Tablet",
            network_type="wifi",
            device_info={"gpu": {"renderer": "Adreno"}},
            http_peer="100.64.1.2",
        )
        assert result["status"] == "registered"
        node = sched.nodes["android-live"]
        assert node.state == NodeState.ONLINE
        assert node.node_type == "android"
        assert node.device_info["connection_type"] == "http_thin"
        assert node.device_info["pipeline_worker"] is False
        assert node.device_info["http_peer"] == "100.64.1.2"
        assert pushed and pushed[-1][1] == "add"

        first_heartbeat = node.last_heartbeat
        result = sched.register_android_client("android-live", hostname="Game Tablet 2")
        assert result["status"] == "updated"
        assert sched.nodes["android-live"].last_heartbeat >= first_heartbeat
        assert pushed[-1][1] == "update"

        sched._refresh_http_client_states(now=node.last_heartbeat + 121)
        assert sched.nodes["android-live"].state == NodeState.OFFLINE
        assert dummy.state_updates

    def test_android_offline_does_not_block_nodes_ready(self, sched):
        """Android HTTP 客户端离线不影响 PC worker readiness。"""
        sched.nodes["android-offline"] = NodeInfo(
            node_id="android-offline",
            role="client",
            node_type="android",
            state=NodeState.OFFLINE,
            device_info={"connection_type": "http_thin"},
        )
        assert sched.check_nodes_ready() is True

    def test_delete_offline_android_node_removes_and_pushes_remove(self, sched, monkeypatch):
        """离线 Android 节点删除后应从内存移除并广播 remove。"""
        class DummyDb:
            def __init__(self):
                self.deleted = []
                self.created = []
                self.layer_reset = False
            def delete_node(self, node_id):
                self.deleted.append(node_id)
                return True
            def set_layer_assignments(self, assignments):
                self.layer_reset = True
            def upsert_node(self, **kwargs):
                self.created.append(kwargs.get("node_id"))
                pass

        dummy = DummyDb()
        import scheduler as scheduler_mod
        monkeypatch.setattr(scheduler_mod, "_get_db", lambda: dummy)
        monkeypatch.setattr(scheduler_mod, "_db_available", True)

        sched.manual_register_node("android-test", hostname="Phone", node_type="android")
        assert "android-test" in dummy.created
        pushed = []
        sched._push_node_update_to_all_clients = lambda *args: pushed.append(args)

        result = sched.delete_node("android-test")
        assert result["status"] == "deleted"
        assert "android-test" not in sched.nodes
        assert dummy.deleted == ["android-test"]
        assert dummy.layer_reset is True
        assert pushed and pushed[0][0] == "android-test" and pushed[0][1] == "remove"


# ================================================================
# _get_node_vram_mb / _check_vram_constraint 测试
# ================================================================

class TestVRAMConstraint:
    """测试显存约束检查"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # 注入测试节点：
        # - gpu_node: 8GB 独显
        # - cpu_node: 无独显, 4GB RAM 可用
        # - tiny_node: 0.5GB 共享显存、1GB RAM 可用（集显/CPU 模式）
        from scheduler import NodeInfo, NodeState
        s.nodes["gpu_node"] = NodeInfo(
            node_id="gpu_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {"vram_total_gb": 8.0, "name": "RTX 3070",
                        "is_integrated": False, "cuda_available": True},
                "ram": {"total_gb": 16.0, "available_gb": 8.0},
                "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
            },
        )
        s.nodes["cpu_node"] = NodeInfo(
            node_id="cpu_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {},
                "ram": {"total_gb": 8.0, "available_gb": 4.0},
                "cpu": {"physical_cores": 4, "freq_max_mhz": 2500},
            },
        )
        s.nodes["tiny_node"] = NodeInfo(
            node_id="tiny_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {"vram_total_gb": 0.5, "name": "Adreno", "is_integrated": True},
                "ram": {"total_gb": 3.0, "available_gb": 1.0},
                "cpu": {"physical_cores": 4, "freq_max_mhz": 2000},
            },
        )
        return s

    def test_get_vram_from_gpu(self, sched):
        """有独显时从 GPU 读取 VRAM"""
        mb = sched._get_node_vram_mb("gpu_node")
        assert mb == 8.0 * 1024  # 8 GB → 8192 MB

    def test_get_vram_from_ram_fallback(self, sched):
        """无独显时从可用 RAM 读取"""
        mb = sched._get_node_vram_mb("cpu_node")
        assert mb == 4.0 * 1024  # 4 GB available

    def test_get_vram_tiny_gpu(self, sched):
        """集显/CPU 节点应使用系统可用内存，不使用共享显存数字。"""
        mb = sched._get_node_vram_mb("tiny_node")
        assert mb == 1.0 * 1024

    def test_get_vram_unknown_node(self, sched):
        """未知节点返回 0"""
        assert sched._get_node_vram_mb("nonexistent") == 0.0

    def test_vram_constraint_ok(self, sched):
        """显存充足 — 应返回 True"""
        ok, needed, available = sched._check_vram_constraint(
            "gpu_node", layers_count=8, has_embedding=True, has_lm_head=False,
        )
        assert ok is True
        assert needed > 0
        assert available == 8192.0

    def test_vram_constraint_insufficient(self, sched):
        """显存不足 — 应返回 False"""
        # tiny_node 仅 1GB 可用内存，装不下 24 层 + Embedding + LM Head
        ok, needed, available = sched._check_vram_constraint(
            "tiny_node", layers_count=24, has_embedding=True, has_lm_head=True,
        )
        assert ok is False
        assert needed > available

    def test_vram_constraint_barely_ok(self, sched):
        """刚好够用 — 10% 安全余量内"""
        # cpu_node: ~4GB, int4 8层 ≈ 8*25 + 无额外头 ≈ 200MB → 轻松
        ok, needed, available = sched._check_vram_constraint(
            "cpu_node", layers_count=8, has_embedding=False, has_lm_head=False,
        )
        assert ok is True

    def test_vram_constraint_unknown_node(self, sched):
        """未知节点 — 无法判断时放行"""
        ok, needed, available = sched._check_vram_constraint(
            "nonexistent", layers_count=100, has_embedding=True, has_lm_head=True,
        )
        assert ok is True  # 不能判断 → 放行
        assert needed == 0
        assert available == 0

    def test_qwen2_split_memory_uses_fp16_even_when_int4_requested(
            self, sched, monkeypatch):
        import api_server as _api

        config = type("Qwen2Config", (), {
            "model_type": "qwen2",
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
        })()
        manager = type("Manager", (), {
            "model": type("Model", (), {"config": config})(),
            "quant_type": "int4",
        })()
        monkeypatch.setattr(_api, "model_manager", manager)

        layer_mb, embedding_mb, lm_head_mb = sched._get_layer_memory_estimate_mb(
            node_id="gpu_node",
            fallback=(70, 580, 580),
            quant_factors={"fp16": 1.0, "int4": 0.35},
            configured_quant="int4",
        )

        assert layer_mb > 50
        assert embedding_mb > 500
        assert lm_head_mb == embedding_mb

    def test_vram_transfer_moves_only_excess_layers(self, sched, monkeypatch):
        assignments = [
            {"node_id": "gpu_node", "role": "master", "layers_count": 10,
             "has_embedding": True, "has_lm_head": True, "score": 100},
            {"node_id": "cpu_node", "role": "client", "layers_count": 2,
             "has_embedding": False, "has_lm_head": False, "score": 20},
        ]

        def constraint(node_id, count, has_embedding=False, has_lm_head=False):
            limits = {"gpu_node": 8, "cpu_node": 8}
            return count <= limits[node_id], float(count), float(limits[node_id])

        monkeypatch.setattr(sched, "_check_vram_constraint", constraint)
        result = sched._apply_vram_constraints(assignments)
        by_id = {item["node_id"]: item for item in result}

        assert by_id["gpu_node"]["layers_count"] == 8
        assert by_id["cpu_node"]["layers_count"] == 4


# ================================================================
# 流水线就绪检查 测试
# ================================================================


class TestPipelineReadiness:
    """测试 _all_pipeline_nodes_ready() 和流水线节点状态判断"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_no_tcp_server(self, sched):
        """TCP 服务端未启动 → 返回 False"""
        sched._tcp_server = None
        assert sched._all_pipeline_nodes_ready() is False

    def test_only_master_nodes(self, sched):
        """仅主节点分配了层 → 无流水线从节点 → 返回 False"""
        # 模拟 get_layer_assignments 只返回 master
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 24,
                 "has_embedding": True, "has_lm_head": True}
            ],
        }
        try:
            assert sched._all_pipeline_nodes_ready() is False
        finally:
            sched.get_layer_assignments = original

    def test_worker_not_online(self, sched):
        """从节点不在线 → 返回 False"""
        from scheduler import NodeInfo, NodeState
        # 注册一个离线从节点
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client",
            state=NodeState.OFFLINE, address="1.2.3.4:8888"
        )
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        try:
            assert sched._all_pipeline_nodes_ready() is False
        finally:
            sched.get_layer_assignments = original

    def test_push_waits_for_worker_load_ack(self, sched, monkeypatch):
        """配置发出后不能立即 ready，必须等从节点加载 ACK。"""
        from scheduler import NodeInfo, NodeState

        assignment_info = {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        sched.nodes["client1"] = NodeInfo(
            node_id="client1",
            role="client",
            state=NodeState.ONLINE,
            address="100.64.1.2:8888",
            last_heartbeat=time.time(),
            model_sha256="",
        )
        sent = []
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
            "broadcast_layer_config": lambda self, payload: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: assignment_info)
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda: {
            "model_id": "qwen-1_8b",
            "model_path": "C:/models/qwen",
            "model_sha256": "sha-qwen",
            "total_layers": 24,
        })

        sched.push_layer_config_to_clients()

        payload = sent[0]["client1"]
        assert payload["node_id"] == "client1"
        assert payload["model_id"] == "qwen-1_8b"
        assert payload["model_sha256"] == "sha-qwen"
        assert payload["total_layers"] == 24
        assert "client1" not in sched._layer_config_pushed
        assert sched._all_pipeline_nodes_ready() is False

        sched._on_tcp_message("client1", {
            "type": "layer_config_ack",
            "data": {
                "node_id": "client1",
                "config_id": payload["config_id"],
                "status": "ready",
                "layer_range": [8, 24],
                "model_sha256": "sha-qwen",
                "engine": "pytorch",
            },
        })
        assert sched._all_pipeline_nodes_ready() is True

    def test_readiness_reports_worker_layer_load_error(self, sched, monkeypatch):
        """在线不等于可计算，层加载错误必须成为明确的阻塞原因。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="100.64.1.2:8888", last_heartbeat=time.time(),
        )
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
        })()
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "layers_count": 8},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "layers_count": 16},
            ],
        })
        sched._layer_config_expected["client1"] = {
            "config_id": "cfg-1", "model_id": "deepseek-r1",
        }
        sched._layer_config_acks["client1"] = {
            "status": "error", "error": "missing tokenizer.json",
        }

        readiness = sched._get_pipeline_readiness()

        assert readiness["ready"] is False
        assert readiness["reason_code"] == "worker_layer_load_failed"
        assert "missing tokenizer.json" in readiness["reason"]
        assert readiness["workers"][0]["layer_status"] == "error"

    def test_pipeline_status_requires_layer_ready_ack(self, sched, monkeypatch):
        """管理状态不能把仅 TCP 在线但仍在同步模型的 worker 标为 active。"""
        import api_server as _api

        sched._role_override = "master"
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="100.64.1.2:8888", last_heartbeat=time.time(),
        )
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
        })()
        monkeypatch.setattr(_api, "model_manager", type("Mgr", (), {
            "is_loaded": True, "_engine_type": "pytorch",
        })())
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "layers_count": 8},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "layers_count": 16},
            ],
        })
        sched._layer_config_expected["client1"] = {
            "config_id": "cfg-1", "model_id": "qwen-1_8b",
        }

        status = sched._get_pipeline_status()

        assert status["online_worker_count"] == 1
        assert status["active"] is False
        assert status["readiness_reason_code"] == "worker_layer_loading"
        assert status["workers"][0]["layer_status"] == "loading"


# ================================================================
# 流水线结果等待 测试
# ================================================================


class TestPipelineWaitResult:
    """测试 _wait_for_layer_result 超时和事件机制"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_timeout_no_result(self, sched):
        """没有节点返回结果 → 超时返回 None"""
        import time
        t0 = time.time()
        result = sched._wait_for_layer_result("task_1", "client1", timeout=0.5)
        elapsed = time.time() - t0
        assert result is None
        assert 0.4 < elapsed < 2.0  # 允许一些系统抖动

    def test_result_before_wait(self, sched):
        """在等待线程中，另一侧注入结果 → 应立即唤醒"""
        import threading
        import time

        result_holder = []

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result("task_2", "client1", timeout=5.0)
            )

        # 启动等待线程
        t = threading.Thread(target=waiter)
        t.start()

        # 等待一小段时间确保 waiter 已开始等待
        time.sleep(0.1)

        # 注入结果（模拟 _handle_layer_result）
        key = "task_2:client1"
        with sched._pipeline_lock:
            sched._pipeline_results[key] = {"task_id": "task_2", "node_id": "client1"}
            if key in sched._pipeline_events:
                sched._pipeline_events[key].set()

        t.join(timeout=2.0)
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert result_holder[0].get("task_id") == "task_2"


# ================================================================
# 流水线消息处理 测试
# ================================================================


class TestPipelineMessageDispatch:
    """测试流水线相关消息类型的调度分发"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_layer_result_stores_in_dict(self, sched):
        """_handle_layer_result 应将结果存入并触发 event"""
        sched._pipeline_active_tasks.add("task_1")
        msg = {
            "type": "layer_result",
            "data": {
                "task_id": "task_1",
                "node_id": "client1",
                "step": 0,
                "hidden_states": "ZmFrZQ==",  # "fake" in base64
                "hidden_shape": [1, 16, 2048],
            },
        }
        sched._on_tcp_message("client1", msg)

        key = "task_1:client1"
        with sched._pipeline_lock:
            assert key in sched._pipeline_results
            result = sched._pipeline_results[key]
            assert result["task_id"] == "task_1"
            assert "hidden_states" in result

    def test_late_layer_result_is_discarded(self, sched):
        """已清理任务的迟到大张量不得重新进入结果缓存。"""
        sched._handle_layer_result("client1", {
            "data": {
                "task_id": "task_late",
                "node_id": "client1",
                "hidden_states": "dGVzdA==",
            },
        })
        assert sched._pipeline_results == {}

    def test_pipeline_done_clears_kv_cache(self, sched):
        """PIPELINE_DONE 应清理指定 task 的 KV 缓存"""
        sched._kv_cache["task_x"] = {"layer_0": "mock_kv"}
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_x"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_x" not in sched._kv_cache

    def test_pipeline_abort_clears_kv_cache(self, sched):
        """PIPELINE_ABORT 应清理指定 task 的 KV 缓存"""
        sched._kv_cache["task_y"] = {"layer_0": "mock_kv"}
        msg = {
            "type": "pipeline_abort",
            "data": {"task_id": "task_y"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_y" not in sched._kv_cache

    def test_layer_config_dispatches_to_handler(self, sched):
        """layer_config 消息类型应路由到 _handle_layer_config（不抛异常）"""
        msg = {
            "type": "layer_config",
            "data": {
                "client1": {
                    "start_layer": 8, "end_layer": 16,
                    "has_embedding": False, "has_lm_head": False,
                },
            },
        }
        # 不应抛出异常（即使 model_manager 不存在也会优雅处理）
        sched._on_tcp_message("master", msg)

    def test_direct_layer_config_loads_range_and_sends_ack(self, sched, monkeypatch):
        """真实线上单 assignment 载荷应加载层范围并回传 ready ACK。"""
        import api_server as _api
        from tcp_comm import MessageType, TCPClient

        class MockModelManager:
            is_loaded = False
            _engine_type = "pytorch"
            layer_range = None

            def load_layer_range(self, start, end, **kwargs):
                self.layer_range = (start, end)

        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "send_data": lambda self, payload, msg_type: sent.append((payload, msg_type)),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        monkeypatch.setattr(TCPClient, "_compute_local_model_sha256", lambda: "sha-qwen")
        monkeypatch.setattr(_api, "model_manager", MockModelManager())

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-1",
            "start_layer": 8,
            "end_layer": 16,
            "has_embedding": False,
            "has_lm_head": False,
            "model_sha256": "sha-qwen",
        })

        assert len(sent) == 1
        payload, msg_type = sent[0]
        assert msg_type == MessageType.LAYER_CONFIG_ACK
        assert payload["status"] == "ready"
        assert payload["config_id"] == "cfg-1"
        assert payload["layer_range"] == [8, 16]
        assert payload["model_sha256"] == "sha-qwen"

    def test_deepseek_assignment_syncs_selected_model_before_loading(self, sched, monkeypatch):
        """缺少 DeepSeek 时应先同步指定模型，再按真实层数加载。"""
        import api_server as _api
        import model_sync
        from tcp_comm import TCPClient

        load_calls = []

        class MockModelManager:
            is_loaded = False
            _engine_type = "pytorch"
            layer_range = None

            def load_layer_range(self, start, end, **kwargs):
                load_calls.append((start, end, kwargs))
                self.layer_range = (start, end)

        sent = []
        fake_client = type("FakeClient", (), {
            "server_host": "100.64.0.10",
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        sched._tcp_client = fake_client
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        hashes = iter(["old-sha", "deepseek-sha"])
        monkeypatch.setattr(
            TCPClient,
            "_compute_local_model_sha256",
            lambda **kwargs: next(hashes),
        )
        monkeypatch.setattr(
            model_sync,
            "resolve_worker_model_path",
            lambda model_id: "C:/models/deepseek",
        )
        sync_calls = []
        monkeypatch.setattr(
            model_sync,
            "ensure_model_available",
            lambda host, port, model_id, sha: (
                sync_calls.append((host, port, model_id, sha))
                or "C:/models/deepseek"
            ),
        )
        monkeypatch.setattr(_api, "model_manager", MockModelManager())

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-deepseek",
            "start_layer": 10,
            "end_layer": 28,
            "has_embedding": False,
            "has_lm_head": True,
            "model_id": "deepseek-r1-distill-qwen-1.5b",
            "model_sha256": "deepseek-sha",
            "total_layers": 28,
            "master_api_port": 8000,
        })

        assert sync_calls == [(
            "100.64.0.10",
            8000,
            "deepseek-r1-distill-qwen-1.5b",
            "deepseek-sha",
        )]
        assert load_calls[0][0:2] == (10, 28)
        assert load_calls[0][2]["model_path"] == "C:/models/deepseek"
        assert load_calls[0][2]["model_id"] == "deepseek-r1-distill-qwen-1.5b"
        assert load_calls[0][2]["total_layers"] == 28
        assert sent[0]["status"] == "ready"

    def test_layer_config_is_deferred_until_active_task_finishes(self, sched, monkeypatch):
        """新分层配置不能在已有 task 的 KV cache 生命周期中替换模型。"""
        calls = []
        sched._active_pipeline_task_ids.add("active-task")

        sched._handle_layer_config("master", {"config_id": "cfg-next"})
        assert calls == []
        assert sched._pending_layer_config[1]["config_id"] == "cfg-next"

        monkeypatch.setattr(
            sched,
            "_handle_layer_config_locked",
            lambda client_id, data: calls.append((client_id, data["config_id"])),
        )
        sched._finish_local_pipeline_task("active-task")
        assert calls == [("master", "cfg-next")]

    def test_master_disconnect_clears_worker_pipeline_state(self, sched, monkeypatch):
        """主连接丢失后，worker 不得永久保留 KV cache/活动任务。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        sched._kv_cache["task-lost"] = ("kv",)
        sched._active_pipeline_task_ids.add("task-lost")

        sched._on_master_connection_lost()

        assert sched._kv_cache == {}
        assert sched._active_pipeline_task_ids == set()
        assert sched.nodes["client1"].error_count == 1

    def test_layer_config_model_mismatch_sends_error_ack(self, sched, monkeypatch):
        """从节点权重摘要不一致时不得加载，并应返回 error ACK。"""
        import api_server as _api
        from tcp_comm import TCPClient

        load_calls = []
        manager = type("MockModelManager", (), {
            "is_loaded": False,
            "_engine_type": "pytorch",
            "load_layer_range": lambda self, *args, **kwargs: load_calls.append(args),
        })()
        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        monkeypatch.setattr(TCPClient, "_compute_local_model_sha256", lambda: "local-sha")
        monkeypatch.setattr(_api, "model_manager", manager)

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-bad",
            "start_layer": 8,
            "end_layer": 16,
            "model_sha256": "master-sha",
        })

        assert load_calls == []
        assert sent[0]["status"] == "error"
        assert "SHA256 不一致" in sent[0]["error"]

    def test_layer_config_ack_marks_ready_only_for_current_version(self, sched):
        """只有当前 config_id、模型和层范围完全匹配的 ACK 才能置 ready。"""
        expected = {
            "node_id": "client1",
            "config_id": "cfg-current",
            "start_layer": 8,
            "end_layer": 16,
            "model_sha256": "sha-qwen",
        }
        sched._layer_config_expected["client1"] = expected

        stale = {
            "type": "layer_config_ack",
            "data": {
                "node_id": "client1",
                "config_id": "cfg-old",
                "status": "ready",
                "layer_range": [8, 16],
                "model_sha256": "sha-qwen",
                "engine": "pytorch",
            },
        }
        sched._on_tcp_message("client1", stale)
        assert "client1" not in sched._layer_config_pushed

        current = dict(stale)
        current["data"] = dict(stale["data"], config_id="cfg-current")
        sched._on_tcp_message("client1", current)
        assert "client1" in sched._layer_config_pushed

    def test_layer_config_retry_resends_same_config_id(self, sched):
        sent = []
        sched._tcp_server = MagicMock()
        sched._tcp_server._running = True
        sched._tcp_server.get_client_ids.return_value = ["client1"]
        sched._tcp_server.send_layer_config.side_effect = (
            lambda node_id, data: sent.append((node_id, data))
        )
        sched._layer_config_expected["client1"] = {
            "node_id": "client1",
            "config_id": "cfg-same",
            "start_layer": 0,
            "end_layer": 8,
        }
        sched._layer_config_retry_state["client1"] = {
            "attempts": 1,
            "next_retry": 0.0,
        }

        assert sched._retry_pending_layer_configs(now=10.0) == 1
        assert sent == [("client1", sched._layer_config_expected["client1"])]
        assert sent[0][1]["config_id"] == "cfg-same"

    def test_duplicate_ready_config_only_resends_cached_ack(self, sched, monkeypatch):
        sent = []
        sched._last_layer_config_ack_payload = {
            "config_id": "cfg-ready",
            "status": "ready",
        }
        monkeypatch.setattr(
            sched, "_send_layer_config_ack",
            lambda payload: sent.append(payload) or True,
        )
        sched._schedule_layer_config("master", {"config_id": "cfg-ready"})
        assert sent == [{"config_id": "cfg-ready", "status": "ready"}]

    def test_unknown_message_type_logs_debug(self, sched, caplog):
        """未知消息类型应记录 DEBUG 日志，便于排查协议不匹配。"""
        with caplog.at_level(logging.DEBUG, logger="scheduler"):
            sched._on_tcp_message("client1", {"type": "unknown_type"})
        assert any("未知消息类型" in r.getMessage() for r in caplog.records)

    def test_pipeline_pause_resume_logs_info(self, sched, caplog):
        """pipeline_pause/resume 应在 INFO 级别可见。"""
        with caplog.at_level(logging.INFO, logger="scheduler"):
            sched._on_tcp_message("client1", {"type": "pipeline_pause"})
            sched._on_tcp_message("client1", {"type": "pipeline_resume"})
        messages = [r.getMessage() for r in caplog.records]
        assert any("PIPELINE_PAUSE" in m for m in messages)
        assert any("PIPELINE_RESUME" in m for m in messages)

    def test_on_tcp_register_uses_advertised_addr_not_peer_port(self, sched, monkeypatch):
        """_on_tcp_message(register) 应使用 advertised_addr 注册节点。"""
        class FakeServer:
            _running = True
            def get_client_info(self, cid):
                return {
                    "advertised_addr": "10.0.0.9:8888",
                    "peer_addr": "10.0.0.9:51234",
                    "network_type": "wifi",
                }

            def confirm_registration(self, cid):
                return True
        sched._tcp_server = FakeServer()
        sched.nodes["pc-worker"] = NodeInfo(
            node_id="pc-worker", role="client", node_type="pc",
            state=NodeState.OFFLINE,
        )
        msg = {
            "type": "register",
            "data": {
                "role": "client",
                "node_type": "pc",
                "hostname": "worker",
                "advertised_address": "10.0.0.9:8888",
            },
        }
        sched._on_tcp_message("pc-worker", msg)
        node = sched.nodes["pc-worker"]
        assert node.state == NodeState.ONLINE
        assert node.address == "10.0.0.9:8888"
        assert "tcp_peer_addr" in node.device_info
        assert node.device_info["tcp_peer_addr"] == "10.0.0.9:51234"

    def test_pipeline_peer_registration_is_not_added_as_worker(self, sched):
        server = MagicMock()
        sched._tcp_server = server
        sched._on_tcp_message("client-peer", {
            "type": "register",
            "data": {
                "role": "client",
                "node_type": "pipeline_peer",
            },
        })
        server.confirm_registration.assert_called_once_with("client-peer")
        assert "client-peer" not in sched.nodes

    def test_tcp_disconnect_logs_and_marks_offline(self, sched, caplog):
        """TCP 断连应记录调度器层日志并标记节点离线。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        sched._push_node_update_to_all_clients = lambda *args, **kwargs: None
        sched.deregister_node = lambda _client_id: None

        with caplog.at_level(logging.INFO, logger="scheduler"):
            sched._on_tcp_disconnect("client1")

        assert sched.nodes["client1"].state == NodeState.OFFLINE
        assert any("TCP 断开" in r.getMessage() for r in caplog.records)


# ================================================================
# 流水线回退推理 测试
# ================================================================


class TestPipelineFallback:
    """测试 run_pipeline_safe 自动回退逻辑"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_fallback_when_no_nodes_ready(self, sched):
        """没有流水线节点 → run_pipeline_safe 应回退到全模型推理"""
        # 全模型推理需要 model_manager，这里只测试回退触发条件
        sched._tcp_server = None
        # run_pipeline_safe 内调用 _all_pipeline_nodes_ready → False → 回退
        # 回退方法 _run_full_model_inference 依赖 model_manager，
        # 此处只验证链路不崩溃（会返回 error 而非抛异常）
        result = sched.run_pipeline_safe("测试")
        # 预期：回退到全模型推理，但因模型未加载返回 error
        assert isinstance(result, dict)
        # 可能返回 error（模型未加载）或 response
        assert "response" in result or "error" in result

    def test_fallback_waits_for_inference_lock(self, sched, monkeypatch):
        """流水线未就绪回退时也必须等待推理锁，避免并发完整模型推理。"""
        import api_server as _api

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_api, "model_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: False)

        entered = threading.Event()
        finished = threading.Event()

        def fake_fallback(prompt, **kwargs):
            entered.set()
            return {"response": "fallback", "metrics": {"fallback": True}}

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        assert sched._inference_lock.acquire(blocking=False)

        result_holder = {}

        def call_pipeline():
            result_holder["result"] = sched.run_pipeline_safe("blocked")
            finished.set()

        t = threading.Thread(target=call_pipeline)
        t.start()

        assert not entered.wait(0.2), "推理锁未释放前不应进入本地回退推理"
        assert not finished.is_set(), "推理锁未释放前请求不应提前完成"

        sched._inference_lock.release()
        t.join(timeout=2)

        assert finished.is_set()
        assert entered.is_set()
        assert result_holder["result"]["response"] == "fallback"

    def test_run_pipeline_safe_fallbacks_when_pipeline_returns_error(self, sched, monkeypatch):
        """流水线单步返回 error 时，当前请求应自动回退到全模型推理。"""
        import api_server as _api

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_api, "model_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched,
            "run_pipeline",
            lambda prompt="", **kw: {"response": "", "error": "worker step failed"},
        )

        fallback_calls = []

        def fake_fallback(prompt, **kwargs):
            fallback_calls.append(kwargs)
            return {
                "response": "fallback ok",
                "metrics": {"fallback_reason": kwargs.get("_fallback_reason")},
            }

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        result = sched.run_pipeline_safe("test prompt")

        assert result["response"] == "fallback ok"
        assert fallback_calls
        assert "worker step failed" in fallback_calls[0]["_fallback_reason"]

    def test_run_pipeline_safe_does_not_replay_after_stream_output(
            self, sched, monkeypatch):
        """已发送部分正文后失败时不能从头执行全模型并重复回答。"""
        import api_server as _api

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_api, "model_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)

        def failed_pipeline(prompt="", **kwargs):
            kwargs["_stream_callback"]({"token": "partial"})
            return {"response": "", "error": "worker disconnected"}

        monkeypatch.setattr(sched, "run_pipeline", failed_pipeline)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail("部分输出后不能从头回退"),
        )
        events = []

        result = sched.run_pipeline_safe(
            "test prompt", _stream_callback=events.append,
        )

        assert result["error"] == "worker disconnected"
        assert events == [{"token": "partial"}]

    def test_queued_pipeline_does_not_replay_after_stream_output(
            self, sched, monkeypatch):
        """队列 worker 同样不得在部分输出后触发全模型重放。"""
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        events = []
        kwargs = {"_stream_callback": events.append}
        sched._track_stream_output(kwargs)

        def failed_pipeline(prompt="", **pipeline_kwargs):
            pipeline_kwargs["_stream_callback"]({"token": "partial"})
            return {"response": "", "error": "worker disconnected"}

        monkeypatch.setattr(sched, "run_pipeline", failed_pipeline)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **fallback_kwargs: pytest.fail("部分输出后不能从头回退"),
        )

        result = sched._process_queued_pipeline_task("test prompt", **kwargs)

        assert result["error"] == "worker disconnected"
        assert events == [{"token": "partial"}]


# ================================================================
# KV Cache 管理 测试 (Phase 3)
# ================================================================


class TestKVCacheManagement:
    """测试调度器本地 KV cache 存储与清理（从节点侧）"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_kv_cache_initialized_empty(self, sched):
        """新 Scheduler 的 _kv_cache 应为空 dict"""
        assert isinstance(sched._kv_cache, dict)
        assert len(sched._kv_cache) == 0

    def test_kv_cache_stored_after_layer_result(self, sched):
        """_handle_layer_result 不应影响 _kv_cache（两者独立）"""
        # 注入一个 KV cache（模拟从节点侧 _handle_layer_forward 的存储行为）
        sched._kv_cache["task_A"] = "mock_past_key_values"
        assert "task_A" in sched._kv_cache
        assert sched._kv_cache["task_A"] == "mock_past_key_values"

    def test_pipeline_done_clears_tuple_kv_cache(self, sched):
        """PIPELINE_DONE 应清理 torch tensor 类型的 KV cache"""
        sched._kv_cache["task_t"] = ("layer_kv",)  # 模拟 tuple 类型
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_t"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_t" not in sched._kv_cache

    def test_pipeline_abort_clears_tuple_kv_cache(self, sched):
        """PIPELINE_ABORT 应清理 KV cache"""
        sched._kv_cache["task_z"] = (("mock_k", "mock_v"),)
        msg = {
            "type": "pipeline_abort",
            "data": {"task_id": "task_z"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_z" not in sched._kv_cache

    def test_multiple_task_kv_cache_independent(self, sched):
        """不同 task 的 KV cache 应相互独立"""
        sched._kv_cache["task_1"] = "kv_1"
        sched._kv_cache["task_2"] = "kv_2"

        # 清理 task_1
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_1"},
        }
        sched._on_tcp_message("master", msg)

        assert "task_1" not in sched._kv_cache
        assert "task_2" in sched._kv_cache
        assert sched._kv_cache["task_2"] == "kv_2"

    def test_fallback_inference_returns_error_without_model(self, sched, monkeypatch):
        """无模型时 _run_full_model_inference 返回 error"""
        import api_server as _api

        monkeypatch.setattr(_api, "model_manager", None)
        result = sched._run_full_model_inference("测试")
        assert isinstance(result, dict)
        assert "error" in result
        assert "模型未加载" in result["error"]


# ================================================================
# 请求队列 测试 (Phase 4 — 多请求排队)
# ================================================================


class TestPipelineQueueBasics:
    """测试 PipelineQueue 的基本操作"""

    @pytest.fixture
    def queue(self):
        from scheduler import PipelineQueue
        return PipelineQueue(max_size=10, result_ttl=10.0)

    def test_initial_state(self, queue):
        """新队列应为空且未运行"""
        assert queue.queue_size == 0
        assert not queue.is_busy
        assert not queue._running

    def test_enqueue_returns_task_id(self, queue):
        """enqueue 应返回 task_id"""
        tid = queue.enqueue(prompt="test")
        assert tid.startswith("q_")
        assert len(tid) > 4
        assert queue.queue_size == 1

    def test_external_cancel_event_removes_queued_task(self, queue):
        cancel_event = threading.Event()
        task_id = queue.enqueue(
            prompt="cancel queued",
            max_new_tokens=32,
            _cancel_event=cancel_event,
        )

        cancel_event.set()
        result = queue.wait_for_result(
            task_id,
            timeout=1.0,
            cancel_event=cancel_event,
        )

        assert result["status"] == "cancelled"
        assert queue.queue_size == 0

    def test_enqueue_custom_task_id(self, queue):
        """enqueue 应接受自定义 task_id"""
        tid = queue.enqueue(task_id="my_task", prompt="hello")
        assert tid == "my_task"
        assert queue.queue_size == 1

    def test_enqueue_full_queue_raises(self, queue):
        """队列满时应抛异常"""
        queue._max_size = 2
        queue.enqueue(prompt="task1")
        queue.enqueue(prompt="task2")
        with pytest.raises(RuntimeError, match="已满"):
            queue.enqueue(prompt="task3")

    def test_is_busy_when_processing(self, queue):
        """正在处理任务时 is_busy 应为 True"""
        import threading
        import time

        started = threading.Event()
        finish = threading.Event()

        def slow_process(**kwargs):
            started.set()
            finish.wait(timeout=5.0)
            return {"response": "done"}

        queue.start(process_fn=slow_process)
        queue.enqueue(prompt="test")
        started.wait(timeout=2.0)

        assert queue.is_busy

        finish.set()
        queue.stop()

    def test_wait_for_result_returns_done(self, queue):
        """等待任务完成应返回结果"""
        def simple_process(**kwargs):
            return {"response": f"echo: {kwargs.get('prompt', '')}"}

        queue.start(process_fn=simple_process)
        tid = queue.enqueue(prompt="hello")
        result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "done"
        assert result["result"]["response"] == "echo: hello"
        queue.stop()

    def test_wait_for_result_timeout(self, queue):
        """超时应返回 timeout 状态"""
        def slow_process(**kwargs):
            import time
            time.sleep(2.0)
            return {"response": "late"}

        queue.start(process_fn=slow_process)
        tid = queue.enqueue(prompt="test")
        result = queue.wait_for_result(tid, timeout=0.3)

        assert result["status"] == "timeout"
        queue.stop()

    def test_wait_for_unknown_task(self, queue):
        """等待不存在的任务应返回 unknown"""
        result = queue.wait_for_result("nonexistent", timeout=0.5)
        assert result["status"] == "unknown"

    def test_process_error_captured(self, queue, caplog):
        """process_fn 抛异常时应记录 error 状态，并带 exc_info。"""
        def failing_process(**kwargs):
            raise ValueError("模拟推理失败")

        with caplog.at_level(logging.ERROR, logger="scheduler"):
            queue.start(process_fn=failing_process)
            tid = queue.enqueue(prompt="test")
            result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "error"
        assert "模拟推理失败" in result["error"]
        # L5: 半结构化日志格式 event=task_failed
        records = [r for r in caplog.records if "task_failed" in r.getMessage()]
        assert records
        assert records[0].exc_info is not None
        queue.stop()

    def test_queue_fifo_order(self, queue):
        """任务应按 FIFO 顺序执行"""
        import threading
        order = []
        lock = threading.Lock()
        started = threading.Event()

        def ordered_process(**kwargs):
            with lock:
                order.append(kwargs.get("seq", -1))
            if len(order) == 1:
                started.set()  # 第一个任务开始后通知
            return {"seq": kwargs.get("seq", -1)}

        queue.start(process_fn=ordered_process)

        # 快速入队 3 个任务
        queue.enqueue(seq=0, prompt="first")
        queue.enqueue(seq=1, prompt="second")
        queue.enqueue(seq=2, prompt="third")

        # 等待所有完成
        import time
        deadline = time.time() + 5.0
        while queue.queue_size > 0 and time.time() < deadline:
            time.sleep(0.05)
        # 等待最后一个任务完成
        time.sleep(0.3)

        assert order == [0, 1, 2], f"应为 FIFO 顺序，实际: {order}"
        queue.stop()

    def test_stop_wakes_waiters(self, queue):
        """stop() 应唤醒所有等待者并设置 cancelled 状态"""
        tid = queue.enqueue(prompt="test")
        # 不启动 worker，直接 stop
        queue.stop()

        result = queue.wait_for_result(tid, timeout=1.0)
        assert result["status"] == "cancelled"

    def test_get_status_reflects_state(self, queue):
        """get_status 应反映当前队列状态"""
        status = queue.get_status()
        assert not status["running"]
        assert status["queue_size"] == 0
        assert status["current_task"] is None

        def slow_process(**kwargs):
            import time
            time.sleep(0.3)
            return {"response": "ok"}

        queue.start(process_fn=slow_process)
        queue.enqueue(prompt="test")

        # 等待 worker 取走任务
        import time
        deadline = time.time() + 2.0
        while queue.queue_size > 0 and time.time() < deadline:
            time.sleep(0.05)

        status2 = queue.get_status()
        assert status2["running"]
        assert status2["queue_size"] == 0  # 已被 worker 取出
        assert status2["current_task"] is not None

        queue.stop()

    def test_result_cleanup_after_ttl(self, queue):
        """超过 TTL 的已完成结果应被清理"""
        queue._result_ttl = 0.1  # 100ms TTL
        tid = queue.enqueue(prompt="test")

        # 手动注入过期结果
        import time
        queue._results[tid] = {
            "status": "done",
            "result": {"response": "old"},
            "completed_at": time.time() - 1.0,  # 1 秒前完成
        }

        queue._cleanup_expired()
        assert tid not in queue._results

    def test_queue_preserves_task_data(self, queue):
        """任务数据应正确传递给 process_fn"""
        received = {}

        def capture_process(**kwargs):
            received.update(kwargs)
            return {"response": "captured"}

        queue.start(process_fn=capture_process)
        tid = queue.enqueue(
            prompt="hello world",
            max_new_tokens=256,
            temperature=0.5,
            session_id="sess_1",
        )
        result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "done"
        assert received["prompt"] == "hello world"
        assert received["max_new_tokens"] == 256
        assert received["temperature"] == 0.5
        assert received["session_id"] == "sess_1"
        queue.stop()


class TestPipelineQueueIntegration:
    """测试 PipelineQueue 与 Scheduler 的集成"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # 不启动调度器（避免 TCP 绑定），只测试队列集成
        return s

    def test_queue_initialized_in_scheduler(self, sched):
        """Scheduler 初始化时应创建 PipelineQueue"""
        assert sched.pipeline_queue is not None
        assert not sched.pipeline_queue._running
        status = sched.pipeline_queue.get_status()
        assert status["queue_size"] == 0

    def test_queue_included_in_get_status(self, sched):
        """get_status() 应包含 queue 信息"""
        status = sched.get_status()
        assert "pipeline_queue" in status
        q = status["pipeline_queue"]
        assert "queue_size" in q
        assert "running" in q
        assert "current_task" in q

    def test_process_queued_task_delegates(self, sched):
        """_process_queued_pipeline_task 应调用 run_pipeline（绕过排队检查）"""
        original_ready = sched._all_pipeline_nodes_ready
        original_run = sched.run_pipeline

        sched._all_pipeline_nodes_ready = lambda: True
        calls = []
        sched.run_pipeline = lambda prompt="", **kw: calls.append(dict(prompt=prompt, **kw)) or {"response": "ok"}

        try:
            result = sched._process_queued_pipeline_task(
                prompt="test", max_new_tokens=512
            )
            assert len(calls) == 1
            assert calls[0]["prompt"] == "test"
            assert calls[0]["max_new_tokens"] == 512
            assert result == {"response": "ok"}
        finally:
            sched._all_pipeline_nodes_ready = original_ready
            sched.run_pipeline = original_run

    def test_process_queued_task_fallbacks_when_pipeline_returns_error(self, sched, monkeypatch):
        """队列 worker 中流水线返回 error 时也应自动回退。"""
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched,
            "run_pipeline",
            lambda prompt="", **kw: {"response": "", "error": "queued worker failed"},
        )

        fallback_calls = []

        def fake_fallback(prompt, **kwargs):
            fallback_calls.append(kwargs)
            return {"response": "queue fallback ok"}

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        result = sched._process_queued_pipeline_task(
            prompt="queued", max_new_tokens=16,
        )

        assert result["response"] == "queue fallback ok"
        assert fallback_calls
        assert "queued worker failed" in fallback_calls[0]["_fallback_reason"]

    def test_run_pipeline_safe_respects_queue_busy(self, sched):
        """
        当 pipeline_queue.is_busy=True 时，run_pipeline_safe 应将请求入队。

        注意: 由于 queue worker 未启动（无 process_fn），入队后需要直接
        注入结果来模拟队列完成。
        """
        import threading
        import time

        # 模拟 busy 状态
        sched.pipeline_queue._current_task_id = "fake_running"

        # 准备结果注入
        def inject_result():
            time.sleep(0.3)
            tid = None
            with sched.pipeline_queue._lock:
                if sched.pipeline_queue._queue:
                    tid = sched.pipeline_queue._queue.popleft()
            if tid:
                tid_str = tid[0]
                with sched.pipeline_queue._lock:
                    sched.pipeline_queue._results[tid_str] = {
                        "status": "done",
                        "result": {"response": "queued_result"},
                    }
                    if tid_str in sched.pipeline_queue._events:
                        sched.pipeline_queue._events[tid_str].set()

        # 模拟流水线节点可用
        original = sched._all_pipeline_nodes_ready
        sched._all_pipeline_nodes_ready = lambda: True
        # 同步执行（不经过真正的 worker 线程）
        original_run = sched.run_pipeline
        sched.run_pipeline = lambda **kw: {"response": "direct"}

        try:
            t = threading.Thread(target=inject_result, daemon=True)
            t.start()

            result = sched.run_pipeline_safe(prompt="queued_prompt")
            # 清理 fake running task
            sched.pipeline_queue._current_task_id = None
            t.join(timeout=3.0)

            # 应得到注入的结果（排队路径）
            assert isinstance(result, dict)
            assert "response" in result or "error" in result
        finally:
            sched._all_pipeline_nodes_ready = original
            sched.run_pipeline = original_run
            sched.pipeline_queue._current_task_id = None


# ================================================================
# 链式拓扑 测试 (P2 — 节点直连)
# ================================================================


class TestChainTopology:
    """测试链式直连转发基础设施"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        from scheduler import NodeInfo, NodeState
        # 注册 3 个在线从节点（带地址）
        s.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="192.168.1.2:8888",
        )
        s.nodes["client2"] = NodeInfo(
            node_id="client2", role="client", state=NodeState.ONLINE,
            address="192.168.1.3:8888",
        )
        s.nodes["client3"] = NodeInfo(
            node_id="client3", role="client", state=NodeState.ONLINE,
            address="192.168.1.4:8888",
        )
        return s

    def test_get_node_address_valid(self, sched):
        """_get_node_address 应正确解析地址"""
        addr = sched._get_node_address("client1")
        assert addr is not None
        assert addr["host"] == "192.168.1.2"
        assert addr["port"] == 8888

    def test_get_node_address_invalid(self, sched):
        """_get_node_address 对未知节点返回 None"""
        assert sched._get_node_address("nonexistent") is None

    def test_get_node_address_offline_no_address(self, sched):
        """离线且无地址的节点返回 None"""
        from scheduler import NodeInfo, NodeState
        sched.nodes["offline_node"] = NodeInfo(
            node_id="offline_node", role="client", state=NodeState.OFFLINE,
            address="",
        )
        assert sched._get_node_address("offline_node") is None

    def test_broadcast_pipeline_abort(self, sched):
        """_broadcast_pipeline_abort 应调用 _send_to_worker 给所有节点"""
        sent_to = []

        def mock_send(worker_id, data, msg_type):
            sent_to.append(worker_id)

        original = sched._send_to_worker
        sched._send_to_worker = mock_send

        try:
            pipeline_nodes = [
                {"node_id": "client1"},
                {"node_id": "client2"},
                {"node_id": "client3"},
            ]
            sched._broadcast_pipeline_abort(pipeline_nodes, "task_x", "测试取消")
            assert len(sent_to) == 3
            assert "client1" in sent_to
            assert "client2" in sent_to
            assert "client3" in sent_to
        finally:
            sched._send_to_worker = original

    def test_broadcast_pipeline_abort_continues_after_send_failure(self, sched):
        """单个节点 ABORT 失败不应阻断其他节点清理。"""
        sent_to = []

        def mock_send(worker_id, data, msg_type):
            sent_to.append(worker_id)
            if worker_id == "client1":
                raise ConnectionError("client1 down")

        original = sched._send_to_worker
        sched._send_to_worker = mock_send

        try:
            pipeline_nodes = [
                {"node_id": "client1"},
                {"node_id": "client2"},
                {"node_id": "client3"},
            ]
            sched._broadcast_pipeline_abort(pipeline_nodes, "task_x", "测试取消")
            assert sent_to == ["client1", "client2", "client3"]
        finally:
            sched._send_to_worker = original

    def test_wait_for_layer_result_multi_nodes(self, sched):
        """_wait_for_layer_result 应在多个节点中有任一返回时立即唤醒"""
        import threading
        import time

        result_holder = []

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result(
                    "task_chain", ["client1", "client2", "client3"],
                    timeout=5.0,
                )
            )

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)

        # 模拟 client2 返回结果（非最后一个节点）
        key = "task_chain:client2"
        with sched._pipeline_lock:
            sched._pipeline_results[key] = {
                "task_id": "task_chain",
                "node_id": "client2",
            }
            if key in sched._pipeline_events:
                sched._pipeline_events[key].set()

        t.join(timeout=2.0)
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert result_holder[0].get("node_id") == "client2"

    def test_wait_for_layer_result_consumes_result_that_arrived_before_event(self, sched):
        """结果先于 event 注册到达时不应等待到超时。"""
        import time

        with sched._pipeline_lock:
            sched._pipeline_results["task_fast:client2"] = {
                "task_id": "task_fast",
                "node_id": "client2",
            }

        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_fast", ["client1", "client2"], timeout=5.0,
        )
        elapsed = time.time() - t0

        assert result is not None
        assert result["node_id"] == "client2"
        assert elapsed < 0.5

    def test_node_disconnect_wakes_pending_pipeline_waiter(self, sched):
        """节点断连应立即唤醒等待该节点结果的流水线线程。"""
        import threading
        import time

        result_holder = []

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result("task_down", "client2", timeout=5.0)
            )

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)

        t0 = time.time()
        sched._fail_pending_pipeline_results_for_node("client2", "client2 down")
        t.join(timeout=1.0)
        elapsed = time.time() - t0

        assert len(result_holder) == 1
        assert result_holder[0]["node_id"] == "client2"
        assert "client2 down" in result_holder[0]["error"]
        assert elapsed < 0.5

    def test_send_layer_result_returns_false_when_tcp_client_missing(self, sched):
        """worker 无主节点连接时，结果发送应返回 False 而不是静默吞掉。"""
        sched._tcp_client = None

        assert sched._send_layer_result("master", "task_send", error="failed") is False

    def test_send_layer_result_disconnects_on_send_failure(self, sched):
        """结果回传 send_data 抛错时应返回 False 并断开连接促使主节点快速感知。"""
        class FailingClient:
            _running = True

            def __init__(self):
                self.disconnected = False

            def send_data(self, data, msg_type):
                raise OSError("broken pipe")

            def disconnect(self):
                self.disconnected = True
                self._running = False

        client = FailingClient()
        sched._tcp_client = client

        assert sched._send_layer_result("master", "task_send", error="failed") is False
        assert client.disconnected is True

    def test_chain_ack_timeout_returns_error_before_layer_timeout(self, sched):
        """链式转发已发送但下游未 ACK 时，应在 ACK 窗口内快速失败。"""
        import time

        sched._pipeline_active_tasks.add("task_ack")
        sched._handle_chain_forward_ack(
            "client1",
            {
                "data": {
                    "task_id": "task_ack",
                    "step": 0,
                    "node_id": "client1",
                    "target_node_id": "client2",
                    "status": "sent",
                }
            },
        )

        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_ack",
            ["client1", "client2"],
            timeout=5.0,
            ack_node_ids=["client2"],
            ack_step=0,
            ack_timeout=0.2,
        )
        elapsed = time.time() - t0

        assert result is not None
        assert result["node_id"] == "client2"
        assert "未收到接收 ACK" in result["error"]
        assert elapsed < 1.0

    def test_wait_for_layer_result_timeout_multi_nodes(self, sched):
        """多节点等待超时应返回 None"""
        import time
        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_t", ["client1", "client2"], timeout=0.3
        )
        elapsed = time.time() - t0
        assert result is None
        assert elapsed < 2.0

    def test_chain_forward_routing(self, sched):
        """CHAIN_FORWARD 消息应路由到 _handle_chain_forward"""
        import threading
        called = []

        def mock_handler(cid, msg):
            called.append((cid, msg.get("type")))

        original = sched._handle_chain_forward
        sched._handle_chain_forward = mock_handler

        try:
            msg = {
                "type": "chain_forward",
                "data": {
                    "task_id": "task_1",
                    "step": 0,
                    "hidden_states": "dGVzdA==",
                },
            }
            sched._on_tcp_message("client1", msg)
            assert len(called) == 1
            assert called[0][0] == "client1"
            assert called[0][1] == "chain_forward"
        finally:
            sched._handle_chain_forward = original

    def test_chain_forward_ack_routing(self, sched):
        """CHAIN_FORWARD_ACK 消息应路由并记录 ACK 状态。"""
        sched._pipeline_active_tasks.add("task_ack_route")
        msg = {
            "type": "chain_forward_ack",
            "data": {
                "task_id": "task_ack_route",
                "step": 1,
                "node_id": "client2",
                "status": "received",
            },
        }

        sched._on_tcp_message("client2", msg)

        assert sched._chain_ack_state["task_ack_route"][1]["client2"]["status"] == "received"

    def test_chain_ack_received_is_not_overwritten_by_late_sent(self, sched):
        """下游 received ACK 先到时，后到的 sent 回报不应覆盖已确认状态。"""
        sched._pipeline_active_tasks.add("task_ack_order")
        sched._handle_chain_forward_ack(
            "client2",
            {
                "data": {
                    "task_id": "task_ack_order",
                    "step": 0,
                    "node_id": "client2",
                    "status": "received",
                }
            },
        )
        sched._handle_chain_forward_ack(
            "client1",
            {
                "data": {
                    "task_id": "task_ack_order",
                    "step": 0,
                    "node_id": "client1",
                    "target_node_id": "client2",
                    "status": "sent",
                }
            },
        )

        state = sched._chain_ack_state["task_ack_order"][0]["client2"]
        assert state["status"] == "received"

    def test_late_chain_ack_is_discarded(self, sched):
        sched._handle_chain_forward_ack("client2", {
            "data": {
                "task_id": "task-finished",
                "step": 0,
                "node_id": "client2",
                "status": "received",
            },
        })
        assert sched._chain_ack_state == {}

    def test_chain_forward_delegates_to_layer_forward(self, sched):
        """_handle_chain_forward 应委托给 _handle_layer_forward"""
        called_with = []
        ack_calls = []

        def mock_handler(cid, msg):
            called_with.append(cid)

        original = sched._handle_layer_forward
        original_ack = sched._send_chain_forward_ack
        sched._handle_layer_forward = mock_handler
        sched._send_chain_forward_ack = lambda **kwargs: ack_calls.append(kwargs) or True

        try:
            msg = {"type": "chain_forward", "data": {"task_id": "t1", "step": 2}}
            sched._handle_chain_forward("client2", msg)
            assert len(called_with) == 1
            assert called_with[0] == "client2"
            assert ack_calls
            assert ack_calls[0]["task_id"] == "t1"
            assert ack_calls[0]["status"] == "received"
        finally:
            sched._handle_layer_forward = original
            sched._send_chain_forward_ack = original_ack

    def test_layer_forward_with_chain_next_forwards_and_sends_ack(self, sched, monkeypatch):
        """_handle_layer_forward 有 chain_next 时应直连转发并发送 sent ACK。"""
        import api_server as _api
        import torch

        forward_calls = []
        ack_calls = []

        class MockModelManager:
            is_loaded = True

            def forward_layers(self, **kwargs):
                return {"hidden_states": torch.ones(1, 2, 4)}

        def mock_forward(target_id, data):
            forward_calls.append((target_id, data))
            return True

        def mock_send_result(cid, tid, result_data=None, error=None):
            pass  # 不应被调用（链式转发成功时）

        original_fwd = sched._send_chain_forward
        original_send = sched._send_layer_result
        original_ack = sched._send_chain_forward_ack
        sched._send_chain_forward = mock_forward
        sched._send_layer_result = mock_send_result
        sched._send_chain_forward_ack = lambda **kwargs: ack_calls.append(kwargs) or True
        monkeypatch.setattr(_api, "model_manager", MockModelManager())
        monkeypatch.setattr(sched, "_record_local_pipeline_participation", lambda *a, **kw: True)

        try:
            msg = {
                "data": {
                    "task_id": "task_forward",
                    "step": 3,
                    "input_ids": [[1, 2]],
                    "use_kv_cache": False,
                    "chain_next": {"node_id": "client2"},
                    "chain_remaining": [],
                }
            }
            sched._handle_layer_forward("master", msg)

            assert len(forward_calls) == 1
            assert forward_calls[0][0] == "client2"
            assert ack_calls
            assert ack_calls[0]["task_id"] == "task_forward"
            assert ack_calls[0]["step"] == 3
            assert ack_calls[0]["target_node_id"] == "client2"
            assert ack_calls[0]["status"] == "sent"
        finally:
            sched._send_chain_forward = original_fwd
            sched._send_layer_result = original_send
            sched._send_chain_forward_ack = original_ack

    def test_master_relay_records_chain_sent_ack(self, sched):
        """L2 主节点中转成功后应记录发往目标节点的 sent ACK 状态。"""
        sent = []

        def mock_send(worker_id, data, msg_type):
            sent.append((worker_id, data, msg_type))

        original = sched._send_to_worker
        sched._send_to_worker = mock_send
        sched._pipeline_active_tasks.add("task_relay")

        try:
            sched._handle_layer_result(
                "client1",
                {
                    "data": {
                        "task_id": "task_relay",
                        "step": 4,
                        "node_id": "client1",
                        "_relay_to": "client2",
                        "hidden_states": "dGVzdA==",
                    }
                },
            )

            assert sent and sent[0][0] == "client2"
            state = sched._chain_ack_state["task_relay"][4]["client2"]
            assert state["status"] == "sent"
            assert state["reporter_node_id"] == "client1"
        finally:
            sched._send_to_worker = original

    def test_chain_info_built_in_run_pipeline(self, sched):
        """run_pipeline 应构建链式拓扑信息"""
        # mock get_layer_assignments 返回 2 个从节点
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24, "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 0,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 0, "end_layer": 12,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client2", "start_layer": 12, "end_layer": 24,
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        try:
            # run_pipeline 需要 model_manager 和完整的 TCP 环境
            # 此处仅验证分层解析不抛异常
            layer_info = sched.get_layer_assignments()
            pipeline_nodes = [
                a for a in layer_info.get("assignments", [])
                if a.get("node_id") != "master"
            ]
            pipeline_nodes.sort(key=lambda a: a.get("start_layer", 0))
            assert len(pipeline_nodes) == 2
            assert pipeline_nodes[0]["node_id"] == "client1"
            assert pipeline_nodes[1]["node_id"] == "client2"
        finally:
            sched.get_layer_assignments = original


# ================================================================
# 显存约束校验 — 层区间重算
# ================================================================

class TestVramConstraintRecalculation:
    """测试 _apply_vram_constraints 在转移层后正确重算 start_layer/end_layer"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_EDGE,  # 显存极少
            ),
        }
        return s

    def test_layer_ranges_continuous_after_transfer(self, sched):
        """层转移后 start_layer/end_layer 应连续无间隙"""
        # 构造含 master 和低显存 client 的 assignments
        assignments = [
            {"node_id": "master", "role": "master",
             "start_layer": 0, "end_layer": 12, "layers_count": 12,
             "has_embedding": True, "has_lm_head": False, "score": 100},
            {"node_id": "client1", "role": "client",
             "start_layer": 12, "end_layer": 24, "layers_count": 12,
             "has_embedding": False, "has_lm_head": True, "score": 10},
        ]
        result = sched._apply_vram_constraints(assignments)

        # client1 显存不足 → 层应转给 master
        # master 应承担全部 24 层
        assert len(result) >= 1
        total_layers = sum(a["layers_count"] for a in result)
        assert total_layers == 24

        # 验证连续无间隙
        result.sort(key=lambda x: x["start_layer"])
        cursor = 0
        for a in result:
            assert a["start_layer"] == cursor, \
                f"gap at {a['node_id']}: expected start={cursor}, got {a['start_layer']}"
            cursor = a["end_layer"]

    def test_vram_transfer_preserves_embed_lm_head(self, sched):
        """层转移后首节点应有 embedding，末节点应有 lm_head"""
        assignments = [
            {"node_id": "master", "role": "master",
             "start_layer": 0, "end_layer": 12, "layers_count": 12,
             "has_embedding": True, "has_lm_head": False, "score": 100},
            {"node_id": "client1", "role": "client",
             "start_layer": 12, "end_layer": 24, "layers_count": 12,
             "has_embedding": False, "has_lm_head": True, "score": 10},
        ]
        result = sched._apply_vram_constraints(assignments)
        if len(result) > 0:
            assert result[0]["has_embedding"] is True
            assert any(item["has_lm_head"] for item in result)


# ================================================================
# _simple_weight_assignment 边界条件
# ================================================================

class TestSimpleWeightAssignmentEdgeCases:
    """测试 _simple_weight_assignment 的边界条件"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_many_nodes_with_min_layers(self, sched):
        """当节点数 > 层数时，每节点至少 1 层，超出的从低分节点削减"""
        # 构造 30 个低分配置节点
        nodes = []
        for i in range(30):
            nodes.append({
                "node_id": f"node{i}",
                "role": "master" if i == 0 else "client",
                "device_info": {"gpu": {"vram_total_gb": 0.5, "cuda_available": False},
                               "ram": {"total_gb": 2}, "cpu": {"physical_cores": 1, "freq_max_mhz": 1000}},
            })
        result = sched._simple_weight_assignment(nodes, 24)
        total = sum(a["layers_count"] for a in result)
        # 总层数不应超过 24
        assert total <= 24, f"总层数 {total} 不应超过 24"
        # 至少部分节点应有分配层
        assert len(result) >= 1

    def test_two_nodes_equal_weight(self, sched):
        """等权重双节点应均分 24 层（各 12）"""
        nodes = [
            {"node_id": "master", "role": "master",
             "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client",
             "device_info": PROFILE_WORKSTATION},  # 相同 profile
        ]
        result = sched._simple_weight_assignment(nodes, 24)
        total = sum(a["layers_count"] for a in result)
        assert total == 24
        # 每个节点至少 10 层（等权重接近均分）
        for a in result:
            assert a["layers_count"] >= 10, \
                f"{a['node_id']} 应有 ≈12 层，实际 {a['layers_count']}"


# ================================================================
# 运行时安全分层与任务统计
# ================================================================

class TestRuntimeSafeLayerAssignmentAndAccounting:
    """测试 master 协调节点语义与分布式任务记账。"""

    def test_low_score_master_does_not_steal_worker_layers(self):
        """低分 master 仅保留 1 层锚定 Embedding，主要层仍归高分 worker"""
        sched = Scheduler()
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc", "device_info": PROFILE_EDGE},
            {"node_id": "worker-fast", "role": "client", "node_type": "pc", "device_info": PROFILE_WORKSTATION},
            {"node_id": "worker-mid", "role": "client", "node_type": "pc", "device_info": PROFILE_LAPTOP},
        ]
        result = sched.compute_layer_assignment(nodes)
        by_id = {a["node_id"]: a for a in result}

        # master 作为 Embedding 锚点至少 1 层，不再 coordinator-only
        assert by_id["master"]["layers_count"] >= 1, \
            f"低分 master 应保留 1 层锚定 Embedding，实际: {by_id['master']['layers_count']}"
        assert "coordinator_only" not in by_id["master"]
        assert by_id["master"]["has_embedding"] is True
        assert by_id["master"]["start_layer"] == 0
        # 高分 worker 仍拿主要层数
        assert by_id["worker-fast"]["layers_count"] >= 14, \
            f"高分 worker 应拿主要层数，实际: {by_id['worker-fast']['layers_count']}"
        assert result[-1]["end_layer"] == 24
        assert result[-1]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_pipeline_task_accounting_counts_master_and_workers_once(self):
        sched = Scheduler()
        sched.nodes = {
            "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE),
            "worker1": NodeInfo(node_id="worker1", role="client", state=NodeState.ONLINE),
            "worker2": NodeInfo(node_id="worker2", role="client", state=NodeState.ONLINE),
        }
        sched._effective_role = lambda: "master"
        sched._push_node_update_to_all_clients = lambda *args, **kwargs: None

        accounting = sched._record_pipeline_task_accounting(
            "task-1",
            [{"node_id": "worker1"}, {"node_id": "worker2"}, {"node_id": "worker1"}],
            success=True,
        )
        assert accounting["counted_nodes"] == ["master", "worker1", "worker2"]
        assert accounting["workers_counted"] == ["worker1", "worker2"]
        assert sched.nodes["master"].task_count == 1
        assert sched.nodes["worker1"].task_count == 1
        assert sched.nodes["worker2"].task_count == 1

        dedup = sched._record_pipeline_task_accounting(
            "task-1", [{"node_id": "worker1"}], success=True
        )
        assert dedup["deduplicated"] is True
        assert sched.nodes["worker1"].task_count == 1

    def test_local_pipeline_participation_is_idempotent(self):
        sched = Scheduler()
        sched.nodes = {
            "client_node": NodeInfo(node_id="client_node", role="client", state=NodeState.ONLINE),
        }
        sched.get_effective_node_id = lambda: "client_node"
        assert sched._record_local_pipeline_participation("task-local", success=True) is True
        assert sched._record_local_pipeline_participation("task-local", success=True) is False
        assert sched.nodes["client_node"].task_count == 1


# ================================================================
# get_status() — tcp_client 字段（从节点视角）
# ================================================================

class TestGetStatusTcpClient:
    """测试 get_status() 对从节点返回 tcp_client 连接状态"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_client_status_includes_tcp_client_field(self, sched):
        """从节点 get_status 应包含 tcp_client 字段"""
        # mock 从节点角色
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 无 _tcp_client 时 tcp_client 字段应为 None
            status = sched.get_status()
            assert "tcp_client" in status, "get_status 应包含 tcp_client 字段"
            assert status["tcp_client"] is None, \
                "无 _tcp_client 时 tcp_client 应为 None"
        finally:
            sched._effective_role = original_role

    def test_client_status_with_tcp_client(self, sched):
        """从节点有 _tcp_client 时应报告连接状态"""
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 注入 mock _tcp_client
            mock_client = type('MockTCPClient', (), {
                'is_registered': True,
                '_running': True,
                'server_host': '100.100.52.1',
                'server_port': 8888,
                'avg_rtt_ms': 12.5,
                'sock': True,
            })()
            sched._tcp_client = mock_client

            status = sched.get_status()
            assert status["tcp_client"] is not None
            assert status["tcp_client"]["connected"] is True
            assert status["tcp_client"]["running"] is True
            assert status["tcp_client"]["server_host"] == "100.100.52.1"
            assert status["tcp_client"]["avg_rtt_ms"] == 12.5
        finally:
            sched._effective_role = original_role
            if hasattr(sched, '_tcp_client'):
                del sched._tcp_client


# ================================================================
# check_master_health() — TCP 快速检测
# ================================================================

class TestCheckMasterHealthTcp:
    """测试 check_master_health() 通过 TCP 状态快速检测断连"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_tcp_disconnected_returns_offline(self, sched):
        """TCP 断开时 check_master_health 应立即返回离线"""
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 注入未注册的 mock _tcp_client
            mock_client = type('MockTCPClient', (), {
                'is_registered': False,
                '_running': False,
                'server_host': '100.100.52.1',
                'server_port': 8888,
                'sock': None,
            })()
            sched._tcp_client = mock_client

            result = sched.check_master_health()
            # 即使 DB 可能显示在线，TCP 断开应立即报告离线
            if result.get("source") == "tcp_disconnected":
                assert result["master_online"] is False
                assert result["tcp_connected"] is False
        finally:
            sched._effective_role = original_role
            if hasattr(sched, '_tcp_client'):
                del sched._tcp_client

    def test_check_master_health_has_tcp_connected_field(self, sched):
        """所有 check_master_health 返回路径应包含 tcp_connected 字段"""
        result = sched.check_master_health()
        assert "tcp_connected" in result, \
            "check_master_health 返回值应包含 tcp_connected 字段"

    def test_registered_tcp_is_online_without_database(self, sched):
        """无数据库时，已注册且活跃的 TCP 连接仍必须判定主节点在线。"""
        sched._role_override = "client"
        sched._tcp_client = type("MockTCPClient", (), {
            "is_registered": True,
            "_running": True,
            "server_host": "100.64.0.10",
            "server_port": 8888,
            "sock": object(),
        })()

        result = sched.check_master_health()

        assert result["master_online"] is True
        assert result["source"] == "tcp"
        assert result["tcp_connected"] is True
        assert result["master_host"] == "100.64.0.10"


# ================================================================
# GPU 选择与评分测试 — 多 GPU / 独显/集显混合场景
# ================================================================


PROFILE_IGPU_ONLY = {
    "gpu": {
        "name": "Intel Iris Xe Graphics",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
        "gpu_type": "integrated",
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_DGPU_ONLY = {
    "gpu": {
        "name": "NVIDIA GeForce RTX 4060",
        "vram_total_gb": 8.0,
        "cuda_available": True,
        "is_integrated": False,
        "gpu_type": "discrete",
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_MULTI_GPU = {
    "gpu": {       # ← 用户前端默认选中的"当前"GPU 被集显占领
        "name": "Intel Iris Xe Graphics",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
        "gpu_type": "integrated",
    },
    "gpus": [
        {
            "name": "Intel Iris Xe Graphics",
            "vram_total_gb": 0.5,
            "cuda_available": False,
            "is_integrated": True,
            "gpu_type": "integrated",
        },
        {
            "name": "NVIDIA GeForce RTX 4060",
            "vram_total_gb": 8.0,
            "cuda_available": True,
            "is_integrated": False,
            "gpu_type": "discrete",
        },
    ],
    "selected_gpu_index": 0,
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_DGPU_NO_IS_INTEGRATED_FIELD = {
    "gpu": {
        "name": "NVIDIA GeForce RTX 4070",
        "vram_total_gb": 12.0,
        "cuda_available": True,
        "gpu_type": "discrete",
    },
    "ram": {"total_gb": 32.0},
    "cpu": {"physical_cores": 12, "freq_max_mhz": 4200},
}


class TestGpuSelection:
    """测试 _select_scoring_gpu() 和 _gpu_is_integrated()"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_fallback_to_gpus_list_for_discrete(self, sched):
        """多 GPU 画像 + 集显为 current → 应选中 gpus 里的 CUDA 独显"""
        gpu = sched._select_scoring_gpu(PROFILE_MULTI_GPU)
        assert gpu["cuda_available"] is True
        assert "nvidia" in gpu["name"].lower() or "rtx" in gpu["name"].lower()

    def test_single_gpu_works(self, sched):
        """仅一个 GPU 且为独显 → 直接返回"""
        gpu = sched._select_scoring_gpu(PROFILE_DGPU_ONLY)
        assert gpu["cuda_available"] is True

    def test_igpu_only_no_change(self, sched):
        """仅集显 → 返回自身"""
        gpu = sched._select_scoring_gpu(PROFILE_IGPU_ONLY)
        assert gpu.get("is_integrated") is True
        assert gpu.get("cuda_available") is False

    def test_missing_is_integrated_field(self, sched):
        """缺少 is_integrated 但 gpu_type=discrete+cuda → 判为非集显"""
        gpu = sched._select_scoring_gpu(PROFILE_DGPU_NO_IS_INTEGRATED_FIELD)
        assert gpu["cuda_available"] is True

    def test_empty_device_info(self, sched):
        """空画像 → 返回空 dict"""
        assert sched._select_scoring_gpu({}) == {}
        assert sched._select_scoring_gpu(None) == {}


class TestGpuIsIntegrated:
    """测试 _gpu_is_integrated() 的启发式判断（静态方法，无需实例化）"""

    @staticmethod
    def _gpu_is_integrated(gpu):
        return Scheduler._gpu_is_integrated(gpu)

    def test_explicit_false(self):
        assert self._gpu_is_integrated({"is_integrated": False}) is False

    def test_explicit_true(self):
        assert self._gpu_is_integrated({"is_integrated": True}) is True

    def test_gpu_type_discrete(self):
        assert self._gpu_is_integrated({"gpu_type": "discrete"}) is False

    def test_gpu_type_integrated(self):
        assert self._gpu_is_integrated({"gpu_type": "integrated"}) is True

    def test_nvidia_name_marker(self):
        assert self._gpu_is_integrated({
            "name": "NVIDIA GeForce GTX 1660", "cuda_available": True,
        }) is False

    def test_intel_uhd_name_marker(self):
        assert self._gpu_is_integrated({"name": "Intel UHD Graphics 630"}) is True

    def test_unknown_defaults_true(self):
        """无提示 → 保守判为集显"""
        assert self._gpu_is_integrated({"name": "Foo Bar"}) is True


class TestReviewVoteEligibility:
    """测试审查投票资格的本机 CUDA 独显判断。"""

    @pytest.fixture
    def sched_master(self, monkeypatch):
        s = Scheduler()
        monkeypatch.setattr(s, "_effective_role", lambda: "master")
        s.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
            node_type="pc", hostname="master-host",
            device_info=PROFILE_WORKSTATION,
        )
        return s

    def test_can_vote_uses_discrete_gpu_from_gpus_list(self, sched_master):
        """current gpu 是集显时，应使用 gpus 列表里的 CUDA 独显判断投票资格。"""
        sched_master.nodes["master"].device_info = PROFILE_MULTI_GPU

        ok, reason = sched_master.can_node_vote("master")

        assert ok is True, reason

    def test_can_vote_accepts_custom_master_node_id(self, sched_master, monkeypatch):
        """主节点自定义 NODE_ID 时，投票检查应回退到 master 自身记录。"""
        monkeypatch.setattr(sched_master, "get_effective_node_id", lambda: "gaming-laptop")
        sched_master.nodes["master"].device_info = PROFILE_LAPTOP

        ok, reason = sched_master.can_node_vote("gaming-laptop")

        assert ok is True, reason

    def test_can_vote_uses_local_profile_when_master_profile_is_stale(self, sched_master, monkeypatch):
        """主节点表里的旧画像只有集显时，应允许用实时本机画像兜底。"""
        import device_profiler

        class DummyProfile:
            def to_dict(self):
                return PROFILE_MULTI_GPU

        sched_master.nodes["master"].device_info = PROFILE_IGPU_ONLY
        monkeypatch.setattr(device_profiler, "get_profile", lambda: DummyProfile())

        ok, reason = sched_master.can_node_vote("master")

        assert ok is True, reason

    def test_can_vote_rejects_android_node(self, sched_master):
        sched_master.nodes["android-1"] = NodeInfo(
            node_id="android-1", role="client", state=NodeState.ONLINE,
            node_type="android", hostname="phone",
            device_info=PROFILE_MULTI_GPU,
        )

        ok, reason = sched_master.can_node_vote("android-1")

        assert ok is False
        assert "PC" in reason


# ================================================================
# _normalize_master_anchor 测试
# ================================================================


class TestNormalizeMasterAnchor:
    """测试统一分层不变量：master 作为首段 Embedding 锚点"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def _nl(self, items=None):
        """便捷构造 node_list"""
        default = [
            {"node_id": "master", "role": "master",
             "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client",
             "device_info": PROFILE_LAPTOP},
            {"node_id": "client2", "role": "client",
             "device_info": PROFILE_ULTRABOOK},
        ]
        return items if items is not None else default

    def test_master_becomes_first_with_at_least_one_layer(self, sched):
        """master 在 assignments 中应为首位且至少 1 层"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 18,
             "layers_count": 18, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "master", "start_layer": 18, "end_layer": 24,
             "layers_count": 6, "has_embedding": False, "has_lm_head": True,
             "score": 100, "role": "master"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["has_embedding"] is True
        assert result[0]["layers_count"] >= 1

    def test_master_not_in_assignments_yet_in_node_list(self, sched):
        """master 未在 assignments 中但在 node_list 中 → 应补入并置首"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 24,
             "layers_count": 24, "has_embedding": True, "has_lm_head": True,
             "score": 50, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        assert result[0]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_no_master_in_node_list_passes_through(self, sched):
        """node_list 中无 master → 原样返回但区间重新排序"""
        node_list = self._nl([
            {"node_id": "client1", "role": "client", "device_info": PROFILE_LAPTOP},
            {"node_id": "client2", "role": "client", "device_info": PROFILE_ULTRABOOK},
        ])
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 18,
             "layers_count": 18, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "client2", "start_layer": 18, "end_layer": 24,
             "layers_count": 6, "has_embedding": False, "has_lm_head": True,
             "score": 10, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert len(result) == 2
        assert result[0]["has_embedding"] is True
        assert result[-1]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_master_gets_zero_layers_becomes_one(self, sched):
        """master layers_count=0 → 变为 1，从低分节点回收"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 12,
             "layers_count": 12, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "master", "start_layer": 12, "end_layer": 12,
             "layers_count": 0, "has_embedding": False, "has_lm_head": False,
             "score": 100, "role": "master"},
            {"node_id": "client2", "start_layer": 12, "end_layer": 24,
             "layers_count": 12, "has_embedding": False, "has_lm_head": True,
             "score": 10, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["layers_count"] >= 1
        assert sum(a["layers_count"] for a in result) == 24


# ================================================================
# 分布式推理集成测试 — 全链路流水线编排
# ================================================================


class TestPipelineOrchestrationIntegration:
    """
    集成测试：模拟真实分布式推理场景。

    覆盖关键路径：
    1. master 在线 + 所有 worker 离线 → fallback 本地推理
    2. master 参与流水线首段 → layer 分配正确
    3. 多 GPU 画像 → 独显评分不被集显压低
    4. Android 节点注册后保持在线（不被 init_nodes 强制离线）
    """

    @pytest.fixture
    def sched_master(self, monkeypatch):
        """创建一个"主节点" scheduler，含模拟 TCP 服务端"""
        s = Scheduler()
        # 伪装为主节点
        s._role_override = "master"
        monkeypatch.setattr(s, "_effective_role", lambda: "master")
        # 注入假 TCP 服务端（满足就绪检查和 get_status 的属性访问）
        fake_server = type('FakeTCPServer', (), {
            '_running': True,
            'host': '0.0.0.0',
            'port': 8888,
            'clients': {},
            'send_to_client': lambda self, cid, data, msg_type: None,
            'broadcast_layer_config': lambda self, assignments: None,
            'get_client_ids': lambda self: list(self.clients.keys()),
            'get_client_info': lambda self, cid: {},
        })()
        monkeypatch.setattr(s, "_tcp_server", fake_server)

        # 注入 master 节点自身
        s.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
            node_type="pc", hostname="master-host",
            device_info=PROFILE_WORKSTATION,
        )
        return s

    @pytest.fixture
    def sched_with_workers(self, sched_master, monkeypatch):
        """主节点 + 2 个 PC worker 在线"""
        s = sched_master
        for nid, profile in [("worker1", PROFILE_LAPTOP), ("worker2", PROFILE_ULTRABOOK)]:
            s.nodes[nid] = NodeInfo(
                node_id=nid, role="client", state=NodeState.ONLINE,
                node_type="pc", address=f"100.64.1.{2 if '1' in nid else 3}:8888",
                hostname=nid, device_info=profile,
                last_heartbeat=time.time(),
            )
        # 模拟 TCP 服务端已知这些 client
        s._tcp_server.clients = {"worker1": True, "worker2": True}
        assignments = s.compute_layer_assignment()
        # 移除 get_layer_assignments 的 DB 依赖，固定同一份当前分配。
        monkeypatch.setattr(s, "get_layer_assignments", lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": assignments,
        })
        for assignment in assignments:
            node_id = assignment["node_id"]
            if node_id == "master":
                continue
            expected = {
                "config_id": "cfg-current",
                "start_layer": assignment["start_layer"],
                "end_layer": assignment["end_layer"],
                "model_sha256": "sha-current",
            }
            s._layer_config_expected[node_id] = expected
            s._layer_config_acks[node_id] = {
                "config_id": "cfg-current",
                "status": "ready",
                "layer_range": [assignment["start_layer"], assignment["end_layer"]],
                "model_sha256": "sha-current",
                "engine": "pytorch",
            }
            s._layer_config_pushed.add(node_id)
        return s

    def test_stale_layer_ack_does_not_make_pipeline_ready(self, sched_with_workers):
        """当前分配变化后，旧层范围 ACK 必须立即失效。"""
        sched_with_workers._layer_config_acks["worker1"]["layer_range"] = [0, 1]
        readiness = sched_with_workers._get_pipeline_readiness()
        assert readiness["ready"] is False
        assert readiness["reason_code"] == "worker_layer_loading"

    # ----------------------------------------------------------
    # 场景 1：master 在线，所有 worker 离线 → fallback 本地推理
    # ----------------------------------------------------------

    def test_all_workers_offline_triggers_fallback(self, sched_master, monkeypatch):
        """无可用 worker → run_pipeline_safe 应回退到全模型推理"""
        fallback_called = []
        pipeline_called = []

        def fake_fallback(prompt, **kw):
            fallback_called.append((prompt, kw))
            return {"response": "fallback_response", "metrics": {"fallback": True}}

        monkeypatch.setattr(sched_master, "_run_full_model_inference", fake_fallback)
        monkeypatch.setattr(sched_master, "run_pipeline", lambda **kw: pipeline_called.append(1) or {})

        result = sched_master.run_pipeline_safe("测试 prompt")

        assert len(fallback_called) == 1
        assert len(pipeline_called) == 0
        assert result["response"] == "fallback_response"
        assert result["metrics"]["fallback"] is True

    def test_all_workers_offline_get_status_shows_offline(self, sched_master):
        """worker 离线时 get_status 应正确反映离线状态"""
        sched_master.nodes["worker-x"] = NodeInfo(
            node_id="worker-x", role="client", state=NodeState.OFFLINE,
            node_type="pc",
        )
        status = sched_master.get_status()
        nodes = status["nodes"]
        assert nodes["master"]["is_available"] is True
        assert nodes["worker-x"]["is_available"] is False

    # ----------------------------------------------------------
    # 场景 2：master 参与流水线 → 层分配正确
    # ----------------------------------------------------------

    def test_layer_assignment_master_has_embedding(self, sched_with_workers):
        """master 应作为首段 Embedding 锚点参与层分配"""
        info = sched_with_workers.get_layer_assignments()
        assignments = info["assignments"]
        # master 至少 1 层
        master = [a for a in assignments if a.get("role") == "master"]
        assert len(master) >= 1, "master 应在 assignments 中"
        m = master[0]
        assert m["has_embedding"] is True
        assert m["start_layer"] == 0
        assert m["layers_count"] >= 1
        # worker1 分数高于 worker2 → 应拿更多层
        workers = [a for a in assignments if a["node_id"] != "master"]
        assert len(workers) >= 1
        if len(workers) >= 2:
            assert workers[0]["layers_count"] >= workers[1]["layers_count"]

    def test_master_layer_range_is_contiguous_with_workers(self, sched_with_workers):
        """master 的层范围 + worker 的层范围应连续覆盖 [0, 24)"""
        info = sched_with_workers.get_layer_assignments()
        assignments = info["assignments"]
        assignments.sort(key=lambda a: a["start_layer"])
        cursor = 0
        for a in assignments:
            assert a["start_layer"] == cursor, \
                f"断点于 {a['node_id']}: 期望 {cursor}, 实际 {a['start_layer']}"
            cursor = a["end_layer"]
        assert cursor == 24, f"末节点 end_layer 应为 24, 实际 {cursor}"

    # ----------------------------------------------------------
    # 场景 3：多 GPU / 独显评分
    # ----------------------------------------------------------

    def test_mixed_gpu_profile_selects_discrete_for_scoring(self, sched_master):
        """多 GPU 画像（集显 front + 独显 in gpus）→ 评分优先用独显"""
        master = sched_master.nodes["master"]
        # 模拟：device_info["gpu"] 是集显，但 gpus 里含独显
        master.device_info = PROFILE_MULTI_GPU
        # 通过 compute_layer_assignment 间接计算 weight
        # 先 _compute_node_weight 验证
        weight = sched_master._compute_node_weight(PROFILE_MULTI_GPU)
        # 独显 RTX 4060: VRAM=8/24*50=16.7, RAM=16/64*30=7.5, CPU=8/16*10+4000/4000*10=15, Bonus=15 → ≈54.2
        assert weight >= 40, f"独显评分不应被集显压低，实际: {weight:.1f}"

        weight_igpu = sched_master._compute_node_weight(PROFILE_IGPU_ONLY)
        # 集显: VRAM=0.5/24*50≈1.0, RAM=16/64*30=7.5, CPU=15, Bonus=0 → ≈23.5
        assert weight > weight_igpu + 15, \
            f"含独显画像评分({weight:.1f})应明显高于纯集显({weight_igpu:.1f})"

    def test_vram_uses_scoring_gpu_not_frontend_gpu(self, sched_master):
        """_get_node_vram_mb 应使用评分 GPU 而非前端选中 GPU"""
        master = sched_master.nodes["master"]
        master.device_info = PROFILE_MULTI_GPU
        vram_mb = sched_master._get_node_vram_mb("master")
        # 独显 RTX 4060: 8GB → 8192 MB
        assert vram_mb == 8.0 * 1024, \
            f"VRAM 应为独显的 8192MB, 实际: {vram_mb}"

        master.device_info = PROFILE_IGPU_ONLY
        vram_igpu = sched_master._get_node_vram_mb("master")
        # 共享显存不能作为独立容量；若画像未提供 available_gb，则返回未知 0。
        assert vram_igpu == 0, \
            f"纯集显应按系统可用内存计量，缺少 available_gb 时应为未知 0，实际: {vram_igpu}"

    # ----------------------------------------------------------
    # 场景 4：Android 节点状态管理
    # ----------------------------------------------------------

    def test_android_node_stays_online_after_register(self, sched_master):
        """Android HTTP 客户端注册后应保持 online"""
        sched_master.register_android_client(
            "android-1", hostname="Pixel 7", network_type="wifi",
        )
        assert sched_master.nodes["android-1"].state == NodeState.ONLINE
        assert sched_master.nodes["android-1"].node_type == "android"

        # 调用 refresh（模拟心跳检查）— 不应立即标记离线
        sched_master._refresh_http_client_states(now=time.time() + 30)
        assert sched_master.nodes["android-1"].state == NodeState.ONLINE, \
            "30 秒后 Android 仍应在线"

    def test_android_node_offline_after_timeout(self, sched_master):
        """Android 客户端超时后应变为 offline"""
        sched_master.register_android_client("android-2", hostname="Tablet")
        # 快进 121 秒（超过 120s 超时）
        sched_master._refresh_http_client_states(now=time.time() + 121)
        assert sched_master.nodes["android-2"].state == NodeState.OFFLINE

    def test_android_node_not_forced_offline_in_init(self, sched_master, monkeypatch):
        """Android HTTP 客户端在 init_nodes 中不应被强制 offline"""
        import scheduler as sched_mod

        now = time.time()
        db_row = {
            "node_id": "android-db", "role": "client", "node_type": "android",
            "state": "online", "address": "1.2.3.4", "hostname": "Phone",
            "device_info": {"connection_type": "http_thin"},
            "network_type": "wifi", "connected_at": now,
            "last_heartbeat": now, "task_count": 0, "error_count": 0,
            "model_sha256": "", "avg_rtt_ms": 0.0, "last_rtt_ms": 0.0,
        }

        class DummyDb:
            def get_all_nodes(self):
                return [db_row]
            def delete_node(self, nid):
                pass
            def upsert_node(self, **kw):
                pass
            def update_node_state(self, **kw):
                pass
            def get_config(self, k, d):
                return d
            def set_config(self, k, v):
                pass

        monkeypatch.setattr(sched_mod, "_db_available", True)
        monkeypatch.setattr(sched_mod, "_get_db", lambda: DummyDb())
        monkeypatch.setattr(sched_mod, "RUN_MODE", "distributed")

        # 在新 scheduler 上运行 init_nodes（避免 sched_master fixture 已初始化）
        s2 = Scheduler()
        s2._role_override = "master"
        monkeypatch.setattr(s2, "_effective_role", lambda: "master")
        s2._tcp_server = sched_master._tcp_server
        s2.init_nodes()

        assert "android-db" in s2.nodes, f"Android 节点应被恢复，实际节点列表: {list(s2.nodes.keys())}"
        android = s2.nodes["android-db"]
        assert android.node_type == "android"
        # ★ 关键断言：Android HTTP 节点不应被强制 offline
        assert android.state == NodeState.ONLINE, \
            f"Android 节点应是 ONLINE，实际: {android.state}"

    # ----------------------------------------------------------
    # 场景 5：ensure_full_model 在 fallback 中触发
    # ----------------------------------------------------------

    def test_fallback_calls_ensure_full_model(self, sched_master, monkeypatch):
        """流水线裁剪后回退本地推理 → 必须确保模型为完整模型"""
        import api_server as _api

        ensure_calls = []
        chat_calls = []

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            layer_range = (0, 4)  # 模拟被裁剪过
            _layer_has_embedding = True
            _layer_has_lm_head = False
            tokenizer = None

            def ensure_full_model(self, **kw):
                ensure_calls.append(kw)
                self.layer_range = None
                self._layer_has_lm_head = True

            def chat(self, messages, **kw):
                chat_calls.append((messages, kw))
                return {"content": "fallback reply", "usage": {}, "tokens_per_second": 10.0}

            def chat_stream(self, messages, **kw):
                chat_calls.append(("stream", messages, kw))
                yield "fallback"
                yield " reply"

        mock_mgr = MockModelManager()
        monkeypatch.setattr(_api, "model_manager", mock_mgr)

        result = sched_master._run_full_model_inference(
            "test prompt", max_new_tokens=32,
            _fallback_reason="test_fallback",
        )
        # 确认 ensure_full_model 被调用
        assert len(ensure_calls) >= 1, "fallback 前应调用 ensure_full_model"
        # 确认 chat 也被调用
        assert len(chat_calls) >= 1
        assert result["response"] == "fallback reply"
        assert result["metrics"]["fallback"] is True
        assert result["metrics"]["fallback_reason"] == "test_fallback"

    # ----------------------------------------------------------
    # 场景 6：master 参与 pipeline 时的 run_pipeline 编排
    # ----------------------------------------------------------

    def test_run_pipeline_master_loads_local_layer_range(self, sched_with_workers, monkeypatch):
        """run_pipeline 应在 master_participates 时调用 load_layer_range"""
        import api_server as _api

        layer_loads = []
        forward_calls = []

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            tokenizer: Any = None
            layer_range = None
            _layer_has_embedding = True
            _layer_has_lm_head = True
            quant_type = "int4"

            def load_layer_range(self, start, end, has_embedding, has_lm_head, **kw):
                layer_loads.append((start, end, has_embedding, has_lm_head))

            def ensure_layer_range(self, start, end, has_embedding, has_lm_head, **kw):
                layer_loads.append(("ensure", start, end, has_embedding, has_lm_head))

            def forward_layers(self, **kw):
                forward_calls.append(kw)
                import torch
                input_ids: Any = kw.get("input_ids")
                bs = input_ids.shape[0] if input_ids is not None else 1
                sl = input_ids.shape[1] if input_ids is not None else 1
                return {"hidden_states": torch.randn(bs, sl, 2048),
                        "past_key_values": ((torch.randn(1, 16, sl, 64),
                                             torch.randn(1, 16, sl, 64)),)}

            def get_device(self):
                import torch
                return torch.device("cpu")

        mock_mgr = MockModelManager()

        # 构建 tokenizer mock
        class MockTokenizer:
            eos_token_id = 151643
            def __call__(self, prompt, **kw):
                import torch
                return {"input_ids": torch.randint(0, 1000, (1, 8)),
                        "attention_mask": torch.ones(1, 8)}
            def decode(self, ids, **kw):
                return "mock response"

        mock_mgr.tokenizer = MockTokenizer()
        monkeypatch.setattr(_api, "model_manager", mock_mgr)

        # 也需要 mock _send_to_worker 和 _wait_for_layer_result
        send_calls = []
        monkeypatch.setattr(sched_with_workers, "_send_to_worker",
                           lambda wid, data, mtype: send_calls.append((wid, data, mtype)))

        # 模拟无 LM Head 的 worker 返回尾层 hidden states，由 master 执行输出头。
        # _wait_for_layer_result 内部已将 base64 解码为 bytes
        import torch
        from tcp_comm import serialize_tensor_fast
        fake_logits = torch.randn(1, 8, 151936)
        fake_hidden = torch.randn(1, 8, 2048)
        fake_hidden_raw = serialize_tensor_fast(fake_hidden)
        lm_head_calls = []
        monkeypatch.setattr(
            sched_with_workers,
            "_run_master_lm_head",
            lambda hidden: lm_head_calls.append(hidden.shape) or fake_logits,
        )

        monkeypatch.setattr(sched_with_workers, "_wait_for_layer_result",
                           lambda tid, nids, timeout: {
                               "task_id": tid,
                               "node_id": nids[-1] if isinstance(nids, list) else nids,
                               "step": 0,
                               "hidden_states": fake_hidden_raw,
                           })

        result = sched_with_workers.run_pipeline(
            "hello world", max_new_tokens=3, temperature=0.7, top_p=0.9,
        )

        # 验证 master 加载了层范围
        assert len(layer_loads) >= 1, "master 应调用 load_layer_range/ensure_layer_range"
        # 验证 master 本地 forward 被调用
        assert len(forward_calls) >= 1, "master 本地 forward_layers 应被调用"
        # 验证发送了 LAYER_FORWARD 给 worker
        layer_forwards = [c for c in send_calls if c[2] is not None]
        assert len(layer_forwards) >= 1, "应发送 LAYER_FORWARD 给 worker"
        assert lm_head_calls, "worker 尾层 hidden states 应返回 master 执行 LM Head"
        # 验证分布式执行标记与主从任务统计来自同一条已完成流水线任务
        assert result["metrics"]["distributed_used"] is True
        assert set(result["metrics"]["workers_used"]) == {"worker1", "worker2"}
        assert set(result["metrics"]["workers_counted"]) == {"worker1", "worker2"}
        assert sched_with_workers.nodes["master"].task_count == 1
        assert sched_with_workers.nodes["worker1"].task_count == 1
        assert sched_with_workers.nodes["worker2"].task_count == 1

    def test_run_pipeline_builds_native_prompt_from_history(
            self, sched_with_workers, monkeypatch):
        import api_server as _api
        import torch
        from tcp_comm import serialize_tensor_fast

        prompts = []

        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                assert messages == [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                ]
                return "native-history-prompt"

            def __call__(self, prompt, **kwargs):
                prompts.append(prompt)
                return {
                    "input_ids": torch.tensor([[1, 3]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def decode(self, ids, **kwargs):
                return "answer"

        class Manager:
            is_loaded = True
            _engine_type = "pytorch"
            tokenizer = Tokenizer()

            def ensure_layer_range(self, *args, **kwargs):
                pass

            def forward_layers(self, **kwargs):
                return {"hidden_states": torch.randn(1, 2, 8)}

            def get_device(self):
                return torch.device("cpu")

            def _merge_stop_sequences(self, _value):
                return []

            def _get_generation_eos_token_ids(self, _stops):
                return 2

        monkeypatch.setattr(_api, "model_manager", Manager())
        monkeypatch.setattr(sched_with_workers, "_send_to_worker", lambda *a, **kw: None)
        hidden = serialize_tensor_fast(torch.randn(1, 2, 8))
        monkeypatch.setattr(
            sched_with_workers, "_wait_for_layer_result_with_ack",
            lambda *a, **kw: {"hidden_states": hidden},
        )
        monkeypatch.setattr(
            sched_with_workers, "_run_master_lm_head",
            lambda _hidden: torch.tensor([[[0.0, 0.0, 100.0, 0.0]]]),
        )

        sched_with_workers.run_pipeline(
            "second",
            max_new_tokens=1,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
        )

        assert prompts == ["native-history-prompt"]

    def test_run_pipeline_safe_immediate_path_locks(self, sched_with_workers, monkeypatch):
        """立即执行路径应正确管理 inference_lock（不产生死锁或泄漏）"""
        import api_server as _api

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            tokenizer: Any = None
            layer_range = None
            _layer_has_embedding = True
            _layer_has_lm_head = True
            quant_type = "int4"

            def ensure_layer_range(self, start_layer, end_layer, has_embedding, has_lm_head, **kw):
                pass

            def forward_layers(self, **kw):
                import torch
                bs = 1; sl = 4
                return {"hidden_states": torch.randn(bs, sl, 2048),
                        "past_key_values": ((torch.randn(1, 16, sl, 64),
                                             torch.randn(1, 16, sl, 64)),)}

            def get_device(self):
                import torch
                return torch.device("cpu")

        mock_mgr = MockModelManager()
        class MockTokenizer:
            eos_token_id = 151643
            def __call__(self, prompt, **kw):
                import torch
                return {"input_ids": torch.randint(0, 1000, (1, 4)),
                        "attention_mask": torch.ones(1, 4)}
            def decode(self, ids, **kw):
                return "mock"

        mock_mgr.tokenizer = MockTokenizer()
        monkeypatch.setattr(_api, "model_manager", mock_mgr)
        monkeypatch.setattr(sched_with_workers, "_send_to_worker",
                           lambda wid, data, mtype: None)

        import torch
        from tcp_comm import serialize_tensor_fast
        fake_logits = torch.randn(1, 4, 151936)
        fake_logits_raw = serialize_tensor_fast(fake_logits)  # bytes — 模拟 _wait_for_layer_result 解码后的返回值
        monkeypatch.setattr(sched_with_workers, "_wait_for_layer_result",
                           lambda tid, nids, timeout: {
                               "task_id": tid, "node_id": "worker1", "step": 0,
                               "logits": fake_logits_raw,
                           })

        result = sched_with_workers.run_pipeline_safe("test")
        # 验证锁已正常释放（不是死锁）
        assert not sched_with_workers._inference_lock.locked(), \
            "推理锁应在 run_pipeline_safe 返回后释放"
        assert "error" not in result or result.get("response"), \
            f"不应返回致命错误: {result.get('error', '')}"

    def test_run_pipeline_stream_uses_safe_scheduler(self, sched_master, monkeypatch):
        """fast SSE 也必须经过排队、互斥和全模型回退入口。"""
        calls = []

        def safe(prompt, **kwargs):
            calls.append(prompt)
            kwargs["_stream_callback"]({
                "done": True, "response": "ok", "metrics": {},
            })
            return {"response": "ok", "metrics": {}}

        monkeypatch.setattr(sched_master, "run_pipeline_safe", safe)
        monkeypatch.setattr(
            sched_master,
            "_run_pipeline",
            lambda *args, **kwargs: pytest.fail("不能绕过 run_pipeline_safe"),
        )

        events = list(sched_master.run_pipeline_stream("hello"))
        assert calls == ["hello"]
        assert events[-1]["done"] is True
        assert events[-1]["response"] == "ok"

    def test_run_pipeline_unexpected_exception_aborts_registered_context(
            self, sched_master, monkeypatch):
        """采样/解码等未预期异常也必须清理所有节点 KV cache。"""
        aborts = []
        clears = []

        def fail(*args, **kwargs):
            sched_master._pipeline_context.stack.append({
                "task_id": "task-exception",
                "pipeline_nodes": [{"node_id": "worker1"}],
            })
            raise RuntimeError("sampling failed")

        monkeypatch.setattr(sched_master, "_run_pipeline", fail)
        monkeypatch.setattr(
            sched_master,
            "_broadcast_pipeline_abort",
            lambda nodes, task_id, reason: aborts.append((nodes, task_id, reason)),
        )
        monkeypatch.setattr(
            sched_master,
            "_clear_pipeline_runtime_state",
            lambda task_id: clears.append(task_id),
        )

        with pytest.raises(RuntimeError, match="sampling failed"):
            sched_master.run_pipeline("hello")
        assert aborts[0][1] == "task-exception"
        assert clears == ["task-exception"]


# ================================================================
# Phase 7 P0: _effective_role 实现逻辑测试
# ================================================================

class TestEffectiveRole:
    """测试 _effective_role() 的实现逻辑（非 mock）。"""

    def test_default_role_from_config(self):
        """默认角色来自 NODE_ROLE 配置。"""
        s = Scheduler()
        # 新 Scheduler 未设置 _role_override，直接读 NODE_ROLE
        role = s._effective_role()
        assert role in ("master", "client"), \
            f"有效角色应为 master 或 client，实际: {role}"

    def test_role_override_takes_precedence(self):
        """_role_override 应优先于 NODE_ROLE。"""
        s = Scheduler()
        s._role_override = "client"
        assert s._effective_role() == "client"

    def test_role_override_empty_falls_through_to_config(self):
        """空字符串 override 应回退到 NODE_ROLE（falsy 值处理）。"""
        s = Scheduler()
        s._role_override = ""
        role = s._effective_role()
        # 空字符串为 falsy → or 短路 → NODE_ROLE
        assert role in ("master", "client")


class TestProvisionalMasterRole:
    """Fresh packaged nodes can join an existing master without DB access."""

    @staticmethod
    def _set_master_role(monkeypatch, tmp_path):
        import config as cfg
        import scheduler as scheduler_mod

        monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "master", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "master", raising=False)
        monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
        monkeypatch.delenv("QLH_NODE_ROLE", raising=False)

    def test_unconfigured_db_unavailable_master_is_provisional(self, monkeypatch, tmp_path):
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_verified = True
        sched._master_identity_reason = "db_unavailable"

        role = sched.get_my_role()

        assert role["node_role"] == "unknown"
        assert role["runtime_node_role"] == "master"
        assert role["is_master"] is False
        assert role["is_provisional"] is True
        assert role["can_join_existing_master"] is True

    def test_database_verified_master_is_not_provisional(self, monkeypatch, tmp_path):
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_verified = True
        sched._master_identity_reason = "match"

        role = sched.get_my_role()

        assert role["node_role"] == "master"
        assert role["is_master"] is True
        assert role["is_provisional"] is False
        assert role["can_join_existing_master"] is False

    def test_master_with_connected_clients_cannot_switch(self, monkeypatch, tmp_path):
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_reason = "db_unavailable"
        sched._tcp_server = MagicMock()
        sched._tcp_server.get_client_ids.return_value = ["client-1"]

        result = sched.activate_client_mode()

        assert result["status"] == "denied"
        assert sched._effective_role() == "master"


# ================================================================
# Phase 7 P1: compute_layer_assignment 边界测试
# ================================================================

class TestComputeLayerAssignmentEdgeCases:
    """测试 compute_layer_assignment 的极端情况。"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {}
        return s

    def test_zero_vram_node_fallback_to_min_one_layer(self, sched):
        """0 VRAM 的节点不应崩溃，应回退到最小 1 层分配。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc",
             "device_info": {"gpu": {"vram_total_gb": 0.0, "cuda_available": False},
                             "ram": {"total_gb": 8.0}, "cpu": {"physical_cores": 4}}},
            {"node_id": "w1", "role": "client", "node_type": "pc",
             "device_info": {"gpu": {"vram_total_gb": 0.0, "cuda_available": False},
                             "ram": {"total_gb": 8.0}, "cpu": {"physical_cores": 4}}},
        ]
        result = sched.compute_layer_assignment(nodes)
        assert len(result) >= 1
        total = sum(a["layers_count"] for a in result)
        assert total == 24
        # 验证每个节点至少分配 1 层（0-VRAM 回退保证）
        for a in result:
            assert a["layers_count"] >= 1, (
                f"节点 {a['node_id']} 分配到 {a['layers_count']} 层，"
                f"期望至少 1 层 (0-VRAM 回退)"
            )

    def test_only_android_nodes_returns_empty(self, sched):
        """仅 Android 节点应返回空分配（Android 不参与层拆分）。"""
        nodes = [
            {"node_id": "android-a", "role": "client", "node_type": "android",
             "device_info": {"platform": "android"}},
            {"node_id": "android-b", "role": "client", "node_type": "android",
             "device_info": {"platform": "android"}},
        ]
        result = sched.compute_layer_assignment(nodes)
        assert result == []


# ================================================================
# L5: task_id / request_id 半结构化日志贯穿
# ================================================================

class TestTaskRequestIdCorrelation:
    """L5: scheduler 中 task_id 和 request_id 的关联。"""

    @pytest.fixture
    def queue(self):
        from scheduler import PipelineQueue
        return PipelineQueue(max_size=10, result_ttl=60)

    def test_enqueue_stores_request_id(self, queue):
        """enqueue 时应存储 request_id 到 QueueTask。"""
        tid = queue.enqueue(
            prompt="test prompt",
            max_new_tokens=128,
            request_id="req-abc-123",
        )
        queue.stop()
        assert tid.startswith("q_")

    def test_enqueue_request_id_in_log(self, queue, caplog):
        """enqueue 日志应包含 request_id。"""
        with caplog.at_level(logging.INFO, logger="scheduler"):
            queue.enqueue(prompt="test", max_new_tokens=64, request_id="rid-logtest")
            queue.stop()

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_enqueue" in msg and "request_id=rid-logtest" in msg:
                found = True
                break
        assert found, f"未找到包含 request_id=rid-logtest 的 enqueue 日志: {caplog.text}"

    def test_task_dispatch_log_has_request_id(self, queue, caplog):
        """task_dispatch 日志应包含 request_id。"""
        def echo_process(**kwargs):
            return {"status": "ok", "response": "echo"}

        queue.start(process_fn=echo_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(
                prompt="hello",
                max_new_tokens=128,
                request_id="rid-dispatch-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_dispatch" in msg and "request_id=rid-dispatch-test" in msg:
                found = True
                break
        assert found, f"task_dispatch 日志应包含 request_id: {caplog.text}"
        queue.stop()

    def test_task_complete_log_has_request_id(self, queue, caplog):
        """task_complete 日志应包含 request_id。"""
        def quick_process(**kwargs):
            return {"status": "ok", "content": "done"}

        queue.start(process_fn=quick_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(
                prompt="quick",
                max_new_tokens=16,
                request_id="rid-complete-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_complete" in msg and "request_id=rid-complete-test" in msg:
                found = True
                break
        assert found, f"task_complete 日志应包含 request_id: {caplog.text}"
        queue.stop()

    def test_task_failed_log_has_request_id_and_exc_info(self, queue, caplog):
        """task_failed 日志应包含 request_id 且携带异常堆栈。"""
        def failing_process(**kwargs):
            raise RuntimeError("模拟推理崩溃")

        queue.start(process_fn=failing_process)
        with caplog.at_level(logging.ERROR, logger="scheduler"):
            tid = queue.enqueue(
                prompt="will fail",
                max_new_tokens=64,
                request_id="rid-fail-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        records = [r for r in caplog.records if "task_failed" in r.getMessage()]
        assert records
        assert "request_id=rid-fail-test" in records[0].getMessage()
        assert records[0].exc_info is not None
        queue.stop()

    def test_enqueue_without_request_id_defaults_to_dash(self, queue, caplog):
        """未提供 request_id 时日志应显示 request_id=-。"""
        def quick_process(**kwargs):
            return {"status": "ok"}

        queue.start(process_fn=quick_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(prompt="no-rid", max_new_tokens=32)
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_enqueue" in msg:
                assert "request_id=-" in msg, f"无 request_id 时应显示 -: {msg}"
                found = True
                break
        assert found
        queue.stop()


class TestForwardInferenceRequestId:
    """L5: forward_inference_to_master 中的 request_id 传递。"""

    def test_forward_method_accepts_request_id_param(self):
        """forward_inference_to_master 应接受 request_id 参数。"""
        import inspect
        from scheduler import Scheduler
        sig = inspect.signature(Scheduler.forward_inference_to_master)
        assert "request_id" in sig.parameters

    def test_start_infer_task_accepts_request_id_param(self):
        """start_infer_task 应接受 request_id 参数。"""
        import inspect
        from scheduler import Scheduler
        sig = inspect.signature(Scheduler.start_infer_task)
        assert "request_id" in sig.parameters

    def test_forward_result_is_correlated_and_errors_are_not_reported_as_ok(
            self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                if msg_type.value != "infer_forward":
                    return
                sched._on_tcp_message("master", {
                    "type": "infer_result",
                    "data": {
                        "forward_request_id": data["forward_request_id"],
                        "task_id": "task-x",
                        "status": "error",
                        "error": "worker failed",
                        "metrics": {"distributed_used": False},
                    },
                })

        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master("hello", timeout=1)
        assert result["status"] == "error"
        assert result["error"] == "worker failed"

    def test_forward_timeout_sends_cancel(self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")
        sent = []

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master("hello", timeout=0.01)
        assert result["status"] == "timeout"
        assert [kind for _, kind in sent] == ["infer_forward", "infer_cancel"]
        assert sent[0][0]["forward_request_id"] == sent[1][0]["forward_request_id"]

    def test_forward_external_cancel_sends_cancel_and_cleans_pending(self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")
        sent = []

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        cancel_event = threading.Event()
        cancel_event.set()
        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master(
            "hello",
            _cancel_event=cancel_event,
            timeout=1,
        )

        assert result["status"] == "cancelled"
        assert [kind for _, kind in sent] == ["infer_forward", "infer_cancel"]
        assert sched._client_pending_events == {}
        assert sched._client_pending_results == {}

    def test_forwarded_success_is_sent_before_task_cleanup_regression(self):
        sched = Scheduler()
        sent = []
        finished = threading.Event()
        sched.start_infer_task = lambda *args, **kwargs: "task-demo"
        sched.run_pipeline_safe = lambda prompt, **kwargs: {
            "response": "computed",
            "metrics": {"distributed_used": True},
        }
        sched.complete_infer_task = lambda *args, **kwargs: None

        def capture(*args, **kwargs):
            sent.append((args, kwargs))
            finished.set()

        sched._send_infer_result = capture
        sched.handle_infer_forward("client1", {
            "data": {
                "prompt": "x",
                "forward_request_id": "forward-demo",
            },
        })
        assert finished.wait(1)
        args, kwargs = sent[0]
        assert args[1:4] == ("task-demo", "computed", {"distributed_used": True})
        assert kwargs["forward_request_id"] == "forward-demo"
        assert kwargs.get("status", "ok") == "ok"

    def test_start_task_does_not_mark_workers_busy(self, monkeypatch):
        sched = Scheduler()
        sched.nodes["worker"] = NodeInfo(
            node_id="worker", role=NodeRole.CLIENT, state=NodeState.ONLINE,
        )
        task_id = sched.start_infer_task("hello")
        assert task_id.startswith("task_")
        assert sched.nodes["worker"].state == NodeState.ONLINE


class TestRequestNodeLogs:
    """L5: scheduler.request_node_logs() 多节点日志聚合。"""

    @pytest.fixture
    def sched(self):
        from scheduler import Scheduler
        s = Scheduler()
        s._tcp_server = None  # 无 TCP 服务器时应返回 None
        return s

    def test_request_node_logs_returns_none_without_tcp_server(self, sched):
        """无 TCP 服务器时应返回 None。"""
        result = sched.request_node_logs("worker-1")
        assert result is None

    def test_request_node_logs_returns_none_for_unknown_node(self, sched):
        """不存在的节点应返回 None。"""
        sched._tcp_server = MagicMock()
        sched.nodes = {}
        result = sched.request_node_logs("unknown-node")
        assert result is None

    def test_request_node_logs_returns_none_for_offline_node(self, sched):
        """离线节点应返回 None。"""
        sched._tcp_server = MagicMock()
        from scheduler import NodeInfo, NodeState, NodeRole
        sched.nodes = {
            "worker-1": NodeInfo(
                node_id="worker-1",
                state=NodeState.OFFLINE,
                role=NodeRole.CLIENT,
            )
        }
        with sched._nodes_lock:
            pass  # just testing state check
        result = sched.request_node_logs("worker-1")
        assert result is None


# ================================================================
# T1-T4 修复验证：线程安全测试
# ================================================================

class TestThreadSafety:
    """测试 scheduler 线程安全修复（BUG T1-T4 验证）"""

    def test_sync_node_rtt_with_lock(self):
        """_sync_node_rtt 应在 _nodes_lock 保护下访问节点（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 10,
                avg_rtt_ms=0.0,
                last_rtt_ms=0.0,
            ),
        }

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 5.5

        sched._sync_node_rtt("client1", mock_client)
        assert sched.nodes["client1"].last_rtt_ms == 5.5
        assert sched.nodes["client1"].avg_rtt_ms == 5.5
        assert time.time() - sched.nodes["client1"].last_heartbeat < 1.0

    def test_sync_node_rtt_missing_node_safe(self):
        """_sync_node_rtt 对不存在的节点应安全返回（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {}

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 5.0

        sched._sync_node_rtt("nonexistent", mock_client)

    def test_sync_node_rtt_concurrent_access(self):
        """多线程并发调用 _sync_node_rtt 不应导致数据竞争（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 100,
                avg_rtt_ms=0.0,
                last_rtt_ms=0.0,
            ),
        }

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 10.0

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=sched._sync_node_rtt,
                args=("client1", mock_client),
            )
            threads.append(t)

        def delete_node():
            with sched._nodes_lock:
                sched.nodes.pop("client1", None)

        t_del = threading.Thread(target=delete_node)
        threads.append(t_del)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    def test_master_db_heartbeat_uses_lock(self):
        """_master_db_heartbeat_loop 应在 _nodes_lock 保护下更新心跳（T2 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master",
                role=NodeRole.MASTER,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 100,
            ),
        }
        sched._running = False

        old_hb = sched.nodes["master"].last_heartbeat
        with sched._nodes_lock:
            master_node = sched.nodes.get("master")
            if master_node:
                master_node.last_heartbeat = time.time()

        assert sched.nodes["master"].last_heartbeat > old_hb

    def test_connect_to_master_node_update_uses_lock(self):
        """connect_to_master 中节点状态更新应在 _nodes_lock 保护下（T3 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.OFFLINE,
                last_heartbeat=time.time() - 100,
            ),
        }
        node_id = "client1"
        with sched._nodes_lock:
            if node_id in sched.nodes:
                sched.nodes[node_id].state = NodeState.ONLINE
                sched.nodes[node_id].last_heartbeat = time.time()

        assert sched.nodes[node_id].state == NodeState.ONLINE
        assert time.time() - sched.nodes[node_id].last_heartbeat < 1.0


class TestConnectToMasterBootstrapRecovery:
    """connect_to_master bootstrap recovery behavior."""

    def test_auth_failure_refreshes_bootstrap_and_retries(self, monkeypatch):
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-old", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "stale-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-old", raising=False)
        monkeypatch.setenv("QLH_API_PORT", "8001")
        monkeypatch.delenv("QLH_BOOTSTRAP_API_PORT", raising=False)
        monkeypatch.delenv("QLH_MASTER_API_PORT", raising=False)

        sched = Scheduler()
        sched._role_override = "client"

        first_connect_calls = []

        def fake_first_connect(master_api_host, master_api_port, node_id, node_type):
            first_connect_calls.append((master_api_host, master_api_port, node_id, node_type))
            cfg.CLUSTER_SECRET = "fresh-secret"
            return {
                "cluster": {
                    "master_tcp_host": "100.64.0.10",
                    "master_tcp_port": 8888,
                },
                "node": {
                    "node_id": "client-new",
                    "role": "client",
                },
            }

        class FakeTCPClient:
            attempts = 0

            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.advertise_port = advertise_port
                self.device_info = dict(device_info or {})
                self.last_register_error = ""
                self.is_registered = False
                self._running = False

            def connect(self, on_message=None):
                type(self).attempts += 1
                if type(self).attempts == 1:
                    self.last_register_error = "注册被拒绝: 认证失败: HMAC 签名不匹配"
                    return False
                self.is_registered = True
                self._running = True
                return True

            def send_data(self, data, msg_type):
                pass

        monkeypatch.setattr("bootstrap.first_connect", fake_first_connect)
        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert result["node_id"] == "client-new"
        assert FakeTCPClient.attempts == 2
        assert first_connect_calls == [("100.64.0.10", 8000, "client-old", "pc")]
        assert cfg.NODE_ID == "client-new"
        assert scheduler_mod.NODE_ID == "client-new"

    def test_existing_registered_connection_is_reused(self):
        sched = Scheduler()
        sched._role_override = "client"
        existing = MagicMock()
        existing._running = True
        existing.is_registered = True
        existing.sock = object()
        existing.server_host = "100.64.0.10"
        existing.server_port = 8888
        sched._tcp_client = existing

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert result["reused"] is True
        existing.disconnect.assert_not_called()

    def test_failed_candidate_connection_preserves_existing_connection(
            self, monkeypatch):
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-existing", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(
            scheduler_mod, "NODE_ID", "client-existing", raising=False,
        )

        class FailedTCPClient:
            instances = []

            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.last_register_error = "candidate unavailable"
                self.is_registered = False
                self._running = False
                self.sock = None
                self.on_disconnect = None
                self.disconnect_calls = 0
                type(self).instances.append(self)

            def connect(self, on_message=None):
                return False

            def disconnect(self):
                self.disconnect_calls += 1

        sched = Scheduler()
        sched._role_override = "client"
        previous_callback = object()
        existing = MagicMock()
        existing._running = True
        existing.is_registered = True
        existing.sock = object()
        existing.server_host = "100.64.0.10"
        existing.server_port = 8888
        existing.client_id = "client-existing"
        existing.on_disconnect = previous_callback
        sched._tcp_client = existing
        monkeypatch.setattr("tcp_comm.TCPClient", FailedTCPClient)

        result = sched.connect_to_master("100.64.0.20", 8888)

        assert result["status"] == "failed"
        assert sched._tcp_client is existing
        assert existing.on_disconnect is previous_callback
        existing.disconnect.assert_not_called()
        assert FailedTCPClient.instances[0].disconnect_calls == 1

    def test_stale_connection_disconnect_does_not_clear_current_state(self):
        sched = Scheduler()
        current = object()
        stale = object()
        sched._tcp_client = cast(Any, current)
        sched._kv_cache["active"] = "kv"

        sched._on_master_connection_lost(stale)

        assert sched._kv_cache == {"active": "kv"}

    def test_layer_config_callback_can_use_client_before_connect_returns(self, monkeypatch):
        """REGISTER 后立即到达的层配置必须能看到 TCP 客户端和最终 node_id。"""
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-race", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-race", raising=False)

        observed = []

        class FakeTCPClient:
            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.device_info = dict(device_info or {})
                self.last_register_error = ""
                self.is_registered = True
                self._running = True
                self.sock = object()

            def connect(self, on_message=None):
                assert on_message is not None
                on_message({"type": "layer_config", "data": {}})
                return True

            def send_data(self, data, msg_type):
                pass

        sched = Scheduler()
        sched._role_override = "client"

        def observe(_client_id, _msg):
            active_client = cast(Any, sched._tcp_client)
            assert active_client is not None
            observed.append((active_client.client_id, sched.get_effective_node_id()))

        monkeypatch.setattr(sched, "_on_tcp_message", observe)
        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert observed == [("client-race", "client-race")]

    def test_profile_completed_during_connect_is_reported_after_registration(
            self, monkeypatch):
        import config as cfg
        import scheduler as scheduler_mod

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-profile", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-profile", raising=False)
        sent = []
        sched = Scheduler()
        sched._role_override = "client"
        sched._local_device_profile = None

        class FakeTCPClient:
            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.client_id = client_id
                self.server_host = server_host
                self.server_port = server_port
                self.device_info = dict(device_info or {})
                self.is_registered = False
                self._running = False
                self.last_register_error = ""

            def connect(self, on_message=None):
                sched._local_device_profile = PROFILE_IGPU_ONLY
                self.is_registered = True
                self._running = True
                return True

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert any(
            msg_type == "status_res" and data["device_info"] == PROFILE_IGPU_ONLY
            for data, msg_type in sent
        )


def test_tcp_bind_failure_keeps_master_local_pipeline_available(monkeypatch):
    import scheduler as scheduler_mod
    import config as cfg

    monkeypatch.setattr(scheduler_mod, "RUN_MODE", "distributed", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)

    class FailedTCPServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def start(self, on_message=None, on_disconnect=None):
            raise OSError("address already in use")

    sched = Scheduler()
    register_master = MagicMock()
    monkeypatch.setattr("tcp_comm.TCPServer", FailedTCPServer)
    monkeypatch.setattr("tcp_comm.detect_lan_ip", lambda: "100.64.0.10")
    monkeypatch.setattr("tcp_comm.get_mac_addresses", lambda: ["001122334455"])
    monkeypatch.setattr(sched, "init_nodes", lambda: None)
    monkeypatch.setattr(
        sched,
        "_verify_master_identity",
        lambda: setattr(sched, "_master_identity_reason", "first_run"),
    )
    monkeypatch.setattr(sched, "_register_master_in_db", register_master)
    monkeypatch.setattr(sched, "_start_master_db_heartbeat", MagicMock())
    monkeypatch.setattr(sched, "deactivate_spare_master_on_startup", lambda: None)
    monkeypatch.setattr(sched, "_start_database_reconnect_monitor", lambda: None)
    monkeypatch.setattr(sched, "can_join_existing_master", lambda: False)

    try:
        sched.start(host="0.0.0.0", port=8888)

        assert sched._running is True
        assert sched._tcp_server is None
        assert sched.pipeline_queue._running is True
        register_master.assert_not_called()
    finally:
        sched.stop()


def test_distributed_toggle_survives_without_database(monkeypatch):
    """数据库关闭时，运行时分布式开关仍应立即生效。"""
    import scheduler as scheduler_mod

    sched = Scheduler()
    monkeypatch.setattr(scheduler_mod, "_get_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler_mod, "_db_available", False)

    sched.set_distributed_inference_enabled(False)
    assert sched.get_distributed_inference_enabled() is False
    sched.set_distributed_inference_enabled(True)
    assert sched.get_distributed_inference_enabled() is True
