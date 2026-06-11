"""
单元测试 — 调度器动态分层算法
=============================
测试 _compute_node_weight() 和 compute_layer_assignment()
使用模拟设备数据，无需真实硬件/数据库连接。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from scheduler import Scheduler


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
        """工作站应获得最高分（接近 100 + 15 = 115）"""
        weight = sched._compute_node_weight(PROFILE_WORKSTATION)
        # VRAM: 24/24*50=50, RAM: 64/64*30=30, CPU: 16/16*10+4500/4000*10=21.25
        # Bonus: 15 → total ≈ 116.25
        assert 100 <= weight <= 120, f"工作站权重应在 100-120，实际: {weight:.1f}"

    def test_laptop_moderate_score(self, sched):
        """游戏本应获得中等分数"""
        weight = sched._compute_node_weight(PROFILE_LAPTOP)
        # VRAM: 8/24*50=16.7, RAM: 16/64*30=7.5, CPU: 8/16*10+4000/4000*10=15
        # Bonus: 15 → total ≈ 54.2
        assert 40 <= weight <= 70, f"游戏本权重应在 40-70，实际: {weight:.1f}"

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
        """独显应获得 +15 奖励"""
        with_gpu = PROFILE_LAPTOP  # 独显
        without_gpu = {**PROFILE_LAPTOP, "gpu": {**PROFILE_LAPTOP["gpu"], "cuda_available": False}}
        w_gpu = sched._compute_node_weight(with_gpu)
        w_no_gpu = sched._compute_node_weight(without_gpu)
        # 独显奖励应接近 15
        bonus = w_gpu - w_no_gpu
        assert 12 <= bonus <= 18, f"独显奖励应在 12-18，实际: {bonus:.1f}"

    def test_mobile_arm_score(self, sched):
        """移动设备应有合理分数"""
        weight = sched._compute_node_weight(PROFILE_MOBILE)
        assert 0 <= weight <= 15, f"移动设备权重应在 0-15，实际: {weight:.1f}"


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
            "master": type('NodeInfo', (), {
                'node_id': 'master', 'role': 'master',
                'device_info': PROFILE_WORKSTATION,
            })(),
            "client1": type('NodeInfo', (), {
                'node_id': 'client1', 'role': 'client',
                'device_info': PROFILE_LAPTOP,
            })(),
            "client2": type('NodeInfo', (), {
                'node_id': 'client2', 'role': 'client',
                'device_info': PROFILE_ULTRABOOK,
            })(),
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
        """第一个节点（master）应有 Embedding"""
        result = sched.compute_layer_assignment()
        result.sort(key=lambda x: x["start_layer"])
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0

    def test_last_node_has_lm_head(self, sched):
        """最后一个节点应有 LM Head"""
        result = sched.compute_layer_assignment()
        result.sort(key=lambda x: x["start_layer"])
        assert result[-1]["has_lm_head"] is True
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
        """权重高的节点应分配更多层"""
        result = sched.compute_layer_assignment()

        scores = {a["node_id"]: a["score"] for a in result}
        layers = {a["node_id"]: a["layers_count"] for a in result}

        # 工作站 > 游戏本 > 轻薄本
        assert scores["master"] > scores["client1"] > scores["client2"], \
            f"权重排序错误: {scores}"
        # 工作站应获得最多层
        assert layers["master"] >= layers["client1"], \
            f"权重高应分配更多层: {layers}"
        assert layers["client1"] >= layers["client2"], \
            f"权重高应分配更多层: {layers}"

    def test_empty_nodes_returns_empty(self, sched):
        """空节点列表应返回空"""
        result = sched.compute_layer_assignment([])
        assert result == []

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
        """两个节点时的分层应连续"""
        two = [
            {"node_id": "master", "role": "master", "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client", "device_info": PROFILE_LAPTOP},
        ]
        result = sched.compute_layer_assignment(two)
        result.sort(key=lambda x: x["start_layer"])

        assert len(result) == 2
        assert result[0]["start_layer"] == 0
        assert result[0]["end_layer"] == result[1]["start_layer"]
        assert result[1]["end_layer"] == 24
        assert result[0]["has_embedding"] is True
        assert result[1]["has_lm_head"] is True

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


# ================================================================
# get_layer_assignments 测试
# ================================================================

class TestGetLayerAssignments:
    """测试分层配置获取（内存模式，无 DB）"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {
            "master": type('NodeInfo', (), {
                'node_id': 'master', 'role': 'master',
                'device_info': PROFILE_LAPTOP,
            })(),
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
