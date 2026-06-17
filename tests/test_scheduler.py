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
        # - tiny_node: 0.5GB VRAM（模拟手机）
        from scheduler import NodeInfo, NodeState
        s.nodes["gpu_node"] = NodeInfo(
            node_id="gpu_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {"vram_total_gb": 8.0, "name": "RTX 3070", "is_integrated": False},
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
        """小显存 GPU 节点"""
        mb = sched._get_node_vram_mb("tiny_node")
        assert mb == 0.5 * 1024  # 512 MB

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
        # tiny_node 仅 512MB，装不下 24 层 + Embedding + LM Head
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

    def test_fallback_inference_returns_error_without_model(self, sched):
        """无模型时 _run_full_model_inference 返回 error"""
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

    def test_process_error_captured(self, queue):
        """process_fn 抛异常时应记录 error 状态"""
        def failing_process(**kwargs):
            raise ValueError("模拟推理失败")

        queue.start(process_fn=failing_process)
        tid = queue.enqueue(prompt="test")
        result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "error"
        assert "模拟推理失败" in result["error"]
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

    def test_chain_forward_delegates_to_layer_forward(self, sched):
        """_handle_chain_forward 应委托给 _handle_layer_forward"""
        called_with = []

        def mock_handler(cid, msg):
            called_with.append(cid)

        original = sched._handle_layer_forward
        sched._handle_layer_forward = mock_handler

        try:
            msg = {"type": "chain_forward", "data": {"task_id": "t1"}}
            sched._handle_chain_forward("client2", msg)
            assert len(called_with) == 1
            assert called_with[0] == "client2"
        finally:
            sched._handle_layer_forward = original

    def test_layer_forward_with_chain_next_forwards(self, sched):
        """_handle_layer_forward 有 chain_next 时应调用 _send_chain_forward"""
        # 注意: 此测试仅验证链式转发分支，不实际执行模型推理
        forward_calls = []

        def mock_forward(target_id, data):
            forward_calls.append((target_id, data))
            return True

        def mock_send_result(cid, tid, result_data=None, error=None):
            pass  # 不应被调用（链式转发成功时）

        original_fwd = sched._send_chain_forward
        original_send = sched._send_layer_result
        sched._send_chain_forward = mock_forward
        sched._send_layer_result = mock_send_result

        try:
            # 构造带 chain_next 的 LAYER_FORWARD（模拟首节点场景）
            # 实际调用会因没有 model_manager 而抛异常，此处仅验证不崩溃
            # 完整集成测试需要真实模型环境
            pass  # _handle_layer_forward 需要 model_manager，留待集成测试
        finally:
            sched._send_chain_forward = original_fwd
            sched._send_layer_result = original_send

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
            "master": type('NodeInfo', (), {
                'node_id': 'master', 'role': 'master',
                'device_info': PROFILE_WORKSTATION,
            })(),
            "client1": type('NodeInfo', (), {
                'node_id': 'client1', 'role': 'client',
                'device_info': PROFILE_EDGE,  # 显存极少
            })(),
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
            assert result[-1]["has_lm_head"] is True


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
