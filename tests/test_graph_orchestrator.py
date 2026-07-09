"""
单元测试 — 图算法智能编排模块
=============================
测试 UnionFind、GraphOrchestrator（图转树 + DFS 路径搜索）、
以及 Scheduler.compute_layer_assignment() 图算法集成。

使用模拟设备数据，无需真实硬件/网络连接。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, patch

from graph_orchestrator import UnionFind, GraphOrchestrator


# ================================================================
# Mock NodeInfo — 模拟节点对象（兼容 GraphOrchestrator 接口）
# ================================================================

class MockNode:
    """模拟 NodeInfo，提供 graph_orchestrator 所需属性"""
    def __init__(self, node_id, role="client", device_info=None,
                 avg_rtt_ms=0.0, network_type="unknown", address=""):
        self.node_id = node_id
        self.role = role
        self.device_info = device_info or {}
        self.avg_rtt_ms = avg_rtt_ms
        self.network_type = network_type
        self.address = address


# ================================================================
# 模拟设备画像（与 test_scheduler.py 一致）
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

PROFILE_HIGH_VRAM = {
    "gpu": {
        "name": "NVIDIA A100",
        "vram_total_gb": 80.0,
        "cuda_available": True,
        "is_integrated": False,
    },
    "ram": {"total_gb": 256.0},
    "cpu": {"physical_cores": 32, "freq_max_mhz": 3800},
}


# ================================================================
# UnionFind 测试
# ================================================================

class TestUnionFind:
    """测试并查集基本操作"""

    def test_initial_state(self):
        """初始每个节点独立"""
        uf = UnionFind(["a", "b", "c", "d"])
        assert uf.component_count == 4
        for v in ["a", "b", "c", "d"]:
            assert uf.find(v) == v

    def test_union_same_set(self):
        """合并同集合应返回 False"""
        uf = UnionFind(["a", "b"])
        assert uf.union("a", "b") is True
        assert uf.union("a", "b") is False

    def test_union_reduces_components(self):
        """合并应减少连通分量数"""
        uf = UnionFind(["a", "b", "c", "d", "e"])
        assert uf.component_count == 5
        uf.union("a", "b")
        assert uf.component_count == 4
        uf.union("b", "c")
        assert uf.component_count == 3
        uf.union("d", "e")
        assert uf.component_count == 2

    def test_connected_after_union(self):
        """合并后节点应连通"""
        uf = UnionFind(["x", "y", "z"])
        uf.union("x", "y")
        assert uf.connected("x", "y")
        assert not uf.connected("x", "z")
        uf.union("y", "z")
        assert uf.connected("x", "z")  # 传递性

    def test_path_compression(self):
        """路径压缩：链式连接后 find 应将所有节点直接指向根"""
        uf = UnionFind(["a", "b", "c", "d", "e"])
        # 构造链 a-b-c-d-e
        uf.parent = {"a": "b", "b": "c", "c": "d", "d": "e", "e": "e"}
        # find 时应压缩路径
        root = uf.find("a")
        assert root == "e"
        # 压缩后 a 的父节点应直接是 e（或中间节点也被压缩）
        assert uf.parent["a"] == "e"

    def test_large_set(self):
        """较大集合（100 节点）"""
        vertices = [f"n{i}" for i in range(100)]
        uf = UnionFind(vertices)
        assert uf.component_count == 100
        # 合并所有偶数节点
        for i in range(0, 100, 2):
            uf.union("n0", f"n{i}")
        # 合并所有奇数节点
        for i in range(1, 100, 2):
            uf.union("n1", f"n{i}")
        assert uf.component_count == 2
        # 合并两组
        uf.union("n0", "n1")
        assert uf.component_count == 1
        assert uf.connected("n42", "n99")

    def test_empty(self):
        """空顶点集合"""
        uf = UnionFind([])
        assert uf.component_count == 0


# ================================================================
# GraphOrchestrator — 图构建 测试
# ================================================================

class TestGraphBuilding:
    """测试通信图构建"""

    def test_build_graph_basic(self):
        """基本图构建：3 个异构节点"""
        nodes = {
            "master": MockNode("master", "master",
                               PROFILE_WORKSTATION, avg_rtt_ms=0,
                               network_type="localhost"),
            "client1": MockNode("client1", "client",
                                PROFILE_LAPTOP, avg_rtt_ms=2.0,
                                network_type="wifi"),
            "client2": MockNode("client2", "client",
                                PROFILE_EDGE, avg_rtt_ms=10.0,
                                network_type="wifi"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000, total_layers=24)
        graph = orch._build_graph()

        # 顶点
        assert len(graph['vertices']) == 3
        assert 'master' in graph['vertices']
        assert 'client1' in graph['vertices']
        assert 'client2' in graph['vertices']

        # 边（全连接 C(3,2)=3）
        assert len(graph['edges']) == 3

        # 顶点属性
        master_v = graph['vertices']['master']
        assert master_v['vram_mb'] == 24.0 * 1024  # 24GB → MB
        assert master_v['compute_weight'] > 70  # 高端工作站权重高
        assert master_v['compute_delay'] < 5.0

        edge_v = graph['vertices']['client2']
        assert edge_v['compute_weight'] < 30  # 边缘设备权重低
        assert edge_v['compute_delay'] > 5.0

    def test_build_graph_bandwidth_ordering(self):
        """带宽高的节点对应边权重更大"""
        nodes = {
            "fast1": MockNode("fast1", "client", PROFILE_WORKSTATION,
                              avg_rtt_ms=1.0, network_type="ethernet"),
            "fast2": MockNode("fast2", "client", PROFILE_WORKSTATION,
                              avg_rtt_ms=1.5, network_type="ethernet"),
            "slow1": MockNode("slow1", "client", PROFILE_EDGE,
                              avg_rtt_ms=20.0, network_type="wifi"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        graph = orch._build_graph()

        # 找 fast1-fast2 边和 fast1-slow1 边
        edges_by_pair = {}
        for e in graph['edges']:
            key = tuple(sorted([e['u'], e['v']]))
            edges_by_pair[key] = e

        ff_edge = edges_by_pair.get(('fast1', 'fast2'))
        fs_edge = edges_by_pair.get(('fast1', 'slow1'))

        assert ff_edge is not None
        assert fs_edge is not None
        # fast-fast 带宽应高于 fast-slow
        assert ff_edge['bandwidth'] > fs_edge['bandwidth']
        # fast-fast 延迟应低于 fast-slow
        assert ff_edge['latency_ms'] < fs_edge['latency_ms']

    def test_build_graph_complete(self):
        """全连接图：N 节点应有 N*(N-1)/2 条边"""
        for n in [2, 3, 5, 8]:
            nodes = {
                f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                                  avg_rtt_ms=5.0)
                for i in range(n)
            }
            orch = GraphOrchestrator(nodes, model_memory_mb=2000)
            graph = orch._build_graph()
            expected_edges = n * (n - 1) // 2
            assert len(graph['edges']) == expected_edges, \
                f"{n} 节点应有 {expected_edges} 条边，实际 {len(graph['edges'])}"

    def test_build_graph_no_device_info(self):
        """无 device_info 的节点：应使用默认值"""
        nodes = {
            "a": MockNode("a", "client", {}, avg_rtt_ms=10.0),
            "b": MockNode("b", "client", None, avg_rtt_ms=15.0),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        graph = orch._build_graph()

        for v in ['a', 'b']:
            vdata = graph['vertices'][v]
            assert vdata['vram_mb'] == 0.0
            assert vdata['compute_weight'] >= 0
            assert vdata['compute_delay'] > 0

    def test_extract_vram_gpu(self):
        """VRAM 提取：GPU 专用显存"""
        orch = GraphOrchestrator({}, model_memory_mb=1000)
        assert orch._extract_vram_mb(PROFILE_WORKSTATION) == 24.0 * 1024
        assert orch._extract_vram_mb(PROFILE_LAPTOP) == 8.0 * 1024

    def test_extract_vram_ram_fallback(self):
        """VRAM 提取：无 GPU 时回退到系统 RAM"""
        orch = GraphOrchestrator({}, model_memory_mb=1000)
        assert orch._extract_vram_mb(PROFILE_EDGE) == 4.0 * 1024
        assert orch._extract_vram_mb(PROFILE_NO_GPU) == 8.0 * 1024

    def test_calc_node_weight_ranking(self):
        """节点权重排名：高端 > 中端 > 低端"""
        orch = GraphOrchestrator({}, model_memory_mb=1000)
        w_workstation = orch._calc_node_weight(PROFILE_WORKSTATION)
        w_laptop = orch._calc_node_weight(PROFILE_LAPTOP)
        w_ultrabook = orch._calc_node_weight(PROFILE_ULTRABOOK)
        w_edge = orch._calc_node_weight(PROFILE_EDGE)
        w_no_gpu = orch._calc_node_weight(PROFILE_NO_GPU)

        # 独显设备 > 集显/无GPU设备
        assert w_workstation > w_laptop > w_ultrabook
        assert w_workstation > 70  # 高端独显
        # 边缘设备（4GB RAM, 2 cores）权重应较低
        assert w_edge < 30
        # NO_GPU（8GB RAM, 4 cores）权重高于 EDGE（4GB RAM, 2 cores）
        assert w_no_gpu > w_edge

    def test_estimate_bandwidth_from_rtt(self):
        """带宽估算：RTT 反推"""
        nodes = {
            "low_rtt": MockNode("low_rtt", "client", {}, avg_rtt_ms=2.0),
            "high_rtt": MockNode("high_rtt", "client", {}, avg_rtt_ms=50.0),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        bw_low = orch._estimate_bandwidth("low_rtt")
        bw_high = orch._estimate_bandwidth("high_rtt")
        assert bw_low > bw_high
        # 2ms RTT → ~333, 50ms RTT → ~19.6
        assert bw_low > 100
        assert bw_high < 100

    def test_estimate_bandwidth_network_type_fallback(self):
        """带宽估算：无 RTT 时按 network_type 估算"""
        nodes = {
            "eth": MockNode("eth", "client", {}, avg_rtt_ms=0, network_type="ethernet"),
            "wifi": MockNode("wifi", "client", {}, avg_rtt_ms=0, network_type="wifi"),
            "unk": MockNode("unk", "client", {}, avg_rtt_ms=0, network_type="unknown"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        bw_eth = orch._estimate_bandwidth("eth")
        bw_wifi = orch._estimate_bandwidth("wifi")
        bw_unk = orch._estimate_bandwidth("unk")
        assert bw_eth > bw_wifi > bw_unk

    def test_get_node_latency(self):
        """节点延迟获取"""
        nodes = {
            "n1": MockNode("n1", "client", {}, avg_rtt_ms=10.0),
            "n2": MockNode("n2", "client", {}, avg_rtt_ms=0, network_type="ethernet"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        lat1 = orch._get_node_latency("n1")
        lat2 = orch._get_node_latency("n2")
        assert lat1 == 5.0  # RTT/2
        assert lat2 == 1.0  # ethernet 默认
        # 未知节点
        assert orch._get_node_latency("nonexist") == 50.0


# ================================================================
# GraphOrchestrator — 最大带宽生成树 测试
# ================================================================

class TestMaxBandwidthSpanningTree:
    """测试 Kruskal 最大带宽生成树"""

    def test_tree_has_correct_edge_count(self):
        """生成树边数 = |V| - 1"""
        for n in [3, 5, 8]:
            nodes = {
                f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                                  avg_rtt_ms=float(i + 1))
                for i in range(n)
            }
            orch = GraphOrchestrator(nodes, model_memory_mb=10000)
            graph = orch._build_graph()
            tree = orch._max_bandwidth_spanning_tree(graph)

            tree_edges = sum(len(neighbors) for neighbors in tree.values()) // 2
            assert tree_edges == n - 1, \
                f"{n} 节点树应有 {n-1} 条边，实际 {tree_edges}"

    def test_tree_is_connected(self):
        """生成树应连通所有节点"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=float(i + 1))
            for i in range(6)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=10000)
        graph = orch._build_graph()
        tree = orch._max_bandwidth_spanning_tree(graph)

        # BFS 验证连通性
        all_nodes = set(tree.keys())
        visited = set()
        stack = [list(all_nodes)[0]]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for neighbor, _, _ in tree[v]:
                if neighbor not in visited:
                    stack.append(neighbor)

        assert visited == all_nodes, \
            f"生成树不连通: 访问了 {len(visited)}/{len(all_nodes)} 节点"

    def test_tree_prefers_high_bandwidth(self):
        """生成树应优先保留高带宽边"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION, avg_rtt_ms=1.0),
            "B": MockNode("B", "client", PROFILE_WORKSTATION, avg_rtt_ms=1.5),
            "C": MockNode("C", "client", PROFILE_EDGE, avg_rtt_ms=50.0),
            "D": MockNode("D", "client", PROFILE_EDGE, avg_rtt_ms=60.0),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=10000)
        graph = orch._build_graph()
        tree = orch._max_bandwidth_spanning_tree(graph)

        # A-B 带宽最高，应该在树中
        ab_in_tree = any(n == "B" for n, _, _ in tree["A"])
        assert ab_in_tree, "高带宽边 A-B 应在生成树中"

        # 收集树中所有边的带宽
        tree_edges = set()
        for u, neighbors in tree.items():
            for v, bw, _ in neighbors:
                if (v, u) not in tree_edges:
                    tree_edges.add((u, v))
        # 树边应有 3 条
        assert len(tree_edges) == 3

    def test_no_cycles_in_tree(self):
        """生成树应无环路"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=float(i))
            for i in range(5)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=10000)
        graph = orch._build_graph()
        tree = orch._max_bandwidth_spanning_tree(graph)

        # DFS 检测环路
        visited = set()

        def check_cycle(node, parent):
            visited.add(node)
            for neighbor, _, _ in tree[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True
                if check_cycle(neighbor, node):
                    return True
            return False

        start = list(tree.keys())[0]
        assert not check_cycle(start, None), "生成树不应有环路"

    def test_disconnected_graph_partial_tree(self):
        """不连通图：生成部分树（森林）"""
        # 创建两组互不连通的节点（通过极高的延迟模拟不连通）
        # 实际上全连接图总是连通的。此测试验证算法在不完整边集下不崩溃。
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION),
            "B": MockNode("B", "client", PROFILE_WORKSTATION),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        graph = orch._build_graph()
        tree = orch._max_bandwidth_spanning_tree(graph)

        # 应有 1 条边
        tree_edges = sum(len(n) for n in tree.values()) // 2
        assert tree_edges == 1

    def test_tree_preserves_all_vertices(self):
        """生成树应保留所有顶点"""
        for n in [2, 4, 7]:
            nodes = {
                f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                                  avg_rtt_ms=float(10 + i))
                for i in range(n)
            }
            orch = GraphOrchestrator(nodes, model_memory_mb=10000)
            graph = orch._build_graph()
            tree = orch._max_bandwidth_spanning_tree(graph)
            assert set(tree.keys()) == set(nodes.keys()), \
                f"树应包含所有 {n} 个顶点"


# ================================================================
# GraphOrchestrator — DFS 路径搜索 测试
# ================================================================

class TestDFSPathSearch:
    """测试 DFS 路径搜索"""

    def _make_tree_and_vertices(self, nodes_dict):
        """辅助：构建通信图 + 生成树 + 顶点数据"""
        orch = GraphOrchestrator(nodes_dict, model_memory_mb=10000)
        graph = orch._build_graph()
        tree = orch._max_bandwidth_spanning_tree(graph)
        return tree, graph['vertices']

    def test_finds_path_when_memory_sufficient(self):
        """显存充足时应找到路径"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION),  # 24GB
            "B": MockNode("B", "client", PROFILE_LAPTOP),       # 8GB
            "C": MockNode("C", "client", PROFILE_ULTRABOOK),    # 0.5GB
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=5000)  # 需要 5GB
        tree, vertices = self._make_tree_and_vertices(nodes)
        path = orch._dfs_path_search(tree, vertices)

        assert len(path) >= 1
        # 路径上的显存总和应 ≥ 5000MB
        total_mem = sum(vertices[n]['vram_mb'] for n in path)
        assert total_mem >= 5000

    def test_returns_best_effort_when_memory_insufficient(self):
        """显存不足时应返回最大覆盖路径"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_ULTRABOOK),  # 0.5GB
            "B": MockNode("B", "client", PROFILE_EDGE),       # 4GB
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=50000)  # 需要 50GB (不可能)
        tree, vertices = self._make_tree_and_vertices(nodes)
        path = orch._dfs_path_search(tree, vertices)

        # 应返回路径（best-effort）
        assert len(path) >= 1
        total_mem = sum(vertices[n]['vram_mb'] for n in path)
        assert total_mem > 0

    def test_path_is_simple_no_duplicates(self):
        """返回的路径应无重复节点"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=float(5 + i))
            for i in range(6)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        tree, vertices = self._make_tree_and_vertices(nodes)
        path = orch._dfs_path_search(tree, vertices)

        assert len(path) == len(set(path)), "路径不应有重复节点"

    def test_path_contains_valid_tree_nodes(self):
        """路径节点应都在树中"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=float(5 + i))
            for i in range(5)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        tree, vertices = self._make_tree_and_vertices(nodes)
        path = orch._dfs_path_search(tree, vertices)

        for n in path:
            assert n in tree, f"路径节点 {n} 应在树中"

    def test_high_bandwidth_nodes_preferred(self):
        """高带宽节点应优先出现在路径中"""
        nodes = {
            "fast_A": MockNode("fast_A", "client", PROFILE_WORKSTATION,
                               avg_rtt_ms=1.0, network_type="ethernet"),
            "fast_B": MockNode("fast_B", "client", PROFILE_LAPTOP,
                               avg_rtt_ms=2.0, network_type="ethernet"),
            "slow_C": MockNode("slow_C", "client", PROFILE_ULTRABOOK,
                               avg_rtt_ms=100.0, network_type="wifi"),
            "slow_D": MockNode("slow_D", "client", PROFILE_EDGE,
                               avg_rtt_ms=120.0, network_type="wifi"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        tree, vertices = self._make_tree_and_vertices(nodes)
        path = orch._dfs_path_search(tree, vertices)

        # fast 节点应在 slow 节点前面（因为延迟更低）
        fast_indices = [i for i, n in enumerate(path) if n.startswith("fast")]
        slow_indices = [i for i, n in enumerate(path) if n.startswith("slow")]
        if fast_indices and slow_indices:
            assert max(fast_indices) < max(slow_indices) or \
                   min(fast_indices) < min(slow_indices), \
                   "高带宽节点应倾向排在前面"

    def test_empty_tree_returns_empty(self):
        """空树应返回空路径"""
        orch = GraphOrchestrator({}, model_memory_mb=1000)
        path = orch._dfs_path_search({}, {})
        assert path == []


# ================================================================
# GraphOrchestrator — 层分配 测试
# ================================================================

class TestLayerAssignment:
    """测试沿路径的层均衡分配"""

    def _make_orchestrator(self, nodes_dict, model_memory_mb=2000):
        return GraphOrchestrator(nodes_dict, model_memory_mb=model_memory_mb,
                                 total_layers=24)

    def test_total_layers_sum_to_24(self):
        """分配的总层数应为 24"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION),
            "B": MockNode("B", "client", PROFILE_LAPTOP),
            "C": MockNode("C", "client", PROFILE_EDGE),
        }
        orch = self._make_orchestrator(nodes)
        assignments = orch._assign_layers(["A", "B", "C"], {
            "A": {"vram_mb": 24*1024, "compute_weight": 100, "compute_delay": 1.0, "role": "client"},
            "B": {"vram_mb": 8*1024, "compute_weight": 60, "compute_delay": 4.0, "role": "client"},
            "C": {"vram_mb": 4*1024, "compute_weight": 20, "compute_delay": 8.0, "role": "client"},
        })

        total = sum(a["layers_count"] for a in assignments)
        assert total == 24, f"总层数应为 24，实际 {total}"

    def test_first_has_embedding_last_has_lm_head(self):
        """首节点含 Embedding，末节点含 LM Head"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION),
            "B": MockNode("B", "client", PROFILE_LAPTOP),
        }
        orch = self._make_orchestrator(nodes)
        assignments = orch._assign_layers(["A", "B"], {
            "A": {"vram_mb": 24*1024, "compute_weight": 100, "compute_delay": 1.0, "role": "client"},
            "B": {"vram_mb": 8*1024, "compute_weight": 60, "compute_delay": 4.0, "role": "client"},
        })

        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["has_lm_head"] is False
        assert assignments[-1]["has_embedding"] is False
        assert assignments[-1]["has_lm_head"] is True

    def test_layers_continuous(self):
        """层区间应连续无空隙"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP)
            for i in range(5)
        }
        orch = self._make_orchestrator(nodes)
        vertices = {
            f"n{i}": {"vram_mb": 8000, "compute_weight": 50, "compute_delay": 5.0, "role": "client"}
            for i in range(5)
        }
        assignments = orch._assign_layers(
            [f"n{i}" for i in range(5)], vertices
        )

        for i in range(len(assignments) - 1):
            assert assignments[i]["end_layer"] == assignments[i + 1]["start_layer"], \
                f"层区间不连续: [{assignments[i]['end_layer']}] vs [{assignments[i+1]['start_layer']}]"

        assert assignments[0]["start_layer"] == 0
        assert assignments[-1]["end_layer"] == 24

    def test_min_one_layer_per_node(self):
        """每节点至少 1 层"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP)
            for i in range(30)  # 多于 24 层
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000, total_layers=24)
        vertices = {
            f"n{i}": {"vram_mb": 8000, "compute_weight": 50, "compute_delay": 5.0, "role": "client"}
            for i in range(30)
        }
        # 只取前 10 个（层数足够每人至少 1 层）
        path = [f"n{i}" for i in range(10)]
        assignments = orch._assign_layers(path, {
            k: v for k, v in vertices.items() if k in path
        })

        for a in assignments:
            assert a["layers_count"] >= 1, \
                f"每个节点至少 1 层，{a['node_id']} 有 {a['layers_count']}"

    def test_vram_proportional_allocation(self):
        """显存大的节点应分配更多层"""
        nodes = {
            "big": MockNode("big", "client", PROFILE_WORKSTATION),     # 24GB
            "small": MockNode("small", "client", PROFILE_ULTRABOOK),   # 0.5GB
        }
        orch = self._make_orchestrator(nodes)
        assignments = orch._assign_layers(["big", "small"], {
            "big": {"vram_mb": 24*1024, "compute_weight": 100, "compute_delay": 1.0, "role": "client"},
            "small": {"vram_mb": 512, "compute_weight": 20, "compute_delay": 9.0, "role": "client"},
        })

        big_layers = assignments[0]["layers_count"]
        small_layers = assignments[1]["layers_count"]
        assert big_layers > small_layers, \
            f"显存大的节点应分配更多层: big={big_layers}, small={small_layers}"

    def test_single_node_gets_all_layers(self):
        """单节点路径：全部分配"""
        nodes = {"A": MockNode("A", "client", PROFILE_WORKSTATION)}
        orch = self._make_orchestrator(nodes)
        assignments = orch._assign_layers(["A"], {
            "A": {"vram_mb": 24*1024, "compute_weight": 100, "compute_delay": 1.0, "role": "client"},
        })

        assert len(assignments) == 1
        assert assignments[0]["layers_count"] == 24
        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["has_lm_head"] is True

    def test_empty_path_returns_empty(self):
        """空路径应返回空列表"""
        orch = self._make_orchestrator({})
        assert orch._assign_layers([], {}) == []

    def test_equal_split_when_zero_vram(self):
        """显存全为 0 时均分"""
        nodes = {"A": MockNode("A"), "B": MockNode("B"), "C": MockNode("C")}
        orch = self._make_orchestrator(nodes)
        assignments = orch._assign_layers(["A", "B", "C"], {
            "A": {"vram_mb": 0, "compute_weight": 0, "compute_delay": 5.0, "role": "client"},
            "B": {"vram_mb": 0, "compute_weight": 0, "compute_delay": 5.0, "role": "client"},
            "C": {"vram_mb": 0, "compute_weight": 0, "compute_delay": 5.0, "role": "client"},
        })

        assert len(assignments) == 3
        layers = [a["layers_count"] for a in assignments]
        assert sum(layers) == 24
        assert max(layers) - min(layers) <= 1  # 尽量均匀


# ================================================================
# GraphOrchestrator — 完整编排 (orchestrate) 测试
# ================================================================

class TestOrchestrateFull:
    """测试完整编排流程"""

    def test_orchestrate_returns_valid_assignments(self):
        """完整编排应返回有效的分层方案"""
        nodes = {
            "master": MockNode("master", "master", PROFILE_WORKSTATION,
                               network_type="localhost"),
            "client1": MockNode("client1", "client", PROFILE_LAPTOP,
                                avg_rtt_ms=2.0, network_type="wifi"),
            "client2": MockNode("client2", "client", PROFILE_ULTRABOOK,
                                avg_rtt_ms=5.0, network_type="wifi"),
            "client3": MockNode("client3", "client", PROFILE_EDGE,
                                avg_rtt_ms=10.0, network_type="wifi"),
            "client4": MockNode("client4", "client", PROFILE_HIGH_VRAM,
                                avg_rtt_ms=3.0, network_type="ethernet"),
            "client5": MockNode("client5", "client", PROFILE_NO_GPU,
                                avg_rtt_ms=8.0, network_type="wifi"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        assignments = orch.orchestrate()

        # 基本结构验证
        assert isinstance(assignments, list)
        assert len(assignments) >= 1

        for a in assignments:
            assert "node_id" in a
            assert "role" in a
            assert "start_layer" in a
            assert "end_layer" in a
            assert "layers_count" in a
            assert "has_embedding" in a
            assert "has_lm_head" in a
            assert "score" in a
            assert a["layers_count"] >= 1
            assert a["start_layer"] < a["end_layer"]

        # 连续性
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24

    def test_single_node_full(self):
        """单节点：orchestrate 返回全部层"""
        nodes = {"master": MockNode("master", "master", PROFILE_WORKSTATION)}
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        assignments = orch.orchestrate()

        assert len(assignments) == 1
        assert assignments[0]["layers_count"] == 24
        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["has_lm_head"] is True

    def test_get_chain_topology(self):
        """get_chain_topology 返回纯节点 ID 列表"""
        nodes = {
            "master": MockNode("master", "master", PROFILE_WORKSTATION,
                               network_type="localhost"),
            "client1": MockNode("client1", "client", PROFILE_LAPTOP,
                                avg_rtt_ms=2.0),
            "client2": MockNode("client2", "client", PROFILE_HIGH_VRAM,
                                avg_rtt_ms=3.0),
            "client3": MockNode("client3", "client", PROFILE_ULTRABOOK,
                                avg_rtt_ms=5.0),
            "client4": MockNode("client4", "client", PROFILE_EDGE,
                                avg_rtt_ms=10.0),
            "client5": MockNode("client5", "client", PROFILE_NO_GPU,
                                avg_rtt_ms=8.0),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        chain = orch.get_chain_topology()

        assert isinstance(chain, list)
        assert len(chain) >= 1
        assert all(isinstance(n, str) for n in chain)
        # 顺序应与 assignments 一致
        for n in chain:
            assert n in nodes

    def test_fallback_when_empty_nodes(self):
        """空节点：返回空"""
        orch = GraphOrchestrator({}, model_memory_mb=500)
        assert orch.orchestrate() == []

    def test_fallback_weight_assignment(self):
        """回退方案应生成有效分配，且 master 排在首位"""
        nodes = {
            "B": MockNode("B", "client", PROFILE_EDGE),
            "master": MockNode("master", "master", PROFILE_LAPTOP),  # 低权重master
            "A": MockNode("A", "client", PROFILE_WORKSTATION),       # 高权重client
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        assignments = orch._fallback_weight_assignment()

        assert len(assignments) == 3
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24
        # master 应排在第一位（无论权重高低）
        assert assignments[0]["node_id"] == "master"
        assert assignments[0]["has_embedding"] is True
        assert assignments[-1]["has_lm_head"] is True

    def test_edge_case_identical_nodes(self):
        """全相同配置的节点：应均分"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=5.0)
            for i in range(6)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        assignments = orch.orchestrate()

        total = sum(a["layers_count"] for a in assignments)
        assert total == 24
        # 所有节点层数差不超过 1
        layers = [a["layers_count"] for a in assignments]
        assert max(layers) - min(layers) <= 1, \
            f"同构节点应均分，实际 {layers}"

    def test_orchestrate_model_memory_conservative(self):
        """大模型显存需求 — 验证约束感知"""
        nodes = {
            "big1": MockNode("big1", "client", PROFILE_HIGH_VRAM),    # 80GB
            "big2": MockNode("big2", "client", PROFILE_WORKSTATION),  # 24GB
            "small": MockNode("small", "client", PROFILE_EDGE),       # 4GB
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=50000, total_layers=24)
        assignments = orch.orchestrate()

        # 需要 50GB，big1(80GB) 单独够
        used_nodes = [a["node_id"] for a in assignments]
        assert "big1" in used_nodes  # big1 有 80GB，可单独满足

        # 若需要 100GB，则 big1+big2 才够
        orch2 = GraphOrchestrator(nodes, model_memory_mb=100000, total_layers=24)
        assignments2 = orch2.orchestrate()
        used2 = [a["node_id"] for a in assignments2]
        assert "big1" in used2
        assert "big2" in used2 or len(used2) >= 2  # 需要至少 2 个节点


# ================================================================
# Scheduler 集成 — 图算法编排阈值
# ================================================================

class TestSchedulerGraphIntegration:
    """测试 Scheduler.compute_layer_assignment() 中的图算法集成"""

    @pytest.fixture
    def scheduler(self):
        """创建模拟 Scheduler（无 DB/TCP 依赖）"""
        from scheduler import Scheduler
        s = Scheduler()
        # 手动填充节点
        return s

    def _add_nodes_to_scheduler(self, scheduler, nodes_dict):
        """将 MockNode 添加到 scheduler.nodes"""
        for nid, mn in nodes_dict.items():
            from scheduler import NodeInfo, NodeState, NodeRole
            role = NodeRole.MASTER if mn.role == "master" else NodeRole.CLIENT
            scheduler.nodes[nid] = NodeInfo(
                node_id=nid,
                role=role,
                state=NodeState.ONLINE,
                device_info=mn.device_info,
                network_type=mn.network_type,
                avg_rtt_ms=mn.avg_rtt_ms,
            )

    def test_uses_simple_weight_when_below_threshold(self, scheduler):
        """节点数 ≤ 阈值时使用简单权重分配"""
        nodes = {
            "master": MockNode("master", "master", PROFILE_WORKSTATION),
            "client1": MockNode("client1", "client", PROFILE_LAPTOP),
            "client2": MockNode("client2", "client", PROFILE_ULTRABOOK),
        }
        self._add_nodes_to_scheduler(scheduler, nodes)

        with patch('config.GRAPH_ORCHESTRATOR_THRESHOLD', 5):
            assignments = scheduler.compute_layer_assignment()

        assert len(assignments) >= 1
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24

    def test_uses_graph_orchestrator_above_threshold(self, scheduler):
        """节点数 > 阈值时应使用图算法智能编排"""
        nodes = {}
        for i in range(7):
            role = "master" if i == 0 else "client"
            profile = [PROFILE_WORKSTATION, PROFILE_LAPTOP, PROFILE_HIGH_VRAM,
                       PROFILE_ULTRABOOK, PROFILE_EDGE, PROFILE_NO_GPU, PROFILE_MOBILE][i]
            nodes[f"n{i}"] = MockNode(f"n{i}", role, profile,
                                      avg_rtt_ms=float(i + 1))
        self._add_nodes_to_scheduler(scheduler, nodes)

        with patch('config.GRAPH_ORCHESTRATOR_THRESHOLD', 5):
            assignments = scheduler.compute_layer_assignment()

        assert len(assignments) >= 1
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24

        # master 参与首段层执行，所有节点至少 1 层
        for a in assignments:
            assert a["layers_count"] >= 1, \
                f"节点 {a['node_id']} 应至少 1 层，实际: {a['layers_count']}"

    def test_simple_weight_excluded_master_first(self, scheduler):
        """简单权重：master 作为首段 Embedding 锚点排在第一位"""
        nodes = {
            "master": MockNode("master", "master", PROFILE_WORKSTATION),
            "z_client": MockNode("z_client", "client", PROFILE_HIGH_VRAM),  # 权重更高但角色为 client
            "a_client": MockNode("a_client", "client", PROFILE_LAPTOP),
        }
        self._add_nodes_to_scheduler(scheduler, nodes)

        with patch('config.GRAPH_ORCHESTRATOR_THRESHOLD', 5):
            assignments = scheduler.compute_layer_assignment()

        # master 在首位，作为首段 Embedding 锚点至少执行 1 层
        assert assignments[0]["node_id"] == "master"
        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["layers_count"] >= 1, \
            f"master 应至少执行 1 层，实际: {assignments[0]['layers_count']}"
        assert "coordinator_only" not in assignments[0]

    def test_fallback_on_graph_error(self, scheduler):
        """图算法异常时应回退到简单权重分配"""
        nodes = {}
        for i in range(7):
            nodes[f"n{i}"] = MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                                      avg_rtt_ms=float(i + 1))
        self._add_nodes_to_scheduler(scheduler, nodes)

        with patch('config.GRAPH_ORCHESTRATOR_THRESHOLD', 5):
            with patch('graph_orchestrator.GraphOrchestrator.orchestrate',
                       side_effect=RuntimeError("模拟图算法崩溃")):
                assignments = scheduler.compute_layer_assignment()

        # 应回退成功
        assert len(assignments) >= 1
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24

    def test_vram_constraint_applied_after_orchestration(self, scheduler):
        """图算法编排后仍应用显存约束校验"""
        nodes = {}
        for i in range(7):
            # 全部使用边缘设备（4GB VRAM），模型需要较多显存
            nodes[f"n{i}"] = MockNode(f"n{i}", "client", PROFILE_EDGE,
                                      avg_rtt_ms=float(i + 1))
        self._add_nodes_to_scheduler(scheduler, nodes)

        with patch('config.GRAPH_ORCHESTRATOR_THRESHOLD', 5):
            assignments = scheduler.compute_layer_assignment()

        assert len(assignments) >= 1
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24


# ================================================================
# 边界情况
# ================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_two_nodes_single_edge(self):
        """2 节点：图有 1 条边"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION, avg_rtt_ms=5.0),  # 24GB
            "B": MockNode("B", "client", PROFILE_LAPTOP, avg_rtt_ms=10.0),      # 8GB
        }
        # A=24GB < 30GB → 必须两节点都参与
        orch = GraphOrchestrator(nodes, model_memory_mb=30000)
        graph = orch._build_graph()
        assert len(graph['edges']) == 1

        tree = orch._max_bandwidth_spanning_tree(graph)
        tree_edges = sum(len(n) for n in tree.values()) // 2
        assert tree_edges == 1

        path = orch._dfs_path_search(tree, graph['vertices'])
        assert len(path) == 2  # 两节点都必须参与才够 30GB

    def test_all_nodes_ethernet(self):
        """全以太网节点：低延迟高带宽"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              network_type="ethernet")
            for i in range(6)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500)
        graph = orch._build_graph()

        # 所有边应有高带宽
        for e in graph['edges']:
            assert e['bandwidth'] >= 50, \
                f"以太网带宽应较高: {e['bandwidth']}"
            assert e['latency_ms'] <= 5.0, \
                f"以太网延迟应较低: {e['latency_ms']}"

    def test_invalid_node_id_handling(self):
        """不存在的节点 ID 应返回默认值"""
        orch = GraphOrchestrator({}, model_memory_mb=500)
        assert orch._estimate_bandwidth("nonexist") == 10.0
        assert orch._get_node_latency("nonexist") == 50.0

    def test_large_node_count_performance(self):
        """较大的节点数（30）— 验证算法不崩溃且高效"""
        nodes = {
            f"n{i}": MockNode(f"n{i}", "client", PROFILE_LAPTOP,
                              avg_rtt_ms=float(5 + i % 20))
            for i in range(30)
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=500, total_layers=24)
        assignments = orch.orchestrate()

        assert len(assignments) >= 1
        total = sum(a["layers_count"] for a in assignments)
        assert total == 24
        # 30 个节点的编排应在合理时间内完成（远小于 1 秒）

    def test_assign_layers_handles_excess_nodes(self):
        """_assign_layers 在节点数 > 层数时不应崩溃，总层数不超过 model 层数"""
        from graph_orchestrator import GraphOrchestrator

        # 构造 30 个低配节点
        nodes = {}
        for i in range(30):
            nid = f"node{i}"
            nodes[nid] = type('MockNode', (), {
                'node_id': nid,
                'role': 'client',
                'device_info': {},
                'avg_rtt_ms': 0.0,
                'network_type': 'wifi',
            })()

        orch = GraphOrchestrator(nodes=nodes, model_memory_mb=1000, total_layers=24)

        # 直接测试 _assign_layers：30 节点路径，仅 24 层
        vertices = {
            nid: {'vram_mb': 100, 'compute_weight': 10, 'compute_delay': 5.0,
                  'role': 'client'}
            for nid in nodes
        }
        path = list(nodes.keys())
        result = orch._assign_layers(path, vertices)

        total = sum(a["layers_count"] for a in result)
        assert total <= 24, f"总层数 {total} 不应超过 24，节点数={len(result)}"
        # 至少应有部分节点被分配（≤ 24 个节点有层）
        assert 1 <= len(result) <= 24

    def test_assign_layers_overflow_squeezes_nodes_to_zero(self):
        """diff<0 时，最低显存节点被削减至 0 层后被正确跳过。"""
        nodes = {}
        for i in range(30):
            nid = f"n{i}"
            nodes[nid] = type('MockNode', (), {
                'node_id': nid,
                'role': 'client',
                'device_info': {},
                'avg_rtt_ms': 0.0,
                'network_type': 'wifi',
            })()

        orch = GraphOrchestrator(nodes=nodes, model_memory_mb=1000, total_layers=24)
        # 所有节点显存相同 → 均分 → 30节点×1层=30 超过24 → diff=-6
        vertices = {
            nid: {'vram_mb': 100, 'compute_weight': 10, 'compute_delay': 5.0,
                  'role': 'client'}
            for nid in nodes
        }
        path = list(nodes.keys())
        result = orch._assign_layers(path, vertices)

        total = sum(a["layers_count"] for a in result)
        assert total == 24
        assert result[0]["has_embedding"] is True
        assert result[-1]["has_lm_head"] is True
        # 应有恰好 24 个非零节点
        assert len(result) == 24

    def test_assign_layers_zero_layer_nodes_not_in_result(self):
        """被削减至 0 层的节点不应出现在结果中。"""
        # 3 节点，total_layers=2 → 均分 1/1/0
        nodes = {
            "A": type('MockNode', (), {
                'node_id': 'A', 'role': 'client',
                'device_info': {}, 'avg_rtt_ms': 0.0, 'network_type': 'wifi',
            })(),
            "B": type('MockNode', (), {
                'node_id': 'B', 'role': 'client',
                'device_info': {}, 'avg_rtt_ms': 0.0, 'network_type': 'wifi',
            })(),
            "C": type('MockNode', (), {
                'node_id': 'C', 'role': 'client',
                'device_info': {}, 'avg_rtt_ms': 0.0, 'network_type': 'wifi',
            })(),
        }
        orch = GraphOrchestrator(nodes=nodes, model_memory_mb=1000, total_layers=2)
        vertices = {
            nid: {'vram_mb': 100, 'compute_weight': 10, 'compute_delay': 5.0,
                  'role': 'client'}
            for nid in nodes
        }
        path = ["A", "B", "C"]
        result = orch._assign_layers(path, vertices)
        assert len(result) == 2
        node_ids = {a["node_id"] for a in result}
        assert len(node_ids) == 2


# ================================================================
# GraphOrchestrator — 延迟与带宽模型 测试
# ================================================================


class TestLatencyBandwidthModel:
    """测试通信图中的延迟与带宽估算模型"""

    def test_compute_delay_uses_max_weight_constant(self):
        """compute_delay 应使用 _MAX_NODE_WEIGHT (115) 归一化。"""
        from graph_orchestrator import _MAX_NODE_WEIGHT
        assert _MAX_NODE_WEIGHT == 115.0

        nodes = {
            "full_score": MockNode("full_score", "client", PROFILE_WORKSTATION),
            "zero_score": MockNode("zero_score", "client", {
                "gpu": {}, "ram": {"total_gb": 0}, "cpu": {"physical_cores": 1, "freq_max_mhz": 1000},
            }),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=1000)
        graph = orch._build_graph()

        # 高分节点 → 低延迟
        assert graph['vertices']['full_score']['compute_delay'] < 5.0
        # 低分节点 → 接近 max_delay = 10.0
        assert graph['vertices']['zero_score']['compute_delay'] >= 9.0

    def test_edge_latency_symmetric(self):
        """同一条边两个方向的 latency 应相等。"""
        nodes = {
            "A": MockNode("A", "client", PROFILE_WORKSTATION, avg_rtt_ms=10.0),
            "B": MockNode("B", "client", PROFILE_LAPTOP, avg_rtt_ms=20.0),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        graph = orch._build_graph()
        edge = graph['edges'][0]
        assert edge['latency_ms'] == orch._get_node_latency("A") + orch._get_node_latency("B")
        assert edge['latency_ms'] == 5.0 + 10.0  # RTT/2

    def test_bandwidth_min_of_both_ends(self):
        """边带宽 = min(bw_u, bw_v) × 0.8"""
        nodes = {
            "fast": MockNode("fast", "client", PROFILE_WORKSTATION,
                            avg_rtt_ms=1.0, network_type="ethernet"),
            "slow": MockNode("slow", "client", PROFILE_EDGE,
                            avg_rtt_ms=0, network_type="wifi"),
        }
        orch = GraphOrchestrator(nodes, model_memory_mb=2000)
        graph = orch._build_graph()
        edge = graph['edges'][0]
        bw_fast = orch._estimate_bandwidth("fast")     # from RTT: 1000/(1+1)=500
        bw_slow = orch._estimate_bandwidth("slow")      # from network_type: 30
        expected = round(min(bw_fast, bw_slow) * 0.8, 2)
        assert edge['bandwidth'] == expected

    def test_unknown_node_returns_defaults(self):
        """未知节点应返回安全的默认带宽和延迟。"""
        orch = GraphOrchestrator({}, model_memory_mb=500)
        assert orch._estimate_bandwidth("ghost") == 10.0
        assert orch._get_node_latency("ghost") == 50.0

    def test_node_without_device_info_has_zero_vram(self):
        """无 device_info 节点的 VRAM 应为 0。"""
        orch = GraphOrchestrator({}, model_memory_mb=500)
        assert orch._extract_vram_mb({}) == 0.0
        assert orch._extract_vram_mb(None) == 0.0

