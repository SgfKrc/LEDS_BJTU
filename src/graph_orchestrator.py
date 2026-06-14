"""
图算法智能编排模块 — 最大带宽生成树 + DFS 路径搜索
==================================================
将边缘通信全图转换为最大带宽生成树（图转树），再通过树上受限 DFS 搜索，
在多项式时间内得到最优流水线节点编排方案。

核心算法（参见 docs/图算法.md）:
  1. 构建全连接通信图（节点=顶点, 链路=边, 权重=带宽+延迟）
  2. Kruskal 贪心 → 最大带宽生成树（优先保留高带宽链路）
  3. DFS 带剪枝搜索 → 满足显存约束的最小延迟路径
  4. 按显存占比均衡分配模型层 → 输出链式拓扑

适用条件: 节点数 > GRAPH_ORCHESTRATOR_THRESHOLD（默认 5，见 config.py）
退化方案: 简单权重比例分配（节点数 ≤ 阈值）

复杂度:
  - 生成树: O(|E| log |E|)
  - DFS 搜索: O(|V|²)
  - 总体: O(|V|² log |V|)（全连接图 |E| ≈ |V|²/2）
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ================================================================
# Union-Find（并查集）— 路径压缩 + 按秩合并
# ================================================================

class UnionFind:
    """
    并查集 — 用于 Kruskal 算法快速判断连通性与合并分量。

    特性:
      - 路径压缩: find() 时将沿途节点直接挂到根节点
      - 按秩合并: union() 时将秩小的树挂到秩大的树下
      - 近似常数时间复杂度 O(α(N))，α 为反 Ackermann 函数
    """

    def __init__(self, vertices: List[str]):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, v: str) -> str:
        """查找根节点（路径压缩）"""
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u: str, v: str) -> bool:
        """
        合并两个连通分量（按秩优化）。

        Returns:
            True  合并成功（之前不连通）
            False 已在同一分量（无操作）
        """
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return False
        if self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        elif self.rank[ru] > self.rank[rv]:
            self.parent[rv] = ru
        else:
            self.parent[rv] = ru
            self.rank[ru] += 1
        return True

    def connected(self, u: str, v: str) -> bool:
        """检查两节点是否在同一连通分量"""
        return self.find(u) == self.find(v)

    @property
    def component_count(self) -> int:
        """获取当前连通分量数量"""
        roots = set(self.find(v) for v in self.parent)
        return len(roots)


# ================================================================
# GraphOrchestrator — 智能编排
# ================================================================

class GraphOrchestrator:
    """
    智能编排模块。

    将全连接通信图通过「图转树」降维为最大带宽生成树，
    再在树上进行 DFS 搜索，找到满足显存约束的最优流水线路径。

    使用方式:
        orchestrator = GraphOrchestrator(nodes, model_memory_mb, total_layers)
        assignments = orchestrator.orchestrate()
        # assignments 可直接作为 compute_layer_assignment() 的返回值
    """

    def __init__(self, nodes: dict, model_memory_mb: float,
                 total_layers: int = 24, quant_factor: float = 1.0):
        """
        Args:
            nodes: {node_id: object with device_info, avg_rtt_ms, network_type, role}
            model_memory_mb: 模型总显存需求 (MB)
            total_layers: 模型 Transformer 层总数
            quant_factor: 量化显存修正系数 (fp16=1.0, int8=0.55, int4=0.35)
        """
        self.nodes = nodes
        self.model_memory_mb = model_memory_mb
        self.total_layers = total_layers
        self.quant_factor = quant_factor

        logger.info(
            f"🧠 GraphOrchestrator 初始化: nodes={len(nodes)}, "
            f"model_memory={model_memory_mb:.0f}MB, "
            f"total_layers={total_layers}, quant_factor={quant_factor}"
        )

    # ================================================================
    # 公开接口
    # ================================================================

    def orchestrate(self) -> List[dict]:
        """
        主入口：执行完整编排流程。

        Returns:
            [{node_id, role, start_layer, end_layer, layers_count,
              has_embedding, has_lm_head, score}]
            按流水线链式拓扑顺序排列
        """
        if len(self.nodes) < 2:
            return self._single_node_assignment()

        # Step 1: 构建全连接通信图
        graph = self._build_graph()
        logger.info(
            f"📊 通信图已构建: {len(graph['vertices'])} 节点, "
            f"{len(graph['edges'])} 条边"
        )

        # Step 2: 最大带宽生成树
        tree = self._max_bandwidth_spanning_tree(graph)
        tree_edges = sum(len(neighbors) for neighbors in tree.values()) // 2
        logger.info(
            f"🌲 最大带宽生成树: {len(tree)} 节点, {tree_edges} 条边"
        )

        # Step 3: DFS 路径搜索
        optimal_path = self._dfs_path_search(tree, graph['vertices'])
        if not optimal_path or len(optimal_path) < 1:
            logger.warning("DFS 未找到可行路径，回退到权重排序")
            return self._fallback_weight_assignment()

        logger.info(
            f"✅ 最优流水线路径: {' → '.join(optimal_path)} "
            f"({len(optimal_path)} 节点)"
        )

        # Step 4: 层分配
        assignments = self._assign_layers(optimal_path, graph['vertices'])

        return assignments

    def get_chain_topology(self) -> List[str]:
        """
        获取推荐链式拓扑顺序（仅节点 ID 列表）。

        用于 run_pipeline() 中的节点排序。
        """
        assignments = self.orchestrate()
        return [a["node_id"] for a in assignments]

    # ================================================================
    # Step 1: 构建全连接通信图
    # ================================================================

    def _build_graph(self) -> dict:
        """
        构建全局异构通信图 G = (V, E)。

        V 属性: vram_mb, compute_weight, compute_delay, role
        E 属性: bandwidth（带宽评分，越高越好）, latency_ms（通信延迟）

        带宽估算优先级:
          1. 有 RTT 测量 → bw = 1000 / (avg_rtt_ms + 1)
          2. 无 RTT → 按 network_type 估算
          3. 节点间带宽: min(bw_u, bw_v) × 0.8（保守估计）
        """
        vertices = {}
        node_list = list(self.nodes.keys())

        for nid in node_list:
            node = self.nodes[nid]
            device_info = self._get_device_info(node)

            vram_mb = self._extract_vram_mb(device_info)
            weight = self._calc_node_weight(device_info)
            compute_delay = max(0.5, (115.0 - weight) / 115.0 * 10.0)

            vertices[nid] = {
                'vram_mb': vram_mb,
                'compute_weight': weight,
                'compute_delay': compute_delay,
                'role': self._get_role(node),
            }

        # 构建全连接边（无向）
        edges = []
        for i, u in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                v = node_list[j]
                bw_u = self._estimate_bandwidth(u)
                bw_v = self._estimate_bandwidth(v)
                bandwidth = min(bw_u, bw_v) * 0.8
                latency = (self._get_node_latency(u) +
                           self._get_node_latency(v))

                edges.append({
                    'u': u,
                    'v': v,
                    'bandwidth': round(bandwidth, 2),
                    'latency_ms': round(latency, 2),
                })

        return {'vertices': vertices, 'edges': edges}

    # ---- 节点属性提取 ----

    @staticmethod
    def _get_device_info(node) -> dict:
        """安全获取节点的 device_info"""
        return getattr(node, 'device_info', {}) or {}

    @staticmethod
    def _get_role(node) -> str:
        """安全获取节点角色"""
        return getattr(node, 'role', 'client') or 'client'

    def _extract_vram_mb(self, device_info: dict) -> float:
        """
        从设备画像提取可用显存 (MB)。

        优先级: GPU 专用显存 > 系统可用 RAM
        """
        if not device_info:
            return 0.0
        gpu = device_info.get('gpu', {})
        ram = device_info.get('ram', {})
        if isinstance(gpu, dict) and gpu.get('vram_total_gb', 0) > 0:
            return gpu['vram_total_gb'] * 1024
        if isinstance(ram, dict):
            return ram.get('available_gb', ram.get('total_gb', 0)) * 1024
        return 0.0

    def _calc_node_weight(self, device_info: dict) -> float:
        """
        计算节点算力权重（满分 100 + 独显奖励 15）。

        与 Scheduler._compute_node_weight() 保持一致:
          - GPU 显存: 50%
          - 系统内存: 30%
          - CPU 核心+频率: 20%
          - 独显奖励: +15
        """
        gpu = device_info.get('gpu', {}) if device_info else {}
        ram = device_info.get('ram', {}) if device_info else {}
        cpu = device_info.get('cpu', {}) if device_info else {}

        # VRAM 得分 (0–50)
        vram_gb = gpu.get('vram_total_gb', 0) if isinstance(gpu, dict) else 0
        vram_score = min(vram_gb / 24.0, 1.0) * 50.0 if vram_gb > 0 else 0

        # RAM 得分 (0–30)
        ram_gb = ram.get('total_gb', 4) if isinstance(ram, dict) else 4
        ram_score = min(ram_gb / 64.0, 1.0) * 30.0

        # CPU 得分 (0–20)
        cpu_cores = cpu.get('physical_cores', 2) if isinstance(cpu, dict) else 2
        cpu_freq = cpu.get('freq_max_mhz', 2000) if isinstance(cpu, dict) else 2000
        core_score = min(cpu_cores / 16.0, 1.0) * 10.0
        freq_score = min(cpu_freq / 4000.0, 1.0) * 10.0

        # 独显奖励 (+15)
        cuda_avail = gpu.get('cuda_available', False)
        is_integrated = gpu.get('is_integrated', True)
        discrete_bonus = 15.0 if (cuda_avail and not is_integrated) else 0.0

        return vram_score + ram_score + core_score + freq_score + discrete_bonus

    def _estimate_bandwidth(self, node_id: str) -> float:
        """
        估算节点的可用带宽评分（越大越好，量纲无关仅用于比较）。

        优先级:
          1. RTT 反推: bw = 1000 / (avg_rtt_ms + 1)
          2. network_type 估算: ethernet=100, wifi=30, unknown=10
        """
        node = self.nodes.get(node_id)
        if not node:
            return 10.0

        rtt = getattr(node, 'avg_rtt_ms', 0.0) or 0.0
        if rtt > 0:
            return 1000.0 / (rtt + 1.0)

        net_type = getattr(node, 'network_type', 'unknown') or 'unknown'
        bw_map = {
            'ethernet': 100.0,
            'wifi': 30.0,
            'localhost': 1000.0,
            'unknown': 10.0,
        }
        return bw_map.get(net_type, 10.0)

    def _get_node_latency(self, node_id: str) -> float:
        """
        获取节点通信延迟 (ms，单向)。

        优先级:
          1. RTT/2
          2. network_type 估算
        """
        node = self.nodes.get(node_id)
        if not node:
            return 50.0

        rtt = getattr(node, 'avg_rtt_ms', 0.0) or 0.0
        if rtt > 0:
            return rtt / 2.0

        net_type = getattr(node, 'network_type', 'unknown') or 'unknown'
        latency_map = {
            'ethernet': 1.0,
            'wifi': 5.0,
            'localhost': 0.1,
            'unknown': 20.0,
        }
        return latency_map.get(net_type, 20.0)

    # ================================================================
    # Step 2: 最大带宽生成树（Kruskal）
    # ================================================================

    def _max_bandwidth_spanning_tree(self, graph: dict
                                     ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        基于 Kruskal 贪心算法构建最大带宽生成树。

        步骤:
          1. 所有边按带宽从大到小排序
          2. 依次遍历，若两节点不在同一连通分量 → 选入生成树
          3. 够 |V|-1 条边后停止

        返回:
            {node_id: [(neighbor_id, bandwidth, latency_ms), ...]}
        """
        vertices = list(graph['vertices'].keys())
        edges = sorted(graph['edges'], key=lambda e: e['bandwidth'], reverse=True)
        uf = UnionFind(vertices)

        tree: Dict[str, List[Tuple[str, float, float]]] = {
            v: [] for v in vertices
        }

        selected = 0
        target = len(vertices) - 1

        for edge in edges:
            u, v = edge['u'], edge['v']
            if uf.union(u, v):
                bw = edge['bandwidth']
                lat = edge['latency_ms']
                tree[u].append((v, bw, lat))
                tree[v].append((u, bw, lat))
                selected += 1
                if selected >= target:
                    break

        if selected < target:
            logger.warning(
                f"生成树不完整: 选中 {selected}/{target} 条边 "
                f"（图可能不连通，已选 {uf.component_count} 个连通分量）"
            )

        return tree

    # ================================================================
    # Step 3: 树上 DFS 路径搜索
    # ================================================================

    def _dfs_path_search(self, tree: Dict[str, List[Tuple[str, float, float]]],
                         vertices: dict) -> List[str]:
        """
        在生成树上执行 DFS，搜索满足显存约束的最小总延迟路径。

        约束:
          - 硬性: 路径上所有节点显存总和 ≥ model_memory_mb
          - 优化: 最小化 总计算延迟 + 总通信延迟

        剪枝策略:
          1. 当前累计延迟 ≥ 已知最优延迟 → 回溯
          2. 剩余未访问节点显存不足以补足缺口 → 回溯

        Returns:
            最优路径节点 ID 列表（按流水线顺序）
        """
        best_path: List[str] = []
        best_delay: float = float('inf')
        best_memory: float = 0.0

        all_nodes = list(tree.keys())
        total_available = sum(
            vertices[n].get('vram_mb', 0) for n in all_nodes
        )

        if total_available < self.model_memory_mb:
            logger.warning(
                f"⚠️ 总显存不足: {total_available:.0f}MB < "
                f"需要 {self.model_memory_mb:.0f}MB，将返回最大覆盖路径"
            )

        def _dfs(current: str, visited: set, path: List[str],
                 mem_sum: float, delay_sum: float):
            nonlocal best_path, best_delay, best_memory

            visited.add(current)
            path.append(current)
            vdata = vertices.get(current, {})
            new_mem = mem_sum + vdata.get('vram_mb', 0)
            new_delay = delay_sum + vdata.get('compute_delay', 5.0)

            # 检查约束 → 更新最优解
            if new_mem >= self.model_memory_mb:
                if new_delay < best_delay:
                    best_delay = new_delay
                    best_path = list(path)
                    best_memory = new_mem
                    logger.debug(
                        f"  发现更优路径: {' → '.join(path)}, "
                        f"延迟={new_delay:.2f}, 显存={new_mem:.0f}MB"
                    )
            elif new_mem > best_memory and best_delay == float('inf'):
                # 无可行解时记录最大显存路径（best-effort）
                best_memory = new_mem
                best_path = list(path)

            # 剪枝 1: 当前延迟已超最优（不可能更优）
            if new_delay >= best_delay:
                visited.discard(current)
                path.pop()
                return

            # 剪枝 2: 剩余显存不足以补足缺口
            remaining = sum(
                vertices[n].get('vram_mb', 0)
                for n in all_nodes if n not in visited
            )
            if new_mem + remaining < self.model_memory_mb:
                visited.discard(current)
                path.pop()
                return

            # 遍历邻接节点
            for neighbor, _bw, lat in tree.get(current, []):
                if neighbor not in visited:
                    _dfs(neighbor, visited, path,
                         new_mem, new_delay + lat)

            # 回溯
            visited.discard(current)
            path.pop()

        # 以每个节点为起点执行 DFS
        for start_node in all_nodes:
            _dfs(start_node, set(), [], 0.0, 0.0)

        if best_delay < float('inf'):
            logger.info(
                f"✅ DFS 搜索完成: {len(best_path)} 节点, "
                f"总延迟={best_delay:.2f}, 总显存={best_memory:.0f}MB"
            )
        else:
            logger.warning(
                f"⚠️ DFS 未找到满足显存约束的路径, "
                f"最佳部分路径: {len(best_path)} 节点, "
                f"显存={best_memory:.0f}MB / 需要={self.model_memory_mb:.0f}MB"
            )

        return best_path

    # ================================================================
    # Step 4: 节点层均衡分配（后处理）
    # ================================================================

    def _assign_layers(self, path: List[str], vertices: dict) -> List[dict]:
        """
        沿最优路径按显存占比均衡分配模型层。

        规则:
          - 首节点含 Embedding 层
          - 末节点含 LM Head 层
          - 各节点按显存占比分配 Transformer 层
          - 最少每节点 1 层，rounding 误差修正

        Returns:
            [{node_id, role, start_layer, end_layer, layers_count,
              has_embedding, has_lm_head, score}]
        """
        if not path:
            return []

        path_nodes = []
        for nid in path:
            node = self.nodes.get(nid)
            vdata = vertices.get(nid, {})
            path_nodes.append({
                'node_id': nid,
                'role': self._get_role(node) if node else 'client',
                'vram_mb': vdata.get('vram_mb', 0),
                'compute_weight': vdata.get('compute_weight', 0),
            })

        total_vram = sum(n['vram_mb'] for n in path_nodes)

        if total_vram <= 0:
            return self._equal_split_assignment(path_nodes)

        # 按显存比例分配
        distributable = self.total_layers
        raw_layers = []
        for n in path_nodes:
            proportion = n['vram_mb'] / total_vram
            raw = max(1, round(proportion * distributable))
            raw_layers.append(raw)

        # 修正 rounding 误差
        diff = distributable - sum(raw_layers)
        if diff > 0:
            sorted_idx = sorted(
                range(len(path_nodes)),
                key=lambda i: path_nodes[i]['vram_mb'],
                reverse=True,
            )
            for i in range(diff):
                raw_layers[sorted_idx[i % len(sorted_idx)]] += 1
        elif diff < 0:
            sorted_idx = sorted(
                range(len(path_nodes)),
                key=lambda i: path_nodes[i]['vram_mb'],
            )
            for _ in range(-diff):
                for idx in sorted_idx:
                    if raw_layers[idx] > 1:
                        raw_layers[idx] -= 1
                        break

        # 构建分配结果
        assignments = []
        cursor = 0
        for i, n in enumerate(path_nodes):
            count = raw_layers[i]
            assignments.append({
                'node_id': n['node_id'],
                'role': n['role'],
                'start_layer': cursor,
                'end_layer': cursor + count,
                'layers_count': count,
                'has_embedding': (i == 0),
                'has_lm_head': (i == len(path_nodes) - 1),
                'score': round(n['compute_weight'], 1),
            })
            cursor += count

        logger.info(
            f"📊 层分配完成 ({len(assignments)} 节点, "
            f"{self.total_layers} 层):"
        )
        for a in assignments:
            logger.info(
                f"  {a['node_id']}: Layer {a['start_layer']}-{a['end_layer']} "
                f"({a['layers_count']}层) embed={a['has_embedding']} "
                f"lm_head={a['has_lm_head']}"
            )

        return assignments

    # ================================================================
    # 回退 / 辅助方法
    # ================================================================

    def _single_node_assignment(self) -> List[dict]:
        """单节点：全部层分配给该节点"""
        node_list = list(self.nodes.values())
        if not node_list:
            return []
        n = node_list[0]
        return [{
            'node_id': getattr(n, 'node_id', 'master'),
            'role': self._get_role(n),
            'start_layer': 0,
            'end_layer': self.total_layers,
            'layers_count': self.total_layers,
            'has_embedding': True,
            'has_lm_head': True,
            'score': 50.0,
        }]

    def _equal_split_assignment(self, path_nodes: List[dict]) -> List[dict]:
        """均分分配（当无法计算显存比例时）"""
        n = len(path_nodes)
        if n == 0:
            return []
        base = self.total_layers // n
        remainder = self.total_layers % n

        assignments = []
        cursor = 0
        for i, pn in enumerate(path_nodes):
            count = base + (1 if i < remainder else 0)
            assignments.append({
                'node_id': pn['node_id'],
                'role': pn.get('role', 'client'),
                'start_layer': cursor,
                'end_layer': cursor + count,
                'layers_count': count,
                'has_embedding': (i == 0),
                'has_lm_head': (i == n - 1),
                'score': round(pn.get('compute_weight', 0), 1),
            })
            cursor += count
        return assignments

    def _fallback_weight_assignment(self) -> List[dict]:
        """
        回退方案：按算力权重降序排列 + 比例分配。

        当图算法无法找到满足显存约束的路径时使用。
        """
        node_list = list(self.nodes.values())
        if not node_list:
            return []

        scored = []
        for n in node_list:
            di = self._get_device_info(n)
            weight = self._calc_node_weight(di)
            scored.append({
                'node_id': getattr(n, 'node_id', '?'),
                'role': self._get_role(n),
                'score': weight,
                'vram_mb': self._extract_vram_mb(di),
            })

        scored.sort(key=lambda x: x['score'], reverse=True)

        # 复用 _assign_layers
        fake_vertices = {
            s['node_id']: {
                'vram_mb': s['vram_mb'],
                'compute_weight': s['score'],
                'compute_delay': max(0.5, (115.0 - s['score']) / 115.0 * 10.0),
                'role': s['role'],
            }
            for s in scored
        }
        return self._assign_layers(
            [s['node_id'] for s in scored], fake_vertices
        )
