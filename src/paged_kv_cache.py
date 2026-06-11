"""
轻量化分页KV缓存模块 — 借鉴操作系统内存分页思想
==================================================
功能职责:
1. 内存页管理、页表映射
2. KV 动态分配、追加写入
3. 会话结束页面回收、清空缓存
4. 适配 Prefill / Decode 双阶段

设计原理:
- 将KV缓存拆分为固定大小的内存页（PAGE_SIZE个Token/页）
- 通过页表映射实现非连续存储，避免显存碎片
- 仅本地节点维护，不跨设备传输
- Prefill 阶段批量写入 prompt KV，Decode 阶段逐 token 追加

核心优化:
- get_all_kv(): O(n) 单趟扫描，按物理页连续区间批量切片，避免逐 token cat
- append_kv_single(): Decode 单 token 快速路径，跳过循环逻辑
- 支持 device / dtype 参数，张量在正确设备上分配

依赖: torch
"""

import logging
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN

logger = logging.getLogger(__name__)


@dataclass
class KVPage:
    """单个KV内存页"""
    page_id: int
    k: torch.Tensor       # shape: [num_heads, page_size, head_dim]
    v: torch.Tensor       # shape: [num_heads, page_size, head_dim]
    used: int = 0         # 当前已使用的 token 槽位数
    is_free: bool = True

    @property
    def remaining(self) -> int:
        """本页剩余可用槽位数"""
        return self.k.shape[1] - self.used

    @property
    def capacity(self) -> int:
        """本页总槽位数"""
        return self.k.shape[1]

    def __repr__(self) -> str:
        return (f"KVPage(id={self.page_id}, used={self.used}/{self.capacity}, "
                f"free={self.is_free}, shape={list(self.k.shape)})")


class PagedKVCache:
    """
    轻量化分页KV缓存管理器

    借鉴操作系统内存分页机制:
    - 固定大小内存页（PAGE_SIZE 个 Token/页）
    - 页表映射（逻辑token位置 → 物理页+偏移）
    - 动态分配、自动回收
    - 适配 Prefill（批量写入）和 Decode（单Token追加）双阶段

    使用示例:
        >>> cache = PagedKVCache(page_size=128, max_pages=256, device="cuda")

        >>> # Prefill 阶段: 批量写入 prompt 的 KV
        >>> cache.append_kv(k_prefill, v_prefill)   # [H, prompt_len, D]

        >>> # Decode 阶段: 逐 token 追加（优化路径）
        >>> cache.append_kv_single(k_new, v_new)     # [H, 1, D]

        >>> # Attention 计算前: 获取完整 KV 序列
        >>> all_k, all_v = cache.get_all_kv()
    """

    def __init__(
        self,
        page_size: int = None,
        max_pages: int = None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """
        初始化分页KV缓存。

        Args:
            page_size: 单页容纳Token数量，默认 config.PAGE_SIZE (128)
            max_pages: 最大内存页数，默认 config.MAX_PAGE_NUM (256)
            device: 张量存储设备，默认 "cuda"（如有）否则 "cpu"
            dtype: 张量数据类型，默认 torch.float16
        """
        self.page_size = page_size or PAGE_SIZE
        self.max_pages = max_pages or MAX_PAGE_NUM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16

        # ---- 页表: 逻辑 token 位置 → 物理页 ID ----
        self.page_table: List[int] = []

        # ---- 页面池 ----
        self.free_pages: List[KVPage] = []
        self.allocated_pages: List[KVPage] = []

        # ---- page_id → KVPage 快速索引 (O(1) 查找) ----
        self._page_index: dict[int, KVPage] = {}

        # ---- 当前活跃页（正在写入）----
        self._current_page: Optional[KVPage] = None

        # ---- 计数器 ----
        self._total_tokens: int = 0
        self._page_counter: int = 0

        # ---- 统计信息 ----
        self._append_call_count: int = 0
        self._total_appended_tokens: int = 0
        self._get_all_kv_call_count: int = 0

        max_tokens = self.page_size * self.max_pages
        logger.info(
            f"PagedKVCache 初始化: page_size={self.page_size}, "
            f"max_pages={self.max_pages}, device={self.device}, dtype={self.dtype}, "
            f"max_tokens={max_tokens} ({max_tokens * 2 * 2 // 1024}K 槽位)"
        )

    @classmethod
    def from_profile(cls, profile: dict, device: str = None,
                     dtype: torch.dtype = None, num_heads: int = 16,
                     head_dim: int = 64) -> "PagedKVCache":
        """
        根据设备画像自动选择 page_size 和 max_pages。

        Args:
            profile: DeviceProfiler.to_dict() 返回的设备画像 dict
            device: 张量设备（默认自动检测）
            dtype: 张量数据类型（默认 float16）
            num_heads: 注意力头数（用于日志）
            head_dim: 单头维度（用于日志）

        Returns:
            自适应大小的 PagedKVCache 实例
        """
        tier = profile.get("tier", "laptop") if profile else "laptop"
        gpu = profile.get("gpu", {}) if profile else {}

        # 按设备档位选择 KV 缓存大小
        tier_config = {
            "workstation":  (128, 512, 4096),   # (page_size, max_pages, max_seq_len)
            "laptop":       (128, 256, 2048),
            "ultrabook":    (64,  128, 1024),
            "edge":         (64,  64,  512),
            "mobile":       (32,  32,  256),
        }
        page_size, max_pages, max_seq = tier_config.get(tier, (128, 256, 2048))

        # 进一步根据实际 VRAM 微调
        vram_gb = gpu.get("vram_total_gb", 0) if gpu else 0
        if vram_gb >= 12:
            max_pages = min(512, int(max_pages * 1.5))
        elif vram_gb >= 6:
            pass  # 保持默认
        elif vram_gb > 0:
            max_pages = max(32, int(max_pages * 0.5))

        logger.info(
            f"PagedKVCache.from_profile: tier={tier} → "
            f"page_size={page_size}, max_pages={max_pages}, max_seq={max_seq}"
        )

        return cls(
            page_size=page_size,
            max_pages=max_pages,
            device=device,
            dtype=dtype,
        )

    # ================================================================
    # 页面管理
    # ================================================================

    def _allocate_page(self, num_heads: int, head_dim: int) -> KVPage:
        """
        分配一个新内存页。优先从空闲池取，否则创建新页。

        Args:
            num_heads: 注意力头数量
            head_dim: 每个头的维度

        Returns:
            可用的 KVPage

        Raises:
            RuntimeError: 达到最大页数限制
        """
        # 优先复用空闲页（零分配开销）
        if self.free_pages:
            page = self.free_pages.pop()
            page.is_free = False
            page.used = 0
            self.allocated_pages.append(page)
            self._page_index[page.page_id] = page
            logger.debug(f"复用空闲页: {page}")
            return page

        # 检查上限
        if len(self.allocated_pages) >= self.max_pages:
            raise RuntimeError(
                f"已达到最大页数限制 ({self.max_pages})，"
                f"当前 {len(self.allocated_pages)} 页已全部占用。"
                f"请增大 config.MAX_PAGE_NUM 或调用 clear() 回收。"
            )

        # 创建新页（在目标设备上分配零张量）
        page = KVPage(
            page_id=self._page_counter,
            k=torch.zeros(
                num_heads, self.page_size, head_dim,
                device=self.device, dtype=self.dtype,
            ),
            v=torch.zeros(
                num_heads, self.page_size, head_dim,
                device=self.device, dtype=self.dtype,
            ),
            is_free=False,
        )
        self._page_counter += 1
        self.allocated_pages.append(page)
        self._page_index[page.page_id] = page
        logger.debug(f"分配新页: {page}")
        return page

    def _get_page_by_id(self, page_id: int) -> KVPage:
        """O(1) 按物理页ID查找"""
        try:
            return self._page_index[page_id]
        except KeyError:
            raise KeyError(f"页面 page_id={page_id} 不存在（已回收或从未分配）")

    # ================================================================
    # KV 写入接口
    # ================================================================

    def append_kv(self, new_k: torch.Tensor, new_v: torch.Tensor) -> int:
        """
        追加新 Token 的 K、V 到缓存（支持批量写入，适配 Prefill 阶段）。

        自动跨页写入：如果新 token 超过当前页剩余空间，自动分配新页。

        Args:
            new_k: 新 Key 张量   shape: [num_heads, num_new_tokens, head_dim]
            new_v: 新 Value 张量  shape: [num_heads, num_new_tokens, head_dim]

        Returns:
            写入后缓存中的总 token 数

        Raises:
            RuntimeError: 超过 max_pages 限制
        """
        num_new = new_k.shape[1]
        if num_new == 0:
            return self._total_tokens

        num_heads, _, head_dim = new_k.shape

        # 确保张量与缓存在同一设备
        target_device = torch.device(self.device)
        if new_k.device != target_device:
            new_k = new_k.to(target_device)
            new_v = new_v.to(target_device)

        tokens_written = 0
        while tokens_written < num_new:
            # 无活跃页或当前页已满 → 分配新页
            if self._current_page is None or self._current_page.remaining == 0:
                self._current_page = self._allocate_page(num_heads, head_dim)

            space = self._current_page.remaining
            to_write = min(space, num_new - tokens_written)

            # 写入当前页
            start = self._current_page.used
            self._current_page.k[:, start:start + to_write, :] = \
                new_k[:, tokens_written:tokens_written + to_write, :]
            self._current_page.v[:, start:start + to_write, :] = \
                new_v[:, tokens_written:tokens_written + to_write, :]
            self._current_page.used += to_write

            # 更新页表：每个新 token 记录其所在物理页
            pid = self._current_page.page_id
            self.page_table.extend([pid] * to_write)

            tokens_written += to_write

        self._total_tokens += num_new
        self._append_call_count += 1
        self._total_appended_tokens += num_new

        return self._total_tokens

    def append_kv_single(self, k: torch.Tensor, v: torch.Tensor) -> int:
        """
        追加单个 Token 的 K、V（优化版，适配 Decode 阶段）。

        相比 append_kv()，跳过批量循环逻辑，减少 Python 开销。
        Decode 阶段每步只生成 1 个 token，这是最高频的调用路径。

        Args:
            k: [num_heads, 1, head_dim] 或 [num_heads, head_dim]
            v: [num_heads, 1, head_dim] 或 [num_heads, head_dim]

        Returns:
            写入后缓存中的总 token 数
        """
        # 统一维度: [num_heads, head_dim] → [num_heads, 1, head_dim]
        if k.dim() == 2:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        num_heads, _, head_dim = k.shape

        # 当前页满 → 分配新页
        if self._current_page is None or self._current_page.remaining == 0:
            self._current_page = self._allocate_page(num_heads, head_dim)

        # 单 token 写入
        pos = self._current_page.used
        self._current_page.k[:, pos:pos + 1, :] = k.to(self.device)
        self._current_page.v[:, pos:pos + 1, :] = v.to(self.device)
        self._current_page.used += 1

        # 页表追加
        self.page_table.append(self._current_page.page_id)
        self._total_tokens += 1

        return self._total_tokens

    # ================================================================
    # KV 读取接口
    # ================================================================

    def get_all_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取全部历史 KV（按逻辑顺序拼接）。

        算法: O(n) 单趟扫描页表，检测物理连续区间后批量切片。
        相比逐 token cat（n 次 kernel launch），仅需 ~page_count 次 cat。

        Returns:
            (all_k, all_v): 按 token 顺序拼接的完整 K、V 张量
                           shape: [num_heads, total_tokens, head_dim]
        """
        if self._total_tokens == 0:
            shape = self._empty_shape()
            return (
                torch.empty(shape, device=self.device, dtype=self.dtype),
                torch.empty(shape, device=self.device, dtype=self.dtype),
            )

        self._get_all_kv_call_count += 1

        # 单趟扫描：追踪每个物理页当前的物理偏移，同时检测连续区间
        page_offset: dict[int, int] = {}   # page_id → 该页已扫描到的物理偏移
        chunks_k: List[torch.Tensor] = []
        chunks_v: List[torch.Tensor] = []

        # 当前连续区间的状态
        run_page: Optional[int] = None     # 区间所属物理页ID
        run_start: int = 0                 # 区间在物理页内的起始偏移
        run_len: int = 0                   # 区间长度（token数）

        for logical_idx in range(self._total_tokens):
            page_id = self.page_table[logical_idx]
            offset = page_offset.get(page_id, 0)
            page_offset[page_id] = offset + 1

            # 检查是否能延长当前连续区间
            # 条件：同一物理页 + 物理偏移连续
            if run_page == page_id and offset == run_start + run_len:
                run_len += 1
            else:
                # 提交上一个区间
                if run_page is not None and run_len > 0:
                    page = self._get_page_by_id(run_page)
                    chunks_k.append(page.k[:, run_start:run_start + run_len, :])
                    chunks_v.append(page.v[:, run_start:run_start + run_len, :])
                # 开始新区间
                run_page = page_id
                run_start = offset
                run_len = 1

        # 提交最后一个区间
        if run_page is not None and run_len > 0:
            page = self._get_page_by_id(run_page)
            chunks_k.append(page.k[:, run_start:run_start + run_len, :])
            chunks_v.append(page.v[:, run_start:run_start + run_len, :])

        all_k = torch.cat(chunks_k, dim=1)
        all_v = torch.cat(chunks_v, dim=1)
        return all_k, all_v

    def get_kv_window(self, last_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取最近 N 个 token 的 KV（滑动窗口注意力）。

        适用于 window attention 或只需最近上下文的场景，
        避免读出全部历史 KV 后再截断。

        Args:
            last_n: 需要最近多少个 token

        Returns:
            (k_window, v_window): [num_heads, min(total_tokens, last_n), head_dim]
        """
        if last_n <= 0:
            shape = self._empty_shape()
            return (
                torch.empty(shape, device=self.device, dtype=self.dtype),
                torch.empty(shape, device=self.device, dtype=self.dtype),
            )
        if last_n >= self._total_tokens:
            return self.get_all_kv()

        start_idx = self._total_tokens - last_n

        # 先扫描前缀 [0, start_idx) 以确定窗口起始时的物理偏移
        page_offset: dict[int, int] = {}
        for i in range(start_idx):
            pid = self.page_table[i]
            page_offset[pid] = page_offset.get(pid, 0) + 1

        # 扫描窗口 [start_idx, total_tokens)，检测连续区间
        chunks_k: List[torch.Tensor] = []
        chunks_v: List[torch.Tensor] = []
        run_page, run_start, run_len = None, 0, 0

        for logical_idx in range(start_idx, self._total_tokens):
            page_id = self.page_table[logical_idx]
            offset = page_offset.get(page_id, 0)
            page_offset[page_id] = offset + 1

            if run_page == page_id and offset == run_start + run_len:
                run_len += 1
            else:
                if run_page is not None and run_len > 0:
                    page = self._get_page_by_id(run_page)
                    chunks_k.append(page.k[:, run_start:run_start + run_len, :])
                    chunks_v.append(page.v[:, run_start:run_start + run_len, :])
                run_page = page_id
                run_start = offset
                run_len = 1

        if run_page is not None and run_len > 0:
            page = self._get_page_by_id(run_page)
            chunks_k.append(page.k[:, run_start:run_start + run_len, :])
            chunks_v.append(page.v[:, run_start:run_start + run_len, :])

        return torch.cat(chunks_k, dim=1), torch.cat(chunks_v, dim=1)

    # ================================================================
    # 缓存管理
    # ================================================================

    def clear(self) -> None:
        """清空当前会话缓存，回收所有页面至空闲池（零释放，可复用）"""
        for page in self.allocated_pages:
            page.used = 0
            page.is_free = True
            self.free_pages.append(page)

        self.allocated_pages.clear()
        self._page_index.clear()
        self.page_table.clear()
        self._current_page = None
        self._total_tokens = 0

        logger.info(
            f"KV缓存已清空 — {len(self.free_pages)} 个页面已回收至空闲池 "
            f"（下次 append 将零分配复用）"
        )

    def to(self, device: str) -> "PagedKVCache":
        """将所有已分配页面和空闲页面移动到指定设备"""
        self.device = device
        for page in self.allocated_pages:
            page.k = page.k.to(device)
            page.v = page.v.to(device)
        for page in self.free_pages:
            page.k = page.k.to(device)
            page.v = page.v.to(device)
        logger.info(f"所有 KV 页面已迁移至设备: {device}")
        return self

    def _empty_shape(self) -> tuple:
        """
        返回空张量的正确形状 [num_heads, 0, head_dim]。
        优先从已分配页推断，无页时回退到 (0,)。
        """
        if self.allocated_pages:
            p = self.allocated_pages[0]
            return (p.k.shape[0], 0, p.k.shape[2])
        if self._current_page is not None:
            return (self._current_page.k.shape[0], 0, self._current_page.k.shape[2])
        return (0,)

    # ================================================================
    # 属性与统计
    # ================================================================

    @property
    def total_tokens(self) -> int:
        """缓存中当前的 token 总数"""
        return self._total_tokens

    @property
    def allocated_page_count(self) -> int:
        """已分配的物理页数"""
        return len(self.allocated_pages)

    @property
    def free_page_count(self) -> int:
        """空闲池中的页数"""
        return len(self.free_pages)

    def get_stats(self) -> dict:
        """
        获取缓存统计信息，用于性能监控和可视化。

        Returns:
            {
                "total_tokens": 当前 token 数,
                "max_tokens": 理论最大容量,
                "allocated_pages": 已分配页数,
                "free_pages": 空闲池页数,
                "max_pages": 最大页数限制,
                "page_size": 每页 token 容量,
                "utilization": 总容量利用率,
                "page_utilization": 已分配页的填充率,
                "estimated_memory_mb": 估算显存占用,
                "append_call_count": append 调用次数,
                "total_appended_tokens": 累计写入 token 数,
                "get_all_kv_call_count": get_all_kv 调用次数,
            }
        """
        total_slots = self.max_pages * self.page_size
        used_slots = self._total_tokens

        # 估算显存占用
        mem_bytes = 0
        if self.allocated_pages:
            p = self.allocated_pages[0]
            mem_bytes = (p.k.numel() + p.v.numel()) * p.k.element_size()

        return {
            "total_tokens": self._total_tokens,
            "max_tokens": total_slots,
            "allocated_pages": len(self.allocated_pages),
            "free_pages": len(self.free_pages),
            "max_pages": self.max_pages,
            "page_size": self.page_size,
            "utilization": round(used_slots / total_slots, 4) if total_slots > 0 else 0.0,
            "page_utilization": (
                round(used_slots / (len(self.allocated_pages) * self.page_size), 4)
                if self.allocated_pages else 0.0
            ),
            "estimated_memory_mb": round(mem_bytes * len(self.allocated_pages) / (1024 ** 2), 2),
            "append_call_count": self._append_call_count,
            "total_appended_tokens": self._total_appended_tokens,
            "get_all_kv_call_count": self._get_all_kv_call_count,
        }
