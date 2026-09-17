"""
轻量化分页KV缓存模块 — 借鉴操作系统内存分页思想
==================================================
功能职责:
1. 内存页管理、页表映射
2. KV 动态分配、追加写入
3. 尾部回滚、会话结束页面回收、清空缓存
4. 适配 Prefill / Decode 双阶段

设计原理:
- 将KV缓存拆分为固定大小的内存页（PAGE_SIZE个Token/页）
- 通过页表映射实现非连续存储，避免显存碎片
- 仅本地节点维护，不跨设备传输
- Prefill 阶段批量写入 prompt KV，Decode 阶段逐 token 追加

核心优化:
- get_all_kv(): O(n) 单趟扫描，按物理页连续区间批量切片，避免逐 token cat
- append_kv_single(): Decode 单 token 快速路径，跳过循环逻辑
- truncate(n): 回滚尾部 n 个 token，回收完整空页并保留尾页复用
- 支持 device / dtype 参数，张量在正确设备上分配

依赖: torch（仅在实际创建/读写张量时延迟导入）
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Mapping, Tuple, Optional



class _LazyTorch:
    """Resolve torch only when the D-tier cache actually touches a tensor."""

    def __getattr__(self, name: str):
        return getattr(importlib.import_module("torch"), name)


torch = _LazyTorch()

from config import PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN
from cache_unit_layout import CacheUnitLayout

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


@dataclass(frozen=True)
class ColdPageRecord:
    """Integrity metadata for one page stored in the disk cold tier."""

    page_id: int
    path: Path
    used: int
    sha256: str
    bytes_count: int
    shape: tuple[int, ...]


def _synchronized(method):
    """Serialize cache operations while allowing nested calls through RLock."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


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
        cold_cache_dir: str | os.PathLike[str] | None = None,
        cold_max_pages: int | None = None,
        cache_unit_size: int | None = None,
        cold_bytes_reserve: Callable[[int], None] | None = None,
        cold_bytes_commit: Callable[[int], None] | None = None,
        cold_bytes_release: Callable[[int], None] | None = None,
        cold_bytes_release_reservation: Callable[[int], None] | None = None,
    ):
        """
        初始化分页KV缓存。

        Args:
            page_size: 单页容纳Token数量，默认 config.PAGE_SIZE (128)
            max_pages: 最大内存页数，默认 config.MAX_PAGE_NUM (256)
            device: 张量存储设备，默认 "cuda"（如有）否则 "cpu"
            dtype: 张量数据类型，默认 torch.float16
            cold_cache_dir: 可选的磁盘冷层根目录；启用后每个缓存实例使用独立子目录
            cold_max_pages: 磁盘冷层最大页数；省略时使用 max_pages
            cache_unit_size: 固定缓存单元大小；必须整除 page_size，默认等于 page_size
        """
        self.page_size = page_size or PAGE_SIZE
        self.max_pages = max_pages or MAX_PAGE_NUM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        self.cache_unit_layout = CacheUnitLayout(
            page_size=self.page_size,
            unit_size=self.page_size if cache_unit_size is None else cache_unit_size,
        )
        if cold_max_pages is not None and (
            isinstance(cold_max_pages, bool) or not isinstance(cold_max_pages, int)
        ):
            raise TypeError("cold_max_pages must be an integer")
        if cold_max_pages is not None and cold_max_pages < 0:
            raise ValueError("cold_max_pages cannot be negative")
        if cold_cache_dir is None and cold_max_pages not in (None, 0):
            raise ValueError("cold_cache_dir is required when cold_max_pages is enabled")
        self.cold_max_pages = (
            (self.max_pages if cold_max_pages is None else cold_max_pages)
            if cold_cache_dir is not None else 0
        )
        self._lock = threading.RLock()
        self.cold_cache_dir: Path | None = None
        if cold_cache_dir is not None and self.cold_max_pages > 0:
            root = Path(cold_cache_dir).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            # A cache instance owns one isolated directory; stale files from a
            # previous process can never collide with its page ids.
            self.cold_cache_dir = root / f"session-{uuid.uuid4().hex}"
            self.cold_cache_dir.mkdir()
        elif cold_cache_dir is not None:
            raise ValueError("cold_max_pages must be positive when cold_cache_dir is provided")

        self._cold_pages: dict[int, ColdPageRecord] = {}
        self._page_order: list[int] = []
        self._cold_hit_count = 0
        self._cold_miss_count = 0
        self._cold_eviction_count = 0
        self._cold_read_last_ms = 0.0
        self._cold_read_max_ms = 0.0
        self._resident_peak_pages = 0
        self._cold_bytes_reserve = cold_bytes_reserve
        self._cold_bytes_commit = cold_bytes_commit
        self._cold_bytes_release = cold_bytes_release
        self._cold_bytes_release_reservation = cold_bytes_release_reservation

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
        self._truncate_call_count: int = 0
        self._total_truncated_tokens: int = 0

        max_tokens = self.page_size * self._logical_max_pages
        logger.info(
            f"PagedKVCache 初始化: page_size={self.page_size}, "
            f"max_pages={self.max_pages}, device={self.device}, dtype={self.dtype}, "
            f"max_tokens={max_tokens} ({max_tokens * 2 * 2 // 1024}K 槽位)"
        )

    @classmethod
    def from_profile(cls, profile: dict, device: str = None,
                     dtype: torch.dtype = None, num_heads: int = 16,
                     head_dim: int = 64,
                     cold_cache_dir: str | os.PathLike[str] | None = None,
                     cold_max_pages: int | None = None,
                     cache_unit_size: int | None = None,
                     cold_bytes_reserve: Callable[[int], None] | None = None,
                     cold_bytes_commit: Callable[[int], None] | None = None,
                     cold_bytes_release: Callable[[int], None] | None = None,
                     cold_bytes_release_reservation: Callable[[int], None] | None = None) -> "PagedKVCache":
        """
        根据设备画像自动选择 page_size 和 max_pages。

        Args:
            profile: DeviceProfiler.to_dict() 返回的设备画像 dict
            device: 张量设备（默认自动检测）
            dtype: 张量数据类型（默认 float16）
            num_heads: 注意力头数（用于日志）
            head_dim: 单头维度（用于日志）
            cold_cache_dir: 可选的磁盘冷层根目录
            cold_max_pages: 磁盘冷层最大页数；省略时使用 max_pages
            cache_unit_size: 固定缓存单元大小；必须整除 page_size

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
            cold_cache_dir=cold_cache_dir,
            cold_max_pages=cold_max_pages,
            cache_unit_size=cache_unit_size,
            cold_bytes_reserve=cold_bytes_reserve,
            cold_bytes_commit=cold_bytes_commit,
            cold_bytes_release=cold_bytes_release,
            cold_bytes_release_reservation=cold_bytes_release_reservation,
        )

    @property
    def _cold_enabled(self) -> bool:
        return self.cold_cache_dir is not None and self.cold_max_pages > 0

    @property
    def _logical_max_pages(self) -> int:
        return self.max_pages + self.cold_max_pages

    def _write_cold_manifest(self) -> None:
        if not self._cold_enabled:
            return
        manifest = {
            "schema": "qlh.paged_kv_cold.v1",
            "records": {
                str(page_id): {
                    "file": record.path.name,
                    "used": record.used,
                    "sha256": record.sha256,
                    "bytes": record.bytes_count,
                    "shape": list(record.shape),
                }
                for page_id, record in sorted(self._cold_pages.items())
            },
        }
        target = self.cold_cache_dir / "manifest.json"
        temporary = self.cold_cache_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _store_cold_page(self, page: KVPage) -> None:
        if not self._cold_enabled:
            raise RuntimeError("disk cold tier is not enabled")
        if len(self._cold_pages) >= self.cold_max_pages:
            raise RuntimeError(
                f"disk cold tier is full ({self.cold_max_pages} pages); "
                "truncate or clear the cache before appending"
            )
        final_path = self.cold_cache_dir / f"page-{page.page_id}.pt"
        temporary = self.cold_cache_dir / f".page-{page.page_id}-{uuid.uuid4().hex}.tmp"
        reserved_bytes = 0
        committed = False
        payload = {
            "schema": "qlh.paged_kv_page.v1",
            "page_id": page.page_id,
            "used": page.used,
            "k": page.k.detach().to("cpu").contiguous(),
            "v": page.v.detach().to("cpu").contiguous(),
        }
        try:
            torch.save(payload, temporary)
            bytes_count = temporary.stat().st_size
            if self._cold_bytes_reserve is not None:
                self._cold_bytes_reserve(bytes_count)
                reserved_bytes = bytes_count
            os.replace(temporary, final_path)
            record = ColdPageRecord(
                page_id=page.page_id,
                path=final_path,
                used=page.used,
                sha256=self._file_sha256(final_path),
                bytes_count=bytes_count,
                shape=tuple(int(value) for value in page.k.shape),
            )
            self._cold_pages[page.page_id] = record
            try:
                self._write_cold_manifest()
            except Exception:
                self._cold_pages.pop(page.page_id, None)
                final_path.unlink(missing_ok=True)
                raise
            if self._cold_bytes_commit is not None:
                self._cold_bytes_commit(bytes_count)
            committed = True
        except Exception:
            if not committed:
                self._cold_pages.pop(page.page_id, None)
                final_path.unlink(missing_ok=True)
            if (
                reserved_bytes
                and not committed
                and self._cold_bytes_release_reservation is not None
            ):
                self._cold_bytes_release_reservation(reserved_bytes)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def _load_cold_page(self, page_id: int) -> KVPage:
        record = self._cold_pages.get(page_id)
        if record is None:
            raise KeyError(f"page_id={page_id} is not resident or stored in the cold tier")
        started = time.perf_counter()
        try:
            if not record.path.is_file():
                raise RuntimeError("cold page file is missing")
            if self._file_sha256(record.path) != record.sha256:
                raise RuntimeError("cold page SHA-256 mismatch")
            payload = torch.load(record.path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("schema") != "qlh.paged_kv_page.v1":
                raise RuntimeError("cold page schema is invalid")
            if payload.get("page_id") != page_id or payload.get("used") != record.used:
                raise RuntimeError("cold page metadata is inconsistent")
            k = payload.get("k")
            v = payload.get("v")
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise RuntimeError("cold page tensors are missing")
            if tuple(int(value) for value in k.shape) != record.shape or tuple(v.shape) != record.shape:
                raise RuntimeError("cold page tensor shape is invalid")
            if not 0 <= record.used <= k.shape[1]:
                raise RuntimeError("cold page used count is invalid")
            page = KVPage(
                page_id=page_id,
                k=k.to(self.device),
                v=v.to(self.device),
                used=record.used,
                is_free=False,
            )
        except Exception as exc:
            self._cold_miss_count += 1
            raise RuntimeError(f"cannot read cold page {page_id}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._cold_hit_count += 1
        self._cold_read_last_ms = elapsed_ms
        self._cold_read_max_ms = max(self._cold_read_max_ms, elapsed_ms)
        return page

    def _spill_oldest(self, *, exclude_page_id: int | None = None) -> KVPage:
        if not self._cold_enabled:
            raise RuntimeError("disk cold tier is not enabled")
        current_id = (
            self._current_page.page_id
            if self._current_page is not None and self._current_page.remaining > 0
            else None
        )
        candidates = [
            page
            for page in self.allocated_pages
            if page.page_id != exclude_page_id and page.page_id != current_id
        ]
        if not candidates:
            raise RuntimeError("no sealed page is available for cold-tier eviction")
        order = {page_id: index for index, page_id in enumerate(self._page_order)}
        page = min(candidates, key=lambda item: order.get(item.page_id, len(order)))
        self._store_cold_page(page)
        self.allocated_pages[:] = [item for item in self.allocated_pages if item is not page]
        self._page_index.pop(page.page_id, None)
        self._cold_eviction_count += 1
        return page

    def _hydrate_page(self, page_id: int) -> KVPage:
        resident = self._page_index.get(page_id)
        if resident is not None:
            return resident
        if len(self.allocated_pages) >= self.max_pages and not self.free_pages:
            self._spill_oldest(exclude_page_id=page_id)
        page = self._load_cold_page(page_id)
        record = self._cold_pages.pop(page_id)
        record.path.unlink(missing_ok=True)
        if self._cold_bytes_release is not None:
            self._cold_bytes_release(record.bytes_count)
        if self.free_pages:
            self.free_pages.pop()
            page = KVPage(page_id=page_id, k=page.k, v=page.v, used=page.used, is_free=False)
        self.allocated_pages.append(page)
        self._page_index[page_id] = page
        self._write_cold_manifest()
        return page

    def _delete_cold_page(self, page_id: int) -> None:
        record = self._cold_pages.get(page_id)
        if record is None:
            return
        record.path.unlink(missing_ok=True)
        self._cold_pages.pop(page_id, None)
        try:
            self._write_cold_manifest()
        except Exception:
            self._cold_pages[page_id] = record
            raise
        if self._cold_bytes_release is not None:
            self._cold_bytes_release(record.bytes_count)

    def _page_for_read(self, page_id: int) -> KVPage:
        page = self._page_index.get(page_id)
        return page if page is not None else self._load_cold_page(page_id)

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
            if self._cold_enabled:
                self._page_order.append(page.page_id)
            self._resident_peak_pages = max(self._resident_peak_pages, len(self.allocated_pages))
            logger.debug(f"复用空闲页: {page}")
            return page

        # 检查上限
        if self._cold_enabled and len(self.allocated_pages) >= self.max_pages:
            self._spill_oldest()

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
        if self._cold_enabled:
            self._page_order.append(page.page_id)
        self._resident_peak_pages = max(self._resident_peak_pages, len(self.allocated_pages))
        logger.debug(f"分配新页: {page}")
        return page

    def _get_page_by_id(self, page_id: int) -> KVPage:
        """O(1) 按物理页ID查找"""
        try:
            return self._page_index[page_id]
        except KeyError:
            if self._cold_enabled and page_id in self._cold_pages:
                return self._page_for_read(page_id)
            raise KeyError(f"页面 page_id={page_id} 不存在（已回收或从未分配）")

    # ================================================================
    # KV 写入接口
    # ================================================================

    @_synchronized
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

        max_tokens = self.page_size * self._logical_max_pages
        if self._total_tokens + num_new > max_tokens:
            raise RuntimeError(
                f"追加 {num_new} 个 token 将超过缓存容量 {max_tokens} "
                f"（当前 {self._total_tokens}）。请增大 config.MAX_PAGE_NUM "
                f"或先调用 truncate()/clear() 回收。"
            )

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

    @_synchronized
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
        max_tokens = self.page_size * self._logical_max_pages
        if self._total_tokens >= max_tokens:
            raise RuntimeError(
                f"追加 1 个 token 将超过缓存容量 {max_tokens} "
                f"（当前 {self._total_tokens}）。请先调用 truncate()/clear() 回收。"
            )

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
        self._resident_peak_pages = max(self._resident_peak_pages, len(self.allocated_pages))

        return self._total_tokens

    # ================================================================
    # KV 读取接口
    # ================================================================

    @_synchronized
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

    @_synchronized
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

    def _release_resident_page(self, page: KVPage) -> None:
        self._page_index.pop(page.page_id, None)
        self.allocated_pages[:] = [item for item in self.allocated_pages if item is not page]
        page.used = 0
        page.is_free = True
        self.free_pages.append(page)

    def _truncate_cold(self, num_tokens: int) -> int:
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens 必须是整数")
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")
        if num_tokens > self._total_tokens:
            raise ValueError(
                f"无法回滚 {num_tokens} 个 token：缓存当前只有 {self._total_tokens} 个 token"
            )
        if num_tokens == 0:
            return self._total_tokens

        del self.page_table[-num_tokens:]
        remaining = num_tokens
        while remaining > 0:
            if not self._page_order:
                raise RuntimeError("KV 缓存内部状态损坏：冷层页序列为空")
            page_id = self._page_order[-1]
            page = self._page_index.get(page_id)
            cold_record = self._cold_pages.get(page_id)
            if page is None and cold_record is not None and remaining >= cold_record.used:
                remaining -= cold_record.used
                self._page_order.pop()
                self._delete_cold_page(page_id)
                self._current_page = None
                continue
            if page is None:
                page = self._hydrate_page(page_id)
            if remaining < page.used:
                page.used -= remaining
                remaining = 0
                self._current_page = page
                break
            remaining -= page.used
            self._page_order.pop()
            if page_id in self._cold_pages:
                self._delete_cold_page(page_id)
            else:
                self._release_resident_page(page)
            self._current_page = None

        if self._page_order and self._current_page is None:
            self._current_page = self._hydrate_page(self._page_order[-1])
        self._total_tokens -= num_tokens
        self._truncate_call_count += 1
        self._total_truncated_tokens += num_tokens
        return self._total_tokens

    @_synchronized
    def truncate(self, num_tokens: int) -> int:
        """
        从缓存尾部回滚指定数量的 token。

        完整移除的尾页会进入空闲池；仍保留部分 token 的尾页会同步回退
        ``used``，并成为下一次 append 的当前写页。已回滚位置的张量无需
        清零，因为 ``used`` 和页表共同限定可读范围，后续写入会覆盖它们。

        Args:
            num_tokens: 要从尾部移除的 token 数，范围为
                        ``0 <= num_tokens <= total_tokens``。

        Returns:
            回滚后缓存中的总 token 数。

        Raises:
            TypeError: ``num_tokens`` 不是整数或是 bool。
            ValueError: 数量为负数或超过当前 token 数。
        """
        if self._cold_enabled:
            return self._truncate_cold(num_tokens)
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
            raise TypeError("num_tokens 必须是整数")
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")
        if num_tokens > self._total_tokens:
            raise ValueError(
                f"无法回滚 {num_tokens} 个 token：缓存当前只有 "
                f"{self._total_tokens} 个 token"
            )
        if num_tokens == 0:
            return self._total_tokens

        remaining = num_tokens
        del self.page_table[-num_tokens:]

        while remaining > 0:
            page = self._current_page
            if page is None or not self.allocated_pages:
                raise RuntimeError("KV 缓存内部状态损坏：页表与已分配页面不一致")

            if remaining < page.used:
                page.used -= remaining
                remaining = 0
                break

            remaining -= page.used
            released = self.allocated_pages.pop()
            if released is not page:
                raise RuntimeError("KV 缓存内部状态损坏：当前页不是分配序列尾页")
            self._page_index.pop(page.page_id, None)
            page.used = 0
            page.is_free = True
            self.free_pages.append(page)
            self._current_page = (
                self.allocated_pages[-1] if self.allocated_pages else None
            )

        self._total_tokens -= num_tokens
        self._truncate_call_count += 1
        self._total_truncated_tokens += num_tokens
        return self._total_tokens

    @_synchronized
    def clear(self) -> None:
        """清空当前会话缓存，回收所有页面至空闲池（零释放，可复用）"""
        if self._cold_enabled:
            for page_id in list(self._cold_pages):
                self._delete_cold_page(page_id)
            self._page_order.clear()

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

    @_synchronized
    def close(self) -> None:
        """清空并删除当前缓存实例自有的冷层会话目录。"""
        self.clear()
        if self.cold_cache_dir is None:
            return
        try:
            shutil.rmtree(self.cold_cache_dir)
        except FileNotFoundError:
            pass

    @_synchronized
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
        if self._cold_pages:
            shape = next(iter(self._cold_pages.values())).shape
            return (shape[0], 0, shape[2])
        if self.free_pages:
            p = self.free_pages[0]
            return (p.k.shape[0], 0, p.k.shape[2])
        return (0,)

    @_synchronized
    def cache_unit_boundaries(self, total_tokens: int | None = None) -> tuple[int, ...]:
        """Return complete fixed-token cache-unit boundaries for inspection."""
        target = self._total_tokens if total_tokens is None else total_tokens
        return self.cache_unit_layout.unit_boundaries(target)

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

    @_synchronized
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
                "cache_unit_size": 固定缓存单元大小,
                "cache_unit_count": 已完成缓存单元数,
                "cache_unit_remainder": 未形成完整单元的尾部 token 数,
                "utilization": 总容量利用率,
                "page_utilization": 逻辑页的填充率,
                "estimated_memory_mb": 估算显存占用,
                "cold_enabled": 是否启用磁盘冷层,
                "cold_pages": 冷层页数,
                "cold_hit_count": 冷层成功读取次数,
                "cold_miss_count": 冷层读取失败次数,
                "cold_read_max_ms": 观测到的最大冷读耗时,
                "append_call_count": append 调用次数,
                "total_appended_tokens": 累计写入 token 数,
                "get_all_kv_call_count": get_all_kv 调用次数,
                "truncate_call_count": truncate 调用次数,
                "total_truncated_tokens": 累计回滚 token 数,
            }
        """
        total_slots = self._logical_max_pages * self.page_size
        used_slots = self._total_tokens
        logical_pages = len(self.allocated_pages) + len(self._cold_pages)
        complete_cache_tokens = self.cache_unit_layout.complete_token_count(used_slots)

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
            "logical_max_pages": self._logical_max_pages,
            "page_size": self.page_size,
            "cache_unit_size": self.cache_unit_layout.unit_size,
            "cache_units_per_page": self.cache_unit_layout.units_per_page,
            "cache_unit_count": complete_cache_tokens // self.cache_unit_layout.unit_size,
            "cache_unit_remainder": used_slots - complete_cache_tokens,
            "cold_enabled": self._cold_enabled,
            "cold_pages": len(self._cold_pages),
            "cold_max_pages": self.cold_max_pages,
            "cold_hit_count": self._cold_hit_count,
            "cold_miss_count": self._cold_miss_count,
            "cold_eviction_count": self._cold_eviction_count,
            "cold_bytes": sum(record.bytes_count for record in self._cold_pages.values()),
            "resident_peak_pages": self._resident_peak_pages,
            "cold_read_last_ms": round(self._cold_read_last_ms, 3),
            "cold_read_max_ms": round(self._cold_read_max_ms, 3),
            "utilization": round(used_slots / total_slots, 4) if total_slots > 0 else 0.0,
            "page_utilization": (
                round(used_slots / (logical_pages * self.page_size), 4)
                if logical_pages else 0.0
            ),
            "estimated_memory_mb": round(mem_bytes * len(self.allocated_pages) / (1024 ** 2), 2),
            "append_call_count": self._append_call_count,
            "total_appended_tokens": self._total_appended_tokens,
            "get_all_kv_call_count": self._get_all_kv_call_count,
            "truncate_call_count": self._truncate_call_count,
            "total_truncated_tokens": self._total_truncated_tokens,
        }
