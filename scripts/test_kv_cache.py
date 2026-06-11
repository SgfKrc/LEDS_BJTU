"""
分页KV缓存模块 — 单元测试与性能基准
======================================
用法:
    python scripts/test_kv_cache.py              # 完整测试
    python scripts/test_kv_cache.py --quick       # 快速冒烟测试
    python scripts/test_kv_cache.py --benchmark   # 仅性能基准
"""

import argparse
import logging
import os
import sys
import time
import math

import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from paged_kv_cache import PagedKVCache, KVPage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- 测试参数 ----
NUM_HEADS = 16
HEAD_DIM = 64
PAGE_SIZE = 128
MAX_PAGES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PASS, FAIL = 0, 0


def check(condition: bool, name: str, detail: str = "") -> bool:
    """断言并计数"""
    global PASS, FAIL
    if condition:
        PASS += 1
        logger.info(f"  ✅ {name}")
    else:
        FAIL += 1
        logger.error(f"  ❌ {name}  — {detail}")
    return condition


def random_kv(tokens: int, heads: int = NUM_HEADS, dim: int = HEAD_DIM):
    """生成随机 K、V 张量"""
    k = torch.randn(heads, tokens, dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(heads, tokens, dim, device=DEVICE, dtype=torch.float16)
    return k, v


# ================================================================
# 1. 基础功能测试
# ================================================================

def test_basic_append():
    """基础: 单页内追加 + 读取"""
    logger.info("--- 1.1 基础追加 ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)

    k, v = random_kv(50)
    cache.append_kv(k, v)
    check(cache.total_tokens == 50, "token数=50", f"got {cache.total_tokens}")
    check(cache.allocated_page_count == 1, "分配1页", f"got {cache.allocated_page_count}")

    all_k, all_v = cache.get_all_kv()
    check(all_k.shape == (NUM_HEADS, 50, HEAD_DIM), f"K shape={list(all_k.shape)}")
    check(all_v.shape == (NUM_HEADS, 50, HEAD_DIM), f"V shape={list(all_v.shape)}")
    check(torch.allclose(all_k, k, atol=1e-5), "K值完整一致")
    check(torch.allclose(all_v, v, atol=1e-5), "V值完整一致")

    cache.clear()
    check(cache.total_tokens == 0, "clear后token=0")
    check(cache.free_page_count == 1, "页面已回收至空闲池")


def test_multi_page_span():
    """跨页写入: token跨越多个物理页"""
    logger.info("--- 1.2 跨页写入 ---")
    cache = PagedKVCache(page_size=16, max_pages=MAX_PAGES, device=DEVICE)

    total = 50  # 需 ceil(50/16)=4 页
    k, v = random_kv(total)
    cache.append_kv(k, v)

    check(cache.total_tokens == total, f"token数={total}")
    check(cache.allocated_page_count == 4, f"分配4页（实际{cache.allocated_page_count}）")

    all_k, all_v = cache.get_all_kv()
    check(torch.allclose(all_k, k, atol=1e-5), "跨页K值完整")
    check(torch.allclose(all_v, v, atol=1e-5), "跨页V值完整")

    # 验证页表长度
    check(len(cache.page_table) == total, f"页表长度={total}")

    cache.clear()


def test_page_reuse_after_clear():
    """页面复用: clear 后 append 应复用旧页"""
    logger.info("--- 1.3 页面复用 ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)

    k1, v1 = random_kv(30)
    cache.append_kv(k1, v1)
    pages_before = cache.allocated_page_count
    cache.clear()

    k2, v2 = random_kv(30)
    cache.append_kv(k2, v2)
    pages_after = cache.allocated_page_count

    check(pages_after == pages_before, f"复用后仍为{pages_before}页")
    check(cache.free_page_count == 0, "空闲池已用尽")
    check(cache.total_tokens == 30, "token数正确")

    all_k, _ = cache.get_all_kv()
    check(torch.allclose(all_k, k2, atol=1e-5), "复用后数据正确")

    cache.clear()


def test_append_kv_single():
    """Decode 优化路径: 逐 token 追加"""
    logger.info("--- 1.4 单token追加 (decode路径) ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)

    k_full, v_full = random_kv(100)

    # 逐 token 追加
    for i in range(100):
        k_i = k_full[:, i:i+1, :]
        v_i = v_full[:, i:i+1, :]
        cache.append_kv_single(k_i, v_i)

    check(cache.total_tokens == 100, "token数=100")

    # 2D 输入（无 seq 维度）
    k_2d = torch.randn(NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v_2d = torch.randn(NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    cache.append_kv_single(k_2d, v_2d)
    check(cache.total_tokens == 101, "2D输入自动unsqueeze")

    all_k, all_v = cache.get_all_kv()
    check(all_k.shape[1] == 101, "总序列长度=101")
    # 验证前100个token
    check(torch.allclose(all_k[:, :100, :], k_full, atol=1e-5), "逐token写入数据正确")

    cache.clear()


def test_get_kv_window():
    """滑动窗口: 获取最近N个token"""
    logger.info("--- 1.5 滑动窗口 ---")
    cache = PagedKVCache(page_size=16, max_pages=MAX_PAGES, device=DEVICE)

    k, v = random_kv(80)
    cache.append_kv(k, v)

    # 正常窗口
    wk, wv = cache.get_kv_window(20)
    check(wk.shape[1] == 20, "窗口大小=20")
    check(torch.allclose(wk, k[:, -20:, :], atol=1e-5), "窗口数据=最后20个")

    # 窗口 > 总量 → 返回全部
    wk, wv = cache.get_kv_window(200)
    check(wk.shape[1] == 80, "窗口>总量返回全部")

    # 窗口 = 0 → 空张量（形状可能是 (0,) 或 [H, 0, D]）
    wk, wv = cache.get_kv_window(0)
    check(wk.numel() == 0, "窗口=0返回空张量")

    # 多页场景验证
    wk, wv = cache.get_kv_window(50)
    check(torch.allclose(wk, k[:, -50:, :], atol=1e-5), "跨页窗口数据正确")

    cache.clear()


def test_empty_cache():
    """边界: 空缓存"""
    logger.info("--- 1.6 空缓存边界 ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)

    all_k, all_v = cache.get_all_kv()
    check(all_k.numel() == 0, "空缓存get_all_kv返回空张量")
    check(cache.total_tokens == 0, "初始token=0")

    # 追加0 token
    k, v = random_kv(0)
    cache.append_kv(k, v)
    check(cache.total_tokens == 0, "追加0 token不改变状态")

    cache.clear()


def test_exact_page_boundary():
    """边界: 恰好填满一整页"""
    logger.info("--- 1.7 页边界 ---")
    cache = PagedKVCache(page_size=16, max_pages=MAX_PAGES, device=DEVICE)

    # 写满第1页
    k1, v1 = random_kv(16)
    cache.append_kv(k1, v1)
    check(cache.allocated_page_count == 1, "满页仍为1页")
    check(cache._current_page.remaining == 0, "当前页无剩余空间")

    # 再写1个token → 触发新页分配
    k2, v2 = random_kv(1)
    cache.append_kv(k2, v2)
    check(cache.allocated_page_count == 2, "超过边界自动分配第2页")
    check(cache.total_tokens == 17, "总token=17")

    all_k, _ = cache.get_all_kv()
    check(torch.allclose(all_k[:, :16, :], k1, atol=1e-5), "第1页数据正确")
    check(torch.allclose(all_k[:, 16:, :], k2, atol=1e-5), "第2页数据正确")

    cache.clear()


def test_max_capacity():
    """边界: 达到最大容量限制"""
    logger.info("--- 1.8 最大容量限制 ---")
    tiny_max = 3  # 只允许3页
    cache = PagedKVCache(page_size=16, max_pages=tiny_max, device=DEVICE)

    # 写满3页
    k, v = random_kv(48)  # 3 × 16
    cache.append_kv(k, v)
    check(cache.allocated_page_count == 3, "写满3页")
    check(cache.total_tokens == 48, "token=48")

    # 再写触发异常
    k2, v2 = random_kv(1)
    try:
        cache.append_kv(k2, v2)
        check(False, "应触发RuntimeError", "超过max_pages未抛异常")
    except RuntimeError as e:
        check(True, "超容量正确抛出RuntimeError")
        logger.info(f"      异常信息: {str(e)[:80]}...")

    cache.clear()


# ================================================================
# 2. 设备与数据类型测试
# ================================================================

def test_device_dtype():
    """设备和数据类型"""
    logger.info("--- 2.1 设备/dtype ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE, dtype=torch.float16)

    k, v = random_kv(10)
    cache.append_kv(k, v)

    all_k, all_v = cache.get_all_kv()
    check(all_k.is_cuda, f"输出在{DEVICE}上", f"实际在{all_k.device}")
    check(all_k.dtype == torch.float16, f"dtype=float16", f"实际{all_k.dtype}")

    cache.clear()


# ================================================================
# 3. 性能基准
# ================================================================

def benchmark_get_all_kv():
    """性能: get_all_kv 大序列吞吐"""
    logger.info("--- 3.1 get_all_kv 性能基准 ---")
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)

    SEQ_LENS = [256, 512, 1024, 2048]
    REPEAT = 20

    for seq_len in SEQ_LENS:
        cache.clear()
        k, v = random_kv(seq_len)
        cache.append_kv(k, v)

        # 预热
        for _ in range(5):
            _ = cache.get_all_kv()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        # 计时
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = cache.get_all_kv()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / REPEAT) * 1000
        pages = cache.allocated_page_count
        logger.info(
            f"  seq={seq_len:5d} | {pages:2d}页 | "
            f"avg={avg_ms:7.3f}ms | {'⚠️ 慢' if avg_ms > 5 else '✅ 快'}"
        )

    cache.clear()


def benchmark_append():
    """性能: append 吞吐"""
    logger.info("--- 3.2 append 性能基准 ---")

    # Prefill: 批量
    cache = PagedKVCache(page_size=PAGE_SIZE, max_pages=MAX_PAGES, device=DEVICE)
    BATCH_SIZES = [128, 256, 512, 1024]
    REPEAT = 50

    for bs in BATCH_SIZES:
        if bs > PAGE_SIZE * MAX_PAGES:
            continue
        cache.clear()
        k, v = random_kv(bs)

        # 预热
        cache.clear()
        cache.append_kv(k, v)

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            cache.clear()
            cache.append_kv(k, v)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / REPEAT) * 1000
        logger.info(f"  Prefill batch={bs:5d} | avg={avg_ms:8.3f}ms | {bs / (elapsed / REPEAT):.0f} tok/s")

    # Decode: 单token
    cache.clear()
    k_single, v_single = random_kv(1)
    REPEAT_DECODE = 200

    # 预热
    for _ in range(10):
        cache.append_kv_single(k_single[:, 0, :], v_single[:, 0, :])
    cache.clear()

    t0 = time.perf_counter()
    for _ in range(REPEAT_DECODE):
        cache.append_kv_single(k_single[:, 0, :], v_single[:, 0, :])
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg_us = (elapsed / REPEAT_DECODE) * 1_000_000
    logger.info(f"  Decode 单token ×{REPEAT_DECODE} | avg={avg_us:8.1f}μs/call")

    cache.clear()


# ================================================================
# 4. 统计信息测试
# ================================================================

def test_stats():
    """统计信息正确性"""
    logger.info("--- 4.1 统计信息 ---")
    cache = PagedKVCache(page_size=128, max_pages=8, device=DEVICE)

    k, v = random_kv(300)  # ceil(300/128)=3页
    cache.append_kv(k, v)
    _ = cache.get_all_kv()
    _ = cache.get_all_kv()

    stats = cache.get_stats()
    check(stats["total_tokens"] == 300, "total_tokens=300")
    check(stats["allocated_pages"] == 3, "allocated_pages=3")
    check(stats["max_pages"] == 8, "max_pages=8")
    check(stats["page_size"] == 128, "page_size=128")
    check(0 < stats["utilization"] < 1, "利用率在0~1之间")
    check(stats["append_call_count"] == 1, "append调用1次")
    check(stats["total_appended_tokens"] == 300, "累计写入300 token")
    check(stats["get_all_kv_call_count"] == 2, "get_all_kv调用2次")
    check(stats["estimated_memory_mb"] > 0, "显存估算>0")

    logger.info(f"      统计: {stats}")

    cache.clear()
    stats_empty = cache.get_stats()
    check(stats_empty["total_tokens"] == 0, "clear后total_tokens=0")


# ================================================================
# 主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="分页KV缓存测试")
    parser.add_argument("--quick", action="store_true", help="仅冒烟测试")
    parser.add_argument("--benchmark", action="store_true", help="仅性能基准")
    args = parser.parse_args()

    logger.info(f"{'='*60}")
    logger.info(f"分页KV缓存模块测试 — device={DEVICE}")
    logger.info(f"{'='*60}")

    if args.benchmark:
        benchmark_append()
        benchmark_get_all_kv()
        print_result()
        return

    # ---- 功能测试 ----
    test_basic_append()
    test_multi_page_span()
    test_page_reuse_after_clear()
    test_append_kv_single()
    test_get_kv_window()
    test_empty_cache()
    test_exact_page_boundary()
    test_max_capacity()
    test_device_dtype()
    test_stats()

    if not args.quick:
        benchmark_append()
        benchmark_get_all_kv()

    print_result()


def print_result():
    logger.info(f"\n{'='*60}")
    total = PASS + FAIL
    if FAIL == 0:
        logger.info(f"🎉 全部通过! ({total} tests, {PASS} pass, {FAIL} fail)")
    else:
        logger.error(f"❌ 有失败项! ({total} tests, {PASS} pass, {FAIL} fail)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
