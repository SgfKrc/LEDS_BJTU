"""SSE 流式首 token 延迟对比（2.6 验收：新网关 vs 旧 FastAPI 偏差 <10%）

用法：python scripts/sse_ttft_compare.py [--runs 5] [--old URL] [--new URL]
"""
import argparse
import json
import statistics
import time
import urllib.request


def sse_first_token_latency(base: str, runs: int) -> list:
    """POST /api/chat/stream，返回每次的首 token 延迟（秒）。"""
    url = base + "/api/chat/stream"
    body = json.dumps({"message": "你好，请简单介绍一下你自己", "session_id": "s1", "max_new_tokens": 64}).encode("utf-8")
    latencies = []
    for i in range(runs):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first_tok = None
        with urllib.request.urlopen(req, timeout=120) as resp:
            buf = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buf += chunk
                # 第一个 data: 行（SSE 事件帧首字节数据到达即算首 token）
                if b"\ndata:" in buf or buf.startswith(b"data:"):
                    first_tok = time.perf_counter() - t0
                    break
        if first_tok is None:
            raise RuntimeError(f"{base} 未收到 SSE data 事件")
        latencies.append(first_tok)
        print(f"  run {i + 1}: {first_tok:.3f}s")
    return latencies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--old", default="http://127.0.0.1:8000")
    ap.add_argument("--new", default="http://127.0.0.1:8100")
    args = ap.parse_args()

    print(f"== SSE 首 token 延迟对比（{args.runs} 次，交替测量）")
    # 交替测量消除顺序/CPU 状态偏差；取中位数（抗离群）
    old, new = [], []
    for i in range(args.runs):
        print(f"-- round {i + 1}")
        old.append(sse_first_token_latency(args.old, 1)[0])
        new.append(sse_first_token_latency(args.new, 1)[0])
    old_med = statistics.median(old)
    new_med = statistics.median(new)
    dev = (new_med - old_med) / old_med * 100 if old_med else float("inf")
    print(f"\n旧网关中位数: {old_med:.3f}s  新网关中位数: {new_med:.3f}s")
    print(f"偏差: {dev:+.1f}%（判据 <10%）")
    ok = abs(dev) < 10.0
    print("结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
