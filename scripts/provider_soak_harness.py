#!/usr/bin/env python3
"""RAG provider 长时 soak harness CLI（包装 src/provider_soak，后置可准备）。

用法示例：
  python scripts/provider_soak_harness.py --iterations 500 --failure-rate 0.05 --hang-rate 0.02
  python scripts/provider_soak_harness.py --iterations 2000 --seed 1

默认用 fake embedding provider（确定性摘要）跑注入式 soak；真实
Ollama / llama.cpp provider 的长时真跑由 RAG 后置验收注入 embed_fn 完成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from provider_soak import SoakSpec, run_provider_soak  # noqa: E402


def _fake_embed(index: int) -> str:
    """模拟离线 embedding：返回确定性摘要（只读、无网络、无副作用）。"""
    return hashlib.sha256(f"qlh-rag-soak-{index}".encode("utf-8")).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG embedding provider soak harness")
    parser.add_argument("--iterations", type=int, default=500, help="总调用次数")
    parser.add_argument("--failure-rate", type=float, default=0.05, help="注入失败比例 [0,1]")
    parser.add_argument("--hang-rate", type=float, default=0.05, help="注入挂起/超时比例 [0,1]")
    parser.add_argument("--timeout", type=float, default=2.0, help="单次超时参考（秒，用于报告）")
    parser.add_argument("--seed", type=int, default=20260821, help="确定性随机种子")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = SoakSpec(
        iterations=args.iterations,
        failure_rate=args.failure_rate,
        hang_rate=args.hang_rate,
        timeout_s=args.timeout,
        seed=args.seed,
    )
    report = run_provider_soak(_fake_embed, spec=spec)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        latency = report["latency_ms"]
        print(
            f"soak: iter={report['iterations']} ok={report['successes']} "
            f"fail={report['failures']} timeout={report['timeouts']} "
            f"recovered={report['recovered']} "
            f"p50={latency['p50']}ms p95={latency['p95']}ms avg={latency['avg']}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
