#!/usr/bin/env python3
"""RAG sqlite-vec 采用决策基准 CLI（后置可准备）。

输出三层：
  1) decision：复用 rag_ann.evaluate_ann_decision（语料在扫描预算内或扩展缺失
     → NO_GO；语料超预算且 sqlite-vec 可用 → benchmark-gate-only GO）；
  2) sqlite_vec_available：本机扩展可用性；
  3) 说明：当前基线 = SQLite FTS5 + 有界 cosine 扫描；真正的大规模 ANN 对比
     基准需要真实语料 + 扩展安装，属 RAG 后置验收。

用法示例：
  python scripts/rag_ann_benchmark.py --corpus-chunks 50000 --scan-budget 1000 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rag_ann import evaluate_ann_decision, sqlite_vec_available  # noqa: E402

BENCHMARK_SCHEMA_VERSION = "qlh.rag.ann_benchmark.v1"


def format_report(
    corpus_chunks: int, scan_budget: int, extension_available: bool | None = None,
) -> dict[str, object]:
    """组装采用决策报告（path-free、保守）。"""
    available = sqlite_vec_available() if extension_available is None else bool(extension_available)
    decision = evaluate_ann_decision(
        corpus_chunks=corpus_chunks, scan_budget=scan_budget,
        extension_available=available,
    )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "corpus_chunks": int(corpus_chunks),
        "scan_budget": int(scan_budget),
        "sqlite_vec_available": bool(available),
        "decision": decision["decision"],
        "reason": decision["reason"],
        "baseline": "FTS5 + bounded cosine scan",
        "note": "大规模 sqlite-vec ANN 对比需真实语料与扩展安装（RAG 后置）",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG ANN 采用决策基准")
    parser.add_argument("--corpus-chunks", type=int, required=True, help="语料分块数")
    parser.add_argument("--scan-budget", type=int, default=1000, help="有界扫描预算")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = format_report(args.corpus_chunks, args.scan_budget)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"decision={report['decision']} reason={report['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
