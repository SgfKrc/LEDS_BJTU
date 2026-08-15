#!/usr/bin/env python3
"""P7 (2026-08-16): 跨节点日志聚合 CLI——`qlh log`。

从主节点聚合本地与全部在线从节点的最近日志行（经
``GET /api/cluster/nodes/log-aggregate``），按节点标注输出。

用法:
    python scripts/qlh_log.py [--host HOST] [--port PORT] [--lines N]
                              [--level LEVEL] [--name NAME] [--token TOKEN]

安全: 只输出日志行与 node_id；认证 token 仅用于请求头，不写入输出；
单节点失败显示截断错误摘要，不中断其余节点。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def _request(base_url: str, path: str, token: str | None) -> dict[str, Any]:
    url = f"{base_url}{path}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"[error] 日志聚合请求失败: {str(exc)[:120]}")
    if not isinstance(payload, dict):
        raise SystemExit("[error] 日志聚合响应格式无效")
    return payload


def _print_node(node_id: str, logs: list[str]) -> None:
    print(f"── {node_id} ──")
    for line in logs[-20:]:
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="聚合各节点最近日志（qlh log）")
    parser.add_argument("--host", default="127.0.0.1", help="主节点 API 地址")
    parser.add_argument("--port", type=int, default=8000, help="主节点 API 端口")
    parser.add_argument("--lines", type=int, default=50, help="每节点最近行数上限")
    parser.add_argument("--level", default="", help="日志级别过滤 (ERROR/WARNING/INFO/DEBUG)")
    parser.add_argument("--name", default="", help="logger 名称过滤")
    parser.add_argument("--token", default="", help="日志 API 访问 token（不写入输出）")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON（供脚本消费）")
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    query = f"limit={args.lines}"
    if args.level:
        query += f"&level={args.level}"
    if args.name:
        query += f"&name={args.name}"

    payload = _request(base, f"/api/cluster/nodes/log-aggregate?{query}", args.token)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0

    local = payload.get("local") or {}
    print(f"== 本地节点 {local.get('node_id', '?')}（{payload.get('total_workers', 0)} 个在线 worker）==")
    _print_node(local.get("node_id", "local"), local.get("logs") or [])
    for worker in payload.get("workers") or []:
        if worker.get("error"):
            print(f"── {worker.get('node_id', '?')} ── [error] {worker['error']}")
            continue
        _print_node(worker.get("node_id", "worker"), worker.get("logs") or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
