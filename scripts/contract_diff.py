"""阶段 2 验收工具：双网关响应契约对比（旧 FastAPI vs TS 网关）

对比对象：
  --old  旧 FastAPI 网关（api_server.py，退役前基线，默认 http://127.0.0.1:8000）
  --new  TS 网关（gateway/dist/main.js，默认 http://127.0.0.1:8100）

对比方法：
  - 端点清单自动从 src/api_server.py 提取（@app.<method> 装饰器）
  - 对每个端点分别请求两个网关，比较：
      ① HTTP 状态码
      ② 响应 JSON 结构指纹（递归：dict 的 key 集合 + list 元素结构 + 值类型）
  - **值差异不计为失败**（旧网关读真实状态/DB，新网关读桩/上游，数值必然不同；
    结构一致即契约一致）
  - 结构差异/状态码差异计入报告，退出码 1

用法：
  python scripts/contract_diff.py [--old URL] [--new URL] [--list-only] [--endpoint 过滤子串]
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 常用 POST/PUT 请求体模板（满足 api_server Pydantic 校验；未列出的端点
# 若遇 422 会标记"待复测"而非契约差异——422 是旧网关请求校验，非网关行为差异）
BODY_TEMPLATES: dict = {
    ("POST", "/api/chat"): {"message": "你好", "session_id": "s1", "max_new_tokens": 128},
    ("POST", "/api/chat/stream"): {"message": "你好", "session_id": "s1", "max_new_tokens": 128},
    ("POST", "/api/chat/clear"): {},
    ("POST", "/api/models/load"): {"model_id": "qwen-1_8b-chat", "quant_type": "int4", "engine": "torch"},
    ("POST", "/api/models/switch"): {"model_id": "qwen-1_8b-chat", "quant_type": "int4", "engine": "auto"},
    ("POST", "/api/device/select-gpu"): {"gpu_index": 0},
    ("POST", "/api/device/auto-configure"): {},
    ("PUT", "/api/cluster/config/max-nodes"): {"max_nodes": 8},
    ("POST", "/api/cluster/connect"): {"master_host": "127.0.0.1", "master_port": 8888, "switch_to_client": False},
    ("POST", "/api/cluster/nodes/register"): {"node_id": "t1", "hostname": "h", "address": "a", "network_type": "wifi", "node_type": "pc"},
    ("POST", "/api/cluster/android/register"): {"node_id": "a1", "hostname": "h", "node_type": "android"},
    ("POST", "/api/cluster/android/heartbeat"): {"node_id": "a1"},
    ("POST", "/api/cluster/queue/strategy"): {"strategy": "mlfq"},
    ("PUT", "/api/cluster/config/distributed-inference"): {"enabled": True},
    ("POST", "/api/cluster/spare-master"): {"target_node_id": "x"},
    ("POST", "/api/cluster/review/create"): {"title": "t", "description": "d"},
    ("POST", "/api/cluster/review/vote"): {"ticket_id": "x", "vote": "approve"},
    ("PUT", "/api/user/settings"): {"settings": {}},
    ("POST", "/api/models/registry"): {"model_id": "m1", "name": "M1"},
    ("POST", "/api/bootstrap/first-connect"): {"host": "1.2.3.4", "port": 8888},
    ("POST", "/api/workflows/register"): {"name": "w1"},
    ("POST", "/api/sessions"): {"title": "s1"},
}


def extract_endpoints() -> list:
    """从 api_server.py 提取 (method, path) 列表。"""
    src = (REPO_ROOT / "src" / "api_server.py").read_text(encoding="utf-8")
    return [(m.upper(), p) for m, p in
            re.findall(r'@app\.(get|post|put|delete)\(["\']([^"\']+)', src)]


def request_json(base: str, method: str, path: str):
    """请求并返回 (status, body_or_error_text)。POST/PUT 用 BODY_TEMPLATES 模板。"""
    url = base + path
    body = BODY_TEMPLATES.get((method, path))
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(100000)
            try:
                return resp.status, json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                return resp.status, {"__raw__": raw[:200].decode("utf-8", "replace")}
    except urllib.error.HTTPError as e:
        raw = e.read(100000)
        try:
            return e.code, json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return e.code, {"__raw__": raw[:200].decode("utf-8", "replace")}
    except Exception as e:
        return -1, {"__error__": str(e)}


def fingerprint(node, depth: int = 0) -> object:
    """响应结构指纹：dict→排序 key 列表映射，list→元素指纹(去重)，忽略具体值。

    int/float 统一为 "num"（2026-08-03 修复）：JSON 语义上二者等价——
    Python json 保留 0.0（float），JS JSON.stringify(0.0) 输出 0（int），
    双网关对同一数值产生不同类型是编码差异而非契约差异。
    """
    if depth > 6:
        return "…"
    if isinstance(node, dict):
        out = {}
        for k in sorted(node.keys()):
            out[k] = fingerprint(node[k], depth + 1)
        return out
    if isinstance(node, list):
        if not node:
            return []
        return sorted({json.dumps(fingerprint(x, depth + 1), ensure_ascii=False)
                       for x in node[:3]})
    if isinstance(node, bool):
        return "bool"
    if isinstance(node, (int, float)):
        return "num"
    return type(node).__name__


def compare_one(old_base: str, new_base: str, method: str, path: str):
    """返回 (类别, 说明)。类别: ok | diff | body_422。"""
    old_st, old_body = request_json(old_base, method, path)
    new_st, new_body = request_json(new_base, method, path)
    # 422 且无模板：旧网关请求校验拒绝（测试 body 不完整），非契约差异
    if old_st == 422 and (method, path) not in BODY_TEMPLATES:
        return "body_422", "旧网关 422（测试 body 未提供模板）"
    issues = []
    if old_st != new_st:
        issues.append(f"状态码 {old_st} != {new_st}")
    old_fp = fingerprint(old_body)
    new_fp = fingerprint(new_body)
    if old_fp != new_fp:
        issues.append("结构不一致")
        if isinstance(old_fp, dict) and isinstance(new_fp, dict):
            miss_new = sorted(set(old_fp) - set(new_fp))
            miss_old = sorted(set(new_fp) - set(old_fp))
            if miss_new:
                issues.append("旧有新无: " + ", ".join(miss_new))
            if miss_old:
                issues.append("新有旧无: " + ", ".join(miss_old))
    return ("ok" if not issues else "diff"), "; ".join(issues) if issues else "一致"


def main() -> int:
    parser = argparse.ArgumentParser(description="双网关契约对比")
    parser.add_argument("--old", default="http://127.0.0.1:8000")
    parser.add_argument("--new", default="http://127.0.0.1:8100")
    parser.add_argument("--list-only", action="store_true", help="只列出端点不请求")
    parser.add_argument("--endpoint", default="", help="只对比路径包含此子串的端点")
    parser.add_argument("--skip", default="upload", help="跳过路径包含子串的端点（默认 upload，multipart 未实现）")
    args = parser.parse_args()

    endpoints = extract_endpoints()
    print(f"端点总数: {len(endpoints)}")
    if args.list_only:
        for m, p in endpoints:
            print(f"  {m:6} {p}")
        return 0

    same = diff = pending = err = 0
    print(f"对比 {args.old}  vs  {args.new}\n")
    for m, p in endpoints:
        if args.endpoint and args.endpoint not in p:
            continue
        if args.skip and args.skip in p:
            print(f"  [跳过] {m:6} {p}（{args.skip}）")
            continue
        cat, note = compare_one(args.old, args.new, m, p)
        if cat == "ok":
            same += 1
        elif cat == "body_422":
            pending += 1
            print(f"  [待复测] {m:6} {p}: {note}")
        else:
            diff += 1
            print(f"  [差异] {m:6} {p}: {note}")
    print(f"\n一致 {same} / 差异 {diff} / 待复测(body) {pending} / 跳过 {len(endpoints) - same - diff - pending}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
