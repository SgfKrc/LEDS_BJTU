"""QLH legacy-control 日志服务桩（阶段 2 T6）

承载 /logs/* 三个端点，模拟未来从 api_server 剥离的控制面遗留进程
（见 docs/TUI适配实施计划.md §3.3 时序决策）。纯标准库，零第三方依赖，
可作为未来 legacy-control 进程（阶段 2.5 扩展 review/email/sessions 等）的原型。

内部端点（网关透传：对外 /api/logs/* 去掉 /api 前缀后转发至此）：
  GET /logs/recent?limit&level   日志缓冲尾部 + buffer 统计
  GET /logs                      日志文件列表
  GET /logs/stats                缓冲与文件统计

认证：X-QLH-Log-Token 可选（允许无 token；有 token 时当前不校验，
未来真实实现按 api_server 日志端点语义校验）。

用法：python src/legacy_control.py [--port N] [--log-dir PATH]
启动后向 stdout 打印 LEGACY_CONTROL_LISTENING:<port> 供编排/测试探活。
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

BUFFER_CAPACITY = 1000
DEFAULT_LOG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
)

# 模拟进程内日志缓冲（对齐 api_server 的 logging 内存缓冲语义：buffer_size 等字段）
SAMPLE_LOGS = [
    {"level": "INFO", "time": "2026-08-02 10:00:00", "name": "scheduler", "message": "节点注册成功 node_id=test-client"},
    {"level": "INFO", "time": "2026-08-02 10:00:05", "name": "scheduler", "message": "层分配已计算 strategy=graph"},
    {"level": "WARN", "time": "2026-08-02 10:01:00", "name": "tcp_comm", "message": "心跳超时 node_id=test-client"},
    {"level": "ERROR", "time": "2026-08-02 10:02:00", "name": "model_module", "message": "模型加载失败，回退 llama_cpp"},
    {"level": "INFO", "time": "2026-08-02 10:03:00", "name": "api_server", "message": "chat 请求完成 duration_ms=1234"},
]

# ---- 控制面端点桩（阶段 2 过渡：legacy-control 承载控制面域） ----
# 阶段 3 由 control-svc(TS) 实现真实逻辑；此处返回契约形状的空数据/默认值。
# 覆盖域：sessions / conversations / settings / review / workflows / bootstrap /
#         models registry / downloadable / gguf / files / download
CONTROL_EMPTY = {
    "sessions": [],
    "conversations": [],
    "workflows": [],
    "review/tickets": [],
    "models/registry": [],
    "models/downloadable": [],
    "models/gguf": [],
}


def _control_response(path: str, method: str):
    """控制面端点的桩响应：GET 列表类返回空结构，其余返回操作成功确认。"""
    if method == "GET":
        for key in ("models/registry", "models/downloadable", "models/gguf"):
            if path.startswith("/" + key):
                return {"status": "ok", key.split("/")[-1]: []}
        if path.startswith("/review/tickets"):
            return {"tickets": []}
        if path.startswith("/review/can-vote"):
            return {"can_vote": True}
        # /cluster/review/*（审查票在 cluster 路径下，api_server 无独立 /api/review）
        if path.startswith("/cluster/review/tickets"):
            return {"tickets": []}
        if path.startswith("/cluster/review/can-vote"):
            return {"can_vote": True}
        # /logs 扩展端点（export/download/node/nodes-summary/单文件）
        if path.startswith("/logs/nodes-summary"):
            return {"nodes": {}, "count": 0}
        if path.startswith("/logs/node/"):
            return {"count": 0, "logs": [], "node_id": path.split("/")[-2]}
        if path.startswith("/logs/export"):
            return {"status": "ok", "archive": "qlh-logs.zip", "files": 0}
        if path.startswith("/logs/download"):
            return {"status": "ok", "filename": "qlh.log", "size": 0}
        if path.startswith("/logs/"):
            return {"status": "ok", "filename": path.split("/")[-1], "size": 0}
        if path.startswith("/models/download/"):
            return {"status": "ok", "filename": path.split("/")[-1], "size": 0}
        if path.startswith("/sessions"):
            return {"sessions": []}
        if path.startswith("/conversations"):
            return {"conversations": [], "sync_status": {"dirty": False}}
        if path.startswith("/workflows"):
            return {"workflows": []}
        if path.startswith("/settings"):
            return {"settings": {}}
        if path.startswith("/bootstrap/info"):
            return {"available": True, "identity_verified": False}
        if path.startswith("/review/can-vote"):
            return {"can_vote": True}
        if path == "/presets":
            return {
                "presets": [
                    {"id": "intro", "icon": "👋", "label": "自我介绍",
                     "question": "请简单介绍一下你自己，你能做什么？",
                     "estimated_prompt_tokens": 25, "estimated_response_tokens": 120,
                     "estimated_memory_mb": 13.6, "estimated_seconds": 4.1},
                    {"id": "edge_computing", "icon": "🌐", "label": "边缘计算科普",
                     "question": "什么是边缘计算？它和云计算有什么区别？",
                     "estimated_prompt_tokens": 35, "estimated_response_tokens": 200,
                     "estimated_memory_mb": 22.0, "estimated_seconds": 6.9},
                ],
                "current_speed_tok_s": 29,
                "current_quant": "int4",
                "max_new_tokens": 512,
            }
        if path == "/user/settings":
            return {"settings": {}, "source": "none"}
        if path == "/db/health":
            return {
                "status": "retired",
                "backend": "sqlite",
                "effective_mode": "local_only",
                "message": "远端 PostgreSQL 已退场；用户数据由主节点 SQLite 持有",
            }
        return {"status": "ok"}
    # POST/PUT/DELETE：操作确认（空体场景禁止 204）
    if path.startswith("/cluster/review/mail-poll"):
        return {"status": "ok", "polled": 0}
    if path.startswith("/cluster/review/expire-check"):
        return {"status": "ok", "expired": 0}
    if path.startswith("/cluster/review/vote"):
        return {"status": "ok"}
    if path.startswith("/cluster/review/create"):
        return {"ticket_id": "stub-ticket", "status": "ok"}
    if path.startswith("/cluster/review/tickets"):
        return {"status": "deleted", "count": 0}
    if path.startswith("/logs/client-error"):
        return {"status": "ok", "received": 1}
    if path.startswith("/logs"):
        return {"status": "ok", "deleted": 0}
    if path.startswith("/bootstrap/first-connect"):
        return {"status": "ok", "identity_verified": True}
    if path.startswith("/sessions"):
        return {"session_id": "stub-session", "status": "ok"}
    if path.startswith("/review"):
        return {"ticket_id": "stub-ticket", "status": "ok"}
    if path.startswith("/workflows"):
        return {"workflow_id": "stub-workflow", "status": "ok"}
    if path.startswith("/models/registry"):
        return {"model_id": "stub-model", "status": "ok"}
    if path.startswith("/models/download/"):
        return {"status": "ok", "downloaded": True}
    return {"status": "ok"}


def _list_log_files(log_dir: str) -> list:
    files = []
    if os.path.isdir(log_dir):
        for name in sorted(os.listdir(log_dir)):
            fp = os.path.join(log_dir, name)
            if os.path.isfile(fp):
                try:
                    st = os.stat(fp)
                    files.append({"name": name, "size": st.st_size, "modified": int(st.st_mtime)})
                except OSError:
                    continue
    return files


def _stats(log_dir: str) -> dict:
    files = _list_log_files(log_dir)
    levels: dict = {}
    for entry in SAMPLE_LOGS:
        levels[entry["level"]] = levels.get(entry["level"], 0) + 1
    return {
        "log_dir": log_dir,
        "files_count": len(files),
        "files_total_bytes": sum(f["size"] for f in files),
        "buffer_size": len(SAMPLE_LOGS),
        "buffer_capacity": BUFFER_CAPACITY,
        "buffer_total_seen": 42,
        "buffer_dropped_estimate": 0,
        "levels": levels,
        "nodes": {},
    }


class Handler(BaseHTTPRequestHandler):
    # 记录收到的请求头（含 X-QLH-Log-Token 透传情况），供测试断言
    seen_requests: list = []

    def _send(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        token = self.headers.get("X-QLH-Log-Token", "")
        Handler.seen_requests.append({"method": "GET", "path": path, "token": token})

        if path == "/logs/recent":
            limit = int(qs.get("limit", ["50"])[0])
            level = qs.get("level", [""])[0]
            filtered = [
                e for e in SAMPLE_LOGS if (not level or e["level"] == level)
            ][:limit]
            self._send(200, {
                "count": len(filtered),
                "matched": len(filtered),
                "buffer_size": len(SAMPLE_LOGS),
                "buffer_capacity": BUFFER_CAPACITY,
                "logs": filtered,
            })
        elif path == "/logs":
            self._send(200, {"files": _list_log_files(self.log_dir)})
        elif path == "/logs/stats":
            self._send(200, _stats(self.log_dir))
        elif path.startswith(("/sessions", "/conversations", "/settings",
                              "/review", "/workflows", "/bootstrap",
                              "/models/registry", "/models/downloadable",
                              "/models/gguf", "/models/files", "/models/download",
                              "/presets", "/user", "/db",
                              "/cluster/review", "/logs")):
            self._send(200, _control_response(path, "GET"))
        else:
            self._send(404, {"detail": f"Route GET:{path} not found"})

    def _handle_write(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        Handler.seen_requests.append({"method": method, "path": path, "token": self.headers.get("X-QLH-Log-Token", "")})
        if path.startswith(("/sessions", "/conversations", "/settings",
                            "/review", "/workflows", "/bootstrap",
                            "/models/registry", "/models/downloadable",
                            "/models/download/", "/models/gguf", "/user", "/db",
                            "/cluster/review", "/logs")):
            self._send(200, _control_response(path, method))
        else:
            self._send(404, {"detail": f"Route {method}:{path} not found"})

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        try:
            self._handle_write("POST")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_PUT(self) -> None:
        try:
            self._handle_write("PUT")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_DELETE(self) -> None:
        try:
            self._handle_write("DELETE")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt: str, *args) -> None:  # 静默访问日志
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="QLH legacy-control 日志服务桩")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    Handler.log_dir = args.log_dir  # type: ignore[attr-defined]
    print(f"LEGACY_CONTROL_LISTENING:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
