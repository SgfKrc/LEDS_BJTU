"""QLH 开发实测桩（scheduler-svc + inference-svc 综合）

供阶段 2 网关开发/演示/实测使用：单进程起两个端口
  :8020  scheduler-svc（/cluster/*、/device/*，字段对齐 docs/TUI适配实施计划.md §2.2）
  :8010  inference-svc（/v1/*，契约对齐 docs/微服务架构改造计划.md §4.1）

数据与 gateway/test/fake-scheduler.ts、fake-inference.ts 保持一致；
这是独立可运行版本（fake-*.ts 无独立入口，只能被 jest 加载）。

用法：python scripts/dev_stubs.py [--scheduler-port 8020] [--inference-port 8010]
"""
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# ============================================================
# scheduler-svc 路由表
# ============================================================
SCHEDULER_ROUTES: list = [
    ("GET", r"^/cluster/status$", {"run_mode": "distributed", "node_role": "master", "node_id": "test-master", "max_nodes": 8}),
    ("GET", r"^/cluster/my-role$", {"is_master": True, "is_provisional": False, "runtime_node_role": "master", "node_role": "master", "node_id": "test-master"}),
    ("GET", r"^/cluster/nodes$", {
        "count": 2, "online_count": 1, "offline_count": 1,
        "nodes": [
            {"node_id": "test-master", "role": "master", "node_type": "pc", "state": "online",
             "address": "100.64.0.1", "network_type": "tailscale", "avg_rtt_ms": 12.5,
             "task_count": 1, "error_count": 0, "last_heartbeat": 1754100000},
            {"node_id": "test-client", "role": "client", "node_type": "pc", "state": "offline",
             "address": "100.64.0.2", "network_type": "tailscale", "avg_rtt_ms": None,
             "task_count": 0, "error_count": 2, "last_heartbeat": 1754000000},
        ],
    }),
    ("GET", r"^/cluster/invite$", {"master_host": "100.64.0.1", "master_port": 8888, "node_count": 2, "max_nodes": 8, "identity_verified": True}),
    ("GET", r"^/cluster/spare-master$", {"spare_master": {"node_id": "test-client", "hostname": "spare-pc", "is_online": True}}),
    ("GET", r"^/cluster/master-health$", {"master_online": True, "master_host": "100.64.0.1", "master_port": 8888, "last_seen_seconds_ago": 0, "stale": False}),
    ("GET", r"^/cluster/discover$", {"found": True, "master_host": "100.64.0.1", "master_port": 8888, "source": "stub", "stale": False}),
    ("GET", r"^/cluster/transfer-logs$", {"count": 1, "logs": [{"direction": "master->client", "from_role": "master", "to_role": "client", "related_node": "test-client", "timestamp": 1754100000, "outcome": "ok"}]}),
    ("GET", r"^/cluster/config$", {"network": {"server_ip": "0.0.0.0", "server_port": 8888, "heartbeat_interval_s": 5}, "model": {"quant_type": "int4", "page_size": 512, "max_page_num": 100, "max_seq_len": 8192}}),
    ("GET", r"^/cluster/config/distributed-inference$", {"enabled": True, "default": False}),
    ("GET", r"^/cluster/layers$", {"total": 24, "strategy": "graph", "computed_at": 1754100000, "assignments": [{"node_id": "test-master", "role": "master", "start_layer": 0, "end_layer": 11, "has_embedding": True, "has_lm_head": False, "score": 0.9}]}),
    ("GET", r"^/cluster/queue$", {
        "paused": False, "strategy": "mlfq", "current_task": None, "queue_size": 1, "max_size": 100,
        "q0_depth": 1, "q1_depth": 0, "q2_depth": 0, "completed_count": 5,
        "aging_params": {"q0_max_tokens": 256, "q1_max_tokens": 512, "q1_to_q0_s": 30, "q2_to_q1_s": 60},
        "preempt_stats": {"count": 1, "total_overhead_ms": 120},
        "q0": [{"task_id": "t1", "original_level": 0, "wait_seconds": 3.2, "max_new_tokens": 256, "is_aged": False, "session_id": "s1"}],
        "q1": [], "q2": [],
    }),
    ("GET", r"^/device/profile$", {
        "os": {"system": "Windows", "release": "10.0.22631"}, "hostname": "test-pc",
        "cpu": {"model": "Intel(R) Core(TM) i5-12400F", "brand": "Intel", "physical_cores": 6, "logical_cores": 12},
        "ram": {"total_gb": 16.0, "available_gb": 8.5}, "memory": {"total_gb": 16.0, "available_gb": 8.5},
        "disk": {"free_gb": 100.0, "total_gb": 512.0},
        "gpus": [{"name": "NVIDIA GeForce RTX 3060", "gpu_type": "nvidia", "cuda_available": True, "vram_total_gb": 12.0}],
        "selected_gpu_index": 0, "tier_label": "laptop", "tier": 2, "score_total": 85.5,
        "recommendations": ["推荐使用 INT4 量化档位"], "warnings": [],
    }),
    ("POST", r"^/cluster/connect$", {"status": "connected", "message": "已连接主节点"}),
    ("POST", r"^/cluster/nodes/register$", {"status": "registered", "reason": "", "message": "注册成功"}),
    ("POST", r"^/cluster/android/register$", {"status": "registered", "node_id": "android-dev-1", "message": "Android 注册成功"}),
    ("POST", r"^/cluster/android/heartbeat$", {"status": "ok", "node_id": "android-dev-1"}),
    ("GET", r"^/cluster/spare-master/logs$", {"count": 0, "logs": []}),
    ("POST", r"^/cluster/nodes/[^/]+/deregister$", {"status": "deregistered"}),
    ("POST", r"^/cluster/transfer-master$", {"status": "transferred", "message": "主节点身份已转让"}),
    ("POST", r"^/cluster/spare-master$", {"status": "set", "message": "已设置备用主节点"}),
    ("POST", r"^/cluster/config/max-nodes$", {"status": "updated"}),
    ("PUT", r"^/cluster/config/max-nodes$", {"status": "updated"}),
    ("POST", r"^/cluster/email-test$", {"message": "测试邮件已发送", "status": "sent"}),
    ("POST", r"^/cluster/reset-identity$", {"status": "reset"}),
    ("POST", r"^/cluster/config/distributed-inference$", {"status": "updated"}),
    ("PUT", r"^/cluster/config/distributed-inference$", {"status": "updated"}),
    ("POST", r"^/cluster/layers$", {"status": "applied", "message": "已应用分层"}),
    ("PUT", r"^/cluster/layers$", {"status": "applied", "message": "已应用分层"}),
    ("POST", r"^/cluster/queue/strategy$", {"strategy": "mlfq"}),
    ("POST", r"^/cluster/queue/pause$", {"paused": True}),
    ("POST", r"^/cluster/queue/resume$", {"paused": False}),
    ("POST", r"^/cluster/queue/clear$", {"success": True, "cleared": 3}),
    ("DELETE", r"^/cluster/nodes/delete-ok$", {"status": "deleted"}, 200),
    ("DELETE", r"^/cluster/nodes/[^/]+$", {"detail": "节点不存在"}, 404),
    ("DELETE", r"^/cluster/spare-master$", {"status": "cleared", "message": "已清除备用主节点"}),
    ("DELETE", r"^/cluster/layers$", {"status": "cleared"}),
    ("DELETE", r"^/cluster/queue/task/cancel-ok$", {"success": True, "task_id": "cancel-ok", "message": "任务已取消"}),
    ("DELETE", r"^/cluster/queue/task/[^/]+$", {"success": False, "task_id": "x", "message": "任务不存在或已经完成，无法取消"}),
    ("POST", r"^/device/select-gpu$", {"selected_gpu": {"name": "NVIDIA GeForce RTX 3060"}, "selected_gpu_index": 0, "warning": ""}),
    ("POST", r"^/device/auto-configure$", {"applied_config": {"description": "laptop 档配置已应用"}, "tier": 2, "score": 85.5}),
]

# ============================================================
# inference-svc 路由表
# ============================================================
INFERENCE_ROUTES: list = [
    ("GET", r"^/v1/status$", {
        "model_loaded": True, "current_quant": "int4", "model_name": "Qwen/Qwen-1.8B-Chat",
        "active_model_id": "qwen-1_8b-chat", "engine": "torch",
        "gpu": {"name": "NVIDIA GeForce RTX 3060", "total_mb": 12288, "allocated_mb": 1792.0,
                "reserved_mb": 2048.0, "utilization": 35.0},
        "kv_cache": {"total_tokens": 512, "max_tokens": 65536, "allocated_pages": 4, "free_pages": 124,
                     "max_pages": 128, "page_size": 512, "utilization": 0.0313,
                     "estimated_memory_mb": 64.0, "rounds": 3, "total_time_s": 12.5},
    }),
    ("GET", r"^/v1/models/current$", {"loaded": True, "model_id": "qwen-1_8b-chat", "quant_type": "int4",
                                      "model_name": "Qwen/Qwen-1.8B-Chat", "model_path": "models/qwen-1_8b-chat",
                                      "engine": "torch", "total_params": 1842333696, "device": "cuda",
                                      "gpu_allocated_gb": 1.75, "gpu_reserved_gb": 2.0}),
    ("GET", r"^/v1/models/available$", {"models": [
        {"model_id": "qwen-1_8b-chat", "name": "Qwen 1.8B Chat", "engine": "torch"},
        {"model_id": "qwen-1_8b-chat-gguf", "name": "Qwen 1.8B Chat GGUF", "engine": "llama_cpp"}]}),
    ("GET", r"^/v1/models$", {"models": [
        {"model_id": "qwen-1_8b-chat", "name": "Qwen 1.8B Chat", "engine": "torch"},
        {"model_id": "qwen-1_8b-chat-gguf", "name": "Qwen 1.8B Chat GGUF", "engine": "llama_cpp"}],
        "active_model_id": "qwen-1_8b-chat"}),
    ("POST", r"^/v1/chat$", {"reply": "桩回复：你好，我是 QLH。", "session_id": "stub-session"}),
    ("POST", r"^/v1/chat/clear$", {"cleared": True}),
    ("POST", r"^/v1/chat/cancel$", {"status": "cancelled", "generation_id": "stub-gen"}),
    ("POST", r"^/v1/models/load$", {"status": "loaded", "model_id": "qwen-1_8b-chat", "engine": "torch"}),
    ("POST", r"^/v1/models/switch$", {"status": "switched", "model_id": "qwen-1_8b-chat-gguf", "engine": "llama_cpp"}),
    ("POST", r"^/v1/speculative/run$", {"accepted": 0, "drafted": 0, "verified": 1, "note": "stub"}),
]

SSE_BODY = "\n".join([
    'data: {"delta":"你"}', '', 'data: {"delta":"好"}', '', 'data: [DONE]', '',
])


def _make_handler(routes, sse_paths=None):
    sse_paths = sse_paths or {}  # {method: (regex, body_text)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, method):
            parsed = urlparse(self.path)
            path = parsed.path
            if method in sse_paths:
                rx, body_text = sse_paths[method]
                if rx.search(path):
                    body = body_text.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            for route in routes:
                if route[0] != method:
                    continue
                m = re.match(route[1], path)
                if m:
                    status = route[3] if len(route) > 3 else 200
                    self._send(status, route[2])
                    return
            self._send(404, {"detail": f"Route {method}:{path} not found"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_PUT(self):
            self._handle("PUT")

        def do_DELETE(self):
            self._handle("DELETE")

        def log_message(self, fmt, *args):
            pass

    return Handler


def _build_scheduler_routes(client_mode: bool) -> list:
    """构造 scheduler 路由表；--client-mode 时 /cluster/my-role 模拟从节点身份。"""
    if not client_mode:
        return list(SCHEDULER_ROUTES)
    my_role = {
        "is_master": False,
        "is_provisional": False,
        "runtime_node_role": "client",
        "node_role": "client",
        "node_id": "test-client",
    }
    out = []
    for route in SCHEDULER_ROUTES:
        method, pattern, data = route[0], route[1], route[2]
        rest = route[3:]
        if pattern == r"^/cluster/my-role$":
            data = my_role
        out.append((method, pattern, data, *rest))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="QLH 开发实测桩")
    parser.add_argument("--scheduler-port", type=int, default=8020)
    parser.add_argument("--inference-port", type=int, default=8010)
    parser.add_argument("--client-mode", action="store_true",
                        help="模拟从节点身份（/cluster/my-role 返回 is_master=False）")
    args = parser.parse_args()

    sched_server = HTTPServer(("127.0.0.1", args.scheduler_port),
                              _make_handler(_build_scheduler_routes(args.client_mode)))
    inf_server = HTTPServer(("127.0.0.1", args.inference_port),
                            _make_handler(INFERENCE_ROUTES,
                                          {"POST": (re.compile(r"^/v1/chat/stream$"), SSE_BODY)}))
    print(f"DEV_STUBS scheduler=:{args.scheduler_port} inference=:{args.inference_port}", flush=True)
    threading.Thread(target=sched_server.serve_forever, daemon=True).start()
    try:
        inf_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sched_server.server_close()
        inf_server.server_close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
