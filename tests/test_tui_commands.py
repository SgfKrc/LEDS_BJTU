"""TUI 命令系统测试（/ 开头命令：模型/量化/引擎切换、退出、设置等）

用假 API 桩验证命令解析、参数校验、请求构造与退出标志，
不依赖真实后端与终端。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import tui_admin as t


class FakeApi:
    """记录调用并返回预设响应的 API 桩。"""

    host = "127.0.0.1"
    port = 8000
    timeout = 5.0
    log_token = ""

    def __init__(self):
        self.calls = []

    @property
    def base_url(self):
        return "http://%s:%s" % (self.host, self.port)

    def get(self, path, params=None, with_log_token=False):
        self.calls.append(("GET", path))
        if path == "/models/current":
            return {"loaded": True, "model_id": "qwen-1.8b",
                    "engine": "pytorch", "quant_type": "int4",
                    "model_name": "Qwen", "device": "cuda"}
        if path == "/models/available":
            return {"models": [{"id": "int4", "name": "INT4"}],
                    "current": "int4", "current_engine": "pytorch"}
        if path == "/models":
            return {"models": [{"model_id": "qwen-1.8b", "name": "Qwen",
                                "supported_engines": ["pytorch", "llama_cpp"],
                                "description": ""}]}
        if path == "/cluster/my-role":
            return {"node_role": "client", "node_id": "n1"}
        if path == "/cluster/config/distributed-inference":
            return {"enabled": False}
        if path == "/cluster/queue":
            return {"strategy": "mlfq", "paused": False, "queue_size": 0,
                    "q0_depth": 0, "q1_depth": 0, "q2_depth": 0}
        if path == "/device/profile":
            return {"gpus": [{"name": "RTX 4060"}, {"name": "Tesla T4"}],
                    "selected_gpu_index": 0, "tier_label": "laptop",
                    "tier": "laptop", "score_total": 58.8,
                    "recommendations": [{"description": "推荐 INT4 量化"}],
                    "warnings": []}
        if path == "/presets":
            return {"presets": [
                {"id": "intro", "icon": "👋", "label": "自我介绍",
                 "question": "你好？", "estimated_prompt_tokens": 10,
                 "estimated_response_tokens": 20, "estimated_memory_mb": 1.0,
                 "estimated_seconds": 1.0},
                {"id": "code", "icon": "💻", "label": "代码助手",
                 "question": "写个函数", "estimated_prompt_tokens": 5,
                 "estimated_response_tokens": 30, "estimated_memory_mb": 2.0,
                 "estimated_seconds": 2.0},
            ], "current_quant": "int4",
               "current_speed_tok_s": 29, "max_new_tokens": 512}
        if path == "/cluster/nodes":
            return {"count": 1, "online_count": 1,
                    "nodes": [{"node_id": "n1"}]}
        return {}

    def post(self, path, body=None, params=None):
        self.calls.append(("POST", path, body))
        if path == "/models/switch":
            return {"success": True, "model_name": "Qwen",
                    "quant_type": body.get("quant_type"),
                    "engine": body.get("engine")}
        if path == "/models/load":
            return {"model_name": "Qwen", "current_quant": "int8"}
        if path == "/device/select-gpu":
            return {"selected_gpu_index": 1, "selected_gpu": {"name": "Tesla T4"}}
        if path == "/system/shutdown":
            return {"ok": True}
        if path == "/chat/generations/g-1/cancel":
            raise t.ApiError("HTTP 404: not found")
        return {"status": "ok", "message": ""}

    def put(self, path, body=None):
        self.calls.append(("PUT", path, body))
        return {"status": "ok"}

    def delete(self, path, with_log_token=False):
        self.calls.append(("DELETE", path))
        return {"status": "ok"}


class FakeApp(t.BaseApp):
    def __init__(self, api=None):
        super().__init__(api or FakeApi(), 3.0, is_plain=True)
        self.out = []

    def show_output(self, lines, title=""):
        self.out = list(lines)


@pytest.fixture
def app():
    return FakeApp()


class TestCommandParsing:
    def test_unknown_command(self, app):
        msg, style = app.exec_command("/bogus")
        assert style == "err"
        assert "未知命令" in msg

    def test_missing_slash(self, app):
        msg, style = app.exec_command("quit")
        assert style == "err"

    def test_missing_args(self, app):
        msg, style = app.exec_command("/quant")
        assert style == "warn"
        assert "用法" in msg

    def test_extra_args(self, app):
        msg, style = app.exec_command("/gpu 1 2")
        assert style == "warn"

    def test_help_lists_commands(self, app):
        msg, style = app.exec_command("/help")
        assert style == "ok"
        assert app.out and "── 系统 ──" in app.out[0]
        assert any("/switch" in line for line in app.out)
        assert any("/shutdown" in line for line in app.out)

    def test_alias(self, app):
        msg, style = app.exec_command("/h")
        assert style == "ok"


class TestExitCommands:
    def test_quit_exits_tui_only(self, app):
        app.exec_command("/quit")
        assert app.exit_requested is True
        assert app.shutdown_backend is False

    def test_shutdown_requests_backend(self, app):
        msg, style = app.exec_command("/shutdown")
        assert style == "ok"
        assert app.exit_requested is True
        assert app.shutdown_backend is True
        assert any(c[0] == "POST" and c[1] == "/system/shutdown"
                   for c in app.api.calls)

    def test_shutdown_failure_keeps_tui(self, app):
        class DownApi(FakeApi):
            def post(self, path, body=None, params=None):
                if path == "/system/shutdown":
                    raise t.ApiError("HTTP 503: 后端已停止")
                return super().post(path, body)
        a = FakeApp(DownApi())
        msg, style = a.exec_command("/shutdown")
        assert style == "err"
        assert a.exit_requested is False


class TestModelCommands:
    def test_switch_builds_body(self, app):
        msg, style = app.exec_command(
            "/switch qwen-1.8b --quant int8 --engine pytorch --compile")
        assert style == "ok"
        post = [c for c in app.api.calls if c[1] == "/models/switch"]
        assert post, "POST /models/switch 未发出"
        body = post[0][2]
        assert body == {"model_id": "qwen-1.8b", "quant_type": "int8",
                        "engine": "pytorch", "use_compile": True}

    def test_load_default_quant(self, app):
        msg, style = app.exec_command("/load qwen-1.8b")
        assert style == "ok"
        body = [c for c in app.api.calls if c[1] == "/models/load"][0][2]
        assert body["quant_type"] == "int4" and body["engine"] == "auto"

    def test_quant_uses_current_model_and_engine(self, app):
        msg, style = app.exec_command("/quant int8")
        assert style == "ok"
        body = [c for c in app.api.calls if c[1] == "/models/switch"][0][2]
        assert body["model_id"] == "qwen-1.8b"
        assert body["quant_type"] == "int8"
        assert body["engine"] == "pytorch"

    def test_engine_rejects_invalid(self, app):
        msg, style = app.exec_command("/engine onnx")
        assert style == "err"

    def test_engine_keeps_quant(self, app):
        msg, style = app.exec_command("/engine llama_cpp")
        assert style == "ok"
        body = [c for c in app.api.calls if c[1] == "/models/switch"][0][2]
        assert body["engine"] == "llama_cpp"
        assert body["quant_type"] == "int4"

    def test_model_not_loaded_hint(self, app):
        class NoModelApi(FakeApi):
            def get(self, path, params=None, with_log_token=False):
                if path == "/models/current":
                    return {"loaded": False}
                return super().get(path)
        msg, style = FakeApp(NoModelApi()).exec_command("/quant int8")
        assert style == "warn"
        assert "未加载" in msg


class TestDeviceAndClusterCommands:
    def test_gpu_list(self, app):
        msg, style = app.exec_command("/gpu")
        assert style == "ok"
        assert "共 2 块 GPU" in msg

    def test_gpu_select(self, app):
        msg, style = app.exec_command("/gpu 1")
        assert style == "ok"
        assert "Tesla T4" in msg
        assert ("POST", "/device/select-gpu", {"gpu_index": 1}) in app.api.calls

    def test_dist_toggle(self, app):
        msg, style = app.exec_command("/dist toggle")
        assert style == "ok"
        put = [c for c in app.api.calls
               if c[0] == "PUT" and c[1] == "/cluster/config/distributed-inference"]
        assert put[0][2] == {"enabled": True}

    def test_queue_strategy(self, app):
        msg, style = app.exec_command("/queue strategy fifo")
        assert style == "ok"
        assert ("POST", "/cluster/queue/strategy", {"strategy": "fifo"}) in app.api.calls

    def test_queue_cancel(self, app):
        msg, style = app.exec_command("/queue cancel abc")
        assert style == "ok"
        assert ("DELETE", "/cluster/queue/task/abc") in app.api.calls

    def test_connect(self, app):
        msg, style = app.exec_command("/connect 100.64.0.1 8888")
        assert style == "ok"
        body = [c for c in app.api.calls if c[1] == "/cluster/connect"][0][2]
        assert body == {"master_host": "100.64.0.1", "master_port": 8888,
                        "switch_to_client": False}


class TestSettingsCommands:
    def test_host(self, app):
        msg, style = app.exec_command("/host 100.64.0.8 9000")
        assert style == "ok"
        assert app.api.host == "100.64.0.8" and app.api.port == 9000

    def test_interval_clamped(self, app):
        app.exec_command("/interval 99")
        assert app.interval == 60.0

    def test_timeout_clamped(self, app):
        app.exec_command("/timeout 999")
        assert app.api.timeout == 120.0

    def test_token(self, app):
        app.exec_command("/token secret")
        assert app.api.log_token == "secret"


class TestScreenAndMisc:
    def test_screen_by_number(self, app):
        msg, style = app.exec_command("/screen 3")
        assert style == "ok"
        assert "分布式" in msg

    def test_screen_by_name(self, app):
        msg, style = app.exec_command("/screen 日志")
        assert style == "ok"
        assert "日志查看" in msg

    def test_screen_invalid(self, app):
        msg, style = app.exec_command("/screen 99")
        assert style == "err"

    def test_cancel_falls_back_to_workflow(self, app):
        msg, style = app.exec_command("/cancel g-1")
        assert style == "ok"
        assert "工作流" in msg

    def test_chat_clear(self, app):
        msg, style = app.exec_command("/chat clear")
        assert style == "ok"
        assert ("POST", "/chat/clear", None) in app.api.calls


class TestStatusRefreshScreen:
    """/status、/refresh、/screen 无参数分支"""

    def test_status_opens_screen1(self, app):
        msg, style = app.exec_command("/status")
        assert style == "ok"
        assert "系统状态总览" in msg

    def test_status_alias_st(self, app):
        msg, style = app.exec_command("/st")
        assert style == "ok"
        assert "系统状态总览" in msg

    def test_refresh_no_current_screen_warn(self, app):
        # plain 模式 open_screen 不设置 current → 恒 warn 分支
        msg, style = app.exec_command("/refresh")
        assert style == "warn"
        assert "不在任何屏幕" in msg

    def test_screen_no_args_usage_warn(self, app):
        # /screen 注册了 min_args=1：无参数走参数校验分支（cmd_screen 内部
        # 的无参数列表分支为不可达代码，行为以 min_args 为准）
        msg, style = app.exec_command("/screen")
        assert style == "warn"
        assert "参数不足" in msg


class TestModelInfoCommands:
    """/model、/models、/presets 只读信息命令"""

    def test_model_loaded_details(self, app):
        msg, style = app.exec_command("/model")
        assert style == "ok"
        assert "Qwen" in msg
        assert "int4" in "\n".join(app.out)

    def test_model_not_loaded_warn(self, app):
        class NoModelApi(FakeApi):
            def get(self, path, params=None, with_log_token=False):
                if path == "/models/current":
                    return {"loaded": False}
                return super().get(path)
        msg, style = FakeApp(NoModelApi()).exec_command("/model")
        assert style == "warn"
        assert "未加载" in msg

    def test_models_list(self, app):
        msg, style = app.exec_command("/models")
        assert style == "ok"
        assert "共 1 个模型配置" in msg
        assert "qwen-1.8b" in "\n".join(app.out)
        assert "INT4" in "\n".join(app.out)  # /models/available 选项

    def test_presets(self, app):
        msg, style = app.exec_command("/presets")
        assert style == "ok"
        assert "共 2 个预设" in msg
        out = "\n".join(app.out)
        assert "自我介绍" in out and "预估" in out


class TestDeviceAndNodesCommands:
    """/device（auto/profile/非法）、/nodes"""

    def test_device_auto(self, app):
        msg, style = app.exec_command("/device auto")
        assert style == "ok"
        assert "自动配置完成" in msg

    def test_device_profile(self, app):
        msg, style = app.exec_command("/device profile")
        assert style == "ok"
        assert "laptop" in msg
        assert "58.8" in "\n".join(app.out)  # score_total
        assert "推荐 INT4" in "\n".join(app.out)  # recommendations

    def test_device_invalid_subcommand(self, app):
        msg, style = app.exec_command("/device bogus")
        assert style == "err"
        assert "用法" in msg

    def test_nodes(self, app):
        msg, style = app.exec_command("/nodes")
        assert style == "ok"
        assert "节点 1 个" in msg
        assert "n1" in "\n".join(app.out)

    def test_nodes_shows_role(self, app):
        msg, style = app.exec_command("/nodes")
        assert "从节点" in "\n".join(app.out)  # FakeApi /cluster/my-role → client


class TestLogCommands:
    """/logs（行数校验）、/log（filter/token/用法）"""

    def test_logs_default(self, app):
        msg, style = app.exec_command("/logs")
        assert style == "ok"
        assert "日志查看" in msg

    def test_logs_lines(self, app):
        msg, style = app.exec_command("/logs 50")
        assert style == "ok"

    def test_logs_invalid_lines(self, app):
        msg, style = app.exec_command("/logs abc")
        assert style == "err"
        assert "行数无效" in msg

    def test_log_filter(self, app):
        msg, style = app.exec_command("/log filter ERROR")
        assert style == "ok"
        assert "ERROR" in msg

    def test_log_filter_all(self, app):
        msg, style = app.exec_command("/log filter")
        assert style == "ok"
        assert "全部" in msg

    def test_log_filter_invalid_level(self, app):
        msg, style = app.exec_command("/log filter TRACE")
        assert style == "err"
        assert "级别无效" in msg

    def test_log_token(self, app):
        msg, style = app.exec_command("/log token secret")
        assert style == "ok"
        assert app.api.log_token == "secret"

    def test_log_token_clear(self, app):
        app.api.log_token = "old"
        msg, style = app.exec_command("/log token")
        assert style == "ok"
        assert app.api.log_token == ""

    def test_log_usage(self, app):
        msg, style = app.exec_command("/log bogus")
        assert style == "err"
        assert "用法" in msg
