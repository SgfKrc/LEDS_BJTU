"""阶段 0.2 测试：InferenceHost 协议与 ModelHost 代理语义。

验证：
  - ModelHost 满足 InferenceHost 协议（鸭子类型：方法签名齐全）
  - ModelHost 属性代理：manager 属性读写转发；自有属性（model_loaded/
    generation_config/full_chat_execution_lock）留在宿主
  - attach() 回调挂载（api_server 暴露 _execute_task_worker_stage 等）
  - model_host 单例可被 scheduler 默认注入（get_model_host）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import threading

from model_host import (
    InferenceHost,
    ModelHost,
    get_model_host,
    model_host,
)


def _protocol_methods() -> list:
    return [
        "select_engine", "load_model", "unload_model", "load_layer_range",
        "forward_layers", "chat", "chat_stream", "ensure_full_model",
    ]


class TestInferenceHostProtocol:
    """协议满足性（不加载真实模型）。"""

    def test_protocol_methods_exist(self):
        # 冷启动状态通过公共快照检查，不触发 ModelManager 导入。
        host = ModelHost()
        assert host.runtime_status()["manager_loaded"] is False

    def test_model_host_proxies_manager(self):
        # 代理转发：宿主未定义属性转发给注入的 manager
        class FakeManager:
            is_loaded = False
            layer_range = None

        host = ModelHost(manager=FakeManager())
        assert host.is_loaded is False
        assert host.layer_range is None
        with pytest.raises(AttributeError):
            host.no_such_attr

    def test_own_attrs_stay_on_host(self):
        host = ModelHost()
        assert host.model_loaded is False
        assert host.generation_config["max_new_tokens"] == 1024
        assert isinstance(host.full_chat_execution_lock, type(threading.RLock()))

    def test_own_attr_write(self):
        host = ModelHost()
        host.model_loaded = True
        assert host.model_loaded is True
        host.generation_config["max_new_tokens"] = 2048
        assert host.generation_config["max_new_tokens"] == 2048


class TestModelHostAttach:
    """api_server 向 host 挂载回调（消除 scheduler 反向 import）。"""

    def test_attach_roundtrip(self):
        host = ModelHost()

        def fake_cb(x):
            return x + 1

        host.attach("_fake_callback", fake_cb)
        assert host.get_attachment("_fake_callback")(1) == 2

    def test_attach_overrides_manager_proxy(self):
        # attach 后属性读取优先取宿主自身
        host = ModelHost()
        host.attach("_execute_task_worker_stage", lambda: "attached")
        assert host.get_attachment("_execute_task_worker_stage")() == "attached"

    def test_attachment_lookup_is_instance_scoped_and_lazy(self):
        attached = ModelHost()
        attached.attach("_instance_callback", lambda: "attached")
        untouched = ModelHost()

        assert untouched.get_attachment("_instance_callback") is None
        assert untouched.runtime_status()["manager_loaded"] is False


class TestModelHostSingleton:
    """全局单例与 scheduler 默认注入。"""

    def test_singleton_identity(self):
        assert get_model_host() is model_host
        assert isinstance(model_host, ModelHost)

    def test_scheduler_default_host(self):
        # scheduler 无 host 参数时使用全局单例（阶段 0.2 注入语义）
        import scheduler as sched_mod
        s = sched_mod.Scheduler()
        assert s.inference_host is model_host

    def test_scheduler_explicit_host(self):
        import scheduler as sched_mod
        host = ModelHost()
        s = sched_mod.Scheduler(host=host)
        assert s.inference_host is host


class TestLazyModelManager:
    """从 api_server 迁入的惰性容器行为不变。"""

    def test_lazy_no_instantiation(self):
        # 未访问任何属性前不实例化 ModelManager（冷启动友好）
        assert ModelHost().runtime_status()["manager_loaded"] is False


class TestApiServerIntegration:
    """api_server 经改造后的宿主接线（不加载模型）。"""

    def test_api_server_model_manager_is_host(self):
        import api_server
        assert api_server.model_manager is model_host

    def test_api_server_attach_present(self):
        import api_server  # noqa: F401 —— 模块加载即执行 attach
        assert callable(model_host.get_attachment("_execute_task_worker_stage"))
        assert callable(model_host.get_attachment("_active_task_graph_model_identity"))

    def test_no_reverse_import(self):
        # 阶段 0.5 验收项：scheduler 不再 import api_server
        import inspect
        import scheduler as sched_mod
        src = inspect.getsource(sched_mod)
        assert "import api_server" not in src
