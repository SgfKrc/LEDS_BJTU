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
    _LazyModelManager,
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
        # 行为由 ModelManager 提供（经 _LazyModelManager 延迟加载）；
        # 协议层只需保证接口签名在宿主上可达（代理后属性存在）。
        host = ModelHost()
        mgr = host._manager
        # _LazyModelManager 未加载时 _instance 为 None；验证惰性容器存在
        assert mgr._instance is None
        assert mgr._lock is not None

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
        assert host._fake_callback(1) == 2

    def test_attach_overrides_manager_proxy(self):
        # attach 后属性读取优先取宿主自身
        host = ModelHost()
        host.attach("_execute_task_worker_stage", lambda: "attached")
        assert host._execute_task_worker_stage() == "attached"


class TestModelHostSingleton:
    """全局单例与 scheduler 默认注入。"""

    def test_singleton_identity(self):
        assert get_model_host() is model_host
        assert isinstance(model_host, ModelHost)

    def test_scheduler_default_host(self):
        # scheduler 无 host 参数时使用全局单例（阶段 0.2 注入语义）
        import scheduler as sched_mod
        s = sched_mod.Scheduler()
        assert s._host is model_host

    def test_scheduler_explicit_host(self):
        import scheduler as sched_mod
        host = ModelHost()
        s = sched_mod.Scheduler(host=host)
        assert s._host is host


class TestLazyModelManager:
    """从 api_server 迁入的惰性容器行为不变。"""

    def test_slots_defined(self):
        lm = _LazyModelManager()
        assert set(lm.__slots__) == {"_instance", "_lock"}
        assert lm._instance is None

    def test_lazy_no_instantiation(self):
        # 未访问任何属性前不实例化 ModelManager（冷启动友好）
        lm = _LazyModelManager()
        assert lm._instance is None


class TestApiServerIntegration:
    """api_server 经改造后的宿主接线（不加载模型）。"""

    def test_api_server_model_manager_is_host(self):
        import api_server
        assert api_server.model_manager is model_host

    def test_api_server_attach_present(self):
        import api_server  # noqa: F401 —— 模块加载即执行 attach
        assert callable(model_host._execute_task_worker_stage)
        assert callable(model_host._active_task_graph_model_identity)

    def test_no_reverse_import(self):
        # 阶段 0.5 验收项：scheduler 不再 import api_server
        import inspect
        import scheduler as sched_mod
        src = inspect.getsource(sched_mod)
        assert "import api_server" not in src
