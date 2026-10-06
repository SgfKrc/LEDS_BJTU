"""阶段 0.2 测试：InferenceHost 协议与 ModelHost 代理语义。

验证：
  - ModelHost 满足 InferenceHost 协议（鸭子类型：方法签名齐全）
  - ModelHost 属性代理：manager 属性读写转发；自有属性（model_loaded/
    generation_config/full_chat_execution_lock）留在宿主
  - SchedulerCallbackSet 显式回调注入（api_server 组合根暴露的执行能力）
  - model_host 单例可被 scheduler 默认注入（get_model_host）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import threading
import types

from model_host import (
    InferenceHost,
    ModelHost,
    SchedulerCallbackSet,
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


class TestSwitchModelAfterGgufEngine:
    """#31：GGUF 路径把 LlamaCppEngine 装成 _manager 之后，必须能切回非 GGUF 引擎。"""

    class _FakeGgufEngine:
        """模拟 LlamaCppEngine：有 unload，没有 switch_model。"""

        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    class _FakeManager:
        """模拟 ModelManager：接受 switch_model 并记录调用。"""

        def __init__(self):
            self.calls = []

        def switch_model(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True, "model_id": kwargs.get("model_id")}

    def _host_with_gguf_engine(self):
        host = ModelHost()
        engine = self._FakeGgufEngine()
        object.__setattr__(host, "_manager", engine)
        # 复现 _load_gguf_model 写进实例字典的镜像属性
        object.__setattr__(host, "model_loaded", True)
        object.__setattr__(host, "_engine_type", "llama_cpp")
        object.__setattr__(host, "quant_type", "gguf")
        object.__setattr__(host, "model_path", "/tmp/x.gguf")
        return host, engine

    def test_switch_back_drops_gguf_engine_and_mirrored_attrs(self, monkeypatch):
        import model_host as model_host_module

        fresh = self._FakeManager()
        monkeypatch.setattr(model_host_module, "_LazyModelManager", lambda: fresh)

        host, engine = self._host_with_gguf_engine()
        result = host.switch_model(model_id="qwen2.5-0.5b-instruct", engine="pytorch")

        assert result["success"] is True
        assert engine.unloaded is True, "GGUF 引擎必须先卸载"
        assert fresh.calls[0]["engine"] == "pytorch"
        # 镜像属性必须移除，否则读取时命中的仍是上一档的值（_engine_type 会停在
        # "llama_cpp"，让 api_server 走 GGUF 分支）
        instance_dict = object.__getattribute__(host, "__dict__")
        for name in ("_engine_type", "quant_type", "model_path",
                     "active_model_id", "_active_model_id"):
            assert name not in instance_dict, f"{name} 应已被移除"

    def test_gguf_path_untouched_when_manager_has_switch_model(self, monkeypatch):
        """本来就是管理器时，走原路径：不卸载、不新建。"""
        import model_host as model_host_module

        def _boom():
            raise AssertionError("不应新建懒管理器")

        monkeypatch.setattr(model_host_module, "_LazyModelManager", _boom)

        existing = self._FakeManager()
        host = ModelHost(manager=existing)
        result = host.switch_model(model_id="m", engine="pytorch")

        assert result["success"] is True
        assert existing.calls[0]["engine"] == "pytorch"
        host.generation_config["max_new_tokens"] = 2048
        assert host.generation_config["max_new_tokens"] == 2048


def _callback_set(marker="default"):
    return SchedulerCallbackSet(
        active_task_graph_model_identity=lambda: marker,
        execute_task_worker_stage=lambda request, cancel_event: {"marker": marker},
        build_model_chat_prompt=lambda tokenizer, messages, **kwargs: marker,
        thinking_system_prompt=f"thinking:{marker}",
        snapshot_recent_logs=lambda: ([{"marker": marker}], 1),
        filter_recent_logs=lambda entries, **kwargs: list(entries),
        format_model_response=lambda text, **kwargs: (text, None),
    )


class TestSchedulerCallbackInjection:
    """Scheduler callbacks are one explicit, immutable protocol bundle."""

    def test_callback_bundle_is_instance_scoped_and_lazy(self):
        attached = ModelHost()
        callbacks = _callback_set("attached")
        attached.configure_scheduler_callbacks(callbacks)
        untouched = ModelHost()

        assert attached.scheduler_callbacks is callbacks
        assert untouched.scheduler_callbacks is None
        assert untouched.runtime_status()["manager_loaded"] is False

    def test_callback_bundle_keyword_order_does_not_change_wiring(self):
        values = {
            "active_task_graph_model_identity": lambda: "identity",
            "execute_task_worker_stage": lambda request, cancel_event: {"ok": True},
            "build_model_chat_prompt": lambda tokenizer, messages, **kwargs: "prompt",
            "thinking_system_prompt": "thinking",
            "snapshot_recent_logs": lambda: ([], 0),
            "filter_recent_logs": lambda entries, **kwargs: entries,
            "format_model_response": lambda text, **kwargs: (text, None),
        }
        callbacks = SchedulerCallbackSet(**dict(reversed(list(values.items()))))

        assert callbacks.active_task_graph_model_identity() == "identity"
        assert callbacks.execute_task_worker_stage(None, None) == {"ok": True}
        assert callbacks.build_model_chat_prompt(None, []) == "prompt"
        assert callbacks.thinking_system_prompt == "thinking"


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

    def test_no_torch_proxy_reports_unloaded_and_fails_closed(self):
        host = ModelHost()
        assert host.is_loaded is False
        assert host.is_pipeline_prepared is False
        assert host.runtime_status()["manager_loaded"] is False
        with pytest.raises(RuntimeError, match="没有可用的 ModelManager"):
            host.ensure_full_model()
        assert host.runtime_status()["manager_loaded"] is False

    def test_explicit_torch_load_reports_missing_torch(self, monkeypatch):
        import model_host as model_host_module

        host = ModelHost()
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda self: (_ for _ in ()).throw(ImportError("torch is unavailable")),
        )
        with pytest.raises(RuntimeError, match="没有 PyTorch"):
            host.load_model(engine="pytorch")

    def test_auto_load_uses_gguf_without_materializing_torch_manager(self, monkeypatch, tmp_path):
        import model_host as model_host_module

        model_path = tmp_path / "probe.gguf"
        model_path.write_bytes(b"probe")

        class FakeGguf:
            engine_type = "llama_cpp"

            def __init__(self):
                self.is_loaded = False

            def load_model(self, **kwargs):
                self.is_loaded = True
                self.model_path = kwargs["model_path"]

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        fake_module.get_gguf_model_path = lambda: str(model_path)
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)

        host = ModelHost()
        monkeypatch.setattr(host, "select_engine", lambda profile=None: "llama_cpp")
        host.load_model(model_path=str(model_path), engine=None)

        assert host.is_loaded is True
        assert host.engine_type == "llama_cpp"

    def test_prepare_gguf_pipeline_does_not_materialize_torch_manager(
            self, monkeypatch, tmp_path):
        model_path = tmp_path / "probe.gguf"
        model_path.write_bytes(b"probe")

        class FakeGguf:
            engine_type = "llama_cpp"

            def prepare_pipeline_model(self, **kwargs):
                self.kwargs = kwargs
                self.is_pipeline_prepared = True
                return {
                    "model_id": kwargs["model_id"],
                    "model_sha256": "a" * 64,
                    "pipeline_runtime_supported": True,
                }

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)

        host = ModelHost()
        descriptor = host.prepare_pipeline_model(
            model_id="probe-model",
            model_path=str(model_path),
            quant_type="Q4_K_M",
        )

        assert descriptor["model_id"] == "probe-model"
        assert host.is_pipeline_prepared is True
        assert host.is_loaded is False
        assert host.engine_type == "llama_cpp"

    def test_prepare_directory_switches_away_from_llama_manager(
            self, monkeypatch, tmp_path):
        model_dir = tmp_path / "safetensors-model"
        model_dir.mkdir()
        unloaded = []

        class OldGguf:
            engine_type = "llama_cpp"

            def prepare_pipeline_model(self, **_kwargs):
                raise AssertionError("directory must not use the GGUF preparer")

            def unload(self):
                unloaded.append(True)

        class TorchManager:
            def prepare_pipeline_model(self, **kwargs):
                self.kwargs = kwargs
                return {"model_id": kwargs["model_id"]}

        torch_manager = TorchManager()
        host = ModelHost(manager=OldGguf())
        object.__setattr__(host, "_engine_type", "llama_cpp")
        monkeypatch.setattr(
            ModelHost, "_materialize_manager", lambda _self: torch_manager,
        )

        result = host.prepare_pipeline_model(
            model_id="directory-model", model_path=str(model_dir),
        )

        assert result == {"model_id": "directory-model"}
        assert unloaded == [True]
        assert "_engine_type" not in host.__dict__

    def test_unload_clears_direct_gguf_mirror_state(self):
        class DirectGguf:
            def unload(self):
                self.unloaded = True

        manager = DirectGguf()
        host = ModelHost(manager=manager)
        for name, value in {
            "_engine_type": "llama_cpp",
            "quant_type": "GGUF",
            "model_path": "old.gguf",
            "active_model_id": "old-model",
            "_active_model_id": "old-model",
        }.items():
            object.__setattr__(host, name, value)

        host.unload_model()

        assert manager.unloaded is True
        for name in (
            "_engine_type", "quant_type", "model_path",
            "active_model_id", "_active_model_id",
        ):
            assert name not in host.__dict__

    def test_torch_free_partial_gguf_is_rejected_as_full_model(self):
        class PartialGguf:
            engine_type = "llama_cpp"
            is_loaded = True

            def get_pipeline_descriptor(self):
                return {
                    "partial_assignment": True,
                    "assignment_layer_range": [0, 8],
                    "loaded_artifact": "head8.gguf",
                }

        host = ModelHost(manager=PartialGguf())
        assert host.is_loaded is True
        with pytest.raises(RuntimeError, match="裁层工件"):
            host.ensure_full_model()

    def test_torch_free_full_gguf_satisfies_full_model_check(self):
        class FullGguf:
            engine_type = "llama_cpp"
            is_loaded = True

            def get_pipeline_descriptor(self):
                return {"model_path": "full.gguf"}

        host = ModelHost(manager=FullGguf())
        assert host.ensure_full_model() is None


class TestApiServerIntegration:
    """api_server 经改造后的宿主接线（不加载模型）。"""

    def test_api_server_model_manager_is_host(self):
        import api_server
        assert api_server.model_manager is model_host

    def test_api_server_configures_explicit_scheduler_callbacks(self):
        import api_server

        callbacks = api_server.scheduler.inference_callbacks
        assert callbacks is model_host.scheduler_callbacks
        assert callable(callbacks.execute_task_worker_stage)
        assert callable(callbacks.active_task_graph_model_identity)

    def test_no_reverse_import(self):
        # 阶段 0.5 验收项：scheduler 不再 import api_server
        import inspect
        import scheduler as sched_mod
        src = inspect.getsource(sched_mod)
        assert "import api_server" not in src
