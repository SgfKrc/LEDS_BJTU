"""推理宿主（ModelHost）—— 阶段 0 边界治理（0.2/0.4）

统一持有：
  - ModelManager 实例（经 _LazyModelManager 延迟导入 model_module，
    避免冷启动 import 全量加载——原 api_server.py:113-155 迁入）
  - 模型运行时可变状态：model_loaded / generation_config
  - 推理执行锁：full_chat_execution_lock（原 api_server._full_chat_execution_lock）

api_server 与 scheduler 共享同一 model_host 单例，消除 scheduler 对
api_server 的运行时反向 import（scheduler.py 中 12 处 `import api_server`）。

属性代理：ModelHost 上未显式定义的属性转发给内部 manager（ModelManager），
调用方 `host.forward_layers(...)`、`host.chat(...)` 与直接使用
ModelManager 等价。
"""
import threading
from typing import Any, Optional, Protocol


class InferenceHost(Protocol):
    """推理宿主协议：行为由 model_module.ModelManager 提供（阶段 1 由
    inference-svc 以 HTTP 契约实现同构接口）。"""

    def select_engine(self, profile): ...

    def load_model(self, engine, quant_type, use_compile, model_id): ...

    def unload_model(self): ...

    def load_layer_range(self, layer_range, embed, lm_head): ...

    def forward_layers(self, layer_range, hidden, past_key_values, **kw): ...

    def chat(self, messages, **kw): ...

    def chat_stream(self, messages, **kw): ...

    def ensure_full_model(self): ...


class _LazyModelManager:
    """Delay importing model_module until the model manager is first used."""

    __slots__ = ("_instance", "_lock")
    _instance: Any
    _lock: Any

    def __init__(self):
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.RLock())

    def _get_instance(self):
        instance = self._instance
        if instance is not None:
            return instance
        with self._lock:
            instance = self._instance
            if instance is None:
                from model_module import ModelManager

                instance = ModelManager()
                object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._get_instance(), name, value)

    def __delattr__(self, name):
        if name in self.__slots__:
            raise AttributeError(name)
        delattr(self._get_instance(), name)

    def __repr__(self):
        instance = self._instance
        if instance is None:
            return "<_LazyModelManager unloaded>"
        return repr(instance)


# ModelHost 自身持有（不代理给 manager）的属性名（可用 attach 扩展）
_OWN_ATTRS = {
    "_manager", "model_loaded", "generation_config", "current_quant",
    "full_chat_execution_lock", "_db_available",
}


class ModelHost:
    """推理宿主：统一持有 ModelManager + 模型运行时状态 + 执行锁。

    属性代理：未在 _OWN_ATTRS 中的属性读写转发给内部 ModelManager。
    """

    def __init__(self, manager: Any = None):
        object.__setattr__(self, "_manager", manager if manager is not None else _LazyModelManager())
        object.__setattr__(self, "model_loaded", False)
        try:
            import config as _cfg
            _initial_quant = str(getattr(_cfg, "QUANT_TYPE", "int4"))
        except Exception:
            _initial_quant = "int4"
        object.__setattr__(self, "current_quant", _initial_quant)
        try:
            import db as _db_mod  # noqa: F401 —— 探测 psycopg2 可用性
            _db_importable = True
        except Exception:
            _db_importable = False
        object.__setattr__(self, "_db_available", _db_importable)
        object.__setattr__(self, "generation_config", {
            "max_new_tokens": 1024,          # laptop 档默认值
            "tier_max_new_tokens": 1024,     # 设备档位上限（auto_configure 后更新）
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        })
        object.__setattr__(self, "full_chat_execution_lock", threading.RLock())

    def attach(self, name: str, value: Any) -> None:
        """注册自有属性（存于 ModelHost 自身，不走 manager 代理）。

        供 api_server 在模块加载完成后挂载需向 scheduler 暴露的回调
        （如 _execute_task_worker_stage / _active_task_graph_model_identity），
        避免 scheduler 反向 import api_server（阶段 0.2）。
        """
        _OWN_ATTRS.add(name)
        object.__setattr__(self, name, value)

    def peek_manager(self) -> Any:
        """Return an already-created manager without triggering lazy import."""

        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            return object.__getattribute__(manager, "_instance")
        return manager

    def has_loaded_model(self) -> bool:
        """Check LLM ownership without materializing the lazy manager."""

        if bool(object.__getattribute__(self, "model_loaded")):
            return True
        manager = self.peek_manager()
        return bool(manager is not None and getattr(manager, "is_loaded", False))

    def __getattr__(self, name):
        # object.__getattribute__ 直接取 _manager，避免 _manager 被 del 后
        # 经 __getattr__ 访问自身导致无限递归
        mgr = object.__getattribute__(self, "_manager")
        if mgr is None:
            return None  # manager 缺失（测试注入 None）时模拟原 getattr(..., None) 语义
        return getattr(mgr, name)

    def __setattr__(self, name, value):
        if name in _OWN_ATTRS:
            object.__setattr__(self, name, value)
            return
        setattr(self._manager, name, value)

    def __delattr__(self, name):
        if name in _OWN_ATTRS:
            object.__delattr__(self, name)
            return
        delattr(self._manager, name)

    def __repr__(self):
        return "<ModelHost manager=%r model_loaded=%s>" % (
            self._manager, self.model_loaded)


# 全局单例：api_server 与 scheduler 共享（0.4）
model_host: InferenceHost = ModelHost()


def get_model_host() -> ModelHost:
    """返回全局单例（显式获取入口，避免 from-import 时绑定旧实例）。"""
    return model_host
