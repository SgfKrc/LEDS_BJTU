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
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from koakuma_engine import (
    BackendCapabilities,
    backend_capabilities,
    backend_id_for,
    select_backend,
)


class InferenceHost(Protocol):
    """推理宿主协议：行为由 model_module.ModelManager 提供（阶段 1 由
    inference-svc 以 HTTP 契约实现同构接口）。"""

    def select_engine(self, profile): ...

    @property
    def engine_type(self) -> str: ...

    @property
    def backend_id(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def supports(self, capability: str) -> bool: ...

    def load_model(self, engine, quant_type, use_compile, model_id): ...

    def unload_model(self): ...

    def load_layer_range(self, layer_range, embed, lm_head): ...

    def forward_layers(self, layer_range, hidden, past_key_values, **kw): ...

    def chat(self, messages, **kw): ...

    def chat_image(self, image_path, prompt, **kw): ...

    def chat_stream(self, messages, **kw): ...

    def native_vision_available(self) -> bool: ...

    def ensure_full_model(self): ...

    @property
    def scheduler_callbacks(self) -> Optional["SchedulerCallbacks"]: ...


class SchedulerCallbacks(Protocol):
    """Explicit callbacks required by scheduler control-plane features."""

    active_task_graph_model_identity: Callable[[], Any]
    execute_task_worker_stage: Callable[[Any, Any], dict]
    build_model_chat_prompt: Callable[..., str]
    thinking_system_prompt: str
    snapshot_recent_logs: Callable[[], tuple[list[dict], int]]
    filter_recent_logs: Callable[..., list[dict]]
    format_model_response: Callable[..., tuple[str, Optional[str]]]


@dataclass(frozen=True)
class SchedulerCallbackSet:
    """Immutable scheduler dependency bundle assembled by the API composition root."""

    active_task_graph_model_identity: Callable[[], Any]
    execute_task_worker_stage: Callable[[Any, Any], dict]
    build_model_chat_prompt: Callable[..., str]
    thinking_system_prompt: str
    snapshot_recent_logs: Callable[[], tuple[list[dict], int]]
    filter_recent_logs: Callable[..., list[dict]]
    format_model_response: Callable[..., tuple[str, Optional[str]]]


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


# ModelHost owns these runtime attributes rather than proxying them to a manager.
_OWN_ATTRS = {
    "_manager", "model_loaded", "generation_config", "current_quant",
    "full_chat_execution_lock", "scheduler_callbacks",
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
        object.__setattr__(self, "generation_config", {
            "max_new_tokens": 1024,          # laptop 档默认值
            "tier_max_new_tokens": 1024,     # 设备档位上限（auto_configure 后更新）
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        })
        object.__setattr__(self, "full_chat_execution_lock", threading.RLock())
        object.__setattr__(self, "scheduler_callbacks", None)

    @property
    def engine_type(self) -> str:
        """Public backend identity; implementation details stay behind the host."""

        own_id = object.__getattribute__(self, "__dict__").get("_engine_type")
        if own_id:
            return str(own_id)
        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            manager = object.__getattribute__(manager, "_instance")
            if manager is None:
                return ""
        return backend_id_for(manager, default="")

    @property
    def backend_id(self) -> str:
        return self.engine_type

    @property
    def capabilities(self) -> BackendCapabilities:
        return backend_capabilities(self.engine_type)

    def supports(self, capability: str) -> bool:
        return self.capabilities.supports(capability)

    def select_engine(self, profile: dict | None = None) -> str:
        """Select the node backend through the Koakuma boundary."""

        try:
            import config as _cfg

            requested = getattr(_cfg, "INFERENCE_ENGINE", "auto")
            island_enabled = bool(getattr(_cfg, "ISLAND_ENABLED", False))
            island_base_url = str(getattr(_cfg, "ISLAND_BASE_URL", "") or "")
        except Exception:
            requested = "auto"
            island_enabled = False
            island_base_url = ""
        return select_backend(
            profile,
            requested=requested,
            island_enabled=island_enabled,
            island_base_url=island_base_url,
        )

    def configure_scheduler_callbacks(self, callbacks: SchedulerCallbacks) -> None:
        """Install the scheduler's complete, typed callback contract atomically."""

        if callbacks is None:
            raise TypeError("scheduler callbacks are required")
        object.__setattr__(self, "scheduler_callbacks", callbacks)

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

    def runtime_status(self) -> dict:
        """Return the public, lazy-safe runtime state for diagnostics and tests."""

        return {
            "manager_loaded": self.peek_manager() is not None,
            "model_loaded": self.has_loaded_model(),
            "current_quant": object.__getattribute__(self, "current_quant"),
        }

    # ------------------------------------------------------------ engine dispatch
    def load_model(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        model_id: str = None,
        engine: str = None,
        db_experimental_models: list = None,
    ) -> None:
        """Load a model, routing the GGUF engine *around* the torch-backed manager.

        The GGUF/llama.cpp engine has its own implementation (``llama_engine``) and needs
        no PyTorch, so it is loaded directly here and installed as this host's manager.
        Every other engine keeps using ModelManager unchanged.

        Routing GGUF through ModelManager would import ``model_module`` -- which imports
        torch at module scope -- leaving the L-tier (no-torch) environment unable to load
        a model at all.  Argument order matches ModelManager.load_model, so existing
        positional and keyword callers keep working.
        """

        if str(engine or "").strip().lower() in ("llama_cpp", "llama.cpp", "llama-cpp", "gguf"):
            self._load_gguf_model(model_path=model_path, model_id=model_id, profile=profile)
            return
        self._manager.load_model(
            model_path=model_path,
            quant_type=quant_type,
            profile=profile,
            model_id=model_id,
            engine=engine,
            db_experimental_models=db_experimental_models,
        )

    def _load_gguf_model(self, *, model_path: str = None, model_id: str = None, profile: dict = None) -> None:
        """Load a GGUF file via llama_engine and make it this host's manager."""

        import os

        from llama_engine import LlamaCppEngine, get_gguf_model_path

        try:
            from config import GGUF_MODEL_PATH
        except Exception:  # pragma: no cover - config defines it in practice
            GGUF_MODEL_PATH = ""

        if model_path and str(model_path).endswith(".gguf"):
            gguf_path = model_path
        elif GGUF_MODEL_PATH and os.path.isfile(GGUF_MODEL_PATH):
            gguf_path = GGUF_MODEL_PATH
        else:
            gguf_path = get_gguf_model_path()
        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF 模型文件未找到: {gguf_path or '(未解析到路径)'}")

        resolved_id = model_id
        if not resolved_id:
            try:
                from model_config import get_builtin_models, resolve_model_path

                absolute_path = os.path.abspath(gguf_path)
                for candidate in get_builtin_models():
                    if candidate.gguf_path and os.path.abspath(
                        resolve_model_path(candidate.gguf_path),
                    ) == absolute_path:
                        resolved_id = candidate.model_id
                        break
            except Exception:
                resolved_id = None

        model_metadata = {}
        if resolved_id:
            try:
                from model_config import get_model_profile_metadata

                model_metadata = get_model_profile_metadata(resolved_id)
            except Exception:
                model_metadata = {}

        tier = (profile or {}).get("tier", "laptop")
        n_ctx = {"edge": 1024, "mobile": 512, "ultrabook": 2048}.get(tier, 4096)

        engine_obj = LlamaCppEngine()
        engine_obj.load_model(
            model_path=gguf_path,
            n_ctx=n_ctx,
            model_profile=model_metadata,
        )
        # Install the engine as the manager: chat/chat_stream/is_loaded are then proxied
        # to it unchanged (their signatures already match ModelManager's).
        object.__setattr__(self, "_manager", engine_obj)
        object.__setattr__(self, "model_loaded", True)
        # Callers (api_server.get_status, scheduler) read these manager-side fields, so
        # mirror them here.  __getattr__ only fires when the instance dict misses, so
        # setting them on the instance keeps the proxy working normally otherwise.
        resolved_id = resolved_id or gguf_path
        object.__setattr__(self, "active_model_id", resolved_id)
        object.__setattr__(self, "_active_model_id", resolved_id)
        object.__setattr__(self, "_engine_type", "llama_cpp")
        object.__setattr__(self, "quant_type", "gguf")
        object.__setattr__(self, "model_path", gguf_path)
        try:
            import config as _cfg
            object.__setattr__(self, "current_quant", str(getattr(_cfg, "QUANT_TYPE", "int4")))
        except Exception:  # pragma: no cover
            pass

    def _gguf_available(self) -> bool:
        """Whether a GGUF file can be resolved *without* importing torch."""

        import os

        try:
            from config import GGUF_MODEL_PATH
            from llama_engine import get_gguf_model_path
        except Exception:  # pragma: no cover - llama_engine is import-safe
            return False
        if GGUF_MODEL_PATH and os.path.isfile(GGUF_MODEL_PATH):
            return True
        return bool(get_gguf_model_path())

    def switch_model(
        self,
        model_id: str = None,
        quant_type: str = None,
        profile: dict = None,
        engine: str = None,
        model_path: str = None,
        db_experimental_models: list = None,
    ) -> dict:
        """Switch models, keeping the GGUF engine on the torch-free path.

        ``api_server`` calls ``switch_model`` (for its rollback protection), not
        ``load_model``, so the GGUF routing has to live here too.  ``engine=None``/"auto"
        asks the backend to decide: when a GGUF file resolves *and* torch is absent, GGUF
        is the only engine that can actually run, so prefer it.  With torch present the
        D-tier default is kept exactly as before.
        """

        import importlib.util

        requested = str(engine or "").strip().lower()
        wants_gguf = requested in ("llama_cpp", "llama.cpp", "llama-cpp", "gguf")
        if not wants_gguf and requested in ("", "auto"):
            torch_absent = importlib.util.find_spec("torch") is None
            wants_gguf = torch_absent and self._gguf_available()
        if wants_gguf:
            self._load_gguf_model(model_path=model_path, model_id=model_id, profile=profile)
            # Mirror ModelManager.switch_model's contract -- api_server reads "success".
            return {
                "success": True,
                "model_id": model_id,
                "model_name": model_id,
                "error": None,
            }
        return self._manager.switch_model(
            model_id=model_id,
            quant_type=quant_type,
            profile=profile,
            engine=engine,
            model_path=model_path,
            db_experimental_models=db_experimental_models,
        )

    def unload_model(self) -> None:
        """Unload whichever engine is active -- GGUF (direct) or ModelManager."""

        manager = object.__getattribute__(self, "_manager")
        if isinstance(manager, _LazyModelManager):
            manager = object.__getattribute__(manager, "_instance")
        if manager is None:
            pass
        elif hasattr(manager, "unload_model"):  # ModelManager
            manager.unload_model()
        elif hasattr(manager, "unload"):  # LlamaCppEngine
            manager.unload()
        object.__setattr__(self, "model_loaded", False)

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
model_host = ModelHost()


def get_model_host() -> ModelHost:
    """返回全局单例（显式获取入口，避免 from-import 时绑定旧实例）。"""
    return model_host
