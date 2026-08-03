"""模型托管宿主（inference-svc 进程内）。

并行共存（§1.4）：本类在 inference-svc 独立进程内新建 ModelHost 实例，
与 api_server 进程内的 model_host 单例互不共享、互不影响。

1.1（本文件）：薄委托 ModelHost——其内部 _LazyModelManager 延迟 import
model_module，未加载模型时冷启动成本 <100ms（§2.3：model_module 子树
9.7s 全部发生在首次 load 请求时）。
1.2：把 api_server 数据面执行段（_execute_chat_full 等 8 函数）与
scheduler 流水线段（_run_pipeline 等）复制为本宿主方法，源文件保持
不动（复制迁移，禁改源）。
"""
import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .protocol import ChatRequest


class EngineHost:
    """推理宿主：进程内模型托管 + 执行段宿主 + generation 取消注册表。"""

    def __init__(self):
        # 延迟 import：保持模块顶层轻量（config 69ms 可接受，model_module 不在此触发）
        from model_host import ModelHost

        self._host: ModelHost = ModelHost()
        self._layers: List[str] = []  # 已加载层段（1.2 与 model 侧真实状态对齐）
        self._gen_lock = threading.RLock()
        self._generations: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------
    # 模型生命周期（委托 ModelHost / ModelManager）
    # ------------------------------------------------------------------
    def select_engine(self, profile=None) -> str:
        return self._host.select_engine(profile)

    def load_model(
        self,
        engine: Optional[str] = None,
        quant_type: Optional[str] = None,
        use_compile: bool = False,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self._host.load_model(
            engine=engine,
            quant_type=quant_type,
            use_compile=use_compile,
            model_id=model_id,
        )
        return result if isinstance(result, dict) else {"success": True, "data": result}

    def unload_model(self) -> Dict[str, Any]:
        self._host.unload_model()
        self._layers.clear()
        return {"success": True, "message": "模型已卸载"}

    def switch_model(self, model_id: str, engine: Optional[str] = None) -> Dict[str, Any]:
        result = self._host.switch_model(model_id=model_id, engine=engine)
        return result if isinstance(result, dict) else {"success": True, "data": result}

    def current_model(self) -> Dict[str, Any]:
        """当前模型信息（/v1/models/current）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        mgr = self._host
        if not loaded:
            return {"loaded": False}
        try:
            model_id = mgr.active_model_id() if callable(getattr(mgr, "active_model_id", None)) else None
        except Exception:
            model_id = None
        return {
            "loaded": True,
            "engine": getattr(mgr, "_engine_type", None),
            "model_id": model_id,
            "quant_type": getattr(self._host, "current_quant", None),
        }

    # ------------------------------------------------------------------
    # 层段接口（client 角色；master 角色本地层段同用）
    # ------------------------------------------------------------------
    def load_layer_range(
        self, layer_range: str, embed: bool = False, lm_head: bool = False
    ) -> Dict[str, Any]:
        """加载层段。layer_range 形如 "0-12"（[start, end)，对齐
        ModelManager.load_layer_range(start_layer, end_layer,
        has_embedding, has_lm_head) 语义。"""
        try:
            start_str, end_str = layer_range.split("-", 1)
            start_layer, end_layer = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"非法 layer_range: {layer_range!r}（期望如 '0-12'）")
        result = self._host.load_layer_range(
            start_layer=start_layer,
            end_layer=end_layer,
            has_embedding=embed,
            has_lm_head=lm_head,
        )
        if layer_range not in self._layers:
            self._layers.append(layer_range)
        return result if isinstance(result, dict) else {"success": True, "layer_range": layer_range}

    def unload_layer_range(self, layer_range: str) -> Dict[str, Any]:
        if layer_range in self._layers:
            self._layers.remove(layer_range)
        return {"success": True, "unloaded": layer_range}

    def forward_layers(
        self,
        layer_range: str,
        hidden,
        past_key_values=None,
        **kwargs,
    ):
        return self._host.forward_layers(
            layer_range=layer_range,
            hidden=hidden,
            past_key_values=past_key_values,
            **kwargs,
        )

    def embedding(self, input_ids):
        """Embedding 段（1.2 从 scheduler 流水线段复制真实实现）。"""
        raise NotImplementedError("embedding 段在 1.2 随流水线执行段复制接入")

    def lm_head(self, hidden):
        """LM Head 段（1.2 从 scheduler._run_master_lm_head 复制真实实现）。"""
        raise NotImplementedError("lm_head 段在 1.2 随流水线执行段复制接入")

    def ready(self) -> Dict[str, Any]:
        """/v1/ready：模型或层段就绪即 ready（冷启动方案 §5.4 语义）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        return {
            "ready": loaded or bool(self._layers),
            "model_loaded": loaded,
            "layers": list(self._layers),
        }

    def status(self) -> Dict[str, Any]:
        """/v1/status：引擎、当前模型、显存、层段。"""
        mgr = self._host
        loaded = bool(getattr(self._host, "model_loaded", False))
        model = getattr(mgr, "model", None)
        device = str(getattr(model, "device", "")) if model is not None else None
        try:
            model_id = mgr.active_model_id() if callable(getattr(mgr, "active_model_id", None)) else None
        except Exception:
            model_id = None
        return {
            "engine": getattr(mgr, "_engine_type", None),
            "model_id": model_id,
            "device": device,
            "model_loaded": loaded,
            "quant_type": getattr(self._host, "current_quant", None),
            "layers": list(self._layers),
        }

    # ------------------------------------------------------------------
    # 对话（1.1 薄实现：本地模型 chat/chat_stream；
    # 1.2 替换为 _execute_chat_full / fast 模式副本，含历史/追问/持久化）
    # ------------------------------------------------------------------
    def chat_full(self, req: ChatRequest) -> Dict[str, Any]:
        """完整对话响应（对齐 api_server /api/chat 响应形状）。"""
        messages = [{"role": "user", "content": req.message}]
        result = self._host.chat(
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        if isinstance(result, dict):
            response = result.get("content") or result.get("response") or str(result)
            metrics = {
                k: result[k]
                for k in ("tokens_per_second", "usage")
                if k in result
            }
        else:
            response = str(result)
            metrics = {}
        return {
            "response": response,
            "followups": [],
            "metrics": metrics,
            "request_id": "-",
        }

    def chat_stream_events(self, req: ChatRequest, cancel_event: Optional[threading.Event]):
        """SSE 事件序列（1.1 薄实现；1.2 替换为 fast 模式副本）。

        Yields:
            dict 事件：{"token": ...} 或 {"done": True, "response": ..., "metrics": ...}
        """
        messages = [{"role": "user", "content": req.message}]
        chunks = self._host.chat_stream(
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        for chunk in chunks:
            if cancel_event is not None and cancel_event.is_set():
                break
            yield {"token": chunk}
        yield {"done": True, "response": "", "followups": [], "metrics": {}, "request_id": "-"}

    def speculative_run(self, req) -> Dict[str, Any]:
        """投机解码实验端点（1.2 复制 _run_speculative_experiment 后接入真实实现）。"""
        raise NotImplementedError("speculative 实验端点随 1.2 执行段复制接入")

    # ------------------------------------------------------------------
    # generation 注册表（取消语义，对齐 api_server._register_generation）
    # ------------------------------------------------------------------
    def register_generation(
        self, generation_id: Optional[str] = None
    ) -> Tuple[str, threading.Event]:
        with self._gen_lock:
            gid = generation_id or f"gen_{uuid4().hex[:12]}"
            ev = self._generations.get(gid)
            if ev is None:
                ev = threading.Event()
                self._generations[gid] = ev
            return gid, ev

    def unregister_generation(self, generation_id: str) -> None:
        with self._gen_lock:
            self._generations.pop(generation_id, None)

    def cancel_generation(self, generation_id: str) -> bool:
        """置取消事件；返回 False 表示 generation_id 未知（404 语义）。"""
        with self._gen_lock:
            ev = self._generations.get(generation_id)
            if ev is None:
                return False
            ev.set()
            return True
