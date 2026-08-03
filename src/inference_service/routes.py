"""inference-svc HTTP 路由（微服务架构改造计划 §4.1 契约，前缀 /v1）。

依赖注入：request.app.state.engine_host / request.app.state.kv_host
（生产由 inference_svc_main 组装；契约测试注入 Fake，见
tests/test_inference_service_protocol.py）。

SSE 事件格式对齐 api_server /api/chat/stream（2026-08-03 基线）：
  data: {"token": "..."}                                        # fast 模式逐 token
  data: {"done": true, "response": "...", "followups": [...],
         "metrics": {...}, "request_id": "..."}                 # 结束事件
  data: {"done": true, "error": "...", "request_id": "..."}     # 错误事件
无 event: 行，纯 data 事件（前端 EventSource 依赖逐字节保真）。
"""
import base64
import json
import logging
from typing import Any, Dict, Iterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .protocol import (
    ChatCancelRequest,
    ChatRequest,
    EmbeddingRequest,
    KVFreeRequest,
    KVInitRequest,
    LayerForwardRequest,
    LayerLoadRequest,
    LayerUnloadRequest,
    LMHeadRequest,
    LoadModelRequest,
    SpeculativeRunRequest,
    SwitchModelRequest,
    UnloadModelRequest,
)
from .tensor_transport import deserialize_tensor, serialize_tensor

router = APIRouter(prefix="/v1")

logger = logging.getLogger("inference_service.routes")


# ----------------------------------------------------------------------
# 依赖获取
# ----------------------------------------------------------------------
def _engine_host(request: Request):
    host = getattr(request.app.state, "engine_host", None)
    if host is None:
        raise HTTPException(status_code=503, detail="engine host 未初始化")
    return host


def _kv_host(request: Request):
    host = getattr(request.app.state, "kv_host", None)
    if host is None:
        raise HTTPException(status_code=503, detail="kv host 未初始化")
    return host


def _require_master_role(request: Request) -> None:
    """1.3 角色感知：client 角色（从节点）无完整模型，chat 端点 404。"""
    if getattr(request.app.state, "node_role", "master") == "client":
        raise HTTPException(
            status_code=404,
            detail="client 角色不提供 chat 接口（从节点无完整模型）",
        )


# ----------------------------------------------------------------------
# SSE 序列化（对齐 api_server 事件格式）
# ----------------------------------------------------------------------
def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: Any, request_id: str = "-") -> str:
    if isinstance(message, dict):
        message = message.get("message") or json.dumps(message, ensure_ascii=False)
    return _sse_event({"done": True, "error": message, "request_id": request_id})


def _encode_tensor(tensor) -> str:
    return base64.b64encode(serialize_tensor(tensor)).decode("ascii")


def _decode_tensor(tensor_ref: str):
    try:
        data = base64.b64decode(tensor_ref.encode("ascii"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"tensor_ref base64 解码失败: {e}")
    return deserialize_tensor(data)


# ----------------------------------------------------------------------
# 存活/就绪/状态
# ----------------------------------------------------------------------
@router.get("/health", response_model=None)
async def health():
    return {"status": "ok", "service": "inference-svc", "version": __version__}


@router.get("/ready")
async def ready(request: Request):
    return _engine_host(request).ready()


@router.get("/status")
async def status(request: Request):
    host = _engine_host(request)
    kv = _kv_host(request)
    result = host.status()
    result["kv_cache"] = kv.status()
    return result


# ----------------------------------------------------------------------
# 模型生命周期
# ----------------------------------------------------------------------
@router.post("/models/load")
async def models_load(req: LoadModelRequest, request: Request):
    host = _engine_host(request)
    result = host.load_model(
        engine=req.engine,
        quant_type=req.quant_type,
        use_compile=req.use_compile,
        model_id=req.model_id,
    )
    if req.layer_range:
        host.load_layer_range(layer_range=req.layer_range)
    return result


@router.post("/models/unload")
async def models_unload(req: UnloadModelRequest, request: Request):
    return _engine_host(request).unload_model()


@router.post("/models/switch")
async def models_switch(req: SwitchModelRequest, request: Request):
    return _engine_host(request).switch_model(model_id=req.model_id, engine=req.engine)


@router.get("/models/current")
async def models_current(request: Request):
    return _engine_host(request).current_model()


# ----------------------------------------------------------------------
# 对话
# ----------------------------------------------------------------------
@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """完整对话（JSON；1.2c 已接入 _execute_chat_full 完整复制）。"""
    _require_master_role(request)
    host = _engine_host(request)
    request_id = request.headers.get("X-QLH-Request-ID", "-")
    generation_id, cancel_event = host.register_generation(req.generation_id)
    try:
        result = host.chat_full(req, cancel_event)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        host.unregister_generation(generation_id)
    result["request_id"] = request_id
    return result


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE 流式（事件格式与 api_server /api/chat/stream 一致）。"""
    _require_master_role(request)
    host = _engine_host(request)
    request_id = request.headers.get("X-QLH-Request-ID", "-")
    generation_id, cancel_event = host.register_generation(req.generation_id)

    if req.streaming_mode == "full":
        # full：完整功能，推理完成后一次性返回单个 done 事件（SSE 格式）
        try:
            result = host.chat_full(req, cancel_event)
        except HTTPException as e:
            return StreamingResponse(
                iter([_sse_error(e.detail, request_id)]),
                media_type="text/event-stream",
            )
        except Exception as e:
            return StreamingResponse(
                iter([_sse_error(str(e), request_id)]),
                media_type="text/event-stream",
            )
        finally:
            host.unregister_generation(generation_id)
        payload = {
            "done": True,
            "response": result.get("response", ""),
            "followups": result.get("followups", []),
            "metrics": result.get("metrics", {}),
            "request_id": request_id,
        }
        return StreamingResponse(
            iter([_sse_event(payload)]), media_type="text/event-stream"
        )

    # fast：真流式逐 token
    async def _generate():
        completed_normally = False
        try:
            for event in host.chat_stream_events(req, cancel_event):
                yield _sse_event(event)
            completed_normally = True
        except Exception as e:
            yield _sse_error(str(e), request_id)
        finally:
            if not completed_normally:
                cancel_event.set()
            host.unregister_generation(generation_id)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/chat/cancel")
async def chat_cancel(req: ChatCancelRequest, request: Request):
    _require_master_role(request)
    host = _engine_host(request)
    if not host.cancel_generation(req.generation_id):
        raise HTTPException(
            status_code=404, detail=f"未知 generation_id: {req.generation_id}"
        )
    return {"success": True, "message": "已请求取消", "generation_id": req.generation_id}


# ----------------------------------------------------------------------
# 实验端点
# ----------------------------------------------------------------------
@router.post("/speculative/run")
async def speculative_run(req: SpeculativeRunRequest, request: Request):
    """投机解码 draft-verify 实验端点（1.2b 接入真实实现；
    门控/异常映射复制自 api_server.experimental_speculative_chat）。"""
    _require_master_role(request)
    import config as _cfg

    host = _engine_host(request)
    if not getattr(_cfg, "SPEC_ENABLED", False):
        raise HTTPException(
            404,
            "投机解码实验未启用。请设置 QLH_SPEC_ENABLED=true 并配置 "
            "QLH_SPEC_VERIFY_BASE_URL（或复用 QLH_EXTERNAL_BASE_URL）后重启。",
        )
    from external_provider import ExternalScopeDeniedError
    from speculative import (
        SpeculativeCapabilityError,
        SpeculativeConfigError,
        SpeculativeError,
    )

    try:
        result = host.speculative_run(req)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="speculative 实验端点未接入")
    except ExternalScopeDeniedError as exc:
        logger.info(
            "数据作用域拒绝投机解码外部校验: scope=%s（消息正文未发送）",
            getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
        )
        raise HTTPException(403, str(exc)) from None
    except SpeculativeConfigError as exc:
        raise HTTPException(409, str(exc)) from None
    except SpeculativeCapabilityError as exc:
        raise HTTPException(502, str(exc)) from None
    except SpeculativeError as exc:
        raise HTTPException(502, str(exc)) from None

    return {
        "content": result["content"],
        "finish_reason": result["finish_reason"],
        "metrics": result["metrics"],
        "rounds": result["rounds"],
        "request_id": request.headers.get("X-QLH-Request-ID", "-"),
    }


# ----------------------------------------------------------------------
# 层段（client 角色；master 角色本地层段同用）
# ----------------------------------------------------------------------
@router.post("/layers/load")
async def layers_load(req: LayerLoadRequest, request: Request):
    return _engine_host(request).load_layer_range(
        layer_range=req.layer_range, embed=req.embed, lm_head=req.lm_head
    )


@router.post("/layers/unload")
async def layers_unload(req: LayerUnloadRequest, request: Request):
    return _engine_host(request).unload_layer_range(layer_range=req.layer_range)


@router.post("/layers/forward")
async def layers_forward(req: LayerForwardRequest, request: Request):
    host = _engine_host(request)
    hidden = _decode_tensor(req.tensor_ref)
    past_key_values = None
    if req.past_key_values_ref:
        kv_host = _kv_host(request)
        past_key_values = kv_host.get(req.past_key_values_ref)
        if past_key_values is None:
            raise HTTPException(
                status_code=404,
                detail=f"未知 KV 任务引用: {req.past_key_values_ref}",
            )
    try:
        result = host.forward_layers(
            layer_range=req.layer_range,
            hidden=hidden,
            past_key_values=past_key_values,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # 结果张量（hidden states 或 logits）回传
    if hasattr(result, "detach"):  # torch.Tensor
        return {"output_ref": _encode_tensor(result), "task_id": req.task_id}
    return {"output": result, "task_id": req.task_id}


@router.post("/layers/embedding")
async def layers_embedding(req: EmbeddingRequest, request: Request):
    host = _engine_host(request)
    input_ids = _decode_tensor(req.tensor_ref)
    try:
        result = host.embedding(input_ids)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="embedding 段未接入（1.2）")
    return {"output_ref": _encode_tensor(result)}


@router.post("/layers/lm_head")
async def layers_lm_head(req: LMHeadRequest, request: Request):
    host = _engine_host(request)
    hidden = _decode_tensor(req.tensor_ref)
    try:
        result = host.lm_head(hidden)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="lm_head 段未接入（1.2）")
    return {"output_ref": _encode_tensor(result)}


# ----------------------------------------------------------------------
# KV 缓存生命周期
# ----------------------------------------------------------------------
@router.post("/kv/init")
async def kv_init(req: KVInitRequest, request: Request):
    kv_host = _kv_host(request)
    return kv_host.init(
        task_id=req.task_id,
        device=req.device,
        page_size=req.page_size,
        max_pages=req.max_pages,
    )


@router.post("/kv/free")
async def kv_free(req: KVFreeRequest, request: Request):
    kv_host = _kv_host(request)
    try:
        return kv_host.free(task_id=req.task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知任务: {req.task_id}")
