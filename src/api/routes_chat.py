"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter, File
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = (
    "ChatRequest",
    "ChatResponse",
    "Request",
    "SpeculativeExperimentRequest",
    "UploadFile",
)

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['upload_file', 'speculative_experiment_capability', 'experimental_speculative_chat', 'chat', 'chat_stream', 'cancel_chat_generation', 'clear_chat']}

async def upload_file(file: UploadFile = File(...)):
    """
    上传文本文件，返回解析后的内容。

    支持 txt / md / csv / py / json / log 等纯文本格式。
    限制 5 MB，超过 5000 行自动截断（保留前 5000 行）。
    """
    import os as _os

    # 1. 校验扩展名
    filename = file.filename or "untitled"
    ext = _os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_TEXT_EXTENSIONS:
        raise _api_module.HTTPException(
            400,
            f"不支持的文件类型: {ext}。"
            f"支持的格式: {', '.join(sorted(ALLOWED_TEXT_EXTENSIONS))}",
        )

    # 2. 读取内容
    try:
        raw = await file.read()
    except Exception as e:
        raise _api_module.HTTPException(400, f"文件读取失败: {e}")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise _api_module.HTTPException(
            413,
            f"文件过大 ({len(raw) / 1024 / 1024:.1f} MB)，"
            f"限制 {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
        )

    # 3. 解码（尝试 UTF-8 → GBK → latin-1）
    content = None
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise _api_module.HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8 或 GBK")

    # 4. 统计 + 截断
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines > MAX_UPLOAD_LINES:
        content = "\n".join(lines[:MAX_UPLOAD_LINES])
        truncated = True
    else:
        truncated = False

    # 统计字符数和词数近似值
    char_count = len(content)
    word_count = len(content.split())

    # 检测语言类型（用于前端代码高亮）
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html", ".css": "css",
        ".json": "json", ".md": "markdown", ".csv": "csv",
        ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
        ".cpp": "cpp", ".c": "c", ".h": "c", ".java": "java",
        ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".sql": "sql", ".r": "r", ".swift": "swift", ".kt": "kotlin",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    }
    language = lang_map.get(ext, "plaintext")

    _api_module.logger.info(
        f"文件上传: {filename} ({ext}) {char_count} 字符 "
        f"{total_lines} 行{' (已截断)' if truncated else ''}"
    )

    return {
        "filename": filename,
        "extension": ext,
        "language": language,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": total_lines if not truncated else MAX_UPLOAD_LINES,
        "total_lines": total_lines,
        "truncated": truncated,
        "truncated_lines": total_lines - MAX_UPLOAD_LINES if truncated else 0,
        "size_bytes": len(raw),
        "content": content,
    }

async def speculative_experiment_capability():
    """Return a zero-network, fail-closed capability snapshot for the UI."""
    try:
        import config as cfg
        from speculative import resolve_verify_config

        enabled = bool(getattr(cfg, "SPEC_ENABLED", False))
        verify = resolve_verify_config()
        configured = bool(verify.get("base_url"))
        available = enabled and configured
        reason_code = (
            "ready"
            if available
            else "disabled_by_config"
            if not enabled
            else "verify_endpoint_missing"
        )
        return {
            "enabled": enabled,
            "configured": configured,
            "available": available,
            "execution_mode": "speculative_assisted",
            "local_only": not configured,
            "data_scope": str(getattr(cfg, "EXTERNAL_DATA_SCOPE", "opt_in")),
            "reason_code": reason_code,
            "verify_model": str(verify.get("model") or ""),
            "gamma": int(verify.get("gamma", 0)),
            "max_rounds": int(verify.get("max_rounds", 0)),
        }
    except Exception as exc:
        _api_module.logger.warning("投机实验能力探针失败: %s", exc)
        return {
            "enabled": False,
            "configured": False,
            "available": False,
            "execution_mode": "speculative_assisted",
            "local_only": True,
            "reason_code": "capability_probe_error",
        }

async def experimental_speculative_chat(req: SpeculativeExperimentRequest):
    """
    投机解码 draft-verify 实验端点（路线 C-1，阶段 0-1 探索性 PoC）。

    仅在 QLH_SPEC_ENABLED=true 时存在；关闭时一律 404，主聊天路径不受影响。
    数据作用域与路线 B 共用 QLH_EXTERNAL_DATA_SCOPE（deny 档位一个包都不发）。
    """
    import config as _cfg

    if not getattr(_cfg, "SPEC_ENABLED", False):
        raise _api_module.HTTPException(
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
        result = await _api_module.run_in_threadpool(_api_module._run_speculative_experiment, req)
    except ExternalScopeDeniedError as exc:
        _api_module.logger.info(
            "数据作用域拒绝投机解码外部校验: scope=%s（消息正文未发送）",
            getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
        )
        raise _api_module.HTTPException(403, str(exc)) from None
    except SpeculativeConfigError as exc:
        raise _api_module.HTTPException(409, str(exc)) from None
    except SpeculativeCapabilityError as exc:
        # 前提条件不满足时大声失败：静默降级会让输出分布不再等于 verify 模型
        raise _api_module.HTTPException(502, str(exc)) from None
    except SpeculativeError as exc:
        raise _api_module.HTTPException(502, str(exc)) from None

    return {
        "content": result["content"],
        "finish_reason": result["finish_reason"],
        "metrics": result["metrics"],
        "rounds": result["rounds"],
        "request_id": _api_module._request_id_ctx.get("-"),
    }

async def chat(req: ChatRequest):
    """
    发送消息并获取模型回复（多轮对话）。

    自动维护对话历史 + KV 缓存。
    若模型未加载，自动尝试加载默认模型。
    """
    generation_id, cancel_event = _api_module._register_generation(req.generation_id)
    req.generation_id = generation_id
    # 路线 B：请求带外部 flag 但被数据作用域拒绝时记一条 INFO（每请求一次）
    _api_module._maybe_log_external_scope_denial(req, _api_module._external_route_decision(req))

    def _run_chat_request():
        if req.execution_mode == "task_graph":
            if not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded:
                raise _api_module.HTTPException(
                    409,
                    "任务链实验要求先加载本地完整模型。",
                )
            return _api_module._execute_requested_chat(req, cancel_event)
        with _api_module.model_host.full_chat_execution_lock:
            if not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded:
                try:
                    _api_module._ensure_chat_model_or_forwarding(req)
                except FileNotFoundError:
                    raise _api_module.HTTPException(
                        400,
                        "模型未加载且未找到可自动加载的模型文件。"
                        "请先在控制面板中加载模型。",
                    )
                except Exception as exc:
                    raise _api_module.HTTPException(
                        500,
                        f"自动加载模型失败: {exc}。请手动在控制面板中加载模型。",
                    ) from exc
            return _api_module._execute_requested_chat(req, cancel_event)

    try:
        result = await _api_module.run_in_threadpool(_run_chat_request)
    except _api_module.ChatGenerationCancelled as exc:
        raise _api_module.HTTPException(
            409,
            {"message": "生成已取消", "generation_id": exc.generation_id},
        ) from exc
    finally:
        _api_module._unregister_generation(generation_id, cancel_event)
    return _api_module.ChatResponse(
        content=result["content"],
        thinking_content=result.get("thinking_content"),
        metrics=result["metrics"],
        followups=result["followups"],
    )

async def chat_stream(req: ChatRequest, request: Request):
    """
    流式聊天端点 (Server-Sent Events)。

    支持两种模式（通过 streaming_mode 参数切换）:

    fast（默认）— 真流式，逐 token 推送:
      - 路径1: 分布式流水线 → 逐 token SSE
      - 路径2: 单机 PyTorch → 逐 token SSE（TextIteratorStreamer）
      - 路径3: llama.cpp / 其他 → 假流式回退（单次 done 事件）
      - 注意: fast 模式跳过了历史/追问/DB持久化，专注低延迟

    full — 假流式，完整功能:
      - 走 /api/chat 全流程：会话管理、对话历史、追问生成、DB 持久化
      - 推理完成后一次性返回单个 done 事件（SSE 格式）
      - 功能与 /api/chat 完全一致，仅响应格式不同

    事件格式:
        data: {"token": "你"}
        data: {"done": true, "response": "...", "followups": [...], "metrics": {...}}
    """
    import asyncio as _asyncio
    import json as _json
    request_id = _api_module._request_id_ctx.get("-")
    generation_id, cancel_event = _api_module._register_generation(req.generation_id)
    req.generation_id = generation_id

    async def _generate():
        previous_request_id = _api_module._request_id_ctx.get("-")
        _api_module._request_id_ctx.set(request_id)
        completed_normally = False
        try:
            async for chunk in _generate_events():
                yield chunk
            completed_normally = True
        finally:
            if not completed_normally:
                cancel_event.set()
            _api_module._unregister_generation(generation_id, cancel_event)
            # StreamingResponse may close an async generator from a different
            # task context. ContextVar tokens cannot be reset across contexts.
            _api_module._request_id_ctx.set(previous_request_id)

    async def _run_with_request_id(loop, func):
        def _runner():
            token = _api_module._request_id_ctx.set(request_id)
            try:
                return func()
            finally:
                _api_module._request_id_ctx.reset(token)

        return await loop.run_in_executor(None, _runner)

    def _error_event(message) -> str:
        if isinstance(message, dict):
            message = message.get("message") or _api_module.json.dumps(
                message, ensure_ascii=False,
            )
        return f"data: {_json.dumps({'done': True, 'error': message, 'request_id': request_id}, ensure_ascii=False)}\n\n"

    async def _iterate_sync_generator(iterable):
        """Bridge a blocking generator without blocking the ASGI event loop."""
        queue = _asyncio.Queue()
        done = object()

        def _pump():
            try:
                for item in iterable:
                    loop.call_soon_threadsafe(queue.put_nowait, (item, None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, (None, exc))
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, (done, None))
                except RuntimeError:
                    pass

        loop = _asyncio.get_running_loop()
        _api_module.threading.Thread(
            target=_pump,
            name=f"chat-stream-bridge-{generation_id[-8:]}",
            daemon=True,
        ).start()
        try:
            while True:
                item, error = await queue.get()
                if item is done:
                    break
                if error is not None:
                    raise error
                yield item
        finally:
            cancel_event.set()

    async def _generate_events():
        # 路线 B：请求带外部 flag 但被数据作用域拒绝时记一条 INFO（每请求一次）
        _api_module._maybe_log_external_scope_denial(req, _api_module._external_route_decision(req))
        # T9.5：distributed_required 无分布式路径时明确失败（interactive/fast 共用）
        routing_gate = _api_module._routing_gate_error(req)
        if routing_gate:
            yield _error_event(routing_gate)
            return
        # ================================================================
        # ★ interactive 模式（T9 聊天页契约）：真流式逐 token +
        #    完成时会话事务提交（user + assistant 一次写入）
        # ================================================================
        if req.streaming_mode == "interactive":
            target_session_id = req.session_id or _api_module.active_session_id
            if target_session_id and target_session_id != _api_module.active_session_id:
                try:
                    _api_module._switch_session(target_session_id)
                except Exception:
                    pass
            history = _api_module._get_active_history()
            if target_session_id and len(history) == 0:
                try:
                    _api_module._auto_title_session(target_session_id, req.message)
                except Exception:
                    pass

            yield f"data: {_json.dumps({'start': True, 'generation_id': generation_id, 'request_id': request_id, 'session_id': target_session_id, 'routing_preference': req.routing_preference}, ensure_ascii=False)}\n\n"

            response_parts: list[str] = []
            thinking_parts: list[str] = []
            metrics: dict = {}
            error: _api_module.Optional[str] = None
            cancelled = False
            distributed_used = False

            def _append_event(event: dict) -> None:
                nonlocal metrics, error
                if event.get("token"):
                    response_parts.append(str(event["token"]))
                if event.get("thinking"):
                    thinking_parts.append(str(event["thinking"]))
                if isinstance(event.get("metrics"), dict):
                    metrics.update(event["metrics"])
                if event.get("error"):
                    error = str(event["error"])

            loop = _asyncio.get_running_loop()

            def _token_frame(event: dict) -> Optional[str]:
                """token 事件 → SSE 帧；非 token 事件返回 None。"""
                if event.get("token") is not None:
                    return f"data: {_json.dumps({'token': event['token']}, ensure_ascii=False)}\n\n"
                return None

            try:
                # ---- 外部路由（数据作用域门控，与 fast/full 同语义）----
                _ext_decision = _api_module._external_route_decision(req)
                if _ext_decision.use_external:
                    async for event in _iterate_sync_generator(
                        _api_module._external_stream_events(req, cancel_event),
                    ):
                        _append_event(event)
                        frame = _token_frame(event)
                        if frame:
                            yield frame
                        if event.get("done"):
                            break
                elif req.image_data_urls:
                    error = (
                        "本地原生图像请求仅支持 full 响应模式；"
                        "请改用 streaming_mode=full 和 local_only"
                    )
                elif (req.routing_preference != "local_only"
                      and _api_module._should_forward_chat_to_master()):
                    # 计划 §9.5：聊天请求不绕过网关直打本地模型；从节点
                    # interactive 明确失败并引导连接主节点，不静默降级。
                    error = (
                        "interactive 模式要求连接主节点；当前节点是从节点，"
                        "请用 --host 指定主节点 endpoint 后重试"
                    )
                elif _api_module._pipeline_worker_is_reserved():
                    error = (
                        "本设备正作为 PyTorch 分层从节点，interactive 模式不可用；"
                        "请连接主节点"
                    )
                else:
                    if (
                        (not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded)
                        and not _api_module._pipeline_model_is_prepared()
                    ):
                        try:
                            await _run_with_request_id(loop, _api_module._auto_load_default_model)
                        except Exception as exc:
                            error = f"本地回退模型加载失败: {exc}"
                    if error is None:
                        distributed_used = False
                        # 路径 1: 分布式流水线流式（master；local_only 强制跳过）
                        if (req.routing_preference != "local_only"
                                and _api_module.scheduler.get_distributed_inference_enabled()
                                and _api_module.RUN_MODE == "distributed"
                                and _api_module.scheduler._effective_role() == "master"
                                and _api_module.runtime_supports(
                                    _api_module.model_manager, _api_module.Capability.FORWARD_LAYERS,
                                )):
                            distributed_used = True
                            async for event in _iterate_sync_generator(_api_module.scheduler.run_pipeline_stream(
                                req.message,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                session_id=req.session_id,
                                messages=[{"role": "user", "content": req.message}],
                                show_thinking=req.show_thinking,
                                _require_distributed=(req.routing_preference == "distributed_required"),
                                _force_distributed_assignment=True,
                                _cancel_event=cancel_event,
                            )):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        # 路径 2: 单机 PyTorch 流式
                        elif (_api_module.backend_id_for(_api_module.model_manager) == "llama_cpp"
                              and _api_module.model_manager.is_loaded
                              and callable(getattr(_api_module.model_manager, "chat_stream", None))
                              and callable(getattr(
                                  _api_module.scheduler, "_run_full_model_inference_stream", None,
                              ))):
                            async for event in _iterate_sync_generator(
                                _api_module.scheduler._run_full_model_inference_stream(
                                    req.message,
                                    max_new_tokens=req.max_new_tokens,
                                    temperature=req.temperature,
                                    top_p=req.top_p,
                                    messages=[{"role": "user", "content": req.message}],
                                    show_thinking=req.show_thinking,
                                    _cancel_event=cancel_event,
                                ),
                            ):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        elif (_api_module.backend_id_for(_api_module.model_manager) == "pytorch"
                                and _api_module.model_manager.is_loaded):
                            async for event in _iterate_sync_generator(_api_module.scheduler._run_full_model_inference_stream(
                                req.message,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                session_id=req.session_id,
                                messages=[{"role": "user", "content": req.message}],
                                show_thinking=req.show_thinking,
                                _cancel_event=cancel_event,
                            )):
                                _append_event(event)
                                frame = _token_frame(event)
                                if frame:
                                    yield frame
                        # 路径 3: llama.cpp / 其他 → 假流式回退（整段作为单 token 事件）
                        else:
                            result = await _run_with_request_id(
                                loop,
                                lambda: _api_module.scheduler.run_pipeline_safe(
                                    req.message,
                                    max_new_tokens=req.max_new_tokens,
                                    temperature=req.temperature,
                                    top_p=req.top_p,
                                    session_id=req.session_id,
                                    _require_distributed=(req.routing_preference == "distributed_required"),
                                    _force_distributed_assignment=(req.routing_preference != "local_only"),
                                    _cancel_event=cancel_event,
                                ),
                            )
                            if isinstance(result.get("metrics"), dict):
                                metrics.update(result["metrics"])
                            response = result.get("response", "")
                            if response:
                                response_parts.append(str(response))
                                yield f"data: {_json.dumps({'token': str(response)}, ensure_ascii=False)}\n\n"
                            if result.get("error"):
                                error = str(result["error"])
            except _api_module.ChatGenerationCancelled:
                cancelled = True
            except _api_module.HTTPException as exc:
                error = exc.detail
            except Exception as exc:
                _api_module.logger.error(f"interactive 模式推理失败: {exc}", exc_info=True)
                error = str(exc)

            if cancelled:
                partial = "".join(response_parts)
                yield f"data: {_json.dumps({'cancelled': True, 'generation_id': generation_id, 'request_id': request_id, 'session_id': target_session_id, 'partial': partial}, ensure_ascii=False)}\n\n"
                return
            if error:
                yield _error_event(error)
                return

            response_text = "".join(response_parts)
            metrics.setdefault("request_id", request_id)
            metrics["generation_id"] = generation_id
            metrics["routing_preference"] = req.routing_preference
            metrics["distributed_requested"] = req.routing_preference in (
                "distributed_preferred", "distributed_required",
            )
            metrics["distributed_used"] = distributed_used
            if (req.routing_preference in ("distributed_preferred",
                                           "distributed_required")
                    and not distributed_used):
                # 请求分布式但实际本地：必须展示回退原因（计划 §9.5）
                metrics.setdefault("fallback", True)
                metrics.setdefault(
                    "fallback_reason",
                    "distributed_unavailable_fallback_to_local",
                )
            committed = _api_module._commit_interactive_history(
                target_session_id, req.message, response_text, metrics,
            )
            done_payload = {
                "done": True,
                "response": response_text,
                "thinking_content": "".join(thinking_parts) or None,
                "followups": [],
                "metrics": metrics,
                "generation_id": generation_id,
                "request_id": request_id,
                "session_id": target_session_id,
                "history_committed": committed,
            }
            yield f"data: {_json.dumps(done_payload, ensure_ascii=False)}\n\n"
            return

        # ================================================================
        # ★ full 模式：完整 chat 流程，假流式（SSE 单事件）
        # ================================================================
        if req.streaming_mode == "full" or req.execution_mode == "task_graph":
            loop = _asyncio.get_running_loop()

            def _execute_full_stream_request():
                if req.execution_mode == "task_graph":
                    if not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded:
                        raise _api_module.HTTPException(
                            409,
                            "任务链实验要求先加载本地完整模型。",
                        )
                    return _api_module._execute_requested_chat(req, cancel_event)
                with _api_module.model_host.full_chat_execution_lock:
                    if not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded:
                        _api_module._ensure_chat_model_or_forwarding(req)
                    return _api_module._execute_requested_chat(req, cancel_event)

            try:
                result = await _run_with_request_id(
                    loop, _execute_full_stream_request,
                )
                _done_payload = _json.dumps({
                    'done': True,
                    'response': result['content'],
                    'thinking_content': result.get('thinking_content'),
                    'followups': result['followups'],
                    'metrics': result['metrics'],
                    'request_id': request_id,
                }, ensure_ascii=False)
                yield f"data: {_done_payload}\n\n"
            except _api_module.HTTPException as e:
                yield _error_event(e.detail)
            except Exception as e:
                _api_module.logger.error(f"full 模式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))
            return

        # ================================================================
        # fast 模式：真流式，跳过历史/追问/DB 持久化（低延迟）
        # ================================================================
        # ---- 路线 B: 外部推理服务真流式（数据作用域门控，默认不出集群）----
        external_fallback_reason = ""
        _ext_decision = _api_module._external_route_decision(req)
        if req.image_data_urls and not _ext_decision.use_external:
            yield _error_event(
                "多模态请求必须使用已启用且获数据作用域授权的 external_api："
                f"{_ext_decision.reason}"
            )
            return
        if _ext_decision.use_external:
            loop = _asyncio.get_running_loop()
            ext_events = _api_module._external_stream_events(req, cancel_event)
            first_event = None
            external_error = None
            try:
                # 首个事件同步拉取：失败发生在任何 SSE 数据发出之前，
                # 可以干净地回退到下方既有本地路径
                first_event = await _run_with_request_id(
                    loop, lambda: next(ext_events, None),
                )
            except Exception as exc:
                external_error = exc
            if external_error is None:
                if first_event is not None:
                    yield f"data: {_json.dumps(first_event, ensure_ascii=False)}\n\n"
                    if not first_event.get("done"):
                        try:
                            async for event in _iterate_sync_generator(ext_events):
                                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            _api_module.logger.error(f"外部推理服务流式失败: {e}")
                            yield _error_event(str(e))
                return
            # 外部失败 → 与 /api/chat 同语义：优先回退本地路径
            if req.image_data_urls:
                yield _error_event(
                    "多模态外部推理服务调用失败，禁止丢弃图片后回退："
                    f"{external_error}"
                )
                return
            if req.prefer_external and not (
                (_api_module.model_host.model_loaded and _api_module.model_manager.is_loaded)
                or _api_module._pipeline_model_is_prepared()
            ):
                yield _error_event(
                    f"外部推理服务调用失败，且本地无可用推理引擎：{external_error}"
                )
                return
            external_fallback_reason = f"external_api_failed: {external_error}"
            _api_module.logger.warning(
                f"外部推理服务流式调用失败: {external_error}，回退到本地路径"
            )

        # ---- 路径 0: 分布式 client 优先转发主节点 ----
        # A pipeline worker may already hold a valid PyTorch segment. It must
        # never enter the local PyTorch streaming branch, which restores the
        # full model and invalidates the master's ready ACK.
        # T9.5: local_only 强制不走转发，仅本地执行
        if (req.routing_preference != "local_only"
                and _api_module._should_forward_chat_to_master()):
            loop = _asyncio.get_running_loop()
            result = await _run_with_request_id(
                loop,
                lambda: _api_module.scheduler.forward_inference_to_master(
                    message=req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    show_thinking=req.show_thinking,
                    routing_preference=req.routing_preference,
                    session_id=req.session_id,
                    messages=[{"role": "user", "content": req.message}],
                    request_id=request_id,
                    _cancel_event=cancel_event,
                ),
            )
            if result.get("status") == "ok":
                metrics = result.get("metrics", {}) or {}
                metrics.setdefault("request_id", request_id)
                _done_payload = _json.dumps({
                    'done': True,
                    'response': result.get('content', ''),
                    'thinking_content': result.get('thinking_content'),
                    'metrics': metrics,
                    'request_id': request_id,
                }, ensure_ascii=False)
                yield f"data: {_done_payload}\n\n"
                return
            if _api_module._pipeline_worker_is_reserved():
                yield _error_event(
                    result.get("error")
                    or "本设备正作为 PyTorch 分层从节点，无法转发到主节点。"
                )
                return

        if _api_module._pipeline_worker_is_reserved():
            yield _error_event(
                "本设备正作为 PyTorch 分层从节点，"
                "请先断开主节点或明确切换本地模型。"
            )
            return

        if (
            (not _api_module.model_host.model_loaded or not _api_module.model_manager.is_loaded)
            and not _api_module._pipeline_model_is_prepared()
        ):
            loop = _asyncio.get_running_loop()
            try:
                await _run_with_request_id(loop, _api_module._auto_load_default_model)
            except Exception as exc:
                yield _error_event(f"本地回退模型加载失败: {exc}")
                return

        # ---- 路径 1: 分布式流水线流式（local_only 强制跳过，走单机）----
        if (req.routing_preference != "local_only"
                and _api_module.scheduler.get_distributed_inference_enabled()
                and _api_module.RUN_MODE == "distributed"
                and _api_module.scheduler._effective_role() == "master"
                and _api_module.runtime_supports(_api_module.model_manager, _api_module.Capability.FORWARD_LAYERS)):
            try:
                async for event in _iterate_sync_generator(_api_module.scheduler.run_pipeline_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=[{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                    _require_distributed=(req.routing_preference == "distributed_required"),
                    _force_distributed_assignment=True,
                    _cancel_event=cancel_event,
                )):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))

        # ---- 路径 2: 单机 PyTorch 流式 ----
        elif (_api_module.backend_id_for(_api_module.model_manager) == "llama_cpp"
                and _api_module.model_manager.is_loaded
                and callable(getattr(_api_module.model_manager, "chat_stream", None))
                and callable(getattr(
                    _api_module.scheduler, "_run_full_model_inference_stream", None,
                ))):
            try:
                async for event in _iterate_sync_generator(
                    _api_module.scheduler._run_full_model_inference_stream(
                        req.message,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        messages=[{"role": "user", "content": req.message}],
                        show_thinking=req.show_thinking,
                        _cancel_event=cancel_event,
                    ),
                ):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"llama.cpp 流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))

        elif (_api_module.backend_id_for(_api_module.model_manager) == "pytorch"
                and _api_module.model_manager.is_loaded):
            try:
                async for event in _iterate_sync_generator(_api_module.scheduler._run_full_model_inference_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=[{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                    _cancel_event=cancel_event,
                )):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                _api_module.logger.error(f"单机流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))

        # ---- 路径 3: llama.cpp / 从节点 / 模型未加载 → 假流式回退 ----
        else:
            loop = _asyncio.get_running_loop()
            result = await _run_with_request_id(
                loop,
                lambda: _api_module.scheduler.run_pipeline_safe(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    _require_distributed=(req.routing_preference == "distributed_required"),
                    _force_distributed_assignment=(req.routing_preference != "local_only"),
                    _cancel_event=cancel_event,
                )
            )
            # 一次性返回完整结果（SSE 格式，单事件）
            metrics = result.get('metrics', {}) or {}
            metrics.setdefault("request_id", request_id)
            if external_fallback_reason and not metrics.get("fallback_reason"):
                metrics["fallback"] = True
                metrics["fallback_reason"] = external_fallback_reason
            yield f"data: {_json.dumps({'done': True, 'response': result.get('response', ''), 'error': result.get('error'), 'metrics': metrics, 'request_id': request_id}, ensure_ascii=False)}\n\n"

    return _api_module.StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
            "X-Generation-ID": generation_id,
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )

async def cancel_chat_generation(generation_id: str):
    return {
        "status": _api_module._request_generation_cancel(generation_id),
        "generation_id": generation_id,
    }

def clear_chat(session_id: str = "default"):
    """清空当前活跃会话的对话历史与 KV 缓存"""
    global kv_cache, conversation_stats
    result = _api_module.delete_conversations(session_id)
    _api_module.conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    if _api_module.kv_cache:
        _api_module.kv_cache.clear()
    _api_module._init_kv_cache()
    _api_module.logger.info("对话历史已清空: session=%s", result["session_id"])
    return {
        "status": "cleared",
        "session_id": result["session_id"],
        "deleted_count": result.get("deleted_count", 0),
        "conversation_turns": 0,
    }


def register_routes() -> None:
    for name in ('clear_chat',):
        handler = _api_module._serialized_conversation_mutation(globals()[name])
        globals()[name] = handler
    router.add_api_route('/api/chat/upload', upload_file, methods=['POST'])
    router.add_api_route('/api/experimental/speculative/capability', speculative_experiment_capability, methods=['GET'])
    router.add_api_route('/api/experimental/speculative', experimental_speculative_chat, methods=['POST'])
    router.add_api_route('/api/chat', chat, methods=['POST'], response_model=ChatResponse)
    router.add_api_route('/api/chat/stream', chat_stream, methods=['POST'])
    router.add_api_route('/api/chat/generations/{generation_id}/cancel', cancel_chat_generation, methods=['POST'])
    router.add_api_route('/api/chat/clear', clear_chat, methods=['POST'])
