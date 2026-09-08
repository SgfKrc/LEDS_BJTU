"""FastAPI app factory for the standalone harness."""

import json
import base64
import uuid
from dataclasses import replace
from typing import Any, Iterable, Mapping

from ..adapters.base import AdapterError, ChatAdapter
from ..context_engine import ContextBudget, ContextPolicy, ContextMessage
from ..image_workbench.assets import ImageAssetStore
from ..image_workbench.contracts import ImageAdapter, ImageAdapterError, ImageRequest, ImageRequestError
from ..rag import RagStore, build_context
from ..session import SessionStore
from .mapping import APIRequestError, chunk_to_openai, parse_chat_request, response_to_openai


class APIUnavailable(RuntimeError):
    """Raised when the optional HTTP API dependency is not installed."""


def _error_body(message: str, *, code: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": code}}


def _sse(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def create_app(
    adapter: ChatAdapter,
    *,
    context_policy: ContextPolicy | None = None,
    context_budget: ContextBudget | None = None,
    image_adapter: ImageAdapter | None = None,
    image_store: ImageAssetStore | None = None,
    rag_store: RagStore | None = None,
    session_store: SessionStore | None = None,
) -> Any:
    """Create an OpenAI-compatible app around one explicit adapter.

    Context policy is opt-in until S2 has a model-specific profile.  Passing a
    policy without a budget (or vice versa) is rejected to avoid hidden ctx
    defaults.
    """

    if (context_policy is None) != (context_budget is None):
        raise ValueError("context_policy and context_budget must be provided together")
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise APIUnavailable("fastapi is required for the harness API layer") from exc

    app = FastAPI(title="QLH Harness Workbench", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> Any:
        try:
            return {"status": "ok", "backend": adapter.capabilities().backend}
        except AdapterError as exc:
            return JSONResponse(
                _error_body(str(exc), code=exc.code, error_type="backend_error"),
                status_code=exc.status_code,
            )

    @app.get("/v1/models")
    async def models() -> Any:
        try:
            values = adapter.models()
        except AdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)
        return {"object": "list", "data": [item.as_dict() for item in values]}

    @app.get("/v1/capabilities")
    async def capabilities() -> Any:
        try:
            return adapter.capabilities().as_dict()
        except AdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)

    @app.get("/v1/images/capabilities")
    async def image_capabilities() -> Any:
        if image_adapter is None:
            return JSONResponse(
                _error_body("image adapter is not configured", code="images_unavailable", error_type="backend_error"),
                status_code=503,
            )
        try:
            return image_adapter.capabilities().as_dict()
        except ImageAdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)

    @app.post("/v1/images/generations")
    async def image_generations(request: Request) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if image_adapter is None:
            return JSONResponse(
                _error_body("image adapter is not configured", code="images_unavailable", error_type="backend_error"),
                status_code=503,
                headers={"X-Request-ID": request_id},
            )
        try:
            payload = await request.json()
            image_request = ImageRequest.from_mapping(payload)
            generated = image_adapter.generate(image_request)
        except ImageRequestError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code), status_code=400, headers={"X-Request-ID": request_id})
        except ImageAdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code, headers={"X-Request-ID": request_id})
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_image_result", error_type="backend_error"), status_code=502, headers={"X-Request-ID": request_id})

        record = None
        if image_store is not None:
            record = image_store.put(generated, prompt=image_request.prompt, owner_scope=image_request.user or "local")
        if image_request.response_format == "url":
            if record is None:
                return JSONResponse(_error_body("url response requires an image asset store", code="image_url_unavailable"), status_code=503, headers={"X-Request-ID": request_id})
            item: dict[str, Any] = {"url": f"/v1/images/assets/{record.asset_id}", "asset_id": record.asset_id}
        else:
            item = {"b64_json": base64.b64encode(generated.data).decode("ascii")}
            if record is not None:
                item["asset_id"] = record.asset_id
        if record is not None:
            item["metadata"] = record.as_dict()
        return JSONResponse({"created": int(__import__("time").time()), "data": [item]}, headers={"X-Request-ID": request_id})

    @app.get("/v1/images/assets/{asset_id}")
    async def image_asset(asset_id: str) -> Any:
        if image_store is None:
            return JSONResponse(_error_body("image asset store is not configured", code="images_unavailable", error_type="backend_error"), status_code=503)
        try:
            record, data = image_store.read(asset_id)
        except ImageAdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)
        from fastapi.responses import Response

        return Response(content=data, media_type=record.mime_type, headers={"Cache-Control": "private, no-store"})

    @app.get("/v1/rag/health")
    async def rag_health() -> Any:
        if rag_store is None:
            return JSONResponse(_error_body("RAG store is not configured", code="rag_unavailable", error_type="backend_error"), status_code=503)
        return rag_store.health()

    @app.post("/v1/rag/sources")
    async def rag_add_source(request: Request) -> Any:
        if rag_store is None:
            return JSONResponse(_error_body("RAG store is not configured", code="rag_unavailable", error_type="backend_error"), status_code=503)
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be an object")
            result = rag_store.add_document(
                source_ref=payload.get("source_ref", ""),
                title=payload.get("title", ""),
                text=payload.get("text", ""),
                owner_scope=payload.get("owner_scope", "local"),
                max_chars=payload.get("max_chars", 1200),
                overlap_chars=payload.get("overlap_chars", 120),
            )
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_rag_source"), status_code=400)
        return JSONResponse(result, status_code=201)

    @app.post("/v1/rag/search")
    async def rag_search(request: Request) -> Any:
        if rag_store is None:
            return JSONResponse(_error_body("RAG store is not configured", code="rag_unavailable", error_type="backend_error"), status_code=503)
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be an object")
            query = payload.get("query", "")
            hits = rag_store.search(query, owner_scope=payload.get("owner_scope", "local"), limit=payload.get("limit", 8))
            context = build_context([hit.as_dict() for hit in hits], max_chars=payload.get("max_chars", 8_000))
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_rag_query"), status_code=400)
        return {"query": query, "hits": [hit.as_dict() for hit in hits], "context": context.as_dict()}

    @app.post("/v1/sessions")
    async def create_session(request: Request) -> Any:
        if session_store is None:
            return JSONResponse(_error_body("session store is not configured", code="sessions_unavailable", error_type="backend_error"), status_code=503)
        try:
            payload = await request.json()
            payload = payload if isinstance(payload, Mapping) else {}
            session = session_store.create(owner_scope=payload.get("owner_scope", "local"), title=payload.get("title", "New session"))
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_session"), status_code=400)
        return JSONResponse(session.as_dict(), status_code=201)

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request) -> Any:
        if session_store is None:
            return JSONResponse(_error_body("session store is not configured", code="sessions_unavailable", error_type="backend_error"), status_code=503)
        try:
            owner_scope = request.query_params.get("owner_scope")
            return session_store.get(session_id, owner_scope=owner_scope)
        except KeyError:
            return JSONResponse(_error_body("session not found", code="session_not_found"), status_code=404)

    @app.post("/v1/sessions/{session_id}/messages", status_code=201)
    async def append_session_message(session_id: str, request: Request) -> Any:
        if session_store is None:
            return JSONResponse(_error_body("session store is not configured", code="sessions_unavailable", error_type="backend_error"), status_code=503)
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be an object")
            owner_scope = payload.get("owner_scope", "local")
            session_store.get(session_id, owner_scope=owner_scope)
            message = session_store.append_message(
                session_id,
                role=payload.get("role", "user"),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
            )
        except KeyError:
            return JSONResponse(_error_body("session not found", code="session_not_found"), status_code=404)
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_session_message"), status_code=400)
        return message.as_dict()

    @app.post("/v1/sessions/{session_id}/assets", status_code=201)
    async def attach_session_asset(session_id: str, request: Request) -> Any:
        if session_store is None:
            return JSONResponse(_error_body("session store is not configured", code="sessions_unavailable", error_type="backend_error"), status_code=503)
        try:
            payload = await request.json()
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be an object")
            owner_scope = payload.get("owner_scope", "local")
            session_store.get(session_id, owner_scope=owner_scope)
            asset = session_store.attach_asset(
                session_id,
                asset_id=payload.get("asset_id", ""),
                kind=payload.get("kind", "image"),
                metadata=payload.get("metadata", {}),
            )
        except KeyError:
            return JSONResponse(_error_body("session not found", code="session_not_found"), status_code=404)
        except (TypeError, ValueError) as exc:
            return JSONResponse(_error_body(str(exc), code="invalid_session_asset"), status_code=400)
        return asset.as_dict()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        try:
            payload = await request.json()
            adapter_request = parse_chat_request(payload, request_id=request_id)
            if context_policy is not None and context_budget is not None:
                snapshot = context_policy.build(
                    [ContextMessage.from_value(message) for message in adapter_request.messages],
                    context_budget,
                )
                adapter_request = replace(
                    adapter_request,
                    messages=tuple(message.as_dict() for message in snapshot.messages),
                )
        except APIRequestError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code), status_code=exc.status_code)
        except ValueError as exc:
            return JSONResponse(_error_body(str(exc), code="context_invalid"), status_code=400)
        except AdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)

        if adapter_request.stream:
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"

            def events() -> Iterable[str]:
                try:
                    for chunk in adapter.stream(adapter_request):
                        yield _sse(
                            chunk_to_openai(
                                chunk,
                                fallback_id=stream_id,
                                fallback_model=adapter_request.model,
                            )
                        )
                    yield "data: [DONE]\n\n"
                except AdapterError as exc:
                    yield _sse(_error_body(str(exc), code=exc.code, error_type="backend_error"))
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                },
            )
        try:
            response = adapter.complete(adapter_request)
        except AdapterError as exc:
            return JSONResponse(_error_body(str(exc), code=exc.code, error_type="backend_error"), status_code=exc.status_code)
        return JSONResponse(response_to_openai(response), headers={"X-Request-ID": request_id})

    return app
