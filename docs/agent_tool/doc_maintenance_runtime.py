#!/usr/bin/env python3
"""文档维护 Agent M2.3：受限的 OpenAI 兼容调用与缓存编排。"""
from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from doc_maintenance_cache import JudgementCache
from doc_maintenance_llm import (
    DocAgentConfig,
    Judgement,
    JudgementBatch,
    ProviderPlan,
    ProviderSpec,
    build_provider_payload,
    parse_judgement_response,
    prepare_judgement_batches,
    resolve_provider_plan,
    sanitize_text,
)

DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_TIMEOUT_SECONDS = 60.0


class ProviderUnavailable(RuntimeError):
    """仅携带固定类别，防止 URL、响应体或密钥写入报告。"""


@dataclass(frozen=True)
class ProviderReply:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleProvider:
    """最小 OpenAI chat/completions 客户端，不依赖 requests 或 SDK。"""

    def __init__(self, spec: ProviderSpec, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout must be within (0, {MAX_TIMEOUT_SECONDS}]")
        self.name = spec.name
        self.model = spec.model
        self._base_url = spec.base_url.rstrip("/")
        self._api_key = spec.api_key
        self._timeout_seconds = timeout_seconds

    def judge(self, batch: JudgementBatch) -> ProviderReply:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "仅返回符合 response_schema 的 JSON 对象。"
                        "不确定、上下文不足或证据冲突时 judgement 必须为 needs_review。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        build_provider_payload(batch), ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        # opencode 网关会拒绝 Python urllib 的默认 User-Agent；固定产品标识不含用户信息。
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "QLH-DocAgent/0.1",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise ProviderUnavailable(f"http_{exc.code}") from exc
        except (URLError, TimeoutError, socket.timeout):
            raise ProviderUnavailable("transport_error") from None
        try:
            decoded = json.loads(raw)
            content = decoded["choices"][0]["message"]["content"]
            usage = decoded.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            raise ProviderUnavailable("invalid_envelope") from None
        if not isinstance(content, str):
            raise ProviderUnavailable("invalid_content")
        return ProviderReply(content, prompt_tokens, completion_tokens)


def _report_item(batch: JudgementBatch, judgement: Judgement, *, source: str,
                 provider: str | None, model: str | None) -> dict:
    return {
        "doc": batch.doc,
        "judgement": judgement.judgement,
        "confidence": judgement.confidence,
        "suggestion": judgement.suggestion,
        "source": source,
        "provider": sanitize_text(provider or "", 128) or None,
        "model": sanitize_text(model or "", 256) or None,
    }


def _invalid_provider_result(batch: JudgementBatch) -> Judgement:
    return Judgement(
        doc_ref=batch.doc_ref,
        judgement="needs_review",
        confidence=0.0,
        suggestion="所有已配置 provider 均不可用，请人工核对",
        error="all_providers_unavailable",
    )


def run_llm_judgements(audit: dict, repo_root: Path, config: DocAgentConfig,
                       cache_path: Path, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                       plan: ProviderPlan | None = None) -> dict:
    """执行 M2 编排。网络仅在调用方显式要求 --llm 后才发生。"""
    selected_plan = plan or resolve_provider_plan(config)
    batches = prepare_judgement_batches(audit, repo_root)
    run_id = uuid.uuid4().hex
    report = {
        "enabled": True,
        "run_id": run_id,
        "provider_plan": [
            {"name": spec.name, "model": sanitize_text(spec.model, 256)}
            for spec in selected_plan.providers
        ],
        "warnings": list(selected_plan.warnings),
        "judgements": [],
        "cost": {"cache_hits": 0, "provider_calls": 0, "needs_review": 0},
    }
    with JudgementCache(cache_path) as cache:
        for batch in batches:
            cached = cache.lookup(batch)
            if cached is not None:
                report["cost"]["cache_hits"] += 1
                if cached.judgement.judgement == "needs_review":
                    report["cost"]["needs_review"] += 1
                report["judgements"].append(_report_item(
                    batch, cached.judgement, source=cached.source,
                    provider=cached.provider, model=cached.model,
                ))
                continue

            final: Judgement | None = None
            final_provider: ProviderSpec | None = None
            for spec in selected_plan.providers:
                provider = OpenAICompatibleProvider(spec, timeout_seconds)
                try:
                    reply = provider.judge(batch)
                except ProviderUnavailable:
                    cache.log_cost(
                        run_id=run_id, provider=spec.name, model=spec.model,
                        prompt_tokens=0, completion_tokens=0, hits=0, misses=1,
                    )
                    report["cost"]["provider_calls"] += 1
                    continue
                report["cost"]["provider_calls"] += 1
                parsed = parse_judgement_response(
                    reply.content, batch, confidence_floor=config.confidence_floor,
                )
                cache.log_cost(
                    run_id=run_id, provider=spec.name, model=spec.model,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens, hits=0, misses=1,
                )
                # 非法 schema 是 provider 失败，允许后备 provider 尝试；低置信度是有效结论。
                if parsed.error in {
                    "invalid_json", "invalid_schema", "doc_ref_mismatch", "invalid_judgement",
                    "invalid_confidence", "invalid_schema_value",
                }:
                    continue
                final = parsed
                final_provider = spec
                break

            if final is None:
                final = _invalid_provider_result(batch)
                source = "none"
                provider_name = None
                model = None
            else:
                source = "llm"
                provider_name = final_provider.name if final_provider else None
                model = final_provider.model if final_provider else None
                cache.store(
                    batch, final, source="llm", provider=provider_name or "unknown",
                    model=model or "unknown",
                )
            if final.judgement == "needs_review":
                report["cost"]["needs_review"] += 1
            report["judgements"].append(_report_item(
                batch, final, source=source, provider=provider_name, model=model,
            ))
    return report
