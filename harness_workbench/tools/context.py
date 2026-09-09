"""Bounded tool-result context injection for normal answer models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..model_profiles import CapabilityGate, GateDecision, ModelProfile
from .network import MAX_SEARCH_ITEMS, NetworkPolicy, NetworkToolError, _validate_url


TOOL_CONTEXT_SCHEMA = "qlh.tool_context.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOOL_NAMES = frozenset({"web_search", "web_fetch"})


class ToolContextError(ValueError):
    """Stable, non-sensitive error raised before result injection."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


@dataclass(frozen=True, slots=True)
class ToolContextPolicy:
    """Server-owned limits for the data object supplied to an answer model."""

    max_items: int = MAX_SEARCH_ITEMS
    max_citations: int = MAX_SEARCH_ITEMS
    max_item_chars: int = 4_096
    max_total_chars: int = 12_000

    def __post_init__(self) -> None:
        for name in ("max_items", "max_citations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_SEARCH_ITEMS:
                raise ValueError(f"{name} must be between 1 and {MAX_SEARCH_ITEMS}")
        if isinstance(self.max_item_chars, bool) or not isinstance(self.max_item_chars, int) or not 256 <= self.max_item_chars <= 32 * 1024:
            raise ValueError("max_item_chars must be between 256 and 32768")
        if isinstance(self.max_total_chars, bool) or not isinstance(self.max_total_chars, int) or not 512 <= self.max_total_chars <= 64 * 1024:
            raise ValueError("max_total_chars must be between 512 and 65536")
        if self.max_total_chars < self.max_item_chars:
            raise ValueError("max_total_chars must cover max_item_chars")


class ToolContextBuilder:
    """Validate capability evidence and create a bounded tool-role object."""

    def __init__(
        self,
        *,
        policy: ToolContextPolicy | None = None,
        network_policy: NetworkPolicy | None = None,
        capability_gate: CapabilityGate | None = None,
    ) -> None:
        self.policy = policy or ToolContextPolicy()
        self.network_policy = network_policy or NetworkPolicy()
        self.capability_gate = capability_gate or CapabilityGate()

    def build(
        self,
        request: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        profile: ModelProfile | None = None,
        decision: GateDecision | None = None,
        mode: str = "host_router",
    ) -> dict[str, Any]:
        self._capability(profile=profile, decision=decision, mode=mode)
        tool_name, request_id = self._identity(request, result)
        items = result.get("items")
        citations = result.get("citations")
        if not isinstance(items, list) or not isinstance(citations, list):
            raise ToolContextError("invalid_result", "tool result items and citations are required")
        if len(items) > MAX_SEARCH_ITEMS or len(citations) > MAX_SEARCH_ITEMS:
            raise ToolContextError("invalid_result", "tool result exceeds the item limit")

        normalized_items, item_urls, truncated = self._items(items, tool_name)
        normalized_citations = self._citations(citations, item_urls)
        if normalized_items and not normalized_citations:
            raise ToolContextError("citation_required", "tool context items require citations")
        return {
            "schema": TOOL_CONTEXT_SCHEMA,
            "role": "tool",
            "name": tool_name,
            "request_id": request_id,
            "items": normalized_items,
            "citations": normalized_citations,
            "truncated": bool(result.get("truncated", False) or truncated),
        }

    def _capability(
        self,
        *,
        profile: ModelProfile | None,
        decision: GateDecision | None,
        mode: str,
    ) -> GateDecision:
        if mode not in {"host_router", "autonomous_tools"}:
            raise ToolContextError("invalid_mode", "tool context mode is invalid")
        if profile is not None:
            computed = self.capability_gate.evaluate(profile, role="tool_router" if mode == "autonomous_tools" else "answer")
            if decision is not None and decision.profile_id != computed.profile_id:
                raise ToolContextError("capability_mismatch", "capability decision does not match the profile")
            decision = computed
        if decision is None:
            raise ToolContextError("capability_not_verified", "tool result reinjection requires verified capability evidence")
        reinjection = decision.capabilities.get("tool_result_reinjection")
        if decision.status != "verified" or reinjection is None or reinjection.status != "verified":
            raise ToolContextError("capability_not_verified", "tool result reinjection is not verified")
        if mode == "autonomous_tools" and not decision.can("autonomous_tools"):
            raise ToolContextError("autonomous_tools_not_allowed", "autonomous tool mode is not admitted")
        return decision

    def _identity(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise ToolContextError("invalid_request", "tool context inputs must be objects")
        tool_name = request.get("tool_name")
        request_id = request.get("request_id") or result.get("request_id")
        if tool_name not in _TOOL_NAMES or not isinstance(request_id, str) or not request_id:
            raise ToolContextError("invalid_identity", "tool context identity is invalid")
        if result.get("request_id") != request_id:
            raise ToolContextError("request_id_mismatch", "tool result request_id does not match the request")
        result_tool = result.get("tool_name")
        if result_tool is not None and result_tool != tool_name:
            raise ToolContextError("tool_name_mismatch", "tool result name does not match the request")
        if result.get("status") != "ok":
            raise ToolContextError("tool_result_error", "only successful tool results may be injected")
        if result.get("schema") not in {"qlh.tool_result.v1", "qlh.harness.tool_result.v1"}:
            raise ToolContextError("invalid_schema", "tool result schema is unsupported")
        return str(tool_name), request_id

    def _items(self, items: list[Any], tool_name: str) -> tuple[list[dict[str, str]], set[str], bool]:
        normalized: list[dict[str, str]] = []
        urls: set[str] = set()
        used_chars = 0
        truncated = False
        for raw in items:
            if len(normalized) >= self.policy.max_items:
                truncated = True
                break
            item = self._item(raw, tool_name)
            title, url, snippet, item_truncated = item
            truncated = truncated or item_truncated
            remaining = self.policy.max_total_chars - used_chars
            if remaining <= 0:
                truncated = True
                break
            allowed = min(self.policy.max_item_chars, remaining)
            if len(title) + len(snippet) > allowed:
                title_budget = min(len(title), max(1, allowed // 4))
                title = title[:title_budget]
                snippet_budget = max(1, allowed - len(title))
                snippet = snippet[:snippet_budget]
                truncated = True
            used_chars += len(title) + len(snippet)
            normalized.append({"title": title, "url": url, "snippet": snippet})
            urls.add(url)
        return normalized, urls, truncated

    def _item(self, value: Any, tool_name: str) -> tuple[str, str, str, bool]:
        if not isinstance(value, Mapping):
            raise ToolContextError("invalid_result", "tool result item must be an object")
        title = value.get("title")
        url = value.get("url") or value.get("final_url")
        snippet = value.get("snippet")
        if snippet is None and tool_name == "web_fetch":
            snippet = value.get("text")
        if not isinstance(title, str):
            title = str(url or "")
        if not isinstance(snippet, str) or not snippet.strip():
            raise ToolContextError("invalid_result", "tool result item has no bounded text")
        try:
            safe_url = _validate_url(url, self.network_policy)
        except NetworkToolError as exc:
            raise ToolContextError(exc.code, str(exc)) from exc
        normalized_title = title.strip()
        item_truncated = len(normalized_title) > self.policy.max_item_chars
        return normalized_title[: self.policy.max_item_chars], safe_url, snippet.strip(), item_truncated

    def _citations(self, citations: list[Any], item_urls: set[str]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in citations:
            if len(normalized) >= self.policy.max_citations:
                break
            if not isinstance(raw, Mapping) or set(raw) - {"url", "sha256"}:
                raise ToolContextError("invalid_citation", "tool citation fields are invalid")
            digest = raw.get("sha256")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ToolContextError("invalid_citation", "tool citation digest is invalid")
            try:
                url = _validate_url(raw.get("url"), self.network_policy)
            except NetworkToolError as exc:
                raise ToolContextError(exc.code, str(exc)) from exc
            if url not in item_urls or url in seen:
                continue
            seen.add(url)
            normalized.append({"url": url, "sha256": digest})
        return normalized


def build_tool_result_context(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    profile: ModelProfile | None = None,
    decision: GateDecision | None = None,
    mode: str = "host_router",
    policy: ToolContextPolicy | None = None,
    network_policy: NetworkPolicy | None = None,
) -> dict[str, Any]:
    """Functional entry point matching the main project's tool helper."""

    return ToolContextBuilder(policy=policy, network_policy=network_policy).build(
        request,
        result,
        profile=profile,
        decision=decision,
        mode=mode,
    )


__all__ = [
    "TOOL_CONTEXT_SCHEMA",
    "ToolContextBuilder",
    "ToolContextError",
    "ToolContextPolicy",
    "build_tool_result_context",
]
