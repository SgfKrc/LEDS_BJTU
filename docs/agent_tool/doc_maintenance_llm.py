#!/usr/bin/env python3
"""文档维护 Agent M2.1：离线协议、配置、脱敏与批处理基础层。

本模块不发起网络请求。M2.3 才会为 ``JudgementProvider`` 接入 HTTP provider。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

MAX_EXCERPT_BYTES = 4096
MAX_SUGGESTION_BYTES = 1024
VALID_JUDGEMENTS = {"stale", "accurate", "needs_review"}
# ``opencode`` 是 opencode go 网关的明确名称；``deepseek`` 保留为早期配置别名。
VALID_PROVIDERS = {"opencode", "deepseek", "ollama"}

RULE_SIGNALS = {
    "R1": "状态行与完成标记可能矛盾",
    "R2": "文档存在未提交改动",
    "R3": "关联代码提交晚于文档更新日期",
    "R4": "检测到失效的内部链接",
    "R5": "文档头部缺少状态行",
}

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?im)^(\s*(?:export\s+)?[A-Z0-9_.-]*"
    r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|GRANT|BLOB|URL|PATH)"
    r"[A-Z0-9_.-]*\s*[:=]\s*).*$"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_TOKEN_RE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{8,}|gh[pousr]_[a-z0-9_]{12,}|hf_[a-z0-9]{12,})\b"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*\b")
_URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s`<>\"]+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\[^\s`<>\"]+")
_ABS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|tmp|var|opt|mnt|srv)/[^\s`<>\"]+")
_REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.\.?/)*(?:src|docs|tests|scripts|models|build)/[^\s`<>\")\]]+"
)
_MARKDOWN_TARGET_RE = re.compile(r"(\]\()([^)]+)(\))")
_FILE_REF_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+\.)"
    r"(?:py|md|json|ya?ml|toml|ini|env|sqlite3?|db|log|bin|safetensors)(?![A-Za-z0-9_])"
)


class JudgementProvider(Protocol):
    """M2 provider 的最小接口；具体联网实现留给 M2.3。"""

    name: str
    model: str

    def judge(self, payload: dict) -> str:
        """返回 provider 的原始 JSON 文本。"""
        ...


@dataclass(frozen=True)
class DocAgentConfig:
    requested_provider: str
    provider_explicit: bool
    confidence_floor: float = 0.6
    deepseek_base_url: str = ""
    deepseek_model: str = ""
    deepseek_api_key: str = field(default="", repr=False, compare=False)
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "gemma4:12b"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True)
class ProviderPlan:
    providers: tuple[ProviderSpec, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgementBatch:
    """单文档批次。doc 仅供本地映射，绝不进入 provider_input。"""

    doc: str = field(repr=False)
    doc_ref: str
    doc_sha256: str
    status_line: str
    excerpt: str
    findings: tuple[dict, ...]
    related_commits: tuple[str, ...]

    def provider_input(self) -> dict:
        return {
            "doc": self.doc_ref,
            "status_line": self.status_line,
            "excerpt": self.excerpt,
            "findings": list(self.findings),
            "related_commits": list(self.related_commits),
        }

    def canonical_input(self) -> dict:
        value = self.provider_input()
        value["doc_sha256"] = self.doc_sha256
        return value

    @property
    def cache_key(self) -> str:
        encoded = json.dumps(
            self.canonical_input(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Judgement:
    doc_ref: str
    judgement: str
    confidence: float
    suggestion: str
    error: str | None = None


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 兼容 .env 常见的 ``KEY=value  # 注释``，但保留引号中的 #。
        quote: str | None = None
        for index, char in enumerate(value):
            if char in {"'", '"'}:
                if quote is None:
                    quote = char
                elif quote == char:
                    quote = None
            elif char == "#" and quote is None and (
                index == 0 or value[index - 1].isspace()
            ):
                value = value[:index].rstrip()
                break
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key.startswith("DOCAGENT_"):
            values[key] = value
    return values


def _valid_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not (
        parsed.username or parsed.password
    )


def load_docagent_config(path: Path, cli_provider: str | None = None) -> DocAgentConfig:
    """只读取独立 .env.docagent；不会合并主 .env 或进程环境。"""
    values = _read_env_file(path)
    env_provider = values.get("DOCAGENT_PROVIDER", "").strip().lower()
    requested = (cli_provider or env_provider or "deepseek").strip().lower()
    if requested not in VALID_PROVIDERS:
        raise ValueError("unsupported DOCAGENT_PROVIDER")
    explicit = cli_provider is not None or bool(env_provider)
    try:
        confidence_floor = float(values.get("DOCAGENT_CONFIDENCE_FLOOR", "0.6"))
    except ValueError as exc:
        raise ValueError("DOCAGENT_CONFIDENCE_FLOOR must be a number") from exc
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("DOCAGENT_CONFIDENCE_FLOOR must be between 0 and 1")
    return DocAgentConfig(
        requested_provider=requested,
        provider_explicit=explicit,
        confidence_floor=confidence_floor,
        deepseek_base_url=values.get("DOCAGENT_DEEPSEEK_BASE_URL", "").rstrip("/"),
        deepseek_model=values.get("DOCAGENT_DEEPSEEK_MODEL", ""),
        deepseek_api_key=values.get("DOCAGENT_DEEPSEEK_API_KEY", ""),
        ollama_base_url=values.get(
            "DOCAGENT_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
        ).rstrip("/"),
        ollama_model=values.get("DOCAGENT_OLLAMA_MODEL", "gemma4:12b"),
    )


def resolve_provider_plan(config: DocAgentConfig) -> ProviderPlan:
    """生成 fail-closed 回退计划；这里只校验配置，不探测网络。"""
    providers: list[ProviderSpec] = []
    warnings: list[str] = []
    if config.requested_provider in {"opencode", "deepseek"}:
        complete = (
            config.provider_explicit
            and _valid_base_url(config.deepseek_base_url)
            and bool(config.deepseek_model)
            and bool(config.deepseek_api_key)
        )
        if complete:
            providers.append(ProviderSpec(
                name=config.requested_provider, base_url=config.deepseek_base_url,
                model=config.deepseek_model, api_key=config.deepseek_api_key,
            ))
        else:
            warnings.append("remote_provider_disabled_incomplete_or_not_explicit")

    if _valid_base_url(config.ollama_base_url) and config.ollama_model:
        providers.append(ProviderSpec(
            name="ollama", base_url=config.ollama_base_url,
            model=config.ollama_model,
        ))
    else:
        warnings.append("ollama_disabled_invalid_config")
    return ProviderPlan(tuple(providers), tuple(warnings))


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def sanitize_text(value: str, max_bytes: int = MAX_EXCERPT_BYTES) -> str:
    """清除 provider 禁止字段，并按 UTF-8 字节数截断。"""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)
    text = _TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED_TOKEN]", text)
    text = _MARKDOWN_TARGET_RE.sub(r"\1[REDACTED_TARGET]\3", text)
    text = _URL_RE.sub("[REDACTED_URL]", text)
    text = _WINDOWS_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _ABS_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _REPO_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _FILE_REF_RE.sub("[REDACTED_FILE]", text)
    normalized = "\n".join(" ".join(line.split()) for line in text.splitlines())
    return _truncate_utf8(normalized.strip(), max_bytes)


def _commit_titles(record: dict) -> tuple[str, ...]:
    titles: list[str] = []
    for value in record.get("related_commits", ()):
        title = value.get("subject", "") if isinstance(value, dict) else str(value)
        sanitized = sanitize_text(title, 512)
        if sanitized:
            titles.append(sanitized)
    return tuple(sorted(set(titles)))


def prepare_judgement_batches(audit: dict, repo_root: Path) -> list[JudgementBatch]:
    """把 M1 结果按文档合并为稳定、可缓存且可安全发送的批次。"""
    docs_root = (repo_root / "docs").resolve()
    batches: list[JudgementBatch] = []
    for record in audit.get("docs", ()): 
        findings = record.get("findings") or []
        if not findings:
            continue
        rel = str(record.get("doc", ""))
        path = (repo_root / rel).resolve()
        if path != docs_root and docs_root not in path.parents:
            raise ValueError(f"audit document escapes docs root: {rel}")
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        doc_sha256 = str(record.get("sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest())
        safe_findings = tuple(sorted(
            ({
                "rule": str(finding.get("rule", "unknown")),
                "level": str(finding.get("level", "info")),
                "signal": RULE_SIGNALS.get(
                    str(finding.get("rule", "")), "检测到未分类的机械化信号"
                ),
            } for finding in findings),
            key=lambda item: (item["rule"], item["level"], item["signal"]),
        ))
        batches.append(JudgementBatch(
            doc=rel,
            doc_ref=hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16],
            doc_sha256=doc_sha256,
            status_line=sanitize_text(str(record.get("status_line", "")), 512),
            excerpt=sanitize_text(text[:16384], MAX_EXCERPT_BYTES),
            findings=safe_findings,
            related_commits=_commit_titles(record),
        ))
    return sorted(batches, key=lambda batch: batch.doc_ref)


def build_provider_payload(batch: JudgementBatch) -> dict:
    """构造不含本地路径、密钥和 URL 的单文档判定请求。"""
    return {
        "task": "判断文档状态是否过时；不确定时必须返回 needs_review",
        "input": batch.provider_input(),
        "response_schema": {
            "doc": batch.doc_ref,
            "judgement": "stale|accurate|needs_review",
            "confidence": "number:0..1",
            "suggestion": "string",
        },
    }


def _needs_review(batch: JudgementBatch, error: str) -> Judgement:
    return Judgement(
        doc_ref=batch.doc_ref,
        judgement="needs_review",
        confidence=0.0,
        suggestion="provider 输出无效，请人工核对",
        error=error,
    )


def parse_judgement_response(raw: str, batch: JudgementBatch,
                             confidence_floor: float = 0.6) -> Judgement:
    """严格校验 provider JSON；任何偏差均 fail-closed 到 needs_review。"""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return _needs_review(batch, "invalid_json")
    required = {"doc", "judgement", "confidence", "suggestion"}
    if not isinstance(value, dict) or set(value) != required:
        return _needs_review(batch, "invalid_schema")
    if value["doc"] != batch.doc_ref:
        return _needs_review(batch, "doc_ref_mismatch")
    judgement = value["judgement"]
    confidence = value["confidence"]
    suggestion = value["suggestion"]
    if judgement not in VALID_JUDGEMENTS:
        return _needs_review(batch, "invalid_judgement")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return _needs_review(batch, "invalid_confidence")
    if not 0.0 <= float(confidence) <= 1.0 or not isinstance(suggestion, str):
        return _needs_review(batch, "invalid_schema_value")
    final_judgement = judgement if float(confidence) >= confidence_floor else "needs_review"
    return Judgement(
        doc_ref=batch.doc_ref,
        judgement=final_judgement,
        confidence=float(confidence),
        suggestion=sanitize_text(suggestion, MAX_SUGGESTION_BYTES),
        error="below_confidence_floor" if final_judgement != judgement else None,
    )
