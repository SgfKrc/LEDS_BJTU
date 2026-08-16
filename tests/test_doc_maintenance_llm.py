"""文档维护 Agent M2.1：配置、脱敏、批处理与严格判定协议。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_llm import (  # noqa: E402
    MAX_EXCERPT_BYTES,
    build_provider_payload,
    load_docagent_config,
    parse_judgement_response,
    prepare_judgement_batches,
    resolve_provider_plan,
    sanitize_text,
)


def _audit(doc: str, findings: list[dict]) -> dict:
    return {
        "docs": [{
            "doc": doc,
            "status_line": "> 状态：规划",
            "sha256": "a" * 64,
            "findings": findings,
            "related_commits": ["feat: update task graph"],
        }]
    }


def test_deepseek_requires_explicit_complete_configuration(tmp_path):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text(
        "DOCAGENT_DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
        "DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash\n"
        "DOCAGENT_DEEPSEEK_API_KEY=sk-super-secret-key\n",
        encoding="utf-8",
    )
    config = load_docagent_config(env_path)
    plan = resolve_provider_plan(config)
    assert [provider.name for provider in plan.providers] == ["ollama"]
    assert "sk-super-secret-key" not in repr(config)
    assert "sk-super-secret-key" not in repr(plan)


def test_deepseek_then_ollama_when_explicit_and_complete(tmp_path):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text(
        "DOCAGENT_PROVIDER=deepseek\n"
        "DOCAGENT_DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
        "DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash\n"
        "DOCAGENT_DEEPSEEK_API_KEY=sk-super-secret-key\n",
        encoding="utf-8",
    )
    plan = resolve_provider_plan(load_docagent_config(env_path))
    assert [provider.name for provider in plan.providers] == ["deepseek", "ollama"]


def test_opencode_provider_name_uses_existing_remote_configuration(tmp_path):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text(
        "DOCAGENT_PROVIDER=opencode\n"
        "DOCAGENT_DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
        "DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash\n"
        "DOCAGENT_DEEPSEEK_API_KEY=sk-super-secret-key\n",
        encoding="utf-8",
    )
    plan = resolve_provider_plan(load_docagent_config(env_path))
    assert [provider.name for provider in plan.providers] == ["opencode", "ollama"]


def test_dotenv_inline_comments_are_ignored_outside_quotes(tmp_path):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text(
        "DOCAGENT_PROVIDER=deepseek  # preferred remote\n"
        "DOCAGENT_DEEPSEEK_BASE_URL=https://example.invalid/v1 # endpoint\n"
        "DOCAGENT_DEEPSEEK_MODEL='model#variant'\n"
        "DOCAGENT_DEEPSEEK_API_KEY=sk-secret # not part of value\n",
        encoding="utf-8",
    )
    config = load_docagent_config(env_path)
    assert config.requested_provider == "deepseek"
    assert config.deepseek_model == "model#variant"
    assert config.deepseek_api_key == "sk-secret"


def test_explicit_ollama_never_enables_remote_provider(tmp_path):
    config = load_docagent_config(tmp_path / "missing", cli_provider="ollama")
    plan = resolve_provider_plan(config)
    assert [provider.name for provider in plan.providers] == ["ollama"]


def test_invalid_provider_value_is_not_echoed(tmp_path):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text("DOCAGENT_PROVIDER=sk-should-not-leak\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported DOCAGENT_PROVIDER") as exc_info:
        load_docagent_config(env_path)
    assert "sk-should-not-leak" not in str(exc_info.value)


@pytest.mark.parametrize("value", ["-0.1", "1.1", "not-a-number"])
def test_invalid_confidence_floor_is_rejected(tmp_path, value):
    env_path = tmp_path / ".env.docagent"
    env_path.write_text(f"DOCAGENT_CONFIDENCE_FLOOR={value}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_docagent_config(env_path)


def test_sanitize_text_removes_secrets_urls_and_paths():
    raw = (
        "API_KEY=sk-super-secret-key\n"
        "Bearer abc.def.secret\n"
        "https://example.com/private?q=1\n"
        "C:\\Users\\name\\secret.env\n"
        "src/private/worker.py docs/secret.md [link](../private.md)\n"
        "ghp_abcdefghijklmnopqrstuvwxyz\n"
    )
    safe = sanitize_text(raw)
    for forbidden in (
        "sk-super", "abc.def", "example.com", "C:\\Users", "src/private",
        "docs/secret", "../private", "ghp_",
    ):
        assert forbidden not in safe


def test_batches_group_findings_and_provider_payload_has_no_local_doc_path(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    doc = docs / "secret-plan.md"
    doc.write_text(
        "> 状态：规划\nAPI_KEY=sk-super-secret-key\n"
        "源码位于 src/private/worker.py，说明见 https://example.com/private\n"
        + "文档片段" * 3000,
        encoding="utf-8",
    )
    findings = [
        {"rule": "R3", "level": "info", "message": "src/private/worker.py"},
        {"rule": "R1", "level": "warn", "message": "secret"},
    ]
    batches = prepare_judgement_batches(_audit("docs/secret-plan.md", findings), repo)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.findings) == 2
    assert len(batch.excerpt.encode("utf-8")) <= MAX_EXCERPT_BYTES
    payload_text = json.dumps(build_provider_payload(batch), ensure_ascii=False)
    for forbidden in (
        "docs/secret-plan.md", "src/private", "example.com", "sk-super-secret-key",
    ):
        assert forbidden not in payload_text


def test_canonical_hash_is_stable_when_finding_order_changes(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("> 状态：规划\n", encoding="utf-8")
    findings = [
        {"rule": "R1", "level": "warn"},
        {"rule": "R3", "level": "info"},
    ]
    first = prepare_judgement_batches(_audit("docs/x.md", findings), repo)[0]
    second = prepare_judgement_batches(_audit("docs/x.md", list(reversed(findings))), repo)[0]
    assert first.cache_key == second.cache_key


def test_strict_response_accepts_valid_json_and_applies_confidence_floor(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("> 状态：规划\n", encoding="utf-8")
    batch = prepare_judgement_batches(
        _audit("docs/x.md", [{"rule": "R1", "level": "warn"}]), repo
    )[0]
    raw = json.dumps({
        "doc": batch.doc_ref,
        "judgement": "stale",
        "confidence": 0.59,
        "suggestion": "改为已完成",
    })
    result = parse_judgement_response(raw, batch, confidence_floor=0.6)
    assert result.judgement == "needs_review"
    assert result.error == "below_confidence_floor"


@pytest.mark.parametrize("raw,error", [
    ("```json\n{}\n```", "invalid_json"),
    ('{"doc":"wrong","judgement":"accurate","confidence":0.9,"suggestion":"ok"}',
     "doc_ref_mismatch"),
    ('{"doc":"REF","judgement":"yes","confidence":0.9,"suggestion":"ok"}',
     "doc_ref_mismatch"),
])
def test_invalid_response_fails_closed(tmp_path, raw, error):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("> 状态：规划\n", encoding="utf-8")
    batch = prepare_judgement_batches(
        _audit("docs/x.md", [{"rule": "R1", "level": "warn"}]), repo
    )[0]
    result = parse_judgement_response(raw, batch)
    assert result.judgement == "needs_review"
    assert result.error == error
