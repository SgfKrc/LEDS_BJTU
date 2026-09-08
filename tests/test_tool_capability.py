"""Offline tool-capability registry contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.model_tools.cli import main
from scripts.model_tools.tool_capability import probe_model_asset, validate_tool_fixture


def _write_asset(root: Path, *, with_tools: bool = True) -> Path:
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"fixture weights")
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        encoding="utf-8",
    )
    template = "{{ tools | tojson }} {% if role == 'tool' %} tool_result {% endif %} {{ arguments | tojson }}" if with_tools else "{{ messages }}"
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def _fixture() -> dict:
    return {
        "assistant": {"tool_calls": [{"name": "web_search", "arguments": {"query": "QLH", "top_k": 3}}]},
        "tool_result": {"role": "tool", "name": "web_search", "content": {"status": "ok", "items": []}},
        "final_response": "已完成检索。",
    }


def test_tool_fixture_distinguishes_three_capabilities():
    result = validate_tool_fixture(_fixture())
    assert result["valid"] is True
    assert result["capabilities"] == {
        "json_output": "verified",
        "tool_call_generation": "verified",
        "tool_result_reinjection": "verified",
    }

    invalid = validate_tool_fixture({**_fixture(), "assistant": {"tool_calls": [{"name": "unknown", "arguments": {}}]}})
    assert invalid["valid"] is False
    assert invalid["capabilities"]["tool_call_generation"] == "rejected"


def test_static_probe_is_read_only_and_does_not_admit_unverified_model(tmp_path: Path):
    asset = _write_asset(tmp_path / "private-model")
    report = probe_model_asset(asset, model_id="fixture-qwen", sidecar_version="littlelamb-0.1")

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["network_used"] is False
    assert report["weights_loaded"] is False
    assert report["admission"]["status"] == "unknown"
    assert report["admission"]["production_eligible"] is False
    assert report["capabilities"]["tool_call_generation"]["status"] == "declared"
    assert report["capabilities"]["json_output"]["status"] == "declared"
    assert str(tmp_path) not in json.dumps(report)
    assert report["model"]["path_digest"]


def test_fixture_verification_stays_candidate_until_runtime_gate(tmp_path: Path):
    report = probe_model_asset(_write_asset(tmp_path / "fixture"), fixture=_fixture())
    assert report["valid"] is True
    assert report["admission"]["status"] == "candidate"
    assert report["admission"]["production_eligible"] is False
    assert all(item["status"] == "verified" for item in report["capabilities"].values())


def test_cli_can_probe_explicit_asset_and_keep_fixture_offline(tmp_path: Path, capsys):
    asset = _write_asset(tmp_path / "cli-model", with_tools=False)
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(_fixture(), ensure_ascii=False), encoding="utf-8")
    assert main(["tool_capability_probe", "--model", str(asset), "--fixture", str(fixture_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"network_used": false' in output
    assert '"weights_loaded": false' in output
    assert str(tmp_path) not in output

