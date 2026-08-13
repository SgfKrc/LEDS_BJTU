from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.model_tools.cli import main
from scripts.model_tools.qwen3_sidecar_probe import run_qwen3_sidecar_probe
from scripts.model_tools.qwen3_sidecar_probe_worker import execute_request


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "qwen3"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "transformers_version": "4.51.0",
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fixture"}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    files = []
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        data = (root / name).read_bytes()
        import hashlib
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (root / ".qlh-model-asset.json").write_text(json.dumps({
        "schema_version": 1,
        "artifact_kind": "transformers_safetensors",
        "asset": {"asset_id": "qwen3-4b", "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "revision": "fixture"},
        "files": files,
    }), encoding="utf-8")
    return root


def _loader(*, version="4.51.0", rendered="<|im_start|>assistant\n"):
    class Tokenizer:
        chat_template = "fixture"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return rendered

    class Config:
        model_type = "qwen3"

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return Config()

    return lambda name: SimpleNamespace(__version__=version, AutoTokenizer=Tokenizer, AutoConfig=AutoConfig)


def test_qwen3_preflight_passes_config_tokenizer_and_hard_thinking_off(tmp_path: Path):
    result = execute_request({"model_path": str(_artifact(tmp_path))}, module_loader=_loader())

    assert result["status"] == "ready_for_qwen3_smoke"
    assert result["gate_passed"] is True
    assert result["artifact"]["model_type"] == "qwen3"
    assert result["tokenizer"]["enable_thinking_supported"] is True
    assert result["tokenizer"]["rendered_without_thinking"] is True
    assert str(tmp_path) not in str(result)


def test_qwen3_preflight_rejects_old_transformers_and_thinking_output(tmp_path: Path):
    artifact = _artifact(tmp_path)
    old = execute_request({"model_path": str(artifact)}, module_loader=_loader(version="4.47.1"))
    thinking = execute_request({"model_path": str(artifact)}, module_loader=_loader(rendered="<think>hidden</think>answer"))

    assert old["status"] == "runtime_rejected"
    assert old["errors"][0]["code"] == "transformers_too_old"
    assert thinking["status"] == "preflight_failed"
    assert thinking["errors"][0]["code"] == "tokenizer_preflight_failed"


def test_qwen3_preflight_accepts_official_empty_thinking_scaffold(tmp_path: Path):
    result = execute_request(
        {"model_path": str(_artifact(tmp_path))},
        module_loader=_loader(rendered="<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    )

    assert result["gate_passed"] is True
    assert result["tokenizer"]["rendered_without_thinking"] is True


def test_qwen3_controller_requires_model_and_accepts_isolated_worker_result(tmp_path: Path):
    missing = run_qwen3_sidecar_probe(model=None)
    result = run_qwen3_sidecar_probe(
        model=tmp_path / "qwen3",
        worker_runner=lambda request, timeout: {"tool": "qwen3_sidecar_probe", "schema_version": 1, "gate_passed": True},
    )

    assert missing["status"] == "invalid_request"
    assert missing["errors"][0]["code"] == "model_path_required"
    assert result["gate_passed"] is True


def test_qwen3_cli_reports_preflight(monkeypatch, capsys, tmp_path: Path):
    report = {
        "tool": "qwen3_sidecar_probe", "valid": True, "gate_passed": True,
        "status": "ready_for_qwen3_smoke",
        "runtime": {"transformers_version": "4.51.0", "isolated": True},
        "artifact": {"model_type": "qwen3"},
        "tokenizer": {"loaded": True, "chat_template_available": True,
                       "enable_thinking_supported": True, "rendered_without_thinking": True},
        "errors": [],
    }
    monkeypatch.setattr("scripts.model_tools.cli.run_qwen3_sidecar_probe", lambda **kwargs: report)

    assert main(["qwen3-sidecar-probe", "--model", str(tmp_path), "--json"]) == 0
    assert '"ready_for_qwen3_smoke"' in capsys.readouterr().out
