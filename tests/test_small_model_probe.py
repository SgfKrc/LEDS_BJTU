"""B1/DSW-D1 offline model probe contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.model_tools import small_model_probe as probe


def _write_fixture(root: Path, template: str) -> Path:
    root.mkdir()
    values = {
        "config.json": {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 40960},
        "tokenizer_config.json": {"chat_template": template, "eos_token": "<|im_end|>"},
        "generation_config.json": {"eos_token_id": [151645, 151643]},
        "tokenizer.json": {"fixture": True},
        "model.safetensors": "weights are not read",
    }
    manifest_files = []
    for name, value in values.items():
        path = root / name
        if isinstance(value, dict):
            path.write_text(json.dumps(value), encoding="utf-8")
        else:
            path.write_text(value, encoding="utf-8")
        data = path.read_bytes()
        manifest_files.append({"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (root / "model.manifest.json").write_text(json.dumps({"source": "fixture/model", "files": manifest_files}), encoding="utf-8")
    return root


def test_directory_probe_extracts_template_switch_and_manifest_without_weights(tmp_path: Path) -> None:
    model = _write_fixture(tmp_path / "qwen3", "{{ enable_thinking }} {{ messages }}")
    report = probe.probe_artifact(model, model_id="qwen3-0.6b")

    assert report["valid"] is True
    assert report["weights_loaded"] is False
    assert report["network_used"] is False
    assert report["template"]["thinking"]["status"] == "declared"
    assert report["manifest"]["valid"] is True
    assert report["manifest"]["content_hashes_checked"] == 4


def test_b1_report_keeps_runtime_smoke_explicit(monkeypatch) -> None:
    config = SimpleNamespace(model_id="minicpm4-0.5b")
    valid = {"valid": True, "format": "safetensors", "template": {"thinking": {"status": "not_declared"}}}
    gguf = {"valid": True, "format": "gguf", "architecture": "minicpm"}
    monkeypatch.setattr(probe, "_core_config", lambda _model_id: (config, Path("model-dir"), Path("model.gguf")))
    monkeypatch.setattr(probe, "probe_artifact", lambda path, **_kwargs: valid if Path(path).name == "model-dir" else gguf)
    monkeypatch.setattr(probe, "_architecture_probe", lambda *_args: {"status": "metadata_pass"})

    report = probe.run_b1_probe(model_ids=["minicpm4-0.5b"])

    assert report["gate_passed"] is True
    assert report["weights_loaded"] is False
    assert report["gguf_smoke"] is None


def test_dsw_d1_report_combines_existing_gguf_and_read_only_conversion_plan(monkeypatch) -> None:
    config = SimpleNamespace(model_id="distilqwen25-ds3-0324-7b", name="DistilQwen2.5-DS3-0324-7B")
    source = {"valid": True, "format": "safetensors"}
    gguf = {"valid": True, "format": "gguf", "architecture": "qwen2"}
    monkeypatch.setattr(probe, "_core_config", lambda _model_id: (config, Path("source"), Path("output.gguf")))
    monkeypatch.setattr(probe, "probe_artifact", lambda path, **_kwargs: source if Path(path).name == "source" else gguf)
    monkeypatch.setattr(probe, "_architecture_probe", lambda *_args: {"status": "metadata_pass"})
    monkeypatch.setattr(probe, "plan_conversion", lambda **_kwargs: {"valid": True, "toolchain": {}, "space": {}, "errors": []})

    report = probe.run_dsw_d1()

    assert report["valid"] is True
    assert report["gate_passed"] is True
    assert report["read_only"] is True
    assert report["conversion"]["plan_valid"] is True
