"""Assignment-scoped manifest tests."""

import hashlib
import json
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "src")

from pipeline_assignment_manifest import build_assignment_manifest  # noqa: E402


def _fixture(root):
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "num_hidden_layers": 2}),
        encoding="utf-8",
    )
    save_file({
        "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.norm.weight": torch.zeros(2, dtype=torch.float16),
        "lm_head.weight": torch.zeros(4, 2, dtype=torch.float16),
    }, str(root / "model.safetensors"))
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_manifest_contains_only_assigned_keys_and_identity(tmp_path):
    _fixture(tmp_path)
    manifest = build_assignment_manifest(
        tmp_path,
        model_id="fixture",
        model_sha256="a" * 64,
        config_id="config-1",
        plan_id="plan-1",
        node_id="worker-1",
        start_layer=0,
        end_layer=1,
        total_layers=2,
        has_embedding=True,
        has_lm_head=False,
    )
    assert manifest["manifest_kind"] == "pytorch_pipeline_assignment"
    assert manifest["layer_range"] == [0, 1]
    assert manifest["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    weight_files = [item for item in manifest["files"] if item.get("kind") == "weights"]
    assert len(weight_files) == 1
    assert set(weight_files[0]["keys"]) == {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
    }
    assert all("model.layers.1" not in key for key in weight_files[0]["keys"])


def test_manifest_rejects_invalid_range(tmp_path):
    _fixture(tmp_path)
    try:
        build_assignment_manifest(
            tmp_path, model_id="fixture", model_sha256="a" * 64,
            config_id="config-1", plan_id="plan-1", node_id="worker-1",
            start_layer=2, end_layer=2, total_layers=2,
        )
    except ValueError as exc:
        assert "layer range" in str(exc)
    else:
        raise AssertionError("invalid assignment range was accepted")


def test_tied_output_assignment_includes_shared_embedding_weight(tmp_path):
    _fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_type"] = "qwen3"
    config["tie_word_embeddings"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    save_file({
        "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.norm.weight": torch.zeros(2, dtype=torch.float16),
    }, str(tmp_path / "model.safetensors"))

    manifest = build_assignment_manifest(
        tmp_path, model_id="qwen3", model_sha256="b" * 64,
        config_id="config-2", plan_id="plan-2", node_id="worker-2",
        start_layer=1, end_layer=2, total_layers=2,
        has_embedding=False, has_lm_head=True,
    )

    keys = {
        key
        for item in manifest["files"] if item.get("kind") == "weights"
        for key in item["keys"]
    }
    assert "model.embed_tokens.weight" in keys
    assert manifest["tie_word_embeddings"] is True
    assert manifest["output_weight_source"] == "model.embed_tokens.weight"
