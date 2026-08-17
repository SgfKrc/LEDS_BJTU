import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_model_assets import discover_local_model_assets


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discovery_pairs_manifest_safetensors_with_filesystem_gguf(tmp_path):
    root = tmp_path / "models"
    weights = root / "qwen3-5-2b"
    gguf = root / "qwen3-5-2b-gguf"
    weights.mkdir(parents=True)
    gguf.mkdir()
    _write_json(weights / "config.json", {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "max_position_embeddings": 262144,
    })
    (weights / "model.safetensors").write_bytes(b"weights")
    _write_json(weights / ".qlh-model-asset.json", {
        "schema_version": 1,
        "artifact_kind": "transformers_safetensors",
        "asset": {"asset_id": "qwen3-5-2b", "repo_id": "Qwen/Qwen3.5-2B"},
        "files": [
            {"path": "config.json"},
            {"path": "model.safetensors"},
        ],
    })
    (gguf / "Qwen3.5-2B-Q4_K_M.gguf").write_bytes(b"gguf")

    inventory = discover_local_model_assets(root)

    assert inventory["summary"]["total"] == 1
    asset = inventory["assets"][0]
    assert asset["model_id"] == "qwen3-5-2b"
    assert asset["huggingface_id"] == "Qwen/Qwen3.5-2B"
    assert asset["model_type"] == "both"
    assert asset["available_formats"] == ["safetensors", "gguf"]
    assert asset["max_context"] == 262144
    assert asset["runtime_profile"] == "qwen3_sidecar"
    assert asset["runtime_status"] == "inventory_only"
    assert asset["runtime_action"] == "qwen3_preflight"
    assert asset["model_path"].endswith("qwen3-5-2b")
    assert asset["gguf_path"].endswith("Qwen3.5-2B-Q4_K_M.gguf")


def test_discovery_ignores_incomplete_manifest_and_non_llm_directory(tmp_path):
    root = tmp_path / "models"
    incomplete = root / "qwen3-4b"
    incomplete.mkdir(parents=True)
    _write_json(incomplete / "config.json", {"model_type": "qwen3"})
    (incomplete / "model.safetensors").write_bytes(b"weights")
    _write_json(incomplete / ".qlh-model-asset.json", {
        "artifact_kind": "transformers_safetensors",
        "asset": {"asset_id": "qwen3-4b"},
        "files": [{"path": "config.json"}, {"path": "missing.safetensors"}],
    })
    non_llm = root / "diffusion"
    non_llm.mkdir()
    _write_json(non_llm / "config.json", {"model_type": "controlnet"})
    (non_llm / "model.safetensors").write_bytes(b"weights")
    unsupported = root / "opaque-package"
    unsupported.mkdir()
    (unsupported / "model.gguf").write_bytes(b"gguf")
    _write_json(unsupported / ".qlh-model-asset.json", {
        "artifact_kind": "unknown_format",
        "asset": {"asset_id": "opaque-package"},
        "files": [{"path": "model.gguf"}],
    })

    inventory = discover_local_model_assets(root)

    assert inventory == {"assets": [], "summary": {"total": 0, "total_bytes": 0}}
