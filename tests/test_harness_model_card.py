import hashlib
import json
import struct

import pytest

from harness_workbench.model_profiles import builtin_profiles
from harness_workbench.tools import (
    MODEL_CARD_SCHEMA,
    build_model_card_from_health,
    build_model_card_from_json,
)
from harness_workbench.tools.model_card import main


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_fixture() -> bytes:
    metadata = (
        _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("qwen"),
        _gguf_string("qwen.context_length") + struct.pack("<II", 4, 8192),
        _gguf_string("general.file_type") + struct.pack("<II", 4, 15),
        _gguf_string("tokenizer.ggml.model") + struct.pack("<I", 8) + _gguf_string("gpt2"),
    )
    return b"GGUF" + struct.pack("<IQQ", 3, 195, len(metadata)) + b"".join(metadata)


def _models_fixture(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "models"
    models.mkdir(parents=True)
    artifact = models / "Qwen-1_8B-Chat.Q4_K_M.gguf"
    payload = _gguf_fixture()
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    artifact.with_name(artifact.name + ".sha256").write_text(digest, encoding="utf-8")
    (repo / ".gitignore").write_text("models/*.gguf\n", encoding="utf-8")
    return repo, models


def test_model_card_reads_selected_gguf_header_metadata_without_weights(tmp_path):
    _, models = _models_fixture(tmp_path)

    report = build_model_card_from_health(models)

    header = report.metadata.gguf_header
    assert report.valid is True
    assert report.status == "complete"
    assert report.weights_loaded is False
    assert report.network_used is False
    assert report.metadata.context_length == 8192
    assert report.metadata.quantization == "Q4_K_M"
    assert report.metadata.parameter_hint == "1.8B (filename hint)"
    assert header["read_status"] == "ok"
    assert header["tensor_count"] == 195
    assert header["metadata"]["general.architecture"] == "qwen"


def test_model_card_merges_candidate_profile_without_claiming_unknown_capabilities(tmp_path):
    _, models = _models_fixture(tmp_path)
    profile = next(item for item in builtin_profiles() if item.model_id == "QW1.8B")

    report = build_model_card_from_health(models, profile=profile)

    assert report.model_id == "QW1.8B"
    assert report.profile_digest == profile.digest
    assert report.metadata.backend == "llama_server"
    assert report.metadata.revision == "builtin-qw1-v1"
    assert report.metadata.profile_status == "candidate"
    assert report.metadata.production_eligible is False
    assert report.metadata.context_length == 8192
    assert report.metadata.capabilities == ()
    assert report.metadata.chat_template == "qwen_chat_v1"
    assert any("configured as 4096" in item for item in report.metadata.notes)


def test_model_card_marks_manifest_gaps_incomplete_but_auditable(tmp_path):
    _, models = _models_fixture(tmp_path)
    (models / "missing.lock.json").write_text(
        json.dumps({"artifacts": {"main": {"path": "missing.gguf"}}}),
        encoding="utf-8",
    )

    report = build_model_card_from_health(models)

    assert report.status == "incomplete"
    assert report.valid is True
    assert report.manifest_health["valid"] is False
    assert "missing.lock.json:artifacts.main:missing" in report.manifest_health["errors"]
    assert any("before release" in item for item in report.limitations)


def test_model_card_missing_root_fails_structurally(tmp_path):
    report = build_model_card_from_health(tmp_path / "missing")

    assert report.status == "invalid"
    assert report.valid is False
    assert report.checks["artifacts_present"] is False
    assert report.manifest_health["errors"] == ["model_root_missing"]


def test_model_card_digest_and_markdown_are_stable_and_redacted(tmp_path):
    _, models = _models_fixture(tmp_path)
    first = build_model_card_from_health(models)
    second = build_model_card_from_health(models)

    markdown = first.to_markdown()
    assert first.digest == second.digest
    assert first.as_dict()["schema"] == MODEL_CARD_SCHEMA
    assert str(tmp_path) not in json.dumps(first.as_dict())
    assert "quality or performance claim" in markdown
    assert "weights loaded: `false`" in markdown


def test_model_card_accepts_manifest_health_json(tmp_path):
    _, models = _models_fixture(tmp_path)
    source_report = build_model_card_from_health(models).manifest_health
    health = {
        "schema": "qlh.harness.manifest_health.v1",
        "root_label": "models",
        "valid": source_report["valid"],
        "artifacts": [],
        "manifests": [],
        "scanned_file_count": source_report["scanned_file_count"],
        "ignored_file_count": source_report["ignored_file_count"],
        "warnings": [],
        "errors": [],
    }
    path = tmp_path / "health.json"
    path.write_text(json.dumps(health), encoding="utf-8")

    report = build_model_card_from_json(path)

    assert report.status == "invalid"
    assert report.manifest_health["scanned_file_count"] == 2


def test_model_card_rejects_unsafe_health_paths(tmp_path):
    health = {
        "schema": "qlh.harness.manifest_health.v1",
        "root_label": "models",
        "artifacts": [{"path": "C:/secret/model.gguf", "kind": "gguf", "size_bytes": 1, "sha256_status": "missing"}],
        "manifests": [],
        "scanned_file_count": 1,
        "ignored_file_count": 0,
        "warnings": [],
        "errors": [],
    }

    with pytest.raises(ValueError, match="unsafe path"):
        build_model_card_from_health(tmp_path, health_report=health)


def test_model_card_rejects_stale_or_invalid_gguf_as_complete(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "broken.gguf"
    artifact.write_bytes(b"not-a-gguf")

    report = build_model_card_from_health(models)

    assert report.status == "incomplete"
    assert report.valid is False
    assert report.checks["gguf_headers_readable"] is False
    assert report.metadata.gguf_header["read_status"] == "not_gguf_or_truncated"


def test_model_card_json_input_requires_known_schema(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"schema": "unknown.v1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model card input schema"):
        build_model_card_from_json(path)


def test_model_card_cli_writes_json_and_markdown(tmp_path):
    _, models = _models_fixture(tmp_path)
    json_path = tmp_path / "card.json"
    markdown_path = tmp_path / "card.md"

    result = main([
        "--root", str(models),
        "--model-id", "QW1.8B",
        "--json", str(json_path),
        "--markdown", str(markdown_path),
    ])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["model_id"] == "QW1.8B"
    assert payload["weights_loaded"] is False
    assert markdown_path.read_text(encoding="utf-8").startswith("# QW1.8B model card")


def test_model_card_cli_rejects_unknown_builtin_profile(tmp_path):
    _, models = _models_fixture(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(["--root", str(models), "--model-id", "missing-model"])

    assert exc.value.code == 2
