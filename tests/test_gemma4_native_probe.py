"""Contracts for the isolated, fail-closed Gemma 4 native preflight."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from scripts.model_tools.cli import main
from scripts.model_tools.gemma4_native_probe import run_native_probe
from scripts.model_tools.gemma4_native_probe_worker import TOOL, execute_request


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(model: Path, mmproj: Path, *, require_audio: bool = False) -> dict:
    return {
        "schema_version": 1,
        "operation": "native_multimodal_preflight",
        "tool": TOOL,
        "read_only": True,
        "network_access": "disabled",
        "project_llama_cpp_revision": "fixture-revision",
        "artifact_requested": True,
        "model_path": str(model),
        "mmproj_path": str(mmproj),
        "model_sha256": _sha256(model),
        "mmproj_sha256": _sha256(mmproj),
        "n_ctx": 128,
        "require_audio": require_audio,
    }


def _fake_loader(*, vision: bool = True, audio: bool = True):
    closed: list[bool] = []

    class FakeLlama:
        def __init__(self, **kwargs):
            self.model = object()
            self.kwargs = kwargs

        def close(self):
            closed.append(True)

    context = object()
    mtmd = SimpleNamespace(
        mtmd_context_params_default=lambda: object(),
        mtmd_free=lambda value: closed.append(value is context),
        mtmd_init_from_file=lambda path, model, params: context,
        mtmd_support_audio=lambda value: audio,
        mtmd_support_vision=lambda value: vision,
    )
    binding = SimpleNamespace(__version__="fixture", Llama=FakeLlama)

    def loader(name: str):
        return {"llama_cpp": binding, "llama_cpp.mtmd_cpp": mtmd}[name]

    return loader, closed


def test_binding_import_failure_is_fail_closed_without_artifact_paths():
    result = execute_request(
        {
            "schema_version": 1,
            "operation": "native_multimodal_preflight",
            "tool": TOOL,
            "read_only": True,
            "network_access": "disabled",
            "artifact_requested": False,
        },
        module_loader=lambda _name: (_ for _ in ()).throw(ImportError("fixture")),
    )

    assert result["valid"] is True
    assert result["gate_passed"] is False
    assert result["status"] == "binding_unavailable"
    assert result["binding"]["mtmd_abi"] is False
    assert result["errors"][0]["code"] == "binding_import_failed"


def test_verified_artifacts_need_vision_and_do_not_expose_paths(tmp_path: Path):
    model = tmp_path / "private-model.gguf"
    mmproj = tmp_path / "private-mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projector")
    loader, closed = _fake_loader(vision=True, audio=False)

    result = execute_request(_request(model, mmproj), module_loader=loader)

    assert result["status"] == "ready_for_image_smoke"
    assert result["gate_passed"] is True
    assert result["artifacts"]["verified"] is True
    assert result["capabilities"]["vision"] is True
    assert result["capabilities"]["audio"] is False
    assert len(closed) == 2
    assert str(tmp_path) not in str(result)


def test_audio_requirement_digest_mismatch_and_resource_rejection_fail_closed(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projector")
    loader, _closed = _fake_loader(vision=True, audio=False)

    audio_required = execute_request(_request(model, mmproj, require_audio=True), module_loader=loader)
    mismatch = _request(model, mmproj)
    mismatch["mmproj_sha256"] = "0" * 64
    rejected = execute_request(mismatch, module_loader=loader)
    resource = _request(model, mmproj)
    resource["required_free_ram_bytes"] = 1024
    monkeypatch.setattr(
        "scripts.model_tools.gemma4_native_probe_worker._available_ram_bytes",
        lambda: 512,
    )
    resource_rejected = execute_request(resource, module_loader=loader)

    assert audio_required["status"] == "required_capability_missing"
    assert audio_required["gate_passed"] is False
    assert audio_required["errors"][0]["code"] == "audio_not_supported"
    assert rejected["status"] == "artifact_rejected"
    assert rejected["gate_passed"] is False
    assert rejected["errors"][0]["code"] == "artifact_verification_failed"
    assert str(tmp_path) not in str(rejected)
    assert resource_rejected["status"] == "resource_rejected"
    assert resource_rejected["resources"]["admitted"] is False
    assert resource_rejected["errors"][0]["code"] == "insufficient_ram"


def test_controller_requires_complete_explicit_artifact_identity():
    incomplete = run_native_probe(model=Path("candidate.gguf"))
    invalid_hash = run_native_probe(
        model=Path("candidate.gguf"),
        mmproj=Path("projector.gguf"),
        model_sha256="not-a-digest",
        mmproj_sha256="0" * 64,
    )

    assert incomplete["valid"] is False
    assert incomplete["status"] == "invalid_request"
    assert invalid_hash["valid"] is False
    assert "SHA-256" in invalid_hash["errors"][0]["message"]


def test_cli_binding_preflight_exits_zero_and_artifact_gate_is_reported(monkeypatch, capsys):
    binding_report = {
        "schema_version": 1,
        "tool": TOOL,
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "status": "binding_ready",
        "gate_passed": False,
        "binding": {"status": "available", "package_version": "fixture", "mtmd_abi": True},
        "artifacts": {"verified": False},
        "capabilities": {"vision": False, "audio": False},
        "errors": [],
    }
    monkeypatch.setattr("scripts.model_tools.cli.run_native_probe", lambda **_kwargs: binding_report)

    assert main(["gemma4-native-probe", "--json"]) == 0
    assert '"status": "binding_ready"' in capsys.readouterr().out
