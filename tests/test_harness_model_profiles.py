"""S1.5 model profile, probe and capability-gate contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_workbench.model_profiles import (
    CapabilityGate,
    CapabilityState,
    ModelProfile,
    ProfileRegistry,
    builtin_profiles,
    profile_diff,
    probe_local_model,
)
from harness_workbench.model_profiles.schema import ProfileValidationError


def _verified_capabilities() -> dict[str, CapabilityState]:
    return {
        name: CapabilityState("verified", ("runtime_fixture_v1",))
        for name in (
            "json_output",
            "tool_call_generation",
            "tool_result_reinjection",
            "multimodal",
            "thinking_control",
        )
    }


def _profile(*, status: str = "candidate", production: bool = False) -> ModelProfile:
    return ModelProfile(
        model_id="test-small",
        revision="fixture-v1",
        backend="llama_server",
        artifact_sha256="a" * 64,
        tokenizer_digest="b" * 64,
        chat_template_digest="c" * 64,
        context={"n_ctx": 2048, "input_budget": 1400, "max_new_tokens": 256},
        generation={"temperature": 0.7, "stop": ["<eos>"]},
        adaptation={"prompt_family": "generic_chat_v1", "tool_mode": "host_router"},
        roles=("answer", "tool_router"),
        resources={"kv_cache": "q8_0"},
        capabilities=_verified_capabilities(),
        status=status,
        production_eligible=production,
        evidence={
            "fixture_set": "small-model-core-v1",
            "runtime_verified": True,
            "artifact_digest_mode": "full_stream",
        },
    )


def test_builtin_profiles_are_conservative_candidates() -> None:
    profiles = builtin_profiles()
    assert {profile.model_id for profile in profiles} == {
        "QW1.8B",
        "Qwen3-0.6B",
        "Qwen2.5-0.5B",
        "MiniCPM4-0.5B",
        "DistilQwen2.5-DS3-0324-7B",
        "Gemma-small",
    }
    assert all(profile.status == "candidate" for profile in profiles)
    assert all(profile.production_eligible is False for profile in profiles)
    assert all(
        profile.capabilities["tool_call_generation"].status == "unknown"
        for profile in profiles
    )
    by_id = {profile.model_id: profile for profile in profiles}
    assert by_id["Qwen2.5-0.5B"].adaptation["prompt_family"] == "qwen_chat_v1"
    assert by_id["MiniCPM4-0.5B"].adaptation["prompt_family"] == "minicpm4_chat_v1"
    assert by_id["DistilQwen2.5-DS3-0324-7B"].context["max_new_tokens"] == 1024


def test_profile_round_trip_and_digest_rejects_tampering() -> None:
    profile = _profile()
    encoded = profile.as_dict()
    assert ModelProfile.from_dict(encoded).digest == profile.digest
    encoded["generation"]["temperature"] = 0.1
    with pytest.raises(ProfileValidationError):
        ModelProfile.from_dict(encoded)


def test_profile_rejects_absolute_paths() -> None:
    with pytest.raises(ProfileValidationError):
        _profile().with_updates(resources={"model_path": "C:\\private\\model.gguf"})


def test_unknown_tool_capability_cannot_open_autonomous_mode() -> None:
    profile = _profile().with_updates(
        capabilities={
            **_verified_capabilities(),
            "tool_call_generation": CapabilityState(),
            "tool_result_reinjection": CapabilityState(),
        }
    )
    decision = CapabilityGate().evaluate(profile, role="tool_router")
    assert decision.status == "candidate"
    assert decision.can("host_router")
    assert not decision.can("autonomous_tools")
    assert "tool_router_requires_verified_tool_contract" in decision.reasons


def test_verified_profile_can_be_production_eligible_only_with_bound_evidence() -> None:
    profile = _profile(status="verified", production=True)
    decision = CapabilityGate().evaluate(profile)
    assert decision.status == "verified"
    assert decision.production_eligible is True
    assert decision.can("autonomous_tools")

    unbound = profile.with_updates(artifact_sha256=None)
    degraded = CapabilityGate().evaluate(unbound)
    assert degraded.status == "candidate"
    assert degraded.production_eligible is False


def test_local_probe_reads_metadata_without_loading_weights(tmp_path: Path) -> None:
    model = tmp_path / "QW1.8B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "max_position_embeddings": 2048}),
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ tools }} role tool arguments json"}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("tokenizer-fixture", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"tensor bytes are never loaded")

    result = probe_local_model(model, model_id="QW1.8B")
    assert result.errors == ()
    assert result.weights_loaded is False
    assert result.network_used is False
    assert result.profile.status == "candidate"
    assert result.profile.capabilities["tool_call_generation"].status == "declared"
    assert result.profile.capabilities["json_output"].status == "declared"
    assert result.profile.evidence["artifact_digest_mode"] == "inventory"
    assert len(result.profile.artifact_sha256 or "") == 64
    assert len(result.profile.tokenizer_digest or "") == 64


def test_missing_asset_is_rejected_without_exposing_path(tmp_path: Path) -> None:
    result = probe_local_model(tmp_path / "missing-model", model_id="missing")
    assert result.profile.status == "rejected"
    assert result.errors == ("asset_missing",)
    encoded = json.dumps(result.as_dict(), ensure_ascii=True)
    assert str(tmp_path).lower() not in encoded.lower()


def test_registry_selects_profiles_and_reports_diff(tmp_path: Path) -> None:
    registry = ProfileRegistry(tmp_path / "profiles")
    candidate = _profile()
    registry.register(candidate)
    selected = registry.select("test-small", backend="llama_server", artifact_sha256="a" * 64)
    assert selected is not None
    assert selected.digest == candidate.digest
    changed = candidate.with_updates(generation={"temperature": 0.2})
    diff = profile_diff(candidate, changed)
    assert "generation.temperature" in diff
    assert registry.rollback("test-small", backend="llama_server", revision="fixture-v1").digest == candidate.digest


def test_registry_alias_keeps_selection_filters(tmp_path: Path) -> None:
    registry = ProfileRegistry(tmp_path / "profiles")
    candidate = _profile().with_updates(aliases=("legacy-small",))
    registry.register(candidate)

    selected = registry.get("legacy-small", backend="llama_server", artifact_sha256="a" * 64)
    assert selected is not None
    assert selected.digest == candidate.digest
    assert registry.get("legacy-small", backend="pytorch") is None


def test_registry_rejects_filename_digest_tampering(tmp_path: Path) -> None:
    registry = ProfileRegistry(tmp_path / "profiles")
    path = registry.register(_profile())
    value = json.loads(path.read_text(encoding="utf-8"))
    value["generation"]["temperature"] = 0.2
    path.write_text(json.dumps(value), encoding="utf-8")
    assert registry.list_profiles() == ()
