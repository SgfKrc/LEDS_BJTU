from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gemma4_pipeline_contract import build_gemma4_pipeline_contract  # noqa: E402
from gemma4_pipeline_multisidecar import Gemma4PipelineMultiSidecar  # noqa: E402
from gemma4_pipeline_sidecar import (  # noqa: E402
    Gemma4PipelineSidecarSession,
    Gemma4SidecarError,
)
from scheduler import Scheduler  # noqa: E402


def _segment(index, start, end):
    return {
        "node_id": f"node-{index}",
        "layer_range": [start, end],
        "has_embedding": start == 0,
        "has_lm_head": end == 4,
        "assignment_manifest_sha256": f"{index + 1:064x}",
        "required_bytes": 1024,
        "execution_device": "cpu",
        "dtype": "float32",
    }


def _contract():
    return build_gemma4_pipeline_contract(
        config_id="config-g4",
        plan_id="plan-g4",
        generation=1,
        model_id="gemma4",
        model_sha256="a" * 64,
        total_layers=4,
        hidden_size=3,
        layer_types=[
            "full_attention", "sliding_attention",
            "full_attention", "sliding_attention",
        ],
        num_kv_shared_layers=2,
        segments=[_segment(0, 0, 2), _segment(1, 2, 4)],
    )


def _runner(request, _timeout):
    phase = request["phase"]
    if phase == "prepare":
        return {
            "schema_version": 1,
            "operation": "gemma4_pipeline_sidecar",
            "phase": phase,
            "status": "prepared",
            "gate_passed": True,
            "cleanup_complete": False,
        }
    if phase == "commit":
        return {
            "schema_version": 1,
            "operation": "gemma4_pipeline_sidecar",
            "phase": phase,
            "status": "committed",
            "gate_passed": True,
            "cleanup_complete": False,
            "execution": {"segment_materialized": True},
        }
    return {
        "schema_version": 1,
        "operation": "gemma4_pipeline_sidecar",
        "phase": phase,
        "status": "aborted" if phase == "abort" else "released",
        "gate_passed": True,
        "cleanup_complete": True,
    }


def test_session_uses_injected_runner_and_keeps_lifecycle_bounded(tmp_path):
    session = Gemma4PipelineSidecarSession(
        model_path=tmp_path,
        model_id="gemma4",
        model_sha256="a" * 64,
        config_id="config-g4",
        plan_id="plan-g4",
        node_id="node-0",
        layer_range=[0, 2],
        total_layers=4,
        has_embedding=True,
        has_lm_head=False,
        produced_shared_kv_types=["full_attention", "sliding_attention"],
        worker_runner=_runner,
    )
    assert session.prepare()["phase"] == "prepared"
    assert session.commit()["phase"] == "committed"
    assert session.snapshot()["full_model_materialized"] is False
    assert session.release()["phase"] == "released"


def test_native_mtmd_environment_is_rejected_for_pipeline_session(tmp_path):
    native_python = Path(".venv-gemma4-native") / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if not native_python.is_file():
        pytest.skip("native Gemma environment is not present")
    with pytest.raises(Gemma4SidecarError, match="llama.cpp/MTMD"):
        Gemma4PipelineSidecarSession(
            model_path=tmp_path,
            model_id="gemma4",
            model_sha256="a" * 64,
            config_id="config-g4",
            plan_id="plan-g4",
            node_id="node-0",
            layer_range=[0, 2],
            total_layers=4,
            has_embedding=True,
            has_lm_head=False,
            sidecar_python=native_python,
        )


class _FakeSession:
    def __init__(self, frame):
        self.frame = frame
        self.phase = "idle"

    def prepare(self):
        self.phase = "prepared"
        return {"phase": self.phase, "gate_passed": True}

    def commit(self):
        self.phase = "committed"
        return {"phase": self.phase, "gate_passed": True}

    def execute(self, *, phase, artifact_root, input_ref, output_ref, **kwargs):
        del kwargs
        input_path = Path(input_ref)
        output_path = Path(output_ref)
        output_path.write_bytes(input_path.read_bytes() + phase.encode("ascii"))
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        return {
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": digest,
                "full_model_materialized": False,
                "multimodal_materialized": False,
                "segment_materialized": True,
            },
            "hidden_handoff": {
                "from_segment": self.frame["segment_index"],
                "to_segment": self.frame["segment_index"] + 1,
            },
        }

    def release(self):
        self.phase = "released"
        return {"phase": self.phase, "gate_passed": True}

    def abort(self):
        self.phase = "aborted"
        return {"phase": self.phase, "gate_passed": True}


def test_multisidecar_prepare_commit_prefill_decode_and_cleanup(tmp_path):
    input_ref = tmp_path / "input.pt"
    input_ref.write_bytes(b"input")
    chain = Gemma4PipelineMultiSidecar.from_contract(
        contract=_contract(),
        artifact_root=tmp_path,
        session_factory=_FakeSession,
    )
    chain.prepare()
    chain.commit()
    chain.prefill(input_ref=input_ref, batch_size=1, sequence_length=3)
    assert chain.snapshot["phase"] == "prefilled"
    prefill_output = chain.final_output_ref("prefill")
    chain.decode(input_ref=input_ref, batch_size=1, sequence_length=4)
    assert chain.snapshot["phase"] == "decoded"
    assert chain.final_output_ref("decode").is_file()
    assert prefill_output.is_file()
    chain.release()
    assert chain.snapshot["phase"] == "released"
    assert chain.snapshot["cleanup_complete"] is True
    assert not list(tmp_path.glob("gemma4-*.pt"))


def test_scheduler_exposes_only_explicit_nonproduction_gemma4_route(tmp_path):
    scheduler = Scheduler()
    scheduler._gemma4_local_artifact_root_override = str(tmp_path)
    input_ref = tmp_path / "scheduler-input.pt"
    input_ref.write_bytes(b"input")

    started = scheduler.begin_gemma4_local_sidecar_chain(
        _contract(), session_factory=_FakeSession,
    )
    assert started["status"] == "committed"
    assert started["production_admitted"] is False
    assert scheduler.get_gemma4_local_sidecar_status() == {
        "active": True,
        "state": scheduler._gemma4_local_chain.snapshot,
        "runtime_environment": ".venv-gemma4-pipeline",
        "production_admitted": False,
    }
    scheduler.run_gemma4_local_prefill(
        input_ref=str(input_ref), batch_size=1, sequence_length=3,
    )
    scheduler.run_gemma4_local_decode(
        input_ref=str(input_ref), batch_size=1, sequence_length=4,
    )
    released = scheduler.release_gemma4_local_sidecar_chain()
    assert released["status"] == "released"
    assert scheduler.get_gemma4_local_sidecar_status()["active"] is False
