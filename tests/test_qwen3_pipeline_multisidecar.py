from __future__ import annotations

from pathlib import Path

import pytest

import sys

sys.path.insert(0, "src")

from qwen3_pipeline_contract import build_kv_contract  # noqa: E402
from qwen3_pipeline_multisidecar import (  # noqa: E402
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
)
from qwen3_pipeline_transaction import build_qwen3_dry_run_contract  # noqa: E402


class _Session:
    def __init__(self, index: int, *, fail_phase: str | None = None) -> None:
        self.index = index
        self.fail_phase = fail_phase
        self.calls: list[str] = []
        self.requests: list[dict] = []
        self.decode_handoff_tokens: int | None = None

    def prepare(self):
        self.calls.append("prepare")
        return {"phase": "prepared", "gate_passed": True}

    def commit(self):
        self.calls.append("commit")
        if self.fail_phase == "commit":
            raise RuntimeError("commit failed")
        return {"phase": "committed", "gate_passed": True, "full_model_materialized": False}

    def execute(self, **request):
        phase = str(request["phase"])
        self.calls.append(phase)
        self.requests.append(dict(request))
        if self.fail_phase == phase:
            raise RuntimeError(f"{phase} failed")
        dtype = str(request["dtype"])
        device = str(request["device"])
        sequence = int(request["sequence_length"])
        batch = int(request["batch_size"])
        segment = int(request["segment_index"])
        Path(request["output_ref"]).write_bytes(b"artifact")
        handoff_sequence = (
            self.decode_handoff_tokens
            if phase == "decode" and self.decode_handoff_tokens is not None
            else sequence
        )
        return {
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
                "artifact_bytes": 8,
                "artifact_sha256": "a" * 64,
            },
            "kv_contract": build_kv_contract(
                chain_id=request["chain_id"], segment_index=segment,
                layer_range=[[0, 2], [2, 4]][segment], sequence_length=sequence,
                batch_size=batch, dtype=dtype, device=device,
                phase=phase, generation=int(request["generation"]),
            ),
            "hidden_handoff": (
                {
                    "schema_version": 1,
                    "chain_id": request["chain_id"],
                    "from_segment": segment,
                    "to_segment": segment + 1,
                    "shape": [batch, handoff_sequence, 4],
                    "batch_size": batch,
                    "sequence_length": handoff_sequence,
                    "hidden_size": 4,
                    "dtype": dtype,
                    "device": device,
                }
                if request["has_next_segment"] else None
            ),
        }

    def release(self):
        self.calls.append("release")
        return {"phase": "released"}

    def abort(self):
        self.calls.append("abort")
        return {"phase": "aborted"}


def _chain(tmp_path: Path, *, fail_phase: str | None = None):
    root = tmp_path / "artifacts"
    root.mkdir()
    source = root / "input.pt"
    source.write_bytes(b"input")
    sessions = [_Session(0, fail_phase=fail_phase), _Session(1, fail_phase=fail_phase)]
    chain = Qwen3PipelineMultiSidecar(
        sessions=sessions,
        segments=[
            {"layer_range": [0, 2], "has_embedding": True, "device": "cpu", "dtype": "torch.float32"},
            {"layer_range": [2, 4], "has_lm_head": True, "device": "cpu", "dtype": "torch.float32"},
        ],
        artifact_root=root,
        chain_id="chain-test",
    )
    return chain, sessions, source, root


def test_multi_sidecar_prefill_decode_and_release_cleans_artifacts(tmp_path):
    chain, sessions, source, root = _chain(tmp_path)
    assert chain.prepare()["phase"] == "prepared"
    assert chain.commit()["phase"] == "committed"
    assert chain.prefill(input_ref=source, batch_size=1, sequence_length=3)["phase"] == "prefilled"
    assert chain.decode(input_ref=source, batch_size=1, sequence_length=4)["phase"] == "decoded"
    assert chain.release()["phase"] == "released"
    assert all(session.calls == ["prepare", "commit", "prefill", "decode", "release"] for session in sessions)
    assert list(root.glob("qwen3-*.pt")) == []


def test_multi_sidecar_rotates_kv_across_repeated_decode_steps(tmp_path):
    chain, sessions, source, root = _chain(tmp_path)
    for session in sessions:
        session.decode_handoff_tokens = 1
    chain.prepare()
    chain.commit()
    prefill = chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    prefill_kv = list(chain._prefill_outputs)

    first = chain.decode(
        input_ref=source, batch_size=1, sequence_length=4,
        input_sequence_length=1,
    )
    first_kv = list(chain._prefill_outputs)
    second = chain.decode(
        input_ref=source, batch_size=1, sequence_length=5,
        input_sequence_length=1,
    )
    second_kv = list(chain._prefill_outputs)

    assert prefill["decode_step_count"] == 0
    assert first["decode_step_count"] == 1
    assert second["decode_step_count"] == 2
    assert second["kv_sequence_length"] == 5
    assert second["decode_history"] == [
        {"step_index": 1, "generation": 1, "sequence_length": 4,
         "input_sequence_length": 1},
        {"step_index": 2, "generation": 2, "sequence_length": 5,
         "input_sequence_length": 1},
    ]
    assert all(not path.exists() for path in prefill_kv + first_kv)
    assert all(path.is_file() for path in second_kv)
    for index, session in enumerate(sessions):
        decode_requests = [value for value in session.requests if value["phase"] == "decode"]
        assert Path(decode_requests[0]["kv_ref"]) == prefill_kv[index]
        assert Path(decode_requests[1]["kv_ref"]) == first_kv[index]
        assert [value["generation"] for value in decode_requests] == [1, 2]
    assert chain.release()["phase"] == "released"
    assert list(root.glob("qwen3-*.pt")) == []


def test_multi_sidecar_repeated_decode_sequence_mismatch_aborts_chain(tmp_path):
    chain, sessions, source, root = _chain(tmp_path)
    for session in sessions:
        session.decode_handoff_tokens = 1
    chain.prepare()
    chain.commit()
    chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    chain.decode(
        input_ref=source, batch_size=1, sequence_length=4,
        input_sequence_length=1,
    )

    with pytest.raises(Qwen3MultiSidecarError) as caught:
        chain.decode(
            input_ref=source, batch_size=1, sequence_length=6,
            input_sequence_length=1,
        )

    assert caught.value.reason_code == "qwen3_multisidecar_sequence_mismatch"
    assert chain.phase == "aborted"
    assert all(session.calls[-1] == "abort" for session in sessions)
    assert list(root.glob("qwen3-*.pt")) == []


def test_multi_sidecar_failure_aborts_all_segments_and_removes_artifacts(tmp_path):
    chain, sessions, source, root = _chain(tmp_path, fail_phase="prefill")
    chain.prepare()
    chain.commit()
    with pytest.raises(Qwen3MultiSidecarError) as exc:
        chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    assert exc.value.reason_code == "qwen3_multisidecar_execution_failed"
    assert all("abort" in session.calls for session in sessions)
    assert chain.phase == "aborted"
    assert list(root.glob("qwen3-*.pt")) == []


def test_multi_sidecar_rejects_handoff_shape_mismatch_and_supports_restart_recovery(tmp_path):
    chain, sessions, source, root = _chain(tmp_path)
    original = sessions[0].execute

    def wrong_handoff(**request):
        report = original(**request)
        if request["has_next_segment"]:
            report["hidden_handoff"]["shape"][1] = 99
        return report

    sessions[0].execute = wrong_handoff
    chain.prepare()
    chain.commit()
    with pytest.raises(Qwen3MultiSidecarError, match="hidden handoff"):
        chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    stale = root / f"qwen3-{chain._chain_token}-0-prefill-stale.pt"
    stale.write_bytes(b"stale")
    recovered = chain.recover_after_restart()
    assert recovered["phase"] == "aborted"
    assert recovered["cleanup_complete"] is True
    assert recovered["last_report"]["abort"]["reason_code"] == "restart_recovery"
    assert list(root.glob("qwen3-*.pt")) == []


def test_multi_sidecar_cancel_is_fail_closed(tmp_path):
    chain, sessions, _source, _root = _chain(tmp_path)
    chain.prepare()
    chain.commit()
    cancelled = chain.cancel()
    assert cancelled["phase"] == "aborted"
    assert cancelled["last_report"]["abort"]["reason_code"] == "cancelled"
    assert all(session.calls[-1] == "abort" for session in sessions)


def test_multi_sidecar_builds_from_canonical_node_local_contract(tmp_path):
    contract = build_qwen3_dry_run_contract(
        config_id="cfg", plan_id="plan", generation=3,
        model_id="qwen3-4b", model_sha256="a" * 64,
        total_layers=4, hidden_size=4, execution_mode="node_local_sidecar",
        segments=[
            {
                "node_id": "worker-a", "layer_range": [0, 2],
                "has_embedding": True, "required_bytes": 100,
                "assignment_manifest_sha256": "b" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
            {
                "node_id": "worker-b", "layer_range": [2, 4],
                "has_lm_head": True, "required_bytes": 100,
                "assignment_manifest_sha256": "c" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
        ],
    )
    frames: list[dict] = []

    def factory(frame):
        frames.append(frame)
        return _Session(len(frames) - 1)

    chain = Qwen3PipelineMultiSidecar.from_contract(
        contract=contract, artifact_root=tmp_path / "artifacts",
        session_factory=factory,
    )
    assert chain.chain_id == contract["contract_sha256"]
    assert chain.generation == 3
    assert [frame["node_id"] for frame in frames] == ["worker-a", "worker-b"]
