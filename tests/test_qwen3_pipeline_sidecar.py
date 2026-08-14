from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, "src")

import qwen3_pipeline_sidecar as sidecar  # noqa: E402
from scripts.model_tools import qwen3_pipeline_runtime_worker as runtime_worker  # noqa: E402


def _session(tmp_path: Path, runner):
    root = tmp_path / "model"
    root.mkdir()
    return sidecar.Qwen3PipelineSidecarSession(
        model_path=root,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        config_id="cfg",
        plan_id="plan",
        node_id="worker-a",
        layer_range=[0, 1],
        total_layers=1,
        has_embedding=True,
        has_lm_head=True,
        assignment_manifest_sha256="b" * 64,
        worker_runner=runner,
    )


def test_sidecar_session_enforces_lifecycle_and_hides_model_path(tmp_path):
    phases: list[str] = []

    def runner(request, _timeout):
        phases.append(request["phase"])
        return {
            "schema_version": 1,
            "operation": "qwen3_pipeline_sidecar",
            "phase": request["phase"],
            "status": {
                "prepare": "prepared", "commit": "committed",
                "release": "released",
            }[request["phase"]],
            "gate_passed": True,
            "cleanup_complete": request["phase"] == "release",
            "assignment": {"selected_tensor_count": 1},
            "resources": {"passed": True},
            "execution": {"segment_materialized": request["phase"] == "commit"},
        }

    session = _session(tmp_path, runner)
    with pytest.raises(sidecar.Qwen3SidecarError) as exc:
        session.commit()
    assert exc.value.reason_code == "qwen3_sidecar_phase_invalid"
    assert session.prepare()["phase"] == "prepared"
    committed = session.commit()
    assert committed["segment_materialized"] is True
    released = session.release()
    assert released["phase"] == "released"
    assert phases == ["prepare", "commit", "release"]
    assert str(tmp_path) not in json.dumps(committed, ensure_ascii=True)


def test_sidecar_session_rejects_oversized_control_report(tmp_path):
    def runner(_request, _timeout):
        return {
            "schema_version": 1,
            "operation": "qwen3_pipeline_sidecar",
            "status": "prepared",
            "gate_passed": True,
            "phase": "prepare",
            "large": "x" * (sidecar.MAX_FRAME_BYTES + 1),
        }

    session = _session(tmp_path, runner)
    with pytest.raises(sidecar.Qwen3SidecarError) as exc:
        session.prepare()
    assert exc.value.reason_code == "qwen3_sidecar_frame_oversize"


def test_sidecar_session_executes_only_scoped_local_artifacts(tmp_path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    artifacts = model_root / "artifacts"
    artifacts.mkdir()
    input_ref = artifacts / "input.pt"
    input_ref.write_bytes(b"input")
    seen: list[dict] = []

    def runner(request, _timeout):
        seen.append(request)
        phase = request["phase"]
        if phase in {"prepare", "commit", "release"}:
            return {
                "schema_version": 1, "operation": "qwen3_pipeline_sidecar",
                "phase": phase,
                "status": {"prepare": "prepared", "commit": "committed", "release": "released"}[phase],
                "gate_passed": True, "cleanup_complete": phase == "release",
            }
        output_path = Path(request["output_ref"])
        output_path.write_bytes(b"output")
        output_bytes, output_sha256 = sidecar._file_evidence(output_path)
        return {
            "schema_version": 1, "operation": "qwen3_pipeline_sidecar",
            "phase": phase, "status": "executed", "gate_passed": True,
            "execution": {
                "segment_materialized": True, "full_model_materialized": False,
                "artifact_bytes": output_bytes, "artifact_sha256": output_sha256,
            },
        }

    session = sidecar.Qwen3PipelineSidecarSession(
        model_path=model_root, model_id="qwen3-4b", model_sha256="a" * 64,
        config_id="cfg", plan_id="plan", node_id="worker-a", layer_range=[0, 1],
        total_layers=1, has_embedding=True, has_lm_head=True, worker_runner=runner,
    )
    session.prepare()
    session.commit()
    report = session.execute(
        phase="prefill", artifact_root=artifacts, input_ref=input_ref,
        output_ref=artifacts / "output.pt", chain_id="chain", segment_index=0,
        sequence_length=2, batch_size=1, has_next_segment=False,
        generation=0, dtype="float32", device="cpu",
    )
    assert report["status"] == "executed"
    assert seen[-1]["data_plane"] == "local_artifact"
    assert "model_path" in seen[-1]  # local control input; never returned in report
    with pytest.raises(sidecar.Qwen3SidecarError) as exc:
        session.execute(
            phase="prefill", artifact_root=artifacts, input_ref=tmp_path / "outside.pt",
            output_ref=artifacts / "output-2.pt", chain_id="chain", segment_index=0,
            sequence_length=2, batch_size=1, has_next_segment=False,
            generation=0, dtype="float32", device="cpu",
        )
    assert exc.value.reason_code == "qwen3_sidecar_artifact_scope"
    session.release()


def test_runtime_worker_prepare_commit_release_cleans_stage(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 1, "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "part.safetensors"}}),
        encoding="utf-8",
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(runtime_worker._RuntimeSession, "_manifest_gate", lambda self, request, path: None)
    monkeypatch.setattr(runtime_worker, "execute_request", lambda request: {
        "gate_passed": True,
        "status": "ready_for_qwen3_pipeline_smoke",
        "assignment": {"selected_tensor_count": 1},
        "resources": {"device": "cpu", "passed": True},
        "runtime": {"isolated": True},
    })
    monkeypatch.setattr(runtime_worker, "select_qwen3_assignment_keys", lambda *args, **kwargs: ["model.layers.0.weight"])
    monkeypatch.setattr(runtime_worker, "_prepare_filtered_assignment", lambda *_args, **_kwargs: stage)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (), {
            "from_pretrained": staticmethod(lambda *args, **kwargs: type("Tokenizer", (), {
                "apply_chat_template": lambda self, *args, **kwargs: "OK",
                "__call__": lambda self, *args, **kwargs: {"input_ids": [1, 2]},
            })()),
        }),
    ))
    monkeypatch.setattr(runtime_worker, "load_qwen3_layer_assignment", lambda *args, **kwargs: (
        object(), {"selected_tensor_bytes": 4, "rss_delta_bytes": 8},
    ))

    request = {
        "schema_version": 1,
        "operation": "qwen3_pipeline_sidecar",
        "read_only": True,
        "network_access": "disabled",
        "phase": "prepare",
        "model_path": str(root),
        "model_id": "qwen3-4b", "model_sha256": "a" * 64,
        "config_id": "cfg", "plan_id": "plan", "node_id": "worker-a",
        "layer_range": [0, 1], "total_layers": 1,
        "has_embedding": True, "has_lm_head": True,
        "assignment_manifest_sha256": "b" * 64,
        "generation": 1, "execution_device": "cpu", "dtype": "float32",
        "controller_python": str(Path(sys.executable).absolute()),
    }
    worker = runtime_worker._RuntimeSession()
    prepared = worker.handle(request)
    assert prepared["status"] == "prepared"
    committed = worker.handle({**request, "phase": "commit"})
    assert committed["segment_materialized"] is True
    released = worker.handle({**request, "phase": "release"})
    assert released["cleanup_complete"] is True
    assert worker.stage is None


def test_runtime_worker_executes_local_prefill_decode_artifacts(tmp_path):
    class _Adapter:
        def forward(self, *, input_ids=None, hidden_states=None, past_key_values=None, use_cache=True):
            if input_ids is not None:
                value = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 4)
            else:
                value = hidden_states + 1
            current = value.unsqueeze(2)
            if past_key_values is not None:
                current = torch.cat((past_key_values[0], current), dim=1)
            return {"hidden_states": value, "logits": value, "past_key_values": (current, current)}

    root = tmp_path / "artifacts"
    root.mkdir()
    prefill_input = root / "prefill-input.pt"
    decode_input = root / "decode-input.pt"
    prefill_output = root / "prefill-output.pt"
    decode_output = root / "decode-output.pt"
    torch.save({"input_ids": torch.tensor([[1, 2, 3]])}, prefill_input)
    torch.save({"input_ids": torch.tensor([[4]])}, decode_input)

    identity = {
        "model_id": "qwen3-4b", "model_sha256": "a" * 64,
        "config_id": "cfg", "plan_id": "plan", "node_id": "worker-a",
        "layer_range": [0, 1], "total_layers": 1,
        "has_embedding": True, "has_lm_head": True,
        "assignment_manifest_sha256": "b" * 64, "generation": 0,
        "execution_device": "cpu", "dtype": "float32",
    }
    worker = runtime_worker._RuntimeSession()
    worker.phase = "committed"
    worker.request = dict(identity)
    worker.adapter = _Adapter()
    worker.resources = {"device": "cpu"}

    prefill_request = {
        "schema_version": 1, "operation": "qwen3_pipeline_sidecar",
        "read_only": True, "network_access": "disabled",
        **identity,
        "phase": "prefill", "data_plane": "local_artifact",
        "artifact_root": str(root), "input_ref": str(prefill_input),
        "input_bytes": runtime_worker._RuntimeSession._file_evidence(prefill_input)[0],
        "input_sha256": runtime_worker._RuntimeSession._file_evidence(prefill_input)[1],
        "output_ref": str(prefill_output), "kv_ref": None,
        "chain_id": "chain-test", "segment_index": 0,
        "sequence_length": 3, "batch_size": 1, "has_next_segment": False,
        "device": "cpu",
    }
    prefilled = worker.handle(prefill_request)
    assert prefilled["status"] == "executed"
    assert prefilled["kv_contract"]["sequence_length"] == 3
    assert prefill_output.is_file()

    decoded = worker.handle({
        **prefill_request,
        "phase": "decode", "input_ref": str(decode_input),
        "output_ref": str(decode_output), "kv_ref": str(prefill_output),
        "input_bytes": runtime_worker._RuntimeSession._file_evidence(decode_input)[0],
        "input_sha256": runtime_worker._RuntimeSession._file_evidence(decode_input)[1],
        "kv_bytes": runtime_worker._RuntimeSession._file_evidence(prefill_output)[0],
        "kv_sha256": runtime_worker._RuntimeSession._file_evidence(prefill_output)[1],
        "sequence_length": 4, "generation": 1,
    })
    assert decoded["status"] == "executed"
    assert decoded["kv_contract"]["phase"] == "decode"
    assert decoded["kv_contract"]["generation"] == 1
    assert decoded["execution"]["full_model_materialized"] is False


def test_network_sidecar_executor_keeps_output_local_and_releases_it(tmp_path):
    artifact_root = tmp_path / "network"
    artifact_root.mkdir()
    input_ref = artifact_root / "input.pt"
    input_ref.write_bytes(b"hidden-input")

    def runner(request, _timeout):
        phase = request["phase"]
        if phase in {"prepare", "commit", "release", "abort"}:
            return {
                "schema_version": 1,
                "operation": "qwen3_pipeline_sidecar",
                "phase": phase,
                "status": {
                    "prepare": "prepared", "commit": "committed",
                    "release": "released", "abort": "aborted",
                }[phase],
                "gate_passed": True,
                "cleanup_complete": phase in {"release", "abort"},
            }
        output = Path(request["output_ref"])
        output.write_bytes(b"hidden-output")
        size, digest = sidecar._file_evidence(output)
        return {
            "schema_version": 1,
            "operation": "qwen3_pipeline_sidecar",
            "phase": phase,
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "segment_materialized": True,
                "full_model_materialized": False,
                "artifact_bytes": size,
                "artifact_sha256": digest,
            },
            "hidden_handoff": {
                "dtype": request["dtype"],
                "device": request["device"],
                "shape": [1, 3, 4],
            },
            "kv_contract": {"present": True, "shape": [1, 3]},
        }

    session = _session(tmp_path, runner)
    session.prepare()
    session.commit()
    executor = sidecar.Qwen3NetworkSidecarExecutor(
        session, artifact_root=artifact_root,
    )
    internal = executor(
        input_ref,
        {
            "transfer_id": "qtx_" + "a" * 32,
            "chain_id": "c" * 64,
            "phase": "prefill",
            "generation": 2,
            "segment_index": 1,
            "sequence_length": 3,
            "batch_size": 1,
            "has_next_segment": False,
            "dtype": "float32",
            "device": "cpu",
        },
    )
    output_path = Path(internal["output_path"])
    assert output_path.is_file()
    assert str(artifact_root) in str(output_path)
    executor.cleanup({"chain_id": "c" * 64, "segment_index": 1}, "cancelled")
    assert not output_path.exists()
