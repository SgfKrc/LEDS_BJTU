from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")

import api_server  # noqa: E402
import qwen3_pipeline_parity  # noqa: E402
import scheduler as scheduler_module  # noqa: E402
import scheduler_svc_http  # noqa: E402
from qwen3_pipeline_transaction import (  # noqa: E402
    Qwen3PipelineProtocolError,
    build_qwen3_dry_run_contract,
)


def _contract(*, generation: int = 3, config_id: str = "cfg") -> dict:
    return build_qwen3_dry_run_contract(
        config_id=config_id, plan_id="plan", generation=generation,
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


class _FakeChain:
    def __init__(self, contract: dict, root: Path) -> None:
        self.chain_id = contract["contract_sha256"]
        self.generation = contract["generation"]
        self.sessions = [object(), object()]
        self.phase = "idle"
        self.cleanup_complete = False
        self.cancelled = False
        self.root = root

    @property
    def snapshot(self):
        return {
            "phase": self.phase,
            "segment_count": len(self.sessions),
            "cleanup_complete": self.cleanup_complete,
        }

    def prepare(self):
        self.phase = "prepared"

    def commit(self):
        self.phase = "committed"

    def prefill(self, **_kwargs):
        self.phase = "prefilled"

    def decode(self, **_kwargs):
        self.phase = "decoded"

    def final_output_ref(self, phase):
        return self.root / f"{phase}-1.pt"

    def artifact_refs(self, phase):
        return [self.root / f"{phase}-0.pt", self.root / f"{phase}-1.pt"]

    def execution_reports(self, _phase):
        return [{}, {}]

    def release(self):
        self.phase = "released"
        self.cleanup_complete = True

    def abort(self):
        self.phase = "aborted"
        self.cleanup_complete = True

    def cancel(self):
        self.cancelled = True
        self.abort()


class _Factory:
    chains = []

    @classmethod
    def from_contract(cls, *, contract, artifact_root, session_factory):
        del session_factory
        chain = _FakeChain(contract, Path(artifact_root))
        cls.chains.append(chain)
        return chain


def _scheduler(monkeypatch, tmp_path):
    sched = scheduler_module.Scheduler()
    memory = {
        "schema_version": 1,
        "phase": "idle",
        "cleanup_complete": True,
    }

    def save(value):
        memory.clear()
        memory.update(value)
        return dict(memory)

    monkeypatch.setattr(sched, "_qwen3_local_state_load", lambda: dict(memory))
    monkeypatch.setattr(sched, "_qwen3_local_state_save", save)
    monkeypatch.setattr(sched, "_qwen3_local_artifact_root_override", str(tmp_path / "artifacts"))
    monkeypatch.setattr(scheduler_module, "Qwen3PipelineMultiSidecar", _Factory)
    _Factory.chains = []
    return sched, memory


def _parity(passed: bool) -> dict:
    return {
        "schema_version": 1,
        "gate": "qwen3_cpu_parity",
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "full_model_fallback": False,
        "full_model_materialized": False,
        "errors": [] if passed else [{"code": "qwen3_parity_logits_mismatch", "message": "mismatch"}],
    }


def test_scheduler_runs_explicit_local_chain_and_fences_replay(monkeypatch, tmp_path):
    sched, memory = _scheduler(monkeypatch, tmp_path)
    contract = _contract()

    started = sched.begin_qwen3_local_sidecar_chain(contract)
    duplicate = sched.begin_qwen3_local_sidecar_chain(contract)
    prefilled = sched.run_qwen3_local_prefill(input_ref="input.pt", batch_size=1, sequence_length=3)
    decoded = sched.run_qwen3_local_decode(input_ref="input.pt", batch_size=1, sequence_length=4)
    monkeypatch.setattr(
        qwen3_pipeline_parity, "evaluate_qwen3_cpu_parity", lambda **_kwargs: _parity(True),
    )
    parity = sched.verify_qwen3_local_cpu_parity(
        reference_prefill="reference-prefill.pt", reference_decode="reference-decode.pt",
    )
    status = sched.get_qwen3_local_chain_status()
    released = sched.release_qwen3_local_sidecar_chain()

    assert started["state"]["phase"] == "committed"
    assert duplicate["status"] == "duplicate"
    assert prefilled["state"]["config_id"] == "cfg"
    assert decoded["state"]["plan_id"] == "plan"
    assert parity["status"] == "passed"
    assert status["active"] is True
    assert status["production_admitted"] is False
    assert status["state"]["phase"] == "parity_passed"
    assert status["state"]["parity"]["gate_passed"] is True
    assert released["state"]["phase"] == "released"
    assert memory["cleanup_complete"] is True
    with pytest.raises(Qwen3PipelineProtocolError, match="fenced"):
        sched.begin_qwen3_local_sidecar_chain(contract)


def test_scheduler_parity_rejection_cancels_entire_chain(monkeypatch, tmp_path):
    sched, memory = _scheduler(monkeypatch, tmp_path)
    sched.begin_qwen3_local_sidecar_chain(_contract())
    sched.run_qwen3_local_prefill(input_ref="input.pt", batch_size=1, sequence_length=3)
    sched.run_qwen3_local_decode(input_ref="input.pt", batch_size=1, sequence_length=4)
    chain = _Factory.chains[-1]
    monkeypatch.setattr(
        qwen3_pipeline_parity, "evaluate_qwen3_cpu_parity", lambda **_kwargs: _parity(False),
    )

    result = sched.verify_qwen3_local_cpu_parity(
        reference_prefill="reference-prefill.pt", reference_decode="reference-decode.pt",
    )

    assert result["status"] == "rejected"
    assert chain.cancelled is True
    assert sched.get_qwen3_local_chain_status()["active"] is False
    assert memory["phase"] == "aborted"
    assert memory["parity"]["gate_passed"] is False


def test_scheduler_reconciles_interrupted_state_and_scoped_artifacts(monkeypatch, tmp_path):
    sched, memory = _scheduler(monkeypatch, tmp_path)
    chain_id = "d" * 64
    memory.update({
        "contract_sha256": chain_id,
        "generation": 5,
        "phase": "decoded",
        "segment_count": 2,
        "cleanup_complete": False,
    })
    root = Path(sched._qwen3_local_artifact_root_override)
    root.mkdir(parents=True)
    token = hashlib.sha256(chain_id.encode("utf-8")).hexdigest()[:20]
    stale = root / f"qwen3-{token}-0-decode-dead.pt"
    unrelated = root / "qwen3-unrelated-0-decode-live.pt"
    stale.write_bytes(b"stale")
    unrelated.write_bytes(b"keep")

    status = sched.get_qwen3_local_chain_status()

    assert status["state"]["phase"] == "recovered_aborted"
    assert status["state"]["cleanup_complete"] is True
    assert not stale.exists()
    assert unrelated.exists()


class _HttpScheduler:
    def __init__(self, role="master") -> None:
        self.role = role
        self.contract = None

    def _effective_role(self):
        return self.role

    def get_qwen3_local_chain_status(self):
        return {"active": False, "production_admitted": False, "state": {"phase": "idle"}}

    def begin_qwen3_local_sidecar_chain(self, contract):
        self.contract = contract
        return {"status": "started", "state": {"phase": "committed"}}


def test_scheduler_http_shell_enforces_master_and_exposes_explicit_entry():
    sched = _HttpScheduler()
    client = TestClient(scheduler_svc_http.build_scheduler_app(sched))

    assert client.get("/cluster/qwen3/local-chain").status_code == 200
    response = client.post("/cluster/qwen3/local-chain/begin", json={"contract": {"x": 1}})
    assert response.status_code == 200
    assert sched.contract == {"x": 1}
    sched.role = "client"
    assert client.get("/cluster/qwen3/local-chain").status_code == 403


def test_scheduler_http_shell_maps_state_conflict_to_409():
    sched = _HttpScheduler()

    def conflict(_contract):
        raise Qwen3PipelineProtocolError("another Qwen3 local chain is active")

    sched.begin_qwen3_local_sidecar_chain = conflict
    response = TestClient(scheduler_svc_http.build_scheduler_app(sched)).post(
        "/cluster/qwen3/local-chain/begin", json={"contract": {"x": 1}},
    )

    assert response.status_code == 409


def test_monolith_http_status_route_uses_same_master_boundary(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        api_server.scheduler, "get_qwen3_local_chain_status",
        lambda: {"active": False, "production_admitted": False, "state": {"phase": "idle"}},
    )
    client = TestClient(api_server.app)

    accepted = client.get("/api/cluster/qwen3/local-chain")
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
    denied = client.get("/api/cluster/qwen3/local-chain")

    assert accepted.status_code == 200
    assert denied.status_code == 403
