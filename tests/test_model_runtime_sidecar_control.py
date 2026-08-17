import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen3_pipeline_transaction import Qwen3PipelineProtocolError
from model_runtime_contracts import (
    build_model_runtime_contract,
    validate_model_runtime_contract,
)
from scheduler import Scheduler
import scheduler_svc_http


def _capacity_assignments():
    return [
        {
            "node_id": "master",
            "start_layer": 0,
            "end_layer": 2,
            "has_embedding": True,
            "has_lm_head": False,
            "required_bytes": 200,
            "capacity_bytes": 400,
            "execution_device": "cpu",
        },
        {
            "node_id": "worker-a",
            "start_layer": 2,
            "end_layer": 4,
            "has_embedding": False,
            "has_lm_head": True,
            "required_bytes": 200,
            "capacity_bytes": 400,
            "execution_device": "cpu",
        },
    ]


def test_runtime_contracts_bind_capacity_assignments_and_validate_canonically():
    descriptor = {
        "total_layers": 4,
        "config": {"hidden_size": 2560},
    }
    contract = build_model_runtime_contract(
        "qwen3_sidecar",
        config_id="model-runtime-qwen3_sidecar-qwen3-4b",
        plan_id="plan-1",
        generation=1,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        descriptor=descriptor,
        assignments=_capacity_assignments(),
    )
    assert contract["execution_mode"] == "node_local_sidecar"
    assert contract["full_model_fallback"] is False
    assert validate_model_runtime_contract("qwen3_sidecar", contract) == contract
    assert all("model_path" not in segment for segment in contract["segments"])


def test_scheduler_binds_and_reuses_same_capacity_plan(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        scheduler,
        "_resolve_model_runtime_descriptor",
        lambda profile, model_id: (
            {
                "model_path": "ignored",
                "model_sha256": "a" * 64,
                "config": {"hidden_size": 2560},
            },
            {"total_layers": 4, "config": {"hidden_size": 2560}},
        ),
    )
    plan = {
        "status": "admitted",
        "admitted": True,
        "plan_id": "plan-1",
        "model_id": "qwen3-4b",
        "model_type": "qwen3",
        "total_layers": 4,
        "assignments": _capacity_assignments(),
    }
    monkeypatch.setattr(scheduler, "get_pipeline_capacity_plan", lambda **_: plan)
    records = []
    monkeypatch.setattr(scheduler, "_load_model_runtime_contract_records", lambda: records)
    monkeypatch.setattr(
        scheduler,
        "_save_model_runtime_contract_records",
        lambda value: (records.clear(), records.extend(value)),
    )

    first = scheduler.bind_model_runtime_contract("qwen3_sidecar", "qwen3-4b")
    second = scheduler.bind_model_runtime_contract("qwen3_sidecar", "qwen3-4b")

    assert first["status"] == "bound"
    assert second["status"] == "already_bound"
    assert first["contract_id"] == second["contract_id"]
    assert first["segment_count"] == 2
    assert scheduler.get_model_runtime_contract(first["contract_id"])["contract_sha256"] == first["contract_id"]
    with pytest.raises(ValueError, match="profile"):
        scheduler.get_model_runtime_contract(first["contract_id"], profile="gemma4_pipeline")


def test_scheduler_projects_sidecar_capabilities_without_admitting_production(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        scheduler, "get_qwen3_local_chain_status",
        lambda: {"active": False, "state": {"phase": "idle"}, "production_admitted": False},
    )
    monkeypatch.setattr(
        scheduler, "get_gemma4_local_sidecar_status",
        lambda: {"active": False, "state": {"phase": "idle"}, "production_admitted": False},
    )

    status = scheduler.get_model_runtime_sidecar_status()

    assert status["schema_version"] == 1
    assert status["control_available"] is True
    assert status["production_admitted"] is False
    assert status["profiles"]["qwen3_sidecar"]["preflight_supported"] is True
    assert status["profiles"]["qwen3_sidecar"]["requires_task_contract"] is True
    assert status["profiles"]["gemma4_pipeline"]["runtime_environment"] == ".venv-gemma4-pipeline"


def test_scheduler_rejects_sidecar_control_on_client_nodes(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "client")

    status = scheduler.get_model_runtime_sidecar_status()

    assert status["control_available"] is False
    assert status["profiles"]["qwen3_sidecar"]["session"]["reason_code"] == "master_only"
    with pytest.raises(Qwen3PipelineProtocolError, match="master"):
        scheduler.begin_model_runtime_sidecar("qwen3_sidecar", {"contract_sha256": "x"})


def test_scheduler_dispatches_only_known_contract_bound_sidecars(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "master")
    calls = []
    monkeypatch.setattr(
        scheduler, "begin_qwen3_local_sidecar_chain",
        lambda contract: calls.append(("qwen3", contract)) or {"status": "started"},
    )
    monkeypatch.setattr(
        scheduler, "release_gemma4_local_sidecar_chain",
        lambda: calls.append(("gemma4-release", None)) or {"status": "released"},
    )
    monkeypatch.setattr(
        scheduler, "cancel_qwen3_local_sidecar_chain",
        lambda: calls.append(("qwen3-cancel", None)) or {"status": "cancelled"},
    )

    started = scheduler.begin_model_runtime_sidecar("qwen3_sidecar", {"execution_mode": "node_local_sidecar"})
    released = scheduler.release_model_runtime_sidecar("gemma4_pipeline")
    cancelled = scheduler.cancel_model_runtime_sidecar("qwen3_sidecar")

    assert started == {"profile": "qwen3_sidecar", "status": "started", "production_admitted": False}
    assert released == {"profile": "gemma4_pipeline", "status": "released", "production_admitted": False}
    assert cancelled == {"profile": "qwen3_sidecar", "status": "cancelled", "production_admitted": False}
    assert calls == [
        ("qwen3", {"execution_mode": "node_local_sidecar"}),
        ("gemma4-release", None),
        ("qwen3-cancel", None),
    ]
    with pytest.raises(Qwen3PipelineProtocolError, match="contract"):
        scheduler.begin_model_runtime_sidecar("qwen3_sidecar", {})
    with pytest.raises(Qwen3PipelineProtocolError, match="unsupported"):
        scheduler.cancel_model_runtime_sidecar("unknown", )


def test_scheduler_resolves_persisted_contract_id_before_sidecar_begin(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "master")
    persisted = {"execution_mode": "node_local_sidecar", "contract_sha256": "c" * 64}
    monkeypatch.setattr(
        scheduler, "get_model_runtime_contract",
        lambda contract_id, *, profile=None: (
            persisted if contract_id == "contract-1" and profile == "qwen3_sidecar" else None
        ),
    )
    calls = []
    monkeypatch.setattr(
        scheduler, "begin_qwen3_local_sidecar_chain",
        lambda contract: calls.append(contract) or {"status": "started"},
    )

    started = scheduler.begin_model_runtime_sidecar(
        "qwen3_sidecar", contract_id="contract-1",
    )

    assert started["status"] == "started"
    assert calls == [persisted]


def test_scheduler_projects_path_free_contract_lifecycle_evidence(monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(scheduler, "_effective_role", lambda: "master")
    contract_id = "d" * 64
    persisted = {
        "contract_sha256": contract_id,
        "execution_mode": "node_local_sidecar",
    }
    records = [{
        "profile": "qwen3_sidecar",
        "model_id": "qwen3-4b",
        "summary": {
            "contract_id": contract_id,
            "profile": "qwen3_sidecar",
            "model_id": "qwen3-4b",
            "generation": 2,
            "segment_count": 2,
        },
        "contract": persisted,
        "audit": [{"at": 1.0, "action": "bound", "phase": "bound"}],
    }]
    monkeypatch.setattr(scheduler, "_load_model_runtime_contract_records", lambda: records)
    def save_records(value):
        records[:] = [dict(item) for item in value]

    monkeypatch.setattr(
        scheduler,
        "_save_model_runtime_contract_records",
        save_records,
    )
    state = {
        "contract_sha256": contract_id,
        "phase": "committed",
        "generation": 2,
        "segment_count": 2,
        "cleanup_complete": False,
    }
    monkeypatch.setattr(
        scheduler, "begin_qwen3_local_sidecar_chain",
        lambda contract: {"status": "started", "state": state},
    )
    monkeypatch.setattr(
        scheduler, "get_qwen3_local_chain_status",
        lambda: {"active": True, "state": state, "production_admitted": False},
    )
    monkeypatch.setattr(
        scheduler, "release_qwen3_local_sidecar_chain",
        lambda: {"status": "released", "state": {**state, "phase": "released", "cleanup_complete": True}},
    )

    scheduler.begin_model_runtime_sidecar("qwen3_sidecar", contract_id=contract_id)
    scheduler.release_model_runtime_sidecar("qwen3_sidecar")
    listed = scheduler.get_model_runtime_contracts()["contracts"][0]
    runtime = scheduler.get_model_runtime_sidecar_status()

    assert [event["action"] for event in listed["execution"]["recent_events"]] == [
        "released", "prepare_succeeded", "bound",
    ]
    assert listed["execution"]["recovery_action"] == "prepare"
    assert runtime["profiles"]["qwen3_sidecar"]["session"]["contract_id"] == contract_id

    monkeypatch.setattr(
        scheduler, "begin_qwen3_local_sidecar_chain",
        lambda contract: (_ for _ in ()).throw(RuntimeError(r"C:\private\model failure")),
    )
    with pytest.raises(RuntimeError):
        scheduler.begin_model_runtime_sidecar("qwen3_sidecar", contract_id=contract_id)
    failed = scheduler.get_model_runtime_contracts()["contracts"][0]["execution"]["last_event"]
    assert failed["action"] == "prepare_failed"
    assert failed["reason_code"] == "runtimeerror"
    assert "path" not in failed
    assert "private" not in str(failed)


class _HttpScheduler:
    def _effective_role(self):
        return "master"

    def get_model_runtime_sidecar_status(self):
        return {"schema_version": 1, "control_available": True, "profiles": {}}

    def get_model_runtime_contracts(self):
        return {"schema_version": 1, "contracts": []}

    def bind_model_runtime_contract(self, profile, model_id):
        return {"status": "bound", "profile": profile, "model_id": model_id}

    def begin_model_runtime_sidecar(self, profile, contract, *, contract_id=None):
        return {
            "profile": profile,
            "contract": contract,
            "contract_id": contract_id,
            "status": "started",
        }

    def release_model_runtime_sidecar(self, profile):
        return {"profile": profile, "status": "released"}

    def cancel_model_runtime_sidecar(self, profile):
        return {"profile": profile, "status": "cancelled"}


def test_scheduler_http_exposes_unified_sidecar_control_contract():
    client = TestClient(scheduler_svc_http.build_scheduler_app(_HttpScheduler()))

    assert client.get("/cluster/model-runtime/sidecars").json()["control_available"] is True
    assert client.get("/cluster/model-runtime/contracts").json()["contracts"] == []
    assert client.post(
        "/cluster/model-runtime/contracts/bind",
        json={"profile": "qwen3_sidecar", "model_id": "qwen3-4b"},
    ).json()["status"] == "bound"
    assert client.post(
        "/cluster/model-runtime/sidecars/begin",
        json={"profile": "qwen3_sidecar", "contract": {"execution_mode": "node_local_sidecar"}},
    ).json()["status"] == "started"
    by_id = client.post(
        "/cluster/model-runtime/sidecars/begin",
        json={"profile": "qwen3_sidecar", "contract_id": "contract-1"},
    ).json()
    assert by_id["contract_id"] == "contract-1"
    assert by_id["contract"] is None
    assert client.post(
        "/cluster/model-runtime/sidecars/release",
        json={"profile": "gemma4_pipeline"},
    ).json()["status"] == "released"
    assert client.delete("/cluster/model-runtime/sidecars/qwen3_sidecar").json()["status"] == "cancelled"
    assert client.post(
        "/cluster/model-runtime/sidecars/begin",
        json={"profile": "unknown", "contract": {}},
    ).status_code == 422
