import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec
import task_graph_payloads
from task_graph_optimization import optimize_task_graph, project_task_graph
from task_graph_payloads import (
    TaskPayloadAuthorizationError,
    TaskPayloadConflict,
    TaskPayloadInUse,
    TaskPayloadNotFound,
    TaskPayloadStore,
    TaskPayloadValidationError,
    bind_payload_plan,
    validate_payload_reference,
)


def _fanout_plan():
    stages = [
        StageSpec("shared", "transform", provider="local", pure=True),
        StageSpec("left", "transform", depends_on=("shared",), provider="local"),
        StageSpec("right", "transform", depends_on=("shared",), provider="local"),
        StageSpec(
            "final", "aggregate", depends_on=("left", "right"), provider="local",
        ),
    ]
    result = optimize_task_graph(
        project_task_graph(stages, "final", graph_id="payload_fanout"),
    )
    return result["payload_plan"][0]


def test_g2_2_binds_fanout_plan_without_body_or_path(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    body = b'{"messages":["shared input"]}'
    reference = bind_payload_plan(
        store, _fanout_plan(), body, data_scope="workflow:wf1",
        media_type="application/json",
    )
    repeated = bind_payload_plan(
        store, _fanout_plan(), body, data_scope="workflow:wf1",
        media_type="application/json",
    )

    assert repeated == reference
    assert validate_payload_reference(reference) == reference
    rendered = json.dumps(reference)
    assert "shared input" not in rendered
    assert "path" not in rendered
    assert "payload:" in reference["payload_id"]
    assert store.stats() == {
        "schema_version": "qlh.task_graph_payload_ref.v1",
        "object_count": 1,
        "manifest_count": 1,
        "active_materialization_count": 0,
        "stored_bytes": len(body),
    }


def test_g2_2_same_content_is_isolated_by_data_scope(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    plan = _fanout_plan()
    first = bind_payload_plan(
        store, plan, b"same", data_scope="workflow:wf1",
    )
    second = bind_payload_plan(
        store, plan, b"same", data_scope="workflow:wf2",
    )

    assert first["content_sha256"] == second["content_sha256"]
    assert first["payload_id"] != second["payload_id"]
    assert store.stats()["object_count"] == 2


def test_g2_2_materialization_is_authorized_temporary_and_release_fenced(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    reference = bind_payload_plan(
        store, _fanout_plan(), b"immutable", data_scope="workflow:wf1",
    )

    with store.materialize(
        reference, consumer_stage_id="left", data_scope="workflow:wf1",
    ) as local_path:
        assert local_path.parent == store.materialized_dir
        assert local_path.read_bytes() == b"immutable"
        assert store.stats()["active_materialization_count"] == 1
        with pytest.raises(TaskPayloadInUse):
            store.release(reference)

    assert not local_path.exists()
    assert store.stats()["active_materialization_count"] == 0
    store.release(reference)
    assert store.stats()["object_count"] == 0
    with pytest.raises(TaskPayloadNotFound):
        with store.materialize(
            reference, consumer_stage_id="left", data_scope="workflow:wf1",
        ):
            pass


def test_g2_2_materialization_rejects_wrong_scope_or_consumer(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    reference = bind_payload_plan(
        store, _fanout_plan(), b"private", data_scope="workflow:wf1",
    )

    with pytest.raises(TaskPayloadAuthorizationError, match="scope"):
        with store.materialize(
            reference, consumer_stage_id="left", data_scope="workflow:wf2",
        ):
            pass
    with pytest.raises(TaskPayloadAuthorizationError, match="consumer"):
        with store.materialize(
            reference, consumer_stage_id="final", data_scope="workflow:wf1",
        ):
            pass


def test_g2_2_detects_immutable_object_tampering(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    reference = bind_payload_plan(
        store, _fanout_plan(), b"original", data_scope="workflow:wf1",
    )
    object_path = store.objects_dir / f"{reference['reference_sha256']}.payload"
    object_path.write_bytes(b"tampered")

    with pytest.raises(TaskPayloadConflict, match="digest"):
        with store.materialize(
            reference, consumer_stage_id="left", data_scope="workflow:wf1",
        ):
            pass
    with pytest.raises(TaskPayloadConflict):
        bind_payload_plan(
            store, _fanout_plan(), b"original", data_scope="workflow:wf1",
        )


def test_g2_2_invalid_manifest_and_failed_copy_leave_no_materialization(
    tmp_path, monkeypatch,
):
    store = TaskPayloadStore(tmp_path / "payloads")
    reference = bind_payload_plan(
        store, _fanout_plan(), b"original", data_scope="workflow:wf1",
    )
    manifest_path = (
        store.manifests_dir / f"{reference['reference_sha256']}.json"
    )
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text('{"schema_version":"bad"}', encoding="utf-8")
    with pytest.raises(TaskPayloadConflict, match="manifest"):
        with store.materialize(
            reference, consumer_stage_id="left", data_scope="workflow:wf1",
        ):
            pass

    manifest_path.write_bytes(original_manifest)

    def corrupt_copy(source, target):
        del source
        target.write_bytes(b"corrupt-copy")

    monkeypatch.setattr(task_graph_payloads.shutil, "copyfile", corrupt_copy)
    with pytest.raises(TaskPayloadConflict):
        with store.materialize(
            reference, consumer_stage_id="left", data_scope="workflow:wf1",
        ):
            pass
    assert list(store.materialized_dir.glob("*.payload")) == []


def test_g2_2_rejects_unsafe_contracts_and_size_overflow(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads", max_payload_bytes=4)
    with pytest.raises(TaskPayloadValidationError, match="size"):
        bind_payload_plan(
            store, _fanout_plan(), b"12345", data_scope="workflow:wf1",
        )
    with pytest.raises(TaskPayloadValidationError, match="identifier"):
        bind_payload_plan(
            store, _fanout_plan(), b"1", data_scope="../escape",
        )
    with pytest.raises(TaskPayloadValidationError, match="fields"):
        bind_payload_plan(
            store, {**_fanout_plan(), "path": "unsafe"}, b"1",
            data_scope="workflow:wf1",
        )
    with pytest.raises(TaskPayloadValidationError, match="bytes-like"):
        store.put(
            "not-bytes",
            data_scope="workflow:wf1",
            source_stage_id="shared",
            consumer_stage_ids=("left", "right"),
        )


def test_g2_2_restart_cleans_only_controlled_materializations(tmp_path):
    root = tmp_path / "payloads"
    store = TaskPayloadStore(root)
    reference = bind_payload_plan(
        store, _fanout_plan(), b"persistent", data_scope="workflow:wf1",
    )
    stale = store.materialized_dir / "stale.payload"
    unrelated = store.materialized_dir / "keep.txt"
    stale.write_bytes(b"stale")
    unrelated.write_bytes(b"keep")

    restarted = TaskPayloadStore(root)

    assert not stale.exists()
    assert unrelated.read_bytes() == b"keep"
    assert restarted.stats()["object_count"] == 1
    with restarted.materialize(
        reference, consumer_stage_id="right", data_scope="workflow:wf1",
    ) as local_path:
        assert local_path.read_bytes() == b"persistent"
