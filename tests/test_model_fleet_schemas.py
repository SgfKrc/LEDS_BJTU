"""
M0 冻结契约：artifact manifest / pull job / deployment / cluster profile
JSON Schema 双语言验证（Python 侧）。
====================================================================
验收口径（一键模型部署计划 §16 M0）：同一 fixture 经 Python/TS schema
validator 得到一致结果；能力枚举禁止用 model_type 推导分布式能力。
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import pytest
import jsonschema

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "fixtures" / "model-fleet"
COMPATIBILITY = SCHEMAS / "model-fleet-compatibility.json"

SCHEMA_FILES = {
    "artifact-manifest": "artifact-manifest.schema.json",
    "pull-job": "pull-job.schema.json",
    "deployment": "deployment.schema.json",
    "cluster-profile": "cluster-profile.schema.json",
    "migration-map": "migration-map.schema.json",
    "fetcher-progress": "fetcher-progress.schema.json",
}

_VALIDATORS = {}

_VALID_SAMPLES = {
    "artifact-manifest": FIXTURES / "artifact-manifest-valid.json",
    "pull-job": FIXTURES / "pull-job-valid.json",
    "deployment": FIXTURES / "deployment-valid.json",
    "cluster-profile": FIXTURES / "cluster-profile-valid.json",
    "migration-map": SCHEMAS / "migration-map.json",
    "fetcher-progress": FIXTURES / "fetcher-progress-valid.json",
}


def _validator(kind: str):
    if kind not in _VALIDATORS:
        schema_path = SCHEMAS / SCHEMA_FILES[kind]
        with schema_path.open(encoding="utf-8") as fh:
            schema = json.load(fh)
        _VALIDATORS[kind] = jsonschema.Draft7Validator(schema)
    return _VALIDATORS[kind]


def _fixture_kind(filename: str) -> str:
    for kind in SCHEMA_FILES:
        if filename.startswith(kind):
            return kind
    raise KeyError(filename)


def _iter_fixtures():
    for path in sorted(FIXTURES.glob("*.json")):
        yield path.name, path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence_errors(data: dict) -> list[str]:
    policy = _load_json(COMPATIBILITY)["fetcher_progress_v1"]
    events = data.get("events")
    if not isinstance(events, list) or not events:
        return ["events must be a non-empty list"]

    errors: list[str] = []
    protocol_version = _load_json(COMPATIBILITY)["schemas"][
        "fetcher-progress"
    ]["schema_version"]
    if data.get("schema_version") != protocol_version:
        errors.append("sequence schema_version does not match the protocol")

    first_job_id = events[0].get("job_id")
    previous_event = None
    previous_time = None
    previous_counters: dict[str, int] = {}
    for index, event in enumerate(events):
        schema_errors = list(_validator("fetcher-progress").iter_errors(event))
        errors.extend(f"events[{index}]: {item.message}" for item in schema_errors)

        event_name = event.get("event")
        phase = event.get("phase")
        allowed_phases = policy["event_phases"].get(event_name, [])
        if phase not in allowed_phases:
            errors.append(f"events[{index}]: phase is invalid for {event_name}")
        for field in policy["required_fields"].get(event_name, []):
            if event.get(field) is None:
                errors.append(f"events[{index}]: {field} is required for {event_name}")
        if event.get("job_id") != first_job_id:
            errors.append(f"events[{index}]: job_id changed within one sequence")

        try:
            current_time = datetime.fromisoformat(event["at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            errors.append(f"events[{index}]: at is not a valid timestamp")
            current_time = None
        if previous_time is not None and current_time is not None:
            if current_time < previous_time:
                errors.append(f"events[{index}]: timestamp regressed")
        if current_time is not None:
            previous_time = current_time

        if previous_event is not None:
            allowed = policy["transitions"].get(previous_event, [])
            if event_name not in allowed:
                errors.append(
                    f"events[{index}]: transition {previous_event} -> {event_name} is invalid"
                )
        previous_event = event_name

        for field in policy["monotonic_fields"]:
            value = event.get(field)
            if isinstance(value, int):
                if field in previous_counters and value < previous_counters[field]:
                    errors.append(f"events[{index}]: {field} regressed")
                previous_counters[field] = value
        if isinstance(event.get("total_bytes"), int) and isinstance(
            event.get("downloaded_bytes"), int
        ):
            if event["downloaded_bytes"] > event["total_bytes"]:
                errors.append(f"events[{index}]: downloaded_bytes exceeds total_bytes")
        if isinstance(event.get("files_total"), int) and isinstance(
            event.get("files_done"), int
        ):
            if event["files_done"] > event["files_total"]:
                errors.append(f"events[{index}]: files_done exceeds files_total")

    if events[0].get("event") != policy["start_event"]:
        errors.append("sequence does not start with started")
    if events[-1].get("event") not in policy["terminal_events"]:
        errors.append("sequence does not end in a terminal event")
    return errors


def test_schema_files_are_valid_json_schemas():
    """四个 schema 文件本身必须是合法 JSON 且可编译。"""
    for name in SCHEMA_FILES.values():
        with (SCHEMAS / name).open(encoding="utf-8") as fh:
            schema = json.load(fh)
        jsonschema.Draft7Validator.check_schema(schema)


# 合法/非法 fixture（文件名约定：<kind>-valid.json / <kind>-invalid-*.json）
_VALID = [
    "artifact-manifest-valid.json",
    "artifact-manifest-gguf-valid.json",
    "pull-job-valid.json",
    "deployment-valid.json",
    "cluster-profile-valid.json",
    "fetcher-progress-valid.json",
]
_INVALID = [
    "artifact-manifest-invalid-bad-digest.json",
    "artifact-manifest-invalid-extra-capability.json",
    "pull-job-invalid-state.json",
    "deployment-invalid-status.json",
    "cluster-profile-invalid-endpoint.json",
    "fetcher-progress-invalid-event.json",
]


@pytest.mark.parametrize("filename", _VALID)
def test_valid_fixtures_pass(filename):
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    errors = list(_validator(_fixture_kind(filename)).iter_errors(data))
    assert errors == [], f"{filename} 应通过校验，但发现: {[e.message for e in errors]}"


@pytest.mark.parametrize("filename", _INVALID)
def test_invalid_fixtures_fail(filename):
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    errors = list(_validator(_fixture_kind(filename)).iter_errors(data))
    assert errors, f"{filename} 应被拒绝，但通过了校验"


def test_capability_enum_is_frozen():
    """能力枚举冻结：不允许额外键（禁止用 model_type 推导能力）。"""
    schema = json.loads((SCHEMAS / "artifact-manifest.schema.json")
                        .read_text(encoding="utf-8"))
    cap_props = schema["properties"]["capabilities"]["properties"]
    assert set(cap_props) == {
        "full_worker", "pytorch_layer_pipeline", "llama_cpp", "task_stage",
    }
    assert schema["properties"]["capabilities"]["additionalProperties"] is False


def test_compatibility_policy_covers_all_schema_versions():
    """兼容清单必须覆盖全部公开 schema，并与 schema_version const 一致。"""
    policy = _load_json(COMPATIBILITY)
    assert set(policy["schemas"]) == set(SCHEMA_FILES)
    for kind, filename in SCHEMA_FILES.items():
        entry = policy["schemas"][kind]
        schema = _load_json(SCHEMAS / filename)
        assert entry["file"] == filename
        assert entry["schema_version"] == schema["properties"][
            "schema_version"
        ]["const"]


def test_root_unknown_field_policy_is_enforced():
    """开放/封闭根对象策略必须与兼容清单一致，避免客户端歧义。"""
    policy = _load_json(COMPATIBILITY)
    for kind, sample_path in _VALID_SAMPLES.items():
        sample = _load_json(sample_path)
        sample["__future_field__"] = "probe"
        errors = list(_validator(kind).iter_errors(sample))
        expected = policy["schemas"][kind]["root_unknown_fields"]
        assert bool(errors) is (expected == "reject"), kind


def test_evolution_policy_marks_breaking_changes():
    policy = _load_json(COMPATIBILITY)["evolution"]
    compatible = set(policy["compatible_changes"])
    breaking = set(policy["version_bump_required_for"])
    assert compatible
    assert breaking
    assert compatible.isdisjoint(breaking)
    assert {
        "add_required_property",
        "change_enum_members",
        "change_unknown_field_policy",
        "change_state_transition",
    } <= breaking


def test_valid_fetcher_sequence_passes_policy():
    data = _load_json(FIXTURES / "fetcher-sequence-valid.json")
    assert _sequence_errors(data) == []


@pytest.mark.parametrize(
    "filename",
    [
        "fetcher-sequence-invalid-transition.json",
        "fetcher-sequence-invalid-after-terminal.json",
        "fetcher-sequence-invalid-job-id.json",
        "fetcher-sequence-invalid-regression.json",
    ],
)
def test_invalid_fetcher_sequences_fail_policy(filename):
    data = _load_json(FIXTURES / filename)
    assert _sequence_errors(data), f"{filename} 应被序列策略拒绝"


def test_migration_map_data_validates_against_schema():
    """migration-map.json 数据文件自身必须通过 migration-map schema。"""
    data = json.loads((SCHEMAS / "migration-map.json").read_text(encoding="utf-8"))
    errors = list(_validator("migration-map").iter_errors(data))
    assert errors == [], f"migration-map.json 应通过校验: {[e.message for e in errors]}"


def test_migration_map_is_self_consistent():
    """迁移映射清单自洽：目标表已登记、源 id 唯一、幂等键与字段映射非空。"""
    data = json.loads((SCHEMAS / "migration-map.json").read_text(encoding="utf-8"))
    target_tables = set(data["target_tables"])
    source_ids = []
    for source in data["sources"]:
        source_ids.append(source["source_id"])
        assert source["target"]["table"] in target_tables, (
            f"{source['source_id']} 的目标表未登记: {source['target']['table']}"
        )
        assert source["idempotency"].strip()
        assert source["target"]["field_map"]
        assert source["id_field"].strip()
    assert len(source_ids) == len(set(source_ids)), "source_id 必须唯一"
    assert len(data["rules"]) >= 1


def test_migration_map_prescribes_no_duplicate_artifacts():
    """验收口径：迁移不产生重复 artifact（摘要去重规则存在）。"""
    data = json.loads((SCHEMAS / "migration-map.json").read_text(encoding="utf-8"))
    joined = " ".join(data["rules"])
    assert "重复" in joined and "sha256" in joined
