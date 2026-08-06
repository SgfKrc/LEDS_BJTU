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
from pathlib import Path

import pytest
import jsonschema

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "fixtures" / "model-fleet"

SCHEMA_FILES = {
    "artifact-manifest": "artifact-manifest.schema.json",
    "pull-job": "pull-job.schema.json",
    "deployment": "deployment.schema.json",
    "cluster-profile": "cluster-profile.schema.json",
    "migration-map": "migration-map.schema.json",
    "fetcher-progress": "fetcher-progress.schema.json",
}

_VALIDATORS = {}


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
