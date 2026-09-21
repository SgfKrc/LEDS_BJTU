"""P1 统一实验驱动的守卫用例：记录 schema、kind/path 一致性与 CLI 预检。

这些用例**不需要模型工件**（只跑 dry-run 与纯函数），因此总是可执行；真模型链路由
`scripts/relay_experiment.py` 在实验机上跑，产物是 relay experiment record。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_experiment_record import (  # noqa: E402
    EXPERIMENT_KINDS,
    IFACE_LLAMA_ENGINE_DOWNSTREAM,
    IFACE_LLAMA_MODEL_LOADER,
    IFACE_LLAMA_UPSTREAM,
    IFACE_MODEL_MODULE_LOADER,
    IFACE_MODEL_MODULE_UPSTREAM,
    IFACE_RAW_LLAMA_DOWNSTREAM,
    KIND_CAPACITY_ONLY,
    KIND_MAINREPO_END_TO_END,
    KIND_RAW_BINDING_PROBE,
    PATH_CAPACITY,
    PATH_D2L_MAINREPO,
    PATH_D2L_RAW,
    PATH_L2L,
    SCHEMA_VERSION,
    build_record,
    classify_path,
    load_schema,
    make_experiment_id,
    validate_record,
    write_record,
)

SCRIPT = ROOT / "scripts" / "relay_experiment.py"


def _cli_module():
    spec = importlib.util.spec_from_file_location("relay_experiment_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_record(**overrides):
    payload = {
        "upstream_iface": IFACE_MODEL_MODULE_UPSTREAM,
        "downstream_iface": IFACE_LLAMA_ENGINE_DOWNSTREAM,
        "path": PATH_D2L_MAINREPO,
        "models": {"upstream": {"id": "qwen2.5-0.5b-instruct"},
                   "downstream": {"path": "cut.gguf", "layers": 12},
                   "whole": {"model_bytes": 988208640}},
        "layer_layout": {"upstream_layers": 12, "downstream_layers": 12},
        "handoff": {"dtype": "float32", "n_embd": 896},
        "load": {"prefill_tokens": 32, "gen_tokens": 32, "batch": 1},
        "verdict": {"passed": True, "tokens_match": True},
        "metrics": {"capacity_gain_x": 1.568},
        "env": None,
        "device_profile": {},
        "evidence": None,
        "artifacts": {},
        "timestamp": "2026-09-21T00:00:00Z",
        "commit": "deadbeef",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------- schema 自检
def test_schema_is_a_valid_json_schema():
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["kind"]["enum"] == list(EXPERIMENT_KINDS)


def test_schema_forbids_mislabeled_kinds_at_the_schema_level():
    """schema 条件规则：mainrepo_end_to_end 只能配 d2l_mainrepo + 主仓下游入口。"""
    validator = jsonschema.Draft202012Validator(load_schema())
    record = build_record(**_minimal_record())
    assert not list(validator.iter_errors(record))

    mislabeled = dict(record, kind=KIND_RAW_BINDING_PROBE)  # path 仍是 d2l_mainrepo
    assert list(validator.iter_errors(mislabeled)), "kind/path 不一致必须被 schema 拒绝"

    wrong_iface = json.loads(json.dumps(record))
    wrong_iface["engines"]["downstream_iface"] = IFACE_RAW_LLAMA_DOWNSTREAM
    assert list(validator.iter_errors(wrong_iface)), "下游接口与 kind 不一致必须被拒绝"


# --------------------------------------------------------------- 分类
@pytest.mark.parametrize(("upstream", "downstream", "expected"), [
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM,
     (KIND_MAINREPO_END_TO_END, PATH_D2L_MAINREPO)),
    (IFACE_MODEL_MODULE_UPSTREAM, IFACE_RAW_LLAMA_DOWNSTREAM,
     (KIND_RAW_BINDING_PROBE, PATH_D2L_RAW)),
    (IFACE_LLAMA_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM,
     (KIND_RAW_BINDING_PROBE, PATH_L2L)),
    (IFACE_MODEL_MODULE_LOADER, IFACE_LLAMA_MODEL_LOADER,
     (KIND_CAPACITY_ONLY, PATH_CAPACITY)),
])
def test_classify_path_maps_every_registered_link(upstream, downstream, expected):
    assert classify_path(upstream, downstream) == expected


def test_classify_path_rejects_unregistered_links():
    with pytest.raises(ValueError, match="未登记的引擎接口组合"):
        classify_path("torch.manual_forward", IFACE_LLAMA_ENGINE_DOWNSTREAM)


# --------------------------------------------------------------- build/validate
def test_build_record_infers_kind_and_fills_identity_fields():
    record = build_record(**_minimal_record())
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["kind"] == KIND_MAINREPO_END_TO_END
    assert record["path"] == PATH_D2L_MAINREPO
    assert record["criterion"] == "per_token_argmax"
    assert record["experiment_id"].startswith("relay-d2l_mainrepo")
    assert record["env"]["dependency_lock"] is None or "sha256" in record["env"]["dependency_lock"]


def test_build_record_rejects_kind_that_contradicts_the_interfaces():
    """★ 防混表核心：把探针/容量数字标成端到端基线必须直接失败。"""
    with pytest.raises(ValueError, match="与引擎接口不一致"):
        build_record(**_minimal_record(kind=KIND_CAPACITY_ONLY))
    with pytest.raises(ValueError, match="与引擎接口不一致"):
        build_record(**_minimal_record(path=PATH_L2L))


def test_validate_record_rejects_missing_fields_and_bad_kind():
    record = build_record(**_minimal_record())
    broken = {k: v for k, v in record.items() if k != "layer_layout"}
    with pytest.raises(ValueError, match="不合法"):
        validate_record(broken)
    with pytest.raises(ValueError, match="不合法"):
        validate_record(dict(record, kind="end_to_end"))


def test_write_record_stamps_artifact_path(tmp_path):
    record = build_record(**_minimal_record())
    target = write_record(record, tmp_path / "rec.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["artifacts"]["record_path"] == str(target)
    validate_record(payload)


def test_make_experiment_id_is_stable_and_filesystem_safe():
    assert make_experiment_id(PATH_L2L, "qwen3-5-2b", 12, 32, 64, 2) == \
        "relay-l2l_llama-qwen3-5-2b-k12-p32-g64-b2"
    assert make_experiment_id(PATH_CAPACITY, "a/b c", None, None, None, None) == \
        "relay-capacity_scan-a-b-c"


# --------------------------------------------------------------- CLI 预检
@pytest.mark.parametrize(("extra", "expected_path", "expected_kind"), [
    (["--path", "d2l_mainrepo"], PATH_D2L_MAINREPO, KIND_MAINREPO_END_TO_END),
    (["--path", "d2l_raw_binding"], PATH_D2L_RAW, KIND_RAW_BINDING_PROBE),
    (["--path", "l2l_llama"], PATH_L2L, KIND_RAW_BINDING_PROBE),
    (["--path", "capacity_scan"], PATH_CAPACITY, KIND_CAPACITY_ONLY),
])
def test_cli_dry_run_emits_a_valid_record_for_every_path(extra, expected_path, expected_kind,
                                                         capsys):
    module = _cli_module()
    code = module.main([*extra, "--dry-run", "--cut-model", "cut.gguf",
                        "--whole-model", "whole.gguf", "--json-out", "-"])
    assert code == 0, "dry-run 预检成功应返回 0"
    out = capsys.readouterr().out
    payload = json.loads(out[: out.rfind("}") + 1])
    assert payload["path"] == expected_path
    assert payload["kind"] == expected_kind
    assert payload["verdict"]["failure"] == "dry_run"
    validate_record(payload)


def test_cli_dry_run_capacity_record_has_no_fake_metrics(capsys):
    module = _cli_module()
    module.main(["--path", "capacity_scan", "--dry-run", "--cut-model", "cut.gguf",
                 "--whole-model", "whole.gguf", "--json-out", "-"])
    out = capsys.readouterr().out
    payload = json.loads(out[: out.rfind("}") + 1])
    assert payload["metrics"] == {}, "预检不得编造任何数字"
    assert payload["models"]["downstream"]["model_bytes"] is None


# --------------------------------------------------------------- keep-head / 三段链路
def test_three_segment_path_has_its_own_signature():
    """三段链路必须与两段 D→L 分开登记（中间段接口参与链路签名），否则就是混表。"""
    from src.relay_experiment_record import (
        IFACE_KEEP_HEAD_UPSTREAM,
        PATH_D2L2L_KEEP_HEAD,
    )

    assert classify_path(IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM) == (
        KIND_MAINREPO_END_TO_END, PATH_D2L_MAINREPO)
    assert classify_path(IFACE_MODEL_MODULE_UPSTREAM, IFACE_LLAMA_ENGINE_DOWNSTREAM,
                         IFACE_KEEP_HEAD_UPSTREAM) == (KIND_MAINREPO_END_TO_END,
                                                       PATH_D2L2L_KEEP_HEAD)


def test_keep_head_paths_require_the_shim():
    """keep-head 链路没有 shim 就直接 fail-loud（不要退回 embeddings 通道）。"""
    from src.relay_experiment_record import PATH_L2L_KEEP_HEAD

    module = _cli_module()
    with pytest.raises(SystemExit, match="keep-head-shim"):
        module._check_l2l_upstream_channel(PATH_L2L_KEEP_HEAD, False, None)
    module._check_l2l_upstream_channel(PATH_L2L_KEEP_HEAD, False, "shim.dll")


@pytest.mark.parametrize("path", ["l2l_keep_head", "d2l2l_keep_head"])
def test_cli_dry_run_covers_keep_head_paths(path, capsys):
    module = _cli_module()
    argv = ["--path", path, "--dry-run", "--cut-model", "cut.gguf",
            "--whole-model", "whole.gguf", "--keep-head-shim", "shim.dll", "--json-out", "-"]
    if path == "l2l_keep_head":
        argv += ["--upstream-model", "head.gguf"]
    else:
        argv += ["--mid-model", "mid.gguf", "--mid-layers", "16"]
    assert module.main(argv) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[: out.rfind("}") + 1])
    assert payload["path"] == path
    validate_record(payload)


# --------------------------------------------------------------- L→L 上游通道守卫
def test_l2l_upstream_channel_guard_is_fail_loud():
    """pip 绑定的 embeddings 通道 = output_norm(H) ⇒ 默认拒绝，必须显式开关才放行。"""
    module = _cli_module()
    with pytest.raises(SystemExit, match="output_norm"):
        module._check_l2l_upstream_channel(PATH_L2L, False)
    module._check_l2l_upstream_channel(PATH_L2L, True)          # 显式开关放行
    module._check_l2l_upstream_channel(PATH_D2L_MAINREPO, False)  # 其它链路不受影响


def test_l2l_dry_run_cli_requires_the_explicit_switch():
    """dry-run 只做字段预检（不触通道），因此不需要开关；真实运行才由守卫拦。"""
    module = _cli_module()
    assert module.main(["--path", "l2l_llama", "--dry-run", "--cut-model", "cut.gguf",
                        "--whole-model", "whole.gguf", "--json-out", "-"]) == 0
