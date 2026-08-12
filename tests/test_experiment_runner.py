"""EX-N1 实验框架测试：plan 解析、runner、调度（资源锁/断点）、gate、报告。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core import collector, plan as plan_mod
from experiment_core.collector import build_record, evaluate_gate
from experiment_core.plan import PlanError, load_plan
from experiment_core.report import build_report
from experiment_core.runner import run_unit
from experiment_core.scheduler import (
    execute_plan, load_completed, resources_conflict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_SET = {
    "id": "ps-v1-zh-en-code",
    "sha256": "8cc555f57fa23d45c16820f77cc90b507da04578a1e90eebf46e19d4eb2568a3",
}

RESULT_WRITER = (
    "import json,sys;"
    "open(sys.argv[1],'w').write(json.dumps({'decode_tok_s': float(sys.argv[2])}))"
)


def _unit(experiment_id="exp-0001", *, name="unit", value=30.0, resources=None,
          baseline=None, gate=None, retries=0, timeout=60, result=None):
    return {
        "experiment_id": experiment_id,
        "name": name,
        "command": [sys.executable, "-c", RESULT_WRITER,
                    "{out_dir}/{experiment_id}.result.json", str(value)],
        "resources": resources or {"gpu": "gpu0"},
        "params": {"seed": 42, "max_new_tokens": 50},
        "model": {"id": "qwen-1_8b-chat", "quant_family": "fp16"},
        "baseline_experiment_id": baseline,
        "gate": gate or {"metric": "decode_tok_s", "op": ">=", "threshold": 10.0},
        "max_retries": retries,
        "timeout_s": timeout,
        "runs": 5,
        **({"result_file": result} if result else {}),
    }


def _plan(tmp_path, units, *, prompt_set=None):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "plan_id": "plan-test-v1",
        "title": "test plan",
        "prompt_set": prompt_set or PROMPT_SET,
        "env": {"os": "test-os", "gpu": "test-gpu"},
        "units": units,
    }, ensure_ascii=False), encoding="utf-8")
    return load_plan(path)


# --------------------------------------------------------------------------
# plan 解析与校验
# --------------------------------------------------------------------------

def test_load_plan_accepts_valid_manifest(tmp_path):
    plan = _plan(tmp_path, [_unit()])
    assert plan.plan_id == "plan-test-v1"
    assert len(plan.units) == 1
    assert plan.units[0].gate is not None
    assert plan.verify_prompt_set().name == "ps-v1-zh-en-code"


def test_plan_rejects_missing_fields(tmp_path):
    raw = {
        "prompt_set": PROMPT_SET,  # 缺 plan_id
        "units": [_unit()],
    }
    path = tmp_path / "bad0.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="plan_id"):
        load_plan(path)
    raw = {
        "plan_id": "plan-x", "prompt_set": PROMPT_SET,
        "units": [{"experiment_id": "exp-0001", "name": "n"}],  # 缺 command
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PlanError, match="command"):
        load_plan(path)


def test_plan_rejects_bad_experiment_id_and_duplicates(tmp_path):
    with pytest.raises(PlanError, match="exp-\\d"):
        _plan(tmp_path, [_unit(experiment_id="exp-1")])
    with pytest.raises(PlanError, match="重复"):
        _plan(tmp_path, [_unit(), _unit()])


def test_plan_rejects_bad_gate(tmp_path):
    with pytest.raises(PlanError, match="threshold 或 baseline_ratio"):
        _plan(tmp_path, [_unit(gate={"metric": "x", "op": ">="})])
    with pytest.raises(PlanError, match="op"):
        _plan(tmp_path, [_unit(gate={"metric": "x", "op": "!=", "threshold": 1})])


def test_plan_rejects_prompt_set_sha_mismatch(tmp_path):
    bad = dict(PROMPT_SET)
    bad["sha256"] = "0" * 64
    with pytest.raises(PlanError, match="SHA-256 mismatch"):
        _plan(tmp_path, [_unit()], prompt_set=bad).verify_prompt_set()


def test_plan_rejects_missing_prompt_set_file(tmp_path):
    bad = dict(PROMPT_SET)
    bad["id"] = "ps-does-not-exist"
    with pytest.raises(PlanError, match="not found"):
        _plan(tmp_path, [_unit()], prompt_set=bad).verify_prompt_set()


# --------------------------------------------------------------------------
# runner：命令模板、结果读取、重试、超时
# --------------------------------------------------------------------------

def test_run_unit_writes_record_and_reads_result_file(tmp_path):
    plan = _plan(tmp_path, [_unit()])
    unit = plan.units[0]
    outcome = run_unit(
        unit,
        out_dir=tmp_path / "out",
        prompt_set_dir=plan.verify_prompt_set(),
        plan_id=plan.plan_id,
    )
    assert outcome.status == "passed"
    assert outcome.metrics["decode_tok_s"] == 30.0
    assert (tmp_path / "out" / "exp-0001.result.json").is_file()
    assert (tmp_path / "out" / "exp-0001.log").is_file()


def test_run_unit_retries_and_records_attempts(tmp_path):
    failing = (
        "import json,sys,os;"
        "marker=os.path.join(os.path.dirname(sys.argv[1]),'attempts.txt');"
        "n=int(open(marker).read()) if os.path.exists(marker) else 0;"
        "open(marker,'w').write(str(n+1));"
        "open(sys.argv[1],'w').write(json.dumps({'decode_tok_s': 1.0}));"
        "import sys as _s; _s.exit(1) if n < 1 else None"
    )
    unit = {
        "experiment_id": "exp-0001", "name": "flaky",
        "command": [sys.executable, "-c", failing,
                    "{out_dir}/{experiment_id}.result.json"],
        "resources": {}, "params": {}, "model": {}, "max_retries": 2,
        "timeout_s": 60, "runs": 1,
        "gate": {"metric": "decode_tok_s", "op": ">=", "threshold": 1.0},
    }
    plan = _plan(tmp_path, [unit])
    outcome = run_unit(
        plan.units[0], out_dir=tmp_path / "out2",
        prompt_set_dir=plan.verify_prompt_set(), plan_id=plan.plan_id,
    )
    assert outcome.status == "passed"
    assert len(outcome.retries) == 1
    assert outcome.retries[0]["exit_code"] == 1
    # 重试数据与首次数据分开标记
    assert outcome.retries[0]["attempt"] == 1


def test_run_unit_timeout_marks_failed(tmp_path):
    unit = {
        "experiment_id": "exp-0001", "name": "slow",
        "command": [sys.executable, "-c", "import time; time.sleep(30)"],
        "resources": {}, "params": {}, "model": {},
        "max_retries": 0, "timeout_s": 1, "runs": 1,
        "gate": None,
    }
    plan = _plan(tmp_path, [unit])
    outcome = run_unit(
        plan.units[0], out_dir=tmp_path / "out3",
        prompt_set_dir=plan.verify_prompt_set(), plan_id=plan.plan_id,
    )
    assert outcome.status == "failed"
    assert "timeout" in outcome.retries[0]["reason"]


# --------------------------------------------------------------------------
# 调度：资源冲突、并行、断点续跑
# --------------------------------------------------------------------------

def test_resources_conflict_on_shared_key():
    assert resources_conflict({"gpu": "gpu0"}, {"gpu": "gpu0"}) is True
    assert resources_conflict({"gpu": "gpu0"}, {"gpu": "gpu1"}) is False
    assert resources_conflict({"gpu": "any"}, {"gpu": "gpu0"}) is True
    assert resources_conflict({"gpu": "gpu0"}, {"port": "9000"}) is False


def test_execute_plan_serial_and_conflict_queues(tmp_path):
    plan = _plan(tmp_path, [
        _unit("exp-0001", resources={"gpu": "gpu0"}),
        _unit("exp-0002", resources={"gpu": "gpu0"}),
    ])
    order: list[str] = []
    outcomes = execute_plan(
        plan, out_dir=tmp_path / "out4", prompt_set_dir=tmp_path,
        run_fn=lambda unit: (order.append(unit.experiment_id), None)[1],
        parallel=2,
    )
    # 同 GPU 冲突 → 串行排队，顺序与 manifest 一致
    assert order == ["exp-0001", "exp-0002"]
    assert set(outcomes) == {"exp-0001", "exp-0002"}


def test_execute_plan_parallel_without_conflict(tmp_path):
    import threading
    import time

    plan = _plan(tmp_path, [
        _unit("exp-0001", resources={"gpu": "gpu0"}),
        _unit("exp-0002", resources={"gpu": "gpu1"}),
        _unit("exp-0003", resources={"gpu": "gpu0"}),
    ])
    active = 0
    max_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def fake_run(unit):
        nonlocal active, max_active
        if unit.experiment_id in ("exp-0001", "exp-0002"):
            # gpu0/gpu1 无冲突单元必须真正并发：互相等待，单线程会死锁超时
            barrier.wait(timeout=5)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return unit

    execute_plan(
        plan, out_dir=tmp_path / "out5", prompt_set_dir=tmp_path,
        run_fn=fake_run, parallel=2,
    )
    # exp-0001/exp-0002 并发（barrier 证明），峰值活跃 ≥ 2
    assert max_active >= 2
    # exp-0003 与 exp-0001 同 GPU → 后续批次，不与 exp-0001 并发
    assert max_active < 3


def test_execute_plan_resume_skips_completed(tmp_path):
    plan = _plan(tmp_path, [
        _unit("exp-0001"), _unit("exp-0002"),
    ])
    out_dir = tmp_path / "out6"
    out_dir.mkdir()
    collector.append_record(out_dir / "records.jsonl", {
        "experiment_id": "exp-0001", "status": "passed",
        "metrics": {"decode_tok_s": 1.0}, "timestamp": "t",
    })
    # failed 的旧记录必须重跑
    collector.append_record(out_dir / "records.jsonl", {
        "experiment_id": "exp-0002", "status": "failed",
        "metrics": {}, "timestamp": "t",
    })
    ran: list[str] = []
    execute_plan(
        plan, out_dir=out_dir, prompt_set_dir=tmp_path,
        run_fn=lambda unit: ran.append(unit.experiment_id),
        resume=True,
    )
    assert ran == ["exp-0002"]  # passed 跳过、failed 重跑
    assert load_completed(out_dir)["exp-0001"]["status"] == "passed"


# --------------------------------------------------------------------------
# gate 判定
# --------------------------------------------------------------------------

def test_gate_threshold_and_baseline_ratio():
    gate = plan_mod.GateSpec("decode_tok_s", ">=", threshold=10.0)
    status, desc = evaluate_gate(gate, {"decode_tok_s": 12.0}, baseline_metrics=None)
    assert status == "passed" and ">= 10" in desc
    status, _ = evaluate_gate(gate, {"decode_tok_s": 8.0}, baseline_metrics=None)
    assert status == "failed"
    # 缺指标 → invalid
    status, reason = evaluate_gate(gate, {}, baseline_metrics=None)
    assert status == "invalid" and "缺少指标" in reason

    ratio_gate = plan_mod.GateSpec("decode_tok_s", ">=", baseline_ratio=0.9)
    status, desc = evaluate_gate(
        ratio_gate, {"decode_tok_s": 27.0}, baseline_metrics={"decode_tok_s": 30.0},
    )
    assert status == "passed" and ">= 27" in desc
    status, _ = evaluate_gate(
        ratio_gate, {"decode_tok_s": 26.0}, baseline_metrics={"decode_tok_s": 30.0},
    )
    assert status == "failed"


# --------------------------------------------------------------------------
# record 组装与报告
# --------------------------------------------------------------------------

def test_build_record_has_all_schema_fields(tmp_path):
    plan = _plan(tmp_path, [_unit()], )
    unit = plan.units[0]
    outcome = run_unit(
        unit, out_dir=tmp_path / "out7",
        prompt_set_dir=plan.verify_prompt_set(), plan_id=plan.plan_id,
    )
    record = build_record(
        plan, unit, outcome,
        prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    assert collector.validate_record(record) == []
    assert record["experiment_id"] == "exp-0001"
    assert record["prompt_set"]["count"] == 30
    assert record["prompt_set"]["sha256"] == PROMPT_SET["sha256"]
    assert record["gate"]["status"] == "passed"
    assert record["metrics"]["decode_tok_s"] == 30.0
    assert record["commit"] == collector.current_commit()


def test_report_contains_comparison_table_and_summary(tmp_path):
    plan = _plan(tmp_path, [
        _unit("exp-0001", value=30.0),
        _unit("exp-0002", value=45.0, baseline="exp-0001"),
    ])
    out_dir = tmp_path / "out8"
    records: dict[str, dict] = {}
    for unit in plan.units:
        outcome = run_unit(
            unit, out_dir=out_dir,
            prompt_set_dir=plan.verify_prompt_set(), plan_id=plan.plan_id,
        )
        record = build_record(
            plan, unit, outcome,
            prompt_set_dir=plan.verify_prompt_set(), records=records,
        )
        records[unit.experiment_id] = record
    report_path, summary_path = build_report(
        out_dir, list(records.values()),
        {"plan_id": plan.plan_id, "title": plan.title,
         "env": plan.env, "prompt_set": plan.prompt_set},
    )
    text = report_path.read_text(encoding="utf-8")
    assert "对照表" in text
    assert "+50.00%" in text  # (45-30)/30
    assert "passed 2 / failed 0 / invalid 0" in text
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == {"passed": 2, "failed": 0, "invalid": 0}


# --------------------------------------------------------------------------
# CLI 端到端
# --------------------------------------------------------------------------

def test_cli_check_and_full_run(tmp_path):
    from experiment_core import cli
    plan_path = tmp_path / "cli-plan.json"
    plan_path.write_text(json.dumps({
        "plan_id": "plan-cli-v1",
        "prompt_set": PROMPT_SET,
        "env": {},
        "units": [_unit("exp-0001")],
    }, ensure_ascii=False), encoding="utf-8")
    assert cli.main(["--plan", str(plan_path), "--check"]) == 0
    out = tmp_path / "cli-out"
    assert cli.main(["--plan", str(plan_path), "--out", str(out)]) == 0
    records = load_completed(out)
    assert records["exp-0001"]["gate"]["status"] == "passed"
    assert (out / "report.md").is_file()
    assert (out / "summary.json").is_file()
    # 断点续跑：二次运行不重复执行（records 不翻倍）
    count_before = len((out / "records.jsonl").read_text(encoding="utf-8").splitlines())
    assert cli.main(["--plan", str(plan_path), "--out", str(out), "--resume"]) == 0
    count_after = len((out / "records.jsonl").read_text(encoding="utf-8").splitlines())
    assert count_after == count_before


def test_cli_refuses_live_output_lock_and_requires_explicit_stale_recovery(tmp_path):
    from experiment_core import cli

    out = tmp_path / "locked-out"
    out.mkdir()
    (out / ".run.lock").write_text(
        json.dumps({"pid": __import__("os").getpid(), "plan_id": "other"}),
        encoding="utf-8",
    )
    plan_path = tmp_path / "locked-plan.json"
    plan_path.write_text(json.dumps({
        "plan_id": "plan-lock-v1", "prompt_set": PROMPT_SET, "env": {},
        "units": [_unit("exp-0001")],
    }), encoding="utf-8")
    assert cli.main(["--plan", str(plan_path), "--out", str(out)]) == 2

    (out / ".run.lock").write_text(json.dumps({"pid": 999999, "plan_id": "old"}), encoding="utf-8")
    assert cli.main(["--plan", str(plan_path), "--out", str(out)]) == 2
    assert cli.main(["--plan", str(plan_path), "--out", str(out), "--recover-lock"]) == 0
    assert not (out / ".run.lock").exists()
