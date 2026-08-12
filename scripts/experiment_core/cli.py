"""run_experiments.py CLI（EX-N1）。

用法：
    python scripts/run_experiments.py --plan fixtures/experiment-plan.example.json
    python scripts/run_experiments.py --plan plan.json --parallel 2 --resume
    python scripts/run_experiments.py --plan plan.json --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .collector import append_record, build_record, validate_record
from .plan import PlanError, load_plan
from .report import build_report
from .runner import RunnerError, run_unit
from .scheduler import ConflictError, execute_plan, load_completed

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_experiments",
        description="QLH 全自动优化实验调度器（EX-N1）。",
    )
    parser.add_argument("--plan", required=True, help="实验计划 manifest（JSON）")
    parser.add_argument("--parallel", type=int, default=1, help="最大并行单元数（资源冲突自动降级排队）")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已完成单元")
    parser.add_argument("--out", help="输出目录（默认 build/experiments/<plan_id>-<时间戳>）")
    parser.add_argument("--check", action="store_true", help="只校验 plan 与提示词集，不执行")
    parser.add_argument(
        "--recover-lock", action="store_true",
        help="仅当先前运行进程已退出时，清理遗留的输出目录执行锁",
    )
    parser.add_argument("--python", default=sys.executable, help="runner 命令使用的 python（默认当前解释器）")
    return parser


def _default_out_dir(plan_id: str) -> Path:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return _PROJECT_ROOT / "build" / "experiments" / f"{plan_id}-{stamp}"


def _pid_is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a signal-0 existence probe on Windows: it can
        # terminate the target. Query a process handle instead.
        import ctypes

        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information | synchronize,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(exit_code),
            ):
                return True
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def output_run_lock(out_dir: Path, *, plan_id: str, recover: bool = False):
    """Make concurrent writers to an experiment output directory fail closed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".run.lock"
    payload = json.dumps({"pid": os.getpid(), "plan_id": plan_id})
    if lock_path.exists():
        try:
            prior = json.loads(lock_path.read_text(encoding="utf-8"))
            prior_pid = int(prior.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            prior_pid = 0
        if _pid_is_alive(prior_pid):
            raise RuntimeError("output directory is already being executed by a live process")
        if not recover:
            raise RuntimeError("output directory has a stale run lock; rerun with --recover-lock after review")
        lock_path.unlink(missing_ok=True)
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError("output directory run lock appeared concurrently") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        prompt_set_dir = plan.verify_prompt_set()
    except PlanError as exc:
        print(f"实验计划校验失败: {exc}", file=sys.stderr)
        return 2
    if args.check:
        print(f"计划校验通过：{plan.plan_id}（{len(plan.units)} 个单元，提示词集 {plan.prompt_set['id']}）")
        return 0

    out_dir = Path(args.out).expanduser() if args.out else _default_out_dir(plan.plan_id)

    def run_one(unit):
        try:
            return run_unit(
                unit,
                out_dir=out_dir,
                prompt_set_dir=prompt_set_dir,
                plan_id=plan.plan_id,
                python=args.python,
            )
        except RunnerError as exc:
            print(f"  {unit.experiment_id}: 执行器错误 {exc}", file=sys.stderr)
            raise

    try:
        lock = output_run_lock(out_dir, plan_id=plan.plan_id, recover=args.recover_lock)
        lock.__enter__()
    except RuntimeError as exc:
        print(f"实验输出目录不可用: {exc}", file=sys.stderr)
        return 2

    records_path = out_dir / "records.jsonl"
    # Keep completed records available to quality baseline comparisons on resume.
    # The directory lock makes this snapshot stable for the duration of this run.
    records: dict[str, dict] = load_completed(out_dir)

    def persist(unit, outcome) -> None:
        record = build_record(
            plan, unit, outcome,
            prompt_set_dir=prompt_set_dir,
            records=records,
        )
        missing = validate_record(record)
        if missing:
            print(f"  {unit.experiment_id}: 记录缺少字段 {missing}", file=sys.stderr)
        if unit.experiment_id in records:
            # 断点续跑重跑（failed/invalid）：替换旧记录，避免 records.jsonl 重复行
            lines = [
                json.dumps(record, ensure_ascii=False)
                for rid, record in records.items()
                if rid != unit.experiment_id
            ]
            lines.append(json.dumps(record, ensure_ascii=False))
            records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            append_record(records_path, record)
        records[unit.experiment_id] = record
        status = record["gate"].get("status", "?")
        metric = record["gate"].get("metric", "")
        value = record["metrics"].get(metric)
        print(
            f"  {unit.experiment_id}: {status}"
            + (f" ({metric}={value})" if value is not None else "")
            + (f" [{outcome.error}]" if outcome.error else "")
        )

    try:
        print(f"运行实验计划 {plan.plan_id}（{len(plan.units)} 单元，parallel={args.parallel}）")
        print(f"输出目录: {out_dir}")
        try:
            execute_plan(
                plan,
                out_dir=out_dir,
                prompt_set_dir=prompt_set_dir,
                run_fn=run_one,
                parallel=args.parallel,
                resume=args.resume,
                on_unit_done=persist,
            )
        except ConflictError as exc:
            print(f"调度失败: {exc}", file=sys.stderr)
            return 2

        all_records = list(load_completed(out_dir).values())
        report_path, summary_path = build_report(
            out_dir, all_records,
            {
                "plan_id": plan.plan_id,
                "title": plan.title,
                "env": plan.env,
                "prompt_set": plan.prompt_set,
                "quality": plan.quality.as_mapping() if plan.quality else None,
            },
        )
        print(f"报告: {report_path}")
        print(f"摘要: {summary_path}")
        status = {r["gate"].get("status") for r in all_records}
        if status == {"passed"}:
            return 0
        if "failed" in status:
            return 1
        return 0 if status <= {"passed", "invalid"} else 1
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
