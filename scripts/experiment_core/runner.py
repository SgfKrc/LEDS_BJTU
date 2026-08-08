"""实验单元执行器（EX-N1）：命令模板替换、超时、重试、日志与结果读取。"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan import ExperimentUnit


class RunnerError(RuntimeError):
    """单元执行失败（非重试性）。"""


@dataclass
class UnitOutcome:
    experiment_id: str
    status: str  # passed | failed | invalid
    exit_code: int | None
    duration_s: float
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    retries: list[dict[str, Any]] = field(default_factory=list)
    raw_log: str = ""


def _read_result_file(path: Path | None, unit_id: str) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{unit_id}: 结果文件不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{unit_id}: 结果文件必须是对象: {path}")
    return value


def run_unit(
    unit: ExperimentUnit,
    *,
    out_dir: Path,
    prompt_set_dir: Path,
    plan_id: str,
    python: str = "python",
) -> UnitOutcome:
    """执行一个实验单元，含超时与重试；每次尝试的原始日志保留。

    约定：命令通过 {out_dir}/{experiment_id}.result.json 输出指标 JSON
    （或经 result_file 指定）；框架只读取该文件，不解析自由文本。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = unit.rendered_result_file(out_dir=out_dir)
    command = unit.render_command(
        out_dir=out_dir, prompt_set_dir=prompt_set_dir, plan_id=plan_id,
    )
    retries: list[dict[str, Any]] = []
    last_exit: int | None = None
    last_error = ""
    started = time.monotonic()
    for attempt in range(unit.max_retries + 1):
        log_path = out_dir / f"{unit.experiment_id}.log"
        try:
            with open(log_path, "wb") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=unit.timeout_s,
                    check=False,
                )
            last_exit = completed.returncode
            if completed.returncode == 0:
                try:
                    metrics = _read_result_file(result_file, unit.experiment_id)
                except RunnerError as exc:
                    # 命令成功但结果文件非法 → 数据不可用，按失败重试
                    last_error = str(exc)
                    retries.append({
                        "attempt": attempt + 1,
                        "exit_code": 0,
                        "reason": str(exc),
                        "log": str(log_path),
                    })
                    if attempt < unit.max_retries:
                        continue
                    return UnitOutcome(
                        experiment_id=unit.experiment_id,
                        status="failed",
                        exit_code=0,
                        duration_s=time.monotonic() - started,
                        error=str(exc),
                        metrics={},
                        retries=retries,
                        raw_log=str(log_path),
                    )
                return UnitOutcome(
                    experiment_id=unit.experiment_id,
                    status="passed",
                    exit_code=0,
                    duration_s=time.monotonic() - started,
                    metrics=metrics,
                    retries=retries,
                    raw_log=str(log_path),
                )
            reason = f"exit code {completed.returncode}"
            last_error = reason
            retries.append({
                "attempt": attempt + 1,
                "exit_code": completed.returncode,
                "reason": reason,
                "log": str(log_path),
            })
        except subprocess.TimeoutExpired as exc:
            last_exit = None
            reason = f"timeout after {unit.timeout_s}s"
            last_error = reason
            retries.append({
                "attempt": attempt + 1,
                "exit_code": None,
                "reason": reason,
                "log": str(log_path),
            })
        except OSError as exc:
            raise RunnerError(f"{unit.experiment_id}: 无法启动命令: {exc}") from exc
    return UnitOutcome(
        experiment_id=unit.experiment_id,
        status="failed",
        exit_code=last_exit,
        duration_s=time.monotonic() - started,
        error=last_error,
        metrics={},
        retries=retries,
        raw_log=str(out_dir / f"{unit.experiment_id}.log"),
    )
