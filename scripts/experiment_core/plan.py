"""实验计划 manifest 解析与校验（EX-N1）。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROMPT_SETS = _PROJECT_ROOT / "fixtures" / "prompt_sets"

_EXPERIMENT_ID_RE = re.compile(r"^exp-[0-9]{4}$")
_GATE_OPS = {">=", "<=", ">", "<", "=="}


class PlanError(ValueError):
    """Manifest 非法或提示词集不匹配。"""


@dataclass(frozen=True)
class GateSpec:
    metric: str
    op: str
    threshold: float | None = None
    baseline_ratio: float | None = None


@dataclass(frozen=True)
class ExperimentUnit:
    experiment_id: str
    name: str
    command: list[str]
    resources: Mapping[str, str]
    params: Mapping[str, Any]
    model: Mapping[str, Any]
    gate: GateSpec | None
    baseline_experiment_id: str | None
    runs: int
    timeout_s: int
    max_retries: int
    result_file: str | None = None
    prompt_set: Mapping[str, Any] = field(default_factory=dict)

    def render_command(self, *, out_dir: Path, prompt_set_dir: Path, plan_id: str) -> list[str]:
        """把 {out_dir}/{experiment_id}/{prompt_set_dir}/{plan_id} 占位符替换为实参。"""
        rendered: list[str] = []
        for token in self.command:
            rendered.append(
                token
                .replace("{out_dir}", str(out_dir))
                .replace("{experiment_id}", self.experiment_id)
                .replace("{prompt_set_dir}", str(prompt_set_dir))
                .replace("{plan_id}", plan_id)
            )
        return rendered

    def rendered_result_file(self, *, out_dir: Path) -> Path | None:
        # 默认约定：{out_dir}/{experiment_id}.result.json（命令把指标 JSON 写到该路径）。
        if not self.result_file:
            return out_dir / f"{self.experiment_id}.result.json"
        return Path(
            self.result_file
            .replace("{out_dir}", str(out_dir))
            .replace("{experiment_id}", self.experiment_id)
        )


@dataclass(frozen=True)
class PlanManifest:
    plan_id: str
    title: str
    prompt_set: Mapping[str, Any]
    env: Mapping[str, Any]
    units: tuple[ExperimentUnit, ...]
    defaults: Mapping[str, Any]
    source_path: Path | None = None

    def prompt_set_dir(self, root: Path | None = None) -> Path:
        base = (root or _PROJECT_ROOT) / "fixtures" / "prompt_sets" / self.prompt_set["id"]
        return base

    def verify_prompt_set(self, root: Path | None = None) -> Path:
        """校验提示词集存在且 SHA-256 与 manifest 一致；返回提示词集目录。"""
        directory = self.prompt_set_dir(root)
        prompts = directory / "prompts.jsonl"
        if not prompts.is_file():
            raise PlanError(
                f"prompt set {self.prompt_set['id']!r} not found: {prompts}"
            )
        declared = self.prompt_set.get("sha256")
        if declared:
            digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
            if digest != declared:
                raise PlanError(
                    f"prompt set {self.prompt_set['id']!r} SHA-256 mismatch: "
                    f"declared {declared}, actual {digest}; 禁止原地覆盖提示词集"
                )
        return directory


def _require(value: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in value or value[key] in (None, ""):
        raise PlanError(f"{what} 缺少字段 {key!r}")
    return value[key]


def _parse_gate(raw: Any, unit_id: str) -> GateSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PlanError(f"{unit_id}: gate 必须是对象")
    metric = _require(raw, "metric", unit_id)
    op = str(raw.get("op", ">="))
    if op not in _GATE_OPS:
        raise PlanError(f"{unit_id}: 不支持的 gate op {op!r}")
    threshold = raw.get("threshold")
    ratio = raw.get("baseline_ratio")
    if threshold is None and ratio is None:
        raise PlanError(f"{unit_id}: gate 必须声明 threshold 或 baseline_ratio")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise PlanError(f"{unit_id}: gate threshold 必须是数字") from exc
    if ratio is not None:
        try:
            ratio = float(ratio)
        except (TypeError, ValueError) as exc:
            raise PlanError(f"{unit_id}: gate baseline_ratio 必须是数字") from exc
    return GateSpec(metric=str(metric), op=op, threshold=threshold, baseline_ratio=ratio)


def _parse_unit(raw: Mapping[str, Any], defaults: Mapping[str, Any], plan_id: str) -> ExperimentUnit:
    experiment_id = str(_require(raw, "experiment_id", "unit"))
    if not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise PlanError(f"{experiment_id}: experiment_id 必须匹配 exp-\\d{{4}}")
    name = str(_require(raw, "name", experiment_id))
    command_raw = raw.get("command") or defaults.get("command")
    if not command_raw:
        raise PlanError(f"{experiment_id}: 缺少 command（单元或 defaults）")
    command = command_raw if isinstance(command_raw, list) else [str(command_raw)]
    if not all(isinstance(token, str) for token in command):
        raise PlanError(f"{experiment_id}: command 必须全是字符串")
    resources_raw = raw.get("resources") or defaults.get("resources") or {}
    if not isinstance(resources_raw, Mapping):
        raise PlanError(f"{experiment_id}: resources 必须是对象")
    resources = {str(k): str(v) for k, v in resources_raw.items()}
    params_raw = raw.get("params") or {}
    if not isinstance(params_raw, Mapping):
        raise PlanError(f"{experiment_id}: params 必须是对象")
    model_raw = raw.get("model") or {}
    if not isinstance(model_raw, Mapping):
        raise PlanError(f"{experiment_id}: model 必须是对象")
    baseline = raw.get("baseline_experiment_id")
    if baseline is not None:
        baseline = str(baseline)
        if not _EXPERIMENT_ID_RE.fullmatch(baseline):
            raise PlanError(f"{experiment_id}: baseline_experiment_id 格式非法")
    runs = int(raw.get("runs", defaults.get("runs", 5)))
    if runs < 1:
        raise PlanError(f"{experiment_id}: runs 必须 >= 1")
    timeout = int(raw.get("timeout_s", defaults.get("timeout_s", 300)))
    if timeout < 1:
        raise PlanError(f"{experiment_id}: timeout_s 必须 >= 1")
    retries = int(raw.get("max_retries", defaults.get("max_retries", 0)))
    if retries < 0:
        raise PlanError(f"{experiment_id}: max_retries 必须 >= 0")
    result_file = raw.get("result_file")
    prompt_set = raw.get("prompt_set") or {}
    if not isinstance(prompt_set, Mapping):
        raise PlanError(f"{experiment_id}: prompt_set 必须是对象")
    return ExperimentUnit(
        experiment_id=experiment_id,
        name=name,
        command=command,
        resources=resources,
        params=dict(params_raw),
        model=dict(model_raw),
        gate=_parse_gate(raw.get("gate"), experiment_id),
        baseline_experiment_id=baseline,
        runs=runs,
        timeout_s=timeout,
        max_retries=retries,
        result_file=str(result_file) if result_file else None,
        prompt_set=dict(prompt_set),
    )


def load_plan(path: str | Path) -> PlanManifest:
    """解析并校验实验计划 manifest。"""
    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"无法解析实验计划: {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise PlanError("实验计划根节点必须是对象")
    plan_id = str(_require(raw, "plan_id", "plan"))
    prompt_set = raw.get("prompt_set")
    if not isinstance(prompt_set, Mapping) or not prompt_set.get("id"):
        raise PlanError("plan 缺少 prompt_set.id")
    env_raw = raw.get("env") or {}
    if not isinstance(env_raw, Mapping):
        raise PlanError("env 必须是对象")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise PlanError("defaults 必须是对象")
    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        raise PlanError("plan 必须包含非空 units 列表")
    seen: set[str] = set()
    units: list[ExperimentUnit] = []
    for item in units_raw:
        if not isinstance(item, Mapping):
            raise PlanError("unit 必须是对象")
        unit = _parse_unit(item, defaults, plan_id)
        if unit.experiment_id in seen:
            raise PlanError(f"重复的 experiment_id: {unit.experiment_id}")
        seen.add(unit.experiment_id)
        units.append(unit)
    return PlanManifest(
        plan_id=plan_id,
        title=str(raw.get("title", plan_id)),
        prompt_set=dict(prompt_set),
        env=dict(env_raw),
        units=tuple(units),
        defaults=dict(defaults),
        source_path=source,
    )
