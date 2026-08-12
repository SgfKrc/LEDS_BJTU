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
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUALITY_CHECKS = {"llm", "sd", "gemma_judge"}


class PlanError(ValueError):
    """Manifest 非法或提示词集不匹配。"""


@dataclass(frozen=True)
class GateSpec:
    metric: str
    op: str
    threshold: float | None = None
    baseline_ratio: float | None = None


@dataclass(frozen=True)
class QualitySpec:
    """Frozen EX-N3 quality contract declared by an experiment plan.

    Only identifiers, hashes, thresholds, and review policy belong here.  The
    rubric source and model completions are deliberately never copied into an
    experiment record.
    """

    required: bool
    llm: Mapping[str, Any] | None = None
    sd: Mapping[str, Any] | None = None
    gemma_judge: Mapping[str, Any] | None = None
    manual_review: Mapping[str, Any] | None = None
    calibration: Mapping[str, Any] | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "llm": dict(self.llm) if self.llm else None,
            "sd": dict(self.sd) if self.sd else None,
            "gemma_judge": dict(self.gemma_judge) if self.gemma_judge else None,
            "manual_review": dict(self.manual_review) if self.manual_review else None,
            "calibration": dict(self.calibration) if self.calibration else None,
        }


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
    quality_checks: tuple[str, ...] = ()

    def render_command(
        self,
        *,
        out_dir: Path,
        prompt_set_dir: Path,
        plan_id: str,
        python: str = "python",
    ) -> list[str]:
        """Replace the bounded command placeholders with execution arguments."""
        rendered: list[str] = []
        for token in self.command:
            rendered.append(
                token
                .replace("{out_dir}", str(out_dir))
                .replace("{experiment_id}", self.experiment_id)
                .replace("{prompt_set_dir}", str(prompt_set_dir))
                .replace("{plan_id}", plan_id)
                .replace("{python}", python)
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
    quality: QualitySpec | None = None
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
        self._verify_quality_contract(directory)
        return directory

    def _verify_quality_contract(self, prompt_set_dir: Path) -> None:
        """Verify plan-pinned EX-N3 rubric identity without persisting it."""
        if self.quality is None or self.quality.llm is None:
            return
        llm = self.quality.llm
        rubric_id = str(llm["rubric_id"])
        rubric_path = _PROJECT_ROOT / "fixtures" / "quality_rubrics" / f"{rubric_id}.json"
        try:
            raw = rubric_path.read_bytes()
            rubric = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"EX-N3 rubric unavailable: {rubric_id}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != llm["rubric_sha256"]:
            raise PlanError("EX-N3 rubric SHA-256 mismatch")
        if not isinstance(rubric, Mapping):
            raise PlanError("EX-N3 rubric must be an object")
        rubric_prompt_set = rubric.get("prompt_set")
        if not isinstance(rubric_prompt_set, Mapping) or (
            rubric_prompt_set.get("id") != self.prompt_set["id"]
            or rubric_prompt_set.get("sha256") != self.prompt_set.get("sha256")
        ):
            raise PlanError("EX-N3 rubric prompt set does not match plan")
        try:
            prompt_ids = {
                str(json.loads(line)["id"])
                for line in (prompt_set_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PlanError("cannot read prompt-set identifiers for EX-N3") from exc
        configured = set(llm["objective_prompt_ids"])
        if not configured.issubset(prompt_ids):
            raise PlanError("EX-N3 objective prompt is absent from the prompt set")
        entries = rubric.get("entries")
        if not isinstance(entries, list):
            raise PlanError("EX-N3 rubric entries must be a list")
        rubric_ids = {
            str(entry.get("prompt_id"))
            for entry in entries if isinstance(entry, Mapping)
        }
        if configured != rubric_ids:
            raise PlanError("EX-N3 rubric entries do not match plan objective prompt IDs")


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


def _bounded_rate(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PlanError(f"{label} must be a number between 0 and 1")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanError(f"{label} must be a number between 0 and 1") from exc
    if not 0.0 <= rate <= 1.0:
        raise PlanError(f"{label} must be a number between 0 and 1")
    return rate


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PlanError(f"{label} must be a bounded identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PlanError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_quality_config(raw: Any, prompt_set: Mapping[str, Any]) -> QualitySpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PlanError("quality must be an object")
    allowed = {"required", "llm", "sd", "gemma_judge", "manual_review", "calibration"}
    if set(raw) - allowed:
        raise PlanError("quality contains unsupported fields")
    required = raw.get("required", False)
    if not isinstance(required, bool):
        raise PlanError("quality.required must be boolean")

    llm_raw = raw.get("llm")
    llm: dict[str, Any] | None = None
    if llm_raw is not None:
        if not isinstance(llm_raw, Mapping):
            raise PlanError("quality.llm must be an object")
        llm_allowed = {
            "prompt_set_id", "prompt_set_sha256", "objective_subset_count",
            "objective_prompt_ids", "rubric_id", "rubric_sha256",
            "correctness_rate_baseline", "format_rate_baseline", "compare_rule",
        }
        if set(llm_raw) - llm_allowed:
            raise PlanError("quality.llm contains unsupported fields")
        required_fields = llm_allowed
        if not required_fields.issubset(llm_raw):
            raise PlanError("quality.llm is incomplete")
        if llm_raw["prompt_set_id"] != prompt_set.get("id"):
            raise PlanError("quality.llm.prompt_set_id must match plan prompt_set")
        if llm_raw["prompt_set_sha256"] != prompt_set.get("sha256"):
            raise PlanError("quality.llm.prompt_set_sha256 must match plan prompt_set")
        prompt_ids = llm_raw["objective_prompt_ids"]
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise PlanError("quality.llm.objective_prompt_ids must be a non-empty list")
        normalized_ids = [_identifier(item, "quality.llm objective prompt") for item in prompt_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise PlanError("quality.llm.objective_prompt_ids must be unique")
        if llm_raw["objective_subset_count"] != len(normalized_ids):
            raise PlanError("quality.llm.objective_subset_count must match objective_prompt_ids")
        rubric_sha256 = llm_raw["rubric_sha256"]
        if not isinstance(rubric_sha256, str) or not _SHA256_RE.fullmatch(rubric_sha256):
            raise PlanError("quality.llm.rubric_sha256 must be a lowercase SHA-256")
        compare_rule = llm_raw["compare_rule"]
        if compare_rule != ">= baseline*0.9":
            raise PlanError("quality.llm.compare_rule must be >= baseline*0.9")
        llm = {
            "prompt_set_id": str(llm_raw["prompt_set_id"]),
            "prompt_set_sha256": str(llm_raw["prompt_set_sha256"]),
            "objective_subset_count": int(llm_raw["objective_subset_count"]),
            "objective_prompt_ids": tuple(normalized_ids),
            "rubric_id": _identifier(llm_raw["rubric_id"], "quality.llm.rubric_id"),
            "rubric_sha256": rubric_sha256,
            "correctness_rate_baseline": _bounded_rate(
                llm_raw["correctness_rate_baseline"], "quality.llm.correctness_rate_baseline",
            ),
            "format_rate_baseline": _bounded_rate(
                llm_raw["format_rate_baseline"], "quality.llm.format_rate_baseline",
            ),
            "compare_rule": compare_rule,
        }

    sd_raw = raw.get("sd")
    sd: dict[str, Any] | None = None
    if sd_raw is not None:
        if not isinstance(sd_raw, Mapping):
            raise PlanError("quality.sd must be an object")
        if set(sd_raw) - {"asset_ids", "gate"} or not {"asset_ids", "gate"}.issubset(sd_raw):
            raise PlanError("quality.sd must contain only asset_ids and gate")
        asset_ids = sd_raw["asset_ids"]
        if not isinstance(asset_ids, list) or not asset_ids:
            raise PlanError("quality.sd.asset_ids must be a non-empty list")
        sd = {
            "asset_ids": tuple(_identifier(item, "quality.sd asset") for item in asset_ids),
            "gate": str(sd_raw["gate"]),
        }
        if sd["gate"] != "quality_gate_sd15 automatic_gate.passed":
            raise PlanError("quality.sd.gate is unsupported")

    gemma_raw = raw.get("gemma_judge")
    gemma_judge: dict[str, Any] | None = None
    if gemma_raw is not None:
        if not isinstance(gemma_raw, Mapping):
            raise PlanError("quality.gemma_judge must be an object")
        allowed_gemma = {
            "model", "judge_contract_id", "judge_contract_sha256",
            "topic_hit_rate_baseline", "key_element_coverage_baseline",
        }
        if set(gemma_raw) - allowed_gemma or not allowed_gemma.issubset(gemma_raw):
            raise PlanError("quality.gemma_judge is incomplete")
        gemma_judge = {
            "model": _identifier(gemma_raw["model"], "quality.gemma_judge.model"),
            "judge_contract_id": _identifier(
                gemma_raw["judge_contract_id"],
                "quality.gemma_judge.judge_contract_id",
            ),
            "judge_contract_sha256": _sha256(
                gemma_raw["judge_contract_sha256"],
                "quality.gemma_judge.judge_contract_sha256",
            ),
            "topic_hit_rate_baseline": _bounded_rate(
                gemma_raw["topic_hit_rate_baseline"], "quality.gemma_judge.topic_hit_rate_baseline",
            ),
            "key_element_coverage_baseline": _bounded_rate(
                gemma_raw["key_element_coverage_baseline"],
                "quality.gemma_judge.key_element_coverage_baseline",
            ),
        }

    manual_raw = raw.get("manual_review")
    manual_review: dict[str, Any] | None = None
    if manual_raw is not None:
        if not isinstance(manual_raw, Mapping):
            raise PlanError("quality.manual_review must be an object")
        if set(manual_raw) - {"reviewers_required", "upgrade_on"} or not {
            "reviewers_required", "upgrade_on",
        }.issubset(manual_raw):
            raise PlanError("quality.manual_review is incomplete")
        if manual_raw["reviewers_required"] != 2 or manual_raw["upgrade_on"] != "2 pass, 0 fail":
            raise PlanError("quality.manual_review must use the frozen 2-pass policy")
        manual_review = {"reviewers_required": 2, "upgrade_on": "2 pass, 0 fail"}

    calibration_raw = raw.get("calibration")
    calibration: dict[str, Any] | None = None
    if calibration_raw is not None:
        if not isinstance(calibration_raw, Mapping):
            raise PlanError("quality.calibration must be an object")
        allowed_calibration = {"series_id", "rounds_required", "threshold_version"}
        if set(calibration_raw) - allowed_calibration or not allowed_calibration.issubset(calibration_raw):
            raise PlanError("quality.calibration is incomplete")
        if calibration_raw["rounds_required"] != 3:
            raise PlanError("quality.calibration.rounds_required must be 3")
        calibration = {
            "series_id": _identifier(calibration_raw["series_id"], "quality.calibration.series_id"),
            "rounds_required": 3,
            "threshold_version": _identifier(
                calibration_raw["threshold_version"], "quality.calibration.threshold_version",
            ),
        }

    if required and llm and (
        llm["correctness_rate_baseline"] < 0.60
        or llm["format_rate_baseline"] < 0.90
    ):
        raise PlanError(
            "quality.required cannot use an unapproved weak LLM calibration floor"
        )
    if required and gemma_judge:
        raise PlanError(
            "gemma_judge cannot be quality.required before real calibration is approved"
        )
    if required and not any((llm, sd, gemma_judge)):
        raise PlanError("quality.required needs at least one configured check")
    return QualitySpec(
        required=required, llm=llm, sd=sd, gemma_judge=gemma_judge,
        manual_review=manual_review, calibration=calibration,
    )


def _parse_unit(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    plan_id: str,
    quality: QualitySpec | None,
) -> ExperimentUnit:
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
    checks_raw = raw.get("quality_checks") or []
    if not isinstance(checks_raw, list) or not all(isinstance(item, str) for item in checks_raw):
        raise PlanError(f"{experiment_id}: quality_checks must be a string list")
    quality_checks = tuple(checks_raw)
    if len(set(quality_checks)) != len(quality_checks) or set(quality_checks) - _QUALITY_CHECKS:
        raise PlanError(f"{experiment_id}: quality_checks contains unsupported entries")
    if quality_checks and quality is None:
        raise PlanError(f"{experiment_id}: quality_checks needs a plan quality contract")
    configured = {
        name for name, value in (
            ("llm", quality.llm if quality else None),
            ("sd", quality.sd if quality else None),
            ("gemma_judge", quality.gemma_judge if quality else None),
        ) if value is not None
    }
    if set(quality_checks) - configured:
        raise PlanError(f"{experiment_id}: quality_checks references an unconfigured check")
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
        quality_checks=quality_checks,
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
    quality = _parse_quality_config(raw.get("quality"), prompt_set)
    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        raise PlanError("plan 必须包含非空 units 列表")
    seen: set[str] = set()
    units: list[ExperimentUnit] = []
    for item in units_raw:
        if not isinstance(item, Mapping):
            raise PlanError("unit 必须是对象")
        unit = _parse_unit(item, defaults, plan_id, quality)
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
        quality=quality,
        source_path=source,
    )
