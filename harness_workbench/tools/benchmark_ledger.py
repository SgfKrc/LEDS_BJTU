"""Read-only benchmark ledger aggregation for ``TOOL-BENCH-LDG-01``.

The ledger normalizes structured experiment JSON into a citation-oriented
table.  It never parses free-form logs, invents missing metrics, or upgrades a
fixture/control-plane measurement into a model or physical dual-host claim.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


BENCHMARK_LEDGER_SCHEMA = "qlh.harness.benchmark_ledger.v1"
BENCHMARK_LEDGER_INPUT_SCHEMA = "qlh.benchmark_ledger.v1"
SUPPORTED_SOURCE_SCHEMAS = frozenset(
    {
        "qlh.defense_benchmark.v1",
        "qlh.real_model_performance.v1",
        "qlh.experiment_record.v1",
        BENCHMARK_LEDGER_INPUT_SCHEMA,
    }
)
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^[A-Za-z]:|^/|^\\\\)")
_IP_TEXT = re.compile(r"(?<![0-9A-Fa-f:.])(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}|[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){2,})(?![0-9A-Fa-f:.])")
_SECRET_KEYS = frozenset({"token", "secret", "password", "authorization", "api_key", "private_key"})
_VOLATILE_KEYS = frozenset({"created_at", "generated_at", "timestamp", "started_at", "ended_at"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _contains_unsafe(value: Any, *, key: str = "") -> bool:
    if key.lower() in _SECRET_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_unsafe(child, key=str(name)) for name, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsafe(child, key=key) for child in value)
    if isinstance(value, str):
        if _ABSOLUTE_PATH.search(value):
            return True
        for match in _IP_TEXT.findall(value):
            try:
                ipaddress.ip_address(match)
            except ValueError:
                continue
            return True
    return False


def _stable_source_digest(payload: Mapping[str, Any]) -> str:
    sanitized = {key: value for key, value in payload.items() if key not in _VOLATILE_KEYS}
    return _digest(sanitized)


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _numeric_metrics(value: Any) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float | int] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and not math.isfinite(float(raw)):
            raise ValueError(f"benchmark metric {key} must be finite")
        number = _finite_number(raw)
        if number is not None:
            result[str(key)] = number
    return result


def _model_id(payload: Mapping[str, Any], default: str = "unspecified") -> str:
    for key in ("model_id", "model", "model_family", "engine"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _claim_class(*, topology: str, host_count: int, model_id: str, status: str) -> str:
    if status in {"not_run", "invalid"}:
        return "not_run" if status == "not_run" else "invalid"
    if host_count >= 2 or "dual_host" in topology or "multi_host" in topology:
        return "dual_host"
    if topology.startswith("single_host"):
        return "single_host"
    return "other"


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    record_id: str
    source_ref: str
    source_digest: str
    model_id: str
    topology: str
    host_count: int
    process_count: int
    sample_count: int
    status: str
    claim_class: str
    metrics: Mapping[str, float | int]
    eligible_for_claim: bool
    claim_scope: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.record_id or not self.source_ref or not self.source_digest:
            raise ValueError("benchmark record identity is required")
        if _ABSOLUTE_PATH.search(self.source_ref) or "://" in self.source_ref:
            raise ValueError("benchmark source_ref must be repository-relative")
        if self.host_count < 0 or self.process_count < 0 or self.sample_count < 0:
            raise ValueError("benchmark counts must be non-negative")
        if self.status not in {"passed", "failed", "not_run", "invalid"}:
            raise ValueError("unsupported benchmark status")
        if self.claim_class not in {"single_host", "dual_host", "other", "not_run", "invalid"}:
            raise ValueError("unsupported benchmark claim class")
        if _contains_unsafe(self.as_dict()):
            raise ValueError("benchmark record contains a path, address or credential")

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
            "model_id": self.model_id,
            "topology": self.topology,
            "host_count": self.host_count,
            "process_count": self.process_count,
            "sample_count": self.sample_count,
            "status": self.status,
            "claim_class": self.claim_class,
            "metrics": dict(self.metrics),
            "eligible_for_claim": self.eligible_for_claim,
            "claim_scope": self.claim_scope,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LedgerGroup:
    group_id: str
    dimension: str
    record_ids: tuple[str, ...]
    record_count: int
    model_ids: tuple[str, ...]
    topologies: tuple[str, ...]
    metric_summary: Mapping[str, Mapping[str, float | int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "dimension": self.dimension,
            "record_ids": list(self.record_ids),
            "record_count": self.record_count,
            "model_ids": list(self.model_ids),
            "topologies": list(self.topologies),
            "metric_summary": {key: dict(value) for key, value in self.metric_summary.items()},
        }


def _summarize_metrics(records: Sequence[BenchmarkRecord]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.metrics.items():
            values.setdefault(key, []).append(float(value))
    return {
        key: {
            "count": len(items),
            "min": min(items),
            "median": median(items),
            "max": max(items),
        }
        for key, items in sorted(values.items())
    }


def _group_records(records: Sequence[BenchmarkRecord]) -> tuple[LedgerGroup, ...]:
    buckets: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        buckets.setdefault(record.claim_class, []).append(record)
    model_ids = {record.model_id for record in records if record.model_id != "unspecified"}
    if len(model_ids) > 1:
        buckets["multi_model"] = list(records)
    groups: list[LedgerGroup] = []
    for dimension, values in sorted(buckets.items()):
        groups.append(
            LedgerGroup(
                group_id=f"{dimension}-group",
                dimension=dimension,
                record_ids=tuple(record.record_id for record in values),
                record_count=len(values),
                model_ids=tuple(sorted({record.model_id for record in values})),
                topologies=tuple(sorted({record.topology for record in values})),
                metric_summary=_summarize_metrics(values),
            )
        )
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class BenchmarkLedgerReport:
    records: tuple[BenchmarkRecord, ...]
    groups: tuple[LedgerGroup, ...]
    ignored_sources: tuple[str, ...] = ()
    schema: str = BENCHMARK_LEDGER_SCHEMA
    runner_kind: str = "input"
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != BENCHMARK_LEDGER_SCHEMA or self.runner_kind not in {"input", "fixture"}:
            raise ValueError("benchmark ledger identity is invalid")
        if not self.records:
            raise ValueError("benchmark ledger requires at least one record")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("benchmark record ids must be unique")
        if self.network_used or self.weights_loaded:
            raise ValueError("benchmark ledger must remain offline and model-free")

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.model_id for record in self.records if record.model_id != "unspecified"}))

    @property
    def valid(self) -> bool:
        return bool(self.records) and all(
            record.status in {"passed", "failed", "not_run", "invalid"}
            and not _contains_unsafe(record.as_dict())
            for record in self.records
        )

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "records_unique": len({record.record_id for record in self.records}) == len(self.records),
            "source_refs_relative": all(not _ABSOLUTE_PATH.search(record.source_ref) and "://" not in record.source_ref for record in self.records),
            "metrics_finite": all(all(_finite_number(value) is not None for value in record.metrics.values()) for record in self.records),
            "claim_scopes_explicit": all(bool(record.claim_scope) for record in self.records),
            "multi_model_aggregated": len(self.model_ids) <= 1 or any(group.dimension == "multi_model" for group in self.groups),
            "offline_boundary": not self.network_used and not self.weights_loaded,
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "record_count": self.record_count,
            "model_ids": list(self.model_ids),
            "records": [record.as_dict() for record in self.records],
            "groups": [group.as_dict() for group in self.groups],
            "ignored_sources": list(self.ignored_sources),
            "checks": self.checks,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-BENCH-LDG-01 benchmark ledger",
            "",
            f"- Valid: `{str(self.valid).lower()}`; records: `{self.record_count}`; models: `{len(self.model_ids)}`; runner: `{self.runner_kind}`",
            "- Metrics are copied only from structured JSON; missing values remain absent. Fixture/control-plane values are never labeled model performance.",
            "",
            "## Citation table",
            "",
            "| record | source | model | claim class | topology | hosts | samples | status | metrics | eligible claim |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        for record in self.records:
            metrics = ", ".join(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}" for key, value in sorted(record.metrics.items())) or "NOT RUN"
            lines.append(
                f"| `{record.record_id}` | `{record.source_ref}` | `{record.model_id}` | `{record.claim_class}` | `{record.topology}` | {record.host_count} | {record.sample_count} | `{record.status}` | {metrics} | `{str(record.eligible_for_claim).lower()}` |"
            )
        lines.extend(("", "## Aggregates", "", "| dimension | records | models | topologies | metrics |", "| --- | ---: | --- | --- | --- |"))
        for group in self.groups:
            metrics = ", ".join(f"{key}: median={summary['median']:g}" for key, summary in group.metric_summary.items()) or "NOT RUN"
            lines.append(f"| `{group.dimension}` | {group.record_count} | {', '.join(group.model_ids) or 'unspecified'} | {', '.join(group.topologies)} | {metrics} |")
        lines.extend(("", "## Checks", ""))
        for name, passed in self.checks.items():
            lines.append(f"- `{name}`: **{'passed' if passed else 'failed'}**")
        if self.ignored_sources:
            lines.extend(("", "Ignored unsupported sources: " + ", ".join(f"`{item}`" for item in self.ignored_sources)))
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def _record_from_series(series: Mapping[str, Any], payload: Mapping[str, Any], source_ref: str, source_digest: str) -> BenchmarkRecord:
    topology = str(series.get("topology", "unknown"))
    host_count = int(series.get("host_count", 0) or 0)
    process_count = int(series.get("process_count", 0) or 0)
    status = str(payload.get("status", series.get("status", "invalid")))
    guard = payload.get("claim_guard") if isinstance(payload.get("claim_guard"), Mapping) else {}
    physical = payload.get("physical_dual_host") if isinstance(payload.get("physical_dual_host"), Mapping) else {}
    eligible = bool(guard.get("real_model_performance", False) or guard.get("physical_dual_host_performance", False)) and status == "passed"
    if topology == "single_host_single_process" or topology == "single_host_dual_process":
        eligible = False
    scope = str(guard.get("allowed_claim", "structured benchmark record only"))
    note = "physical dual-host data not run" if physical.get("status") == "not_run" else ""
    return BenchmarkRecord(
        str(series.get("series_id", "series")),
        source_ref,
        source_digest,
        _model_id(payload, "unspecified"),
        topology,
        host_count,
        process_count,
        int(series.get("sample_count", 0) or 0),
        status,
        _claim_class(topology=topology, host_count=host_count, model_id=_model_id(payload), status=status),
        _numeric_metrics(series.get("metrics")),
        eligible,
        scope,
        note,
    )


def _records_from_payload(payload: Mapping[str, Any], source_ref: str, source_digest: str) -> tuple[BenchmarkRecord, ...]:
    schema = payload.get("schema")
    if schema == "qlh.defense_benchmark.v1":
        series = payload.get("series")
        if not isinstance(series, list) or not series:
            raise ValueError("defense benchmark source requires series")
        records = [
            _record_from_series(item, payload, source_ref, source_digest)
            for item in series
            if isinstance(item, Mapping)
        ]
        physical = payload.get("physical_dual_host")
        if isinstance(physical, Mapping) and str(physical.get("status", "")) == "not_run":
            records.append(
                BenchmarkRecord(
                    "physical-dual-host",
                    source_ref,
                    source_digest,
                    _model_id(payload, "unspecified"),
                    "physical_dual_host",
                    int(physical.get("host_count", 2) or 2),
                    int(physical.get("process_count", 0) or 0),
                    int(physical.get("sample_count", 0) or 0),
                    "not_run",
                    "not_run",
                    {},
                    False,
                    "physical dual-host performance unavailable",
                    str(physical.get("reason_code", "physical_dual_host_data_pending")),
                )
            )
        return tuple(records)
    if schema == "qlh.real_model_performance.v1":
        metrics = _numeric_metrics(payload.get("metrics"))
        status = str(payload.get("status", "invalid"))
        guard = payload.get("claim_guard") if isinstance(payload.get("claim_guard"), Mapping) else {}
        return (
            BenchmarkRecord(
                "real-model-performance",
                source_ref,
                source_digest,
                _model_id(payload, "real-model"),
                str(payload.get("topology", "unknown")),
                int(payload.get("host_count", 0) or 0),
                int(payload.get("process_count", 0) or 0),
                int(payload.get("sample_count", 0) or 0),
                status,
                _claim_class(topology=str(payload.get("topology", "unknown")), host_count=int(payload.get("host_count", 0) or 0), model_id=_model_id(payload), status=status),
                metrics,
                bool(guard.get("eligible_for_model_claim", False)) and status == "passed",
                "real model measurement only when full identity and environment gate passes",
                str(payload.get("reason_code", "")),
            ),
        )
    if isinstance(payload.get("records"), list):
        records: list[BenchmarkRecord] = []
        for index, raw in enumerate(payload["records"]):
            if not isinstance(raw, Mapping):
                raise ValueError(f"benchmark records[{index}] must be an object")
            record_payload = dict(raw)
            record_payload.setdefault("schema", "qlh.experiment_record.v1")
            records.extend(_records_from_payload(record_payload, source_ref, source_digest))
        return tuple(records)
    if schema in {"qlh.experiment_record.v1", BENCHMARK_LEDGER_INPUT_SCHEMA} or "record_id" in payload:
        metrics = _numeric_metrics(payload.get("metrics"))
        topology = str(payload.get("topology", "unknown"))
        host_count = int(payload.get("host_count", 0) or 0)
        status = str(payload.get("status", "invalid"))
        return (
            BenchmarkRecord(
                str(payload.get("record_id", payload.get("experiment_id", "record"))),
                source_ref,
                source_digest,
                _model_id(payload),
                topology,
                host_count,
                int(payload.get("process_count", 0) or 0),
                int(payload.get("sample_count", payload.get("runs", 0)) or 0),
                status,
                _claim_class(topology=topology, host_count=host_count, model_id=_model_id(payload), status=status),
                metrics,
                bool(payload.get("eligible_for_claim", False)) and status == "passed",
                str(payload.get("claim_scope", "structured benchmark record only")),
                str(payload.get("note", "")),
            ),
        )
    raise ValueError(f"unsupported benchmark source schema: {schema}")


def load_benchmark_payload(path: str | Path, *, source_ref: str | None = None) -> tuple[BenchmarkRecord, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark source must be a JSON object")
    if _contains_unsafe(payload):
        raise ValueError("benchmark source contains a path, address or credential")
    ref = source_ref or source.as_posix()
    if _ABSOLUTE_PATH.search(ref) or "://" in ref:
        ref = source.name
    return _records_from_payload(payload, ref.replace("\\", "/"), _stable_source_digest(payload))


def _default_sources() -> tuple[Path, ...]:
    candidates = (
        Path("build/defense-benchmark/latest.json"),
        Path("scripts/demo/real-model-performance-not-run.json"),
    )
    return tuple(path for path in candidates if path.is_file())


def run_benchmark_ledger(
    paths: Sequence[str | Path] | None = None,
    *,
    root: str | Path | None = None,
    strict: bool = True,
) -> BenchmarkLedgerReport:
    if paths and root is not None:
        raise ValueError("provide input paths or root, not both")
    if root is not None:
        root_path = Path(root)
        source_paths = tuple(sorted(root_path.rglob("*.json")))
        strict = False
    else:
        source_paths = tuple(Path(path) for path in paths) if paths else _default_sources()
    if not source_paths:
        raise ValueError("benchmark ledger requires at least one JSON source")
    records: list[BenchmarkRecord] = []
    ignored: list[str] = []
    for path in source_paths:
        try:
            records.extend(load_benchmark_payload(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if strict:
                raise ValueError(f"{path}: {exc}") from exc
            ignored.append(path.name)
    if not records:
        raise ValueError("no supported benchmark records found")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("benchmark record ids collide across input sources")
    return BenchmarkLedgerReport(tuple(records), _group_records(records), tuple(sorted(set(ignored))), runner_kind="input")


def build_benchmark_ledger_report(*args: Any, **kwargs: Any) -> BenchmarkLedgerReport:
    return run_benchmark_ledger(*args, **kwargs)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate structured benchmark JSON into a citation ledger")
    parser.add_argument("--input", dest="inputs", action="append", metavar="PATH", help="JSON source; repeat for multiple files")
    parser.add_argument("--root", metavar="PATH", help="recursively scan JSON sources, ignoring unsupported files")
    parser.add_argument("--strict", action="store_true", help="fail on any unsupported/invalid explicit source")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark_ledger(args.inputs, root=args.root, strict=args.strict or not args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    outputs = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        outputs += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        outputs += 1
    if not outputs:
        print(markdown_text, end="")
    return 0 if report.valid and all(report.checks.values()) else 1


__all__ = [
    "BENCHMARK_LEDGER_INPUT_SCHEMA",
    "BENCHMARK_LEDGER_SCHEMA",
    "BenchmarkLedgerReport",
    "BenchmarkRecord",
    "LedgerGroup",
    "build_benchmark_ledger_report",
    "build_parser",
    "load_benchmark_payload",
    "main",
    "run_benchmark_ledger",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
