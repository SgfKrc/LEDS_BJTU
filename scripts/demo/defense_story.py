"""Validate the five-minute DEF-A1 defense storyline without loading a model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("storyline.json")
DEFAULT_REPORT = ROOT / "build" / "defense-story" / "latest.json"
EXPECTED_TICKETS = {"DEF-P1", "DEF-P2", "DEF-P3", "DEF-P4"}
CLAIM_GUARD = {
    "uses_fixture": True,
    "real_model_demonstrated": False,
    "physical_dual_host_demonstrated": False,
    "model_performance_claimed": False,
    "external_network_required": False,
}
FORBIDDEN_COMMANDS = (
    re.compile(r"(?:^|\s)--mode\s+live(?:\s|$)", re.IGNORECASE),
    re.compile(r"(?:^|\s|[\\/])frontend(?:[\\/]|\s|$)", re.IGNORECASE),
    re.compile(r"(?:model[_-]?(?:smoke|load|switch)|download[_-]?model)", re.IGNORECASE),
)


class StorylineError(ValueError):
    """A stable, user-facing storyline contract failure."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorylineError(f"{label} 不能为空")
    return value.strip()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StorylineError(f"{label} 必须是整数")
    return value


def _clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _repo_path(root: Path, value: object, label: str) -> tuple[str, Path]:
    relative = _text(value, label).replace("\\", "/")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise StorylineError(f"{label} 必须位于仓库内")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise StorylineError(f"{label} 越出仓库") from exc
    return relative, resolved


def validate_storyline(payload: object, *, root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != "qlh.defense_storyline.v1":
        raise StorylineError("故事线 schema 无效")
    if payload.get("ticket") != "DEF-A1":
        raise StorylineError("故事线票号无效")
    if payload.get("claim_guard") != CLAIM_GUARD:
        raise StorylineError("故事线声明边界无效")

    limit = _integer(payload.get("duration_limit_seconds"), "总时长")
    if limit != 300:
        raise StorylineError("答辩故事线必须恰好为 300 秒")

    document_relative, document_path = _repo_path(root, payload.get("document"), "讲稿路径")
    if not document_path.is_file():
        raise StorylineError("讲稿文件不存在")
    document = document_path.read_text(encoding="utf-8")

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise StorylineError("故事线缺少证据索引")
    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_by_path: dict[str, dict[str, Any]] = {}
    required_count = 0
    optional_available = 0
    optional_missing: list[str] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise StorylineError(f"证据 {index + 1} 无效")
        evidence_id = _text(item.get("id"), f"证据 {index + 1} ID")
        if evidence_id in evidence_by_id:
            raise StorylineError(f"证据 ID 重复：{evidence_id}")
        relative, path = _repo_path(root, item.get("path"), f"证据 {evidence_id} 路径")
        if relative in evidence_by_path:
            raise StorylineError(f"证据路径重复：{relative}")
        required = item.get("required")
        if not isinstance(required, bool):
            raise StorylineError(f"证据 {evidence_id} required 无效")
        if item.get("kind") not in {"source", "test", "document", "runtime"}:
            raise StorylineError(f"证据 {evidence_id} 类型无效")
        _text(item.get("purpose"), f"证据 {evidence_id} 用途")
        if required and not path.is_file():
            raise StorylineError(f"必需证据不存在：{relative}")
        if required:
            required_count += 1
        elif path.is_file():
            optional_available += 1
        else:
            optional_missing.append(relative)
        evidence_by_id[evidence_id] = item
        evidence_by_path[relative] = item

    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document):
        if link.startswith(("#", "http://", "https://")):
            continue
        link_path = (document_path.parent / link.split("#", 1)[0]).resolve()
        try:
            link_relative = link_path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise StorylineError(f"讲稿链接越出仓库：{link}") from exc
        if link_relative not in evidence_by_path:
            raise StorylineError(f"讲稿链接未登记为证据：{link_relative}")

    commands = payload.get("rehearsal_commands")
    if not isinstance(commands, dict) or set(commands) != {"windows", "unix"}:
        raise StorylineError("彩排命令必须同时覆盖 Windows 和 Unix")
    for platform, items in commands.items():
        if not isinstance(items, list) or len(items) != 5:
            raise StorylineError(f"{platform} 彩排命令数量无效")
        combined = "\n".join(_text(item, f"{platform} 彩排命令") for item in items)
        if any(pattern.search(combined) for pattern in FORBIDDEN_COMMANDS):
            raise StorylineError(f"{platform} 彩排命令触发真实模型、live 或旧前端禁用项")
        if "checklist" not in combined or "--mode fixtures" not in combined or "--mode failure" not in combined or "benchmark" not in combined or "performance_report" not in combined:
            raise StorylineError(f"{platform} 彩排命令未覆盖 A3 预检、P1-P3 与 A2 汇总")

    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise StorylineError("故事线分段为空")
    cursor = 0
    segment_ids: set[str] = set()
    covered_tickets: set[str] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise StorylineError(f"分段 {index + 1} 无效")
        segment_id = _text(segment.get("id"), f"分段 {index + 1} ID")
        if segment_id in segment_ids:
            raise StorylineError(f"分段 ID 重复：{segment_id}")
        start = _integer(segment.get("start_second"), f"{segment_id} 开始时间")
        end = _integer(segment.get("end_second"), f"{segment_id} 结束时间")
        if start != cursor or end <= start or end > limit:
            raise StorylineError(f"{segment_id} 时间轴存在空洞、重叠或越界")
        window = f"{_clock(start)}–{_clock(end)}"
        if segment_id not in document or window not in document:
            raise StorylineError(f"讲稿缺少 {segment_id} 或时间窗 {window}")
        _text(segment.get("title"), f"{segment_id} 标题")
        _text(segment.get("screen"), f"{segment_id} 屏幕")
        _text(segment.get("action"), f"{segment_id} 动作")
        for key, label in (("say", "讲词"), ("success_markers", "成功标记"), ("evidence_ids", "证据")):
            values = segment.get(key)
            if not isinstance(values, list) or not values or any(not isinstance(item, str) or not item for item in values):
                raise StorylineError(f"{segment_id} {label}无效")
        unknown_evidence = set(segment["evidence_ids"]) - evidence_by_id.keys()
        if unknown_evidence:
            raise StorylineError(f"{segment_id} 引用了未知证据：{','.join(sorted(unknown_evidence))}")
        ticket_refs = segment.get("ticket_refs")
        if not isinstance(ticket_refs, list) or not ticket_refs or not set(ticket_refs) <= EXPECTED_TICKETS:
            raise StorylineError(f"{segment_id} 票号引用无效")
        covered_tickets.update(ticket_refs)
        fallback = segment.get("fallback")
        if not isinstance(fallback, dict):
            raise StorylineError(f"{segment_id} 缺少翻车兜底")
        _text(fallback.get("trigger"), f"{segment_id} 兜底触发条件")
        _text(fallback.get("action"), f"{segment_id} 兜底动作")
        fallback_evidence = _text(fallback.get("evidence_id"), f"{segment_id} 兜底证据")
        if fallback_evidence not in evidence_by_id:
            raise StorylineError(f"{segment_id} 兜底证据未知")
        segment_ids.add(segment_id)
        cursor = end

    if cursor != limit:
        raise StorylineError("故事线未覆盖完整 300 秒")
    if covered_tickets != EXPECTED_TICKETS:
        raise StorylineError("故事线未完整覆盖 DEF-P1 至 DEF-P4")

    return {
        "document": document_relative,
        "duration_seconds": limit,
        "segment_count": len(segments),
        "covered_tickets": sorted(covered_tickets),
        "required_evidence_count": required_count,
        "optional_evidence_available": optional_available,
        "optional_evidence_missing": optional_missing,
        "claim_guard": CLAIM_GUARD,
    }


def validate_file(manifest_path: Path = DEFAULT_MANIFEST, *, root: Path = ROOT) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StorylineError(f"故事线清单不可读取：{type(exc).__name__}") from exc
    return validate_storyline(payload, root=root)


def write_report(summary: dict[str, Any], report_path: Path = DEFAULT_REPORT) -> None:
    payload = {
        "schema": "qlh.defense_storyline.validation.v1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the model-free DEF-A1 defense storyline")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = validate_file(args.manifest)
        write_report(summary, args.report)
    except StorylineError as exc:
        print(f"[QLH-STORY] ERROR {exc}", file=sys.stderr)
        return 2
    print(
        "[QLH-STORY] OK "
        f"duration={summary['duration_seconds']}s "
        f"segments={summary['segment_count']} "
        f"evidence={summary['required_evidence_count']}+{summary['optional_evidence_available']}"
    )
    print(f"[QLH-STORY] DOCUMENT {summary['document']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
