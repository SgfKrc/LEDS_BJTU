#!/usr/bin/env python3
"""文档维护 Agent M2.4：只生成建议 diff，绝不写回源文档。"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from doc_maintenance_llm import sanitize_text

STATUS_RE = re.compile(r"^(?:>\s*)?(?:\*\*)?(?:文档)?状态(?:\*\*)?[:：]")
LIFECYCLE_RE = re.compile(r"文档生命周期")


def _status_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines[:12]):
        if STATUS_RE.search(line.strip()) and not LIFECYCLE_RE.search(line):
            return index
    return None


def _suggested_status(value: str) -> str | None:
    safe = sanitize_text(value, 1024)
    candidates = [line.strip() for line in safe.splitlines() if line.strip()]
    if len(candidates) != 1 or not STATUS_RE.search(candidates[0]):
        return None
    return candidates[0]


def generate_suggestion_diff(llm_report: dict, repo_root: Path, out_dir: Path,
                             *, confidence_floor: float = 0.6) -> dict:
    """产出 suggestions.patch/json；不执行 git apply，也不修改 docs/。"""
    docs_root = (repo_root / "docs").resolve()
    patch_chunks: list[str] = []
    generated: list[str] = []
    skipped: list[dict] = []
    for item in llm_report.get("judgements", ()):
        doc = str(item.get("doc", ""))
        if item.get("judgement") != "stale":
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            skipped.append({"doc": doc, "reason": "invalid_confidence"})
            continue
        if float(confidence) < confidence_floor:
            skipped.append({"doc": doc, "reason": "below_confidence_floor"})
            continue
        suggestion = _suggested_status(str(item.get("suggestion", "")))
        if suggestion is None:
            skipped.append({"doc": doc, "reason": "suggestion_is_not_a_single_status_line"})
            continue
        path = (repo_root / doc).resolve()
        if path != docs_root and docs_root not in path.parents:
            raise ValueError(f"suggestion document escapes docs root: {doc}")
        if not path.is_file():
            skipped.append({"doc": doc, "reason": "document_missing"})
            continue
        original_text = path.read_text(encoding="utf-8", errors="replace")
        original_lines = original_text.splitlines()
        updated_lines = list(original_lines)
        index = _status_index(original_lines)
        if index is None:
            insert_at = 1 if original_lines and original_lines[0].startswith("#") else 0
            insertion = [suggestion, ""]
            if insert_at and len(original_lines) > 1 and original_lines[1].strip():
                insertion.insert(0, "")
            updated_lines[insert_at:insert_at] = insertion
        else:
            if original_lines[index].strip() == suggestion:
                skipped.append({"doc": doc, "reason": "no_change"})
                continue
            updated_lines[index] = suggestion
        diff = "\n".join(difflib.unified_diff(
            original_lines, updated_lines,
            fromfile=f"a/{doc}", tofile=f"b/{doc}", lineterm="",
        ))
        if diff:
            patch_chunks.append(diff)
            generated.append(doc)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / "suggestions.patch"
    manifest_path = out_dir / "suggestions.json"
    patch_path.write_text("\n".join(patch_chunks) + ("\n" if patch_chunks else ""), encoding="utf-8")
    manifest = {
        "generated": generated,
        "skipped": skipped,
        "patch": patch_path.relative_to(repo_root).as_posix()
        if patch_path.is_relative_to(repo_root) else patch_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
