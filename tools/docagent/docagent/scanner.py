"""Read-only, root-relative R1-R5 documentation scanner."""

from __future__ import annotations

import hashlib
import fnmatch
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote

from .gate import report_fingerprint
from .profile import apply_profile, load_profile, profile_metadata, validate_profile
from .rules import load_rules, rules_fingerprint, validate_rules


GIT_LOG_MARKER = "@@DOCAGENT_COMMIT@@"


class ScanError(ValueError):
    """Raised when a repository cannot be scanned safely."""


def _rule(rules: Mapping[str, Any], rule_id: str) -> Mapping[str, Any]:
    return next(rule for rule in rules["rules"] if rule["id"] == rule_id)


def _params(rules: Mapping[str, Any], rule_id: str) -> Mapping[str, Any]:
    return _rule(rules, rule_id)["parameters"]


def _enabled(rules: Mapping[str, Any], rule_id: str) -> bool:
    return bool(_rule(rules, rule_id)["enabled"])


def _level(rules: Mapping[str, Any], rule_id: str) -> str:
    return str(_rule(rules, rule_id)["level"])


def _marker_pattern(markers: object) -> re.Pattern[str]:
    values = [re.escape(str(marker)) for marker in markers if str(marker)]
    return re.compile("|".join(values) if values else r"(?!)", re.IGNORECASE)


def _git(args: list[str], root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _dirty_doc_paths(root: Path, git_scope: str) -> dict[str, str]:
    scope = git_scope.rstrip("/") or "."
    output = _git([
        "-c", "core.quotepath=false", "status", "--short",
        "--untracked-files=all", "--", scope,
    ], root)
    dirty: dict[str, str] = {}
    for line in output.splitlines():
        if len(line) < 4:
            continue
        state = line[:2]
        for path in line[3:].split(" -> "):
            normalized = path.strip().strip('"').replace("\\", "/")
            if normalized.startswith(f"{scope}/") and normalized.endswith(".md"):
                dirty[normalized] = state
    return dirty


def _source_changes(root: Path, since_date: str, source_root: str) -> list[dict[str, Any]]:
    scope = source_root.rstrip("/") or "."
    output = _git([
        "log", f"--since={since_date}T00:00:00", "--date=short",
        f"--format={GIT_LOG_MARKER}%H%x09%cs%x09%s", "--name-only", "--", scope,
    ], root)
    changes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        if line.startswith(GIT_LOG_MARKER):
            fields = line[len(GIT_LOG_MARKER):].split("\t", 2)
            if len(fields) != 3:
                current = None
                continue
            current = {"commit": fields[0], "date": fields[1], "subject": fields[2], "paths": []}
            changes.append(current)
        elif current is not None and line.strip():
            current["paths"].append(line.strip().replace("\\", "/"))
    return changes


def _doc_topics(doc: Path, text: str, source_root: str,
                stop_words: set[str]) -> tuple[set[str], set[str]]:
    heading = next(
        (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
        "",
    )
    source_prefix = source_root.rstrip("/") or "src"
    source_refs = {
        match.rstrip(".,;:)]}\"").lower()
        for match in re.findall(rf"{re.escape(source_prefix)}/[A-Za-z0-9_./-]+", text)
    }
    token_text = " ".join((doc.stem, heading, *source_refs)).lower()
    tokens = {
        token for token in re.findall(r"[a-z][a-z0-9-]{3,}", token_text.replace("_", " "))
        if token not in stop_words
    }
    return tokens, source_refs


def _related_source_change(doc: Path, text: str, updated: str,
                           changes: list[dict[str, Any]], source_root: str,
                           stop_words: set[str]) -> tuple[dict[str, Any], str] | None:
    tokens, source_refs = _doc_topics(doc, text, source_root, stop_words)
    if not tokens and not source_refs:
        return None
    for change in changes:
        if change["date"] <= updated:
            continue
        paths = [path.lower() for path in change["paths"]]
        for ref in source_refs:
            ref_prefix = ref.rstrip("/")
            matched = next((path for path in paths if path.startswith(ref_prefix)), None)
            if matched:
                return change, matched
        haystack = " ".join(paths)
        matched_token = next((token for token in sorted(tokens) if token in haystack), None)
        if matched_token:
            matched_path = next((path for path in paths if matched_token in path), source_root)
            return change, matched_path
    return None


def _status_line(text: str, rules: Mapping[str, Any]) -> str:
    params = _params(rules, "R5")
    pattern = re.compile(str(params["status_pattern"]))
    lifecycle = re.compile(str(params["lifecycle_pattern"]))
    window = int(rules["defaults"]["status_window_lines"])
    for line in text.splitlines()[:window]:
        if pattern.search(line) and (
            not bool(params["ignore_lifecycle_heading"]) or not lifecycle.search(line)
        ):
            return line.strip()
    return ""


def _updated_at(text: str) -> str | None:
    for line in text.splitlines()[:12]:
        match = re.search(r"更新(?:日期|时间)?[:：]\s*(\d{4}-\d{2}-\d{2})", line)
        if match:
            return match.group(1)
    return None


def _extract_links(text: str, ignored_schemes: tuple[str, ...]) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        href = match.group(2).strip()
        if href.startswith(ignored_schemes):
            continue
        href = href.split("#", 1)[0]
        if href:
            links.append((match.group(1), href))
    return links


def _check_link(href: str, base: Path, decode_url_path: bool) -> bool:
    href = unquote(href) if decode_url_path else href
    if href.startswith("../"):
        target = (base.parent / href[3:]).resolve()
    else:
        target = (base / href).resolve()
    return target.is_file()


def _excluded(relative: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatchcase(relative, pattern):
            return True
        if pattern.endswith("/**"):
            prefix = pattern[:-3].rstrip("/")
            if relative == prefix or relative.startswith(f"{prefix}/"):
                return True
        if relative == pattern or relative.startswith(f"{pattern}/"):
            return True
    return False


def scan_document(doc: Path, root: Path, rules: Mapping[str, Any], *,
                  dirty_docs: Mapping[str, str] | None = None,
                  source_changes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = doc.read_text(encoding="utf-8", errors="replace")
    rel = doc.relative_to(root).as_posix()
    findings: list[dict[str, str]] = []
    status = _status_line(text, rules)
    updated = _updated_at(text)
    status_lower = status.lower()
    exemptions = tuple(str(item) for item in rules["exemptions"]["status_contains"])
    exempt = any(item.lower() in status_lower for item in exemptions)

    if _enabled(rules, "R5") and not status and not exempt:
        findings.append({
            "rule": "R5",
            "level": _level(rules, "R5"),
            "message": f"前 {int(rules['defaults']['status_window_lines'])} 行无状态行",
        })
    elif status and _enabled(rules, "R1"):
        r1 = _params(rules, "R1")
        status_stem = (
            re.sub(r"[（(][^（）()]*[）)]", "", status)
            if bool(r1["status_stem_parentheses"]) else status
        )
        done_marks = _marker_pattern(r1["done_markers"])
        if (
            not exempt
            and (not bool(r1["ignore_status_done_markers"]) or not done_marks.search(status))
            and any(str(hint).lower() in status_stem.lower() for hint in r1["stale_status_hints"])
            and done_marks.search("\n".join(text.splitlines()[:int(rules["defaults"]["body_window_lines"])]))
        ):
            findings.append({
                "rule": "R1",
                "level": _level(rules, "R1"),
                "message": f"状态行含未收口词但正文含完成标记：{status[:60]}",
            })

    if updated and _enabled(rules, "R3"):
        r3 = _params(rules, "R3")
        changes = source_changes
        if changes is None:
            changes = _source_changes(root, updated, str(r3["source_root"]))
        related = _related_source_change(
            doc, text, updated, changes, str(r3["source_root"]),
            {str(item) for item in r3["topic_stop_words"]},
        )
        if related:
            change, matched_path = related
            findings.append({
                "rule": "R3",
                "level": _level(rules, "R3"),
                "message": (
                    f"更新日期 {updated} 早于关联代码提交 "
                    f"{change['commit'][:12]} ({change['date']}, {matched_path})"
                ),
            })

    if _enabled(rules, "R4"):
        r4 = _params(rules, "R4")
        ignored = tuple(str(item) for item in r4["ignored_schemes"])
        for label, href in _extract_links(text, ignored):
            if not _check_link(href, doc.parent, bool(r4["decode_url_path"])):
                findings.append({
                    "rule": "R4",
                    "level": _level(rules, "R4"),
                    "message": f"失效链接 [{label}]({href})",
                })

    if _enabled(rules, "R2"):
        dirty = dirty_docs if dirty_docs is not None else _dirty_doc_paths(
            root, str(_params(rules, "R2")["git_scope"])
        )
        if rel in dirty:
            findings.append({
                "rule": "R2",
                "level": _level(rules, "R2"),
                "message": f"当前文档有未提交改动（git status: {dirty[rel]}）",
            })

    return {
        "doc": rel,
        "status_line": status[:120],
        "updated_at": updated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "findings": findings,
    }


def scan_repository(root: str | Path, *, rules_path: str | Path | None = None,
                    rules: Mapping[str, Any] | None = None,
                    since: timedelta | None = None,
                    profile: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Scan a profile-defined documentation tree without modifying the repository."""
    repo_root = Path(root).expanduser().resolve()
    if not repo_root.is_dir():
        raise ScanError(f"repository root is not a directory: {repo_root}")
    if profile is None:
        project_profile = repo_root / ".docagent" / "profile.yaml"
        selected_profile = load_profile(project_profile if project_profile.is_file() else None)
    else:
        selected_profile = load_profile(profile)
    if rules is not None:
        configured = validate_rules(rules)
    else:
        if rules_path is not None:
            configured = load_rules(rules_path)
        else:
            project_rules = repo_root / ".docagent" / "rules.yaml"
            configured = load_rules(project_rules if project_rules.is_file() else None)
    configured = apply_profile(configured, selected_profile)
    docs_dir = repo_root / selected_profile["docs_dir"]
    if not docs_dir.is_dir():
        raise ScanError(f"documentation directory is missing: {docs_dir}")

    since_date = datetime.now() - since if since else None
    docs: list[Path] = []
    update_dates: list[str] = []
    for doc in sorted(docs_dir.rglob("*.md")):
        relative = doc.relative_to(docs_dir).as_posix()
        if _excluded(relative, selected_profile["exclude"]):
            continue
        if since_date and datetime.fromtimestamp(doc.stat().st_mtime) < since_date:
            continue
        docs.append(doc)
        updated = _updated_at(doc.read_text(encoding="utf-8", errors="replace"))
        if updated:
            update_dates.append(updated)

    r2 = _params(configured, "R2")
    r3 = _params(configured, "R3")
    dirty_docs = (
        _dirty_doc_paths(repo_root, str(r2["git_scope"])) if _enabled(configured, "R2") else {}
    )
    source_changes = (
        _source_changes(repo_root, min(update_dates), str(r3["source_root"]))
        if update_dates and _enabled(configured, "R3") else []
    )
    records = [scan_document(
        doc, repo_root, configured, dirty_docs=dirty_docs, source_changes=source_changes,
    ) for doc in docs]
    report = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "rules": {str(rule["id"]): str(rule["name"]) for rule in configured["rules"]},
        "ruleset_version": configured["ruleset_version"],
        "rules_fingerprint": rules_fingerprint(configured),
        "profile": profile_metadata(selected_profile),
        "docs": records,
    }
    report["report_fingerprint"] = report_fingerprint(report)
    return report


__all__ = ["ScanError", "scan_document", "scan_repository"]
