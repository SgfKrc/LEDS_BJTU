"""PATCH-05 流通记录：文档 git 变更时间线（本地、fail-soft）。

- `file_history`：git log --follow --numstat → 提交列表（sha/date/author/subject/±行）
- `render_history`：纯文本时间线（供 TUI 预览区显示）
非 git 仓库/无 git/超时：全部 fail-soft 为 {"ok": False, "error": ...}。
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

DEFAULT_LIMIT = 30
DEFAULT_TIMEOUT = 60
_RECORD_SEP = "\x1e"


def file_history(root: Path, path: str, limit: int = DEFAULT_LIMIT, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """返回 {"ok", "commits": [...], "error"}；commits 新→旧。"""
    root = Path(root)
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--date=short",
                f"--max-count={limit}",
                "--numstat",
                f"--format={_RECORD_SEP}%H|%ad|%an|%s",
                "--",
                path,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # noqa: BLE001
        return {"ok": False, "commits": [], "error": f"subprocess: {exc!r}"}
    if proc.returncode != 0:
        return {"ok": False, "commits": [], "error": (proc.stderr or "").strip()[:200] or f"exit={proc.returncode}"}

    commits: list[dict[str, Any]] = []
    for block in proc.stdout.split(_RECORD_SEP):
        lines = [line for line in block.strip("\n").splitlines()]
        if not lines:
            continue
        meta = lines[0].split("|", 3)
        if len(meta) < 4:
            continue
        added = deleted = 0
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                added += int(parts[0])
                deleted += int(parts[1])
        commits.append(
            {
                "sha": meta[0][:10],
                "date": meta[1],
                "author": meta[2],
                "subject": meta[3][:100],
                "added": added,
                "deleted": deleted,
            }
        )
    return {"ok": True, "commits": commits, "error": None}


def render_history(result: dict[str, Any], path: str, max_lines: int = 30) -> str:
    if not result.get("ok"):
        return f"流通记录不可用：\n{result.get('error')}"
    commits = result.get("commits") or []
    header = [f"== 流通记录（git log --follow）== {path}", f"提交数: {len(commits)}", ""]
    rows = []
    for commit in commits[:max_lines]:
        delta = f"+{commit['added']}/-{commit['deleted']}" if (commit["added"] or commit["deleted"]) else "?" 
        rows.append(f"{commit['date']}  {commit['sha']}  {delta:>10}  {commit['subject']}")
    if len(commits) > max_lines:
        rows.append(f"… 其余 {len(commits) - max_lines} 条省略")
    return "\n".join(header + rows)
