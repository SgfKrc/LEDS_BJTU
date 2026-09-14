"""配置检测与引导（.env.docagent）。

- `env_status`：存在性 + 键名清单（**不读值**，防泄漏）；
- `validate_env`：委托 `docagent config --root`（脱敏 JSON）判定可用性；
- `env_template`：读取官方模板（供 TUI 编辑屏预填）；
- `render_hint`：缺失/不可用时的引导文案。
**写权限边界**：本模块（及 TUI 编辑屏）仅允许写 `root/.env.docagent`，不动其他任何文件。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_NAME = ".env.docagent"
DEFAULT_TIMEOUT = 60


def env_path(root: Path) -> Path:
    return Path(root) / ENV_NAME


def template_path(root: Path) -> Path:
    return Path(root) / "tools" / "docagent" / ".env.docagent.example"


def env_status(root: Path) -> dict[str, Any]:
    """存在性 + 键名（脱敏：绝不返回值）。"""
    path = env_path(root)
    status: dict[str, Any] = {"exists": False, "path": str(path), "keys": [], "template": None}
    template = template_path(root)
    status["template"] = str(template) if template.is_file() else None
    if not path.is_file():
        return status
    status["exists"] = True
    keys: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key:
                keys.append(key)
    except OSError:
        pass
    status["keys"] = keys
    return status


def env_template(root: Path) -> str:
    path = template_path(root)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return (
        "# Safe local-first profile. Copy this file to .env.docagent; never commit the copy.\n"
        "DOCAGENT_PROVIDER=ollama\n"
        "DOCAGENT_PROFILE=qlh\n"
        "DOCAGENT_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1\n"
        "DOCAGENT_OLLAMA_MODEL=gemma4:12b\n"
        "DOCAGENT_CONFIDENCE_FLOOR=0.6\n"
    )


def validate_env(root: Path, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """docagent config --json（脱敏）；fail-soft。"""
    runner = Path(root) / "tools" / "docagent" / "run.py"
    if not runner.is_file():
        return {"ok": False, "error": "docagent run.py not found"}
    try:
        proc = subprocess.run(
            [sys.executable, str(runner), "config", "--root", str(root), "--json"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # noqa: BLE001
        return {"ok": False, "error": f"subprocess: {exc!r}"}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "").strip()[:200] or f"exit={proc.returncode}"}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"json: {exc}"}
    return {"ok": True, "summary": payload, "error": None}


def write_env(root: Path, content: str) -> dict[str, Any]:
    """写入 .env.docagent（唯一允许的写路径；先备份原文件为 .bak）。"""
    path = env_path(root)
    root = Path(root)
    if root.resolve() not in path.resolve().parents:
        return {"ok": False, "error": "path escape blocked"}
    backup = None
    if path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            return {"ok": False, "error": f"backup failed: {exc!r}"}
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        return {"ok": False, "error": f"write failed: {exc!r}"}
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return {"ok": True, "backup": str(backup) if backup else None}


def render_hint(status: dict[str, Any], validation: dict[str, Any] | None = None) -> str:
    if not status["exists"]:
        lines = [
            "== docagent 未配置 ==",
            f"缺少 {ENV_NAME}（{status['path']}）。",
            "按 e 打开配置编辑（已预填官方模板），或手动：",
            f"  copy \"{status['template']}\" \"{status['path']}\"" if status.get("template") else "  （模板文件缺失，使用内置默认模板）",
            "必填：DOCAGENT_PROVIDER（ollama / opencode）与对应 provider 的 URL/MODEL（远程另需 API_KEY）。",
        ]
        return "\n".join(lines)
    if validation and not validation.get("ok"):
        return "\n".join([
            "== docagent 配置校验未通过 ==",
            f"（{ENV_NAME}，键 {len(status['keys'])} 个）",
            f"原因：{validation.get('error')}",
            "按 e 打开配置编辑修改。",
        ])
    keys = " · ".join(status["keys"]) or "（无键）"
    return f"docagent 配置就绪（{ENV_NAME}：{keys}）"
