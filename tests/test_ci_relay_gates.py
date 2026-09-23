"""tests/test_ci_relay_gates.py — A10：CI 闸门自检自身的回归。

要点是 **"该红必须红"**：除了"自检在当前脚本上应当通过"，还要验**负向** —— 把被检脚本的闸门
改坏（silence 掉"该红"）后，自检**必须失败**。负向用例在 `tmp_path` 里的**副本**上做，不碰仓库文件。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
GATE_SCRIPTS = ("relay_quant_budget.py", "relay_health.py", "ci_relay_gates.py")


def _copy_gate_scripts(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in GATE_SCRIPTS:
        (destination / name).write_text((SCRIPTS / name).read_text(encoding="utf-8"),
                                        encoding="utf-8")
    return destination


def _run_gate_self_check(script_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script_dir / "ci_relay_gates.py")],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(REPO_ROOT), check=False)


def test_gate_self_check_passes_on_current_scripts() -> None:
    result = _run_gate_self_check(SCRIPTS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "闸门自检全部通过" in result.stdout


def test_gate_self_check_fails_when_budget_gate_is_silenced(tmp_path: Path) -> None:
    """预算闸门"越线却不红" ⇒ 自检必须失败（否则这条 CI 检查就是假绿）。"""
    copied = _copy_gate_scripts(tmp_path / "gates")
    target = copied / "relay_quant_budget.py"
    text = target.read_text(encoding="utf-8")
    assert "            return 1\n" in text
    target.write_text(text.replace("            return 1\n", "            return 0\n"),
                      encoding="utf-8")

    result = _run_gate_self_check(copied)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "越线集" in result.stdout


def test_gate_self_check_fails_when_health_gate_is_silenced(tmp_path: Path) -> None:
    """健康检查"有死目标却判健康" ⇒ 自检必须失败。"""
    copied = _copy_gate_scripts(tmp_path / "gates")
    target = copied / "relay_health.py"
    text = target.read_text(encoding="utf-8")
    marker = '    healthy = all(item["ok"] for item in results)'
    assert marker in text
    target.write_text(text.replace(marker, "    healthy = True"), encoding="utf-8")

    result = _run_gate_self_check(copied)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "健康检查" in result.stdout
