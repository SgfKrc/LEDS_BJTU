"""tests/test_ci_relay_gates.py — A10：CI 闸门自检自身的回归。

要点是 **"该红必须红"**：除了"自检在当前脚本上应当通过"，还要验**负向** —— 把被检脚本的判定方向
改坏（silence 掉"该红"）后，自检**必须失败**。负向用例只改 `tmp_path` 里的**副本**，不碰仓库文件。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
BUDGET_SCRIPT = "relay_quant_budget.py"
HEALTH_SCRIPT = "relay_health.py"


def _run_self_check(script_dir: Path, name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script_dir / name), "--self-check"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(REPO_ROOT), check=False)


def _copy(script_dir: Path, name: str) -> Path:
    script_dir.mkdir(parents=True, exist_ok=True)
    target = script_dir / name
    target.write_text((SCRIPTS / name).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_budget_gate_self_check_passes() -> None:
    result = _run_self_check(SCRIPTS, BUDGET_SCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "预算闸门自检通过" in result.stdout


def test_health_gate_self_check_passes() -> None:
    result = _run_self_check(SCRIPTS, HEALTH_SCRIPT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "健康检查自检通过" in result.stdout


def test_budget_self_check_fails_when_unsafe_verdict_is_silenced(tmp_path: Path) -> None:
    """把"有余量的 prompt 却翻转 ⇒ unsafe"改成 `safe` ⇒ 自检必须失败。"""
    target = _copy(tmp_path / "gates", BUDGET_SCRIPT)
    text = target.read_text(encoding="utf-8")
    marker = '    if entry["n_headroom_flipped"]:\n        return "unsafe"\n'
    assert marker in text
    target.write_text(
        text.replace(marker, '    if entry["n_headroom_flipped"]:\n        return "safe"\n'),
        encoding="utf-8")

    result = _run_self_check(tmp_path / "gates", BUDGET_SCRIPT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "越线集" in result.stdout


def test_budget_self_check_fails_when_gate_can_never_turn_red(tmp_path: Path) -> None:
    """把等级表压成"永远不红" ⇒ 自检必须失败（这正是 A10 要防的**假绿**）。"""
    target = _copy(tmp_path / "gates", BUDGET_SCRIPT)
    text = target.read_text(encoding="utf-8")
    marker = ('_VERDICT_RANK = {"safe": 0, "safe-headroom-only": 0, "tight": 1, '
              '"no-headroom": 1, "unsafe": 2}')
    assert marker in text
    flattened = ('_VERDICT_RANK = dict.fromkeys('
                 '("safe", "safe-headroom-only", "tight", "no-headroom", "unsafe"), 0)')
    target.write_text(text.replace(marker, flattened), encoding="utf-8")

    result = _run_self_check(tmp_path / "gates", BUDGET_SCRIPT)
    assert result.returncode == 1, result.stdout + result.stderr


def test_health_self_check_fails_when_dead_port_is_reported_healthy(tmp_path: Path) -> None:
    """把"连不上"也判成健康 ⇒ 自检必须失败。"""
    target = _copy(tmp_path / "gates", HEALTH_SCRIPT)
    text = target.read_text(encoding="utf-8")
    marker = ('    except OSError as exc:\n'
              '        return {"ok": False, "reason": "unreachable", "endpoint": endpoint,')
    assert marker in text
    target.write_text(
        text.replace(marker, '    except OSError as exc:\n'
                             '        return {"ok": True, "reason": "injected", "endpoint": endpoint,'),
        encoding="utf-8")

    result = _run_self_check(tmp_path / "gates", HEALTH_SCRIPT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "已释放端口" in result.stdout


def test_health_self_check_fails_when_relay_probe_is_neutered(tmp_path: Path) -> None:
    """★ R-R9：把**协议级探活**的失败也判成健康（等价于"只测端口"）⇒ 自检必须失败。

    这正是 §8.6 的坑：隧道端口**还在监听**、但对端已死（accept 后立刻 reset）。
    若有人把 `_probe_relay` 的失败分支改坏（或干脆退回 `_check_tcp`），自检里
    「假服务必须被 `_probe_relay` 判死」那条会立刻红。
    """
    target = _copy(tmp_path / "gates", HEALTH_SCRIPT)
    text = target.read_text(encoding="utf-8")
    marker = '        return {"ok": False, "reason": "handshake_failed", "endpoint": endpoint,'
    assert marker in text, "变异锚点变了：请同步更新本用例"
    target.write_text(
        text.replace(marker,
                     '        return {"ok": True, "reason": "injected", "endpoint": endpoint,'),
        encoding="utf-8")

    result = _run_self_check(tmp_path / "gates", HEALTH_SCRIPT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "假服务" in result.stdout, result.stdout


def test_health_self_check_fails_when_live_segment_is_rejected(tmp_path: Path) -> None:
    """★ R-R9：把**活段**也判死（握手的成功分支被改坏）⇒ 自检必须失败。

    这条防的是**误杀真段**：探活太严会把好端端的段判死 ⇒ 接力直接起不来，
    比漏报更难查（链路上看不到任何"探活"字样）。「该红必须红」：改坏成功分支，本用例立刻红。
    """
    target = _copy(tmp_path / "gates", HEALTH_SCRIPT)
    text = target.read_text(encoding="utf-8")
    marker = '        return {"ok": True, "reason": "protocol_handshake_ok", "endpoint": endpoint}'
    assert marker in text, "变异锚点变了：请同步更新本用例"
    target.write_text(
        text.replace(marker,
                     '        return {"ok": False, "reason": "injected", "endpoint": endpoint}'),
        encoding="utf-8")

    result = _run_self_check(tmp_path / "gates", HEALTH_SCRIPT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "协议活段" in result.stdout, result.stdout


def test_relay_driver_rejects_fake_segment_before_relaying(tmp_path: Path) -> None:
    """★ R-R9：接力**前置探活**必须拦住「端口在监听但服务已死」的假段（§8.6 的现象）。

    子进程里起一个「accept 后立刻 close」的假服务，再调驱动的 `_require_relay_alive` ⇒
    必须 `SystemExit`（fail-loud），而不是把它放进接力、跑出一个**无法归因**的结果。
    这也是"该红必须红"：把探活退回"只 connect"，本用例立刻红。
    """
    script = tmp_path / "fake_segment_probe.py"
    script.write_text(textwrap.dedent(f'''
        import socket, sys, threading
        sys.path[:0] = [r"{REPO_ROOT / "scripts"}", r"{REPO_ROOT / "src"}", r"{REPO_ROOT}"]

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = int(srv.getsockname()[1])

        def serve():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                conn.close()          # accept 后立刻 close => §8.6 的现象

        threading.Thread(target=serve, daemon=True).start()

        from relay_experiment import _require_relay_alive

        try:
            _require_relay_alive(f"127.0.0.1:{{port}}")
        except SystemExit as exc:
            print("REJECTED", exc)
            raise SystemExit(0)
        print("ACCEPTED")
        raise SystemExit(1)
    '''), encoding="utf-8")

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", cwd=str(REPO_ROOT), check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REJECTED" in result.stdout, result.stdout
