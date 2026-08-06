"""T9 聊天页可选环境引导：创建 .venv-tui 并安装 Textual/httpx。

设计（TUI 适配实施计划 §9.3）：
- 只写入项目内 .venv-tui，禁止静默修改系统/全局解释器；
- 尊重 HTTPS_PROXY / PIP_INDEX_URL 与用户指定镜像；
- 支持 --wheelhouse <dir> 离线安装（无网络环境）；
- 幂等：环境已存在且依赖满足时直接提示跳过；
- 网络失败保留重试命令，不在引导脚本内循环弹窗。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv-tui"
REQUIREMENTS = ROOT / "packaging" / "requirements-tui.txt"

if os.name == "nt":
    PYTHON_BIN = VENV_DIR / "Scripts" / "python.exe"
    PIP_BIN = VENV_DIR / "Scripts" / "pip.exe"
else:
    PYTHON_BIN = VENV_DIR / "bin" / "python"
    PIP_BIN = VENV_DIR / "bin" / "pip"

REQUIRED_MODULES = ("textual", "httpx")


def _venv_ready() -> bool:
    if not PYTHON_BIN.is_file():
        return False
    try:
        probe = subprocess.run(
            [str(PYTHON_BIN), "-c", "import textual, httpx; print('ok')"],
            capture_output=True, text=True, timeout=60,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_tui_env",
        description="创建 .venv-tui 并安装 T9 聊天页可选依赖",
    )
    parser.add_argument(
        "--wheelhouse", default="",
        help="离线 wheelhouse 目录（存在时优先 --no-index 安装）",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if _venv_ready():
        print(f"[T9] .venv-tui 已就绪（{VENV_DIR}）")
        return 0

    wheelhouse = None
    if args.wheelhouse:
        wheelhouse = Path(args.wheelhouse).resolve()
        if not wheelhouse.is_dir():
            print(f"[T9] wheelhouse 目录不存在: {wheelhouse}")
            return 1

    print(f"[T9] 创建虚拟环境: {VENV_DIR}")
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as exc:
        print(f"[T9] 虚拟环境创建失败: {exc}")
        print("[T9] 也可手动安装: pip install -r packaging/requirements-tui.txt")
        return 1

    print(f"[T9] 安装依赖: {REQUIREMENTS.name}")
    env = dict(os.environ)
    cmd = [str(PIP_BIN), "install", "-r", str(REQUIREMENTS)]
    if wheelhouse is not None:
        print(f"[T9] 离线安装（--no-index --find-links {wheelhouse}）")
        cmd = [str(PIP_BIN), "install", "--no-index",
               f"--find-links={wheelhouse}", "-r", str(REQUIREMENTS)]
    try:
        proc = subprocess.run(cmd, env=env)
    except Exception as exc:
        print(f"[T9] 依赖安装启动失败: {exc}")
        return 1
    if proc.returncode != 0:
        print("[T9] 安装失败。可重试:")
        print(f"    {PYTHON_BIN} -m pip install -r packaging/requirements-tui.txt")
        print("    或指定镜像，例如:")
        print("    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple "
              f"{PYTHON_BIN} -m pip install -r packaging/requirements-tui.txt")
        print("    或离线 wheelhouse:")
        print(f"    {PYTHON_BIN} -m pip install --no-index "
              f"--find-links=<wheelhouse目录> -r packaging/requirements-tui.txt")
        return proc.returncode

    if _venv_ready():
        print("[T9] 环境就绪。启动聊天页:")
        print(f"    {PYTHON_BIN} src/tui_chat.py --host http://127.0.0.1:8888")
        return 0
    print("[T9] 安装完成但依赖检查未通过，请查看上方输出")
    return 1


if __name__ == "__main__":
    sys.exit(main())
