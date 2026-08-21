#!/usr/bin/env python3
"""QLH 外部运行时依赖守卫（打包瘦身后由引导器调用）。

背景：安装包不再把 PyTorch/Transformers 等巨无霸打进包体（CPU ~734MB /
CUDA ~13GB），改为「外部运行时依赖」：运行时放在用户目录的独立 venv，
由引导器（qlh_launcher）每次启动时检查缺失模块，缺则 ``pip install``
（CPU 走 PyTorch CPU index，CUDA 走官方源），完成后用该 venv 的 python
以源码方式启动主程序。这样安装包更新只换小包，不反复装卸大 PyTorch。

本模块纯标准库、无网络副作用（可注入 runner 单测），只产出检查结论与
将要执行的命令。
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

RUNTIME_ENV_VAR = "QLH_RUNTIME_DIR"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# 运行时必须可导入的模块（对瘦身包的主程序）。torch 系是核心，其余为主程序
# 启动/推理路径确定会 import 的模块；缺失即触发 pip 引导。注意：这里是"启动必需
# 最小集"，长尾依赖（einops/tiktoken/pandas 等）由 requirements-runtime 安装并
# 在首次引导时一起装上；probe 只防"旧 venv 缺启动必败模块"。
RUNTIME_REQUIRED_MODULES: tuple[str, ...] = (
    "torch", "transformers", "accelerate",
    "fastapi", "uvicorn", "httpx", "psutil", "dotenv", "llama_cpp",
)


@dataclass
class RuntimeContext:
    root: Path                      # 安装根（_internal/ 所在层）
    engine: str = "cpu"             # "cpu" | "cuda"
    requirements: Path | None = None  # 包内 requirements-runtime-<engine>.txt
    python_fallback: Path | None = None  # 无 venv 时回退解释器（默认 sys.executable）
    proxy: str = ""
    runtime_dir: Path | None = None  # 覆盖用户级 runtime 目录（测试/自定义部署用）


def runtime_dir() -> Path:
    """用户级外部运行时目录（默认 %LOCALAPPDATA%\\QLH-Edge-Inference\\runtime）。"""
    override = os.environ.get(RUNTIME_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "QLH-Edge-Inference" / "runtime"


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )


def required_modules(ctx: RuntimeContext) -> tuple[str, ...]:
    # torch 系 CUDA 时额外不需要不同模块集（torch 装 CUDA 源即可），保持同一清单
    return RUNTIME_REQUIRED_MODULES


def probe_missing(python: Path, modules: Sequence[str]) -> list[str]:
    """运行只读 import probe，返回缺失的模块名（不安装任何东西）。"""
    if not python.is_file():
        return list(modules)
    check = (
        "import importlib.util, sys; "
        f"modules = {list(modules)!r}; "
        "missing = [m for m in modules if importlib.util.find_spec(m) is None]; "
        "print('\\n'.join(missing)); "
        "raise SystemExit(1 if missing else 0)"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", check],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return list(modules)
    if result.returncode == 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] or list(modules)


def torch_index(ctx: RuntimeContext) -> str:
    """运行时要用的 PyTorch index URL（cuda="" 表示官方默认源）。"""
    if ctx.engine == "cuda":
        return ""
    return PYTORCH_CPU_INDEX


# fallback（无 requirements 文件时）需保证 probe 最小集可通过的包
RUNTIME_FALLBACK_PACKAGES: tuple[str, ...] = (
    "transformers", "accelerate", "fastapi", "uvicorn",
    "httpx", "psutil", "python-dotenv", "llama-cpp-python",
)


def build_pip_commands(
    python: Path, requirements: Path | None, ctx: RuntimeContext,
) -> list[list[str]]:
    """构造要依次执行的 pip 命令（只构造，不执行）。

    CPU 用 ``--index-url <pytorch-cpu>`` **只给 torch**，其余依赖走默认 PyPI
    （PyTorch 索引只有 torch 生态，若整条 -r 都带 --index-url 会让 transformers
    等全部找不到 —— 详见 review 修复）。CUDA 整个 requirements（含 torch，PyPI
    默认 CUDA wheel）一次装。proxy 逐命令附带。
    """
    base = [str(python), "-m", "pip", "install"]
    if ctx.proxy:
        base = list(base) + ["--proxy", ctx.proxy]
    commands: list[list[str]] = []
    if ctx.engine == "cpu":
        commands.append(list(base) + ["torch", "--index-url", PYTORCH_CPU_INDEX])
        if requirements is not None and requirements.is_file():
            commands.append(list(base) + ["-r", str(requirements)])
        else:
            commands.append(list(base) + list(RUNTIME_FALLBACK_PACKAGES))
    else:
        if requirements is not None and requirements.is_file():
            commands.append(list(base) + ["-r", str(requirements)])
        else:
            commands.append(list(base) + ["torch"] + list(RUNTIME_FALLBACK_PACKAGES))
    return commands


def ensure_runtime(
    ctx: RuntimeContext,
    *,
    runner: Callable[[list[str]], int] | None = None,
    create_venv: bool = True,
) -> dict[str, Any]:
    """确保外部运行时就绪：定位 venv → probe → 缺失则（建 venv +）pip 安装。

    返回报告：state ∈ {ok, installed, failed}、python、missing_before / missing_after。
    真正的 pip/网络执行通过 ``runner`` 注入；默认用 ``subprocess.run``（需联网）。
    """
    runner = runner or (lambda cmd: subprocess.run(cmd, check=False).returncode)
    target_venv = (
        ctx.runtime_dir if ctx.runtime_dir is not None else runtime_dir()
    )
    python = venv_python(target_venv)

    created = False
    if not python.is_file() and create_venv:
        try:
            venv.EnvBuilder(with_pip=True).create(target_venv)
            created = True
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "state": "failed", "python": str(python),
                "error": f"create venv failed: {exc.__class__.__name__}",
            }

    missing = probe_missing(python, required_modules(ctx))
    if not missing:
        return {
            "state": "ok", "python": str(python), "created": created,
            "missing_before": [], "missing_after": [],
        }

    exit_codes: list[int] = []
    for command in build_pip_commands(python, ctx.requirements, ctx):
        exit_codes.append(int(runner(command) or 0))
    missing_after = probe_missing(python, required_modules(ctx))
    if missing_after:
        return {
            "state": "failed", "python": str(python), "created": created,
            "exit": exit_codes,
            "error": "runtime install did not satisfy required modules",
            "missing_before": missing, "missing_after": missing_after,
        }
    return {
        "state": "installed", "python": str(python), "created": created,
        "exit": exit_codes, "missing_before": missing, "missing_after": [],
    }
