#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QLH 一键环境配置：主运行时 + 全部虚拟环境 + (可选) Node 子项目。

用法示例
--------
  python scripts/setup_envs.py --list                    # 查看环境清单
  python scripts/setup_envs.py --all                     # 全部（Python 环境 + Node 子项目）
  python scripts/setup_envs.py --all --no-node           # 仅 Python 环境
  python scripts/setup_envs.py --only test,tui           # 只配指定环境
  python scripts/setup_envs.py --skip packaging,packaging-cuda --all
  python scripts/setup_envs.py --check                   # 只校验现有环境，不安装
  python scripts/setup_envs.py --snapshot                # 从现有 venv 导出 requirements-lock/*.lock.txt
  python scripts/setup_envs.py --torch-index-url https://download.pytorch.org/whl/cu126

平台相关说明
------------
* torch / torchvision / torchaudio **不会自动安装**（体积大、CPU/CUDA 平台相关，
  且各打包 venv 严禁混用版本）。脚本只打印安装命令；用 --torch-index-url 可让提示
  带上你选好的 PyTorch 源（例如 CPU: https://download.pytorch.org/whl/cpu）。
* 需要源码构建的包（如 llama-cpp-python）会自动尝试；缺编译工具链时参照对应
  requirements 文件头注释处理（Windows 需 MSVC Build Tools）。
* 主运行时（main）按 README 标准流程 `pip install -r requirements.txt` 安装，
  torch 会按 pip 默认（Windows 上为 CPU wheel）落下；装完后可自行覆盖为 CUDA 版。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "requirements-lock"

# pip freeze 中需要跳过的自带工具
BASE_SKIP = frozenset({"pip", "setuptools", "wheel", "distribute"})
# 平台相关、不得自动锁定/自动安装的包
TORCH_SET = frozenset({"torch", "torchvision", "torchaudio"})
# 主环境/测试环境安装时保留 torch（README 标准流程）；仅 needs_torch 的 venv 过滤
TORCH_HINT_DEFAULT_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass(frozen=True)
class PyEnv:
    """一个 Python 运行环境（主环境或 venv）。"""

    name: str
    description: str
    venv_dir: str | None          # None => 主环境（系统 Python，不创建 venv）
    requirements: tuple[str, ...] # 相对仓库根的依赖文件
    lock_file: str                # requirements-lock 下的快照文件名
    system_site_packages: bool = False
    needs_torch: bool = False     # True => 自动安装时过滤 torch 系，仅打印提示
    extra_packages: tuple[str, ...] = ()   # 追加到 -r 之外的包（如 pyinstaller）
    python_version_hint: str = "3.12"

    @property
    def is_main(self) -> bool:
        return self.venv_dir is None


ENVS: tuple[PyEnv, ...] = (
    PyEnv(
        name="main",
        description="主运行时（系统 Python；transformers/torch 推理服务与工具脚本）",
        venv_dir=None,
        requirements=("requirements.txt",),
        lock_file="main.lock.txt",
    ),
    PyEnv(
        name="test",
        description="隔离测试环境（.venv-test；全量/定向 pytest，勿装系统 Python）",
        venv_dir=".venv-test",
        requirements=("requirements-test.txt",),
        lock_file="test.lock.txt",
        system_site_packages=True,
        python_version_hint="3.12",
    ),
    PyEnv(
        name="tui",
        description="T9 终端聊天页（.venv-tui；textual）",
        venv_dir=".venv-tui",
        requirements=("packaging/requirements-tui.txt",),
        lock_file="tui.lock.txt",
    ),
    PyEnv(
        name="gemma4-native",
        description="Gemma 4 MTMD 原生运行时（.venv-gemma4-native；llama.cpp GGUF）",
        venv_dir=".venv-gemma4-native",
        requirements=("packaging/requirements-gemma4-native.txt",),
        lock_file="gemma4-native.lock.txt",
    ),
    PyEnv(
        name="gemma4-pipeline",
        description="Gemma 4 PyTorch Transformers 5.10.1 侧车（.venv-gemma4-pipeline）",
        venv_dir=".venv-gemma4-pipeline",
        requirements=("packaging/requirements-gemma4-pipeline-sidecar.txt",),
        lock_file="gemma4-pipeline.lock.txt",
        needs_torch=True,
        python_version_hint="3.12",
    ),
    PyEnv(
        name="qwen3-sidecar",
        description="Qwen3 PyTorch sidecar（.venv-qwen3-sidecar；含 pipeline 执行依赖）",
        venv_dir=".venv-qwen3-sidecar",
        requirements=(
            "packaging/requirements-qwen3-sidecar.txt",
            "packaging/requirements-qwen3-pipeline-sidecar.txt",
        ),
        lock_file="qwen3-sidecar.lock.txt",
        needs_torch=True,
        python_version_hint="3.12",
    ),
    PyEnv(
        name="packaging",
        description="集显版打包（.venv-packaging；torch CPU + PyInstaller）",
        venv_dir=".venv-packaging",
        requirements=("packaging/requirements-cpu.txt",),
        lock_file="packaging.lock.txt",
        needs_torch=True,
        extra_packages=("pyinstaller",),
        python_version_hint="3.12",
    ),
    PyEnv(
        name="packaging-cuda",
        description="独显版打包 + SD 侧车（.venv-packaging-cuda；torch CUDA + PyInstaller）",
        venv_dir=".venv-packaging-cuda",
        requirements=(
            "packaging/requirements-cpu.txt",
            "packaging/requirements-sd15.txt",
        ),
        lock_file="packaging-cuda.lock.txt",
        needs_torch=True,
        extra_packages=("pyinstaller",),
        python_version_hint="3.12",
    ),
)

# Node 子项目（均有 package-lock.json，用 npm ci 可复现安装）
NODE_PROJECTS: tuple[tuple[str, str], ...] = (
    ("frontend", "前端 React 仪表盘"),
    ("gateway", "API 网关"),
    ("control", "控制台服务"),
)

ENV_BY_NAME = {env.name: env for env in ENVS}
NODE_BY_NAME = {name: desc for name, desc in NODE_PROJECTS}


# ---------------------------------------------------------------- path helpers

def _venv_bin_dir(venv_dir: str) -> Path:
    if os.name == "nt":
        return ROOT / venv_dir / "Scripts"
    return ROOT / venv_dir / "bin"


def _venv_python(env: PyEnv) -> Path | None:
    if env.is_main or env.venv_dir is None:
        return None
    return _venv_bin_dir(env.venv_dir) / ("python.exe" if os.name == "nt" else "python")


def _resolve_python(env: PyEnv, base_python: Path) -> Path:
    """返回该环境实际使用的解释器路径（主环境用 base_python，venv 用 venv 内 python）。"""
    if env.is_main:
        return base_python
    return _venv_python(env) or base_python


# ---------------------------------------------------------------- install 逻辑

def _is_torch_line(line: str) -> bool:
    """判断一行 requirement 是否指向 torch 系包（用于安装时过滤）。"""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "-")):
        return False
    m = re.match(r"[A-Za-z0-9_.-]+", stripped)
    return bool(m) and m.group(0).lower() in TORCH_SET


def _write_filtered_reqs(env: PyEnv, tmpdir: Path) -> list[Path]:
    """对 needs_torch 环境，把 requirements 中的 torch 系行过滤后写成临时文件。"""
    filtered: list[Path] = []
    for rel in env.requirements:
        path = ROOT / rel
        kept = [
            line
            for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not _is_torch_line(line)
        ]
        tmp = tmpdir / (env.name + "-" + Path(rel).name)
        tmp.write_text("".join(kept), encoding="utf-8")
        filtered.append(tmp)
    return filtered


def _ensure_venv(env: PyEnv, base_python: Path, dry_run: bool) -> Path:
    py = _venv_python(env)
    assert py is not None
    if py.is_file():
        return py
    command = [str(base_python), "-m", "venv"]
    if env.system_site_packages:
        command.append("--system-site-packages")
    command.append(str(ROOT / env.venv_dir))
    print(f"[{env.name}] 创建虚拟环境: {' '.join(command)}")
    if not dry_run:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(f"[{env.name}] 创建 venv 失败（退出码 {result.returncode}）")
    return py


def _torch_hint(env: PyEnv, index_url: str) -> str:
    if env.name == "packaging":
        return (
            f"[{env.name}] 请手动安装 torch CPU 版（打包 venv 严禁装 CUDA 版）：\n"
            f"    {env.venv_dir}\\Scripts\\python.exe -m pip install torch --index-url"
            f" {index_url if index_url else TORCH_HINT_DEFAULT_INDEX}"
        )
    if env.name == "packaging-cuda":
        return (
            f"[{env.name}] 请手动安装 torch CUDA 版（不是 CPU 版）：\n"
            f"    {env.venv_dir}\\Scripts\\python.exe -m pip install torch"
            + (f" --index-url {index_url}" if index_url else "（默认官方 CUDA 12.x）")
        )
    lines = [
        f"[{env.name}] 该环境需要 torch 或 torchvision，未自动安装。"
        f"请先按平台安装对应版本：",
        f"    {env.venv_dir}\\Scripts\\python.exe -m pip install torch"
        + (f" --index-url {index_url}" if index_url else ""),
    ]
    if env.name == "gemma4-pipeline":
        lines.append("    （Transformers 5.10.1 建议 torch>=2.10,<2.14）")
    if env.name == "qwen3-sidecar":
        lines.append("    装完 torch 后还需 torchvision：python -m pip install 'torchvision>=0.28.0,<0.29.0'")
    return "\n".join(lines)


def install_python_env(env: PyEnv, base_python: Path, args: argparse.Namespace) -> None:
    print(f"\n===== 配置环境: {env.name} —— {env.description} =====")
    py = _resolve_python(env, base_python)
    if env.is_main:
        print(f"[{env.name}] 主环境直接使用 {py}，不创建 venv。")
        if not py.is_file():
            raise SystemExit(f"[{env.name}] 解释器不存在: {py}")
    else:
        py = _ensure_venv(env, base_python, args.dry_run)

    requirements = [str(ROOT / rel) for rel in env.requirements]
    if env.needs_torch:
        with tempfile.TemporaryDirectory(prefix="qlh-setup-") as tmp:
            tmpdir = Path(tmp)
            filtered = _write_filtered_reqs(env, tmpdir)
            install = [str(py), "-m", "pip", "install"]
            if len(filtered) >= 2:
                for f in filtered:
                    install += ["-r", str(f)]
            else:
                install += ["-r", str(filtered[0])]
            for pkg in env.extra_packages:
                install.append(pkg)
            print(f"[{env.name}] 安装依赖: {' '.join(str(i) for i in install[2:])}  （torch 系已过滤）")
            if not args.dry_run:
                result = subprocess.run(install, cwd=ROOT, check=False)
                if result.returncode != 0:
                    raise SystemExit(f"[{env.name}] 依赖安装失败（退出码 {result.returncode}）")
        print(_torch_hint(env, args.torch_index_url))
    else:
        install = [str(py), "-m", "pip", "install"]
        for req in requirements:
            install += ["-r", req]
        for pkg in env.extra_packages:
            install.append(pkg)
        print(f"[{env.name}] 安装依赖: {' '.join(install[2:])}")
        if not args.dry_run:
            result = subprocess.run(install, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise SystemExit(f"[{env.name}] 依赖安装失败（退出码 {result.returncode}）")
    if env.is_main and not args.dry_run:
        # 主环境按 README 标准流程装，装完提示可覆盖为 CUDA 版
        print(f"[{env.name}] 如推理需要 CUDA 版 torch，装完后可自行覆盖：")
        print(f"    {py} -m pip install torch --index-url "
              f"{args.torch_index_url if args.torch_index_url else 'https://download.pytorch.org/whl/cu126'}")
    print(f"[{env.name}] 完成。")


def install_node(project: str, description: str, args: argparse.Namespace) -> None:
    print(f"\n===== 配置 Node 子项目: {project} —— {description} =====")
    npm = shutil.which("npm")
    if npm is None:
        print(f"[node:{project}] 未找到 npm，跳过（可从 https://nodejs.org 安装 Node.js >= 18）")
        return
    project_dir = ROOT / project
    lock = project_dir / "package-lock.json"
    command = [npm] + (["ci"] if lock.is_file() else ["install"])
    print(f"[node:{project}] 运行: {' '.join(command)} @ {project}")
    if not args.dry_run:
        result = subprocess.run(command, cwd=project_dir, check=False)
        if result.returncode != 0:
            raise SystemExit(f"[node:{project}] npm 安装失败（退出码 {result.returncode}）")
    print(f"[node:{project}] 完成。")


# ---------------------------------------------------------------- check 逻辑

def check_python_env(env: PyEnv, base_python: Path) -> bool:
    py = _resolve_python(env, base_python)
    if env.is_main:
        if not py.is_file():
            print(f"[{env.name}] [MISSING] 主解释器不存在: {py}")
            return False
    else:
        py = _venv_python(env)
        if py is None or not py.is_file():
            print(f"[{env.name}] [MISSING] venv 未创建: {env.venv_dir}（可运行 setup_envs.py --only {env.name}）")
            return False
    result = subprocess.run([str(py), "-m", "pip", "check"], capture_output=True, text=True, check=False)
    ok = result.returncode == 0
    if ok:
        detail = "依赖一致性 OK"
    else:
        raw = (result.stdout.strip() or result.stderr.strip()).splitlines()
        detail = raw[0] if raw else "pip check 异常"
    print(f"[{env.name}] {'[OK]' if ok else '[MISSING]'} {detail}")
    if ok and env.needs_torch:
        # torch 相关仅提示，不做硬性校验（用户可能尚未安装）
        print(f"[{env.name}]   （torch 未自动安装；如未装请参照 --snapshot 提示的命令补齐）")
    return ok


def check_node(project: str, description: str) -> bool:
    """校验 Node 子项目是否已安装依赖。description 仅用于信息展示。"""
    node_modules = ROOT / project / "node_modules"
    ok = node_modules.is_dir()
    print(f"[node:{project}] {'[OK]' if ok else '[MISSING]'} node_modules {'存在' if ok else '缺失（可运行 setup_envs.py --only node 安装）'}")
    return ok


# ---------------------------------------------------------------- snapshot 逻辑

def _classify_freeze_line(raw: str, out: list[str]) -> None:
    """把 pip freeze 的一行处理后追加到 out；返回是否命中 torch（由调用方统计版本）。"""
    if not raw or raw.startswith("#"):
        return
    if raw.startswith("-e "):
        out.append("# (editable 安装已跳过，未锁定)")
        return
    if raw.startswith("-"):
        return
    match = re.match(r"([A-Za-z0-9_.-]+)(==.*| @ .*)$", raw)
    if not match:
        out.append(f"# (无法解析跳过) {raw}")
        return
    name, version = match.group(1).lower(), match.group(2)
    if name in BASE_SKIP:
        return
    if version.startswith(" @ file:"):
        out.append("# (本地文件路径安装已跳过，未锁定)")
        return
    if name in TORCH_SET:
        out.append(f"# {name}{version}   # torch 系不自动锁定，手动安装（--torch-index-url）")
        return
    out.append(raw)


def snapshot_env(env: PyEnv, base_python: Path) -> None:
    py = _resolve_python(env, base_python)
    if not py.is_file():
        print(f"[{env.name}] [MISSING] 环境不存在，跳过快照: {py}")
        return
    freeze = subprocess.run(
        [str(py), "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    if freeze.returncode != 0:
        print(f"[{env.name}] [MISSING] pip freeze 失败，跳过")
        return
    lines: list[str] = [
        "# ============================================================",
        f"# {env.name} 依赖精确快照（pip freeze 自动生成，勿手改）",
        f"# 环境: {env.description}",
        "# 生成: python scripts/setup_envs.py --snapshot",
        "# 说明: torch/torchvision/torchaudio 与 editable/本地路径安装不在此锁定；",
        "#       各类依赖按 packaging/requirements-*.txt 安装，torch 按 setup_envs.py 提示手动补齐。",
        "# ============================================================",
    ]
    torch_versions: list[str] = []
    for raw_line in freeze.stdout.splitlines():
        _classify_freeze_line(raw_line.strip(), lines)
        # 记录 torch 系实际版本（若 freeze 里有），便于提示
        m = re.match(r"([A-Za-z0-9_.-]+)(==.*)$", raw_line.strip())
        if m and m.group(1).lower() in TORCH_SET:
            torch_versions.append(f"{m.group(1)}{m.group(2)}")
    if torch_versions:
        lines.insert(4, f"# 当前 torch 系版本（参考）: {', '.join(torch_versions)}")
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    target = LOCK_DIR / env.lock_file
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{env.name}] [OK] 已导出 {target.relative_to(ROOT)}（{len(lines)} 行）")


# ---------------------------------------------------------------- CLI

def _select(args: argparse.Namespace) -> tuple[list[PyEnv], list[tuple[str, str]]]:
    py_envs: list[PyEnv] = []
    node_projects: list[tuple[str, str]] = []

    if args.only:
        names = [n.strip().lower() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in ENV_BY_NAME and n not in NODE_BY_NAME and n != "node"]
        if unknown:
            raise SystemExit(f"未知环境名: {', '.join(unknown)}\n可用: {', '.join(list(ENV_BY_NAME) + ['node'])}")
        py_envs = [ENV_BY_NAME[n] for n in names if n in ENV_BY_NAME]
        if "node" in names:
            node_projects = list(NODE_PROJECTS)
        elif args.no_node is False:
            node_projects = [p for p in NODE_PROJECTS if p[0] in names]
    elif args.all or args.check or args.snapshot:
        # 无副作用的 --check / --snapshot：未显式指定范围时默认全部
        py_envs = list(ENVS)
        if not args.no_node:
            node_projects = list(NODE_PROJECTS)
    else:
        parser.error("请使用 --all 或 --only <name,...> 指定要配置的环境")

    if args.skip:
        skip = {n.strip().lower() for n in args.skip.split(",") if n.strip()}
        unknown_skip = [n for n in skip if n not in ENV_BY_NAME and n not in NODE_BY_NAME and n != "node"]
        if unknown_skip:
            raise SystemExit(f"--skip 含未知环境名: {', '.join(unknown_skip)}")
        py_envs = [e for e in py_envs if e.name not in skip]
        node_projects = [p for p in node_projects if p[0] not in skip and "node" not in skip]
    return py_envs, node_projects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_envs.py",
        description="QLH 一键环境配置：主运行时 + 全部虚拟环境 + Node 子项目。",
    )
    parser.add_argument("--all", action="store_true", help="配置全部 Python 环境（及 Node 子项目）")
    parser.add_argument("--only", metavar="NAME,...", help="只配置指定环境（逗号分隔；node 单配 Node 项目）")
    parser.add_argument("--skip", metavar="NAME,...", help="跳过指定环境（配合 --all）")
    parser.add_argument("--check", action="store_true", help="只校验现有环境，不安装任何东西")
    parser.add_argument("--snapshot", action="store_true", help="从现有 venv 导出 requirements-lock/*.lock.txt")
    parser.add_argument("--no-node", action="store_true", help="跳过 Node 子项目")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的命令，不执行任何安装/创建")
    parser.add_argument("--torch-index-url", default="", metavar="URL",
                        help="PyTorch index URL，用于打印 torch 安装提示（如 .../whl/cpu 或 .../whl/cu126）")
    parser.add_argument("--python", type=Path, default=None, metavar="PATH",
                        help="用于主环境与创建 venv 的 Python 解释器（默认 sys.executable）")
    parser.add_argument("--list", action="store_true", help="打印环境清单后退出")
    return parser


def main(argv: list[str] | None = None) -> int:
    global parser
    # 统一以 UTF-8 输出，保证现代终端 / git-bash 下中文正常（老 cmd 请 `chcp 65001`）
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Python 环境：")
        for env in ENVS:
            lock = LOCK_DIR / env.lock_file
            print(f"  {env.name:<18} {env.description}")
            reqs = ", ".join(env.requirements)
            print(f"    venv: {env.venv_dir or '（系统 Python）'}  deps: {reqs}"
                  + (f"  lock: {'[OK]' if lock.is_file() else '[MISSING]'}" ))
            if env.needs_torch:
                print("    torch: 手动安装（不自动装）")
        print("Node 子项目：")
        for name, desc in NODE_PROJECTS:
            lock = ROOT / name / "package-lock.json"
            print(f"  {name:<18} {desc}  （{'npm ci' if lock.is_file() else 'npm install'}）")
        return 0

    base_python = (args.python.expanduser().resolve(strict=False) if args.python else Path(sys.executable))
    if not base_python.is_file():
        if os.name == "nt":
            base_python = Path(shutil.which("python") or shutil.which("py") or "")
        if not base_python or not base_python.is_file():
            print("找不到 Python 解释器；请用 --python 显式指定", file=sys.stderr)
            return 2

    if args.snapshot:
        py_envs, _ = _select(args)
        if not py_envs:
            print("没有可导出的 Python 环境（--only 或 --all 未指定有效名称）", file=sys.stderr)
            return 2
        for env in py_envs:
            snapshot_env(env, base_python)
        return 0

    py_envs, node_projects = _select(args)
    if not py_envs and not node_projects:
        print("没有选择到任何环境（请检查 --only / --skip / --no-node）", file=sys.stderr)
        return 2

    if args.check:
        ok = True
        for env in py_envs:
            ok = check_python_env(env, base_python) and ok
        for project, description in node_projects:
            ok = check_node(project, description) and ok
        print(f"\n校验完成：{'全部就绪 [OK]' if ok else '存在未就绪的环境 [MISSING]'}")
        return 0 if ok else 1

    print("=" * 64)
    print("QLH 环境一键配置开始")
    print(f"基础解释器: {base_python}")
    if args.torch_index_url:
        print(f"torch index-url: {args.torch_index_url}")
    print("=" * 64)
    for env in py_envs:
        install_python_env(env, base_python, args)
    for project, description in node_projects:
        install_node(project, description, args)
    print("\n===== 全部完成 =====")
    print("本地路径存在的原生依赖（llama-cpp-python 等）如需源码构建，请确保编译工具链可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
