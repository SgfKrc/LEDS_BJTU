"""Runtime selection helpers for optional project-local environments."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence


SD_AUTO_RUNTIME_ENV = "QLH_SD_AUTO_RUNTIME"
SD_RUNTIME_ACTIVE_ENV = "QLH_SD_RUNTIME_ACTIVE"
SD_RUNTIME_ERROR_ENV = "QLH_SD_RUNTIME_ERROR"
SD_REQUIRED_DEPENDENCIES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "PIL",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cuda_python(repo_root: Path) -> Path:
    if os.name == "nt":
        return repo_root / ".venv-packaging-cuda" / "Scripts" / "python.exe"
    return repo_root / ".venv-packaging-cuda" / "bin" / "python"


def _has_managed_sd_asset(repo_root: Path) -> bool:
    models = repo_root / "models"
    return any(
        (models / directory / ".qlh-sd-asset.json").is_file()
        for directory in ("sd15-original-v1", "sd15-90s-retrovers-v1")
    )


def _is_api_server_invocation(
    argv: Optional[Sequence[str]] = None,
    orig_argv: Optional[Sequence[str]] = None,
) -> bool:
    current = list(argv if argv is not None else sys.argv)
    original = list(orig_argv if orig_argv is not None else getattr(sys, "orig_argv", current))
    if current and Path(current[0]).name.lower() == "api_server.py":
        return True
    if original and Path(original[0]).name.lower() in {"uvicorn", "uvicorn.exe"}:
        return True
    return any(
        original[index] == "-m" and original[index + 1] == "uvicorn"
        for index in range(max(0, len(original) - 1))
    )


def _candidate_has_diffusers(candidate: Path) -> bool:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [
                str(candidate),
                "-c",
                "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('diffusers') else 1)",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _reexec_argv(candidate: Path, orig_argv: Optional[Sequence[str]] = None) -> list[str]:
    original = list(orig_argv if orig_argv is not None else getattr(sys, "orig_argv", sys.argv))
    if original and Path(original[0]).name.lower() in {"uvicorn", "uvicorn.exe"}:
        return [str(candidate), "-m", "uvicorn", *original[1:]]
    if original:
        original[0] = str(candidate)
        return original
    return [str(candidate), *sys.argv]


def maybe_reexec_sd_runtime(repo_root: Optional[Path] = None) -> bool:
    """Replace an API-server process with the project CUDA Python when needed."""

    if not _is_api_server_invocation():
        return False
    if os.environ.get(SD_AUTO_RUNTIME_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if os.environ.get(SD_RUNTIME_ACTIVE_ENV) == "1":
        return False
    if importlib.util.find_spec("diffusers") is not None:
        return False

    root = (repo_root or _repo_root()).resolve()
    if not _has_managed_sd_asset(root):
        return False
    candidate = _cuda_python(root).resolve()
    try:
        if Path(sys.executable).resolve() == candidate:
            return False
    except OSError:
        pass
    if not candidate.is_file():
        os.environ[SD_RUNTIME_ERROR_ENV] = "project_cuda_environment_missing"
        return False
    if not _candidate_has_diffusers(candidate):
        os.environ[SD_RUNTIME_ERROR_ENV] = "project_cuda_environment_missing_diffusers"
        return False

    child_env = dict(os.environ)
    child_env[SD_RUNTIME_ACTIVE_ENV] = "1"
    child_env.pop(SD_RUNTIME_ERROR_ENV, None)
    print("检测到本地 SD 资产，正在切换到项目 CUDA 虚拟环境...", file=sys.stderr, flush=True)
    os.execve(str(candidate), _reexec_argv(candidate), child_env)
    return True


def sd_runtime_diagnostics(
    dependencies: Mapping[str, bool],
    repo_root: Optional[Path] = None,
) -> dict[str, object]:
    root = (repo_root or _repo_root()).resolve()
    candidate = _cuda_python(root).resolve()
    try:
        using_project_cuda = Path(sys.executable).resolve() == candidate
    except OSError:
        using_project_cuda = False
    missing = [name for name in SD_REQUIRED_DEPENDENCIES if not dependencies.get(name, False)]
    return {
        "missing_dependencies": missing,
        "runtime_environment": "project_cuda" if using_project_cuda else "default",
        "project_cuda_environment_available": candidate.is_file(),
        "auto_switch_enabled": os.environ.get(SD_AUTO_RUNTIME_ENV, "1").strip().lower()
        not in {"0", "false", "no", "off"},
        "auto_switch_error": os.environ.get(SD_RUNTIME_ERROR_ENV) or None,
        "recommended_command": ".venv-packaging-cuda\\Scripts\\python.exe src\\api_server.py",
    }


__all__ = ["SD_REQUIRED_DEPENDENCIES", "maybe_reexec_sd_runtime", "sd_runtime_diagnostics"]
