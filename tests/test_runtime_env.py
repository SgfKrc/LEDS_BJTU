import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import runtime_env


def _managed_asset(root: Path) -> None:
    target = root / "models" / "sd15-original-v1"
    target.mkdir(parents=True)
    (target / ".qlh-sd-asset.json").write_text("{}", encoding="utf-8")


def _candidate(root: Path) -> Path:
    candidate = runtime_env._cuda_python(root)
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"fixture")
    return candidate.resolve()


def test_runtime_switch_ignores_non_server_imports(tmp_path, monkeypatch):
    _managed_asset(tmp_path)
    _candidate(tmp_path)
    monkeypatch.setattr(runtime_env, "_is_api_server_invocation", lambda: False)
    monkeypatch.setattr(runtime_env.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(runtime_env.os, "execve", lambda *_args: pytest.fail("must not exec"))

    assert runtime_env.maybe_reexec_sd_runtime(tmp_path) is False


def test_runtime_switch_requires_a_managed_sd_asset(tmp_path, monkeypatch):
    _candidate(tmp_path)
    monkeypatch.setattr(runtime_env, "_is_api_server_invocation", lambda: True)
    monkeypatch.setattr(runtime_env.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(runtime_env.os, "execve", lambda *_args: pytest.fail("must not exec"))

    assert runtime_env.maybe_reexec_sd_runtime(tmp_path) is False


def test_runtime_switch_reexecs_server_with_project_cuda_python(tmp_path, monkeypatch):
    _managed_asset(tmp_path)
    candidate = _candidate(tmp_path)
    monkeypatch.setattr(runtime_env, "_is_api_server_invocation", lambda: True)
    monkeypatch.setattr(runtime_env.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(runtime_env, "_candidate_has_diffusers", lambda _path: True)
    monkeypatch.setattr(runtime_env.sys, "executable", str(tmp_path / "global-python.exe"))
    monkeypatch.setattr(
        runtime_env.sys,
        "orig_argv",
        [str(tmp_path / "global-python.exe"), "src/api_server.py", "--flag"],
        raising=False,
    )
    captured = {}

    def fake_execve(path, argv, env):
        captured.update(path=path, argv=argv, env=env)
        raise RuntimeError("exec captured")

    monkeypatch.setattr(runtime_env.os, "execve", fake_execve)

    with pytest.raises(RuntimeError, match="exec captured"):
        runtime_env.maybe_reexec_sd_runtime(tmp_path)

    assert captured["path"] == str(candidate)
    assert captured["argv"] == [str(candidate), "src/api_server.py", "--flag"]
    assert captured["env"][runtime_env.SD_RUNTIME_ACTIVE_ENV] == "1"


def test_runtime_switch_preserves_uvicorn_module_invocation(tmp_path):
    candidate = _candidate(tmp_path)

    assert runtime_env._reexec_argv(
        candidate,
        ["C:/Python/python.exe", "-m", "uvicorn", "src.api_server:app", "--port", "8000"],
    ) == [
        str(candidate),
        "-m",
        "uvicorn",
        "src.api_server:app",
        "--port",
        "8000",
    ]
    assert runtime_env._reexec_argv(
        candidate,
        ["C:/Python/Scripts/uvicorn.exe", "src.api_server:app"],
    ) == [str(candidate), "-m", "uvicorn", "src.api_server:app"]


def test_runtime_diagnostics_reports_missing_packages_without_absolute_paths(tmp_path, monkeypatch):
    _candidate(tmp_path)
    monkeypatch.setattr(runtime_env.sys, "executable", str(tmp_path / "global-python.exe"))
    dependencies = {name: True for name in runtime_env.SD_REQUIRED_DEPENDENCIES}
    dependencies["diffusers"] = False

    report = runtime_env.sd_runtime_diagnostics(dependencies, tmp_path)

    assert report["missing_dependencies"] == ["diffusers"]
    assert report["runtime_environment"] == "default"
    assert report["project_cuda_environment_available"] is True
    assert str(tmp_path) not in str(report)
