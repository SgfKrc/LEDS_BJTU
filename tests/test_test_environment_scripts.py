"""Tests for the dedicated test-environment guard and bootstrapper."""

from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import run_test_channels
from scripts import setup_envs
from scripts import setup_qwen3_sidecar_env
from scripts import setup_test_env


def test_test_channel_guard_rejects_system_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "prefix", "C:/Python312")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is False
    error = capsys.readouterr().err
    assert "setup_test_env.py" in error
    assert "--reuse-runtime" not in error


def test_test_channel_guard_accepts_virtual_environment(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "G:/qlh/.venv-test")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is True


def test_install_command_keeps_proxy_and_requirements_in_venv(monkeypatch):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(proxy="http://127.0.0.1:7897")

    assert command[:4] == [
        str(test_python), "-m", "pip", "install",
    ]
    assert command[4:6] == ["--proxy", "http://127.0.0.1:7897"]
    assert command[-2:] == ["-r", str(setup_test_env.REQUIREMENTS)]


def test_wheelhouse_disables_network_proxy(monkeypatch, tmp_path):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(
        proxy="http://127.0.0.1:7897",
        wheelhouse=tmp_path,
    )

    assert "--no-index" in command
    assert f"--find-links={tmp_path}" in command
    assert "--proxy" not in command


def test_check_rejects_overlay_when_isolation_is_requested(monkeypatch, capsys):
    monkeypatch.setattr(setup_test_env, "_uses_system_site_packages", lambda: True)
    monkeypatch.setattr(
        setup_test_env,
        "_ready",
        lambda: (_ for _ in ()).throw(AssertionError("health check must not run")),
    )

    assert setup_test_env.main(["--check"]) == 2
    assert "existing environment is overlay" in capsys.readouterr().err


def test_check_accepts_explicit_overlay_mode(monkeypatch):
    monkeypatch.setattr(setup_test_env, "_uses_system_site_packages", lambda: True)
    monkeypatch.setattr(setup_test_env, "_ready", lambda: True)

    assert setup_test_env.main(["--check", "--reuse-runtime"]) == 0


def test_unified_setup_keeps_test_environment_isolated():
    test_env = setup_envs.ENV_BY_NAME["test"]

    assert test_env.system_site_packages is False
    assert {"pytest", "xdist", "pytest_timeout"} <= set(test_env.required_modules)


def test_sidecar_checks_require_torch_runtime_modules():
    qwen = setup_envs.ENV_BY_NAME["qwen3-sidecar"]
    gemma = setup_envs.ENV_BY_NAME["gemma4-pipeline"]

    assert {"torch", "torchvision", "transformers"} <= set(qwen.required_modules)
    assert {"torch", "accelerate", "transformers"} <= set(gemma.required_modules)


def test_qwen_sidecar_installs_torch_and_torchvision_from_one_index(monkeypatch):
    readiness = iter((False, True))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        setup_qwen3_sidecar_env,
        "_ready",
        lambda **_kwargs: next(readiness),
    )
    monkeypatch.setattr(
        setup_qwen3_sidecar_env,
        "_python_path",
        lambda: Path(sys.executable),
    )
    monkeypatch.setattr(
        setup_qwen3_sidecar_env.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(list(command)) or SimpleNamespace(returncode=0)
        ),
    )

    assert setup_qwen3_sidecar_env.main([
        "--pipeline",
        "--torch-index-url", "https://download.pytorch.org/whl/cu126",
    ]) == 0

    assert commands[0][-2:] == [
        "torch>=2.0", "torchvision>=0.28,<0.29",
    ]
    assert "https://download.pytorch.org/whl/cu126" in commands[0]


def test_qwen_unified_setup_hint_keeps_torch_wheels_on_one_index():
    hint = setup_envs._torch_hint(
        setup_envs.ENV_BY_NAME["qwen3-sidecar"],
        "https://download.pytorch.org/whl/cu126",
    )

    assert "torchvision>=0.28,<0.29" in hint
    assert hint.count("https://download.pytorch.org/whl/cu126") == 1


def test_qwen_runtime_probe_reports_mixed_wheels(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(
        setup_envs.subprocess,
        "run",
        lambda command, **_kwargs: (
            seen.append(list(command))
            or SimpleNamespace(returncode=1, stdout="mixed Torch builds", stderr="")
        ),
    )

    assert setup_envs._qwen3_torchvision_runtime_issue(Path("python")) == "mixed Torch builds"
    assert "torchvision.ops.nms" in seen[0][2]


# ================================================================
# T-4：测试分类报告脚本（scripts/test_classification_report.py）
# ================================================================

def test_classify_places_quality_gate_and_contract_files():
    from scripts import test_classification_report as tcr

    files_by_class = tcr.classify_test_files(Path("tests"))
    # 质量门文件进 quality_gate 类（marker 入口），不被其他类截胡
    assert "test_sd15_quality_gate.py" in files_by_class["quality_gate"]
    assert "test_sd15_img2img_quality_gate.py" in files_by_class["quality_gate"]
    # 契约类文件（命名启发式）
    assert any("contract" in name for name in files_by_class["contract"])
    # 每个文件恰好落入一个分类（不重不漏：顶层 + simulation 子目录）
    top_level = len(list(Path("tests").glob("test_*.py")))
    sub_level = len(list(Path("tests/simulation").glob("test_*.py")))
    total_classified = sum(len(v) for v in files_by_class.values())
    assert total_classified == top_level + sub_level


def test_classify_unit_is_fallback_for_unmatched_files():
    from scripts import test_classification_report as tcr

    files_by_class = tcr.classify_test_files(Path("tests"))
    # unit 是兜底类，必有内容（多数普通单测文件按命名落入）
    assert len(files_by_class["unit"]) > 0
