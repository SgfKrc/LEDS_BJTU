from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, ".")

from scripts import setup_gemma4_pipeline_env as setup  # noqa: E402


def test_pipeline_environment_is_distinct_from_native_mtmd_environment():
    assert setup.VENV_DIR.name == ".venv-gemma4-pipeline"
    assert setup.NATIVE_VENV_DIR.name == ".venv-gemma4-native"
    assert setup.VENV_DIR != setup.NATIVE_VENV_DIR
    assert setup.TRANSFORMERS_VERSION == "5.10.1"


def test_pip_arguments_use_user_proxy_or_offline_wheelhouse(tmp_path):
    python = Path("python")
    proxied = setup._pip_base(
        python, wheelhouse=None, proxy="http://127.0.0.1:7897",
    )
    assert proxied[-2:] == ["--proxy", "http://127.0.0.1:7897"]

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    offline = setup._pip_base(
        python, wheelhouse=wheelhouse, proxy="http://127.0.0.1:7897",
    )
    assert "--no-index" in offline
    assert "--proxy" not in offline


def test_setup_requires_explicit_torch_source_before_creating_venv(monkeypatch):
    monkeypatch.setattr(setup, "_ready", lambda: False)
    assert setup.main(["--torch-index-url", "", "--proxy", ""]) == 2
