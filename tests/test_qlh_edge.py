from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.edge_preflight import FORBIDDEN_MODULES, run_preflight
from src import qlh_edge


@pytest.fixture()
def edge_client(monkeypatch):
    monkeypatch.setattr(qlh_edge, "_llm", None)
    monkeypatch.setattr(qlh_edge, "_llm_path", "")
    monkeypatch.delenv("QLH_EDGE_MODEL", raising=False)
    return TestClient(qlh_edge.app)


def test_health_and_status_are_minimal_and_do_not_leak_model_path(edge_client):
    health = edge_client.get("/health")
    status = edge_client.get("/status")

    assert health.status_code == 200
    assert health.json()["edition"] == "edge"
    assert "QLH_EDGE_MODEL" not in health.text
    assert status.status_code == 200
    assert status.json()["runtime_whitelist"] == [
        "llama-cpp-python",
        "fastapi",
        "uvicorn",
        "psutil",
        "httpx",
    ]


def test_generate_requires_a_configured_model(edge_client):
    response = edge_client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "QLH_EDGE_MODEL not configured"


def test_generate_lazily_loads_one_gguf_model(edge_client, monkeypatch, tmp_path):
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"GGUF")
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __call__(self, prompt, **kwargs):
            calls.append(("generate", prompt, kwargs))
            return {
                "choices": [{"text": " world"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setenv("QLH_EDGE_MODEL", str(model_path))

    response = edge_client.post(
        "/generate",
        json={"prompt": "hello", "max_tokens": 8, "temperature": 0.2, "top_p": 0.9},
    )

    assert response.status_code == 200
    assert response.json()["text"] == " world"
    assert calls[0][0] == "init"
    assert calls[1][0] == "generate"
    assert calls[1][1] == "hello"
    assert calls[1][2]["max_tokens"] == 8


def test_edge_source_has_no_forbidden_top_level_imports():
    source = Path(__file__).parents[1] / "src" / "qlh_edge.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint(FORBIDDEN_MODULES)


def test_edge_cli_help_is_available():
    completed = subprocess.run(
        [sys.executable, "qlh_edge.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    assert completed.returncode == 0
    assert "minimal edge inference service" in completed.stdout


def test_development_edge_environment_passes_preflight():
    edge_python = Path.cwd() / ".venv-edge" / "Scripts" / "python.exe"
    if not edge_python.is_file():
        pytest.skip("development .venv-edge is not present")

    result = run_preflight(edge_python)

    assert result["ok"], result
