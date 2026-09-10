"""Opt-in real model load/generate/unload smoke test."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = [pytest.mark.real_model]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def test_real_model_load_generate_unload(monkeypatch):
    if os.environ.get("QLH_RUN_REAL_MODEL_SMOKE") != "1":
        pytest.skip("设置 QLH_RUN_REAL_MODEL_SMOKE=1 才运行真实模型冒烟测试")

    model_id = os.environ.get("QLH_SMOKE_MODEL_ID", "qwen-1_8b").strip()
    engine = os.environ.get("QLH_SMOKE_ENGINE", "auto").strip() or "auto"
    quant_type = os.environ.get("QLH_SMOKE_QUANT", "int4").strip() or "int4"
    model_path = os.environ.get("QLH_SMOKE_MODEL_PATH", "").strip()
    max_new_tokens = int(os.environ.get("QLH_SMOKE_MAX_NEW_TOKENS", "8"))
    prompt = os.environ.get("QLH_SMOKE_PROMPT", "用一句话回答：1+1等于几？")

    if max_new_tokens < 1 or max_new_tokens > 64:
        pytest.fail("QLH_SMOKE_MAX_NEW_TOKENS 必须在 1 到 64 之间")
    if not prompt:
        pytest.fail("QLH_SMOKE_PROMPT 不能为空")

    from inference_service.engine_host import EngineHost
    from inference_service.protocol import ChatRequest
    import model_config

    # Redirect a built-in slot for a local experiment without changing files
    # or process state outside this isolated, serial test.
    if model_path:
        config = model_config.get_builtin_model(model_id)
        if config is None:
            pytest.fail(
                "QLH_SMOKE_MODEL_PATH 目前只支持内置 model_id；"
                f"未找到 {model_id!r}"
            )
        path = Path(model_path).expanduser()
        if not path.exists():
            pytest.fail(f"QLH_SMOKE_MODEL_PATH 不存在: {path}")
        selected_engine = engine
        if selected_engine == "auto":
            selected_engine = "llama_cpp" if path.suffix.lower() == ".gguf" else "pytorch"
        field = "gguf_path" if selected_engine == "llama_cpp" else "model_path"
        monkeypatch.setattr(config, field, str(path))
        engine = selected_engine

    host = EngineHost()
    loaded = False
    try:
        load_result = host.load_model(
            engine=engine,
            quant_type=quant_type,
            use_compile=_bool_env("QLH_SMOKE_USE_COMPILE"),
            model_id=model_id,
        )
        assert isinstance(load_result, dict)
        assert load_result.get("success", True) is not False
        assert host._llm_is_loaded(), "EngineHost 报告模型未加载"
        loaded = True

        result = host.chat_full(
            ChatRequest(
                message=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                streaming_mode="full",
            )
        )
        assert isinstance(result, dict)
        content = str(result.get("content") or "").strip()
        assert content, "真实模型生成内容为空"
    finally:
        if loaded:
            host.unload_model()
        host.close()


def test_real_model_switch_success_failure_rollback_and_quant_mismatch(tmp_path):
    """AUD-SW-01: exercise the model lifecycle with actual GGUF runtimes."""
    if os.environ.get("QLH_RUN_REAL_MODEL_SMOKE") != "1":
        pytest.skip("set QLH_RUN_REAL_MODEL_SMOKE=1 to run real model acceptance")

    first_model = os.environ.get("QLH_SWITCH_FIRST_MODEL_ID", "qwen2.5-0.5b").strip()
    second_model = os.environ.get("QLH_SWITCH_SECOND_MODEL_ID", "qwen3-0.6b").strip()

    from fastapi import HTTPException
    import api_server
    from model_module import ModelManager

    manager = ModelManager()
    loaded = False
    try:
        manager.load_model(
            model_id=first_model,
            engine="llama_cpp",
            quant_type="int4",
            profile={"tier": "laptop", "gpu": {"cuda_available": True}},
        )
        loaded = True
        assert manager.is_loaded is True
        assert manager.active_model_id == first_model

        switched = manager.switch_model(second_model, engine="llama_cpp", quant_type="int4")
        assert switched["success"] is True
        assert switched["model_id"] == second_model
        assert manager.is_loaded is True
        assert manager.active_model_id == second_model

        active_engine = manager._llama_engine
        unknown = manager.switch_model("aud-sw-unregistered", engine="llama_cpp")
        assert unknown["success"] is False
        assert unknown["error_code"] == "MODEL_NOT_REGISTERED"
        assert unknown["active_model_preserved"] is True
        assert manager._llama_engine is active_engine
        assert manager.active_model_id == second_model

        broken_model = {
            "model_id": "aud-sw-broken-artifact",
            "name": "AUD-SW broken artifact",
            "model_type": "gguf",
            "model_path": "",
            "gguf_path": str(tmp_path / "missing.gguf"),
            "recommended_vram_gb": 1.0,
            "max_context": 1024,
            "quant_types": ["Q4_K_M"],
            "description": "negative acceptance fixture",
        }
        rolled_back = manager.switch_model(
            broken_model["model_id"],
            engine="llama_cpp",
            quant_type="Q4_K_M",
            db_experimental_models=[broken_model],
        )
        assert rolled_back["success"] is False
        assert rolled_back["error_code"] == "MODEL_LOAD_FAILED_ROLLED_BACK"
        assert rolled_back["model_id"] == second_model
        assert manager.is_loaded is True
        assert manager.active_model_id == second_model

        with pytest.raises(HTTPException) as mismatch:
            api_server._normalize_quant_for_engine("Q4_K_M", "pytorch")
        assert mismatch.value.status_code == 400
        assert manager.is_loaded is True
        assert manager.active_model_id == second_model
    finally:
        if loaded and manager.is_loaded:
            manager.unload_model()
