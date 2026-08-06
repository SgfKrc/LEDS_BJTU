"""Opt-in real model load/generate/unload smoke test."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = [pytest.mark.real_model, pytest.mark.requires_gpu]


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
