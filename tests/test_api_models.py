import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import api_server
import model_config as mc
from model_module import ModelManager


def _both_model(tmp_path, model_id="custom-both"):
    hf_dir = tmp_path / f"{model_id}-hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    gguf = tmp_path / f"{model_id}.Q4_K_M.gguf"
    gguf.write_text("x", encoding="utf-8")
    return mc.ModelConfig(
        model_id=model_id,
        name="Custom Both",
        model_type="both",
        model_path=str(hf_dir),
        gguf_path=str(gguf),
        is_experimental=True,
        quant_types=["fp16", "int4", "Q4_K_M"],
    )


class _TemplateTokenizer:
    def __init__(self, template_name):
        self.template_name = template_name

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        user = next(m["content"] for m in messages if m["role"] == "user")
        if self.template_name == "deepseek":
            return f"<｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜><think>\n"
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def test_build_model_chat_prompt_uses_deepseek_native_template():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("deepseek"),
        [{"role": "user", "content": "Ubuntu和Debian有什么区别？"}],
    )

    assert "<｜User｜>Ubuntu和Debian有什么区别？<｜Assistant｜><think>" in prompt
    assert "<|im_start|>" not in prompt


def test_build_model_chat_prompt_keeps_qwen_chatml_template():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("qwen"),
        [{"role": "user", "content": "Ubuntu和Debian有什么区别？"}],
    )

    assert "<|im_start|>user" in prompt
    assert "<｜User｜>" not in prompt


def test_build_model_chat_prompt_does_not_mix_prefill_with_native_think():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("deepseek"),
        [{"role": "user", "content": "test"}],
        assistant_prefill="【思考】\n",
    )

    assert prompt.endswith("<think>\n")
    assert "【思考】" not in prompt


def test_model_payload_prefers_pytorch_when_cuda_has_both_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    payload = api_server._model_api_payload(_both_model(tmp_path))

    assert payload["supported_engines"] == ["llama_cpp", "pytorch"]
    assert payload["preferred_engine"] == "pytorch"
    assert payload["default_quant_type"] == "int4"


def test_model_payload_prefers_pytorch_without_cuda_when_safetensors_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    payload = api_server._model_api_payload(_both_model(tmp_path))

    assert payload["supported_engines"] == ["llama_cpp", "pytorch"]
    assert payload["preferred_engine"] == "pytorch"
    assert payload["default_quant_type"] == "int4"


def test_available_models_scans_all_registered_model_formats(tmp_path, monkeypatch):
    cfg = _both_model(tmp_path)
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [cfg])
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(api_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api_server, "model_loaded", False)

    result = asyncio.run(api_server.list_available_models())
    engine_ids = {engine["id"] for engine in result["available_engines"]}

    assert engine_ids == {"llama_cpp", "pytorch"}
    assert result["current"] is None
    assert result["current_engine"] is None


class TestChatMlStopHandling:
    """Regression tests for ChatML leakage in local PyTorch generation."""

    class _FakeTokenizer:
        eos_token_id = 2
        unk_token_id = 0

        def __init__(self, decoded=""):
            self.decoded = decoded

        def convert_tokens_to_ids(self, token):
            return {
                "<|im_end|>": 10,
                "<|im_start|>": 11,
            }.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            mapping = {
                "<|im_end|>": [10],
                "<|im_start|>": [11],
                "<not-a-real-token>": [self.unk_token_id],
            }
            return mapping.get(text, [99])

        def decode(self, ids, skip_special_tokens=False):
            return self.decoded

    def test_decode_generated_ids_cuts_at_chatml_stop(self):
        mgr = ModelManager()
        mgr.tokenizer = self._FakeTokenizer(
            "Ubuntu 和 Debian 的区别如下。<|im_end|><|im_start|>user\n下一轮"
        )

        text = mgr._decode_generated_ids([1, 2, 3], mgr._merge_stop_sequences(None))

        assert text == "Ubuntu 和 Debian 的区别如下。"

    def test_unknown_stop_token_is_not_used_as_eos_id(self):
        mgr = ModelManager()
        mgr.tokenizer = self._FakeTokenizer()

        eos_ids = mgr._get_generation_eos_token_ids(["<not-a-real-token>"])

        assert eos_ids == 2

    def test_stream_stop_filter_preserves_chunk_boundary_spaces(self):
        mgr = ModelManager()
        chunks = ["Hello", " world", "<|im_end|>", "ignored"]

        text = "".join(mgr._iter_stream_until_stop(chunks, mgr._merge_stop_sequences(None)))

        assert text == "Hello world"


def test_switch_model_accepts_external_model_path_not_in_builtin_registry(monkeypatch):
    calls = []

    def fake_load_model(self, **kwargs):
        calls.append(kwargs)
        self._active_model_id = kwargs["model_id"]
        self._engine_type = kwargs["engine"]
        self._model_path = kwargs["model_path"]

    monkeypatch.setattr(ModelManager, "load_model", fake_load_model)

    mgr = ModelManager()
    result = mgr.switch_model(
        "db-custom-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
        model_path="models/custom.Q4_K_M.gguf",
    )

    assert result["success"] is True
    assert result["model_id"] == "db-custom-gguf"
    assert calls[0]["model_path"] == "models/custom.Q4_K_M.gguf"


def test_api_switch_model_resolves_db_registered_gguf(tmp_path, monkeypatch):
    gguf = tmp_path / "custom.Q4_K_M.gguf"
    gguf.write_text("x", encoding="utf-8")
    db_entry = {
        "model_id": "db-custom-gguf",
        "name": "DB Custom GGUF",
        "model_type": "gguf",
        "model_path": "",
        "gguf_path": str(gguf),
        "recommended_vram_gb": 4.0,
        "max_context": 4096,
        "quant_types": ["Q4_K_M"],
        "description": "test",
    }
    calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False

        def switch_model(self, **kwargs):
            calls.append(kwargs)
            self.active_model_id = kwargs["model_id"]
            self.is_loaded = True
            return {
                "success": True,
                "model_id": kwargs["model_id"],
                "model_name": "DB Custom GGUF",
                "error": None,
            }

    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [db_entry])
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="db-custom-gguf",
        quant_type="Q4_K_M",
        engine="auto",
    )
    result = asyncio.run(api_server.switch_model(req))

    assert result["success"] is True
    assert calls[0]["engine"] == "llama_cpp"
    assert calls[0]["model_path"] == str(gguf)


def test_model_payload_marks_builtin_slots(monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    builtin_payload = api_server._model_api_payload(
        mc.get_builtin_model("deepseek-r1-distill-qwen-7b")
    )
    custom_payload = api_server._model_api_payload(
        mc.ModelConfig(
            model_id="custom-gguf",
            name="Custom GGUF",
            model_type="gguf",
            gguf_path="models/custom.gguf",
            is_experimental=True,
        )
    )

    assert builtin_payload["is_builtin"] is True
    assert custom_payload["is_builtin"] is False


def test_register_gguf_model_is_allowed_without_cuda(monkeypatch):
    saved = []
    fake_db = types.SimpleNamespace(
        save_experimental_model=lambda model_id, config_json: saved.append((model_id, config_json)) or True
    )

    monkeypatch.setitem(sys.modules, "db", fake_db)
    monkeypatch.setattr(api_server, "_db_available", True)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    req = api_server.RegisterModelRequest(
        model_id="custom-gguf",
        name="Custom GGUF",
        model_type="gguf",
        gguf_path="models/custom.Q4_K_M.gguf",
    )

    result = asyncio.run(api_server.register_model(req))

    assert result == {"status": "registered", "model_id": "custom-gguf"}
    assert saved and saved[0][0] == "custom-gguf"


def test_register_safetensors_model_without_cuda_is_allowed(monkeypatch):
    saved = []
    fake_db = types.SimpleNamespace(
        save_experimental_model=lambda model_id, config_json: saved.append((model_id, config_json)) or True
    )

    monkeypatch.setitem(sys.modules, "db", fake_db)
    monkeypatch.setattr(api_server, "_db_available", True)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    req = api_server.RegisterModelRequest(
        model_id="custom-hf",
        name="Custom HF",
        model_type="safetensors",
        model_path="models/custom-hf",
    )

    result = asyncio.run(api_server.register_model(req))

    assert result == {"status": "registered", "model_id": "custom-hf"}
    assert saved and saved[0][0] == "custom-hf"


# ================================================================
# /api/models/load 端点测试
# ================================================================

def test_load_model_rejects_invalid_engine(monkeypatch):
    """无效 engine 参数 → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.LoadModelRequest(
        engine="invalid_engine",
        quant_type="int4",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 400


def test_load_model_rejects_invalid_quant_for_pytorch(monkeypatch):
    """PyTorch 引擎 + 无效 quant → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.LoadModelRequest(
        engine="pytorch",
        quant_type="q4_k_m",  # GGUF 量化，PyTorch 不支持
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 400


def test_load_model_allows_any_quant_for_llama_cpp(monkeypatch):
    """llama_cpp 引擎放行任意 quant_type（GGUF 自带量化）"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    # 让验证通过（model_id=None 时 _validate_model_load_request 直接 return）
    req = api_server.LoadModelRequest(
        engine="llama_cpp",
        quant_type="Q4_K_M",
    )
    # 不应因 quant_type 被拒绝（llama_cpp 分支放行）
    # 但会因模型未加载等原因失败 — 这里只验证不抛 400
    try:
        asyncio.run(api_server.load_model(req))
    except api_server.HTTPException as e:
        # 可能是 500（模型加载失败）或其他，但不应该是 400
        assert e.status_code != 400, f"llama_cpp 不应拒绝 Q4_K_M quant，但返回了 400: {e.detail}"


def test_load_model_auto_allows_gguf_quant_when_effective_engine_is_llama_cpp(monkeypatch):
    """engine=auto 解析到 llama.cpp 时，GGUF 量化占位值应合法。"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = "llama_cpp"
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self.quant_type = kwargs["quant_type"]
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "GGUF Model",
                "error": None,
            }

    async def fake_get_status():
        return {"status": "ok"}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "_validate_model_load_request", lambda *a, **kw: None)
    monkeypatch.setattr(api_server, "_resolve_model_path_for_engine", lambda *a, **kw: "models/test.gguf")
    monkeypatch.setattr(api_server, "_effective_engine_for_model", lambda *a, **kw: "llama_cpp")
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server, "get_status", fake_get_status)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="Q4_K_M",
        model_id="gguf-model",
    )
    result = asyncio.run(api_server.load_model(req))

    assert result["status"] == "ok"
    assert switch_calls[0]["engine"] == "llama_cpp"
    assert switch_calls[0]["quant_type"] == "Q4_K_M"


def test_load_model_rejects_nonexistent_model_id(monkeypatch):
    """不存在的 model_id → 404"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [])
    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
        model_id="nonexistent-model-xyz",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 404


def test_load_model_uses_switch_model_internally(monkeypatch):
    """验证 /api/models/load 使用 switch_model（B2 修复）"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs.get("model_id", "")
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            self.quant_type = kwargs.get("quant_type")
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)

    async def fake_get_status():
        return {"status": "ok"}

    monkeypatch.setattr(api_server, "get_status", fake_get_status)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
    )
    result = asyncio.run(api_server.load_model(req))
    assert len(switch_calls) == 1, "应调用 switch_model 而非手动 unload/load"
    assert switch_calls[0]["model_id"] == api_server.mc.DEFAULT_MODEL_ID
    assert result["status"] == "ok"


def test_load_model_rollback_on_failure(monkeypatch):
    """加载失败时 switch_model 应触发回滚，/api/models/load 应正确报告"""
    class FakeManager:
        active_model_id = "previous-model"
        is_loaded = True
        _engine_type = "llama_cpp"
        quant_type = "Q4_K_M"
        model = None  # 供 _init_kv_cache 检查

        def switch_model(self, **kwargs):
            # 模拟: 卸载旧模型成功，加载新模型失败，回滚成功
            return {
                "success": False,
                "model_id": "previous-model",
                "model_name": "Previous Model",
                "error": "模型 'new-model' 加载失败: 模拟错误。已回滚到 'Previous Model'。",
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", True)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    # 绕过 _validate_model_load_request（model_id "new-model" 不在注册表中）
    monkeypatch.setattr(api_server, "_validate_model_load_request", lambda *a, **kw: None)
    # 绕过 _init_kv_cache（不需要真实模型）
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
        model_id="new-model",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 500
    assert "已回滚" in exc.value.detail


# ================================================================
# /api/models 端点测试
# ================================================================

def test_list_models_includes_active_model_id(monkeypatch):
    """GET /api/models 应返回 active_model_id"""
    class FakeManager:
        active_model_id = "qwen-1_8b"

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", True)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    result = asyncio.run(api_server.list_models())
    assert result["active_model_id"] == "qwen-1_8b"
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0

    # 默认模型应在列表中
    model_ids = [m["model_id"] for m in result["models"]]
    assert "qwen-1_8b" in model_ids


def test_list_models_returns_null_when_not_loaded(monkeypatch):
    """模型未加载时 active_model_id 为 None"""
    class FakeManager:
        active_model_id = ""

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    result = asyncio.run(api_server.list_models())
    assert result["active_model_id"] is None


# ================================================================
# /api/models/current 端点测试
# ================================================================

def test_current_model_when_not_loaded(monkeypatch):
    """模型未加载时 → loaded=False"""
    monkeypatch.setattr(api_server, "model_loaded", False)
    result = asyncio.run(api_server.get_current_model())
    assert result["loaded"] is False
    assert result["model_id"] is None


def test_current_model_when_loaded(monkeypatch):
    """模型已加载时 → 返回完整信息"""
    class FakeManager:
        active_model_id = "qwen-1_8b"
        _engine_type = "pytorch"

        def get_model_info(self):
            return {
                "model_id": "qwen-1_8b",
                "engine": "pytorch",
                "model_name": "Qwen-1.8B-Chat",
                "model_path": "/fake/path",
                "total_params": "1.8B",
                "device": "cuda:0",
            }

        def get_memory_usage(self):
            return {"gpu_allocated_gb": 1.8, "gpu_reserved_gb": 2.0}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", True)
    monkeypatch.setattr(api_server, "current_quant", "int4")

    result = asyncio.run(api_server.get_current_model())
    assert result["loaded"] is True
    assert result["model_id"] == "qwen-1_8b"
    assert result["engine"] == "pytorch"
    assert result["model_name"] == "Qwen-1.8B-Chat"


# ================================================================
# /api/models/switch 端点测试
# ================================================================

def test_switch_model_rejects_invalid_engine(monkeypatch):
    """无效 engine → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.SwitchModelRequest(
        model_id="qwen-1_8b",
        engine="invalid",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.switch_model(req))
    assert exc.value.status_code == 400


def test_switch_model_calls_manager_switch(monkeypatch):
    """正常切换 → 调用 model_manager.switch_model 并更新全局状态"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    # 绕过 _validate_model_load_request（GGUF 文件实际不存在）
    monkeypatch.setattr(api_server, "_validate_model_load_request", lambda *a, **kw: None)
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="qwen2.5-7b-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
    )
    result = asyncio.run(api_server.switch_model(req))
    assert result["success"] is True
    assert len(switch_calls) == 1
    assert switch_calls[0]["model_id"] == "qwen2.5-7b-gguf"
    assert switch_calls[0]["engine"] == "llama_cpp"


def test_switch_model_rejects_gguf_quant_when_effective_engine_is_pytorch(monkeypatch):
    """engine=auto 解析到 PyTorch 时，不能把 Q4_K_M 传进 PyTorch 加载路径。"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = "pytorch"
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            return {"success": True, "model_id": kwargs["model_id"], "model_name": "bad", "error": None}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "_validate_model_load_request", lambda *a, **kw: None)
    monkeypatch.setattr(api_server, "_resolve_model_path_for_engine", lambda *a, **kw: "models/test-hf")
    monkeypatch.setattr(api_server, "_effective_engine_for_model", lambda *a, **kw: "pytorch")

    req = api_server.SwitchModelRequest(
        model_id="hf-model",
        quant_type="Q4_K_M",
        engine="auto",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.switch_model(req))

    assert exc.value.status_code == 400
    assert switch_calls == []


def test_switch_model_resets_runtime_conversation_state(monkeypatch):
    switch_calls = []
    init_calls = []
    cleared = []

    class FakeKvCache:
        def clear(self):
            cleared.append(True)

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", True)
    monkeypatch.setattr(api_server, "kv_cache", FakeKvCache())
    monkeypatch.setattr(
        api_server,
        "conversation_stats",
        {
            "total_prompt_tokens": 10,
            "total_generated_tokens": 20,
            "total_time_seconds": 3.5,
            "rounds": 2,
        },
    )
    monkeypatch.setattr(
        api_server,
        "session_histories",
        {"default": [{"role": "assistant", "content": "DeepSeek-R1 intro"}]},
    )
    monkeypatch.setattr(api_server, "_validate_model_load_request", lambda *a, **kw: None)
    monkeypatch.setattr(api_server, "_resolve_model_path_for_engine", lambda *a, **kw: "models/test.gguf")
    monkeypatch.setattr(api_server, "_effective_engine_for_model", lambda *a, **kw: "llama_cpp")
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: init_calls.append(True))

    req = api_server.SwitchModelRequest(
        model_id="qwen2.5-7b-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
    )
    result = asyncio.run(api_server.switch_model(req))

    assert result["success"] is True
    assert len(switch_calls) == 1
    assert cleared == [True]
    assert api_server.kv_cache is None
    assert api_server.session_histories == {}
    assert api_server.conversation_stats == {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    assert init_calls == [True]


# ================================================================
# 模型注册表 DB 集成测试
# ================================================================

def test_register_model_passes_db_experimental_models_to_switch(monkeypatch, tmp_path):
    """注册的 DB 模型应可通过 switch_model 端点加载（B1 修复验证）"""
    gguf_file = tmp_path / "registered-model.Q4_K_M.gguf"
    gguf_file.write_text("x", encoding="utf-8")

    db_entry = {
        "model_id": "registered-gguf-model",
        "name": "Registered GGUF Model",
        "model_type": "gguf",
        "model_path": "",
        "gguf_path": str(gguf_file),
        "recommended_vram_gb": 4.0,
        "max_context": 4096,
        "quant_types": ["Q4_K_M"],
        "description": "test",
    }

    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Registered GGUF Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [db_entry])
    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="registered-gguf-model",
        quant_type="Q4_K_M",
        engine="auto",
    )
    result = asyncio.run(api_server.switch_model(req))
    assert result["success"] is True
    assert switch_calls[0]["model_id"] == "registered-gguf-model"
    # 应传递 db_experimental_models（B1 修复验证）
    assert "db_experimental_models" in switch_calls[0]
    assert switch_calls[0]["db_experimental_models"] == [db_entry]


def test_list_model_registry_returns_db_models(monkeypatch):
    """GET /api/models/registry → 返回 DB 注册模型列表"""
    db_models = [
        {"model_id": "m1", "name": "Model 1", "model_type": "gguf"},
        {"model_id": "m2", "name": "Model 2", "model_type": "safetensors"},
    ]
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: db_models)
    result = asyncio.run(api_server.list_model_registry())
    assert result["models"] == db_models


# ================================================================
# _strip_native_thinking_tags 测试（P3修复: DeepSeek-R1 思考标记剥离）
# ================================================================

class TestStripNativeThinkingTags:
    """测试 _strip_native_thinking_tags 对各种思考标记格式的处理"""

    # 使用 chr() 构造 XML 标签，避免测试框架剥离 <xxx> 格式的文本
    LT = chr(60)  # <
    GT = chr(62)  # >

    def test_strips_think_block_with_response_marker(self):
        """  ...  块 +  标记 → 只保留最终回答"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}thinking process here...{L}/think{G}\n{L}response{G}Hello! I'm DeepSeek-R1.{L}/response{G}{L}|im_end|{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "thinking process" not in result
        assert "Hello! I'm DeepSeek-R1." in result
        assert "<|im_end|>" not in result

    def test_strips_think_block_without_response_marker(self):
        """  ...  块（无  标记）→ 回答直接跟在 </think> 后"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}Let me think about this...{L}/think{G}\nThe answer is 42.{L}|im_end|{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "Let me think about this" not in result
        assert "The answer is 42." in result

    def test_strips_leading_reasoning_before_closing_think(self):
        """Native DeepSeek completion may contain only the closing think tag."""
        L, G = self.LT, self.GT
        text = f"reasoning leaked from prompt prefill{L}/think{G}\nFinal answer."
        result = api_server._strip_native_thinking_tags(text)
        assert result == "Final answer."

    def test_strips_standalone_think_tags(self):
        """独立的  和 </think> 标记 → 剥离"""
        L, G = self.LT, self.GT
        # 成对的  块：内容也被移除
        text = f"Here is a response with {L}think{G}think block here.{L}/think{G} around."
        result = api_server._strip_native_thinking_tags(text)
        # 块内内容被移除
        assert "think block here" not in result
        # 普通文字应保留
        assert "Here is a response with" in result
        assert "around." in result

    def test_strips_im_end_tokens(self):
        """<|im_end|> 和 <|im_start|> token → 剥离"""
        text = "Hello world<|im_end|><|im_start|>assistant\n"
        result = api_server._strip_native_thinking_tags(text)
        assert "<|im_end|>" not in result
        assert "<|im_start|>" not in result
        assert "Hello world" in result

    def test_handles_empty_text(self):
        """空文本 → 返回空文本"""
        assert api_server._strip_native_thinking_tags("") == ""
        assert api_server._strip_native_thinking_tags(None) is None

    def test_clean_text_passes_through(self):
        """无思考标记的普通文本 → 原样返回"""
        text = "这是一个普通的回答，没有任何思考标记。"
        result = api_server._strip_native_thinking_tags(text)
        assert result == text

    def test_strips_qwen3_answer_tags(self):
        """Qwen3  格式 → 剥离思考和 answer 标签"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}some reasoning{L}/think{G}\n{L}answer{G}Here is the final response{L}/answer{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "some reasoning" not in result
        assert "Here is the final response" in result

    def test_strips_real_deepseek_r1_output(self):
        """模拟 DeepSeek-R1 真实输出: 自我介绍在  块中泄漏"""
        L, G = self.LT, self.GT
        text = (
            f"{L}think{G}\n"
            "您好！我是由中国的深度求索（DeepSeek）公司独立开发的智能助手DeepSeek-R1，"
            "有关模型和产品的详细内容请参考官方文档。\n"
            f"{L}/think{G}\n"
            f"{L}response{G}\n"
            "关于您的问题，我的回答是：这是正确的答案。"
        )
        result = api_server._strip_native_thinking_tags(text)
        # 思考块中的内容应被完全移除
        assert "深度求索" not in result
        assert "DeepSeek-R1" not in result
        assert "官方文档" not in result
        # 回答内容应保留
        assert "关于您的问题" in result
        assert "这是正确的答案" in result

    def test_preserves_response_content_formatting(self):
        """保留回答内容的换行和格式"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}reasoning here...{L}/think{G}\n{L}response{G}第一行\n第二行\n\n第三行"
        result = api_server._strip_native_thinking_tags(text)
        assert "reasoning here" not in result
        assert "第一行" in result
        assert "第二行" in result
        assert "第三行" in result


class TestParseThinkingResponseNative:
    """测试 _parse_thinking_response 对 DeepSeek-R1 本地格式的解析"""

    def test_parse_closing_only_native_think_format(self):
        """Completion starting inside native <think> should split at </think>."""
        L, G = chr(60), chr(62)
        text = f"reasoning from native template{L}/think{G}\nFinal answer."
        answer, thinking = api_server._parse_thinking_response(text)

        assert answer == "Final answer."
        assert thinking == "reasoning from native template"

    def test_parse_native_think_removes_legacy_prefix_if_present(self):
        """Native parser should not leak the legacy thinking prefill."""
        L, G = chr(60), chr(62)
        text = f"【思考】\nreasoning from native template{L}/think{G}\nFinal answer."
        answer, thinking = api_server._parse_thinking_response(text)

        assert answer == "Final answer."
        assert thinking == "reasoning from native template"

    def test_parse_native_think_format_extracts_thinking(self):
        """  ...  格式 → 正确提取思考和答案"""
        text = "【思考】\nthinking内容...\nresponse这是实际回答"
        # _parse_thinking_response 应先匹配【思考】，找不到【思考结束】则尝试本地格式
        answer, thinking = api_server._parse_thinking_response(text)
        # 由于【思考】标记存在但没有【思考结束】，fallthrough 到情况2
        # 情况2 调用 _strip_native_thinking_tags 清理
        assert answer is not None
        assert "这是实际回答" in answer

    def test_parse_fallback_strips_native_tags(self):
        """格式不匹配 → 回退清理本地标记"""
        text = "【思考】\nsome text\nNo end marker here..."
        answer, thinking = api_server._parse_thinking_response(text)
        assert answer is not None
        # 清理后不应有【思考】标记
        assert "【思考】" not in answer
