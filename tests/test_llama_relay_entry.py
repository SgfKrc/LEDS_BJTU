"""★ 主仓下游接力入口（`LlamaCppEngine.forward_layers_from_hidden`）的单元测试。

背景：QLH 的跨框架层接力（D→L / L→L）此前**只有实验脚本**用裸 `llama_cpp` API 实现
（`build/cross-framework-layer-poc/relay_mainrepo_upstream.py`），主仓 `llama_engine` 没有入口。
2026-09-19 在 `src/llama_engine.py` 新增 `forward_layers_from_hidden()`：把上游 hidden 注入
`llama_batch.embd`，只跑本模型（裁层 GGUF）的层、返回末位 logits —— 使「上下游都走主仓引擎」成立。

⚠️ 真模型用例依赖**裁层 GGUF 工件**（在 gitignored 的 `build/` 下）⇒ **缺工件时条件跳过**
（与本仓既有做法一致），但**形状/错误路径**的用例用**离线构造**保证总是能跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CUT_GGUF = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-cut-k12.gguf"


def _engine_or_skip():
    from llama_engine import LlamaCppEngine

    if not CUT_GGUF.is_file():
        pytest.skip(f"需要裁层 GGUF 工件（{CUT_GGUF.relative_to(ROOT)}），本机缺失")
    try:
        eng = LlamaCppEngine()
        eng.load_model(model_path=str(CUT_GGUF), n_ctx=512, n_threads=4)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"llama.cpp 不可用或加载失败: {type(exc).__name__}")
    return eng


class TestNotLoaded:
    """未加载路径：不需要任何模型，任何环境都能跑。"""

    def test_returns_none_when_not_loaded(self):
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        assert eng.is_loaded is False, "新实例不应处于已加载态"
        assert eng.forward_layers_from_hidden([[0.0] * 8], n_past=0) is None

    def test_is_loaded_is_property_not_method(self):
        """⚠️ 实测坑：`is_loaded` 是 **bool 属性**，调用它会 `TypeError: 'bool' object is not callable`。"""
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        assert isinstance(eng.is_loaded, bool)
        with pytest.raises(TypeError):
            eng.is_loaded()  # type: ignore[operator]


class TestWithRealModel:
    """需要裁层 GGUF 的用例（缺工件则整类跳过）。"""

    def test_logits_shape_and_determinism(self):
        import numpy as np

        eng = _engine_or_skip()
        nm = eng._model._model.model  # ⚠️ 原生指针在 `.model`（`. _model` 是 LlamaModel 包装）
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(nm))
        n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(nm)))

        h = (np.random.default_rng(0).standard_normal((1, n_embd)) * 0.02).astype(np.float32)
        lg1 = eng.forward_layers_from_hidden(h, n_past=0)
        assert lg1 is not None
        assert lg1.shape == (n_vocab,), f"logits 形状应为 (n_vocab,)，实得 {lg1.shape}"
        assert np.isfinite(lg1).all(), "logits 不应出现 NaN/Inf"

        # 同输入重跑 ⇒ 逐位一致（接力对数值稳定性有硬要求）
        # ⚠️ 必须先清 KV：本方法直接在 KV 里占位置，同 n_past 重跑会 llama_decode rc=-1
        #    （注意 `reset_kv_cache()` 是既有的 stateless no-op，用不了）
        eng._model._ctx.kv_cache_clear()
        lg2 = eng.forward_layers_from_hidden(h, n_past=0)
        assert np.array_equal(lg1, lg2), "相同输入两次调用结果必须逐位一致"

    def test_rejects_width_mismatch(self):
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(eng._model._model.model))
        bad = np.zeros((1, n_embd // 2), dtype=np.float32)
        with pytest.raises(ValueError, match="n_embd"):
            eng.forward_layers_from_hidden(bad, n_past=0)

    def test_accepts_1d_hidden(self):
        """1D 输入应被当作单 token（`[n_embd]` ⇒ `[1, n_embd]`）。"""
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd_inp(eng._model._model.model))
        lg = eng.forward_layers_from_hidden(np.zeros(n_embd, dtype=np.float32), n_past=0)
        assert lg is not None and lg.ndim == 1

    def test_upstream_entry_returns_hidden(self):
        """★ 上游入口（对称）：裁层 GGUF 跑前 k 层 ⇒ 出末位 hidden（长度 n_embd），且同输入逐位一致。"""
        import numpy as np

        eng = _engine_or_skip()
        import llama_cpp.llama_cpp as M

        n_embd = int(M.llama_model_n_embd(eng._model._model.model))
        ids = [100, 200, 300, 400]
        h1 = eng.forward_layers_to_hidden(ids, n_past=0)
        assert h1 is not None
        assert h1.shape == (n_embd,), f"hidden 形状应为 (n_embd,)，实得 {h1.shape}"
        assert np.isfinite(h1).all()

        # ⚠️ 同 n_past 重跑前须清 KV（reset_kv_cache 是既有 no-op）
        eng._model._ctx.kv_cache_clear()
        h2 = eng.forward_layers_to_hidden(ids, n_past=0)
        assert np.array_equal(h1, h2), "相同输入两次调用结果必须逐位一致"

    def test_upstream_entry_rejects_empty(self):
        eng = _engine_or_skip()
        with pytest.raises(ValueError, match="不能为空"):
            eng.forward_layers_to_hidden([], n_past=0)

    def test_roundtrip_hidden_width_matches_downstream(self):
        """★ 对称性：上游出的 hidden 宽度 == 下游入口接受的宽度（两端可直连）。"""
        eng = _engine_or_skip()
        h = eng.forward_layers_to_hidden([100, 200], n_past=0)
        eng._model._ctx.kv_cache_clear()
        lg = eng.forward_layers_from_hidden(h, n_past=0)
        assert lg is not None, "上游 hidden 应能被下游入口直接接受"
