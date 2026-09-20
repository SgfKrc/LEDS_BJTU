"""★ 回归测试：`qwen-1_8b` + int4 加载**不得**因 `_init_weights` 覆盖 uint8 权重而崩溃
（BUG-QWEN18B-BYTE-NORMAL，2026-09-19 修复）。

## 缺陷回顾
transformers 5.17 的 `initialize_weights()` 发生在**加载过程中**，而它打
`_is_hf_initialized` 标记是在**加载后**修复 missing keys 时（`modeling_utils.py:4766-4769`）。
时间差使得 **remote code**（Qwen-1.8B 的 `modeling_qwen.py:666 _init_weights`）被调用，
对 `c_proj.weight` 做 `p.data.normal_(...)` —— 而 int4 量化后 `p.data` 是 **`uint8`**
⇒ CUDA 无该 dtype 的 normal 内核 ⇒ `NotImplementedError: "normal_kernel_cuda"
not implemented for 'Byte'` ⇒ 崩在 `Loading weights 1%` 处（HTTP 500）。

## 修法（`model_module._premark_hf_initialized`）
在 `from_pretrained` 窗口内**预打** `_is_hf_initialized`（与 transformers 官方做法一致），
使 `_init_weights` 不被调入。作用域收窄到「transformers≥5 + TRUST_REMOTE_CODE + int4/int8」。

## 本测试锁定什么
1. **有修复能加载**（且**不是坏模型**——真跑一次推理）;
2. **禁用修复即复现崩溃**（证明修复确实在起作用，而不是「本来就不崩」）;
3. 作用域判断（`_needs_premark` 的等效条件）不误伤其他路径。

⚠️ 需要 CUDA + 本机模型（`models/qwen-1_8b-chat`）+ transformers≥5，否则 skip。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_DIR = ROOT / "models" / "qwen-1_8b-chat"


def _skip_unless_applicable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA（uint8 normal 仅在 CUDA 上不可用；本机修复路径也针对 GPU 量化）")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"需要本机模型（{MODEL_DIR.relative_to(ROOT)}），本机缺失")
    import model_module

    if not model_module._is_transformers_5_or_newer():
        pytest.skip("该缺陷只在 transformers ≥5 出现")


def _load(with_fix: bool):
    """加载 qwen-1_8b + int4；with_fix=False 时临时禁用预标记（对照）。"""
    import config as _cfg
    import model_module

    _cfg.TRUST_REMOTE_CODE = True
    model_module.TRUST_REMOTE_CODE = True

    saved = None
    if not with_fix:
        saved = model_module._premark_hf_initialized

        @contextmanager
        def _noop():
            yield

        model_module._premark_hf_initialized = _noop
    try:
        mgr = model_module.ModelManager()
        mgr.load_model(model_path=str(MODEL_DIR), quant_type="int4", engine="pytorch")
        return mgr
    finally:
        if saved is not None:
            model_module._premark_hf_initialized = saved


@pytest.mark.timeout(600)
def test_load_succeeds_with_fix_and_produces_output():
    """① 有修复 ⇒ 加载成功，且**能真跑出 token**（不是「加载成功的坏模型」）。"""
    _skip_unless_applicable()
    mgr = _load(with_fix=True)
    out = mgr.chat([{"role": "user", "content": "1+1=?"}], max_tokens=8)
    text = out.get("content", "") if isinstance(out, dict) else str(out)
    assert isinstance(text, str) and text.strip(), f"应产出非空文本，实得 {out!r}"


@pytest.mark.timeout(600)
def test_control_reproduces_the_crash():
    """② 禁用修复 ⇒ **复现** `not implemented for 'Byte'`。

    这一条是关键：它证明修复**确实在起作用**，而不是「该缺陷本来就不触发」。
    """
    _skip_unless_applicable()
    with pytest.raises(Exception) as ei:
        _load(with_fix=False)
    msg = str(ei.value)
    assert "not implemented for 'Byte'" in msg or "normal_kernel_cuda" in msg, (
        f"对照应复现 uint8 normal 崩溃，实得: {msg[:200]}")


def test_scope_is_narrow():
    """③ 作用域必须收窄：非量化 / 非 remote-code / 4.x 都不应启用预标记。

    这里以「条件表达式」形式做静态断言，避免真的去加载大模型。
    """
    import model_module

    src = (ROOT / "src" / "model_module.py").read_text(encoding="utf-8")
    assert "_needs_premark" in src, "应有显式的启用条件变量"
    # 三个条件必须同时出现（收窄作用域的凭据）
    for cond in ('TRUST_REMOTE_CODE', '_is_transformers_5_or_newer()',
                 'self.quant_type in ("int4", "int8")'):
        assert cond in src, f"作用域条件缺少: {cond}"
    # 且必须 try/finally 恢复全局（不污染 PreTrainedModel）
    helper = src[src.index("def _premark_hf_initialized"):]
    helper = helper[: helper.index("def _is_transformers_5_or_newer")]
    assert "finally" in helper and "PreTrainedModel.initialize_weights = original" in helper, (
        "预标记必须在 finally 中恢复，不得污染全局 PreTrainedModel"
    )
    assert callable(getattr(model_module, "_premark_hf_initialized", None))
