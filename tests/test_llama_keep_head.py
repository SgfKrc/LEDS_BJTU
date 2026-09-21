"""P2「C 路线」keep-head 上游的守卫用例（缺 shim / 缺工件时条件跳过）。

为什么这些用例重要（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §3）：
pip 绑定的 `embeddings` 通道返回 `output_norm(H)`，**不能**当层接力上游。keep-head
（补丁导出的 layer-input / nextn 通道）是 L 段当上游/中间段的唯一经路。这里固定：

1. shim 缺失/符号不全 ⇒ 明确报 `KeepHeadUnavailable`（绝不静默降级到 embeddings 通道）；
2. shim 在位时：hidden 形状 = `[n_tokens, n_embd]`、有限、**同输入两次调用逐位一致**；
3. 「吃 hidden 吐 hidden」（中间段能力）返回同样形状。

真模型路径需要：带补丁的 `libllama`/shim（`scripts/model_tools/build_keep_head_shim.ps1`）
与 head 裁层工件；缺失时跳过（与本仓既有做法一致）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llama_keep_head import (  # noqa: E402
    MODE_CODES,
    SHIM_SYMBOLS,
    KeepHeadUnavailable,
    KeepHeadUpstream,
    _add_dll_dirs,
)

SHIM = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
HEAD12 = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-head12.gguf"
EXTRA_DLL_DIRS = [d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or
                              r"C:\msys64\ucrt64\bin").split(os.pathsep) if d]
NATIVE_WORKER_ENV = "QLH_KEEP_HEAD_NATIVE_WORKER"


def _upstream_or_skip(**kwargs):
    if os.environ.get(NATIVE_WORKER_ENV) != "1":
        pytest.skip("keep-head native ABI tests run in an isolated subprocess")
    if not SHIM.is_file():
        pytest.skip(f"需要 keep-head shim（{SHIM.relative_to(ROOT)}）；"
                    "用 scripts/model_tools/build_keep_head_shim.ps1 生成")
    if not HEAD12.is_file():
        pytest.skip(f"需要 head 裁层工件（{HEAD12.relative_to(ROOT)}），本机缺失")
    try:
        return KeepHeadUpstream(SHIM, HEAD12, extra_dll_dirs=EXTRA_DLL_DIRS,
                                n_ctx=512, n_threads=4, **kwargs)
    except KeepHeadUnavailable as exc:
        pytest.skip(f"keep-head 不可用：{exc}")


def test_dll_directory_handles_are_retained(monkeypatch, tmp_path):
    dll_dir = tmp_path / "dll"
    dll_dir.mkdir()
    handle = object()
    monkeypatch.setattr(os, "add_dll_directory", lambda path: handle)
    # 环境变量会追加额外目录，测试必须隔离环境（否则断言依赖开发机配置）
    monkeypatch.delenv("QLH_KEEP_HEAD_DLL_DIRS", raising=False)

    dirs, handles = _add_dll_dirs(dll_dir)

    assert dirs == [str(dll_dir)]
    assert handles == [handle]


# ------------------------------------------------------------------ 失败路径（不需要模型）
def test_missing_shim_is_explicit_not_silent(tmp_path):
    with pytest.raises(KeepHeadUnavailable, match="找不到 keep-head shim"):
        KeepHeadUpstream(tmp_path / "nope.dll", tmp_path / "nope.gguf")


def test_missing_model_is_explicit(tmp_path):
    shim = tmp_path / "fake.dll"
    shim.write_bytes(b"")
    with pytest.raises(KeepHeadUnavailable, match="找不到模型"):
        KeepHeadUpstream(shim, tmp_path / "nope.gguf")


def test_invalid_mode_is_rejected():
    with pytest.raises(KeepHeadUnavailable, match="mode"):
        KeepHeadUpstream("whatever.dll", "whatever.gguf", mode="guess")


def test_layer_inp_mode_requires_cut_layer():
    with pytest.raises(KeepHeadUnavailable, match="cut_layer"):
        KeepHeadUpstream("whatever.dll", "whatever.gguf", mode="layer_inp")


def test_shim_symbol_contract_is_declared():
    """shim 的符号清单必须包含中间段能力（`forward_embd`）—— 否则三段链路无从谈起。"""
    assert "qlh_kh_forward_embd" in SHIM_SYMBOLS
    assert set(MODE_CODES) == {"nextn", "layer_inp"}


# ------------------------------------------------------------------ 真模型（缺工件跳过）
def test_nextn_upstream_returns_hidden_with_expected_shape():
    import numpy as np

    with _upstream_or_skip(mode="nextn") as up:
        assert up.n_layer == 12, "head12 工件应是 12 层"
        assert up.n_embd > 0
        hidden = up.forward_tokens_to_hidden([100, 200, 300])
        assert hidden.shape == (3, up.n_embd)
        assert np.isfinite(hidden).all()


def test_same_input_is_bitwise_stable():
    """接力对数值稳定性有硬要求：同输入两次必须逐位一致。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn") as first:
        a = first.forward_tokens_to_hidden([100, 200, 300])
    with _upstream_or_skip(mode="nextn") as second:
        b = second.forward_tokens_to_hidden([100, 200, 300])
    assert np.array_equal(a, b)


def test_middle_segment_can_take_hidden_and_return_hidden():
    """中间段能力：吃 hidden（embd 注入）→ 吐 hidden（nextn 末层输出）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn") as up:
        incoming = np.zeros((2, up.n_embd), dtype=np.float32)
        outgoing = up.forward_hidden_to_hidden(incoming)
        assert outgoing.shape == incoming.shape
        assert np.isfinite(outgoing).all()


def test_rejects_wrong_hidden_width():
    with _upstream_or_skip(mode="nextn") as up:
        with pytest.raises(ValueError, match="hidden 形状"):
            up.forward_hidden_to_hidden([[0.0] * max(1, up.n_embd // 2)])


# --------------------------------------------------- P3：多序列数据流契约
def test_token_list_validates_length_without_a_model():
    """`seq_ids` / `positions` 与 n_tokens 必须等长（纯校验，不需要模型）。"""
    assert KeepHeadUpstream._token_list(None, 3, "seq_ids") is None
    assert KeepHeadUpstream._token_list([0, 1, 2], 3, "seq_ids") == [0, 1, 2]
    with pytest.raises(ValueError, match="长度 2 != n_tokens 3"):
        KeepHeadUpstream._token_list([0, 1], 3, "positions")


def test_multi_sequence_binding_is_accepted():
    """P3：2 序列 × 2 token 的显式绑定（seq_ids + positions）必须与单序列同形返回。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=2) as up:
        hidden = np.zeros((4, up.n_embd), dtype=np.float32)
        out = up.forward_hidden_to_hidden(hidden, seq_ids=[0, 0, 1, 1],
                                          positions=[0, 1, 0, 1])
        assert out.shape == hidden.shape
        assert np.isfinite(out).all()


def test_seq_id_beyond_n_seq_max_is_rejected():
    """seq_id ≥ n_seq_max 必须 fail-loud（否则 llama.cpp 直接 rc=-1，错误难定位）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=1) as up:
        hidden = np.zeros((2, up.n_embd), dtype=np.float32)
        with pytest.raises(ValueError, match="n_seq_max"):
            up.forward_hidden_to_hidden(hidden, seq_ids=[0, 3], positions=[0, 0])


def test_single_sequence_positions_stay_consecutive_across_steps():
    """★ 回归：单序列增量必须延续位置（曾因多序列改造丢掉 n_past 而第二步 decode 失败）。"""
    import numpy as np

    with _upstream_or_skip(mode="nextn", n_seq_max=1) as up:
        first = up.forward_hidden_to_hidden(np.zeros((2, up.n_embd), dtype=np.float32),
                                            n_past=0)
        second = up.forward_hidden_to_hidden(np.zeros((1, up.n_embd), dtype=np.float32),
                                             n_past=2)
        assert first.shape[0] == 2 and second.shape[0] == 1


# --------------------------------------------------- P3：hidden 压缩字节口径
def test_hidden_quant_bytes_accounting():
    """压缩档位的**每 token 有效线上字节**（int8 块量化含每 128 维 1 个 f32 scale）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "relay_experiment_cli", ROOT / "scripts" / "relay_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._hidden_bytes(896, "none") == 896 * 4
    assert module._hidden_bytes(896, "f16") == 896 * 2
    assert module._hidden_bytes(896, "int8_block128") == 896 + 7 * 4   # 7 个 128 块
    assert module._hidden_bytes(0, "f16") is None


def test_int8_hidden_quant_supports_non_block_aligned_width():
    import importlib.util
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "relay_experiment_cli_non_aligned", ROOT / "scripts" / "relay_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    hidden = np.arange(3 * 130, dtype=np.float32).reshape(3, 130) - 100.0
    out = module._quantize_hidden(hidden, "int8_block128")
    assert out.shape == hidden.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
