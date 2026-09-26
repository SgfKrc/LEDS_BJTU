"""tests/test_relay_hidden_quant.py — Relay hidden 压缩档（A5）的往返与口径测试。

判据纪律：这里只验**编解码往返**（形状/长度/误差量级/边界）；**上 wire 后的数值结论必须用
per-token argmax 判据**（见 `docs/跨框架接力…`），不得用 cosine 代替。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import relay_hidden_quant as Q  # noqa: E402


def _hidden(tokens: int = 3, width: int = 257) -> np.ndarray:
    """确定性随机 hidden（宽度 257 刻意不是 128 的倍数 ⇒ 覆盖 padding 分支）。"""
    rng = np.random.default_rng(20260923)
    return (rng.standard_normal((tokens, width), dtype=np.float32) * 2.0).astype(np.float32)


@pytest.mark.parametrize("mode", list(Q.HIDDEN_QUANT_MODES))
def test_round_trip_payload_size_matches(mode: str) -> None:
    arr = _hidden()
    tokens, width = arr.shape
    payload = Q.encode_hidden(arr, mode, tokens, width)
    assert len(payload) == Q.expected_quantized_bytes(mode, tokens, width)
    out = np.frombuffer(Q.decode_hidden(payload, mode, tokens, width), dtype="<f4")
    assert out.shape == (tokens * width,)


@pytest.mark.parametrize(
    ("mode", "max_rel"),
    [("none", 0.0), ("f16", 1e-3), ("int8_block128", 2e-2), ("int4_block128", 2e-1)],
)
def test_round_trip_error_is_bounded(mode: str, max_rel: float) -> None:
    """往返误差必须落在该档位的量级内（`none` 必须**逐位相等**）。"""
    arr = _hidden()
    tokens, width = arr.shape
    payload = Q.encode_hidden(arr, mode, tokens, width)
    out = np.frombuffer(Q.decode_hidden(payload, mode, tokens, width),
                        dtype="<f4").reshape(tokens, width)
    if mode == "none":
        assert np.array_equal(out, arr)
        return
    scale = max(float(np.abs(arr).max()), 1e-6)
    assert float(np.abs(out - arr).max()) / scale <= max_rel


@pytest.mark.parametrize("mode", list(Q.HIDDEN_QUANT_MODES))
def test_compressed_modes_are_smaller_than_f32(mode: str) -> None:
    tokens, width = 4, 256
    size = Q.expected_quantized_bytes(mode, tokens, width)
    baseline = tokens * width * 4
    assert size <= baseline
    if mode != "none":
        assert size < baseline


@pytest.mark.parametrize("mode", ["int8_block128", "int4_block128"])
def test_all_zero_block_stays_zero(mode: str) -> None:
    """块内全零（scale 被兜成 1.0）时解码必须仍是 0 —— 不能被 scale 兜底放大成非零。"""
    arr = np.zeros((1, 128), dtype=np.float32)
    payload = Q.encode_hidden(arr, mode, 1, 128)
    out = np.frombuffer(Q.decode_hidden(payload, mode, 1, 128), dtype="<f4")
    assert np.allclose(out, 0.0)


@pytest.mark.parametrize("mode", list(Q.HIDDEN_QUANT_MODES))
def test_tampered_payload_size_is_rejected(mode: str) -> None:
    tokens, width = 2, 128
    payload = Q.encode_hidden(_hidden(tokens, width), mode, tokens, width)
    assert len(payload) > 1
    with pytest.raises(ValueError):
        Q.decode_hidden(payload[:-1], mode, tokens, width)


# ------------------------------------------------------------------ 误差反馈 / 残差补偿（R-R8）
#
# 目的：为 §5.1③ 里「**仍待做**的更激进档（int4 / 2bit）误差反馈」提供**可测基础**。
# 判据纪律不变 —— 下面都是**归因/量化**用的数值断言；真正的验收仍只认 per-token argmax。


def test_roundtrip_f32_matches_encode_decode_pair() -> None:
    """`roundtrip_f32` 必须与 `encode_hidden` + `decode_hidden` **逐位一致**（同一套口径）。"""
    arr = _hidden(3, 257)
    for mode in Q.HIDDEN_QUANT_MODES:
        expected = np.frombuffer(
            Q.decode_hidden(Q.encode_hidden(arr, mode, 3, 257), mode, 3, 257), dtype="<f4"
        ).reshape(3, 257)
        assert np.array_equal(Q.roundtrip_f32(arr, mode, 3, 257), expected)


def test_quant_error_grows_with_aggressive_modes() -> None:
    """误差画像随档位变激进**单调上升**（`none` < `f16` < `int8` < `int4`）—— 供**归因**用。"""
    arr = _hidden(3, 257)
    stats = {mode: Q.quant_error_stats(arr, Q.roundtrip_f32(arr, mode, 3, 257))["rel_rms"]
             for mode in Q.HIDDEN_QUANT_MODES}
    assert stats["none"] == 0.0
    assert stats["f16"] < stats["int8_block128"] < stats["int4_block128"], stats


def test_error_feedback_carries_residual_and_none_has_none() -> None:
    """`none` 档无误差 ⇒ carry 恒为 0；有损档 ⇒ carry 非零（残差真的被带住了）。"""
    arr = _hidden(1, 128)
    _, carry_none = Q.error_feedback_roundtrip(arr, "none", 1, 128)
    assert np.array_equal(carry_none, np.zeros_like(arr))
    _, carry_int8 = Q.error_feedback_roundtrip(arr, "int8_block128", 1, 128)
    assert float(np.abs(carry_int8).max()) > 0.0


def test_error_feedback_carry_is_exactly_the_residual() -> None:
    """★ 定义自洽：`carry' == (x + carry) − roundtrip(x + carry)`。"""
    arr = _hidden(2, 128)
    before = np.full_like(arr, 0.25)
    out, after = Q.error_feedback_roundtrip(arr, "int8_block128", 2, 128, carry=before)
    adjusted = arr + before
    assert np.array_equal(out, Q.roundtrip_f32(adjusted, "int8_block128", 2, 128))
    assert np.array_equal(after, adjusted - out)


def test_error_feedback_bounds_cumulative_error_but_plain_does_not() -> None:
    """★★ 收益的**数学保证**：带补偿时 `Σ(y_i − x_i) = carry_0 − carry_N` ⇒ **有界**；
    无补偿时它是随机游走 ⇒ 随步数增长。

    这条既是"误差反馈有意义"的判据，也是"它**不**改善单步精度"的旁证（单步误差量级不变）。
    """
    rng = np.random.default_rng(20260924)
    width = 128

    def cumulative(steps: int, *, feedback: bool) -> tuple[float, float]:
        carry = None
        total = np.zeros((1, width), dtype=np.float32)
        last_single = 0.0
        for _ in range(steps):
            x = rng.normal(0.0, 1.0, size=(1, width)).astype(np.float32)
            if feedback:
                y, carry = Q.error_feedback_roundtrip(x, "int8_block128", 1, width, carry=carry)
            else:
                y = Q.roundtrip_f32(x, "int8_block128", 1, width)
            total += (y - x)
            last_single = float(np.abs(y - x).max())
        return float(np.abs(total).max()), last_single

    small_gain, small_step = cumulative(8, feedback=True)
    large_gain, large_step = cumulative(64, feedback=True)
    small_plain, _ = cumulative(8, feedback=False)
    large_plain, _ = cumulative(64, feedback=False)

    # 带补偿：累积误差**不随步数增长**（有界）；无补偿：**随步数增长**（随机游走）
    assert large_gain <= small_gain * 3, (small_gain, large_gain)
    assert large_plain > small_plain, (small_plain, large_plain)
    # 单步误差量级不变 ⇒ 印证"不改善单步精度"（int4 那种逐轮失败救不回来）
    assert large_step == pytest.approx(small_step, rel=0.5)


def test_error_feedback_rejects_carry_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        Q.error_feedback_roundtrip(_hidden(1, 128), "int8_block128", 1, 128,
                                   carry=np.zeros((1, 64), dtype=np.float32))


def test_guard_carry_update_must_accumulate() -> None:
    """★「该红必须红」：`carry'` 必须**真的是残差**。

    若有人把 carry 写成"恒为 0"（关掉补偿却仍宣称启用），
    `test_error_feedback_bounds_cumulative_error_but_plain_does_not` 里的「带补偿不增长」
    会退化成随机游走而变红；这里再直接钉一次定义本身。
    """
    arr = _hidden(1, 128)
    out, carry = Q.error_feedback_roundtrip(arr, "int4_block128", 1, 128)
    assert float(np.abs(carry).max()) > 0.0, "int4 的残差不应为 0"
    assert np.allclose(carry, arr - out, atol=1e-6)     # carry 初值为 0 ⇒ carry == x − y


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        Q.expected_quantized_bytes("int2_block128", 1, 128)
    with pytest.raises(ValueError):
        Q.encode_hidden(_hidden(1, 128), "int2_block128", 1, 128)


def test_f16_is_exactly_half_size() -> None:
    assert Q.expected_quantized_bytes("f16", 3, 256) == 3 * 256 * 2
