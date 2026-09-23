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


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        Q.expected_quantized_bytes("int2_block128", 1, 128)
    with pytest.raises(ValueError):
        Q.encode_hidden(_hidden(1, 128), "int2_block128", 1, 128)


def test_f16_is_exactly_half_size() -> None:
    assert Q.expected_quantized_bytes("f16", 3, 256) == 3 * 256 * 2
