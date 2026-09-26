"""relay_hidden_quant.py — Relay `hidden` 的**线上压缩档**（A5）。

背景：此前 hidden 在 wire 上**固定 f32**（`relay_transport.RELAY_DTYPE = "float32_le"`），量化往返只
存在于实验驱动（`scripts/relay_experiment.py` 的 `_quantize_hidden`，**只做本地模拟、不改链路**）。
本模块把同一套算法做成**可上 wire 的编解码**，让跨机传输真的变小：

| 档位 | 布局（每 128 元素一块） | 相对 f32 |
| --- | --- | --- |
| `none` | 原样 f32 | 1.00× |
| `f16` | 半精度 | 0.50× |
| `int8_block128` | `<scale:f32>` + 128 × int8 | ~0.33× |
| `int4_block128` | `<scale:f32>` + 64 B（128 × 4 bit） | ~0.19× |

`int4` 刻意只用到 **−7..7**（对称、无额外偏置，便于把误差变化归因到位宽本身）——与实验驱动的口径一致。

⚠️ 纪律（与 `docs/跨框架接力…` 一致）：启用压缩**必须**用 **per-token argmax** 判据复验，
**不得用 cosine 代替**；分叉就如实标 FAIL。帧头的表达见 `relay_transport.RELAY_QUANT_CODES`
（`flags` 低 3 位；旧对端看到非零 flags 直接拒 ⇒ **天然 fail-closed**、向后兼容）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HIDDEN_QUANT_MODES",
    "QUANT_BLOCK",
    "decode_hidden",
    "encode_hidden",
    "error_feedback_roundtrip",
    "expected_quantized_bytes",
    "quant_error_stats",
    "roundtrip_f32",
]

#: 块大小（与实验驱动 `relay_experiment._quantize_hidden` 的 128 一致）。
QUANT_BLOCK = 128

#: 允许上 wire 的档位（顺序即 `relay_transport.RELAY_QUANT_CODES` 的编码顺序）。
HIDDEN_QUANT_MODES = ("none", "f16", "int8_block128", "int4_block128")


def _as_f32(hidden, n_tokens: int, n_embd: int) -> np.ndarray:
    """把 f32 bytes 或 ndarray 规范成 `[n_tokens, n_embd]` 的 f32 数组（形状不符即抛）。"""
    tokens, width = int(n_tokens), int(n_embd)
    if tokens < 1 or width < 1:
        raise ValueError("invalid_hidden_shape")
    if isinstance(hidden, (bytes, bytearray, memoryview)):
        raw = bytes(hidden)
        if len(raw) != tokens * width * 4:
            raise ValueError("hidden_payload_size_mismatch")
        return np.frombuffer(raw, dtype="<f4").reshape(tokens, width).astype(np.float32)
    arr = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
    if arr.shape != (tokens, width):
        raise ValueError("invalid_hidden_shape")
    return arr


def _blocks(width: int) -> int:
    return (int(width) + QUANT_BLOCK - 1) // QUANT_BLOCK


def expected_quantized_bytes(mode: str, n_tokens: int, n_embd: int) -> int:
    """该档位下 payload 的字节数（`none` ⇒ 与 f32 一致）。"""
    tokens, width = int(n_tokens), int(n_embd)
    if tokens < 1 or width < 1:
        raise ValueError("invalid_hidden_shape")
    if mode == "none":
        return tokens * width * 4
    if mode == "f16":
        return tokens * width * 2
    blocks = _blocks(width)
    if mode == "int8_block128":
        return tokens * blocks * (4 + QUANT_BLOCK)
    if mode == "int4_block128":
        return tokens * blocks * (4 + QUANT_BLOCK // 2)
    raise ValueError(f"unknown hidden quant mode: {mode!r}")


def encode_hidden(hidden, mode: str, n_tokens: int, n_embd: int) -> bytes:
    """把 f32 hidden 压成该档位的 wire 字节（`none` ⇒ 原样返 f32 bytes）。"""
    if mode == "none":
        # 仍走一遍规范化 ⇒ 形状/长度不符在这里就抛，且保证是 f32 little-endian。
        return np.ascontiguousarray(_as_f32(hidden, n_tokens, n_embd), dtype="<f4").tobytes()
    arr = _as_f32(hidden, n_tokens, n_embd)
    if mode == "f16":
        return arr.astype("<f2").tobytes()
    if mode not in ("int8_block128", "int4_block128"):
        raise ValueError(f"unknown hidden quant mode: {mode!r}")

    tokens, width = arr.shape
    blocks = _blocks(width)
    padded_width = blocks * QUANT_BLOCK
    if padded_width != width:
        padded = np.zeros((tokens, padded_width), dtype=np.float32)
        padded[:, :width] = arr
    else:
        padded = arr
    flat = padded.reshape(tokens, blocks, QUANT_BLOCK)
    scale = np.abs(flat).max(axis=2, keepdims=True)
    scale[scale == 0] = 1.0
    levels = 127.0 if mode == "int8_block128" else 7.0
    quantized = np.round(flat / scale * levels).clip(-levels, levels)

    # ⚠️ 先 `view` 再 `reshape`：f32 元素是 4 字节，(tokens, blocks, 1) 的 f32 视图按字节看正好是
    #    (tokens, blocks, 4)；反过来的 reshape 会得到 4 倍长度（踩过）。
    scale_bytes = scale.astype("<f4").view(np.uint8).reshape(tokens, blocks, 4)
    if mode == "int8_block128":
        body = quantized.astype("<i1").view(np.uint8).reshape(tokens, blocks, QUANT_BLOCK)
    else:
        # 4 bit：有符号 −7..7 ⇒ 存无符号 nibble（+8），相邻两个 nibble 打包成一字节。
        nibbles = (quantized.astype(np.int16) + 8).astype(np.uint8)
        low = nibbles[:, :, 0::2]
        high = nibbles[:, :, 1::2]
        body = (low | (high << 4)).astype(np.uint8).reshape(tokens, blocks, QUANT_BLOCK // 2)

    out = np.empty((tokens, blocks, 4 + body.shape[2]), dtype=np.uint8)
    out[:, :, :4] = scale_bytes
    out[:, :, 4:] = body
    return out.tobytes()


def decode_hidden(payload: bytes, mode: str, n_tokens: int, n_embd: int) -> bytes:
    """把该档位的 wire 字节解回 **f32 little-endian bytes**（runner 只认 f32）。

    长度不符即抛 `ValueError("hidden_payload_size_mismatch")` —— 由调用方翻成协议错误码。
    """
    if mode == "none":
        raw = bytes(payload)
        if len(raw) != int(n_tokens) * int(n_embd) * 4:
            raise ValueError("hidden_payload_size_mismatch")
        return raw

    tokens, width = int(n_tokens), int(n_embd)
    if tokens < 1 or width < 1:
        raise ValueError("invalid_hidden_shape")
    expected = expected_quantized_bytes(mode, tokens, width)
    raw = bytes(payload)
    if len(raw) != expected:
        raise ValueError("hidden_payload_size_mismatch")

    if mode == "f16":
        arr = np.frombuffer(raw, dtype="<f2").reshape(tokens, width).astype(np.float32)
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()

    if mode not in ("int8_block128", "int4_block128"):
        raise ValueError(f"unknown hidden quant mode: {mode!r}")

    blocks = _blocks(width)
    body_size = QUANT_BLOCK if mode == "int8_block128" else QUANT_BLOCK // 2
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(tokens, blocks, 4 + body_size)
    scale = packed[:, :, :4].copy().view("<f4").reshape(tokens, blocks, 1)

    if mode == "int8_block128":
        quantized = packed[:, :, 4:].copy().view("<i1").astype(np.float32)
    else:
        bytes_body = packed[:, :, 4:]
        low = (bytes_body & 0x0F).astype(np.uint8)
        high = ((bytes_body >> 4) & 0x0F).astype(np.uint8)
        codes = np.empty((tokens, blocks, QUANT_BLOCK), dtype=np.uint8)
        codes[:, :, 0::2] = low
        codes[:, :, 1::2] = high
        quantized = codes.astype(np.float32) - 8.0

    # 编码是 `q = round(x / scale * levels)`（见 `encode_hidden`）⇒ 解码必须**先除以 levels**
    # 再乘回 scale。⚠️ 漏掉这一步会让结果整体放大 levels 倍（实测：int8 放大 127×、int4 放大 7×）。
    levels = 127.0 if mode == "int8_block128" else 7.0
    flat = (quantized / levels) * scale
    width_padded = blocks * QUANT_BLOCK
    out = flat.reshape(tokens, width_padded)[:, :width]
    return np.ascontiguousarray(out, dtype="<f4").tobytes()


# ------------------------------------------------------------------ 误差反馈 / 残差补偿（R-R8）
#
# 背景（`docs/跨框架接力…` §5.1③ 与 §4）：`f16` / `int8_block128` 在 512 步内**既不累积误差、
# 也不改变任何 token** ⇒ **当前不需要**误差反馈；**仍待做的是更激进档（int4 / 2bit）下的误差反馈**。
# 本段给那条研究提供**可测基础**，并且**不改 wire**（残差留在发送侧，协议零变化）。


def roundtrip_f32(hidden, mode: str, n_tokens: int, n_embd: int) -> np.ndarray:
    """`encode_hidden` → `decode_hidden` 的**往返结果**（f32 数组，形状 `[n_tokens, n_embd]`）。

    纯函数：让"本地模拟"与"真实 wire"走**同一套**编解码，避免两套口径漂移。
    """
    wire = encode_hidden(hidden, mode, n_tokens, n_embd)
    return _as_f32(decode_hidden(wire, mode, n_tokens, n_embd), n_tokens, n_embd)


def error_feedback_roundtrip(hidden, mode: str, n_tokens: int, n_embd: int,
                             carry=None) -> tuple[np.ndarray, np.ndarray]:
    """**残差补偿（error feedback / noise shaping）**的一步（发送侧，**不改 wire**）。

    做法::

        adjusted = x + carry            # 先把上一轮的量化残差加回来
        y        = roundtrip(adjusted)  # 真正上 wire 并解回来
        carry'   = adjusted - y         # 本轮残差留到下一轮

    **为什么它有意义（有数学保证，不是玄学）**：设无补偿时第 i 步误差 `e_i = y_i - x_i`，
    则 `Σ e_i` 是随机游走（约 `√N` 增长）；而带补偿时恒有

        Σ (y_i - x_i) = carry_0 - carry_N

    ⇒ **累积误差有界**（被 `carry` 吸收）。这正是 `tests/test_relay_hidden_quant.py` 里那条
    **恒等式断言**与"有补偿的累积误差显著更小"断言的依据。

    ⚠️ **它不改善单步精度**：`y - x = carry - carry'` 仍是同量级 ⇒ 对 **int4 这种"逐轮就失败"**
    的档位（A5 实测 23/32），**不要指望它救回来** —— 那种失败来自块内 scale 太粗，不是累积。

    返回 `(y, carry')`。
    """
    arr = _as_f32(hidden, n_tokens, n_embd)
    previous = (np.zeros_like(arr) if carry is None
                else np.ascontiguousarray(np.asarray(carry, dtype=np.float32)))
    if previous.shape != arr.shape:
        raise ValueError("error_feedback_carry_shape_mismatch")
    adjusted = arr + previous
    out = roundtrip_f32(adjusted, mode, n_tokens, n_embd)
    return out, np.ascontiguousarray(adjusted - out, dtype=np.float32)


def quant_error_stats(reference, actual) -> dict[str, float]:
    """误差画像（供**归因**与报告）：`max_abs` / `rms` / `rel_rms`（`rms` 用参考的 RMS 归一）。

    ⚠️ 只用于解释"哪一档误差多大、补偿后降了多少"；**验收判据仍只认 per-token argmax**，
    不得拿这些数当准入依据（`docs/跨框架接力…` 的纪律）。
    """
    ref = np.asarray(reference, dtype=np.float32)
    got = np.asarray(actual, dtype=np.float32)
    if ref.shape != got.shape:
        raise ValueError("quant_error_shape_mismatch")
    diff = got - ref
    size = int(diff.size)
    rms = float(np.sqrt(np.mean(np.square(diff)))) if size else 0.0
    base = float(np.sqrt(np.mean(np.square(ref)))) if size else 0.0
    return {
        "max_abs": float(np.abs(diff).max()) if size else 0.0,
        "rms": rms,
        "rel_rms": (rms / base) if base > 0 else 0.0,
    }
