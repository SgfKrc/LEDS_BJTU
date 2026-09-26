"""★ #31 M4 的回归：KV 状态搬运必须支持 **Cache 对象**，并跳过 hybrid 的 `None` 槽位。

为什么要有这个文件：hybrid（`qwen3_5`）的 `forward_layers` 返回的 `past_key_values` **tuple 是有损的**
—— `linear_attention` 层的 recurrent state 在 tuple 里是 `None` 占位
（实测 12 层里 9 层是 `None`，**第 0 层就是**）⇒ 旧写法 `past_kv[0][0].shape` 会 `None[0]` 直接
`TypeError` 硬崩，而**只搬 tuple**还会静默丢掉 recurrent state。完整状态只在 `result["cache"]` 里。

这里用**合成 fixture** 覆盖纯函数逻辑（不需要 transformers / 真模型）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from scheduler_pipeline import _kv_state_seq_len, _prefer_cache_state  # noqa: E402


class _FakeKV:
    """`keys` 形状即真实 KV 张量的形状（取 seq 维用）。"""

    def __init__(self, shape):
        self.keys = _FakeTensor(shape)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class _FakeRecurrent:
    """`linear_attention` 的槽位：**没有 `keys`**（状态在 recurrent_states 里）。"""


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


# ── `_kv_state_seq_len` ─────────────────────────────────────────────────────


def test_kv_state_seq_len_skips_hybrid_none_slots() -> None:
    """★ 核心：hybrid 的 tuple 里第 0 层是 `None` —— 必须**跳过**它取到真正有 KV 的层。

    这正是旧代码 `past_kv[0][0].shape` 会硬崩的形态（`None[0]`）。
    """
    past = (None, None, None, (_FakeTensor((1, 2, 8, 256)), None),
            None, None, None, (_FakeTensor((1, 2, 8, 256)), None))

    layers, seq_len = _kv_state_seq_len(past, "qwen2")

    assert layers == 8          # 槽位数按 tuple 长度算（含 None），与旧口径一致
    assert seq_len == 8         # seq 维取第一个**非空**层：shape[2]（非 qwen 系）
    # 反证：旧写法
    with pytest.raises(TypeError):
        past[0][0].shape  # type: ignore[index]


def test_kv_state_seq_len_qwen_uses_dim_one() -> None:
    """`model_type == "qwen"` 时序列长度取 `shape[1]`（既有口径，不能改）。"""
    past = ((_FakeTensor((2, 5, 3, 64)), None),)

    assert _kv_state_seq_len(past, "qwen") == (1, 5)
    assert _kv_state_seq_len(past, "qwen2") == (1, 3)


def test_kv_state_seq_len_accepts_cache_object() -> None:
    """★ Cache 对象（hybrid 的完整状态载体）：`cache.layers[i].keys` 提供形状。"""
    cache = _FakeCache([_FakeRecurrent(), _FakeRecurrent(), _FakeRecurrent(),
                        _FakeKV((1, 2, 8, 256))])

    layers, seq_len = _kv_state_seq_len(cache, "qwen2")

    assert layers == 4
    assert seq_len == 8


def test_kv_state_seq_len_is_safe_when_no_kv_slot_exists() -> None:
    """全是 recurrent 槽位（没有 `keys`）⇒ 返回 `seq_len=0`，**不抛**。"""
    cache = _FakeCache([_FakeRecurrent(), _FakeRecurrent()])

    assert _kv_state_seq_len(cache, "qwen2") == (2, 0)


def test_kv_state_seq_len_tolerates_empty_state() -> None:
    """空 state（`None` / 空 tuple）⇒ 不抛。"""
    assert _kv_state_seq_len(None, "qwen2") == (0, 0)
    assert _kv_state_seq_len((), "qwen2") == (0, 0)


# ── `_prefer_cache_state` ───────────────────────────────────────────────────


def test_prefer_cache_state_prefers_cache_object() -> None:
    """★ 有 `cache` 就必须用它（tuple 会丢 recurrent state）。"""
    cache = _FakeCache([])
    result = {"cache": cache, "past_key_values": ("tuple-version",)}

    assert _prefer_cache_state(result) is cache


def test_prefer_cache_state_falls_back_to_tuple() -> None:
    """没有 `cache`（纯 attention 模型）⇒ 回落到 tuple，行为与旧版一致。"""
    past = ("tuple-version",)

    assert _prefer_cache_state({"past_key_values": past}) == past
    # `cache` 存在但为 None 时同样回落
    assert _prefer_cache_state({"cache": None, "past_key_values": past}) == past
