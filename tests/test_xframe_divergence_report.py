"""XFRAME-1 的分歧画像单元测试。

`profile()` 是纯函数（只吃两侧的 token id 与 margin），因此可以脱离引擎单测。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from xframe_divergence_report import llama_relay_greedy, profile  # noqa: E402


def test_relay_greedy_rejects_single_segment():
    """分层链至少两段；单段是调用方错误，必须在加载任何模型前 fail-fast。"""
    import pytest

    with pytest.raises(ValueError):
        llama_relay_greedy(["only-one.gguf"], [1, 2, 3], 4)


def test_identical_sequences():
    left = {"ids": [1, 2, 3], "margins": [1.0, 2.0, 3.0]}
    right = {"ids": [1, 2, 3], "margins": [1.1, 2.1, 3.1]}

    row = profile(left, right, eos_id=None)

    assert row["identical"] is True
    assert row["first_divergence_step"] is None
    assert row["length_only_divergence"] is False
    assert row["top1_agreement"] == 1.0


def test_token_level_divergence_reports_step_and_margins():
    left = {"ids": [1, 2, 3], "margins": [0.5, 0.5, 0.5]}
    right = {"ids": [1, 9, 3], "margins": [2.0, 2.0, 2.0]}

    row = profile(left, right, eos_id=None)

    assert row["identical"] is False
    assert row["first_divergence_step"] == 1
    assert row["length_only_divergence"] is False
    # 3 个位置里 0 与 2 相同
    assert row["top1_agreement"] == round(2 / 3, 4)
    # 分歧点的两侧 token 与各自 margin 都要留下，便于判「是否在 margin 低处翻转」
    assert row["divergence_point"]["left_token"] == 2
    assert row["divergence_point"]["right_token"] == 9
    assert row["divergence_point"]["left_margin"] == 0.5
    assert row["divergence_point"]["right_margin"] == 2.0


def test_trailing_eos_is_stripped_before_comparison():
    """两侧的 EOS 处理口径不同，剔除尾部后才能正确判定「其实一致」。"""
    left = {"ids": [1, 2, 99], "margins": [1.0, 1.0, 1.0]}
    right = {"ids": [1, 2], "margins": [1.0, 1.0]}

    row = profile(left, right, eos_id=99)

    assert row["identical"] is True
    assert row["first_divergence_step"] is None


def test_length_only_divergence_is_flagged_separately():
    """前缀全同、只是长度不同 ⇒ 属 EOS 决策分叉，不报成 token 级分歧。"""
    left = {"ids": [1, 2, 3], "margins": [1.0, 1.0, 1.0]}
    right = {"ids": [1, 2], "margins": [1.0, 1.0]}

    row = profile(left, right, eos_id=None)

    assert row["identical"] is False
    assert row["length_only_divergence"] is True
    assert row["first_divergence_step"] is None
    assert row["top1_agreement"] == 1.0
    assert row["left_len"] == 3
    assert row["right_len"] == 2


def test_empty_generation_is_safe():
    row = profile({"ids": [], "margins": []}, {"ids": [], "margins": []}, eos_id=99)

    assert row["identical"] is True
    assert row["top1_agreement"] == 1.0
