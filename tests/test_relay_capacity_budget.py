"""★ R-R5：容量预算（不可分层张量构成 + 收益上限）的回归。

判据纪律：本文件全是**容量账**；数值一致性仍**只认 per-token argmax**。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import relay_capacity_budget as C  # noqa: E402


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """写一个**只有头部、没有数据**的最小 safetensors（本模块只读头部 ⇒ 够用）。"""
    header = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
              for name, (dtype, shape) in tensors.items()}
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_read_header_and_tensor_bytes(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    _write_safetensors(target, {"model.embed_tokens.weight": ("BF16", [151936, 896])})
    header = C.read_safetensors_header(target)
    assert set(header) == {"model.embed_tokens.weight"}
    assert C.tensor_bytes(header["model.embed_tokens.weight"]) == 151936 * 896 * 2


def test_classify_uses_layer_membership_not_name_keywords() -> None:
    """★ 口径：**不在任何 `*.layers.<i>.*` 里**即不可分层（不是"名字里有 embed"）。"""
    keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.23.mlp.down_proj.weight",
        "model.embed_tokens.weight",                      # 层外 ⇒ 不可分层
        "lm_head.weight",                                 # 层外 ⇒ 不可分层
        "model.norm.weight",                              # 层外 ⇒ 不可分层
        "model.visual.blocks.0.mlp.linear_fc1.weight",    # 层外（视觉塔）
        "mtp.fc.weight",                                  # 层外（MTP）
    ]
    splittable, fixed = C.classify_tensors(keys)
    assert len(splittable) == 2
    assert set(fixed) == {
        "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        "model.visual.blocks.0.mlp.linear_fc1.weight", "mtp.fc.weight",
    }


def test_visual_blocks_become_splittable_when_asked() -> None:
    keys = ["model.visual.blocks.0.x.weight", "model.embed_tokens.weight"]
    _, fixed = C.classify_tensors(keys)
    assert "model.visual.blocks.0.x.weight" in fixed
    splittable, rest = C.classify_tensors(keys, extra_layer_patterns=(C.VISUAL_BLOCK_RE,))
    assert splittable == ["model.visual.blocks.0.x.weight"]
    assert rest == ["model.embed_tokens.weight"]


def test_sharded_strategy_hits_the_ceiling() -> None:
    """`C` 能均分 ⇒ 收益恰好等于段数（"不可分层消失"时的理论上限）。"""
    assert C.relay_gain(700.0, 300.0, 2, strategy="fixed-sharded") == pytest.approx(2.0)
    assert C.relay_gain(700.0, 300.0, 3, strategy="fixed-sharded") == pytest.approx(3.0)


def test_role_split_reproduces_documented_qwen25_result() -> None:
    """★ **复现文档 §3 的 1.568×**。

    qwen2.5-0.5b 实测：`L = 715.8 MB`、`C = 272.3 MB`，且 **tie ⇒ `C_out = 0`**（没有 lm_head）
    ⇒ 分母 = `L/2 + C_emb` ⇒ `988.1 / 630.2 = 1.568`。
    """
    gain = C.relay_gain(715.8, 272.3, 2, strategy="fixed-split-by-role",
                        c_upstream_bytes=272.3, c_downstream_bytes=0.0)
    assert gain == pytest.approx(1.568, abs=0.002)


def test_role_split_must_use_real_sides_not_half() -> None:
    """★「该红必须红」：拿 `C/2` 近似会算出**假高收益**。

    tie 模型把 `C` 当"两边各一半" ⇒ 分母偏小 ⇒ 收益虚高（越过 1.6×），而诚实口径是 1.568×。
    """
    honest = C.relay_gain(715.8, 272.3, 2, strategy="fixed-split-by-role",
                          c_upstream_bytes=272.3, c_downstream_bytes=0.0)
    faked = C.relay_gain(715.8, 272.3, 2, strategy="fixed-split-by-role",
                         c_upstream_bytes=136.15, c_downstream_bytes=136.15)
    assert honest < 1.6 < faked


def test_strategy_ordering_and_zero_fixed_degenerates() -> None:
    """`C = 0` ⇒ 三策略都退化为段数（纯层切）；一般情形 `sharded ≥ role-split ≥ upstream`。"""
    assert C.relay_gain(700.0, 0.0, 2, strategy="fixed-upstream") == pytest.approx(2.0)
    assert C.relay_gain(700.0, 0.0, 2, strategy="fixed-sharded") == pytest.approx(2.0)
    upstream = C.relay_gain(700.0, 300.0, 2, strategy="fixed-upstream")
    role = C.relay_gain(700.0, 300.0, 2, strategy="fixed-split-by-role",
                        c_upstream_bytes=300.0, c_downstream_bytes=0.0)
    sharded = C.relay_gain(700.0, 300.0, 2, strategy="fixed-sharded")
    assert sharded >= role >= upstream


def test_unknown_strategy_fails_loud() -> None:
    with pytest.raises(ValueError):
        C.relay_gain(1.0, 1.0, 2, strategy="magic")


def test_capacity_report_on_synthetic_dir(tmp_path: Path) -> None:
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [100, 100]),
        "model.layers.1.self_attn.q_proj.weight": ("BF16", [100, 100]),
        "model.embed_tokens.weight": ("BF16", [1000, 100]),
        "lm_head.weight": ("BF16", [1000, 100]),
        "model.norm.weight": ("BF16", [100]),
    })
    report = C.capacity_report(tmp_path, segments=(2, 3))

    assert report["tensor_count"] == 5
    assert report["splittable_count"] == 2 and report["fixed_count"] == 3
    # 层内 2×10_000×2B = 40_000B；层外 (100_000+100_000+100)×2B = 400_200B
    assert report["splittable_bytes"] == 40_000
    assert report["fixed_bytes"] == 400_200
    assert report["fixed_ratio"] == pytest.approx(400_200 / 440_200)
    # 角色拆分：embed 归上游、lm_head 与 norm 归下游
    assert report["fixed_upstream_bytes"] == 200_000
    assert report["fixed_downstream_bytes"] == 200_200
    assert set(report["gains"]) == set(C.STRATEGIES)
    assert "gain_2seg" in report["gains"]["fixed-sharded"]


def test_capacity_report_requires_safetensors(tmp_path: Path) -> None:
    """缺 safetensors ⇒ 抛 `FileNotFoundError`，**不虚构数字**（调用方据此跳过）。"""
    with pytest.raises(FileNotFoundError):
        C.capacity_report(tmp_path)
