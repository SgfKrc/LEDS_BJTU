"""B16：tied embeddings 的探测与「末节点」支持。

Qwen3.5 等 tied 模型的 safetensors 里**没有** `lm_head.weight`（与 `embed_tokens` 共用），
而分层加载的完整性校验原本**硬要求**它 ⇒ `has_lm_head=True` 必然报
`qwen2 分层权重不完整: lm_head.weight`。本文件覆盖探测逻辑（纯 CPU、离线）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_module import _is_tied_word_embeddings  # noqa: E402


class _Cfg:
    """最小 config 替身（可控 tie_word_embeddings 的有无/取值）。"""

    def __init__(self, tie="unset"):
        if tie != "unset":
            self.tie_word_embeddings = tie


def _write_index(tmp_path: Path, keys) -> str:
    """写一个只含 weight_map 的 index.json（`_iter_safetensors_keys` 会优先读它）。"""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_explicit_true_wins():
    """config 显式 True ⇒ 直接判定 tied，不扫盘。"""
    assert _is_tied_word_embeddings(_Cfg(True), "/nonexistent") is True


def test_explicit_false_wins():
    """config 显式 False ⇒ 直接判定非 tied（即使目录不存在）。"""
    assert _is_tied_word_embeddings(_Cfg(False), "/nonexistent") is False


def test_missing_field_infers_tied_from_keys(tmp_path):
    """字段缺失 + 有 embed_tokens.weight 且无 lm_head.weight ⇒ 推断为 tied。"""
    path = _write_index(tmp_path, [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ])
    assert _is_tied_word_embeddings(_Cfg("unset"), path) is True


def test_missing_field_infers_untied_from_keys(tmp_path):
    """字段缺失 + 有独立的 lm_head.weight ⇒ 推断为非 tied（既有模型行为不变）。"""
    path = _write_index(tmp_path, [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ])
    assert _is_tied_word_embeddings(_Cfg("unset"), path) is False


def test_multimodal_prefix_still_detected(tmp_path):
    """多模态外壳的 `model.language_model.embed_tokens.weight` 也要认出来。"""
    path = _write_index(tmp_path, [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    ])
    assert _is_tied_word_embeddings(_Cfg("unset"), path) is True


def test_probe_failure_is_silent_false():
    """目录不存在 / 无法列举 ⇒ 一律按非 tied 处理（探测必须无副作用）。"""
    assert _is_tied_word_embeddings(_Cfg("unset"), "/definitely/not/a/dir") is False
    assert _is_tied_word_embeddings(_Cfg("unset"), None) is False


def test_no_keys_at_all_is_false(tmp_path):
    """既无 lm_head 也无 embed_tokens（异常目录）⇒ 不算 tied。"""
    path = _write_index(tmp_path, ["model.norm.weight"])
    assert _is_tied_word_embeddings(_Cfg("unset"), path) is False
