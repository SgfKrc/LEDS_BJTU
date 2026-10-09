"""tests/test_cut_multiple_from_gguf.py — `#67` 缺口「`cut_multiple` 自动从 GGUF KV 读」

背景（`docs/已知问题记录.md` 的 `#67`）：
`cut_multiple` 此前**全靠人工传参**（`scripts/relay_cut_plan.py` 的 `--cut-multiple`、
`torch_hetero_plan.HeteroBaseline.cut_multiple`），而两处的**默认值都是 1**。
对 hybrid 架构（Qwen3.5，`full_attention_interval=4`）而言，`1` 是**错的** ——
裁层 GGUF 的层类型按本地层号对 interval 取模推导，非整数倍切点会整体错位、
llama.cpp 报 `missing tensor 'blk.x.<...>'`。

⇒ 正确做法是**从源 GGUF 的 `<arch>.full_attention_interval` 直接读**，让人工参数成为
可选覆盖而不是必需知识。本文件锁住这个"读 + 退回"契约。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.relay_cut_objective import cut_multiple_from_gguf  # noqa: E402


def _make_gguf(path: Path, arch: str = "qwen35", interval: int | None = 4,
               n_layers: int = 8) -> None:
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(8)
    if interval is not None:
        writer.add_key_value(f"{arch}.full_attention_interval", interval,
                             gguf.GGUFValueType.INT32)
    for layer in range(n_layers):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestCutMultipleFromGguf:
    def test_reads_interval_from_hybrid_source(self, tmp_path):
        """hybrid 源 ⇒ 返回其 `full_attention_interval`（Qwen3.5 = 4）。"""
        p = tmp_path / "q35.gguf"
        _make_gguf(p, interval=4)
        assert cut_multiple_from_gguf(p) == 4

    def test_reads_non_four_interval_generically(self, tmp_path):
        """不把 4 写死 —— 读什么就是什么。"""
        p = tmp_path / "other.gguf"
        _make_gguf(p, interval=8)
        assert cut_multiple_from_gguf(p) == 8

    def test_returns_none_when_source_has_no_interval(self, tmp_path):
        """非 hybrid 源（无该 KV）⇒ 返回 None，调用方应保持既有行为。"""
        p = tmp_path / "qwen2.gguf"
        _make_gguf(p, arch="qwen2", interval=None)
        assert cut_multiple_from_gguf(p) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        """文件不存在 ⇒ None（不抛；这是"探测"而不是"校验"）。"""
        assert cut_multiple_from_gguf(tmp_path / "nope.gguf") is None

    def test_interval_one_means_no_constraint(self, tmp_path):
        """`interval=1`（或无约束）⇒ None —— 不能让调用方把 1 当成"必须整除"。"""
        p = tmp_path / "plain.gguf"
        _make_gguf(p, interval=1)
        assert cut_multiple_from_gguf(p) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
