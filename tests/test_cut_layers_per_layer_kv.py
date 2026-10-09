"""tests/test_cut_layers_per_layer_kv.py — ⑩ 按层数组型 KV 重写（#67 缺口 ⑩）

背景（`docs/已知问题记录.md` 的 `#67`）：
llama.cpp 对 hybrid 架构要求 `<arch>.attention.recurrent_layers` 这类**按层数组** KV 的
长度等于**该模型的层数**（`n_layer_all`）。而 `scripts/cut_layers.py` 的 `_copy_kv`
对 `GGUFValueType.ARRAY` 是**原样复制** ⇒ 裁层后长度不符 ⇒ llama.cpp 加载时 throw。

现有真机件能加载 ⇒ 推测当前那个 Q4_K_M 源件**不含**该 KV；但这属于"侥幸可用"，
必须补上：
1. **有该 KV 时正确重写**（按保留区间取子数组，与张量裁层保持同一口径）；
2. **长度与层数不符时 fail-closed 拒绝**（源不可信，绝不让它静默产出坏工件）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
ARCH = "qwen35"
N_LAYERS = 8
INTERVAL = 4


def _make_gguf(path: Path, *, n_layers: int = N_LAYERS,
               recurrent_layers: list | None = None) -> None:
    """合成迷你 GGUF；`recurrent_layers` 可注入按层数组 KV。"""
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(8)
    writer.add_key_value(f"{ARCH}.full_attention_interval", INTERVAL,
                         gguf.GGUFValueType.INT32)
    if recurrent_layers is not None:
        writer.add_key_value(
            f"{ARCH}.attention.recurrent_layers",
            [bool(x) for x in recurrent_layers],
            gguf.GGUFValueType.ARRAY,
            gguf.GGUFValueType.BOOL,
        )
    for layer in range(n_layers):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(8, dtype=np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _canonical_recurrent(n: int, interval: int = INTERVAL) -> list:
    """与 llama.cpp 的推导一致：非 full 层即 recurrent。"""
    return [(i + 1) % interval != 0 for i in range(n)]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _read_kv(path: Path, key: str):
    import gguf

    r = gguf.GGUFReader(str(path))
    f = r.fields.get(key)
    return list(f.contents()) if f is not None else None


class TestPerLayerArrayKV:
    def test_tail_cut_slices_the_per_layer_array(self, tmp_path):
        """tail 裁层（丢前 K 层）⇒ 按层数组同步取 [K:]，长度 == 保留层数。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, recurrent_layers=_canonical_recurrent(N_LAYERS))
        dst = tmp_path / "out.gguf"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4")
        assert r.returncode == 0, (r.stdout, r.stderr)
        got = _read_kv(dst, f"{ARCH}.attention.recurrent_layers")
        assert got is not None, "按层数组 KV 应当被保留"
        assert len(got) == N_LAYERS - 4, f"长度应为保留层数 4，实得 {len(got)}: {got}"
        assert got == _canonical_recurrent(N_LAYERS)[4:], got

    def test_middle_cut_slices_the_per_layer_array(self, tmp_path):
        """middle 裁层 ⇒ 按层数组取 [K:end)。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, recurrent_layers=_canonical_recurrent(N_LAYERS))
        dst = tmp_path / "out.gguf"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4", "--end", "8")
        assert r.returncode == 0, (r.stdout, r.stderr)
        got = _read_kv(dst, f"{ARCH}.attention.recurrent_layers")
        assert got == _canonical_recurrent(N_LAYERS)[4:8], got

    def test_head_cut_keeps_prefix_of_per_layer_array(self, tmp_path):
        """head 模式（保留前 N 层、不重编号）⇒ 按层数组取 [:N]。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, recurrent_layers=_canonical_recurrent(N_LAYERS))
        dst = tmp_path / "out.gguf"
        r = _run("--src", str(src), "--dst", str(dst), "--keep-head", "4")
        assert r.returncode == 0, (r.stdout, r.stderr)
        got = _read_kv(dst, f"{ARCH}.attention.recurrent_layers")
        assert got == _canonical_recurrent(N_LAYERS)[:4], got

    def test_rejects_when_per_layer_array_length_mismatches(self, tmp_path):
        """★ 该红必须红：按层数组长度 ≠ 层数 ⇒ 源不可信 ⇒ fail-closed 拒绝。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, recurrent_layers=_canonical_recurrent(N_LAYERS - 2))
        r = _run("--src", str(src), "--dst", str(tmp_path / "out.gguf"), "--k", "4")
        assert r.returncode != 0, r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        assert "recurrent_layers" in blob, blob
        assert not (tmp_path / "out.gguf").exists(), "拒绝时不得写出工件"

    def test_absent_per_layer_array_changes_nothing(self, tmp_path):
        """源不含该 KV ⇒ 保持既有行为（不引入新失败）。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src, recurrent_layers=None)
        dst = tmp_path / "out.gguf"
        r = _run("--src", str(src), "--dst", str(dst), "--k", "4")
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert _read_kv(dst, f"{ARCH}.attention.recurrent_layers") is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
