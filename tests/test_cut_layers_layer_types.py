"""tests/test_cut_layers_layer_types.py — ③ 层类型序列逐位校验（#67 缺口 ③）

背景（`docs/已知问题记录.md` 的 `#67`）：
llama.cpp **不读** HF 的 `layer_types` 表，而是**按（裁层重编号后的）本地层号对
`full_attention_interval` 取模推导**层类型
（`android/.../llama.cpp/src/models/qwen35.cpp:21-27`：
`is_recr_impl[i] = (i+1) % full_attn_interval != 0`）。

⇒ 由此有两个"错位"来源，此前**只挡了第一个**：
1. **切点/段长不是 interval 的整数倍**（已有 `_check_multiple` 挡住）；
2. **源模型的 `layer_types` 本身不遵循该规律** —— 此时无论怎么切，重推序列都对不上，
   加载必然报 `missing tensor 'blk.x.<...>'`，而生成器却会**静默产出一个坏工件**。
   本文件锁住第 2 条的 fail-closed 行为（`--hf-config` 可选参数）。

同时校验 HF 配置本身的自洽性：`len(layer_types) == block_count`（层数不符 = 源不可信）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
ARCH = "qwen35"
N_LAYERS = 8


def _make_gguf(path: Path, n_layers: int = N_LAYERS, interval: int = 4) -> None:
    """合成迷你 GGUF（与 `test_cut_layers.py::_make_gguf` 同构，仅保留本文件所需字段）。"""
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(8)
    writer.add_key_value(f"{ARCH}.full_attention_interval", interval, gguf.GGUFValueType.INT32)
    for layer in range(n_layers):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight", np.ones(8, dtype=np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_norm.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_hf_config(path: Path, layer_types: list, *, interval: int = 4) -> None:
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "full_attention_interval": interval,
                    "layer_types": layer_types,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _canonical_layer_types(n: int, interval: int) -> list:
    """按 `qwen35.cpp:21-27` 的规律重推：`(local+1) % interval == 0` ⇒ full。"""
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


@pytest.fixture()
def src_gguf(tmp_path: Path) -> Path:
    p = tmp_path / "src.gguf"
    _make_gguf(p)
    return p


class TestLayerTypeContract:
    def test_accepts_source_whose_layer_types_follow_the_interval(self, src_gguf, tmp_path):
        """源 layer_types 符合规律 ⇒ 放行。"""
        cfg = tmp_path / "config.json"
        _write_hf_config(cfg, _canonical_layer_types(N_LAYERS, 4))
        r = _run("--src", str(src_gguf), "--k", "4", "--hf-config", str(cfg), "--dry-run")
        assert r.returncode == 0, (r.stdout, r.stderr)

    def test_rejects_source_whose_layer_types_violate_the_interval(self, src_gguf, tmp_path):
        """★ 该红必须红：源 layer_types 不遵循规律 ⇒ **拒绝生成**（否则静默产出坏工件）。

        构造：把索引 3（本应是 full）改成 linear —— 正是"切点合规但源非标准"的形态。
        """
        bad = _canonical_layer_types(N_LAYERS, 4)
        bad[3] = "linear_attention"
        cfg = tmp_path / "config.json"
        _write_hf_config(cfg, bad)
        r = _run("--src", str(src_gguf), "--k", "4", "--hf-config", str(cfg), "--dry-run")
        assert r.returncode != 0, r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        assert "layer_types" in blob, blob
        # 必须指出是**哪一位**不符，否则现场无从定位
        assert "3" in blob, blob

    def test_rejects_when_layer_types_length_mismatch(self, src_gguf, tmp_path):
        """`len(layer_types) != block_count` ⇒ 源不可信 ⇒ 拒绝。"""
        cfg = tmp_path / "config.json"
        _write_hf_config(cfg, _canonical_layer_types(N_LAYERS - 2, 4))
        r = _run("--src", str(src_gguf), "--k", "4", "--hf-config", str(cfg), "--dry-run")
        assert r.returncode != 0, r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        assert "layer_types" in blob, blob

    def test_without_hf_config_behaviour_is_unchanged(self, src_gguf):
        """不传 `--hf-config` ⇒ 保持既有行为（整数倍校验），不引入新失败。"""
        r = _run("--src", str(src_gguf), "--k", "4", "--dry-run")
        assert r.returncode == 0, (r.stdout, r.stderr)

    def test_cut_point_check_still_applies_with_hf_config(self, src_gguf, tmp_path):
        """即使源 layer_types 合法，非整数倍切点仍必须被拒（两条校验互不替代）。"""
        cfg = tmp_path / "config.json"
        _write_hf_config(cfg, _canonical_layer_types(N_LAYERS, 4))
        r = _run("--src", str(src_gguf), "--k", "3", "--hf-config", str(cfg), "--dry-run")
        assert r.returncode != 0, r.stdout
        assert "整数倍" in ((r.stdout or "") + (r.stderr or ""))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
