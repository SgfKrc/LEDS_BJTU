"""tests/test_cut_layers.py — scripts/cut_layers.py 的自包含测试

自包含策略（遵循本仓惯例）：不依赖任何真实大模型工件 —— 现场合成一个 4 层的迷你 GGUF
（含 ``blk.0..3`` 与 ``token_embd`` 等非 blk 张量），再跑生成器；``gguf`` 缺失时整体跳过。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

gguf = pytest.importorskip("gguf")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cut_layers.py"
ARCH = "qwen2"
N_LAYERS = 4


def _make_gguf(path: Path, n_layers: int = N_LAYERS, interval: int | None = None) -> None:
    """合成一个结构上合法（但不含真实权重语义）的迷你 GGUF。"""
    import numpy as np
    import torch

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(8)
    if interval is not None:
        writer.add_key_value(f"{ARCH}.full_attention_interval", interval,
                             gguf.GGUFValueType.INT32)
    for layer in range(n_layers):
        writer.add_tensor(f"blk.{layer}.attn_norm.weight",
                          np.ones(8, dtype=np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_norm.weight",
                          np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


@pytest.fixture()
def src_gguf(tmp_path: Path) -> Path:
    p = tmp_path / "src.gguf"
    _make_gguf(p)
    return p


class TestDryRun:
    def test_reports_plan_without_writing(self, src_gguf: Path, tmp_path: Path) -> None:
        dst = tmp_path / "should-not-exist.gguf"
        r = _run("--src", str(src_gguf), "--dst", str(dst), "--k", "1", "--dry-run")
        assert r.returncode == 0, r.stderr
        assert "block_count: 4 -> 3" in r.stdout
        assert not dst.exists(), "dry-run 不应写出任何文件"


class TestGenerateAndVerify:
    def test_generate_writes_manifest_and_block_count(self, src_gguf: Path, tmp_path: Path) -> None:
        dst = tmp_path / "cut.gguf"
        manifest = tmp_path / "cut.json"
        r = _run("--src", str(src_gguf), "--dst", str(dst), "--k", "1",
                 "--manifest", str(manifest))
        assert r.returncode == 0, r.stderr
        assert dst.exists() and manifest.exists()

        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["trim_layers"] == 1
        assert data["kept_block_count"] == 3
        # 与主仓合同对齐（relay_contract.RelayTrimPlan）
        assert data["contract"]["trim_plan_valid"] is True
        assert data["contract"]["local_to_source_layer_0"] == 1

        # 产物自身可读，且 block_count 已改写
        reader = gguf.GGUFReader(str(dst))
        assert int(reader.fields[f"{ARCH}.block_count"].contents()) == 3
        names = {t.name for t in reader.tensors}
        assert "blk.0.attn_norm.weight" in names
        assert "blk.3.attn_norm.weight" not in names  # 被丢弃层
        assert "token_embd.weight" in names           # 非 blk 张量原样保留

    def test_verify_manifest_roundtrip(self, src_gguf: Path, tmp_path: Path) -> None:
        dst = tmp_path / "cut.gguf"
        manifest = tmp_path / "cut.json"
        assert _run("--src", str(src_gguf), "--dst", str(dst), "--k", "2",
                    "--manifest", str(manifest)).returncode == 0
        r = _run("--src", str(dst), "--verify-manifest", str(manifest))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "block_count=2" in r.stdout


class TestGuards:
    @pytest.mark.parametrize("k", ["0", "4", "5", "-1"])
    def test_out_of_range_k_rejected(self, src_gguf: Path, k: str) -> None:
        r = _run("--src", str(src_gguf), "--k", k, "--dry-run")
        assert r.returncode == 2
        assert "FAIL" in r.stdout

    def test_missing_k_rejected(self, src_gguf: Path) -> None:
        r = _run("--src", str(src_gguf), "--dry-run")
        assert r.returncode == 2

    def test_hybrid_interval_multiple_enforced(self, tmp_path: Path) -> None:
        """hybrid 架构（有 full_attention_interval）时，K 非整数倍必须被拒绝。"""
        src = tmp_path / "hybrid.gguf"
        _make_gguf(src, n_layers=4, interval=2)
        bad = _run("--src", str(src), "--k", "1", "--dry-run")
        assert bad.returncode == 2 and "full_attention_interval" in bad.stdout
        ok = _run("--src", str(src), "--k", "2", "--dry-run")
        assert ok.returncode == 0, ok.stdout

    def test_missing_source_file(self, tmp_path: Path) -> None:
        r = _run("--src", str(tmp_path / "nope.gguf"), "--k", "1", "--dry-run")
        assert r.returncode == 2
