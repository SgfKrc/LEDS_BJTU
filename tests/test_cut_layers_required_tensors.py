"""tests/test_cut_layers_required_tensors.py — ④ 按层类型断言必需张量集合（#67 缺口 ④）

背景（`docs/已知问题记录.md` 的 `#67`）：
`qwen35.cpp:37-110` 对每个 block 的 `create_tensor` 调用决定了"哪些张量必须有"。
其中少数标注了 `TENSOR_NOT_REQUIRED`（`wqkv`/`wqkv_gate`/`output.weight`）——
**它们缺失是合法的**（tie embeddings、可选 gate）。

现实形态（`qwen35-2b-Q4_K_M.gguf`，335 张量）：
* 非 blk：`token_embd.weight`、`output_norm.weight`（**无** `output.weight`）
* linear 层 14 个 = 2 norm + 2 可选(qkv/gate) + 7 ssm + 3 ffn
* full   层 11 个 = 2 norm + 4 q/k/v/o + 2 q_norm/k_norm + 3 ffn
⇒ `3×14 + 1×11 + 2 = 55`（与 `q35-2b-cut-16-20.manifest.json` 的 `tensors_kept` 一致）

生成器此前**不校验必需集合** ⇒ 缺张量的工件要到设备上加载才炸。本文件锁住 fail-closed。
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

#: linear（recurrent）层必需 —— 去掉 NOT_REQUIRED 的 `attn_qkv` / `attn_gate`
LINEAR_REQUIRED = (
    "attn_norm.weight", "post_attention_norm.weight",
    "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_beta.weight",
    "ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)
#: full attention 层必需
FULL_REQUIRED = (
    "attn_norm.weight", "post_attention_norm.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)


def _is_recurrent(local: int) -> bool:
    """与 `qwen35.cpp:25` 一致：`(i+1) % interval != 0` ⇒ recurrent(linear)。"""
    return (local + 1) % INTERVAL != 0


def _make_gguf(path: Path, *, drop_suffix: str | None = None,
               drop_layer_kind: str | None = None) -> None:
    """合成 GGUF；`drop_suffix` 可从**指定层类型**的层里删掉一个必需张量。

    ⚠️ 删的层必须落在**保留区间内**（测试统一用 `--k 4` ⇒ 保留 `blk.4..blk.7`），
    否则该校验根本不会看到它 —— 早期版本删 `blk.0` 导致测试假绿（实测踩到）。
    """
    import gguf
    import numpy as np

    FIRST_KEPT = 4  # 与测试里的 --k 对齐（保留 blk.4..blk.7）
    # 删哪个层：取**保留区间内该层类型的第一个** —— linear 落在 blk.4，
    # full 落在 blk.7（local=3 ⇒ (3+1)%4==0）。早期版本固定删 blk.0，
    # 那是 linear 且**不在保留区间**，导致测试既没覆盖 full 又假绿（实测踩到）。
    drop_at = None
    if drop_suffix and drop_layer_kind:
        for il in range(FIRST_KEPT, N_LAYERS):
            recr = _is_recurrent(il)
            if ((drop_layer_kind == "linear" and recr)
                    or (drop_layer_kind == "full" and not recr)):
                drop_at = il
                break
    assert drop_suffix is None or drop_at is not None, (
        f"保留区间 [{FIRST_KEPT},{N_LAYERS}) 内没有 {drop_layer_kind} 层，测试构造有误"
    )
    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(8)
    writer.add_key_value(f"{ARCH}.full_attention_interval", INTERVAL,
                         gguf.GGUFValueType.INT32)
    for il in range(N_LAYERS):
        recr = _is_recurrent(il)
        req = LINEAR_REQUIRED if recr else FULL_REQUIRED
        for suffix in req:
            if drop_suffix and il == drop_at and suffix == drop_suffix:
                continue
            writer.add_tensor(f"blk.{il}.{suffix}", np.ones(8, dtype=np.float32))
        # NOT_REQUIRED 的也照实写上一个（真实件里有）
        if recr:
            writer.add_tensor(f"blk.{il}.attn_qkv.weight", np.ones(8, dtype=np.float32))
            writer.add_tensor(f"blk.{il}.attn_gate.weight", np.ones(8, dtype=np.float32))
    writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class TestRequiredTensorContract:
    def test_accepts_complete_source(self, tmp_path):
        """必需集合齐全 ⇒ 放行（回归基线）。"""
        src = tmp_path / "ok.gguf"
        _make_gguf(src)
        r = _run("--src", str(src), "--k", "4", "--dry-run")
        assert r.returncode == 0, (r.stdout, r.stderr)

    def test_rejects_missing_linear_required_tensor(self, tmp_path):
        """★ 该红必须红：linear 层缺 `ssm_conv1d`（必需）⇒ 拒绝。"""
        src = tmp_path / "bad.gguf"
        _make_gguf(src, drop_suffix="ssm_conv1d.weight", drop_layer_kind="linear")
        r = _run("--src", str(src), "--k", "4", "--dry-run")
        assert r.returncode != 0, r.stdout
        blob = (r.stdout or "") + (r.stderr or "")
        assert "ssm_conv1d" in blob, blob

    def test_rejects_missing_full_required_tensor(self, tmp_path):
        """★ full 层缺 `attn_q_norm`（必需）⇒ 拒绝。"""
        src = tmp_path / "bad2.gguf"
        _make_gguf(src, drop_suffix="attn_q_norm.weight", drop_layer_kind="full")
        r = _run("--src", str(src), "--k", "4", "--dry-run")
        assert r.returncode != 0, r.stdout
        assert "attn_q_norm" in ((r.stdout or "") + (r.stderr or ""))

    def test_missing_not_required_tensor_is_allowed(self, tmp_path):
        """`wqkv`/`wqkv_gate` 是 `TENSOR_NOT_REQUIRED` ⇒ 缺失**不得**被判失败。"""
        import gguf
        import numpy as np

        p = tmp_path / "noqkv.gguf"
        writer = gguf.GGUFWriter(str(p), ARCH)
        writer.add_block_count(N_LAYERS)
        writer.add_embedding_length(8)
        writer.add_key_value(f"{ARCH}.full_attention_interval", INTERVAL,
                             gguf.GGUFValueType.INT32)
        for il in range(N_LAYERS):
            req = LINEAR_REQUIRED if _is_recurrent(il) else FULL_REQUIRED
            for suffix in req:
                writer.add_tensor(f"blk.{il}.{suffix}", np.ones(8, dtype=np.float32))
        writer.add_tensor("token_embd.weight", np.ones((8, 8), dtype=np.float32))
        writer.add_tensor("output_norm.weight", np.ones(8, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        r = _run("--src", str(p), "--k", "4", "--dry-run")
        assert r.returncode == 0, (r.stdout, r.stderr)

    def test_missing_output_weight_is_allowed(self, tmp_path):
        """`output.weight` 是 `TENSOR_NOT_REQUIRED`（tie embeddings）⇒ 缺失合法。"""
        src = tmp_path / "tie.gguf"
        _make_gguf(src)  # 本就不写 output.weight
        r = _run("--src", str(src), "--k", "4", "--dry-run")
        assert r.returncode == 0, (r.stdout, r.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
