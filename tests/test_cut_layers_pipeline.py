"""tests/test_cut_layers_pipeline.py — `#67` 缺口「无 wrapper 串起：源 sha + 命令 + 产物 sha + 校验」

背景（`docs/已知问题记录.md` 的 `#67`）：
工件制作链此前是**手工敲 CLI**（`cut_layers.py --src ... --k ... --manifest ...`），
"源 sha / 用的命令 / 产物 sha / 校验结果"四件事没有任何一处**同时**留下 ——
事后只能从 manifest 反推，且**没有一次性把校验串进去**（③⑩④ 各自跑各自的）。

本文件锁住新 wrapper `scripts/cut_layers_pipeline.py` 的契约：
1. 一次调用产出**工件 + manifest + `cut-report.json`** 三件套；
2. `cut-report.json` **同时**记录：源 sha256、实际执行的命令、产物 sha256、各校验结果、
   以及**自动读出的** `cut_multiple`（不再依赖人工）；
3. 源不合法（切点非整数倍 / 缺必需张量）⇒ **拒绝，且不产出 report**（fail-closed）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "scripts" / "cut_layers_pipeline.py"
ARCH = "qwen35"
N_LAYERS = 8
INTERVAL = 4


def _is_recurrent(local: int) -> bool:
    return (local + 1) % INTERVAL != 0


_LINEAR_REQUIRED = (
    "attn_norm.weight", "post_attention_norm.weight",
    "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_beta.weight",
    "ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)
_FULL_REQUIRED = (
    "attn_norm.weight", "post_attention_norm.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
)


def _make_gguf(path: Path, *, drop_suffix: str | None = None) -> None:
    """造**完整**的迷你 qwen35 源（以便通过 #67-④ 的必需表）。"""
    import gguf
    import numpy as np

    writer = gguf.GGUFWriter(str(path), ARCH)
    writer.add_block_count(N_LAYERS)
    writer.add_embedding_length(8)
    writer.add_key_value(f"{ARCH}.full_attention_interval", INTERVAL,
                         gguf.GGUFValueType.INT32)
    for il in range(N_LAYERS):
        recr = _is_recurrent(il)
        for suffix in (_LINEAR_REQUIRED if recr else _FULL_REQUIRED):
            if drop_suffix and il == 4 and suffix == drop_suffix:
                continue
            writer.add_tensor(f"blk.{il}.{suffix}", np.ones(8, dtype=np.float32))
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
        [sys.executable, str(PIPELINE), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


class TestCutLayersPipeline:
    def test_produces_artifact_manifest_and_report(self, tmp_path):
        """一次调用产三件套，且 report 串起了契约要求的四件事。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        outdir = tmp_path / "out"
        r = _run("--src", str(src), "--k", "4", "--outdir", str(outdir))
        assert r.returncode == 0, (r.stdout, r.stderr)

        ggufs = list(outdir.glob("*.gguf"))
        manifests = list(outdir.glob("*.manifest.json"))
        reports = list(outdir.glob("cut-report.json"))
        assert ggufs, f"未产出工件：{list(outdir.iterdir())}"
        assert manifests, "未产出 manifest"
        assert reports, "未产出 cut-report.json"

        rep = json.loads(reports[0].read_text(encoding="utf-8"))
        # ① 源 sha
        assert len(rep["source"]["sha256"]) == 64
        # ② 命令（可复算）
        assert isinstance(rep["command"], list) and rep["command"], rep["command"]
        assert any("cut_layers.py" in part for part in rep["command"])
        # ③ 产物 sha（与 manifest 一致）
        mf = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert rep["artifact"]["sha256"] == mf["artifact_sha256"]
        # ④ 校验结果
        assert rep["checks"]["cut_point_legal"] is True
        # 自动读出的 cut_multiple（不再依赖人工）
        assert rep["source"]["cut_multiple"] == INTERVAL

    def test_rejects_illegal_cut_point_and_writes_no_report(self, tmp_path):
        """★ 该红必须红：切点非 interval 整数倍 ⇒ 拒绝且不留 report。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        outdir = tmp_path / "out"
        r = _run("--src", str(src), "--k", "3", "--outdir", str(outdir))
        assert r.returncode != 0, r.stdout
        assert not list(outdir.glob("cut-report.json")), "失败时不得留下 report"

    def test_rejects_missing_required_tensor(self, tmp_path):
        """源缺必需张量（#67-④）⇒ 拒绝（wrapper 必须把校验串进来，不能绕过）。"""
        src = tmp_path / "bad.gguf"
        _make_gguf(src, drop_suffix="ssm_conv1d.weight")
        outdir = tmp_path / "out"
        r = _run("--src", str(src), "--k", "4", "--outdir", str(outdir))
        assert r.returncode != 0, r.stdout
        assert not list(outdir.glob("cut-report.json"))

    def test_report_records_artifact_and_interval(self, tmp_path):
        """report 里的工件条目要能唯一定位产物。"""
        src = tmp_path / "src.gguf"
        _make_gguf(src)
        outdir = tmp_path / "out"
        r = _run("--src", str(src), "--k", "4", "--outdir", str(outdir))
        assert r.returncode == 0, (r.stdout, r.stderr)
        rep = json.loads((outdir / "cut-report.json").read_text(encoding="utf-8"))
        art = Path(rep["artifact"]["path"])
        assert art.exists(), f"report 指向的工件不存在：{art}"
        assert rep["source"]["architecture"] == ARCH
        assert rep["source"]["source_layer_range"] == [4, N_LAYERS]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
