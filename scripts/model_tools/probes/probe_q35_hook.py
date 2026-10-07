"""定位 qwen3.5 上 shim 的正确挂点。

思路：整模（HF，PyTorch）能给出任意层的中间量，用它当"真值"，逐一试探 shim 的
两种模式 × 不同 cut_layer，找出**真正等价**的那个组合。

候选真值（都由 PyTorch 整模产出，prompt 4 tokens，末位 token 比对）：
  A. blk.15 的输出（= 第 16 层的输入）        ← 接力真正要的（output_norm 之前）
  B. 全部 24 层后的 output_norm 之前          ← 等价于 A 当 K=24
  C. output_norm 之后                        ← 多一次 RMSNorm

shim 侧试探：
  1. mode="nextn"（head16 工件）              → 现用，已知 rel_err=0.84
  2. mode="layer_inp", cut_layer=16（整模）   → 语义等价候选
  3. mode="layer_inp", cut_layer=24（整模）   → 等价于 B

跑法：<venv>/python scripts/model_tools/probes/probe_q35_hook.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

HEAD_GGUF = ROOT / "build" / "keephead" / "master_artifacts" / "q35-2b-head16.gguf"
WHOLE_GGUF = ROOT / "models" / "qwen3-5-2b-gguf" / "qwen35-2b-Q4_K_M.gguf"
SHIM = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
HF_DIR = ROOT / "models" / "qwen3-5-2b"
PROMPT = "1+1=?"
K = 16


def stats(ref, got):
    a = np.asarray(ref, dtype=np.float64).reshape(-1)
    b = np.asarray(got, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return f"shape mismatch {a.shape} vs {b.shape}"
    denom = max(float(np.abs(a).max()), 1e-12)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return (f"rel_err={np.abs(a-b).max()/denom:.3e} cos={cos:.6f}")


def main():
    from transformers import AutoTokenizer
    from model_module import ModelManager

    tok = AutoTokenizer.from_pretrained(
        str(HF_DIR), trust_remote_code=True, local_files_only=True
    )
    ids = tok(PROMPT, return_tensors="pt").input_ids
    print(f"  prompt={PROMPT!r} n_tokens={ids.shape[1]}")

    # ---- 真值：用主仓引擎的层段加载（不要手写逐层前向：
    #      Qwen3_5DecoderLayer.forward() 需要 position_embeddings，手写必错）----
    def seg_hidden(start, end):
        mgr = ModelManager()
        mgr.load_model(model_path=str(HF_DIR), quant_type="none", engine="pytorch")
        mgr.load_layer_range(
            start, end,
            has_embedding=(start == 0), has_lm_head=False,
            model_path=str(HF_DIR), total_layers=24,
        )
        res = mgr.forward_layers(input_ids=ids, hidden_states=None, apply_lm_head=False)
        if isinstance(res, dict):
            res = res.get("hidden_states")
        return np.asarray(res, dtype=np.float32)[0, -1]

    ref_blk15 = seg_hidden(0, K)          # 前 16 层输出 = blk.15 的输出
    ref_last = seg_hidden(0, 24)          # 全 24 层输出（output_norm 之前）
    print(f"  ref blk15 |max|={np.abs(ref_blk15).max():.2f}  ref last |max|={np.abs(ref_last).max():.2f}")

    # ---- shim 三种试探 ----
    from llama_keep_head import KeepHeadUpstream

    toks = ids[0].tolist()
    trials = [
        ("nextn(head16)", dict(mode="nextn")),
        (f"layer_inp(cut={K},whole)", dict(mode="layer_inp", cut_layer=K)),
        (f"layer_inp(cut=24,whole)", dict(mode="layer_inp", cut_layer=24)),
    ]
    for label, kw in trials:
        gguf = HEAD_GGUF if kw.get("mode") == "nextn" else WHOLE_GGUF
        try:
            up = KeepHeadUpstream(str(SHIM), str(gguf), n_ctx=512, **kw)
            out = np.asarray(up.forward_tokens_to_hidden(toks), dtype=np.float32)
            if out.ndim == 3:
                out = out[0]
            got = out[-1]
            print(f"  [{label}] shape={out.shape}")
            print(f"      vs blk.15 out : {stats(ref_blk15, got)}")
            print(f"      vs last out   : {stats(ref_last, got)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] 失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
