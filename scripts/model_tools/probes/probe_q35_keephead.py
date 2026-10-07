"""验证：shim 的 KeepHeadUpstream(nextn) 在 qwen3.5 上取的是否为 output_norm 之前的残差流。

判据：与整模（PyTorch，apply_lm_head=False）的 [0,K) 层输出逐元素对比。
- 若 rel_err ~1e-7 量级 ⇒ shim 挂点正确（master 的 qwen3* 拒绝判据已过时）
- 若 rel_err ~1e-3 量级且 cos<1 ⇒ 多/少一次 RMSNorm ⇒ 挂点确实不对

跑法（在仓库根目录、venv 里）：
  <venv>/python scripts/model_tools/probes/probe_q35_keephead.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

HEAD_GGUF = ROOT / "build" / "keephead" / "master_artifacts" / "q35-2b-head16.gguf"
SHIM = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
HF_DIR = ROOT / "models" / "qwen3-5-2b"
PROMPT = "1+1=?"


def tokenize():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(HF_DIR), trust_remote_code=True, local_files_only=True
    )
    return tok, tok(PROMPT, return_tensors="pt").input_ids


def shim_hidden(ids):
    from llama_keep_head import KeepHeadUpstream

    up = KeepHeadUpstream(str(SHIM), str(HEAD_GGUF), mode="nextn", n_ctx=512)
    toks = ids[0].tolist() if hasattr(ids, "tolist") else list(ids)
    if toks and isinstance(toks[0], list):
        toks = toks[0]
    out = up.forward_tokens_to_hidden(toks)
    return np.asarray(out, dtype=np.float32)


def torch_hidden(ids):
    import torch
    from model_module import ModelManager

    mgr = ModelManager()
    mgr.load_model(model_path=str(HF_DIR), quant_type="none", engine="pytorch")
    with torch.no_grad():
        res = mgr.forward_layers(
            input_ids=ids, hidden_states=None, apply_lm_head=False,
        )
    # 非末节点返回 {"hidden_states": Tensor}
    if isinstance(res, dict):
        res = res.get("hidden_states")
    return res


def stats(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    denom = max(float(np.abs(a).max()), 1e-12)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return (f"max_abs={diff.max():.3e} rel_err={diff.max()/denom:.3e} "
            f"cos={cos:.6f} |ref|max={np.abs(a).max():.1f}")


def main():
    tok, ids = tokenize()
    print(f"  prompt={PROMPT!r} ids={ids.tolist()} n_tokens={ids.shape[1]}")
    sh = shim_hidden(ids)
    print(f"  shim(nextn)  shape={np.asarray(sh).shape} dtype={np.asarray(sh).dtype}")
    th = torch_hidden(ids)
    print(f"  torch        type={type(th)}")
    th_arr = th[0] if isinstance(th, tuple) else th
    th_arr = np.asarray(th_arr, dtype=np.float32)
    if th_arr.ndim == 3:
        th_arr = th_arr[0]
    print(f"  torch        shape={th_arr.shape}")
    print("  --- 对比（末位 token） ---")
    a = np.asarray(sh, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    print("  ", stats(th_arr[-1], a[-1]))


if __name__ == "__main__":
    main()
