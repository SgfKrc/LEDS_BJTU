#!/usr/bin/env python
"""relay_diag_head_norm.py — L→L 上游通道语义诊断（P1 守卫的证据来源）。

问题：`llama_engine.forward_layers_to_hidden()` 底层是 `llama_set_embeddings` +
`llama_get_embeddings_ith`。若该通道返回的是 **`output_norm(H)`**，它就不能当"层接力上游"
（接力需要未过 final norm 的层输出），否则下游必然与整模对拍分叉。

诊断方法：同一 prompt 下
  * PyTorch 上游（`ModelManager.load_layer_range(0, K)` + `forward_layers`）给出第 K 层输出 `H`；
  * head 裁层 GGUF（前 K 层）经 embeddings 通道给出 `E`；
  * 用整模的 `model.norm.weight` 复现 `RMSNorm(H) * w`，比较它与 `E` 的 `rel_err` / `cos`。

判据：`cos(E, RMSNorm(H)*w) > 0.9999` 且 `rel_err < 1e-2` ⇒ 通道 = `output_norm(H)`（守卫依据）。

工件缺失（HF 目录 / head GGUF / safetensors）时按本仓惯例**打印 SKIP 并退出 0**，不伪装成通过。

用法::

    python scripts/relay_diag_head_norm.py \
        --model-dir models/qwen2.5-0.5b-instruct \
        --head-model build/cross-framework-layer-poc/out/qwen25-05b-f16-head12.gguf \
        --layers 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="L→L 上游通道（embeddings vs output_norm）诊断")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head-model", required=True, help="前 K 层的裁层 GGUF（head 工件）")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--prompt", default=str(ROOT / "build" / "cross-framework-layer-poc" / "out"
                                            / "prompt-france.txt"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = Path(args.model_dir)
    head = Path(args.head_model)
    prompt_path = Path(args.prompt)
    for path in (model_dir, head, prompt_path):
        if not path.exists():
            print(f"SKIP: 缺工件 {path}")
            return 0

    import numpy as np
    import torch

    import config as cfg
    import model_module
    from llama_engine import LlamaCppEngine
    from transformers import AutoTokenizer

    cfg.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    cfg.USE_COMPILE = False
    model_module.USE_COMPILE = False

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    ids = tok(prompt_path.read_text(encoding="utf-8").strip(), add_special_tokens=False)["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    print(f"[prompt] {len(ids)} tokens")

    mgr = model_module.ModelManager()
    mgr.load_layer_range(0, args.layers, has_embedding=True, has_lm_head=False,
                         model_path=str(model_dir))
    device = mgr.get_device()
    with torch.no_grad():
        out = mgr.forward_layers(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                                 past_key_values=None, use_cache=True)
    hidden = np.asarray(out["hidden_states"][0].to(torch.float32).cpu().numpy())
    print(f"[pytorch] layer{args.layers} hidden {hidden.shape}")

    engine = LlamaCppEngine()
    engine.load_model(model_path=str(head), n_ctx=512, n_threads=4)
    embeddings = np.asarray(engine.forward_layers_to_hidden(ids, n_past=0, all_positions=True),
                            dtype=np.float32)
    print(f"[llama]   embeddings   {embeddings.shape}")

    last_h, last_e = hidden[-1], embeddings[-1]
    rel = float(np.linalg.norm(last_e - last_h) / np.linalg.norm(last_h))
    cos = float(np.dot(last_e, last_h) / (np.linalg.norm(last_e) * np.linalg.norm(last_h)))
    print(f"[raw]     rel_err={rel:.4f} cos={cos:.6f} "
          f"rms(H)={float(np.sqrt((last_h ** 2).mean())):.4f} "
          f"rms(E)={float(np.sqrt((last_e ** 2).mean())):.4f}")

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("SKIP: 缺 safetensors，无法取 output_norm 权重")
        return 0
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        print("SKIP: 目录里没有 safetensors")
        return 0
    tensors = {}
    for shard in shards:
        tensors.update(load_file(str(shard)))
    norm_key = "model.norm.weight" if "model.norm.weight" in tensors else next(
        (k for k in tensors if k.endswith(".norm.weight") and ".layers." not in k), None)
    if norm_key is None:
        print("SKIP: 找不到 output_norm 权重")
        return 0

    weight = tensors[norm_key].to(torch.float32).numpy()
    eps = 1e-6
    normed = (last_h / np.sqrt((last_h ** 2).mean() + eps)) * weight
    rel_normed = float(np.linalg.norm(last_e - normed) / np.linalg.norm(normed))
    cos_normed = float(np.dot(last_e, normed)
                       / (np.linalg.norm(last_e) * np.linalg.norm(normed)))
    print(f"[normed]  rel_err={rel_normed:.4f} cos={cos_normed:.6f} (权重键 {norm_key})")

    if cos_normed > 0.9999 and rel_normed < 1e-2:
        print("[verdict] embeddings 通道 = output_norm(H) ⇒ 不能直接当层接力上游"
              "（统一驱动 scripts/relay_experiment.py 对 l2l_llama 默认 fail-loud）")
        return 0
    print("[verdict] embeddings 通道未复现 output_norm ⇒ 需继续查因")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
