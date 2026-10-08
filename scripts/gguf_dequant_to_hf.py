"""把 GGUF 权重**反量化**成 HF（safetensors）目录 —— 跨框架"共享量化网格"的构造工具。

为什么需要它
------------
`XFRAME-3` 的 8/8 与 `--same-engine` 的 4/4 都是 **llama.cpp ↔ llama.cpp**（同引擎同量化），
**不是生产形态**。生产 D→L 是 **PyTorch 上游 ↔ llama.cpp 下游**，两边各自持有一套权重：
默认情况下 PyTorch 拿 fp32 原始权重、llama.cpp 拿 4-bit 量化权重 ⇒ 切点两侧**网格不同**，
`rel_err` 达 2e-01 级（真链路实测）。

要验证「**共享量化网格 ⇒ 差异降回实现层下界**」，必须让 PyTorch 侧也用**同一份量化权重**。
本工具把 GGUF 里每个张量按其**自身**的 `tensor_type` 反量化（Q4_K/Q6_K/Q5_0/Q8_0/F32 混合），
按 HF 命名与排布写成 safetensors ⇒ PyTorch 与 llama.cpp **逐张量同源**。

命名/排布约定（Qwen2 系，已核验）
--------------------------------
* `token_embd.weight` → `model.embed_tokens.weight`，`output_norm.weight` → `model.norm.weight`，
  `blk.{i}.{suffix}` → `model.layers.{i}.{hf_suffix}`（后缀表见 `_BLOCK_SUFFIX_MAP`）；
* `output.weight` → `lm_head.weight`，但 `tie_word_embeddings=True` 的模型 GGUF 里它虽存在、
  HF 侧并不需要（HF 复用 embed）⇒ 默认**跳过**并记录；
* ⚠️ **排布**：GGUF 的 `tensor.shape` 记 `[in, out]`（llama.cpp 惯例），但**数据按 `[out, in]`
  存储**，`gguf.quants.dequantize` 实测**已按 HF 排布还原**。故本工具**不无条件转置**，而是
  按 `orient()` 做形状判定（任一侧不吻合即 fail-loud）—— 首版写死"一律转置"曾导致 121 个
  2-D 张量全部错位、HF 加载直接 `ignore_mismatched_sizes` 报错。

用法::

    python scripts/gguf_dequant_to_hf.py \
        --gguf models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        --dst build/keephead/qwen25-05b-q4km-dequant \
        --tokenizer-src models/qwen2.5-0.5b-instruct
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

# llama.cpp 张量后缀 → HF 后缀（单层内）
_BLOCK_SUFFIX_MAP: dict[str, str] = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_TOP_LEVEL_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def hf_name_for(gguf_name: str) -> str | None:
    """GGUF 张量名 → HF 参数名；未知名字返回 None（纯函数，调用方负责 fail-loud）。"""
    if gguf_name in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[gguf_name]
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".")
        if len(parts) < 3:
            return None
        idx, rest = parts[1], ".".join(parts[2:])
        suffix = _BLOCK_SUFFIX_MAP.get(rest)
        if suffix is None:
            return None
        return f"model.layers.{idx}.{suffix}"
    return None


def orient(arr, *, gguf_shape, hf_name: str):
    """把反量化结果摆成 HF 排布（纯函数）。

    ★ 实测（gguf 库 + Qwen2.5-0.5B Q4_K_M）：`gguf.quants.dequantize` **已经返回 HF 排布** ——
    GGUF 的 `tensor.shape` 记的是 `[in, out]`（llama.cpp 惯例），但**数据按 `[out, in]` 存储**，
    反量化按存储顺序还原。不同 gguf 版本可能不同 ⇒ 这里**按形状判定**，两边都不吻合就 fail-loud，
    绝不静默转置错。
    """
    if arr.ndim <= 1:
        return arr
    declared = tuple(int(x) for x in gguf_shape)
    wanted = tuple(reversed(declared))
    if tuple(arr.shape) == wanted:
        return arr
    if tuple(arr.shape) == declared:
        return arr.T
    raise ValueError(
        f"{hf_name}: 反量化形状 {tuple(arr.shape)} 既不等于期望 {wanted} 也不等于声明 {declared}"
    )


def _dequant(tensor):
    """按张量自身的 `tensor_type` 反量化；F32 直接 view。"""
    import numpy as np
    from gguf import quants

    raw = tensor.data
    if str(tensor.tensor_type) == "0":  # F32
        return np.asarray(raw, dtype=np.float32)
    return np.asarray(quants.dequantize(raw, tensor.tensor_type), dtype=np.float32)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GGUF → HF safetensors（反量化，供跨框架共享网格）")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer-src", help="复制 config/tokenizer 的 HF 目录（推荐给出）")
    ap.add_argument("--keep-tied-output", action="store_true",
                    help="保留 GGUF 的 output.weight 为 lm_head.weight（默认在 tied 模型上跳过）")
    ap.add_argument("--manifest", help="输出清单 JSON（含每个张量的 qtype 与形状）")
    ap.add_argument("--dry-run", action="store_true", help="只报告映射与统计，不写文件")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np
    import gguf

    src = pathlib.Path(args.gguf)
    dst = pathlib.Path(args.dst)
    if not src.is_file():
        print(f"SKIP: 缺 GGUF {src}")
        return 0

    reader = gguf.GGUFReader(str(src))
    tensors = {}
    manifest = []
    unknown: list[str] = []
    skipped: list[str] = []
    for t in reader.tensors:
        hf = hf_name_for(t.name)
        if hf is None:
            unknown.append(t.name)
            continue
        if hf == "lm_head.weight" and not args.keep_tied_output:
            skipped.append((t.name, "tie_word_embeddings ⇒ HF 复用 embed_tokens"))
            continue
        arr = orient(_dequant(t), gguf_shape=[int(x) for x in t.shape], hf_name=hf)
        tensors[hf] = np.ascontiguousarray(arr, dtype=np.float32)
        manifest.append({"gguf": t.name, "hf": hf, "qtype": str(t.tensor_type),
                         "shape": list(tensors[hf].shape)})

    if unknown:
        print(f"[FAIL] 有 {len(unknown)} 个 GGUF 张量没有映射（fail-loud，避免静默丢权重）：")
        for name in unknown[:8]:
            print("   -", name)
        return 2

    print(f"[plan] 映射 {len(tensors)} 个张量；跳过 {len(skipped)} 个")
    for name, why in skipped:
        print(f"   [skip] {name}: {why}")
    if args.dry_run:
        print("[dry-run] 不写文件。")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    if args.tokenizer_src:
        ts = pathlib.Path(args.tokenizer_src)
        if not ts.is_dir():
            print(f"SKIP: 缺 tokenizer 源目录 {ts}")
            return 0
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "vocab.json", "merges.txt"):
            p = ts / name
            if p.is_file():
                shutil.copy2(p, dst / name)
        print(f"[copy] tokenizer/config ← {ts}")

    import torch
    from safetensors.torch import save_file

    payload = {k: torch.from_numpy(np.ascontiguousarray(v).copy()) for k, v in tensors.items()}
    save_file(payload, str(dst / "model.safetensors"))
    print(f"[done] {dst / 'model.safetensors'}（{len(payload)} 张量）")

    if args.manifest:
        mp = pathlib.Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"source": str(src), "dst": str(dst),
                                  "n_tensors": len(payload), "tensors": manifest},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[manifest] {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
