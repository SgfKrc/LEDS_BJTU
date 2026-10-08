"""量化「激活」层：llama.cpp 对量化权重的 matmul 会把 F32 激活先量化再整数点积。

为什么需要测它
--------------
跨框架接力（PyTorch ↔ llama.cpp）的残差此前被归因于"权重网格不同"。但权重同源（逐位相同）
之后，4-bit 通路仍有 `3e-3` 级残差，而 f16 通路只有 `1.6e-4`。真因在 **matmul 内部**：

`ggml-cpu.c:1266-1344`（`ggml_compute_forward_mul_mat`）——
```
vec_dot_type  = type_traits_cpu[src0->type].vec_dot_type;   // 权重类型决定
from_float    = type_traits_cpu[vec_dot_type].from_float;
GGML_ASSERT(src1->type == GGML_TYPE_F32);                   // 激活是 F32
from_float((float *) src1->data ..., (void *) wdata ..., ...);   // ← 激活被写进量化缓冲
```
量化权重（`Q5_0`/`Q4_K`/`Q6_K`/`Q8_0`）的 `vec_dot_type` 是 `Q8_0`/`Q8_K` ⇒ **激活被量化**；
F16 权重的是 `F16` ⇒ 只做 f32→f16 转换、**不量化**。

本工具量出这一环的**单层**相对差异，作为 4-bit 跨框架通路的可证下界（对照 XFRAME-4 的
`O(K·ε) ≈ 3.5e-04`）：同一份真实激活 `x` 与权重 `W`，
    `a = x @ Wᵀ`（精确 f32） vs `b = q8_0(x) @ Wᵀ`（模拟 llama.cpp 的激活量化）
给出 `rel(a,b)`。**同权重**下 `rel(a,a) ≡ 0`，故该量全部来自激活量化。

用法::

    python scripts/xframe_actquant_report.py \
        --dequant-dir build/keephead/qwen25-05b-q4km-dequant \
        --prompt "Explain in one short sentence why floating point addition is not associative." \
        --out build/keephead/xframe-actquant.json

工件缺失时按本仓惯例 **打印 SKIP 并退出 0**。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 被测模块的**层内后缀**（覆盖 attn 与 mlp 两种形状）；前缀由 `_find_layers_prefix` 探测
_TARGET_SUFFIXES = [
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.down_proj", "down_proj"),
    ("mlp.gate_proj", "gate_proj"),
]


def _find_layers_prefix(model) -> str | None:
    """探测 text transformer 的 `layers` 路径（纯函数式走法）。

    Qwen2 系是 `model.layers`，Qwen3.5 是 `model.language_model.layers` —— 写死任一个都会
    在另一架构上报 `AttributeError`。返回前缀字符串（如 `model.layers`），找不到返回 None。
    """
    for prefix in ("model.layers", "model.language_model.layers",
                   "language_model.layers", "layers"):
        mod = model
        for part in prefix.split("."):
            if not hasattr(mod, part):
                mod = None
                break
            mod = getattr(mod, part)
        if mod is not None and hasattr(mod, "__len__") and len(mod) > 0:
            return prefix
    return None


def q8_0_quant(x, *, block: int = 32):
    """llama.cpp `quantize_row_q8_0` 的语义：**per-`block` 对称量化**、scale 存 f16（纯函数）。

    `d = amax/127`、`q = round(x/d)`（由 `amax` 保证 |q| ≤ 127，无需 clamp）；
    scale 以 **f16** 存储（llama.cpp 的 `y[i].d` 是 `ggml_fp16_t`）⇒ 这里也做 `.half().float()`。
    最后一维不是 `block` 整数倍时**原样返回**（llama.cpp 要求对齐，不对齐会走别的路径）。
    """
    import torch

    n = x.shape[-1]
    if n % block:
        return x
    orig = x.shape
    xb = x.reshape(-1, n // block, block).float()
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).half().float()
    q = torch.round(xb / scale)
    return (q * scale).reshape(orig).to(x.dtype)


def rel(a, b) -> float:
    """相对 L2 差（纯函数）。"""
    import torch

    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    na = a.norm()
    return float((a - b).norm() / na) if na else 0.0


def _locate(model, dotted: str):
    """按 `layers.0.self_attn.q_proj` 形式的点分路径取子模块（纯函数式的属性/下标走法）。"""
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="量化「激活」层的单层相对差异（4-bit 跨框架通路）")
    ap.add_argument("--dequant-dir", required=True,
                    help="GGUF 反量化出的 HF 目录（权重须与下游 llama.cpp 同源，见 "
                         "scripts/gguf_dequant_to_hf.py）")
    ap.add_argument("--prompt", default="Explain in one short sentence why floating point "
                                        "addition is not associative.")
    ap.add_argument("--targets", default=None,
                    help="逗号分隔的 `dotted.path=label`；缺省用内置 4 个 attn/mlp 模块")
    ap.add_argument("--out", help="把结果写成 JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = pathlib.Path(args.dequant_dir)
    if not model_dir.is_dir():
        print(f"SKIP: 缺反量化目录 {model_dir}")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    targets = None
    if args.targets:
        targets = []
        for item in args.targets.split(","):
            dotted, _, label = item.strip().partition("=")
            targets.append((dotted, label or dotted))

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, trust_remote_code=False).eval()

    if targets is None:
        prefix = _find_layers_prefix(model)
        if prefix is None:
            print("SKIP: 探测不到 text transformer 的 `layers` 路径 —— 请用 --targets 显式给出")
            return 0
        print(f"[layers prefix] {prefix}")
        targets = [(f"{prefix}.0.{suffix}", f"L0.{label}") for suffix, label in _TARGET_SUFFIXES]
        targets.append((f"{prefix}.8.mlp.gate_proj", "L8.gate_proj"))

    captured: dict[str, object] = {}
    hooks = []
    for dotted, label in targets:
        mod = _locate(model, dotted)
        hooks.append(mod.register_forward_pre_hook(
            lambda _m, a, n=label: captured.setdefault(n, a[0].detach())))
    with torch.no_grad():
        model(torch.tensor([ids]))
    for h in hooks:
        h.remove()

    rows = []
    print(f"{'module':16s} {'W shape':>16s} {'rel(精确, 量化激活)':>20s} {'rel(精确, 精确)':>16s}")
    for dotted, label in targets:
        if label not in captured:
            print(f"{label:16s} (未捕获到输入，跳过)")
            continue
        mod = _locate(model, dotted)
        x = captured[label].float()
        w = mod.weight.detach().float()
        a = torch.nn.functional.linear(x, w)
        b = torch.nn.functional.linear(q8_0_quant(x), w)
        c = torch.nn.functional.linear(x, w)
        row = {"module": dotted, "label": label, "w_shape": list(w.shape),
               "rel_quantized_activation": rel(a, b), "rel_control": rel(a, c)}
        rows.append(row)
        print(f"{label:16s} {str(tuple(w.shape)):>16s} {row['rel_quantized_activation']:20.3e} "
              f"{row['rel_control']:16.3e}")

    vals = [r["rel_quantized_activation"] for r in rows if r["rel_quantized_activation"] > 0]
    if vals:
        print(f"\n[summary] n={len(vals)} min={min(vals):.3e} max={max(vals):.3e} "
              f"（对照：XFRAME-4 的归约下界 3.5e-04，f16 通路的端到端 1.6e-4…4.4e-4）")
    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"mode": "actquant", "dequant_dir": str(model_dir),
                                    "prompt": args.prompt, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
