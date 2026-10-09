#!/usr/bin/env python
"""xframe_int8_grid_report.py — int8 共享网格 + **整型累加**：GEMM 输出是否与归约顺序无关。

为什么需要它
------------
XFRAME 已分别证明两件事，但**没有证明第三件**：
* `XFRAME-3`：把**交界处的 hidden** 量化到 int8 逐通道共享网格 ⇒ 跨引擎 argmax 被**压回同格**
  （3/3、逐 token 8/8）；
* `XFRAME-5`：把累加器换成 **float64** ⇒ "只改归约顺序"的差异降到 `0.0`；
* **缺**：上游 GEMM 若走 **int8 网格 + int32 累加**，其输出是否**与归约顺序完全无关**
  （= 跨线程/跨 SIMD 宽度/跨 ISA 都可复现）—— 这是机理文档 §9 的 **L3** 那一环，
  也是"跨框架接力能否从**概率性一致**变成**代数保证**"的关键。

判据
----
对同一份真实权重 `W` 与真实激活 `x`，用**不同的 K 维分块数**（`order`，模拟不同归约树）：

* `f32` GEMM：结果应随 `order` **变化**（这是 XFRAME-4 的 `O(K·ε)` 下界）；
* `int8 网格 + int32 累加`：结果应随 `order` **逐位不变**（整数加满足结合律与交换律）。

同时报出 int8 网格自身的量化误差（相对 f32 参考），与"半格"口径对照。

用法::

    python scripts/xframe_int8_grid_report.py \
        --dequant-dir build/keephead/qwen25-05b-q4km-dequant \
        --orders 1,2,4,8 --out build/keephead/xframe-int8-grid.json

工件缺失时按本仓惯例 **打印 SKIP 并退出 0**。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 被测模块（覆盖 attn 与 mlp 两种形状）
DEFAULT_TARGETS = ("mlp.down_proj", "self_attn.o_proj")

#: 块级网格的块大小（与 ggml 的 `QK8_0` 一致）
BLOCK = 32


def quantize_per_channel(x, scale, levels: float = 127.0):
    """按**给定**的 scale 量化到 `±levels`（共享网格：scale 由调用方提供，纯函数，按广播语义）。

    与 `relay_hidden_quant.encode_hidden` 的口径一致：`q = round(x / scale * levels)`。
    `scale` 为 0 处按 1.0 兜底（否则整行/整列塌成 0）。

    ⚠️ `scale` 必须在**被累加掉的那个维度上取标量**（W 用 per-output-channel、x 用 per-token）——
    若沿 K 维逐元素给 scale，`Σ qx·qw` 之后 scale 信息就丢了、无法广播回来。
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    return np.clip(np.round(arr / s * levels), -levels, levels)


def dequantize_parts(q_acc, x_scale, w_scale, levels: float = 127.0):
    """把 `Σ qx·qw` 的 int32 累加和乘回两侧 scale（纯函数）。

    `Σ (qx·sx/127)(qw·sw/127) ≈ Σ x·w` ⇒ `out = acc · sx[m] · sw[n] / levels²`。
    """
    import numpy as np

    sx = np.asarray(x_scale, dtype=np.float32)
    sw = np.asarray(w_scale, dtype=np.float32)
    sx = np.where(sx == 0, np.float32(1.0), sx)
    sw = np.where(sw == 0, np.float32(1.0), sw)
    return (np.asarray(q_acc, dtype=np.float64) * sx[:, None] * sw[None, :]
            / (float(levels) ** 2)).astype(np.float32)


def split_slices(k: int, order: int) -> list[slice]:
    """把 K 维均分成 `order` 段（最后一段吃余数）；`order<=1` ⇒ 单段（纯函数）。"""
    order = max(1, int(order))
    if order == 1 or k < order:
        return [slice(0, k)]
    bounds = [i * k // order for i in range(order)] + [k]
    return [slice(bounds[i], bounds[i + 1]) for i in range(order)]


def split_matmul_f32(x, w, order: int):
    """f32 GEMM，**按 `order` 段分别求和后再逐段合并**（模拟不同的归约树，纯函数）。

    `x` 为 `[M, K]`、`w` 为 `[N, K]` ⇒ 返回 `[M, N]`。段内仍是 `x·wᵀ`（矩阵乘自身顺序不变），
    变化的是**段间合并顺序与分段边界** —— 这正是不同线程数/SIMD 宽度造成的差异来源。
    """
    import numpy as np

    xa = np.asarray(x, dtype=np.float32)
    wa = np.asarray(w, dtype=np.float32)
    acc = None
    for sl in split_slices(xa.shape[1], order):
        part = xa[:, sl] @ wa[:, sl].T
        acc = part if acc is None else acc + part
    return acc


def split_matmul_int32(qx, qw, order: int):
    """int8 网格上的 GEMM：**int32 累加**，同样按 `order` 分段（纯函数）。

    整数加满足结合律与交换律 ⇒ 与 `order` **无关**（这正是 L3 的立足点）。
    返回 int32 的 `[M, N]` 累加和（尚未乘回 scale）。
    """
    import numpy as np

    qxa = np.asarray(qx, dtype=np.int32)
    qwa = np.asarray(qw, dtype=np.int32)
    acc = None
    for sl in split_slices(qxa.shape[1], order):
        part = qxa[:, sl] @ qwa[:, sl].T
        acc = part if acc is None else acc + part
    return acc


def blockwise_quant(x, block: int = 32, levels: float = 127.0):
    """块级（沿 K 分块，每块一个标量 scale）对称量化 ⇒ `(q, scale)`（纯函数）。

    返回 `q` 形状 `[..., nb, block]`、`scale` 形状 `[..., nb]`（末维不足时右侧补零）。
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    n = arr.shape[-1]
    nb = (n + block - 1) // block
    if nb * block != n:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, nb * block - n)])
    xb = arr.reshape(*arr.shape[:-1], nb, block)
    s = np.abs(xb).max(axis=-1)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.clip(np.round(xb / s[..., None] * levels), -levels, levels)
    return q, s


def split_matmul_blockwise(qx, sx, qw, sw, order: int, levels: float = 127.0):
    """与 ggml `vec_dot` **同构**：块内整数点积 + **块间浮点加权累加**（纯函数）。

    `sumf += d * sumi`（每块一次浮点加）⇒ 量化误差小（块内动态范围窄），但块间仍是浮点加法
    ⇒ 对 `order` **仍有**依赖。实测（K=2048/6144、`order=1,2,4,8`）：块级的顺序敏感度与逐元素
    f32 同量级（`9.3e-08` vs `2.2e-07`，**不是**按块大小线性缩小）—— 因为两者在这一维度上的
    差异都来自"段间合并顺序"（段数同为 `order`），段内归约各自由库实现且对同一分段确定。
    要点是它**非零**：llama.cpp 的量化 GEMM 不是纯整数累加。
    """
    import numpy as np

    sumi = np.einsum("mbk,nbk->mnb", np.asarray(qx, dtype=np.int32), np.asarray(qw, dtype=np.int32))
    scale = (np.asarray(sx, dtype=np.float32)[:, None, :]
             * np.asarray(sw, dtype=np.float32)[None, :, :]) / (float(levels) ** 2)
    contrib = sumi.astype(np.float32) * scale
    acc = None
    for sl in split_slices(contrib.shape[-1], order):
        part = contrib[:, :, sl].sum(axis=-1)
        acc = part if acc is None else acc + part
    return acc


def rel_err(reference, actual) -> float:
    """相对 L2 差（纯函数）。"""
    import numpy as np

    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    nr = float(np.linalg.norm(r))
    return float(np.linalg.norm(r - a) / nr) if nr else 0.0


def _locate(model, dotted: str):
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def _find_layers_prefix(model) -> str | None:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 共享网格 + 整型累加：GEMM 与归约顺序无关性")
    ap.add_argument("--dequant-dir", required=True,
                    help="GGUF 反量化出的 HF 目录（见 scripts/gguf_dequant_to_hf.py）")
    ap.add_argument("--prompt", default="Explain in one short sentence why floating point "
                                        "addition is not associative.")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="层内模块名（逗号分隔），默认覆盖 mlp 与 attn")
    ap.add_argument("--orders", default="1,2,4,8", help="K 维分块数网格（逗号分隔）")
    ap.add_argument("--out", help="把结果写成 JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = pathlib.Path(args.dequant_dir)
    if not model_dir.is_dir():
        print(f"SKIP: 缺反量化目录 {model_dir}")
        return 0

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    orders = [int(x) for x in str(args.orders).split(",") if x.strip()]
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, trust_remote_code=False).eval()
    prefix = _find_layers_prefix(model)
    if prefix is None:
        print("SKIP: 探测不到 text transformer 的 `layers` 路径")
        return 0

    dotted_all = [f"{prefix}.0.{t}" for t in targets]
    captured: dict[str, object] = {}
    hooks = []
    for dotted in dotted_all:
        hooks.append(_locate(model, dotted).register_forward_pre_hook(
            lambda _m, a, n=dotted: captured.setdefault(n, a[0].detach())))
    with torch.no_grad():
        model(torch.tensor([ids]))
    for h in hooks:
        h.remove()

    rows = []
    print(f"[layers prefix] {prefix} | orders={orders}")
    for dotted in dotted_all:
        if dotted not in captured:
            continue
        mod = _locate(model, dotted)
        w = mod.weight.detach().float().numpy()                       # [N, K]
        x = captured[dotted].float().numpy().reshape(-1, w.shape[1])  # [M, K]
        ref = split_matmul_f32(x, w, 1)

        # 共享网格：W 用 per-output-channel（沿 K 取标量）、x 用 per-token —— 两侧都落在
        # 「被累加掉的那一维上的标量 scale」上，故 int32 累加后能原样乘回来。
        w_scale = np.abs(w).max(axis=1).astype(np.float32)            # [N]
        x_scale = np.abs(x).max(axis=1).astype(np.float32)            # [M]
        qx = quantize_per_channel(x, x_scale[:, None])
        qw = quantize_per_channel(w, w_scale[:, None])
        # 块级（llama.cpp 风格）网格：每 32 元素一个 scale
        qxb, sxb = blockwise_quant(x, BLOCK)
        qwb, swb = blockwise_quant(w, BLOCK)

        f32_by_order, int_by_order, int_vals, blk_by_order = {}, {}, {}, {}
        for o in orders:
            f32_by_order[o] = split_matmul_f32(x, w, o)
            acc = split_matmul_int32(qx, qw, o)
            int_by_order[o] = acc
            int_vals[o] = dequantize_parts(acc, x_scale, w_scale)
            blk_by_order[o] = split_matmul_blockwise(qxb, sxb, qwb, swb, o)

        # 顺序敏感度：以 order=1 为基准
        f32_spread = max(rel_err(f32_by_order[1], v) for v in f32_by_order.values())
        int_spread = max(float(np.abs(int_by_order[1] - v).max()) for v in int_by_order.values())
        int_bit_identical = all(np.array_equal(int_by_order[1], v) for v in int_by_order.values())
        blk_spread = max(rel_err(blk_by_order[1], v) for v in blk_by_order.values())
        q_err = rel_err(ref, int_vals[1])
        blk_q_err = rel_err(ref, blk_by_order[1])
        # 跨库：numpy int32 vs torch int32 —— 证明"整数域跨框架/跨库也逐位一致"
        t_acc = (torch.from_numpy(qx.astype(np.int32))
                 @ torch.from_numpy(qw.astype(np.int32)).T).numpy()
        cross_lib = bool(np.array_equal(t_acc, int_by_order[1]))
        row = {
            "module": dotted, "shape_w": list(w.shape), "shape_x": list(x.shape),
            "f32_order_spread_rel": f32_spread,
            "int_order_spread_max_abs": int_spread,
            "int_bit_identical_across_orders": bool(int_bit_identical),
            "int8_row_grid_quant_rel_err": q_err,
            "blockwise_order_spread_rel": blk_spread,
            "int8_block32_quant_rel_err": blk_q_err,
            "torch_int32_matches_numpy": cross_lib,
        }
        rows.append(row)
        print(f"  {dotted.split('.')[-2:][0] + '.' + dotted.split('.')[-1]:24s} "
              f"| 顺序敏感度 f32={f32_spread:.2e} 纯整型={int_spread:.1e}(逐位同={int_bit_identical}) "
              f"块级={blk_spread:.2e} "
              f"| 量化 rel 行级={q_err:.2e} 块级32={blk_q_err:.2e} | torch==numpy={cross_lib}")

    if not rows:
        print("SKIP: 未捕获到目标模块的输入")
        return 0
    fs = [r["f32_order_spread_rel"] for r in rows]
    print(f"\n[summary] n={len(rows)} f32 顺序敏感度 {min(fs):.2e}…{max(fs):.2e}；"
          f"int32 全部逐位相同={all(r['int_bit_identical_across_orders'] for r in rows)}")
    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"mode": "int8-shared-grid", "dequant_dir": str(model_dir),
                                    "prompt": args.prompt, "orders": orders, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
