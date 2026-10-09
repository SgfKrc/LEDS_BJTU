"""XFRAME-6：hybrid 的「递推」是否引入额外不可消除项 —— 真模型可复跑工具。

设计要点（与仓库方法论一致）
--------------------------
**不手写前向**：左引擎一律复用 XFRAME-1 的正式工具 `xframe_divergence_report`
（`hf_greedy` / `_install_ggml_like_rmsnorm`）。本工具只负责**度量与判定**。

对照变量只有一项：**RMSNorm 的归约语义** ——
`rmsnorm="hf"`（HF 原版，输入 dtype 上累加）vs `rmsnorm="ggml-like"`
（ggml 语义：`double` 累加 Σx² + `1/sqrt`，并按类保留 `weight` / `1+weight` 与 gate）。

两种度量
--------
1. **teacher-forcing**（纯数值差）：把**同一条** token 序列整段前向，比较**逐位置** logits。
   不受分叉影响 ⇒ 反映"每步的数值差"，这正是"递推是否放大误差"的判据。
2. **自由解码**（是否分叉）：两侧各自 greedy 若干步，比较生成 token 序列是否一致。

判据
----
对 teacher-forcing 的逐位置 `rel` 做**线性回归**：
- 斜率 ≈ 0（|斜率| 远小于 `ε/位置`）⇒ 递推**不引入**额外不可消除项（误差只复用每步下界）；
- 斜率显著为正且达 `ε` 量级 ⇒ 存在随步数累积的额外项。

用法
----
```
python scripts/xframe6_recursion_report.py --hf-dir models/qwen3-5-2b \
    --gen-steps 128 --decode-steps 24 --out build/keephead/xframe6-report.json
```
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from xframe_divergence_report import (  # noqa: E402
    _install_ggml_like_rmsnorm,
    hf_greedy,
    hf_prompt_ids,
)

DEFAULT_PROMPTS = [
    "Explain in one short sentence why floating point addition is not associative.",
    "Name the three primary colors and say which one you would remove first.",
    "Write a single sentence that contains exactly five words about rain.",
]


def rel_series(ref: list[float], alt: list[float]) -> list[float]:
    """逐位置的**标量**相对差：`|a-b| / max(|a|, 1e-30)`（纯函数，可脱离模型单测）。

    ★ 2026-10-08：入参从"每位置的整份 logits"改为"每位置的 **top1 标量**" ——
    长上下文下保留整份 vocab 会到数 GB，而判据只需要 top1。
    """
    out: list[float] = []
    for a, b in zip(ref, alt):
        denom = max(1e-30, abs(float(a)))
        out.append(abs(float(a) - float(b)) / denom)
    return out


def slope_per_position(rel: list[float]) -> float:
    """`rel` 对位置的线性回归斜率（纯函数）。≈0 ⇒ 不随步数累积。"""
    n = len(rel)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(rel) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(rel))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def summarise(rel: list[float], same: list[bool]) -> dict:
    """把逐位置序列汇总成结论字段（纯函数）。

    判据设计（2026-10-08 定稿）：
    - **主判据** `tail_over_head`：`rel` 的**尾部四分之一均值 / 首部四分之一均值**。
      递推若引入额外累积，该比值应显著 > 1 且随序列长度增长；实测 154 位置长序列为
      **≈1 甚至 < 1**（噪声主导，无趋势）。
    - **辅助** `rel_slope_per_pos`：线性回归斜率 —— 短序列上噪声大，仅作参考。
    - **样本不足**（`positions < 64`）⇒ `no_extra_accumulation = None`（**不下结论**），
      避免用 30 余位置的噪声斜率把它误判成"有额外累积"。
    """
    ordered = sorted(rel)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    n = len(rel)
    q = max(1, n // 4)
    head = sum(rel[:q]) / len(rel[:q]) if rel else 0.0
    tail = sum(rel[-q:]) / len(rel[-q:]) if rel else 0.0
    tail_over_head = (tail / head) if head > 0 else float("inf")
    slope = slope_per_position(rel)
    verdict = None
    if n >= 48:
        verdict = tail_over_head < 2.0
    return {
        "positions": n,
        "argmax_same_positions": sum(1 for s in same if s),
        "argmax_all_same": all(same) if same else True,
        "max_rel": max(rel) if rel else 0.0,
        "median_rel": median,
        "head_mean_rel": head,
        "tail_mean_rel": tail,
        "tail_over_head": tail_over_head,
        "first_rel": rel[0] if rel else 0.0,
        "last_rel": rel[-1] if rel else 0.0,
        "rel_slope_per_pos": slope,
        "no_extra_accumulation": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XFRAME-6 hybrid 递推误差阶（真模型）")
    ap.add_argument("--hf-dir", required=True, help="HF 模型目录（qwen3-5-2b 等 hybrid 模型）")
    ap.add_argument("--prompts", help="每行一个 prompt 的文件；缺省用内置 3 条")
    ap.add_argument("--gen-steps", type=int, default=128,
                    help="teacher-forcing 的序列长度（原 prompt + 生成的 token 数）")
    ap.add_argument(
        "--min-new-tokens", type=int, default=0,
        help="强制至少生成这么多 token（★ 2026-10-08 长上下文扫描用）—— 默认 0，"
             "沿用 HF 的 EOS 早停；设为 >0 时序列长度可控，用于测长上下文下的分叉点",
    )
    ap.add_argument("--decode-steps", type=int, default=24,
                    help="自由解码对照的步数（0 表示跳过）")
    ap.add_argument("--out", help="把报告写成 JSON")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [
            ln for ln in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    print(f"[xframe6] model loaded: {args.hf_dir}", flush=True)

    report = {"hf_dir": args.hf_dir, "gen_steps": args.gen_steps,
              "decode_steps": args.decode_steps, "prompts": []}

    for prompt in prompts:
        pids, _eos = hf_prompt_ids(args.hf_dir, prompt)

        # ① 用原版 greedy 生成一条固定序列（只用于构造 teacher-forcing 的输入）
        gen_kwargs = {"max_new_tokens": args.gen_steps, "do_sample": False}
        if args.min_new_tokens > 0:
            # ★ 长上下文扫描：EOS 早停会让序列长度不可控 ⇒ 强制生成到指定长度
            gen_kwargs["min_new_tokens"] = min(args.min_new_tokens, args.gen_steps)
        with torch.no_grad():
            gen = model.generate(torch.tensor([pids]), **gen_kwargs)
        seq = [int(x) for x in gen[0]]

        # ② teacher-forcing：同一条序列，原版 vs ggml-like 归约语义
        def top1_and_argmax() -> tuple[list[float], list[int]]:
            """逐位置的 **top1 logit** 与 **argmax**（不保留整份 vocab 的 logits）。

            长上下文下 `[seq_len, vocab]` 的 f64 会到数 GB（2k × 150k × 8B ≈ 2.4GB，
            且 teacher-forcing 需要同时持有 ref 与 alt 两份 ⇒ 6GB 量级），而本票判据
            只需要"逐位置 top1 的相对差"与"argmax 是否相同"。这里逐位置提取后立即丢弃，
            峰值只有单行 `[vocab]`。
            """
            top1s: list[float] = []
            argmaxes: list[int] = []
            with torch.no_grad():
                out = model(torch.tensor([seq]))
                for pos in range(out.logits.shape[1]):
                    row = out.logits[0, pos].to(torch.float64)
                    top2 = torch.topk(row, 2).values
                    top1s.append(float(top2[0]))
                    argmaxes.append(int(torch.argmax(row)))
            return top1s, argmaxes

        ref_top1, ref_argmax = top1_and_argmax()
        replaced = _install_ggml_like_rmsnorm(model)   # 同一实例替换 ⇒ 零额外内存
        alt_top1, alt_argmax = top1_and_argmax()

        rel = rel_series(ref_top1, alt_top1)
        same = [a == b for a, b in zip(ref_argmax, alt_argmax)]
        summary = summarise(rel, same)
        summary["replaced_norms"] = replaced
        summary["seq_len"] = len(seq)
        summary["prompt_tokens"] = len(pids)

        # ③ 自由解码对照（可选）：是否分叉
        if args.decode_steps > 0:
            base = hf_greedy(args.hf_dir, pids, args.decode_steps, rmsnorm="hf")
            alt = hf_greedy(args.hf_dir, pids, args.decode_steps, rmsnorm="ggml-like")
            summary["decode_ids_identical"] = base["ids"] == alt["ids"]
            summary["decode_steps"] = args.decode_steps

        report["prompts"].append({"prompt": prompt[:60], **summary})
        print(
            f"[xframe6] {prompt[:40]!r}: argmax {summary['argmax_same_positions']}/"
            f"{summary['positions']} rel_max={summary['max_rel']:.3e} "
            f"rel_first={summary['first_rel']:.3e} rel_last={summary['last_rel']:.3e} "
            f"slope={summary['rel_slope_per_pos']:.3e} "
            f"no_extra={summary['no_extra_accumulation']}"
            + (f" decode_identical={summary.get('decode_ids_identical')}"
               if "decode_ids_identical" in summary else ""),
            flush=True,
        )
        # 还原为原版语义，供下一个 prompt 的 ①/③ 使用
        _restore_original_norms(model)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[xframe6] written {out_path}")
    return 0


def _restore_original_norms(model) -> None:
    """把替换过的 RMSNorm 还原为原实现（依赖 HF 模块自身的类型方法）。"""
    for module in model.modules():
        cls_name = type(module).__name__
        if "RMSNorm" in cls_name and hasattr(module, "weight"):
            if "forward" in module.__dict__:
                del module.__dict__["forward"]


if __name__ == "__main__":
    raise SystemExit(main())
