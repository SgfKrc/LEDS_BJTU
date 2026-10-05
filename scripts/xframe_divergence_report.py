"""XFRAME-1：跨引擎分歧的自动检测与量化。

把这个项目里原本一次性的手工对拍流程固化成一个可重复调用的驱动器，用来回答
**「某项改动是否让分歧推迟」**——这是 `XFRAME-2`（共享 kernel）与 `XFRAME-3`（定点网格）
的判据前置。设计依据与背景见 `docs/跨框架接力数值差异机理-2026-10-05.md`。

**参照系约定**：跨引擎对照的参照系是**同引擎整模**，而不是「另一个引擎的整模」。
用后者测得的是**引擎差异**而非接力差异——本项目已在此踩过坑。因此：

- 本工具当前只做 **`cross-engine`**：HF transformers 整模 vs llama.cpp 整模，
  输出分歧画像（首分歧步、top-1 一致率、分歧点的 top1−top2 margin）。
  这是 `XFRAME-2/3` 需要的**基线**。
- 「同引擎」（整模 vs 分层链，或两段链 vs 整模）需要接线到接力执行路径（`stage_offer_v3`
  或 `forward_layers_from_hidden`），不在本工具范围内，另做。

直接调用引擎（不经 QLH 的 HTTP API），因此可与正在运行的服务并行执行。

用法::

    .venv-test\\Scripts\\python.exe scripts/xframe_divergence_report.py \
        --hf-dir models/qwen2.5-0.5b-instruct \
        --gguf  build/cross-framework-layer-poc/out/qwen25-05b-f16.gguf \
        --max-new-tokens 16 --out local_docs/xframe-report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

DEFAULT_PROMPTS = [
    "Count from 1 to 5, separated by commas.",
    "Name three primary colors.",
    "What is the capital of France?",
    "Say hello in one word.",
]


def hf_prompt_ids(hf_dir: str, prompt: str) -> tuple[list[int], int | None]:
    """用 HF tokenizer 渲染 chat 模板并分词——两侧共用的**唯一**输入来源。

    同时返回 `eos_token_id`，供分歧画像剔除尾部终止符。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=False)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return list(tok(text, add_special_tokens=False)["input_ids"]), tok.eos_token_id


def hf_greedy(hf_dir: str, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    """HF transformers 贪心解码；返回生成的 token id 与每步的 top1−top2 margin。"""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    ids_t = torch.tensor([input_ids])
    with torch.no_grad():
        out = model.generate(
            ids_t, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    gen_ids = [int(x) for x in out.sequences[0][ids_t.shape[1]:]]
    margins = []
    for step in out.scores:
        top2 = torch.topk(step[0].float(), 2).values
        margins.append(round(float(top2[0] - top2[1]), 4))
    return {"ids": gen_ids, "margins": margins}


def llama_greedy(gguf: str, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    """llama.cpp 手动贪心解码（吃外部算好的 ids，绕过它自己的 chat 模板）。

    两侧输入完全相同的 token id 是这套对照成立的前提，已实测确认。
    """
    import numpy as np
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, n_ctx=512, logits_all=True, verbose=False)
    llm.reset()
    llm.eval(list(input_ids))
    eos = llm.token_eos()
    prompt_len = len(input_ids)
    ids: list[int] = []
    margins: list[float] = []
    for _ in range(max_new_tokens):
        row = np.asarray(llm.scores[prompt_len + len(ids) - 1], dtype=np.float32)
        order = np.argsort(row)[::-1]
        margins.append(round(float(row[order[0]] - row[order[1]]), 4))
        nxt = int(order[0])
        if nxt == eos:
            break
        ids.append(nxt)
        llm.eval([nxt])
    return {"ids": ids, "margins": margins}


def profile(left: dict[str, Any], right: dict[str, Any], eos_id: int | None) -> dict[str, Any]:
    """把两侧的生成结果折成分歧画像。

    先剔除尾部 EOS：两侧的 EOS 处理口径不同（HF 的 `generate` 会把终止 token 留在
    `sequences` 里，而本工具的 llama.cpp 路径遇到 EOS 即 break），不剔除会把「只差一个
    终止符」误报成 token 序列分歧。
    """

    def clean(ids: list[int]) -> list[int]:
        if eos_id is None:
            return list(ids)
        out = list(ids)
        while out and out[-1] == eos_id:
            out.pop()
        return out

    a, b = clean(left["ids"]), clean(right["ids"])
    n = min(len(a), len(b))

    first = None
    for i in range(n):
        if a[i] != b[i]:
            first = i
            break

    matched = sum(1 for i in range(n) if a[i] == b[i])
    # 「前缀全同、只是长度不同」是另一种形态（EOS 决策分叉），单独标记而不是混进首分歧。
    length_only = first is None and len(a) != len(b)

    at_first = None
    if first is not None:
        at_first = {
            "left_token": a[first] if first < len(a) else None,
            "right_token": b[first] if first < len(b) else None,
            "left_margin": left["margins"][first] if first < len(left["margins"]) else None,
            "right_margin": right["margins"][first] if first < len(right["margins"]) else None,
        }

    return {
        "identical": a == b,
        "first_divergence_step": first,
        "length_only_divergence": length_only,
        "compared_prefix_len": n,
        "matched_prefix_tokens": matched,
        "top1_agreement": round(matched / n, 4) if n else 1.0,
        "left_len": len(a),
        "right_len": len(b),
        "divergence_point": at_first,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="XFRAME-1 跨引擎分歧画像")
    ap.add_argument("--hf-dir", required=True, help="HF 模型目录（提供 tokenizer 与左引擎）")
    ap.add_argument("--gguf", required=True, help="右引擎的整模 GGUF")
    ap.add_argument("--prompts", help="每行一个 prompt 的文件；缺省用内置 4 条")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", help="把画像写成 JSON")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [
            ln for ln in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    rows = []
    for prompt in prompts:
        pids, eos_id = hf_prompt_ids(args.hf_dir, prompt)
        left = hf_greedy(args.hf_dir, pids, args.max_new_tokens)
        right = llama_greedy(args.gguf, pids, args.max_new_tokens)

        row = {
            "prompt": prompt,
            "left": "hf_transformers",
            "right": "llama_cpp",
            "prompt_tokens": len(pids),
        }
        row.update(profile(left, right, eos_id))
        rows.append(row)

        if row["identical"]:
            print(f"  [{prompt[:36]:36s}] 一致")
        elif row["length_only_divergence"]:
            print(
                f"  [{prompt[:36]:36s}] 前缀全同、仅长度不同 "
                f"(hf={row['left_len']} gg={row['right_len']}) — EOS 决策分叉"
            )
        else:
            step = row["first_divergence_step"]
            dp = row["divergence_point"] or {}
            print(
                f"  [{prompt[:36]:36s}] 首分歧@{step} "
                f"(hf={dp.get('left_token')} margin={dp.get('left_margin')} | "
                f"gg={dp.get('right_token')} margin={dp.get('right_margin')}) "
                f"top1_agreement={row['top1_agreement']}"
            )

    total = len(rows)
    identical = sum(1 for r in rows if r["identical"])
    length_only = sum(1 for r in rows if r["length_only_divergence"])
    diverged = [r["first_divergence_step"] for r in rows if r["first_divergence_step"] is not None]
    summary = {
        "mode": "cross-engine",
        "left": "hf_transformers",
        "right": "llama_cpp",
        "prompts": total,
        "identical": identical,
        "length_only_divergence": length_only,
        "token_level_divergence": total - identical - length_only,
        "agreement_rate": round(identical / total, 4) if total else 1.0,
        "first_divergence_steps": diverged,
        "min_first_divergence_step": min(diverged) if diverged else None,
        "max_new_tokens": args.max_new_tokens,
    }
    print()
    print(
        f"  合计 {identical}/{total} 完全一致；仅长度不同 {length_only}；"
        f"token 级分歧 {total - identical - length_only}；首分歧步 = {diverged}"
    )

    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  已写入 {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
