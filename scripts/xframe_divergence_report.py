"""XFRAME-1：跨引擎分歧的自动检测与量化。

把这个项目里原本一次性的手工对拍流程固化成一个可重复调用的驱动器，用来回答
**「某项改动是否让分歧推迟」**——这是 `XFRAME-2`（共享 kernel）与 `XFRAME-3`（定点网格）
的判据前置。设计依据与背景见 `docs/跨框架接力数值差异机理-2026-10-05.md`。

**参照系约定**：跨引擎对照的参照系是**同引擎整模**，而不是「另一个引擎的整模」。
用后者测得的是**引擎差异**而非接力差异——本项目已在此踩过坑。因此：

- **`cross-engine`**（缺省）：HF transformers 整模 vs llama.cpp 整模，输出分歧画像
  （首分歧步、top-1 一致率、分歧点的 top1−top2 margin）。这是 `XFRAME-2/3` 需要的**基线**。
- **`same-engine`**（`--same-engine`）：llama.cpp **整模** vs llama.cpp **分层链** —— 首段
  keep-head 上游（`forward_tokens_to_hidden`）+ 可选中间段（`forward_hidden_to_hidden`）+
  末段（`forward_layers_from_hidden`），用 `--relay-segments` 给出按层序排列的裁层工件。
  判据是**同引擎组合报告全一致**——这正是「某项改动是否让分歧推迟」的参照系。

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
import os
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


def _install_ggml_like_rmsnorm(model: Any) -> int:
    """把模型里所有 RMSNorm 换成「复刻 ggml 语义」的版本，返回替换个数。

    依据（见 `docs/跨框架接力数值差异机理-2026-10-05.md` §9 与 KT 文档的源码核对）：
    ggml 的 `ggml_rms_norm` 用 **`double`（`ggml_float`）累加**，而 HF/torch 的
    `Qwen2RMSNorm` 在输入 dtype（f32）上累加 —— 这是一个**精度类**差异，且位于放大链的
    **种子**位置（embedding 2.7e-8 → attn_norm 2.0e-6，放大约 72×）。

    本函数只做这一项对齐实验，用于判定「把归一化这一环钉死后，首分歧是否推迟」。
    """
    import torch

    def make_forward(norm: Any):
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))

        def forward(hidden_states):
            # ggml 语义：以 double 累加 Σx²，取 1/sqrt(mean+eps)，最后再乘 weight。
            x = hidden_states.to(torch.float64)
            var = (x * x).mean(-1, keepdim=True)
            normalized = x * (1.0 / torch.sqrt(var + eps))
            return (normalized.to(hidden_states.dtype) * norm.weight)

        return forward

    count = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if "RMSNorm" in cls_name and hasattr(module, "weight"):
            module.forward = make_forward(module)
            count += 1
    return count


def _install_noise_injector(model: Any, layer_idx: int, rel_sigma: float,
                            seed: int = 1234) -> int:
    """在指定 transformer 层的输出上注入**确定性**相对噪声，返回挂载点数。

    用途：量化「这套系统对数值差异有多敏感」——扫 `rel_sigma`，看多大的相对扰动足以
    翻转 argmax。放大链（`docs/跨框架接力数值差异机理-2026-10-05.md` §3）预测系统在
    1e-3 量级的跨引擎差异下必然翻转；本探针给出该预测的**直接测量**。

    噪声按该张量自身的 `|x|` 均值缩放（相对幅度），并用固定 seed 保证可复现。
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def make_hook():
        def hook(_module, _inputs, output):
            def perturb(t):
                if not torch.is_tensor(t) or not t.is_floating_point():
                    return t
                scale = t.detach().abs().mean() * rel_sigma
                noise = torch.randn(t.shape, generator=generator,
                                    dtype=torch.float32) * float(scale)
                return (t.float() + noise).to(t.dtype)

            if isinstance(output, tuple):
                return (perturb(output[0]),) + tuple(output[1:])
            return perturb(output)

        return hook

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or layer_idx >= len(layers):
        raise SystemExit(f"模型没有 layers[{layer_idx}]")
    layers[layer_idx].register_forward_hook(make_hook())
    return 1


def hf_greedy(hf_dir: str, input_ids: list[int], max_new_tokens: int,
              rmsnorm: str = "hf", noise_layer: int | None = None,
              noise_sigma: float = 0.0) -> dict[str, Any]:
    """HF transformers 贪心解码；返回生成的 token id 与每步的 top1−top2 margin。

    `rmsnorm="ggml-like"` 时把模型内所有 RMSNorm 换成复刻 ggml 语义的版本
    （double 累加 + `1/sqrt`），用于 XFRAME-2 的「钉死归一化这一环」实验。
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    if rmsnorm == "ggml-like":
        replaced = _install_ggml_like_rmsnorm(model)
        print(f"  [实验] 已把 {replaced} 个 RMSNorm 替换为 ggml 语义（double 累加 + 1/sqrt）")
    if noise_layer is not None and noise_sigma > 0:
        _install_noise_injector(model, noise_layer, noise_sigma)
        print(f"  [实验] 已在 layers[{noise_layer}] 输出注入相对噪声 sigma={noise_sigma:g}")

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


def llama_relay_greedy(
    segment_paths: list[str],
    input_ids: list[int],
    max_new_tokens: int,
    *,
    eos_id: int | None = None,
    n_ctx: int = 4096,
    n_threads: int = 8,
    shim_path: str | None = None,
) -> dict[str, Any]:
    """llama.cpp **同引擎分层链**贪心解码（`same-engine` 模式的右半边）。

    与 `llama_greedy` 同契约（吃外部算好的 ids、吐 `{"ids","margins"}`），差别是不走
    整模，而是把 `segment_paths`（按层序排列的裁层工件）串成接力链：

      * 首段 `segment_paths[0]`：`KeepHeadUpstream(mode="nextn").forward_tokens_to_hidden`
        —— 吃 token、交 hidden（`output_norm` **之前**）；
      * 中间段（若有）：同型 shim 的 `forward_hidden_to_hidden` —— 吃 hidden、交 hidden；
      * 末段 `segment_paths[-1]`：`LlamaCppEngine.forward_layers_from_hidden` —— 吃 hidden、出 logits。

    左半边（参照系）仍是「同引擎**整模**」`llama_greedy(整模 gguf)` —— 用另一个引擎的整模当
    参照会测到**引擎差**而非**接力差**，本项目已在此踩过坑。

    `shim_path` 缺省走 `QLH_KEEP_HEAD_SHIM` 环境变量（`llama_engine` 的默认查找）；
    末段用显式给出的工件路径，不依赖 `_find_layer_artifact` 的自动命名匹配。
    """
    import numpy as np

    # 驱动器本身不依赖 `src/`（cross-engine 模式只用第三方 llama_cpp），但分层链要直接调
    # 主仓引擎 ⇒ 这里局部注入仓库根与 `src/`，不影响默认模式。
    _root = pathlib.Path(__file__).resolve().parents[1]
    for _p in (str(_root), str(_root / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from llama_engine import LlamaCppEngine
    from llama_keep_head import KeepHeadUpstream

    if len(segment_paths) < 2:
        raise ValueError("分层链至少需要两段（首段 + 末段）")

    # shim 路径解析顺序：显式参数 > `QLH_KEEP_HEAD_SHIM` > 仓库默认构建产物。
    # 末段 `LlamaCppEngine` 自己也按 `QLH_KEEP_HEAD_SHIM` 找，所以统一写回环境变量。
    shim = shim_path or os.environ.get("QLH_KEEP_HEAD_SHIM", "").strip()
    if not shim:
        shim = str(_root / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll")
    os.environ["QLH_KEEP_HEAD_SHIM"] = shim

    ups = [
        KeepHeadUpstream(shim, p, mode="nextn", n_ctx=n_ctx, n_threads=n_threads)
        for p in segment_paths[:-1]
    ]
    engine = LlamaCppEngine()
    engine.load_model(
        model_path=segment_paths[-1], n_ctx=n_ctx, n_threads=n_threads, n_seq_max=1,
    )
    if not engine.is_loaded:
        raise RuntimeError(f"末段加载失败：{segment_paths[-1]}")

    ids: list[int] = []
    margins: list[float] = []
    up_pos = [0] * len(ups)
    down_pos = 0
    try:
        for _ in range(max_new_tokens):
            toks = input_ids if not ids else [ids[-1]]
            hidden = ups[0].forward_tokens_to_hidden(toks, n_past=up_pos[0])
            for k in range(1, len(ups)):
                hidden = ups[k].forward_hidden_to_hidden(hidden, n_past=up_pos[k])
            logits = engine.forward_layers_from_hidden(
                hidden, n_past=down_pos, all_logits=True,
            )
            if logits is None:
                raise RuntimeError("末段 forward_layers_from_hidden 返回 None")
            row = np.asarray(logits, dtype=np.float32)[-1]
            order = np.argsort(row)[::-1]
            margins.append(round(float(row[order[0]] - row[order[1]]), 4))
            nxt = int(order[0])
            if eos_id is not None and nxt == eos_id:
                break
            ids.append(nxt)
            for k in range(len(ups)):
                up_pos[k] += len(toks)
            down_pos += len(toks)
    finally:
        for u in ups:
            u.close()

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
    ap.add_argument(
        "--same-engine", action="store_true",
        help="同引擎模式：左=整模 `llama_greedy(--gguf)`，右=分层链 `llama_relay_greedy"
             "(--relay-segments)`。用于 XFRAME-1 的 same-engine 判据（整模 vs 分层链）",
    )
    ap.add_argument(
        "--relay-segments",
        help="逗号分隔的裁层工件路径（按层序排列），配合 --same-engine；至少两段",
    )
    ap.add_argument(
        "--shim",
        help="keep-head shim 路径；缺省走 QLH_KEEP_HEAD_SHIM 或 "
             "build/keephead/build-cpu/bin/qlh_keep_head.dll",
    )
    ap.add_argument(
        "--hf-rmsnorm", choices=["hf", "ggml-like"], default="hf",
        help="hf=原生 RMSNorm（默认）；ggml-like=复刻 ggml 语义（double 累加 + 1/sqrt），"
             "用于 XFRAME-2 的「钉死归一化这一环」实验",
    )
    ap.add_argument("--out", help="把画像写成 JSON")
    ap.add_argument("--noise-layer", type=int, default=None,
                    help="敏感度探针：在 HF 的该层输出注入相对噪声（需配合 --noise-sigma）")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="敏感度探针：相对噪声幅度（按该张量 |x| 均值缩放）")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [
            ln for ln in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    relay_segments = [s.strip() for s in (args.relay_segments or "").split(",") if s.strip()]
    if args.same_engine and len(relay_segments) < 2:
        ap.error("--same-engine 需要 --relay-segments 给出至少两个裁层工件路径（按层序，逗号分隔）")

    rows = []
    for prompt in prompts:
        pids, eos_id = hf_prompt_ids(args.hf_dir, prompt)
        if args.same_engine:
            # 同引擎：左=整模，右=分层链；两侧都是 llama.cpp ⇒ 测的是**接力差**而非引擎差。
            left = llama_greedy(args.gguf, pids, args.max_new_tokens)
            right = llama_relay_greedy(
                relay_segments, pids, args.max_new_tokens,
                eos_id=eos_id, shim_path=args.shim,
            )
            left_name, right_name = "llama_cpp_whole", "llama_cpp_relay"
        else:
            left = hf_greedy(args.hf_dir, pids, args.max_new_tokens, rmsnorm=args.hf_rmsnorm,
                             noise_layer=args.noise_layer, noise_sigma=args.noise_sigma)
            right = llama_greedy(args.gguf, pids, args.max_new_tokens)
            left_name, right_name = "hf_transformers", "llama_cpp"

        row = {
            "prompt": prompt,
            "left": left_name,
            "right": right_name,
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
        "mode": "same-engine" if args.same_engine else "cross-engine",
        "left": "llama_cpp_whole" if args.same_engine else "hf_transformers",
        "right": "llama_cpp_relay" if args.same_engine else "llama_cpp",
        "relay_segments": relay_segments if args.same_engine else None,
        "hf_rmsnorm": args.hf_rmsnorm,
        "noise_layer": args.noise_layer,
        "noise_sigma": args.noise_sigma,
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
