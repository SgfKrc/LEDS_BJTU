#!/usr/bin/env python
"""xframe_grid_stack.py — DL 接力的**优化叠加性**实证：哪些维度能同时启用、能压到多少。

背景
----
`XFRAME-3` 已证：**交界处**把 hidden 量化到 int8 逐通道共享网格 ⇒ 跨引擎 argmax 被**压回同格**
（prefill 3/3、8 步 8/8）。`XFRAME-4/5` 证归约顺序有 `O(K·ε)` 下界、f64 可绕过。
但**从来没测过这两条能不能叠加** —— 即"上游 GEMM 也改成 int8 网格 + 整型累加"之后，
交界处的网格对齐**是否仍然成立**。

本工具在**同一条真链路**上并列五组，用同一批 ids / 同一整模基线：

| 组 | 上游（前 K 层） | 交界处 hidden | 下游（第 K 层起） |
| --- | --- | --- | --- |
| `A` | llama.cpp 整模（基线，非接力） | — | — |
| `B` | llama.cpp keep-head（**同引擎**，无量化） | 原样 f32 | llama.cpp |
| `C` | **PyTorch** f32 GEMM | **int8 共享网格** | llama.cpp |
| `E` | **PyTorch** int8 网格 + **int32 累加** | **int8 共享网格** | llama.cpp |
| `F` | **PyTorch** int8 网格 + **int32 累加** | 原样 f32 | llama.cpp |

判读：
* `C == A` ⇒ 交界处网格对齐有效（复现 XFRAME-3）；
* `E == C` ⇒ **整型累加可与网格对齐叠加**（两条维度正交）；
* `E != C` 而 `F != B` ⇒ 上游整型累加引入了**跨不过半格**的偏差 ⇒ 两条维度**互斥**。

用法::

    python scripts/xframe_grid_stack.py --gen 8 --out build/keephead/xframe-grid-stack.json

工件缺失时按本仓惯例 **打印 SKIP 并退出 0**。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_POC = _ROOT / "build" / "cross-framework-layer-poc" / "out"


def quantize_shared_per_channel(x, ch_scale, levels: float = 127.0):
    """按**调用方给定**的逐通道 scale 量化到 `±levels` 再乘回（共享网格，纯函数）。

    与 `XFRAME-3` 的 `quant_shared_per_channel` 同口径：`scale` 由**两侧共同决定**
    （`max(|h_pytorch|, |h_llama|)`），不是各自算 —— 各自算会让网格逐 block 偏移、量化反而放大分歧。
    """
    import numpy as np

    arr = np.asarray(x, dtype=np.float32)
    s = np.asarray(ch_scale, dtype=np.float32)
    s = np.where(s == 0, np.float32(1.0), s)
    q = np.clip(np.round(arr / s * levels), -levels, levels)
    return (q / levels * s).astype(np.float32)


def int8_gemm(x, w, levels: float = 127.0):
    """int8 网格上的线性层：per-token 激活 scale + per-output-channel 权重 scale，**int32 累加**。

    `x`: `[..., K]`、`w`: `[N, K]` ⇒ 返回 `[..., N]`（f32）。整数加满足结合律 ⇒ 结果与分块/线程
    划分无关（这是"整型累加"这一维度的全部意义）；代价是量化误差 `≈ scale/levels`。
    **bias 不在本函数内**（调用方加）。
    """
    import torch

    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    xs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)          # [M, 1]
    ws = w.detach().abs().amax(dim=1).clamp(min=1e-30)                  # [N]
    qx = torch.round(flat / xs * levels).clamp(-levels, levels).to(torch.int32)
    qw = torch.round(w.detach() / ws[:, None] * levels).clamp(-levels, levels).to(torch.int32)
    acc = qx @ qw.T                                                     # [M, N] int32
    out = acc.to(torch.float64) * (xs * ws[None, :]) / (float(levels) ** 2)
    return out.to(torch.float32).reshape(*x.shape[:-1], w.shape[0])


def _patch_linear_int8(levels: float = 127.0):
    """把 `torch.nn.Linear.forward` 换成「int8 网格 + int32 累加」；返回原实现以便恢复。"""
    import torch

    original = torch.nn.Linear.forward

    def patched(self, x):
        y = int8_gemm(x, self.weight, levels)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    torch.nn.Linear.forward = patched
    return original


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DL 接力的优化叠加性（网格对齐 × 整型累加）")
    ap.add_argument("--model", default=str(_ROOT / "models" / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--head", default=str(_POC / "qwen25-05b-f16-head12.gguf"))
    ap.add_argument("--cut", default=str(_POC / "qwen25-05b-f16-cut-k12.gguf"))
    ap.add_argument("--whole", default=str(_POC / "qwen25-05b-f16.gguf"))
    ap.add_argument("--shim", default=str(_ROOT / "build" / "keephead" / "build-cpu"
                                          / "bin" / "qlh_keep_head.dll"))
    ap.add_argument("--extra-dll-dir", default=r"C:\msys64\ucrt64\bin")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--levels", type=float, default=127.0)
    ap.add_argument("--prompts", default="What is the capital of France?")
    ap.add_argument("--out", help="把结果写成 JSON")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for p in (args.model, args.head, args.cut, args.whole, args.shim):
        if not pathlib.Path(p).exists():
            print(f"SKIP: 缺工件 {p}")
            return 0

    import numpy as np
    import torch
    import llama_cpp.llama_cpp as M

    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "src"))
    from llama_engine import LlamaCppEngine
    from llama_keep_head import KeepHeadUpstream
    from transformers import AutoModelForCausalLM, AutoTokenizer

    extra = [args.extra_dll_dir] if args.extra_dll_dir else []
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [p.strip() for p in str(args.prompts).split("|") if p.strip()]

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    down = LlamaCppEngine()
    down.load_model(model_path=args.cut, n_ctx=512, n_threads=8, n_seq_max=1)
    if not down.is_loaded:
        print(f"SKIP: 下游加载失败 {args.cut}")
        return 0

    M.llama_backend_init()
    whole_model = M.llama_model_load_from_file(str(args.whole).encode("utf-8"),
                                               M.llama_model_default_params())
    cp = M.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_threads = 512, 512, 512, 8
    n_vocab = int(M.llama_vocab_n_tokens(M.llama_model_get_vocab(whole_model)))
    batch = M.llama_batch_init(512, 0, 1)

    def whole_next(seq: list[int]) -> int:
        ctx = M.llama_init_from_model(whole_model, cp)
        try:
            for i, tid in enumerate(seq):
                batch.token[i] = tid
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 if i == len(seq) - 1 else 0
            batch.n_tokens = len(seq)
            if M.llama_decode(ctx, batch) != 0:
                raise RuntimeError("llama_decode failed")
            lp = M.llama_get_logits_ith(ctx, len(seq) - 1)
            return int(np.ctypeslib.as_array(lp, shape=(n_vocab,)).argmax())
        finally:
            M.llama_free(ctx)

    def upstream_hidden(seq: list[int], group: str) -> "np.ndarray":
        """按组取上游 hidden：`B` 走 keep-head（llama.cpp），其余走 PyTorch（必要时先打 int8 patch）。"""
        if group == "B":
            up = KeepHeadUpstream(args.shim, args.head, mode="nextn", extra_dll_dirs=extra,
                                  n_ctx=512, n_threads=8)
            try:
                return np.asarray(up.forward_tokens_to_hidden(seq, n_past=0), dtype=np.float32)
            finally:
                up.close()
        original = None
        if group in ("E", "F"):
            original = _patch_linear_int8(args.levels)
        try:
            with torch.no_grad():
                hs = hf(torch.tensor([seq]), output_hidden_states=True).hidden_states
            return hs[args.k][0].to(torch.float32).cpu().numpy()
        finally:
            if original is not None:
                torch.nn.Linear.forward = original

    def chain_next(seq: list[int], group: str) -> int:
        h_hf = upstream_hidden(seq, group)
        if group == "B":
            feed = h_hf
        else:
            up = KeepHeadUpstream(args.shim, args.head, mode="nextn", extra_dll_dirs=extra,
                                  n_ctx=512, n_threads=8)
            try:
                h_ll = np.asarray(up.forward_tokens_to_hidden(seq, n_past=0), dtype=np.float32)
            finally:
                up.close()
            if group == "F":
                feed = h_hf                      # 不量化：只暴露上游整型累加自身的偏差
            else:
                ch_scale = np.maximum(np.abs(h_hf).max(axis=0), np.abs(h_ll).max(axis=0))
                feed = quantize_shared_per_channel(h_hf, ch_scale, args.levels)
        down._model._ctx.kv_cache_clear()
        lg = down.forward_layers_from_hidden(feed, n_past=0, all_logits=True)
        return int(np.asarray(lg)[-1].argmax())

    report = {"k": args.k, "gen": args.gen, "levels": args.levels, "prompts": []}
    groups = ("B", "C", "E", "F")
    for prompt in prompts:
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        pids = pids.tolist() if hasattr(pids, "tolist") else list(pids)

        seq = list(pids)
        base: list[int] = []
        for _ in range(args.gen):
            t = whole_next(seq)
            base.append(t)
            seq.append(t)
        print(f"\n[prompt] {prompt!r} ({len(pids)} tokens)")
        print(f"  A 整模          {base}")

        row = {"prompt": prompt, "baseline": base, "groups": {}}
        for group in groups:
            seq = list(pids)
            got: list[int] = []
            for _ in range(args.gen):
                t = chain_next(seq, group)
                got.append(t)
                seq.append(t)
            matched = sum(1 for a, b in zip(got, base) if a == b)
            first = next((i for i, (a, b) in enumerate(zip(got, base)) if a != b), None)
            row["groups"][group] = {"ids": got, "matched": matched, "gen": args.gen,
                                    "first_divergence": first}
            print(f"  {group} {group_name(group):12s} {got}  matched={matched}/{args.gen} "
                  f"first_div={first}")
        report["prompts"].append(row)

    # 汇总判读
    print("\n[verdict]")
    for group in groups:
        tot = sum(r["groups"][group]["matched"] for r in report["prompts"])
        den = sum(r["groups"][group]["gen"] for r in report["prompts"])
        print(f"  {group} {group_name(group):12s} matched={tot}/{den}")
    print(f"  判读: C==A 表示交界处网格对齐有效；E==C 表示整型累加可与网格对齐叠加；"
          f"F!=B 表示上游整型累加自身引入了偏差。")

    M.llama_batch_free(batch)
    M.llama_model_free(whole_model)
    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {dest}")
    return 0


def group_name(group: str) -> str:
    return {"A": "整模基线", "B": "同引擎无量化", "C": "跨引擎+网格",
            "E": "跨引擎+网格+整型累加", "F": "跨引擎+整型累加"}.get(group, group)


if __name__ == "__main__":
    raise SystemExit(main())
