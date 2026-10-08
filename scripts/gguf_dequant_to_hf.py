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

命名/排布约定（两张表，按 `general.architecture` 自动选）
-------------------------------------------------------
* **qwen2 系**：`token_embd.weight` → `model.embed_tokens.weight`，`blk.{i}.{suffix}` →
  `model.layers.{i}.{hf_suffix}`；
* **qwen35（hybrid）**：前缀是 `model.language_model.layers.{i}.`，且层内后缀随**层类型**而变
  —— linear-attention 层是 `attn_qkv`/`attn_gate`/`ssm_*`，全注意力层是 `attn_q/k/v/output` +
  `attn_q_norm/attn_k_norm`；MTP 层（`blk.{block_count-1}`）落到 `mtp.*` 与 `mtp.layers.0.*`；
* `output.weight` → `lm_head.weight`，但 `tie_word_embeddings=True` 的模型 GGUF 里它虽存在、
  HF 侧并不需要（HF 复用 embed）⇒ 默认**跳过**并记录；
* ⚠️ **排布**：GGUF 的 `tensor.shape` 记 `[in, out]`（llama.cpp 惯例），但**数据按 `[out, in]`
  存储**，`gguf.quants.dequantize` 实测**已按 HF 排布还原**。故本工具**不无条件转置**，而是
  按 `orient()` 做形状判定（任一侧不吻合即 fail-loud）—— 首版写死"一律转置"曾导致 121 个
  2-D 张量全部错位、HF 加载直接 `ignore_mismatched_sizes` 报错；
* ⚠️ **表示差**：`blk.{i}.ssm_conv1d.weight` 需补一维（`[channels, 1, kernel]`）；qwen35 的
  RMSNorm 在 GGUF 里是 `1 + weight` 的合并值、`ssm_a` 是 `-exp(A_log)` ⇒ 由 `apply_hf_inverse()`
  还原（不还原会得到 `2e-1…5e-1` 级的错误残差）。

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

# llama.cpp 张量后缀 → HF **层内**后缀（Qwen2 系，已核验）
_BLOCK_QWEN2: dict[str, str] = {
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

# Qwen3.5（hybrid：linear-attention 与全注意力按 `full_attention_interval` 交错 + 1 层 MTP）
# ★ 与 Qwen2 系**不能共用**：
#   * MLP 前的 norm 在 GGUF 里叫 `post_attention_norm`（Qwen2 是 `ffn_norm`）；
#   * 每层按类型二选一 —— linear-attention 层是 `attn_qkv`/`attn_gate`/`ssm_*`，
#     全注意力层是 `attn_q/k/v/output` + `attn_q_norm/attn_k_norm`；
#   * HF 路径前缀是 `model.language_model.layers`（Qwen2 是 `model.layers`）。
_BLOCK_QWEN35: dict[str, str] = {
    # 两类层共有
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # 全注意力层
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # linear-attention（SSM / GatedDeltaNet）层
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

# MTP（nextn）层的张量 → `mtp.` 顶层（Qwen3.5）
_MTP_TOP_QWEN35: dict[str, str] = {
    "nextn.eh_proj.weight": "fc.weight",
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "norm.weight",
}

_ARCH: dict[str, dict] = {
    "qwen2": {
        "layer_prefix": "model.layers.{i}.",
        "top": {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        },
        "block": _BLOCK_QWEN2,
        "mtp_top": {},
    },
    "qwen35": {
        "layer_prefix": "model.language_model.layers.{i}.",
        "top": {
            "token_embd.weight": "model.language_model.embed_tokens.weight",
            "output_norm.weight": "model.language_model.norm.weight",
            "output.weight": "lm_head.weight",
        },
        "block": _BLOCK_QWEN35,
        "mtp_top": _MTP_TOP_QWEN35,
    },
}


def arch_supported(arch: str) -> bool:
    """该 `general.architecture` 是否已有映射表（纯函数）。"""
    return arch in _ARCH


def hf_name_for(gguf_name: str, *, arch: str = "qwen2", mtp_index: int | None = None) -> str | None:
    """GGUF 张量名 → HF 参数名；未知名字返回 None（纯函数，调用方负责 fail-loud）。

    `mtp_index` 给出时（= `block_count - nextn_predict_layers`），`blk.{i}` 是 **MTP 层**：
    其 `nextn.*` 张量映射到 `mtp.{...}`，其余（attention/MLP）映射到 `mtp.layers.0.{...}`
    —— HF 侧确实是这两个位置（已核验 `mtp.layers.0.self_attn.*` 与 `mtp.{fc,norm,...}`）。
    """
    spec = _ARCH.get(arch)
    if spec is None:
        return None
    if gguf_name in spec["top"]:
        return spec["top"][gguf_name]
    if not gguf_name.startswith("blk."):
        return None
    parts = gguf_name.split(".")
    if len(parts) < 3 or not parts[1].isdigit():
        return None
    idx = int(parts[1])
    rest = ".".join(parts[2:])
    if mtp_index is not None and idx == mtp_index:
        if rest.startswith("nextn."):
            tail = spec["mtp_top"].get(rest)
            return f"mtp.{tail}" if tail is not None else None
        tail = spec["block"].get(rest)
        return f"mtp.layers.0.{tail}" if tail is not None else None
    tail = spec["block"].get(rest)
    if tail is None:
        return None
    return spec["layer_prefix"].format(i=idx) + tail


def orient(arr, *, gguf_shape, hf_name: str):
    """把反量化结果摆成 HF 排布（纯函数）。

    ★ 实测（gguf 库 + Qwen2.5-0.5B / Qwen3.5-2B 的 Q4_K_M）：`gguf.quants.dequantize`
    **已经返回 HF 排布** —— GGUF 的 `tensor.shape` 记的是 `[in, out]`（llama.cpp 惯例），
    但**数据按 `[out, in]` 存储**，反量化按存储顺序还原。不同 gguf 版本可能不同 ⇒ 这里
    **按形状判定**，两边都不吻合就 fail-loud，绝不静默转置错。

    例外：`conv1d.weight`（Qwen3.5 的 SSM 深度卷积）—— GGUF 声明 `[kernel, channels]`、
    反量化给 `[channels, kernel]`，而 HF 的 `nn.Conv1d` 权重是 `[channels, 1, kernel]`
    ⇒ 只**补一维**（不转置）。
    """
    import numpy as np

    declared = tuple(int(x) for x in gguf_shape)
    if hf_name.endswith("conv1d.weight"):
        want = (declared[1], 1, declared[0])
        if arr.ndim == 2 and tuple(arr.shape) == (declared[1], declared[0]):
            return np.ascontiguousarray(arr.reshape(want))
        if arr.ndim == 2 and tuple(arr.shape) == (declared[0], declared[1]):
            return np.ascontiguousarray(arr.T.reshape(want))
        raise ValueError(f"{hf_name}: conv1d 形状 {tuple(arr.shape)} 与声明 {declared} 不匹配")
    if arr.ndim <= 1:
        return arr
    wanted = tuple(reversed(declared))
    if tuple(arr.shape) == wanted:
        return arr
    if tuple(arr.shape) == declared:
        return arr.T
    raise ValueError(
        f"{hf_name}: 反量化形状 {tuple(arr.shape)} 既不等于期望 {wanted} 也不等于声明 {declared}"
    )


def apply_hf_inverse(arr, *, hf_name: str, arch: str):
    """把 GGUF 侧的**表示**还原成 HF 侧的表示（纯函数）。Qwen2 系是 no-op。

    Qwen3.5 有两处必须逆变换，否则权重会静默错位（实测：端到端残差因此高达 `2e-1…5e-1`）：

    1. **RMSNorm 的 `1.0 + weight`**：`Qwen3_5RMSNorm` 的 `weight` 是**偏移量**（零初始化，
       前向用 `1 + weight`，见 `modeling_qwen3_5.py` 与 PR #29402），而 llama.cpp 的 GGUF 里
       存的是**合并后的最终值** ⇒ 写入 HF 须**减 1**。
       ⚠️ 例外：`linear_attn.norm.weight` 对应 `Qwen3_5RMSNormGated`，它是**标准 `weight`**
       （实测 delta = 0）⇒ **不减**。
    2. **SSM 的 `A_log`**：GGUF 的 `ssm_a` 存的是 `-exp(A_log)`（实测逐元素吻合）
       ⇒ 还原须 **`log(-x)`**。

    实测依据（`models/qwen3-5-2b` 的 bf16 原权重 vs 反量化结果，122 个 1-D 张量）：
    68 个 norm 的 delta **恰为 `1.00000` 且标准差 0**；`ssm_a` 与 `-exp(A_log)` 逐元素相同。
    """
    import numpy as np

    if arch != "qwen35":
        return arr
    if hf_name.endswith("linear_attn.A_log"):
        return np.log(-np.asarray(arr, dtype=np.float32))
    if any(hf_name.endswith(suffix) for suffix in _QWEN35_NORM_PLUS_ONE):
        return np.asarray(arr, dtype=np.float32) - np.float32(1.0)
    return arr


#: Qwen3.5 里用 `1.0 + weight` 的 norm（**不含** `linear_attn.norm.weight`）
_QWEN35_NORM_PLUS_ONE: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.language_model.norm.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
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
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                    help="输出权重精度；默认 float32（最保真）。2B 级模型 f32 约 8 GB，"
                         "内存吃紧时可用 float16（对'激活量化'这类 1e-3 量级的测量已足够）")
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

    def _meta(key: str):
        fld = reader.fields.get(key)
        if fld is None:
            return None
        try:
            return fld.contents()
        except Exception:
            return None

    arch = str(_meta("general.architecture") or "qwen2")
    if not arch_supported(arch):
        print(f"SKIP: 架构 {arch!r} 尚无映射表（已支持 {sorted(_ARCH)}）")
        return 0
    block_count = _meta(f"{arch}.block_count")
    nextn = int(_meta(f"{arch}.nextn_predict_layers") or 0)
    mtp_index = int(block_count) - nextn if (block_count is not None and nextn > 0) else None
    print(f"[arch] {arch} block_count={block_count} nextn={nextn} mtp_index={mtp_index}")

    tensors = {}
    manifest = []
    unknown: list[str] = []
    skipped: list[str] = []
    for t in reader.tensors:
        hf = hf_name_for(t.name, arch=arch, mtp_index=mtp_index)
        if hf is None:
            unknown.append(t.name)
            continue
        if hf == "lm_head.weight" and not args.keep_tied_output:
            skipped.append((t.name, "tie_word_embeddings ⇒ HF 复用 embed_tokens"))
            continue
        arr = orient(_dequant(t), gguf_shape=[int(x) for x in t.shape], hf_name=hf)
        arr = apply_hf_inverse(arr, hf_name=hf, arch=arch)
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

    def _to_torch(v):
        # 只读数组转 torch 会告警；对只读的（F32 张量的 view）复制一份，其余零拷贝共享内存
        # —— 避免大模型（2B 的 f32 ≈ 8 GB）在转换期内存翻倍。
        if not v.flags.writeable:
            v = v.copy()
        return torch.from_numpy(v)

    if args.dtype != "float32":
        tensors = {k: v.astype(getattr(np, args.dtype)) for k, v in tensors.items()}

    payload = {k: _to_torch(v) for k, v in tensors.items()}
    save_file(payload, str(dst / "model.safetensors"))
    print(f"[done] {dst / 'model.safetensors'}（{len(payload)} 张量，{args.dtype}）")

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
