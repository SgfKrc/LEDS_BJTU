"""relay_capacity_budget.py — 层接力的**容量预算**：可分层 / **不可分层**张量的实测构成与收益上限。

★ R-R5（2026-09-24）：`docs/跨框架层接力-容量收益实测-2026-09-21.md` §3 把"收益为什么不是 2×"
归结为**不可分层张量** `C`，§8 留下后续项「**不可分层张量的缩减**（如 shared embedding 强制
shared ⇒ 提高收益上限）」。本模块给那条后续项提供**实测基础**：**零依赖**读 safetensors header
（不加载权重、不 import torch），逐张量精确统计，再按几种**缩减策略**算收益上限。

⚠️ **口径（本模块采用的精确定义）**：**不可分层 = 不属于任何 `*.layers.<i>.*` 的张量** ——
不只是"词嵌入 + lm_head"。实测发现 `qwen3-5-2b` 还有 `model.visual.*` 视觉塔（约 300 个张量）
与 `mtp.*` **同样层外、同样不可按语言层切**，而它们在文档口径里被漏计（文档记 `C`≈24%，
实测 **37.31%**）。

⚠️ **纪律**：本模块只做**容量账**。数值一致性仍**只认 per-token argmax**；容量收益不构成质量结论。
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

__all__ = [
    "classify_tensors",
    "capacity_report",
    "read_safetensors_header",
    "relay_gain",
    "tensor_bytes",
]

#: "属于某一层"的判据：`... .layers.<i>. ...`（语言模型的层）。
LAYER_RE = re.compile(r"\.layers\.\d+\.")

#: 视觉塔/多模态编码器的**块**也可按块切（`model.visual.blocks.<i>.*`）—— `qwen3.5` 用它省 ~680 MB。
VISUAL_BLOCK_RE = re.compile(r"\.visual\.blocks\.\d+\.")

#: safetensors 支持的 dtype → 字节数（本模块只统计，不解码）。
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

#: `relay_gain` 的策略名。
STRATEGIES = ("fixed-upstream", "fixed-split-by-role", "fixed-sharded")


def _fixed_role_split(tensors: dict[str, dict], fixed: list[str]) -> tuple[float, float]:
    """把**不可分层**张量按角色分成 `(上游字节, 下游字节)` —— 复现文档 §3 的口径。

    * 词嵌入（名字含 `embed`）⇒ **上游**（`has_embedding=True` 只在上游）；
    * lm_head（`lm_head` / `output`）与 `output_norm`（名字以 `norm` 收尾）⇒ **下游**；
    * **其它层外**结构（如 `model.visual.*` 视觉塔、`mtp.*`）⇒ 归**上游**（保守：不假设它们能挪）。
    """
    upstream = 0.0
    downstream = 0.0
    for key in fixed:
        size = float(tensor_bytes(tensors[key]))
        lowered = key.lower()
        if "lm_head" in lowered or lowered.endswith(".output.weight"):
            downstream += size
        elif lowered.endswith("norm.weight"):
            downstream += size
        else:                       # embed / visual / mtp 等
            upstream += size
    return upstream, downstream


def read_safetensors_header(path: Path | str) -> dict[str, dict]:
    """读 safetensors 的**头部元数据**（前 8 字节 = 头部长度，随后是 JSON）。

    **零依赖**：不加载任何权重、不 import torch/transformers ⇒ 秒级、可用于 CI 与大模型。
    返回 `{张量名: {"dtype": ..., "shape": [...]}}`（`__metadata__` 被剔除）。
    """
    target = Path(path)
    with open(target, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(int(size)))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def tensor_bytes(spec: dict) -> int:
    """单个张量占多少字节（`shape` 元素数 × dtype 宽度）。"""
    count = 1
    for dim in spec.get("shape") or ():
        count *= int(dim)
    return count * DTYPE_BYTES.get(str(spec.get("dtype", "")).upper(), 2)


def classify_tensors(keys, *, extra_layer_patterns=()) -> tuple[list[str], list[str]]:
    """把张量名分成 `(可分层, 不可分层)`。

    判据：**属于任一层模式** ⇒ 可分层。默认只认 `*.layers.<i>.*`；传入
    `extra_layer_patterns=(VISUAL_BLOCK_RE,)` 时，视觉塔的块也计入可分层（对应"把视觉塔也切"
    的缩减策略）。
    """
    patterns = [LAYER_RE, *extra_layer_patterns]
    splittable: list[str] = []
    fixed: list[str] = []
    for key in keys:
        name = str(key)
        (splittable if any(p.search(name) for p in patterns) else fixed).append(name)
    return splittable, fixed


def relay_gain(l_bytes: float, c_bytes: float, segments: int, *, strategy: str,
               c_upstream_bytes: float | None = None,
               c_downstream_bytes: float | None = None) -> float:
    """按 `strategy` 算 `segments` 段的**容量收益**（`整模 / 最大段`）。

    三种策略（`C` 的**归属**不同 ⇒ 分母不同）：

    * `fixed-upstream` —— `C` **整份压在第 1 段**（最坏情形：词嵌入与 lm_head 挤在一侧）；
    * `fixed-split-by-role` —— 文档 §3 的口径：`C_emb` 归上游、`C_out` 归下游 ⇒ 分母的 `C`
      取**单侧较大者**。⚠️ 必须传真实的两侧字节（`c_upstream_bytes` / `c_downstream_bytes`）：
      对 **tie 模型** `C_out = 0` ⇒ 分母就是 `L/n + C_emb`（这才是 1.568× 的来源，
      用 `C/2` 近似会算出 ~2.0× 的**假高收益**）；
    * `fixed-sharded` —— `C` **与层一起按段均分**（方案 B「词表维分片」的理想上界）

    收益恒为 `(L + C) / 分母` ⇒ `C → 0` 时上限趋近 `segments`（两段 ⇒ **2.0×**）。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    steps = max(1, int(segments))
    layers, fixed = float(l_bytes), float(c_bytes)
    if strategy == "fixed-upstream":
        denom = layers / steps + fixed
    elif strategy == "fixed-split-by-role":
        upstream = fixed if c_upstream_bytes is None else float(c_upstream_bytes)
        downstream = 0.0 if c_downstream_bytes is None else float(c_downstream_bytes)
        denom = layers / steps + max(upstream, downstream)
    else:                                   # fixed-sharded
        denom = (layers + fixed) / steps
    return ((layers + fixed) / denom) if denom > 0 else 1.0


def capacity_report(model_dir: Path | str, *, extra_layer_patterns=(),
                    segments=(2, 3)) -> dict:
    """对一个**权重目录**出容量账：`L` / `C` / `C 占比` / 各策略在 2 段与 n 段下的收益上限。

    目录里所有 `*.safetensors` 的分片都会被读（只读头部）。缺 safetensors ⇒ 抛 `FileNotFoundError`
    （调用方据此跳过，**不虚构数字**）。
    """
    directory = Path(model_dir)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {directory}")

    tensors: dict[str, dict] = {}
    for shard in shards:
        tensors.update(read_safetensors_header(shard))

    splittable, fixed = classify_tensors(tensors, extra_layer_patterns=extra_layer_patterns)
    l_bytes = float(sum(tensor_bytes(tensors[k]) for k in splittable))
    c_bytes = float(sum(tensor_bytes(tensors[k]) for k in fixed))
    total = l_bytes + c_bytes
    c_upstream, c_downstream = _fixed_role_split(tensors, fixed)

    by_strategy: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        by_strategy[strategy] = {
            f"gain_{int(n)}seg": round(relay_gain(l_bytes, c_bytes, int(n), strategy=strategy,
                                                  c_upstream_bytes=c_upstream,
                                                  c_downstream_bytes=c_downstream), 4)
            for n in segments
        }

    largest_fixed = sorted(((k, tensor_bytes(tensors[k])) for k in fixed),
                           key=lambda item: -item[1])[:8]
    return {
        "model_dir": str(directory),
        "shards": [p.name for p in shards],
        "tensor_count": len(tensors),
        "splittable_count": len(splittable),
        "fixed_count": len(fixed),
        "total_bytes": total,
        "splittable_bytes": l_bytes,
        "fixed_bytes": c_bytes,
        "fixed_ratio": (c_bytes / total) if total > 0 else 0.0,
        "fixed_upstream_bytes": c_upstream,
        "fixed_downstream_bytes": c_downstream,
        "largest_fixed": [{"name": name, "bytes": int(size)} for name, size in largest_fixed],
        "gains": by_strategy,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI：`python src/relay_capacity_budget.py <权重目录> [...]`（只读头部，秒级）。"""
    import argparse

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(
        description="层接力容量预算：不可分层张量构成 + 各缩减策略的收益上限（★ R-R5）")
    ap.add_argument("model_dirs", nargs="+", help="含 *.safetensors 的权重目录")
    ap.add_argument("--segments", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--visual-splittable", action="store_true",
                    help="把 `model.visual.blocks.<i>.*` 也当可分层（对应「视觉塔也切」的策略）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    extra = (VISUAL_BLOCK_RE,) if args.visual_splittable else ()
    reports = []
    for entry in args.model_dirs:
        try:
            reports.append(capacity_report(entry, extra_layer_patterns=extra,
                                           segments=args.segments))
        except FileNotFoundError as exc:
            print(f"[skip] {entry}: {exc}")

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0 if reports else 2

    for report in reports:
        print(f"== {report['model_dir']} ==")
        print(f"  张量 {report['tensor_count']}（层内 {report['splittable_count']} / "
              f"层外 {report['fixed_count']}）   总权重 {report['total_bytes'] / 1e6:.1f} MB")
        print(f"  L = {report['splittable_bytes'] / 1e6:8.1f} MB    "
              f"C = {report['fixed_bytes'] / 1e6:8.1f} MB ({report['fixed_ratio'] * 100:5.2f}%)"
              f"   [上游 {report['fixed_upstream_bytes'] / 1e6:.1f} / "
              f"下游 {report['fixed_downstream_bytes'] / 1e6:.1f}]")
        for strategy, gains in report["gains"].items():
            joined = "  ".join(f"{key}={value:.3f}x" for key, value in gains.items())
            print(f"    {strategy:22s} {joined}")
        for item in report["largest_fixed"][:4]:
            print(f"      · {item['name'][:50]:50s} {item['bytes'] / 1e6:8.1f} MB")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
