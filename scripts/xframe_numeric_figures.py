#!/usr/bin/env python
"""xframe_numeric_figures.py — 把跨框架接力的**数值差异**数据画成图（供报告引用）。

与 `build/cross-framework-layer-poc/make_figures.py`（性能/切点主题，fig1–fig7）配套，
本脚本补**数值主题**：

* `fig8-difference-layers.png` —— 差异分层（对数轴）：从「层段切分 = 0」到「权重网格不同 ≈ 2e-1」；
* `fig9-int8-grid-tradeoff.png` —— int8 网格的**取舍**：顺序敏感度 vs 量化误差（两子图，零值单列标注）。

纪律（与既有画图脚本一致）
--------------------------
* **只画报告里写过的数字**，不推算；**每条都标注来源 JSON 与测试条件**；
* 缺字段/缺文件 ⇒ **跳过并打印**，不臆造；
* 图上不同条目的**测试条件不同**（模型/切点/层），注里写明「不可直接互相比较」。

⚠️ `matplotlib` 只在 `main()` 内导入 ⇒ 本模块可被单测安全 import（`.venv-test` 无 matplotlib）。

用法::

    python scripts/xframe_numeric_figures.py        # 输出到 docs/figures/cross-frame-relay/
"""
from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = _ROOT / "build" / "keephead"
OUTDIR = _ROOT / "docs" / "figures" / "cross-frame-relay"


class Item:
    """图上的一条：`label` / `value` / `source`（JSON 文件名）/ `cond`（测试条件）。"""

    __slots__ = ("label", "value", "source", "cond")

    def __init__(self, label: str, value: float, source: str, cond: str):
        self.label, self.value, self.source, self.cond = label, float(value), source, cond

    def as_tuple(self) -> tuple:
        return (self.label, self.value, self.source, self.cond)


def _load(name: str) -> dict | None:
    p = BUILD / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_difference_layers() -> list[Item]:
    """差异分层图的条目（缺文件的项自动跳过）。每项都来自某个 JSON 的**已有字段**。"""
    items: list[Item] = []

    lr = _load("xframe-layer-range.json")
    if lr and lr.get("rows"):
        items.append(Item("层段切分（ModelManager vs HF 整模）", lr["rows"][0]["rel_err"],
                          "xframe-layer-range.json", "Qwen3.5-2B，K=12…23，identical=True"))

    ig = _load("xframe-int8-grid.json")
    if ig and ig.get("rows"):
        r0 = ig["rows"][0]
        items.append(Item("归约顺序（f32，分段 1/2/4/8）", r0["f32_order_spread_rel"],
                          "xframe-int8-grid.json", f"{r0['module'].split('.')[-1]}，Qwen2.5-0.5B"))
        items.append(Item("块间浮点（量化点积同分段）", r0["blockwise_order_spread_rel"],
                          "xframe-int8-grid.json", "同上；块级 32"))
        items.append(Item("int8 块级32 网格的量化误差", r0["int8_block32_quant_rel_err"],
                          "xframe-int8-grid.json", "同上"))
        items.append(Item("int8 行级网格的量化误差", r0["int8_row_grid_quant_rel_err"],
                          "xframe-int8-grid.json", "同上"))

    aq = _load("xframe-actquant.json")
    if aq and aq.get("rows"):
        worst = max(aq["rows"], key=lambda r: r["rel_quantized_activation"])
        items.append(Item("激活量化（单层，llama.cpp matmul 内）",
                          worst["rel_quantized_activation"], "xframe-actquant.json",
                          f"{worst['label']}，Qwen2.5-0.5B Q4_K 反量化"))

    lp = _load("xframe-layerprofile16.json")
    if lp and lp.get("rows"):
        items.append(Item("权重网格不同（fp32 vs Q4_K）", lp["rows"][0]["rel_err"],
                          "xframe-layerprofile16.json", "Qwen3.5-2B，K=16"))
    lp2 = _load("xframe-layerprofile-05b-q4km-dequant.json")
    if lp2 and lp2.get("rows"):
        items.append(Item("权重同源（Q4_K 反量化）", lp2["rows"][0]["rel_err"],
                          "xframe-layerprofile-05b-q4km-dequant.json", "Qwen2.5-0.5B，K=16"))
    lp3 = _load("xframe-layerprofile-05b-f16-dequant.json")
    if lp3 and lp3.get("rows"):
        items.append(Item("权重同源（f16 反量化）", lp3["rows"][0]["rel_err"],
                          "xframe-layerprofile-05b-f16-dequant.json", "Qwen2.5-0.5B，K=16"))
    return items


def collect_int8_tradeoff() -> list[dict] | None:
    """int8 网格取舍图的条目（三方案：f32 / 纯整型 / 块级量化点积）。"""
    ig = _load("xframe-int8-grid.json")
    if not ig or not ig.get("rows"):
        return None
    r0 = ig["rows"][0]
    return [
        {"name": "f32 GEMM",
         "order_spread": r0["f32_order_spread_rel"],
         "quant_err": 0.0},
        {"name": "int8 行级 + 纯整型累加",
         "order_spread": 0.0,
         "quant_err": r0["int8_row_grid_quant_rel_err"]},
        {"name": "int8 块级32 + 块间浮点",
         "order_spread": r0["blockwise_order_spread_rel"],
         "quant_err": r0["int8_block32_quant_rel_err"]},
    ]


def _setup_fonts() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _setup_fonts()
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    # ---- fig8：差异分层（对数轴）----
    items = collect_difference_layers()
    if not items:
        skipped.append("fig8：没有任何来源 JSON 可读")
    else:
        items.sort(key=lambda it: it.value)
        labels = [it.label for it in items]
        vals = [max(it.value, 1e-12) for it in items]     # 0 用占位画在轴底，另行标注
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        bars = ax.barh(range(len(items)), vals, color="#3d7eb8")
        for i, it in enumerate(items):
            note = "0（逐位相同）" if it.value == 0 else f"{it.value:.2e}"
            ax.text(max(it.value, 1e-12) * 1.35, i, note, va="center", fontsize=7.5, color="#c0392b")
            ax.text(2e-12, i - 0.34, f"来源 {it.source} · {it.cond}", va="center", fontsize=6.0,
                    color="#444444",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.6))
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1e-12, 2.0)
        ax.set_xlabel("相对差异 rel（log 轴）", fontsize=8.5)
        ax.set_title("图 8  跨框架接力的差异分层（各条测试条件不同，不可直接互相比较）",
                     fontsize=9.5)
        ax.grid(axis="x", ls=":", alpha=0.4)
        fig.savefig(OUTDIR / "fig8-difference-layers.png")
        plt.close(fig)
        print(f"[fig8] {len(items)} 条 → {OUTDIR / 'fig8-difference-layers.png'}")

    # ---- fig9：int8 网格取舍（两子图）----
    trade = collect_int8_tradeoff()
    if not trade:
        skipped.append("fig9：缺 xframe-int8-grid.json")
    else:
        names = [d["name"] for d in trade]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
        for ax, key, title, color in (
            (axes[0], "order_spread", "顺序敏感度（越小越好，0 = 逐位相同）", "#c0392b"),
            (axes[1], "quant_err", "量化误差 rel（越小越好）", "#2471a3"),
        ):
            vals = [max(d[key], 1e-12) for d in trade]
            ax.bar(range(len(trade)), vals, color=color)
            for i, d in enumerate(trade):
                t = "0" if d[key] == 0 else f"{d[key]:.2e}"
                ax.text(i, max(d[key], 1e-12) * 1.6, t, ha="center", fontsize=7.5)
            ax.set_yscale("log")
            ax.set_ylim(1e-12, 1.0)
            ax.set_xticks(range(len(trade)))
            ax.set_xticklabels(names, fontsize=7, rotation=12)
            ax.set_title(title, fontsize=8.5)
            ax.grid(axis="y", ls=":", alpha=0.4)
        fig.suptitle("图 9  int8 共享网格的取舍（Qwen2.5-0.5B，down_proj，真实激活）", fontsize=9.5)
        fig.savefig(OUTDIR / "fig9-int8-grid-tradeoff.png")
        plt.close(fig)
        print(f"[fig9] {len(trade)} 方案 → {OUTDIR / 'fig9-int8-grid-tradeoff.png'}")

    for s in skipped:
        print(f"[skip] {s}")
    print(f"[done] 输出目录 {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
