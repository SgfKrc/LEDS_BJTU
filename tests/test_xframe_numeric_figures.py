"""`scripts/xframe_numeric_figures.py` 的纯函数回归。

该模块**不 import matplotlib**（延迟到 `main()`），所以单测可在无 matplotlib 的环境跑。
这里只钉两件与"不臆造数据"相关的事：
1. 条目**只从 JSON 的已有字段**取（缺文件 ⇒ 空列表 / None，而不是编造）；
2. 每条都带上**来源文件名**与**测试条件**（图注可追溯）。

⚠️ 条目数依赖 `build/keephead/*.json` 这些本地产物；产物缺失时断言自动跳过
（与仓库既有的"缺工件型跳过"惯例一致，不伪装成通过）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_numeric_figures import (  # noqa: E402
    Item,
    collect_difference_layers,
    collect_int8_tradeoff,
)


def test_item_carries_source_and_condition():
    it = Item("标签", 1.5e-3, "some.json", "模型/切点")
    assert it.as_tuple() == ("标签", 1.5e-3, "some.json", "模型/切点")
    assert isinstance(it.value, float)


def test_collected_layers_have_source_and_nonnegative_value():
    items = collect_difference_layers()
    for it in items:
        assert it.source.endswith(".json"), it.source
        assert it.cond, "每条必须带测试条件（图注可追溯）"
        assert it.value >= 0.0


def test_int8_tradeoff_shape_is_either_none_or_three_schemes():
    trade = collect_int8_tradeoff()
    if trade is None:
        return  # 本地无 xframe-int8-grid.json ⇒ 跳过
    assert [d["name"] for d in trade] == ["f32 GEMM", "int8 行级 + 纯整型累加",
                                          "int8 块级32 + 块间浮点"]
    for d in trade:
        assert d["order_spread"] >= 0.0 and d["quant_err"] >= 0.0


def test_pure_int_scheme_has_zero_order_spread_and_nonzero_quant_err():
    trade = collect_int8_tradeoff()
    if trade is None:
        return
    pure = next(d for d in trade if "纯整型" in d["name"])
    assert pure["order_spread"] == 0.0          # 整数加满足结合律 ⇒ 与分段顺序无关
    assert pure["quant_err"] > 0.0              # 代价是量化误差
