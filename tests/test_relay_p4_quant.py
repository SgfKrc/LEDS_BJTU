"""P4 新增量化档与边距（margin）判据的回归测试。

覆盖：
- `int4_block128` 的**线路字节口径**（4bit 打包 + 每块 f32 scale）与 int8/f16 的单调关系；
- 量化误差随位宽下降而单调上升（f16 < int8_block128 < int4_block128）；
- 非 128 整倍数宽度（padding 路径）不改变形状；
- `_margin` = top1 - top2，且不改变 argmax 语义；退化输入（<2 元素）返回 nan 而不抛异常；
- 未知模式 fail-loud（不得静默返回原值）。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "scripts" / "relay_experiment.py"


@pytest.fixture(scope="module")
def rx():
    """按路径加载实验驱动（`scripts/` 不是包）。"""
    spec = importlib.util.spec_from_file_location("relay_experiment_under_test", EXPERIMENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hidden_bytes_int4_is_half_of_int8(rx) -> None:
    width = 896
    blocks = (width + 127) // 128
    assert rx._hidden_bytes(width, "int4_block128") == (width + 1) // 2 + blocks * 4
    assert rx._hidden_bytes(width, "int8_block128") == width + blocks * 4
    # 4bit 应当明显小于 8bit，也明显小于 f16
    assert rx._hidden_bytes(width, "int4_block128") < rx._hidden_bytes(width, "int8_block128")
    assert rx._hidden_bytes(width, "int8_block128") < rx._hidden_bytes(width, "f16")


def test_hidden_bytes_odd_width_rounds_up(rx) -> None:
    # 宽为奇数时打包字节要向上取整，不能漏掉最后一个元素
    assert rx._hidden_bytes(3, "int4_block128") == 2 + 1 * 4


def test_quant_error_grows_as_bitwidth_shrinks(rx) -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    data = rng.standard_normal((4, 896), dtype=np.float32)

    def err(mode: str) -> float:
        out = rx._quantize_hidden(data, mode)
        return float(np.abs(np.asarray(out) - data).max())

    f16, int8, int4 = err("f16"), err("int8_block128"), err("int4_block128")
    assert f16 < int8 < int4, (f16, int8, int4)


def test_int4_quantization_keeps_shape_with_padding(rx) -> None:
    np = pytest.importorskip("numpy")
    data = np.zeros((2, 1000), dtype=np.float32)          # 1000 不是 128 的倍数
    out = np.asarray(rx._quantize_hidden(data, "int4_block128"))
    assert out.shape == (2, 1000)
    assert np.isfinite(out).all()


def test_quantize_unknown_mode_fails_loud(rx) -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        rx._quantize_hidden(np.zeros((1, 8), dtype=np.float32), "int3_block64")


def test_margin_is_top1_minus_top2_and_keeps_argmax(rx) -> None:
    np = pytest.importorskip("numpy")
    row = np.array([0.5, 4.0, 7.5, -1.0], dtype=np.float32)
    assert rx._margin(row) == pytest.approx(3.5)          # 7.5 - 4.0
    assert int(row.argmax()) == 2                          # margin 不改变 argmax 语义


def test_margin_degenerate_inputs(rx) -> None:
    np = pytest.importorskip("numpy")
    # 少于两个元素时给 nan，而不是抛异常（记录侧可识别为"无意义"）
    assert math.isnan(rx._margin(np.array([1.0], dtype=np.float32)))
    # 并列最大时边距为 0（最容易翻的情形）
    assert rx._margin(np.array([2.0, 2.0], dtype=np.float32)) == pytest.approx(0.0)


def test_upstream_int8_int4_do_not_use_hf_quant_type(rx) -> None:
    """`int8`/`int4` 必须走 bitsandbytes 显式替换。

    `load_layer_range()` 手工物化权重、不走 `BitsAndBytesConfig` ⇒ 只要把 quant_type 交给 HF，
    就会**静默回退 fp16**（本轮实测：三档 margin 连小数位都一样）。这条测试把该结论钉住。
    """
    for quant in ("int8", "int4", "nf4"):
        assert rx._QUANT_TYPE_BY_UPSTREAM[quant] is None


def test_replace_linear_helpers_exist(rx) -> None:
    """4bit 通用实现 + 三个薄包装（nf4 / fp4 / int8）都要在。"""
    for name in ("_replace_linear_4bit", "_replace_linear_nf4",
                 "_replace_linear_fp4", "_replace_linear_int8"):
        assert callable(getattr(rx, name)), name


def test_param_bytes_counts_packed_params_by_residency(rx) -> None:
    """打包参数必须按**实际驻留**估算。

    上一轮遗留：`Int8Params` 没被显式处理 ⇒ `element_size()` 按 CPU 上的 fp16 副本算，
    int8 档的上游驻留字节被报成与 fp16 相同（实测 511 MB）。
    """
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes")

    p4 = bnb.nn.Params4bit(torch.zeros(64, 128, dtype=torch.float16),
                           requires_grad=False, quant_type="nf4")
    pi8 = bnb.nn.Int8Params(torch.zeros(64, 128, dtype=torch.float16),
                            requires_grad=False, has_fp16_weights=False)
    fp16 = torch.zeros(64, 128, dtype=torch.float16)

    assert rx._param_bytes(p4) == 64 * 128 // 2            # 4bit 打包
    assert rx._param_bytes(pi8) == 64 * 128 + 64 * 4        # int8 权重 + 行级 fp32 scale
    assert rx._param_bytes(fp16) == 64 * 128 * 2            # 未打包：按 dtype
    # int8 必须明显小于 fp16（这正是修复要保证的方向）
    assert rx._param_bytes(pi8) < rx._param_bytes(fp16)
