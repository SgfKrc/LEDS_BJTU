"""★ A7/B7：主仓的两个 5.x 兼容补丁的单元测试。

1. `transformers5_compat.install()` 需补回 `PreTrainedModel.get_head_mask`
   （transformers 5.x 移除；`models/qwen-1_8b-chat/modeling_qwen.py:819` 在前向里调用它）。
2. `model_module._verify_and_repair_loaded_weights()` 需能发现并重载「与 safetensors 不一致」的
   权重（transformers 5.x 的「初始化覆盖已装载权重」缺陷；Qwen-1.8B 24 层 `attn.c_proj.weight` 实测中招）。

本文件**不加载真实模型**：用临时构造的 safetensors + tiny `nn.Module` 验证逻辑（离线、快）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import model_module  # noqa: E402


# ================================================================
# 1) transformers5_compat：get_head_mask
# ================================================================

class TestCompatHeadMask:
    def test_install_adds_get_head_mask(self):
        import transformers5_compat

        added = transformers5_compat.install()
        from transformers.modeling_utils import PreTrainedModel
        assert hasattr(PreTrainedModel, "get_head_mask"), "install() 应补回 get_head_mask"
        # 4.x 环境（已有该方法）时不应被记为"新增"
        assert isinstance(added, list)

    def test_get_head_mask_none_returns_list_of_none(self):
        """★ 关键语义：`head_mask is None` 时必须返回 `[None] * n`（不是 `None`）。

        返回 `None` 会让 remote code 的 `head_mask[i]` 以
        `'NoneType' object is not subscriptable` 再次失败（B15 实际踩过）。
        """
        import transformers5_compat

        transformers5_compat.install()
        from transformers.modeling_utils import PreTrainedModel

        n = 24
        out = PreTrainedModel.get_head_mask(object(), None, n)
        assert isinstance(out, list) and len(out) == n and all(x is None for x in out)

    def test_get_head_mask_non_none_is_fail_loud(self):
        """非 None 时必须 fail-loud（避免静默错算）。"""
        import transformers5_compat

        transformers5_compat.install()
        from transformers.modeling_utils import PreTrainedModel

        with pytest.raises((RuntimeError, TypeError)):
            PreTrainedModel.get_head_mask(object(), torch.zeros(1), 4)


# ================================================================
# 2) _verify_and_repair_loaded_weights：闭环修复
# ================================================================

class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x):  # pragma: no cover - 测试不跑前向
        return self.b(self.a(x))


def _write_safetensors(tmp_path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}), encoding="utf-8")


def test_guard_repairs_mismatched_weights(tmp_path):
    """人为把模型里的某个 weight 改坏 ⇒ 守卫应检测到并**从 safetensors 修回**。"""
    model = _Tiny()
    good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _write_safetensors(tmp_path, good)

    with torch.no_grad():
        model.a.weight.zero_()                     # 模拟被 `_init_weights` 覆盖/未初始化
    assert not torch.equal(model.a.weight, good["a.weight"])

    report = model_module._verify_and_repair_loaded_weights(model, str(tmp_path))
    assert report is not None
    assert report["repaired"] == 1, report
    assert "a.weight" in report["mismatched_head"]
    assert torch.equal(model.a.weight, good["a.weight"]), "守卫应把权重修回 safetensors 的值"


def test_guard_is_noop_when_consistent(tmp_path):
    """全部一致 ⇒ 不做任何修改，`repaired == 0`。"""
    model = _Tiny()
    good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _write_safetensors(tmp_path, good)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = model_module._verify_and_repair_loaded_weights(model, str(tmp_path))
    assert report is not None and report["repaired"] == 0, report
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_guard_returns_none_on_bad_path():
    """目录不可用时**绝不抛**（只返回 None）——守卫不得让加载失败。"""
    model = _Tiny()
    assert model_module._verify_and_repair_loaded_weights(model, "/definitely/not/a/dir") is None
    assert model_module._verify_and_repair_loaded_weights(model, "") is None


def test_is_transformers_5_or_newer_matches_env():
    import transformers

    major = int(str(transformers.__version__).split(".")[0])
    assert model_module._is_transformers_5_or_newer() == (major >= 5)
