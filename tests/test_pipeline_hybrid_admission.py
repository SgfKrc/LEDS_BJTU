"""★ #31 M1/M2 的回归：hybrid（`qwen3_5` / `qwen3_5_text`）在层流水线的**准入与布局**。

为什么要有它：`#31` 的 ①③ 说的是「`pipeline_model_descriptor` 的白名单不含 hybrid」+「另外
**四处独立的** `{"qwen","qwen2"}` 硬编码副本」⇒ 只改一处不够。这里把**单一事实来源**和
**布局登记**都钉住，并守住"`qwen3` 有布局但**无执行器**，不得被连带放行"这条边界（`#31` M2 的风险项）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from pipeline_model_descriptor import (  # noqa: E402
    _ARCHITECTURE_LAYOUTS,
    PIPELINE_RUNTIME_MODEL_TYPES,
)


# ── M1：白名单 ──────────────────────────────────────────────────────────────


def test_hybrid_model_types_are_runtime_supported() -> None:
    """★ `qwen3_5`（多模态包装）与 `qwen3_5_text`（文本子 config）**都**要在白名单里。

    两个都要：外层 config 是 `qwen3_5`，而主仓的层加载 / `forward_lm_head` 路径报出来的是
    **`qwen3_5_text`**（实测 `RuntimeError: 模型架构 qwen3_5_text 缺少最终 Norm`）。
    """
    assert "qwen3_5" in PIPELINE_RUNTIME_MODEL_TYPES
    assert "qwen3_5_text" in PIPELINE_RUNTIME_MODEL_TYPES


def test_qwen3_still_not_runtime_supported() -> None:
    """★ 边界（`#31` M2 明说的风险）：`qwen3` **有布局表但没有执行器** ⇒ 不得被连带放行。"""
    assert "qwen3" in _ARCHITECTURE_LAYOUTS          # 布局在
    assert "qwen3" not in PIPELINE_RUNTIME_MODEL_TYPES  # 但不准入


def test_gemma4_unified_still_blocked() -> None:
    """`gemma4_unified` 仍**不**在白名单（它需要隔离 sidecar，是设计内的 fail-closed）。"""
    assert "gemma4_unified" not in PIPELINE_RUNTIME_MODEL_TYPES


# ── M1：布局登记 ────────────────────────────────────────────────────────────


def test_qwen3_5_text_layout_is_registered() -> None:
    """★ 只加白名单**不生效** —— 布局表也要有，否则 `_ARCHITECTURE_LAYOUTS.get()` 返回 None。"""
    layout = _ARCHITECTURE_LAYOUTS.get("qwen3_5_text")

    assert layout is not None
    assert layout["layer_prefix"] == "model.language_model.layers."
    assert layout["final_norm_prefixes"] == ("model.language_model.norm.",)


def test_qwen3_5_text_layout_matches_real_artifact_keys() -> None:
    """★ 用**真工件的 key 形态**验证前缀口径（`qwen3-5-2b` 的 index 实测 318 个层 key）。

    这里不依赖那份工件，只用它的**键形态**（已在 `#31 §31.6.1` 记录）做合成断言。
    """
    layout = _ARCHITECTURE_LAYOUTS["qwen3_5_text"]
    pattern = layout["layer_pattern"]

    assert pattern.match("model.language_model.layers.0.self_attn.q_proj.weight")
    assert pattern.match("model.language_model.layers.23.mlp.gate_proj.weight")
    # 视觉塔 / MTP 不属于**文本**布局（它们在多模态外层口径里）
    assert not pattern.match("model.visual.blocks.0.attn.qkv.weight")
    assert not pattern.match("mtp.layers.0.self_attn.q_proj.weight")
    assert layout["visual_prefixes"] == ()
    assert layout["mtp_prefixes"] == ()


def test_qwen3_5_layout_keeps_multimodal_prefixes() -> None:
    """`qwen3_5`（外层多模态口径）仍要认得 visual / mtp 分量 —— M3 的"分量归零"依据在这。"""
    layout = _ARCHITECTURE_LAYOUTS["qwen3_5"]

    assert layout["visual_prefixes"] == ("model.visual.", "visual.")
    assert layout["mtp_prefixes"] == ("mtp.",)


# ── M2：单一事实来源 ────────────────────────────────────────────────────────


def test_no_duplicate_hardcoded_model_type_sets_in_src() -> None:
    """★ `#31` M2：`src/` 里**不得**再出现 `{"qwen", "qwen2"}` 这类硬编码副本。

    这是个静态扫描守卫：加了 hybrid 之后，任何漏改的副本都会把 hybrid 静默拦掉
    （`#31` ①③ 描述的就是这种"四处各一份"的形态）。
    """
    import re

    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # ⚠️ 跳过注释行：本仓惯用注释记「此前这里硬编码 {"qwen","qwen2"}」，
            #    那是说明文字、不是代码（首次写这个守卫时就被自己的注释绊了一次）。
            if line.lstrip().startswith("#"):
                continue
            if re.search(r'\{\s*"qwen"\s*,\s*"qwen2"\s*\}', line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")

    assert offenders == [], f"仍有硬编码的架构集合副本：{offenders}"


def test_single_source_of_truth_is_importable_from_consumers() -> None:
    """★ 四个消费方都要能 import 到同一个常量（M2 的"单一事实来源"）。"""
    import inference_service.peer as peer_module
    import scheduler as scheduler_module
    import scheduler_pipeline as pipeline_module

    assert scheduler_module.PIPELINE_RUNTIME_MODEL_TYPES is PIPELINE_RUNTIME_MODEL_TYPES
    assert pipeline_module.PIPELINE_RUNTIME_MODEL_TYPES is PIPELINE_RUNTIME_MODEL_TYPES
    assert peer_module.PIPELINE_RUNTIME_MODEL_TYPES is PIPELINE_RUNTIME_MODEL_TYPES
