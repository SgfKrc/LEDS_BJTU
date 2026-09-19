"""transformers5_compat.py — 让旧栈（transformers_stream_generator）在 transformers 5.x 下可导入

## 问题

主仓曾把运行时的 `transformers` 钉在 4.x，硬原因就是本模块处理的这件事。

`transformers_stream_generator`（Qwen-1.8B remote code 的 `chat_stream` 依赖）在
`main.py` 里做这些导入（实测 0.0.5）：

    from transformers import (
        GenerationConfig, GenerationMixin, LogitsProcessorList, StoppingCriteriaList,
        DisjunctiveConstraint, BeamSearchScorer, PhrasalConstraint,
        ConstrainedBeamSearchScorer, PreTrainedModel,
    )
    from transformers.generation.utils import GenerateOutput, SampleOutput, logger

其中 **`DisjunctiveConstraint` / `BeamSearchScorer` / `PhrasalConstraint` /
`ConstrainedBeamSearchScorer`（顶层）与 `SampleOutput`（`generation.utils`）
在 transformers 5.x 已被移除**（`transformers.generation.beam_constraints` 模块也没了）。

更麻烦的是触发点：transformers 5.x 的 `dynamic_module_utils.check_imports` 在加载
**任何 remote code**（例如 `models/qwen-1_8b-chat/modeling_qwen.py`）时会逐个 import
该文件声明的依赖，而**非 "No module named" 的 ImportError 会直接抛出** ——
于是连「加载模型」这一步都过不去（报 `cannot import name 'DisjunctiveConstraint'`）。

## 做法

在**导入 `transformers` 之后**，把缺失的名字补回对应命名空间。主仓推理路径
**从不调用** remote code 的 `chat_stream`（PyTorch 路径走 tokenizer + `model.generate`
+ 文本流；`_forward_qwen_layers` 只做层前向），因此占位类只需「存在」即可让导入通过。

⚠️ 占位类**一旦真的被实例化或调用就显式报错**（fail-loud），避免"看起来能用、实际静默出错"。
若将来确实需要约束采样/流式，请改用 transformers 5.x 自带的 logits processor 方案。

## 用法

`src/model_module.py` 在导入 `transformers` 之后调用 `install()`。
**不要放到 `src/config.py`**：那里被入口模块导入，而
`tests/test_inference_service_protocol.py::test_entry_module_no_heavy_imports`
明确要求入口顶层不拉 transformers。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 需要补回的符号 → 所在的命名空间路径。
#: `transformers` 顶层 4 个 + `transformers.generation.utils` 1 个（实测 0.0.5 的导入面）。
_REMOVED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "DisjunctiveConstraint": ("transformers",),
    "BeamSearchScorer": ("transformers",),
    "PhrasalConstraint": ("transformers",),
    "ConstrainedBeamSearchScorer": ("transformers",),
    "SampleOutput": ("transformers.generation.utils",),
}

_HINT = (
    "该符号已在 transformers 5.x 中移除；这里只是占位，用于让 transformers_stream_generator "
    "可导入（主仓推理路径不调用它）。若确实需要约束采样/流式，请改用 transformers 5.x 自带的 "
    "logits processor 方案。"
)


class _RemovedInTransformers5:
    """占位类：一旦被实例化或调用就 fail-loud。"""

    _name = "?"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"{self._name}: {_HINT}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - 防御
        raise RuntimeError(f"{self._name}: {_HINT}")


def _resolve(namespace: str):
    """按点分路径取模块（取不到返回 None，不抛）。"""
    import importlib

    try:
        return importlib.import_module(namespace)
    except Exception:  # noqa: BLE001 - 模块不存在时无需补
        return None


def install() -> list[str]:
    """把缺失符号补回对应命名空间；返回本次补回的符号名列表（幂等）。"""
    try:
        import transformers  # noqa: F401  (确保顶层已导入)
    except ImportError:  # pragma: no cover - 环境相关
        logger.debug("transformers 不可导入，跳过 5.x 兼容 shim")
        return []

    added: list[str] = []
    for name, namespaces in _REMOVED_SYMBOLS.items():
        for namespace in namespaces:
            module = _resolve(namespace)
            if module is None or hasattr(module, name):
                continue  # 4.x（或上游将来补回）⇒ 什么都不用做
            placeholder = type(name, (_RemovedInTransformers5,), {"_name": f"{namespace}.{name}"})
            setattr(module, name, placeholder)
            added.append(f"{namespace}.{name}")

    added.extend(_install_legacy_model_apis())

    if added:
        logger.debug(
            "已为 transformers %s 补回 %d 个已移除符号以兼容 transformers_stream_generator: %s",
            getattr(_resolve("transformers"), "__version__", "?"),
            len(added),
            ", ".join(added),
        )
    return added


def _install_legacy_model_apis() -> list[str]:
    """补回 5.x 移除、但 **remote code 的前向会调用**的 `PreTrainedModel` 老 API。

    ★ A7：`models/qwen-1_8b-chat/modeling_qwen.py:819` 在前向里调用
    `self.get_head_mask(head_mask, self.config.num_hidden_layers)`（`:896` 随后用
    `head_mask[i]`），而 `PreTrainedModel.get_head_mask` 在 transformers 5.x **已被移除**
    ⇒ 不补则**加载能过、前向必崩**（`AttributeError: 'QWenModel' object has no attribute
    'get_head_mask'`）。

    这里按 **4.x 语义**补回：`head_mask is None` 时返回 **`[None] * num_hidden_layers`**
    （⚠️ 不是 `None` —— 返回 `None` 会让 remote code 的 `head_mask[i]` 以
    `'NoneType' object is not subscriptable` 再次失败；B15 实际踩过这一步）。
    非 `None` 时 **fail-loud**（避免"看似能用、实际静默错算"）：QLH 推理路径从不传 head_mask。
    """
    added: list[str] = []
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as exc:  # noqa: BLE001 - 环境相关
        logger.debug("补回老 API：无法导入 PreTrainedModel（%s）", exc)
        return added

    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(  # noqa: D401 - 与上游签名保持一致
            self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
        ):
            if head_mask is None:
                return [None] * int(num_hidden_layers)
            raise RuntimeError(
                "QLH 的 5.x 兼容补丁只支持 head_mask=None（推理路径不传 head_mask）；"
                "transformers 5.x 已移除 `PreTrainedModel.get_head_mask` 与其 "
                "`_convert_head_mask_to_5d`，需要非 None 时请改用上游新的 mask 方案。"
            )

        PreTrainedModel.get_head_mask = get_head_mask
        added.append("PreTrainedModel.get_head_mask")

    return added
