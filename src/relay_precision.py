"""relay_precision.py — 接力两侧**权重/工件精度对齐**的实测判据与装配期检查。

背景（2026-10-08）：D→L 跨框架接力的端到端误差**由两侧权重网格差主导** —— 不是归约顺序
（`1e-7` 级）、也不是算子实现（`1e-7` 级）。同一条真链路（Qwen2.5-0.5B、切点 `K=16`、
3 prompt × 160 步）实测：

| 上游（前 K 层） | 下游（裁层 GGUF） | 第 16 层 `rel_err` | 端到端 head/tail `mean_rel` | 逐 token argmax 一致 |
| --- | --- | --- | --- | --- |
| fp32（原始 HF） | `Q4_K_M` | `1.96e-01` | `1.1e-2 … 3.3e-2` | 165/174、163/174、166/172 |
| `Q4_K_M` 反量化（**同源**） | `Q4_K_M` | `4.71e-03` | `3.2e-3 … 8.7e-3` | 173/174、168/174、171/172 |
| `f16` 反量化（**同源**） | `f16` | `3.33e-04` | `1.6e-4 … 4.4e-4` | 173/174、**174/174**、**172/172** |

⇒ **「两侧同源 + 同精度」是唯一能把端到端 `rel` 压到 `1e-4` 级、argmax 基本全同的配置**；
而 `fp32 × Q4_K_M`（旧默认组合）是**最大误差源**（第 16 层差 `1.96e-01`，比同源 f16 大 **590×**）。
本模块把这张表固化成**可调用判据**，并在接力装配时把"对不对齐"记进日志与合同。

纪律：
* **判据只覆盖实测过的组合**；未测组合返回 `unknown` 并显式要求先实测 —— **不猜**。
* **不改变任何路由/准入语义**：本模块只**记录**（见 `RelayXFrameEvidence.precision_verdict`）；
  是否据此改默认部署是**独立决定**。
* 只有 `QLH_RELAY_PRECISION_STRICT=1` 才 fail-loud（抛 `RelayPrecisionMismatch`），默认只 WARN。

`general.file_type` → 档名映射来源：llama.cpp `llama_ftype` 枚举。**本仓实测确认**的是
`1`（f16）与 `15`（Q4_K_M）；其余为枚举直译（未在本仓逐档验证），已在表中逐项标注。
"""

from __future__ import annotations

import os
import re
import threading

__all__ = [
    "GGUF_FILE_TYPE_NAMES",
    "LEVEL_CRITICAL",
    "LEVEL_OK",
    "LEVEL_UNKNOWN",
    "LEVEL_WARN",
    "MEASURED_PRECISION_PAIRS",
    "STRICT_ENV",
    "RelayPrecisionMismatch",
    "RelayPrecisionVerdict",
    "check_relay_precision",
    "evaluate_relay_precision",
    "normalize_precision",
    "note_downstream_precision",
    "note_upstream_precision",
    "precision_from_gguf_file_type",
    "precision_from_path",
    "relay_precision_report",
    "reset_relay_precision_notes",
]

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_CRITICAL = "critical"
LEVEL_UNKNOWN = "unknown"

#: 严格模式开关：`1`/`true`/`yes`/`on` ⇒ 两侧精度不对齐时 fail-loud。
STRICT_ENV = "QLH_RELAY_PRECISION_STRICT"

#: GGUF `general.file_type` → 档名。
#: `1`（f16）与 `15`（Q4_K_M）为**本仓实测**（`models/qwen25-05b-f16.gguf`、
#: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`）；其余为 llama.cpp `llama_ftype` 枚举直译。
GGUF_FILE_TYPE_NAMES = {
    0: "f32",
    1: "f16",          # 实测
    2: "q4_0",
    3: "q4_1",
    7: "int8",         # Q8_0 族
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",      # 实测
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
    33: "bf16",
}

#: 写法归一（键为小写去分隔符后的形态）。
_PRECISION_ALIASES = {
    "f32": "f32", "fp32": "f32", "float32": "f32",
    "f16": "f16", "fp16": "f16", "float16": "f16", "half": "f16",
    "bf16": "bf16", "bfloat16": "bf16",
    "int8": "int8", "q8": "int8", "q80": "int8", "q8_0": "int8",
    # ⚠️ `int4` / `int8` 是 `config.QUANT_TYPE` 的**配置意图**，不是具体网格
    #    （`int4` 可能配 f16 反量化目录）⇒ 原样保留，不去猜成某个 `q*` 档。
    "int4": "int4",
    "q4km": "q4_k_m", "q4_k_m": "q4_k_m", "q4ks": "q4_k_s", "q4_k_s": "q4_k_s",
    "q40": "q4_0", "q4_0": "q4_0", "q41": "q4_1", "q4_1": "q4_1",
    "q5km": "q5_k_m", "q5_k_m": "q5_k_m", "q5ks": "q5_k_s", "q5_k_s": "q5_k_s",
    "q50": "q5_0", "q5_0": "q5_0", "q51": "q5_1", "q5_1": "q5_1",
    "q6k": "q6_k", "q6_k": "q6_k",
    "q3ks": "q3_k_s", "q3_k_s": "q3_k_s", "q3km": "q3_k_m", "q3_k_m": "q3_k_m",
    "q3kl": "q3_k_l", "q3_k_l": "q3_k_l", "q2k": "q2_k", "q2_k": "q2_k",
}

#: 路径标记识别顺序（先长后短，避免 `q4_k_m` 被 `q4_0` 之类抢先）。
_PATH_MARKERS = (
    ("q4_k_m", "q4_k_m"), ("q4km", "q4_k_m"), ("q4_k_s", "q4_k_s"), ("q4ks", "q4_k_s"),
    ("q5_k_m", "q5_k_m"), ("q5km", "q5_k_m"), ("q6_k", "q6_k"), ("q6k", "q6_k"),
    ("q3_k_m", "q3_k_m"), ("q3km", "q3_k_m"), ("q2_k", "q2_k"), ("q2k", "q2_k"),
    ("q4_0", "q4_0"), ("q5_0", "q5_0"),
    ("bf16", "bf16"), ("f32", "f32"), ("fp32", "f32"), ("f16", "f16"), ("fp16", "f16"),
    ("dequant", "unknown"),   # 只说明"是反量化目录"，档位本身由上面的标记决定
)

#: **实测**判据表：`(上游档, 下游档) -> (rel_err, level, 依据文件)`。
#: 数值取自 `build/keephead/xframe-layerprofile-*.json` 的 `rel_err`（第 16 层**单层**相对误差），
#: 保留 JSON 里的**完整精度**，避免"判据比台账粗"造成对不上。
#: `tests/test_relay_precision.py` 会**直接读这些 JSON 交叉校验**（缺文件则跳过）。
MEASURED_PRECISION_PAIRS = {
    ("f32", "q4_k_m"): (1.9634151601782326e-01, LEVEL_CRITICAL, "xframe-layerprofile16.json"),
    ("q4_k_m", "q4_k_m"): (4.713489493590314e-03, LEVEL_WARN,
                           "xframe-layerprofile-05b-q4km-dequant.json"),
    ("f16", "f16"): (3.325006366906236e-04, LEVEL_OK, "xframe-layerprofile-05b-f16-dequant.json"),
}

#: 台账 JSON 的所在目录（判据表的交叉校验用；见 `tests/test_relay_precision.py`）。
EVIDENCE_DIR = "build/keephead"

#: `level` 的排序权重（越大越严重），供"取较坏者"使用。
_LEVEL_RANK = {LEVEL_OK: 0, LEVEL_UNKNOWN: 1, LEVEL_WARN: 2, LEVEL_CRITICAL: 3}


class RelayPrecisionMismatch(RuntimeError):
    """严格模式（`QLH_RELAY_PRECISION_STRICT=1`）下两侧精度未对齐时抛出。"""


def precision_from_gguf_file_type(value) -> str:
    """GGUF `general.file_type` → 档名；未登记的值回 `ftype:<n>`（**不猜**）。

    ⚠️ llama-cpp-python 把该元数据暴露成**字符串**（实测 `'1'` / `'15'`）⇒ 这里先 `int()`。
    """
    try:
        code = int(str(value).strip())
    except (TypeError, ValueError):
        return "unknown"
    return GGUF_FILE_TYPE_NAMES.get(code, f"ftype:{code}")


def precision_from_path(path) -> str:
    """从路径/文件名里的标记识别档位（反量化目录的命名约定）。

    例：`build/keephead/qwen25-05b-f16-dequant` → `f16`；
    `qwen25-05b-q4km-dequant` → `q4_k_m`。识别不出回 `unknown`（**不猜**）。
    """
    if path is None:
        return "unknown"
    text = str(path).lower().replace("\\", "/")
    if not text:
        return "unknown"
    parts = [part for part in text.split("/") if part]
    name = parts[-1] if parts else text
    for marker, label in _PATH_MARKERS:
        # 词边界：`qwen2.5-0.5b` 里不能因为 `q4`/`f16` 之类被误判 —— 用非字母数字边界锚定。
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", name):
            return label
    return "unknown"


def normalize_precision(label) -> str:
    """把各处口径（`fp16` / `int4` / `Q4_K_M` / file_type 码 / 路径）归一到统一档名。

    无法识别 ⇒ `unknown`（**不猜**）。空值 ⇒ `unknown`。
    """
    if label is None:
        return "unknown"
    if isinstance(label, bool):
        return "unknown"
    if isinstance(label, int):
        return precision_from_gguf_file_type(label)
    text = str(label).strip()
    if not text:
        return "unknown"
    if text.isdigit():
        return precision_from_gguf_file_type(text)
    if "/" in text or "\\" in text:
        found = precision_from_path(text)
        if found != "unknown":
            return found
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    found = _PRECISION_ALIASES.get(key)
    if found is not None:
        return found
    # `f16-dequant` / `qwen25-05b-f16.gguf` 这类没有 `/` 但含标记的写法 ⇒ 交给路径识别。
    return precision_from_path(text)


class RelayPrecisionVerdict:
    """两侧精度对齐判据（不可变值对象）。

    `aligned` 只在 `level == ok`（实测 `rel < 1e-3`，即两侧同源 f16）时为 True ——
    与别处"合同字段默认保守"的纪律一致：**没对齐就不算对齐**。
    """

    __slots__ = ("upstream", "downstream", "expected_rel", "level", "advice", "evidence")

    def __init__(self, upstream: str, downstream: str, expected_rel: float | None,
                 level: str, advice: str, evidence: str = "") -> None:
        self.upstream = upstream
        self.downstream = downstream
        self.expected_rel = expected_rel
        self.level = level
        self.advice = advice
        self.evidence = evidence

    @property
    def aligned(self) -> bool:
        """只有 `ok`（实测 `rel < 1e-3`，即两侧同源 f16）才算对齐 —— 没对齐就不算对齐。"""
        return self.level == LEVEL_OK

    def to_dict(self) -> dict:
        return {
            "upstream": self.upstream,
            "downstream": self.downstream,
            "expected_rel": self.expected_rel,
            "level": self.level,
            "aligned": self.aligned,
            "advice": self.advice,
            "evidence": self.evidence,
        }

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f"RelayPrecisionVerdict(up={self.upstream!r}, down={self.downstream!r}, "
                f"level={self.level!r}, expected_rel={self.expected_rel!r})")

    def __eq__(self, other) -> bool:
        if not isinstance(other, RelayPrecisionVerdict):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.upstream, self.downstream, self.level, self.expected_rel))


_ADVICE_OK = ("两侧同源且同精度（实测端到端 rel 1.6e-4…4.4e-4，argmax 基本全同）—— 保持。")
_ADVICE_WARN = ("两侧同源但精度为 4-bit 类（实测第 16 层 rel_err 4.7e-3，端到端 argmax 173/174 等）"
                "—— 可接受；要 argmax 全同需两侧都用 f16。")
_ADVICE_CRITICAL = ("⚠️ 两侧精度不同源（实测第 16 层 rel_err 1.96e-1，端到端 argmax 165/174）"
                    "—— 这是当前最大误差源。请让上游加载**与下游同源**的反量化目录"
                    "（`scripts/gguf_dequant_to_hf.py`），或让两侧都用同一精度的工件。")


def evaluate_relay_precision(upstream_quant, downstream_quant) -> RelayPrecisionVerdict:
    """按**实测**判据表评估两侧精度对齐；未测组合回 `unknown` 并说明要先实测。

    参数接受任意口径（`fp16` / `int4` / `Q4_K_M` / `general.file_type` 码 / 路径），
    内部经 `normalize_precision` 归一。
    """
    up = normalize_precision(upstream_quant)
    down = normalize_precision(downstream_quant)
    pair = MEASURED_PRECISION_PAIRS.get((up, down))
    if pair is not None:
        rel, level, evidence = pair
        advice = {LEVEL_OK: _ADVICE_OK, LEVEL_WARN: _ADVICE_WARN,
                  LEVEL_CRITICAL: _ADVICE_CRITICAL}[level]
        return RelayPrecisionVerdict(up, down, rel, level, advice, evidence)
    advice = (f"组合 上游={up!r} × 下游={down!r} **未实测** —— 先跑 "
              f"`scripts/xframe_dl_relay_positions.py` 取该组合的 rel 与 argmax 一致数，"
              f"再把结果登记进 `MEASURED_PRECISION_PAIRS`。**不要凭推理选档**。")
    return RelayPrecisionVerdict(up, down, None, LEVEL_UNKNOWN, advice, "")


# ------------------------------------------------------------------ 装配期登记与检查
#
# 设计：两个入口（下游 `LlamaCppEngine.forward_layers_from_hidden`、上游
# `ModelManager.load_layer_range`）**各自登记自己一侧的档位**，两侧都到齐时才比对一次。
# 这样**不需要改任何方法签名**（零侵入），也不需要调用方显式传两侧信息。

_notes: dict[str, str | None] = {"upstream": None, "downstream": None}
_checked: dict[tuple[str, str], RelayPrecisionVerdict] = {}
_lock = threading.Lock()


def reset_relay_precision_notes() -> None:
    """清空进程内登记（测试与"换模型后重认"用）。"""
    with _lock:
        _notes["upstream"] = None
        _notes["downstream"] = None
        _checked.clear()


def _strict_enabled(strict: bool | None) -> bool:
    if strict is not None:
        return bool(strict)
    return str(os.environ.get(STRICT_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


def _record(side: str, label) -> str:
    value = normalize_precision(label)
    with _lock:
        _notes[side] = value
    return value


def note_upstream_precision(label=None, *, model_path=None, dtype=None,
                            quant_type=None) -> str:
    """登记**上游**（PyTorch 侧）档位，返回归一结果。

    优先级：显式 `label` > `model_path` 路径标记 > `dtype` 实际精度 > `quant_type` 配置意图。
    ⚠️ `quant_type` 只是**配置意图**（`config.QUANT_TYPE` 的 `fp16`/`int8`/`int4`），
    未必等于实际加载的网格（例如 `int4` 配 f16 反量化目录）⇒ 它排最后。
    """
    if label is not None:
        return _record("upstream", label)
    found = precision_from_path(model_path) if model_path else "unknown"
    if found != "unknown":
        return _record("upstream", found)
    normalized_dtype = normalize_precision(dtype) if dtype is not None else "unknown"
    if normalized_dtype != "unknown":
        return _record("upstream", normalized_dtype)
    return _record("upstream", quant_type)


def note_downstream_precision(label=None, *, metadata=None, model_path=None,
                              quant_type=None) -> str:
    """登记**下游**（llama.cpp 裁层工件）档位，返回归一结果。

    优先级：显式 `label` > `metadata["general.file_type"]`（**权威**，实测存在）
    > `model_path` 路径标记 > `quant_type` 配置意图。
    """
    if label is not None:
        return _record("downstream", label)
    if metadata:
        try:
            file_type = metadata.get("general.file_type")
        except AttributeError:
            file_type = None
        if file_type is not None:
            found = precision_from_gguf_file_type(file_type)
            if found != "unknown":
                return _record("downstream", found)
    found = precision_from_path(model_path) if model_path else "unknown"
    if found != "unknown":
        return _record("downstream", found)
    return _record("downstream", quant_type)


def relay_precision_report() -> dict:
    """当前登记 + 判据（供健康检查/日志/诊断用）。"""
    with _lock:
        up, down = _notes["upstream"], _notes["downstream"]
    if up is None or down is None:
        return {"upstream": up, "downstream": down, "verdict": None,
                "pending": "upstream" if up is None else "downstream"}
    return {"upstream": up, "downstream": down,
            "verdict": evaluate_relay_precision(up, down).to_dict(), "pending": None}


def check_relay_precision(*, strict: bool | None = None, logger=None) -> RelayPrecisionVerdict | None:
    """两侧都登记后比对；**只记录**（默认 WARN），`strict` 时才 fail-loud。

    * 未登记齐 ⇒ 返回 `None`（**不报错**：只用一侧是正常形态，例如纯 D 档或纯 L→L）；
    * 对齐（`ok`）⇒ 缓存后**每次直接返回**，不重复建对象、不重复记日志；
    * 未对齐 ⇒ **只 WARN 一次**（按 `(上游, 下游)` 去重），避免每步刷屏；
    * `strict=True` 且未对齐 ⇒ 抛 `RelayPrecisionMismatch`（每次调用都抛，不给"跑一会儿再说"的机会）。
    """
    with _lock:
        up, down = _notes["upstream"], _notes["downstream"]
        if up is None or down is None:
            return None
        cached = _checked.get((up, down))
    if cached is not None and cached.level == LEVEL_OK:
        return cached
    verdict = cached if cached is not None else evaluate_relay_precision(up, down)
    if verdict.level == LEVEL_OK:
        with _lock:
            _checked[(up, down)] = verdict
        return verdict
    if _strict_enabled(strict):
        raise RelayPrecisionMismatch(
            f"relay 两侧精度未对齐：上游={up} 下游={down} "
            f"(level={verdict.level}, expected_rel={verdict.expected_rel})；{verdict.advice}")
    with _lock:
        first = (up, down) not in _checked
        _checked[(up, down)] = verdict
    if first:
        message = (f"relay 两侧精度未对齐：上游={up} 下游={down} "
                   f"(level={verdict.level}, expected_rel={verdict.expected_rel})；{verdict.advice}")
        if logger is not None:
            try:
                logger.warning(message)
                return verdict
            except Exception:  # noqa: BLE001 - 记日志失败绝不能影响推理
                pass
        import logging
        logging.getLogger(__name__).warning(message)
    return verdict
