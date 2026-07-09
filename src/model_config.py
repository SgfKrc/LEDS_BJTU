"""
多模型配置管理 -- ModelConfig dataclass + 内置模型注册表
========================================================

P3: 多模型实验支持。提供:
  - ModelConfig dataclass: 模型元数据（路径/显存/量化/上下文等）
  - 内置模型列表（默认 Qwen-1.8B + 已知实验模型）
  - 与 DB cluster_config 中用户注册模型合并的统一查询接口

门控: 实验模型仅在 CUDA 可用时暴露（server-side torch.cuda.is_available()）。
      前端通过 hasDedicatedGpu (device profile) 门控。

依赖: config.py (_get_app_root, MODEL_NAME, MODEL_PATH, GGUF_MODEL_PATH)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---- 复用 config.py 的路径工具（避免循环导入） ----


def _get_app_root() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_APP_ROOT = _get_app_root()


# ================================================================
# ModelConfig
# ================================================================

@dataclass
class ModelConfig:
    """单个模型的完整配置元数据。

    字段:
        model_id: 唯一标识符，如 "qwen-1_8b"。用于 API/DB 查找。
        name: 用户可见的显示名称，如 "Qwen-1.8B-Chat"。
        model_path: safetensors 格式模型目录的本地路径（相对 APP_ROOT 或绝对路径）。
        gguf_path: GGUF 文件路径（可选，仅 GGUF/双格式模型需要）。
        model_type: "safetensors" | "gguf" | "both"。
        recommended_vram_gb: 推荐的最低显存（GB），用于 UI 展示。
        max_context: 该模型支持的最大上下文长度。
        is_experimental: True = 用户下载的重模型，False = 内置默认模型。
        huggingface_id: HuggingFace 仓库 ID（下载参考）。
        quant_types: 支持的量化类型列表，如 ["fp16","int8","int4"]。
        description: 单行中文说明，用于 UI tooltip。
        location: "bundled"（内置）| "external"（用户下载）。
    """
    model_id: str
    name: str
    model_type: str = "safetensors"          # "safetensors" | "gguf" | "both"
    model_path: str = ""
    gguf_path: str = ""
    recommended_vram_gb: float = 3.5
    max_context: int = 4096
    is_experimental: bool = False
    huggingface_id: str = ""
    quant_types: list = field(default_factory=lambda: ["fp16", "int8", "int4"])
    description: str = ""
    location: str = "bundled"                # "bundled" | "external"


# ================================================================
# 内置模型注册表
# ================================================================

DEFAULT_MODEL_ID = "qwen-1_8b"

BUILTIN_MODELS: list[ModelConfig] = [
    ModelConfig(
        model_id="qwen-1_8b",
        name="Qwen-1.8B-Chat",
        model_type="both",
        model_path=os.path.join(_APP_ROOT, "models", "qwen-1_8b-chat"),
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen-1_8b-chat-Q4_K_M.gguf"),
        recommended_vram_gb=3.5,
        max_context=4096,
        is_experimental=False,
        huggingface_id="Qwen/Qwen-1.8B-Chat",
        quant_types=["fp16", "int8", "int4"],
        description="默认模型。1.8B 参数，Q4_K_M GGUF (1.16GB) / INT4 Safetensors (1.75GB VRAM)。适合入门级 GPU 和 CPU。",
        location="bundled",
    ),
    ModelConfig(
        model_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "qwen2.5-7b-instruct"),
        gguf_path="",
        recommended_vram_gb=16.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        quant_types=["fp16", "int8", "int4"],
        description="7B 参数指令模型。INT4 量化约需 8GB VRAM，FP16 需 16GB+。需手动下载。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen2.5-14b",
        name="Qwen2.5-14B-Instruct",
        model_type="safetensors",
        model_path=os.path.join(_APP_ROOT, "models", "qwen2.5-14b-instruct"),
        gguf_path="",
        recommended_vram_gb=32.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-14B-Instruct",
        quant_types=["int8", "int4"],
        description="14B 参数指令模型。INT4 量化约需 12GB VRAM。仅限高端 GPU。需手动下载。",
        location="external",
    ),
    ModelConfig(
        model_id="qwen2.5-7b-gguf",
        name="Qwen2.5-7B-Instruct (GGUF)",
        model_type="gguf",
        model_path="",
        gguf_path=os.path.join(_APP_ROOT, "models", "qwen2.5-7b-instruct-Q4_K_M.gguf"),
        recommended_vram_gb=6.0,
        max_context=32768,
        is_experimental=True,
        huggingface_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        quant_types=["Q4_K_M", "Q5_K_M", "Q8_0"],
        description="7B 参数 GGUF 格式。Q4_K_M ~4.7GB，CPU 可运行（速度较慢）。需手动下载。",
        location="external",
    ),
]


# ================================================================
# 查询函数
# ================================================================

def get_builtin_model(model_id: str) -> Optional[ModelConfig]:
    """从内置注册表中按 model_id 查找。"""
    for m in BUILTIN_MODELS:
        if m.model_id == model_id:
            return m
    return None


def get_builtin_models() -> list[ModelConfig]:
    """返回所有内置模型（含实验模型）。"""
    return list(BUILTIN_MODELS)


def get_default_model() -> ModelConfig:
    """返回默认模型配置。"""
    model = get_builtin_model(DEFAULT_MODEL_ID)
    if model is None:
        raise RuntimeError(f"默认模型 '{DEFAULT_MODEL_ID}' 在内置注册表中未找到")
    return model


def is_cuda_available() -> bool:
    """检测 CUDA 是否可用（server-side 门控）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_visible_models(db_experimental_models: Optional[list[dict]] = None) -> list[ModelConfig]:
    """返回当前环境可见的模型列表。

    - 始终包含默认模型
    - 实验模型仅在 CUDA 可用时包含
    - 合并内置实验模型 + DB 中用户注册的实验模型

    Args:
        db_experimental_models: 从 DB cluster_config 读取的实验模型 dict 列表。
                                每个 dict 的字段与 ModelConfig 兼容。

    Returns:
        可见模型配置列表，默认模型排在第一位。
    """
    models: list[ModelConfig] = []
    cuda_ok = is_cuda_available()

    for m in BUILTIN_MODELS:
        if not m.is_experimental:
            models.append(m)
        elif cuda_ok:
            models.append(m)

    # 合并 DB 注册的实验模型
    if cuda_ok and db_experimental_models:
        builtin_ids = {m.model_id for m in models}
        for entry in db_experimental_models:
            mid = entry.get("model_id", "")
            if not mid or mid in builtin_ids:
                continue
            try:
                mc = ModelConfig(
                    model_id=mid,
                    name=entry.get("name", mid),
                    model_type=entry.get("model_type", "safetensors"),
                    model_path=entry.get("model_path", ""),
                    gguf_path=entry.get("gguf_path", ""),
                    recommended_vram_gb=float(entry.get("recommended_vram_gb", 8.0)),
                    max_context=int(entry.get("max_context", 4096)),
                    is_experimental=True,
                    huggingface_id=entry.get("huggingface_id", ""),
                    quant_types=entry.get("quant_types", ["int4"]),
                    description=entry.get("description", ""),
                    location="external",
                )
                models.append(mc)
            except (TypeError, ValueError):
                continue

    return models


def get_model_config(
    model_id: str,
    db_experimental_models: Optional[list[dict]] = None,
) -> Optional[ModelConfig]:
    """按 model_id 查找模型配置（内置 + DB 注册）。

    Args:
        model_id: 模型唯一标识。
        db_experimental_models: DB 注册模型列表（可选）。

    Returns:
        ModelConfig 或 None（未找到时）。
    """
    # 先查内置
    builtin = get_builtin_model(model_id)
    if builtin is not None:
        return builtin

    # 再查 DB 注册
    if db_experimental_models:
        for entry in db_experimental_models:
            if entry.get("model_id") == model_id:
                try:
                    return ModelConfig(
                        model_id=model_id,
                        name=entry.get("name", model_id),
                        model_type=entry.get("model_type", "safetensors"),
                        model_path=entry.get("model_path", ""),
                        gguf_path=entry.get("gguf_path", ""),
                        recommended_vram_gb=float(entry.get("recommended_vram_gb", 8.0)),
                        max_context=int(entry.get("max_context", 4096)),
                        is_experimental=True,
                        huggingface_id=entry.get("huggingface_id", ""),
                        quant_types=entry.get("quant_types", ["int4"]),
                        description=entry.get("description", ""),
                        location="external",
                    )
                except (TypeError, ValueError):
                    return None

    return None
