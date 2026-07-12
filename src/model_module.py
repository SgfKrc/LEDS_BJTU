"""
模型模块 — 模型加载、量化、算子融合、层级拆分、前向推理
==========================================================
功能职责:
1. 双引擎推理 — 自动选择 PyTorch (CUDA) 或 llama.cpp (CPU/集显)
2. 原版 Qwen-1.8B 模型加载（PyTorch + HuggingFace Transformers）
3. INT4/INT8 量化（bitsandbytes 加载时量化，CUDA only）
4. 自动算子融合 torch.compile（可选）
5. 模型层级拆分（按配置分配给主/从节点）
6. 模型前向推理入口

引擎选择逻辑:
  - CUDA 可用 → PyTorch + bitsandbytes（INT4 量化，显存 ~1.75 GB）
  - CPU / 集显 → llama.cpp + GGUF（Q4_K_M 量化，内存 ~1.2 GB）
  - 手动覆盖: config.INFERENCE_ENGINE = "pytorch" | "llama_cpp"

依赖:
  PyTorch 栈: torch, transformers, bitsandbytes
  llama.cpp 栈: llama-cpp-python (pip install llama-cpp-python)
"""

import logging
import os
import threading
import time
from typing import Tuple, Optional, Dict, Any, List

import psutil
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    MODEL_NAME, MODEL_PATH, GGUF_MODEL_PATH,
    QUANT_TYPE, USE_COMPILE,
    DEVICE, TRUST_REMOTE_CODE,
    INFERENCE_ENGINE,
    TOTAL_MODEL_LAYERS, DEFAULT_LAYER_CONFIG,
)

import model_config as mc

logger = logging.getLogger(__name__)


class ModelManager:
    """
    模型管理器：双引擎架构，自动选择最优推理后端。

    引擎:
      - PyTorch: CUDA + bitsandbytes INT4/INT8/FP16 量化（主节点 / 独显设备）
      - llama.cpp: GGUF Q4_K_M 等量化（边缘 / 集显 / CPU-only 设备）

    量化说明:
        bitsandbytes 采用"加载时量化"——权重在磁盘保持 FP16，
        加载到 GPU 时由 BitsAndBytesConfig 实时转换为 INT4/INT8。
        切换量化模式只需修改 config.QUANT_TYPE 并重新加载。
    """

    def __init__(self):
        # PyTorch 引擎
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.quant_type: Optional[str] = None
        self.layer_range: Optional[Tuple[int, int]] = None  # (start, end) 当前加载的层范围；None=完整模型
        self._layer_has_embedding: bool = True
        self._layer_has_lm_head: bool = True
        self._model_layers: int = 0          # 当前加载的层数（range 或 full）
        self._total_model_layers: int = 0    # 完整模型的总层数（加载时记录，load_layer_range 不覆盖）

        # llama.cpp 引擎（延迟导入 + 延迟加载）
        self._llama_engine = None   # LlamaCppEngine 实例
        self._engine_type: str = ""  # "pytorch" | "llama_cpp"

        # P3: 多模型支持 — 当前活跃的模型 ID
        self._active_model_id: str = mc.DEFAULT_MODEL_ID
        self._previous_engine_type: str = ""      # 用于 rollback
        self._previous_quant_type: Optional[str] = None

        # 模型路径记录（供 SHA256 一致性校验等用途）
        self._model_path: Optional[str] = None

        # 并发保护锁 — 防止推理与模型切换之间的数据竞争
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载（兼容 PyTorch 和 llama.cpp 双引擎）。"""
        if self._engine_type == "llama_cpp":
            return self._llama_engine is not None
        return self.model is not None

    # ================================================================
    # 引擎选择
    # ================================================================

    @staticmethod
    def select_engine(profile: dict = None) -> str:
        """
        根据硬件环境和配置选择推理引擎。

        决策优先级:
          1. config.INFERENCE_ENGINE 显式指定（"pytorch" / "llama_cpp"）
          2. "auto" → 检测 CUDA 可用性
             - CUDA 可用 → "pytorch"
             - CUDA 不可用 → "llama_cpp"

        Args:
            profile: 设备画像 dict（可选，用于更精确的判断）

        Returns:
            "pytorch" 或 "llama_cpp"
        """
        # 手动覆盖
        if INFERENCE_ENGINE == "pytorch":
            logger.info("引擎: PyTorch (手动指定)")
            return "pytorch"
        if INFERENCE_ENGINE == "llama_cpp":
            logger.info("引擎: llama.cpp (手动指定)")
            return "llama_cpp"

        # 自动检测
        has_cuda = torch.cuda.is_available()

        # 进一步检查设备画像
        if profile:
            tier = profile.get("tier", "laptop")
            # 检查所有 GPU（而非仅选中 GPU），避免游戏本默认选集显时误判
            gpus = profile.get("gpus", [])
            any_cuda_gpu = any(
                g.get("cuda_available", False)
                for g in gpus
            ) if gpus else has_cuda
            # 单个选中 GPU 的 CUDA 状态（向后兼容）
            gpu_info = profile.get("gpu", {})
            selected_cuda = gpu_info.get("cuda_available", False) if gpu_info else False

            # 关键修正：只要任意 GPU 有 CUDA，就认为 CUDA 可用
            profile_has_cuda = any_cuda_gpu or selected_cuda

            # edge / mobile 档位强制 llama.cpp（即使有 CUDA 也会被层拆分逻辑处理）
            if tier in ("edge", "mobile") and not profile_has_cuda:
                logger.info(f"引擎: llama.cpp (设备档位={tier}, 无 CUDA)")
                return "llama_cpp"
            # ultrabook 档位 + 无 CUDA → llama.cpp
            if tier == "ultrabook" and not profile_has_cuda:
                logger.info(f"引擎: llama.cpp (ultrabook, 集显)")
                return "llama_cpp"

            # 更新 has_cuda 为综合检测结果
            has_cuda = profile_has_cuda

        # 最终判定
        if has_cuda:
            logger.info("引擎: PyTorch + bitsandbytes (CUDA 可用)")
            return "pytorch"
        else:
            logger.info("引擎: llama.cpp + GGUF (CPU/集显)")
            return "llama_cpp"

    # ================================================================
    # 量化配置工厂（PyTorch 专用）
    # ================================================================

    @staticmethod
    def _get_bnb_config(quant_type: str) -> Optional[BitsAndBytesConfig]:
        """
        获取 bitsandbytes 量化配置。

        Args:
            quant_type: "fp16" | "int8" | "int4"

        Returns:
            BitsAndBytesConfig 或 None（fp16 模式）
        """
        if quant_type == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant_type == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=10.0,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )
        else:
            return None  # fp16 — 不使用量化

    # ================================================================
    # 模型加载（双引擎入口）
    # ================================================================

    def load_model(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
        model_id: str = None,
        engine: str = None,
    ) -> None:
        """
        加载模型，自适应选择推理引擎。

        引擎选择:
          - PyTorch: 加载 Safetensors 格式 → 量化 → 融合
          - llama.cpp: 加载 GGUF 格式 → CPU 多线程推理

        Args:
            model_path: 本地模型路径，默认使用 config.MODEL_PATH
            quant_type: PyTorch 量化精度 "fp16" | "int8" | "int4"
            profile: 设备画像 dict
            model_id: 模型唯一标识（P3多模型支持）。
                      若提供且 model_path 未指定，从 model_config 查找路径。
        """
        # P3: 多模型支持 — 根据 model_id 解析路径
        resolved_path = model_path
        resolved_id = model_id or mc.DEFAULT_MODEL_ID

        if not resolved_path and model_id:
            cfg = mc.get_model_config(model_id)
            if cfg is None:
                raise ValueError(f"模型 '{model_id}' 未在注册表中找到")
            # 优先使用 safetensors 路径，其次 GGUF
            if cfg.model_path and os.path.isdir(cfg.model_path):
                resolved_path = cfg.model_path
            elif cfg.gguf_path and os.path.isfile(cfg.gguf_path):
                resolved_path = cfg.gguf_path
            else:
                raise FileNotFoundError(
                    f"模型 '{model_id}' 的路径不存在。\n"
                    f"  safetensors: {cfg.model_path or '(未配置)'}\n"
                    f"  GGUF: {cfg.gguf_path or '(未配置)'}"
                )

        # 确定引擎（尊重 model_type 约束）
        # P3修复: 允许调用者通过 engine 参数强制选择引擎
        if engine and engine != "auto":
            resolved_engine = engine
        else:
            resolved_engine = self.select_engine(profile)
        # model_type 强制约束：GGUF-only 模型必须用 llama.cpp；
        # Safetensors-only 模型在 CPU 上仍需走 PyTorch（或报错）
        if resolved_id != mc.DEFAULT_MODEL_ID:
            cfg = mc.get_model_config(resolved_id)
            if cfg:
                if cfg.model_type == "gguf" and resolved_engine == "pytorch":
                    logger.warning(
                        f"模型 '{resolved_id}' 仅有 GGUF 格式，"
                        f"引擎从 pytorch 切换为 llama_cpp"
                    )
                    resolved_engine = "llama_cpp"
                elif cfg.model_type == "safetensors" and resolved_engine == "llama_cpp":
                    logger.warning(
                        f"模型 '{resolved_id}' 仅有 Safetensors 格式，"
                        f"引擎保持 pytorch（CPU 推理）"
                    )
                    resolved_engine = "pytorch"

        self._engine_type = resolved_engine

        if self._engine_type == "llama_cpp":
            self._load_llama_cpp(resolved_path, profile)
        else:
            self._load_pytorch(resolved_path, quant_type, profile)

        # 记录活跃模型 ID
        self._active_model_id = resolved_id
        self._previous_engine_type = self._engine_type
        self._previous_quant_type = self.quant_type

    def unload_model(self) -> None:
        """
        卸载当前加载的模型，释放 GPU 显存和系统内存。

        同时清理 PyTorch 和 llama.cpp 引擎状态。
        调用后 is_loaded 返回 False，可安全加载新模型。
        """
        logger.info(f"卸载模型: {self._active_model_id} (引擎={self._engine_type})")

        # --- PyTorch 引擎清理 ---
        if self.model is not None:
            self.model = None

        if self.tokenizer is not None:
            self.tokenizer = None

        self.quant_type = None
        self.layer_range = None
        self._layer_has_embedding = True
        self._layer_has_lm_head = True
        self._model_layers = 0
        self._total_model_layers = 0

        # --- llama.cpp 引擎清理 ---
        if self._llama_engine is not None:
            try:
                if hasattr(self._llama_engine, 'close'):
                    self._llama_engine.close()
                elif hasattr(self._llama_engine, 'unload'):
                    self._llama_engine.unload()
            except Exception:
                pass
            self._llama_engine = None

        # --- GPU 显存回收 ---
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        # 重置引擎类型
        self._engine_type = ""
        self._active_model_id = ""  # P3修复: 卸载后清空活跃模型ID
        logger.info("模型已卸载，显存已释放")

    def switch_model(
        self,
        model_id: str,
        quant_type: str = None,
        profile: dict = None,
        engine: str = None,
    ) -> dict:
        """
        切换到另一个模型（P3 多模型支持）。

        流程:
          1. 保存当前模型信息（用于失败时的 best-effort rollback）
          2. 调用 unload_model() 释放显存
          3. 查找新模型配置
          4. 调用 load_model() 加载新模型
          5. 失败时尝试回滚到上一个模型

        Args:
            model_id: 目标模型唯一标识
            quant_type: 量化精度（默认使用当前精度或 QUANT_TYPE）
            profile: 设备画像
            engine: 推理引擎 "pytorch" | "llama_cpp" | "auto" (None=auto)

        Returns:
            {"success": bool, "model_id": str, "model_name": str, "error": str | None}
        """
        with self._lock:
            # 保存回滚信息
            rollback_model_id = self._active_model_id
            rollback_model_name = self._active_model_id  # 将在下面尝试获取可读名称
            rollback_engine = self._previous_engine_type or self._engine_type
            rollback_quant = self._previous_quant_type or quant_type or QUANT_TYPE
            had_model = self.is_loaded

            if had_model:
                # 尝试获取回滚模型的可读名称
                try:
                    rollback_cfg = mc.get_model_config(rollback_model_id)
                    if rollback_cfg:
                        rollback_model_name = rollback_cfg.name
                except Exception:
                    pass

            logger.info(
                f"切换模型: {rollback_model_id} -> {model_id} "
                f"(quant={quant_type}, engine={engine or 'auto'}, "
                f"profile_tier={profile.get('tier', '?') if profile else '?'})"
            )

            # 步骤 1: 卸载当前模型
            if had_model:
                try:
                    self.unload_model()
                except Exception as e:
                    logger.warning(f"卸载当前模型时出现异常（继续切换）: {e}")

            # 步骤 2: 加载新模型
            try:
                cfg = mc.get_model_config(model_id)
                if cfg is None:
                    return {
                        "success": False,
                        "model_id": model_id,
                        "model_name": model_id,
                        "error": f"模型 '{model_id}' 未在注册表中找到。请先注册或下载模型文件。",
                    }
                self.load_model(model_id=model_id, quant_type=quant_type,
                                profile=profile, engine=engine)
                return {
                    "success": True,
                    "model_id": self._active_model_id,
                    "model_name": cfg.name,
                    "error": None,
                }
            except Exception as e:
                logger.error(f"加载模型 '{model_id}' 失败: {e}")

                # 步骤 3: best-effort 回滚（含同模型重载失败的情况）
                if had_model and rollback_model_id:
                    logger.info(f"尝试回滚到上一个模型: {rollback_model_id}")
                    try:
                        self.unload_model()
                        self.load_model(
                            model_id=rollback_model_id,
                            quant_type=rollback_quant,
                            profile=profile,
                            engine=rollback_engine if rollback_engine else None,
                        )
                        return {
                            "success": False,
                            "model_id": rollback_model_id,
                            "model_name": rollback_model_name,
                            "error": f"模型 '{model_id}' 加载失败: {e}。已回滚到 '{rollback_model_name}'。",
                        }
                    except Exception as rollback_err:
                        logger.error(f"回滚也失败: {rollback_err}")
                        return {
                            "success": False,
                            "model_id": None,
                            "model_name": "",
                            "error": f"模型 '{model_id}' 加载失败: {e}。回滚也失败: {rollback_err}。",
                        }

                return {
                    "success": False,
                    "model_id": None,
                    "model_name": "",
                    "error": f"模型 '{model_id}' 加载失败: {e}",
                }

    @property
    def active_model_id(self) -> str:
        """当前活跃的模型 ID。"""
        return self._active_model_id

    def load_layer_range(
        self,
        start_layer: int = 0,
        end_layer: int = 24,
        has_embedding: bool = True,
        has_lm_head: bool = True,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
    ) -> None:
        """
        加载模型的指定层范围（分布式流水线节点专用）。

        加载完整 PyTorch 模型后，仅保留 [start_layer, end_layer) 的
        Transformer 层，根据需要保留/丢弃 Embedding 和 LM Head，
        然后释放不需要的层所占用的显存。

        Args:
            start_layer: 起始层编号（0-based，含）
            end_layer: 结束层编号（0-based，不含）
            has_embedding: 是否保留 Token Embedding（首节点为 True）
            has_lm_head: 是否保留 LM Head 输出层（末节点为 True）
            model_path: 模型路径，默认使用 config.MODEL_PATH
            quant_type: 量化精度，默认使用 config.QUANT_TYPE
            profile: 设备画像 dict

        示例:
            # 主节点：Layer 0-7 + Embedding
            mgr.load_layer_range(0, 8, has_embedding=True, has_lm_head=False)

            # 中间节点：Layer 8-15
            mgr.load_layer_range(8, 16, has_embedding=False, has_lm_head=False)

            # 末节点：Layer 16-24 + LM Head
            mgr.load_layer_range(16, 24, has_embedding=False, has_lm_head=True)
        """
        import torch.nn as nn

        # ---- 参数校验 ----
        from config import TOTAL_MODEL_LAYERS
        if start_layer < 0 or end_layer > TOTAL_MODEL_LAYERS or start_layer >= end_layer:
            raise ValueError(
                f"无效的层范围: [{start_layer}, {end_layer})，"
                f"有效范围: [0, {TOTAL_MODEL_LAYERS})"
            )

        layers_count = end_layer - start_layer
        logger.info(
            f"🎯 层范围加载: Layer {start_layer}-{end_layer} ({layers_count}层), "
            f"embed={has_embedding}, lm_head={has_lm_head}"
        )

        # ---- 先加载完整模型 ----
        self._load_pytorch(model_path=model_path, quant_type=quant_type, profile=profile)
        self._engine_type = "pytorch"

        if self.model is None:
            raise RuntimeError("模型加载失败，无法进行层范围裁剪")

        # ---- 裁剪 Transformer 层 ----
        # Qwen2ForCausalLM 结构:
        #   model.model.embed_tokens   → Embedding
        #   model.model.layers         → nn.ModuleList([24× Qwen2DecoderLayer])
        #   model.model.norm           → RMSNorm
        #   model.lm_head              → Linear(2048, 151936)
        transformer = self.model.model  # Qwen2Model

        # 1. 保留指定范围的 Transformer 层
        all_layers = list(transformer.layers)
        kept = all_layers[start_layer:end_layer]
        transformer.layers = nn.ModuleList(kept)

        # 释放被裁剪层的引用，帮助 GC 回收显存
        for layer in all_layers[:start_layer]:
            del layer
        for layer in all_layers[end_layer:]:
            del layer
        del all_layers

        # 2. 根据需要保留 Embedding
        if not has_embedding:
            if hasattr(transformer, 'embed_tokens'):
                del transformer.embed_tokens
                transformer.embed_tokens = None

        # 3. 根据需要保留 LM Head
        if not has_lm_head:
            if hasattr(self.model, 'lm_head'):
                del self.model.lm_head
                self.model.lm_head = None

        # 4. 清理显存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ---- 记录层范围 ----
        self.layer_range = (start_layer, end_layer)
        self._layer_has_embedding = bool(has_embedding)
        self._layer_has_lm_head = bool(has_lm_head)
        self._model_layers = layers_count
        # _total_model_layers 在 _load_pytorch 中已设为完整模型总层数，此处不覆盖

        # ---- 显存统计 ----
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(
                f"✅ 层范围加载完成: 显存 {mem:.2f} GB, "
                f"层 {start_layer}-{end_layer}, "
                f"embed={has_embedding}, lm_head={has_lm_head}"
            )
        else:
            logger.info(
                f"✅ 层范围加载完成: Layer {start_layer}-{end_layer}, "
                f"embed={has_embedding}, lm_head={has_lm_head}"
            )

    def ensure_layer_range(
        self,
        start_layer: int,
        end_layer: int,
        has_embedding: bool,
        has_lm_head: bool,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
    ) -> None:
        """确保当前 PyTorch 模型已裁剪为指定层范围，避免重复重载。"""
        desired_range = (start_layer, end_layer)
        if (
            self.is_loaded
            and self._engine_type == "pytorch"
            and self.layer_range == desired_range
            and self._layer_has_embedding == bool(has_embedding)
            and self._layer_has_lm_head == bool(has_lm_head)
        ):
            return
        self.load_layer_range(
            start_layer,
            end_layer,
            has_embedding=has_embedding,
            has_lm_head=has_lm_head,
            model_path=model_path,
            quant_type=quant_type,
            profile=profile,
        )

    def ensure_full_model(self, quant_type: str = None,
                          profile: dict = None, engine: str = None) -> None:
        """确保当前模型为完整模型；流水线裁剪后回退本地推理前调用。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        if self._engine_type == "llama_cpp":
            return
        if (
            self._engine_type == "pytorch"
            and self.layer_range is None
            and self._layer_has_embedding
            and self._layer_has_lm_head
        ):
            return

        model_id = self._active_model_id or mc.DEFAULT_MODEL_ID
        q = quant_type or self.quant_type or QUANT_TYPE
        logger.info(
            f"🔄 当前为流水线裁剪模型 layer_range={self.layer_range}，"
            f"重新加载完整模型用于本地推理"
        )
        self.load_model(
            model_id=model_id,
            quant_type=q,
            profile=profile,
            engine=engine or "pytorch",
        )

    def _load_llama_cpp(self, model_path: str = None, profile: dict = None) -> None:
        """
        加载 llama.cpp + GGUF 模型。

        GGUF 文件查找顺序:
          1. 显式指定的 model_path（如果是 .gguf 文件）
          2. config.GGUF_MODEL_PATH
          3. 自动搜索 models/ 目录下的 .gguf 文件
        """
        from llama_engine import LlamaCppEngine, get_gguf_model_path

        # 确定 GGUF 文件路径
        gguf_path = None
        if model_path and model_path.endswith(".gguf"):
            gguf_path = model_path
        elif os.path.isfile(GGUF_MODEL_PATH):
            gguf_path = GGUF_MODEL_PATH
        else:
            gguf_path = get_gguf_model_path()

        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(
                f"GGUF 模型文件未找到。\n"
                f"  配置路径: {GGUF_MODEL_PATH}\n"
                f"  请下载 GGUF 格式的 Qwen-1.8B-Chat 模型:\n"
                f"  - 推荐: Q4_K_M (~1.16 GB) — 速度/质量最佳平衡\n"
                f"  - 下载: https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf\n"
                f"  - 或使用模型下载引导: python src/model_downloader.py"
            )

        # 自适应上下文窗口大小
        n_ctx = 4096
        if profile:
            tier = profile.get("tier", "laptop")
            if tier == "edge":
                n_ctx = 1024
            elif tier == "mobile":
                n_ctx = 512
            elif tier == "ultrabook":
                n_ctx = 2048

        self._llama_engine = LlamaCppEngine()
        self._llama_engine.load_model(
            model_path=gguf_path,
            n_ctx=n_ctx,
        )
        self._model_path = gguf_path

        logger.info("✅ llama.cpp 引擎就绪 (CPU/集显 优化)")

    def _load_pytorch(
        self,
        model_path: str = None,
        quant_type: str = None,
        profile: dict = None,
    ) -> None:
        """
        加载 PyTorch + Transformers 模型（CUDA 路径）。

        核心逻辑:
        - fp16: 直接以 torch.float16 加载，显存 ~3.5 GB
        - int8: bitsandbytes 8-bit 量化加载，显存 ~2.3 GB
        - int4: bitsandbytes 4-bit NF4 双重量化加载，显存 ~1.8 GB
        """
        path = model_path or MODEL_PATH
        self._model_path = path
        self.quant_type = quant_type or QUANT_TYPE

        # ---- 自适应：根据设备画像调整加载策略 ----
        force_cpu = False
        if profile:
            tier = profile.get("tier", "laptop")
            gpu_info = profile.get("gpu", {})
            has_cuda = gpu_info.get("cuda_available", False)
            vram_gb = gpu_info.get("vram_total_gb", 0)

            if tier in ("ultrabook", "edge", "mobile") and not has_cuda:
                force_cpu = True
                logger.info(f"设备档位={tier} 无 CUDA，使用 CPU-only 模式")

            if has_cuda and vram_gb > 0:
                min_vram = {"fp16": 3.5, "int8": 2.3, "int4": 1.8}.get(self.quant_type, 3.5)
                if vram_gb < min_vram:
                    logger.warning(
                        f"⚠️ 显存 {vram_gb:.1f} GB 不足以加载 {self.quant_type} 模型"
                        f"（预计需要 {min_vram:.1f} GB），建议切换量化精度"
                    )

        use_cuda = torch.cuda.is_available() and not force_cpu

        logger.info(f"加载模型: {path}")
        logger.info(f"量化精度: {self.quant_type}  |  算子融合: {USE_COMPILE}")
        logger.info(f"推理设备: {'CUDA' if use_cuda else 'CPU'}")

        # ---- CPU-only 路径 ----
        if not use_cuda:
            if self.quant_type in ("int4", "int8"):
                logger.warning(
                    f"⚠️ bitsandbytes {self.quant_type} 量化不支持 CPU，回退到 FP16 CPU 推理"
                )
                logger.warning(
                    f"💡 建议切换引擎为 llama.cpp (设置 INFERENCE_ENGINE='llama_cpp') "
                    f"以获得更好的 CPU 推理性能（3-5x 加速）"
                )
                self.quant_type = "fp16"

            cpu_cores = profile.get("cpu", {}).get("physical_cores", 4) if profile else 4
            omp_threads = max(2, cpu_cores // 2)
            os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(omp_threads))
            logger.info(f"CPU 线程数: OMP={omp_threads}, MKL={omp_threads}")

            load_kwargs: Dict[str, Any] = dict(
                device_map={"": "cpu"},
                trust_remote_code=TRUST_REMOTE_CODE,
                torch_dtype=torch.float16,
            )
        else:
            # ---- CUDA 路径 ----
            bnb_config = self._get_bnb_config(self.quant_type)

            load_kwargs: Dict[str, Any] = dict(
                device_map="auto",
                trust_remote_code=TRUST_REMOTE_CODE,
            )

            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16

        t0 = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=TRUST_REMOTE_CODE)
        self.layer_range = None
        self._layer_has_embedding = True
        self._layer_has_lm_head = True

        load_time = time.time() - t0

        # 记录模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        param_dtype = next(self.model.parameters()).dtype
        layers = self._count_transformer_layers()
        self._model_layers = layers
        self._total_model_layers = layers

        logger.info(f"模型加载完成 ({load_time:.1f}s)")
        logger.info(f"  参数量: {total_params/1e9:.2f}B  |  类型: {param_dtype}")
        logger.info(f"  设备: {self.model.device}  |  Transformer层数: {self._model_layers}")

        # 显存统计
        if use_cuda and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"  GPU 显存占用: {mem:.2f} GB")
        elif not use_cuda:
            mem = psutil.virtual_memory().used / (1024 ** 3)
            logger.info(f"  CPU 内存占用 (进程): {mem:.1f} GB")

        # 算子融合 — 仅在 FP16 + CUDA 下生效
        if USE_COMPILE:
            if not use_cuda:
                logger.warning("⚠️ torch.compile 需要 CUDA，CPU 模式下已自动跳过")
            elif self.quant_type != "fp16":
                logger.warning(
                    f"⚠️ torch.compile 与 {self.quant_type} 量化不兼容（实测慢 13%），已自动跳过。"
                    f"如需融合，请设置 QUANT_TYPE='fp16'。"
                )
            else:
                self._apply_compile()

    def _count_transformer_layers(self) -> int:
        """统计模型的 Transformer 层数"""
        if self.model is None:
            return 0
        # Qwen2 使用 model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        # GPT-2 / 旧 Llama 使用 model.transformer.h
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        return 0

    def _apply_compile(self) -> None:
        """开启 torch.compile 自动算子融合。"""
        logger.info("开启 torch.compile 算子融合 (mode='reduce-overhead')...")
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("  ✅ torch.compile 已启用 (预计 +8% 推理速度)")
        except Exception as e:
            logger.warning(f"  ❌ torch.compile 启用失败: {e}，回退到普通模式")

    # ================================================================
    # 对话补全（统一接口，内部委托给对应引擎）
    # ================================================================

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        对话补全 — 自动路由到当前活跃引擎。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度 (0-2)
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            {"content": "模型回复文本", "usage": {...}, "tokens_per_second": float}
        """
        if self._engine_type == "llama_cpp":
            if self._llama_engine is None:
                raise RuntimeError("llama.cpp 引擎未加载，请先调用 load_model()")
            return self._llama_engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

        elif self._engine_type == "pytorch":
            # PyTorch 路径: 使用 tokenizer + model.generate()
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("PyTorch 模型未加载，请先调用 load_model()")

            return self._pytorch_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

        else:
            raise RuntimeError(f"未知引擎类型: {self._engine_type}")

    def _pytorch_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """PyTorch 路径的对话补全实现。"""
        t0 = time.time()

        # 使用 tokenizer 的 chat template 构建输入
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Qwen tokenizer 的 chat_template 可能不同，手动构建
            input_text = self._build_qwen_prompt(messages)

        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.get_device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # 解码生成部分（去除输入部分）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        elapsed = time.time() - t0
        completion_tokens = len(generated_ids)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"推理完成 (PyTorch): {completion_tokens} tokens / {elapsed:.1f}s "
            f"= {tok_per_sec:.1f} tok/s"
        )

        return {
            "content": content,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "total_tokens": input_len + completion_tokens,
            },
            "model": MODEL_NAME,
            "finish_reason": "stop",
            "tokens_per_second": round(tok_per_sec, 1),
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ):
        """
        流式对话补全。

        Yields:
            str: 增量文本 chunk
        """
        if self._engine_type == "llama_cpp":
            if self._llama_engine is None:
                raise RuntimeError("llama.cpp 引擎未加载")
            yield from self._llama_engine.chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
        elif self._engine_type == "pytorch":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("PyTorch 模型未加载，请先调用 load_model()")

            try:
                from transformers import TextIteratorStreamer
            except ImportError:
                # 降级：transformers 版本过旧，回退到非流式
                logger.warning("TextIteratorStreamer 不可用，回退到非流式")
                result = self._pytorch_chat(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stop=stop, **kwargs,
                )
                yield result["content"]
                return

            # 构建输入
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                input_text = self._build_qwen_prompt(messages)

            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.get_device()) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            import threading
            t0 = time.time()
            thread = threading.Thread(
                target=self.model.generate, kwargs=generation_kwargs,
            )
            thread.start()

            chunk_count = 0
            for text in streamer:
                if text:
                    chunk_count += 1
                    yield text

            thread.join()
            elapsed = time.time() - t0
            logger.info(
                f"流式推理完成 (PyTorch): {chunk_count} chunks / {elapsed:.1f}s"
            )
        else:
            raise RuntimeError(f"未知引擎类型: {self._engine_type}")

    def _build_qwen_prompt(self, messages: List[Dict[str, str]]) -> str:
        """手动构建 Qwen ChatML 格式 prompt（fallback）。"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ================================================================
    # 模型层级拆分（PyTorch 专用）
    # ================================================================

    def split_model(self, layer_config: Tuple[int, int]) -> nn.Module:
        """
        [已废弃] 模型层级拆分 — 请使用 load_layer_range() 代替。

        此方法在加载完整模型后进行运行时拆分，已被 load_layer_range()
        的"加载即裁剪"方案取代（零额外显存峰值）。

        保留此方法仅为向后兼容，新代码请直接使用:
            mgr.load_layer_range(start, end, has_embedding=..., has_lm_head=...)
            result = mgr.forward_layers(input_ids=...)

        Args:
            layer_config: (start_layer, end_layer) 起止层编号，左闭右开

        Returns:
            None（始终抛 NotImplementedError）
        """
        raise NotImplementedError(
            "split_model() 已废弃，请使用 load_layer_range() 代替。\n"
            "示例: mgr.load_layer_range(start, end, has_embedding=True, has_lm_head=False)"
        )

    # ================================================================
    # 前向推理（PyTorch 分布式专用）
    # ================================================================

    def forward_layers(
        self,
        input_ids: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: tuple = None,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        执行本节点层范围的单步前向传播（分布式流水线节点专用）。

        三种节点角色:
         - 首节点 (has_embedding): input_ids → embed_tokens → layers → hidden_states
         - 中间节点:              hidden_states → layers → hidden_states
         - 末节点 (has_lm_head):  hidden_states → layers → norm → lm_head → logits

        严格遵循 Qwen2Model.forward() 的调用链，复用其 RoPE、causal_mask、
        和 position_embeddings 计算逻辑，确保分布式与单机推理输出一致。

        **KV Cache 支持 (Phase 3 — 增量解码)**:
         - Prefill (past_key_values=None): 处理完整输入序列，构建 KV cache
         - Decode (past_key_values 存在): 仅处理新 token，基于缓存的 KV 增量计算
         - KV Cache 形状: 每层 (batch, num_heads, total_seq_len, head_dim)
         - past_key_values 索引 0..(N-1) 对应本节点的 N 个本地层

        Args:
            input_ids: 输入 token IDs (batch, seq_len)，仅首节点使用
            hidden_states: 中间隐藏状态 (batch, seq_len, hidden_dim)，中间/末节点使用
            attention_mask: 2D 注意力掩码 (batch, seq_len)，1=有效，0=填充
            position_ids: 位置 ID (batch, seq_len)，None 则自动生成
            past_key_values: 已缓存的 KV cache，tuple of (key, value) per local layer
            use_cache: 是否返回新的 KV cache（decode 阶段应为 True）

        Returns:
            {"hidden_states": Tensor}       — 非末节点，形状 (batch, seq_len, hidden_dim)
            {"logits": Tensor}             — 末节点，形状 (batch, seq_len, vocab_size)
            当 use_cache=True 时附加:
            {"past_key_values": tuple}     — 更新后的 KV cache（每层一个 (k,v) 元组）
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model() 或 load_layer_range()")

        if self._engine_type != "pytorch":
            raise RuntimeError("forward_layers 仅支持 PyTorch 引擎")

        # ---- 输入校验 ----
        if input_ids is None and hidden_states is None:
            raise ValueError("必须提供 input_ids 或 hidden_states 之一")
        if input_ids is not None and hidden_states is not None:
            raise ValueError("input_ids 和 hidden_states 不能同时提供")

        device = self.get_device()
        transformer = self.model.model  # Qwen2Model
        dtype = next(self.model.parameters()).dtype

        # 检测节点角色
        has_embed = (
            hasattr(transformer, 'embed_tokens')
            and transformer.embed_tokens is not None
        )
        has_lm_head = (
            hasattr(self.model, 'lm_head')
            and self.model.lm_head is not None
        )

        batch_size: int
        seq_len: int

        # ============================================================
        # 整个前向传播在 torch.no_grad() 下执行，避免梯度追踪开销。
        # ============================================================
        with torch.no_grad():
            # ============================================================
            # Step 1: Embedding（首节点）
            # ============================================================
            if input_ids is not None:
                if not has_embed:
                    raise RuntimeError(
                        "当前节点不含 Embedding 层（加载时未设置 has_embedding=True），"
                        "请传入 hidden_states"
                    )
                input_ids = input_ids.to(device)
                batch_size, seq_len = input_ids.shape
                hidden_states = transformer.embed_tokens(input_ids).to(dtype=dtype)
            else:
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                if hidden_states.device != device:
                    hidden_states = hidden_states.to(device)
                if hidden_states.dtype != dtype:
                    hidden_states = hidden_states.to(dtype)

            # ============================================================
            # Step 2: KV Cache 初始化 — DynamicCache with local indices
            # ============================================================
            # Qwen2SdpaAttention / FlashAttention2 使用 DynamicCache，
            # 其内部以 self_attn.layer_idx 为索引存储 key/value。
            #
            # ★ 分布式节点仅加载部分层，若使用原始 global_idx（如 8-15），
            #    DynamicCache 会产生稀疏空洞（需 None 填充），导致
            #    get_seq_length() 对空洞条目抛 TypeError。
            #
            #    解决方案: 临时将各层的 self_attn.layer_idx 改为本地索引
            #    (0, 1, 2, ...)，使 DynamicCache 按连续本地索引存储。
            #    forward 后恢复原始 global_idx，确保后续调用不受影响。
            from transformers.cache_utils import DynamicCache

            # 保存并覆盖层索引为本地连续编号
            # ★ 先保存原始索引，再在 try 块内补丁，确保 finally 无论何路径都恢复
            saved_layer_indices: list = []
            for local_idx, layer in enumerate(transformer.layers):
                saved_layer_indices.append(layer.self_attn.layer_idx)

            try:
                # ---- 补丁 layer_idx 为本地索引（必须在 try 内，确保异常时恢复） ----
                for local_idx, layer in enumerate(transformer.layers):
                    layer.self_attn.layer_idx = local_idx

                if use_cache:
                    if past_key_values is not None:
                        # Decode: tuple of (k,v) → DynamicCache（本地索引 0..N-1）
                        # Phase 4.3: 验证缓存层数与本地层数一致
                        n_local = len(transformer.layers)
                        if len(past_key_values) != n_local:
                            logger.warning(
                                f"KV cache 层数不匹配: 缓存 {len(past_key_values)} 层, "
                                f"本地 {n_local} 层 — 丢弃不匹配的缓存"
                            )
                            cache = DynamicCache()
                        else:
                            cache = DynamicCache()
                            for layer_idx, (k, v) in enumerate(past_key_values):
                                cache.update(k, v, layer_idx)
                    else:
                        # Prefill: 创建空 DynamicCache
                        cache = DynamicCache()
                else:
                    cache = None

                # ---- 从 cache 获取 past_seen_tokens ----
                # transformers≥5.x: DynamicCache 使用 layers 属性替代 key_cache/value_cache
                if cache is not None and cache.get_seq_length() > 0:
                    try:
                        past_seen_tokens = cache.get_seq_length()
                    except (AttributeError, TypeError, IndexError) as e:
                        # Phase 4.4: DynamicCache 内部 API 变更 → 降级为 0
                        # (不捕获 MemoryError/SystemExit 等真实异常)
                        logger.debug(f"cache.get_seq_length() 失败: {e}")
                        past_seen_tokens = 0
                else:
                    past_seen_tokens = 0

                # ============================================================
                # Step 3: position_ids / cache_position
                # ============================================================
                if position_ids is None:
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + seq_len,
                        device=device
                    )
                    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
                else:
                    position_ids = position_ids.to(device)
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + seq_len,
                        device=device
                    )

                # ============================================================
                # Step 4: causal_mask — 复用 Qwen2 内置逻辑
                # ============================================================
                # transformers≥5.x: _update_causal_mask 已移除，改用 create_causal_mask
                # transformers 4.x: 无此函数，回退到手动构建因果掩码
                # 该函数根据 attention 实现自动选择:
                #   flash_attention_2 → None（flash 内核自行处理因果掩码）
                #   sdpa + 纯因果 → None（SDPA is_causal 路径）
                #   eager / 含填充 → 4D (batch,1,seq,seq) 因果掩码
                try:
                    from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
                    causal_mask = create_causal_mask(
                        config=transformer.config,
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                        past_key_values=cache,
                        position_ids=position_ids,
                    )
                except (ImportError, AttributeError):
                    # transformers 4.x 回退：手动构建 4D 因果掩码
                    seq_len = hidden_states.shape[1]
                    causal_mask = torch.full(
                        (seq_len, seq_len),
                        float('-inf'),
                        device=device,
                    )
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                    causal_mask = causal_mask[None, None, :, :].expand(
                        hidden_states.shape[0], 1, seq_len, seq_len
                    )
                    if attention_mask is not None:
                        attn_mask = attention_mask.to(device)
                        # attn_mask shape: (batch, seq_len) → (batch, 1, 1, seq_len)
                        attn_mask = attn_mask[:, None, None, :]
                        causal_mask = causal_mask.masked_fill(
                            attn_mask == 0, float('-inf')
                        )

                # ============================================================
                # Step 5: position_embeddings — RoPE 旋转位置编码
                # ============================================================
                # Qwen2Model.rotary_emb 会将 position_ids 转换为 cos/sin 元组，
                # 各 attention 层内部通过 apply_rotary_pos_emb 应用到 Q/K 上。
                # 此步在层循环外仅计算一次，所有层共享同一份 position_embeddings。
                position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

                # ============================================================
                # Step 6: Transformer 层前向传播（支持 KV cache）
                # ============================================================
                # DynamicCache 由 SDPA/FlashAttention 在 forward 时原地更新，
                # 每层的 key/value 按 layer_idx 写入 cache.layers 中。
                for i, layer in enumerate(transformer.layers):
                    layer_output = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        past_key_values=cache,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    # transformers≥5.x: DecoderLayer 直接返回 tensor
                    # transformers 4.x: 返回 (hidden_states, present_key_value) 元组
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output
                    del layer_output

                # ============================================================
                # Step 7: 最终 Norm + LM Head（末节点）
                # ============================================================
                result: Dict[str, torch.Tensor] = {}

                if has_lm_head:
                    hidden_states = transformer.norm(hidden_states)
                    logits = self.model.lm_head(hidden_states)
                    result["logits"] = logits
                else:
                    result["hidden_states"] = hidden_states

                # ---- KV Cache: 转为 tuple 存储 ----
                # transformers≥5.x: DynamicCache.layers[i].keys/.values 替代 key_cache/value_cache
                if use_cache and cache is not None and len(cache.layers) > 0:
                    result["past_key_values"] = tuple(
                        (cache.layers[i].keys, cache.layers[i].values)
                        for i in range(len(cache.layers))
                    )

                return result
            finally:
                # ★ 无论成功/异常，必须恢复原始 layer_idx
                #    （finally 覆盖了 layer_idx 补丁 + create_causal_mask +
                #      rotary_emb + 层循环 + LM Head 全路径）
                for layer, orig_idx in zip(transformer.layers, saved_layer_indices):
                    layer.self_attn.layer_idx = orig_idx

    # ================================================================
    # 工具方法
    # ================================================================

    @property
    def engine_type(self) -> str:
        """当前使用的推理引擎类型。"""
        return self._engine_type

    @property
    def is_llama_cpp(self) -> bool:
        """是否使用 llama.cpp 引擎。"""
        return self._engine_type == "llama_cpp"

    @property
    def is_pytorch(self) -> bool:
        """是否使用 PyTorch 引擎。"""
        return self._engine_type == "pytorch"

    def get_device(self) -> torch.device:
        """获取当前模型所在设备（PyTorch 引擎）"""
        if self.model is not None:
            return self.model.device
        return torch.device(DEVICE)

    def get_model_info(self) -> dict:
        """获取模型基本信息，用于调试与日志（双引擎兼容）"""
        if self._engine_type == "llama_cpp" and self._llama_engine:
            info = self._llama_engine.get_model_info()
            info["model_id"] = self._active_model_id
            return info

        info = {
            "model_id": self._active_model_id,
            "engine": "pytorch",
            "model_name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "quant_type": self.quant_type,
            "compile": USE_COMPILE,
            "layer_range": self.layer_range,
            "total_layers": self._total_model_layers or self._model_layers,
            "loaded_layers": self._model_layers,
            "device": str(self.get_device()),
        }
        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        return info

    def get_memory_usage(self) -> dict:
        """获取当前显存/内存占用，用于性能监控（双引擎兼容）"""
        if self._engine_type == "llama_cpp" and self._llama_engine:
            return self._llama_engine.get_memory_usage()

        result = {}
        if torch.cuda.is_available():
            result["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            result["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
            result["gpu_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
        return result

    def reset_kv_cache(self) -> None:
        """清空 KV 缓存（双引擎兼容）"""
        if self._engine_type == "llama_cpp" and self._llama_engine:
            self._llama_engine.reset_kv_cache()
        # PyTorch: KV cache 由 transformers generate() 内部管理，每次调用自动重置

    # ================================================================
    # KV Cache 工具方法（Phase 3 — 增量解码）
    # ================================================================

    @staticmethod
    def _tuple_to_dynamic_cache(past_key_values: tuple, start_layer: int = 0):
        """
        将旧格式 tuple of (k,v) 转换为 DynamicCache。

        NOTE: 此方法当前未被任何代码路径调用（死代码）。
        如果将来重新启用，请注意它使用全局层索引（start_layer + i），
        与 forward_layers 热路径中的本地索引补丁（0..N-1）不兼容。
        混用两者会导致 cache.layers 中出现 None 槽位，进而使
        get_seq_length() 崩溃。

        Qwen2 SDPA/Flash Attention 使用 DynamicCache（transformers >= 4.44），
        每层的 self_attn.layer_idx 为全局层编号，故 DynamicCache 内部
        以全局 layer_idx 为索引存储 key/value。

        Args:
            past_key_values: tuple of (key, value) per layer，索引 0..N-1
            start_layer: 第一个元素的全局层编号（分布式节点专用）

        Returns:
            DynamicCache 对象
        """
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i, kv in enumerate(past_key_values):
            if kv is None:
                continue
            k, v = kv
            global_idx = start_layer + i
            # transformers≥5.x: 使用 update() 按 global_idx 写入
            cache.update(k, v, global_idx)
        return cache
