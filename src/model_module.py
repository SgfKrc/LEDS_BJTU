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
        self.layer_range: Optional[Tuple[int, int]] = None
        self.sub_model: Optional[nn.Module] = None  # 拆分后的子模型
        self._model_layers: int = 0  # 模型总层数

        # llama.cpp 引擎（延迟导入 + 延迟加载）
        self._llama_engine = None   # LlamaCppEngine 实例
        self._engine_type: str = ""  # "pytorch" | "llama_cpp"

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
        """
        # 确定引擎
        self._engine_type = self.select_engine(profile)

        if self._engine_type == "llama_cpp":
            self._load_llama_cpp(model_path, profile)
        else:
            self._load_pytorch(model_path, quant_type, profile)

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

        load_time = time.time() - t0

        # 记录模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        param_dtype = next(self.model.parameters()).dtype
        self._model_layers = self._count_transformer_layers()

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
            # PyTorch 流式: 暂时回退到非流式（后续可用 TextStreamer 实现）
            result = self._pytorch_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
            yield result["content"]
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
        根据配置拆分模型层，返回当前节点负责的子模型。

        Qwen-1.8B 模型结构:
            model.transformer.wte          — Embedding（仅首节点）
            model.transformer.h[0..23]     — 24 层 Transformer
            model.lm_head                  — LM Head（仅末节点）

        注意: llama.cpp 不支持层级拆分，分布式推理需使用 PyTorch 引擎。

        Args:
            layer_config: (start_layer, end_layer) 起止层编号，左闭右开

        Returns:
            拆分后的子模型 nn.Module
        """
        if self._engine_type == "llama_cpp":
            logger.warning(
                "llama.cpp 不支持模型层级拆分。分布式推理请使用 PyTorch 引擎。"
            )
            return None

        self.layer_range = layer_config
        start, end = layer_config
        n_layers = end - start
        logger.info(f"模型拆分: 层 [{start}, {end}) 共 {n_layers} 层")

        # TODO: 实际拆分实现
        # 1. 浅拷贝原始模型结构
        # 2. 如果 start == 0: 保留 Embedding (model.transformer.wte)
        # 3. 切片 model.transformer.h[start:end]
        # 4. 如果 end == total_layers: 保留 LM Head (model.lm_head)
        # 5. 将不用的层设为 None 以释放显存

        self.sub_model = None  # TODO: 替换为拆分后的模型
        return self.sub_model

    # ================================================================
    # 前向推理（PyTorch 分布式专用）
    # ================================================================

    def model_forward(
        self,
        input_ids: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        单步前向推理，输出中间特征或最终结果。

        两种调用模式:
        1. 首节点（Prefill）: 传 input_ids，从 Embedding 开始
        2. 后续节点: 传 hidden_states，仅经过 Transformer 层

        注意: 此接口仅 PyTorch 引擎支持，llama.cpp 使用 chat() 接口。
        """
        model = self.sub_model if self.sub_model is not None else self.model

        if model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        start, end = self.layer_range or (0, self._model_layers)
        is_first = (start == 0)
        is_last = (end >= self._model_layers)

        logger.debug(f"model_forward: layers [{start},{end}), first={is_first}, last={is_last}")

        return {}  # TODO: 实现实际推理逻辑

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
            return self._llama_engine.get_model_info()

        info = {
            "engine": "pytorch",
            "model_name": MODEL_NAME,
            "model_path": MODEL_PATH,
            "quant_type": self.quant_type,
            "compile": USE_COMPILE,
            "layer_range": self.layer_range,
            "total_layers": self._model_layers,
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
