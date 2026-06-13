"""
llama.cpp 推理引擎 — CPU / 集显设备的轻量级推理后端
====================================================
职责:
1. 加载 GGUF 量化模型（Q4_K_M / Q5_K_M / Q8_0 等）
2. ChatML 格式对话补全（兼容 Qwen 家族）
3. 流式输出 + KV 缓存管理
4. 与 PyTorch ModelManager 接口对齐，支持无缝切换

依赖: llama-cpp-python (pip install llama-cpp-python)

Qwen 家族的 ChatML 格式:
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你好<|im_end|>
    <|im_start|>assistant
    你好！有什么可以帮助你的？<|im_end|>

llama-cpp-python 通过 GGUF 元数据中的 tokenizer.chat_template
自动识别 Qwen 的对话格式，通常无需手动指定 chat_format。
若自动检测失败，可手动设置 chat_format="chatml"。

设计原则:
  - CUDA 设备 → PyTorch + bitsandbytes（保留不变）
  - CPU / 集显设备 → llama.cpp + GGUF（本模块）
  - 接口与 ModelManager 对齐，上游调用者无需修改
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Dict, Any, Iterator, List

logger = logging.getLogger(__name__)

# 默认推荐量化类型 → GGUF 文件名
QUANT_FILES = {
    "Q4_K_M": "Qwen-1_8B-Chat-Q4_K_M.gguf",    # 推荐：速度/质量 最佳平衡 (~1.16 GB)
    "Q4_K_S": "Qwen-1_8B-Chat-Q4_K_S.gguf",     # 稍小 (~1.04 GB)
    "Q5_K_M": "Qwen-1_8B-Chat-Q5_K_M.gguf",     # 更高质量 (~1.31 GB)
    "Q8_0":   "Qwen-1_8B-Chat-Q8_0.gguf",        # 近无损 (~1.82 GB)
    "Q3_K_M": "Qwen-1_8B-Chat-Q3_K_M.gguf",      # 更小 (~0.94 GB)
}


class LlamaCppEngine:
    """
    llama.cpp 推理引擎 — 面向 CPU / 集显环境优化。

    使用方式:

        engine = LlamaCppEngine()
        engine.load_model("models/qwen-1_8b-chat-Q4_K_M.gguf")

        # 对话补全
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好"},
        ]
        result = engine.chat(messages, max_tokens=512, temperature=0.7)
        print(result["content"])

        # 流式输出
        for chunk in engine.chat_stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(self):
        self._model = None           # llama_cpp.Llama 实例
        self._model_path: str = ""
        self._quant_type: str = ""
        self._n_ctx: int = 4096      # 上下文窗口大小
        self._n_threads: int = 4     # CPU 线程数
        self._loaded: bool = False

    # ================================================================
    # 模型加载
    # ================================================================

    def load_model(
        self,
        model_path: str = None,
        n_ctx: int = None,
        n_threads: int = None,
        chat_format: str = None,
        **kwargs,
    ) -> None:
        """
        加载 GGUF 量化模型。

        Args:
            model_path: GGUF 文件路径
            n_ctx: 上下文窗口大小（默认 4096，边缘设备建议 2048）
            n_threads: CPU 推理线程数（默认自动检测：物理核心数）
            chat_format: 对话格式（默认自动检测，Qwen 用 "chatml"）
            **kwargs: 传递给 llama_cpp.Llama 的额外参数
        """
        from config import MAX_SEQ_LEN

        self._model_path = model_path

        # 自动检测量化类型（从文件名提取）
        for quant_name, fname in QUANT_FILES.items():
            if fname in (model_path or ""):
                self._quant_type = quant_name
                break
        if not self._quant_type:
            self._quant_type = "GGUF"

        if n_ctx is None:
            n_ctx = MAX_SEQ_LEN if MAX_SEQ_LEN > 0 else 4096
        if n_threads is None:
            n_threads = self._auto_threads()

        self._n_ctx = n_ctx
        self._n_threads = n_threads

        logger.info(f"加载 GGUF 模型: {model_path}")
        logger.info(f"  量化类型: {self._quant_type}")
        logger.info(f"  上下文窗口: {n_ctx} tokens")
        logger.info(f"  CPU 线程数: {n_threads}")

        t0 = time.time()

        try:
            from llama_cpp import Llama

            load_kwargs: Dict[str, Any] = dict(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )

            # chat_format: Qwen 模型必须用 ChatML 格式
            # llama.cpp 的自动检测可能误判为 llama-2，导致对话格式错乱
            if chat_format:
                load_kwargs["chat_format"] = chat_format
            else:
                # 默认使用 chatml（Qwen 家族标准）
                load_kwargs["chat_format"] = "chatml"

            load_kwargs.update(kwargs)

            self._model = Llama(**load_kwargs)
            self._loaded = True

            load_time = time.time() - t0
            logger.info(f"GGUF 模型加载完成 ({load_time:.1f}s)")
            logger.info(f"  引擎: llama.cpp (CPU)")

        except ImportError:
            raise ImportError(
                "llama-cpp-python 未安装。请执行:\n"
                "  pip install llama-cpp-python\n"
                "或从预编译 wheel 安装:\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )
        except Exception as e:
            logger.error(f"GGUF 模型加载失败: {e}")
            self._loaded = False
            raise

    def unload(self) -> None:
        """卸载模型，释放内存。"""
        if self._model is not None:
            del self._model
        self._model = None
        self._loaded = False
        logger.info("GGUF 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    # ================================================================
    # 对话补全（对齐 ModelManager 接口）
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
        对话补全，返回完整结果。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度 (0-2)
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            {
                "content": "模型回复文本",
                "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": T},
                "model": "qwen-1_8b-chat-Q4_K_M",
                "finish_reason": "stop" | "length",
                "tokens_per_second": float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        t0 = time.time()

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        elapsed = time.time() - t0
        choice = response["choices"][0]
        usage = response.get("usage", {})
        content = choice["message"].get("content", "")

        # 计算 tokens/s
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"推理完成: {completion_tokens} tokens / {elapsed:.1f}s "
            f"= {tok_per_sec:.1f} tok/s"
        )

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": os.path.basename(self._model_path),
            "finish_reason": choice.get("finish_reason", "stop"),
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
    ) -> Iterator[str]:
        """
        流式对话补全，逐步 yield token 文本。

        Args:
            同 chat()

        Yields:
            str: 增量文本 chunk
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        stream = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
        )

        for chunk in stream:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    # ================================================================
    # 工具方法
    # ================================================================

    def _auto_threads(self) -> int:
        """自动检测 CPU 核心数，返回推荐线程数（物理核心数）。"""
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            if physical and physical > 0:
                return min(physical, 8)  # 最多 8 线程，避免资源争抢
        except Exception:
            pass

        # fallback: os.cpu_count() 返回逻辑核心数
        logical = os.cpu_count() or 4
        return max(2, min(logical // 2, 8))

    def get_memory_usage(self) -> dict:
        """获取当前内存占用估算。"""
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        proc_mem = process.memory_info().rss / (1024 ** 3)
        return {
            "process_gb": round(proc_mem, 2),
            "system_available_gb": round(mem.available / (1024 ** 3), 1),
            "system_percent": mem.percent,
        }

    def get_model_info(self) -> dict:
        """获取模型基本信息。"""
        info = {
            "engine": "llama.cpp",
            "model_path": self._model_path,
            "quant_type": self._quant_type,
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "loaded": self._loaded,
        }
        if self._loaded:
            info["memory"] = self.get_memory_usage()
        return info

    def reset_kv_cache(self) -> None:
        """
        清空 KV 缓存（用于多会话切换）。

        llama.cpp Python 绑定暂不直接提供 KV cache 重置 API。
        通过重新创建聊天上下文实现：下一次 create_chat_completion
        不传历史消息即为新会话。
        """
        # llama-cpp-python 在每次 create_chat_completion 时独立处理上下文，
        # 不保留跨调用的 KV cache。此方法为接口兼容预留。
        logger.debug("KV cache reset (llama.cpp — stateless, no-op)")

    def tokenize(self, text: str) -> List[int]:
        """将文本转换为 token ID 列表。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        return self._model.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens: List[int]) -> str:
        """将 token ID 列表转换为文本。"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        return self._model.detokenize(tokens).decode("utf-8", errors="replace")


# ================================================================
# 便捷函数
# ================================================================

def check_llama_cpp_available() -> bool:
    """检测 llama-cpp-python 是否可用。"""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def get_gguf_model_path(models_dir: str = None) -> Optional[str]:
    """
    在模型目录中自动查找可用的 GGUF 文件。

    Args:
        models_dir: 模型目录路径，默认 "models/"

    Returns:
        找到的 .gguf 文件完整路径，或 None
    """
    if models_dir is None:
        # PyInstaller 打包后 models/ 与 exe 同级
        if getattr(sys, 'frozen', False):
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(sys.executable)),
                "models",
            )
        else:
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models",
            )

    models_dir = os.path.abspath(models_dir)

    if not os.path.isdir(models_dir):
        return None

    # 查找所有 .gguf 文件
    gguf_files = []
    for fname in os.listdir(models_dir):
        if fname.lower().endswith(".gguf"):
            gguf_files.append(os.path.join(models_dir, fname))

    if not gguf_files:
        return None

    # 优先选择 Q4_K_M
    for path in gguf_files:
        if "Q4_K_M" in os.path.basename(path):
            return path

    # 其次 Q5_K_M
    for path in gguf_files:
        if "Q5_K_M" in os.path.basename(path):
            return path

    # 否则返回第一个找到的
    return gguf_files[0]
