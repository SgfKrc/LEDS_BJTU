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
import hashlib
import json
import os
import sys
import time
from ctypes import byref
from pathlib import Path
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

CHATML_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<|endoftext|>",
    "</s>",
]


def _merge_stop_sequences(stop: List[str] = None) -> List[str]:
    merged: List[str] = []
    for value in (stop or []) + CHATML_STOP_SEQUENCES:
        if value and value not in merged:
            merged.append(value)
    return merged


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
        self._mmproj_path: str = ""
        self._mtmd_context = None
        self._mtmd_module = None
        self._mtmd_capabilities: Dict[str, bool] = {"vision": False, "audio": False}
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
        mmproj_path: str = None,
        mtmd_use_gpu: bool = False,
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

        if self.is_loaded or self._mtmd_context is not None:
            self.unload()
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

            if chat_format:
                load_kwargs["chat_format"] = chat_format
            elif model_path and "qwen-1_8b" in os.path.basename(model_path).lower():
                # Qwen-1.8B GGUF metadata is not always detected reliably.
                load_kwargs["chat_format"] = "chatml"

            load_kwargs.update(kwargs)

            self._model = Llama(**load_kwargs)
            self._loaded = True
            if mmproj_path:
                self.load_mmproj(mmproj_path, use_gpu=mtmd_use_gpu)

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
            self._free_mtmd_context()
            model = self._model
            self._model = None
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("GGUF model release after load failure failed", exc_info=True)
            self._loaded = False
            raise

    @staticmethod
    def _vram_free_bytes() -> Optional[int]:
        """查询当前 VRAM 空闲（nvidia-smi）；不可用时返回 None。"""
        import shutil
        import subprocess as _sp
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            out = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            return int(out.stdout.strip().splitlines()[0]) * 1024**2
        except Exception:
            return None

    @staticmethod
    def estimate_gpu_layers(
        total_layers: int,
        gguf_size_bytes: int,
        vram_free_bytes: int,
        *,
        kv_reserve_bytes: int = 512 * 1024**2,
        projector_reserve_bytes: int = 0,
        safety_margin: float = 1.15,
    ) -> int:
        """G4.5 部分 offload 层数估算（8GB 显存预算门）。

        全量 offload 在本机（RTX 4060 8GB）超预算（Q4 权重 7.1GB + mmproj +
        KV > 可用 ~7.4GB），必须部分 offload：按每层权重字节数计算可容纳层数，
        预留 KV 与安全余量（§5.7 决策：不承诺全量 offload）。
        """
        if total_layers <= 0 or gguf_size_bytes < 0 or vram_free_bytes < 0:
            raise ValueError("GPU layer budget inputs must be non-negative")
        if kv_reserve_bytes < 0 or projector_reserve_bytes < 0 or safety_margin <= 0:
            raise ValueError("GPU layer budget reserves are invalid")
        per_layer = max(1, gguf_size_bytes // total_layers)
        # safety_margin is headroom: required bytes * margin must fit in free VRAM.
        budget = int(
            vram_free_bytes / safety_margin
            - kv_reserve_bytes
            - projector_reserve_bytes
        )
        if budget <= 0:
            return 0
        return max(0, min(total_layers, budget // per_layer))

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_gemma4_native_assets(
        cls,
        lock_path: Path,
        gguf_path: Path,
        mmproj_path: Path,
    ) -> None:
        """Verify the exact frozen Gemma pair before any model initialization."""
        try:
            record = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("gemma4-native lock file is invalid") from exc
        if record.get("schema_version") != 1 or not isinstance(record.get("artifacts"), dict):
            raise RuntimeError("gemma4-native lock schema is invalid")
        artifacts = record["artifacts"]
        for key, path in (("main_gguf", gguf_path), ("mmproj", mmproj_path)):
            expected = artifacts.get(key)
            if not isinstance(expected, dict):
                raise RuntimeError(f"gemma4-native lock is missing {key}")
            filename = expected.get("filename")
            digest = expected.get("sha256")
            size_bytes = expected.get("size_bytes")
            if (
                not isinstance(filename, str)
                or path.name != filename
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes < 1
            ):
                raise RuntimeError(f"gemma4-native lock entry {key} is invalid")
            try:
                actual_size = path.stat().st_size
            except OSError as exc:
                raise FileNotFoundError(f"gemma4-native artifact is unavailable: {key}") from exc
            if actual_size != size_bytes:
                raise RuntimeError(f"gemma4-native {key} size does not match the lock")
            if cls._sha256_file(path) != digest.lower():
                raise RuntimeError(f"gemma4-native {key} SHA-256 does not match the lock")

    def load_gemma4_native(
        self,
        *,
        gguf_path: str = None,
        mmproj_path: str = None,
        n_ctx: int = 768,
        gpu_layers: int = -1,
        require_gpu_layers: int = 0,
        mtmd_use_gpu: bool = True,
    ) -> Dict[str, Any]:
        """G4.5 便捷加载：gemma4 原生工件（受管目录）+ GPU 预算门 + 互斥规则。

        - 缺省工件路径从 models/gemma4-native/gemma4-native.lock.json 读取；
        - gpu_layers=-1 时按显存预算自动估算（部分 offload）；
        - 显存不足（低于 require_gpu_layers 对应预算）时 fail-closed。
        """
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, int):
            raise ValueError("gpu_layers must be an integer")
        if gpu_layers < -1 or gpu_layers > 36:
            raise ValueError("gpu_layers must be -1 or in the range 0..36")
        if isinstance(require_gpu_layers, bool) or not isinstance(require_gpu_layers, int):
            raise ValueError("require_gpu_layers must be an integer")
        if require_gpu_layers < 0 or require_gpu_layers > 36:
            raise ValueError("require_gpu_layers must be in the range 0..36")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < 1:
            raise ValueError("n_ctx must be a positive integer")
        if not isinstance(mtmd_use_gpu, bool):
            raise ValueError("mtmd_use_gpu must be boolean")

        if getattr(sys, "frozen", False):
            asset_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        else:
            asset_root = Path(__file__).resolve().parents[1]
        lock = asset_root / "models" / "gemma4-native" / "gemma4-native.lock.json"
        if not lock.is_file():
            raise RuntimeError("gemma4-native 工件未冻结：先运行 gemma4_native_freeze.py --hash")
        try:
            record = json.loads(lock.read_text(encoding="utf-8"))
            artifacts = record["artifacts"]
            main_record = artifacts["main_gguf"]
            mmproj_record = artifacts["mmproj"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("gemma4-native lock file is invalid") from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or not isinstance(artifacts, dict)
            or not isinstance(main_record, dict)
            or not isinstance(mmproj_record, dict)
            or not isinstance(main_record.get("filename"), str)
            or not isinstance(mmproj_record.get("filename"), str)
        ):
            raise RuntimeError("gemma4-native lock schema is invalid")
        if gguf_path is None:
            gguf_path = str(lock.parent / main_record["filename"])
        if mmproj_path is None:
            mmproj_path = str(lock.parent / mmproj_record["filename"])
        gguf_file = Path(gguf_path).expanduser().absolute().resolve(strict=False)
        mmproj_file = Path(mmproj_path).expanduser().absolute().resolve(strict=False)
        if not gguf_file.is_file() or not mmproj_file.is_file():
            raise FileNotFoundError("gemma4-native 工件缺失：检查 models/gemma4-native/ 与冻结记录")

        load_kwargs: Dict[str, Any] = {}
        if gpu_layers > 0:
            load_kwargs["n_gpu_layers"] = int(gpu_layers)
        elif gpu_layers == -1:
            free = self._vram_free_bytes()
            if free is None:
                raise RuntimeError(
                    "无法查询显存（nvidia-smi 不可用）：请显式传 gpu_layers 或使用 CPU",
                )
            auto = self.estimate_gpu_layers(
                36,
                gguf_file.stat().st_size,
                free,
                projector_reserve_bytes=(mmproj_file.stat().st_size if mtmd_use_gpu else 0),
            )
            load_kwargs["n_gpu_layers"] = auto
            logger.info(
                "gemma4-native 自动 offload 层数: %d（VRAM %.1f GiB 预算门）",
                auto, free / 2**30,
            )
        if require_gpu_layers > 0 and load_kwargs.get("n_gpu_layers", 0) < require_gpu_layers:
            raise RuntimeError(
                f"显存预算不足：需要 ≥{require_gpu_layers} 层 offload，"
                f"实际 {load_kwargs.get('n_gpu_layers', 0)}（fail-closed）"
            )
        self._verify_gemma4_native_assets(lock, gguf_file, mmproj_file)
        self.load_model(
            model_path=str(gguf_file),
            n_ctx=n_ctx,
            mmproj_path=str(mmproj_file),
            mtmd_use_gpu=mtmd_use_gpu,
            **load_kwargs,
        )
        return self.get_capabilities()

    def unload(self) -> None:
        """卸载模型，释放内存。"""
        self._free_mtmd_context()
        model = self._model
        self._model = None
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("GGUF model release failed", exc_info=True)
        self._loaded = False
        logger.info("GGUF 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    # ================================================================
    # Native MTMD multimodal pipeline (G4.3.2B)
    # ================================================================

    def _load_mtmd_module(self):
        if self._mtmd_module is not None:
            return self._mtmd_module
        try:
            import llama_cpp.mtmd_cpp as mtmd
        except ImportError as exc:
            raise RuntimeError("llama.cpp MTMD binding is unavailable") from exc
        required = (
            "mtmd_context_params_default", "mtmd_init_from_file", "mtmd_free",
            "mtmd_support_vision", "mtmd_support_audio", "mtmd_default_marker",
            "mtmd_helper_bitmap_init_from_file", "mtmd_bitmap_free",
            "mtmd_input_chunks_init", "mtmd_input_chunks_free",
            "mtmd_input_chunks_size", "mtmd_input_chunks_get",
            "mtmd_input_chunk_get_type", "mtmd_input_text", "mtmd_bitmap_p_ctypes",
            "mtmd_tokenize",
            "mtmd_helper_eval_chunk_single", "mtmd_batch_init", "mtmd_batch_free",
            "mtmd_batch_add_chunk", "mtmd_batch_encode",
            "mtmd_batch_get_output_embd", "mtmd_helper_post_decode_callback",
            "mtmd_helper_decode_image_chunk",
        )
        missing = [name for name in required if not callable(getattr(mtmd, name, None))]
        if missing:
            raise RuntimeError("llama.cpp MTMD ABI missing symbols: " + ", ".join(missing))
        self._mtmd_module = mtmd
        return mtmd

    def _free_mtmd_context(self) -> None:
        context = self._mtmd_context
        self._mtmd_context = None
        self._mmproj_path = ""
        self._mtmd_capabilities = {"vision": False, "audio": False}
        if context is not None and self._mtmd_module is not None:
            try:
                self._mtmd_module.mtmd_free(context)
            except Exception:
                logger.debug("MTMD context release failed", exc_info=True)

    def load_mmproj(
        self,
        mmproj_path: str,
        *,
        use_gpu: bool = False,
        n_threads: int = None,
    ) -> Dict[str, Any]:
        """Load an explicit local projector and register native capabilities."""
        if not self.is_loaded:
            raise RuntimeError("load_model() must be called before load_mmproj()")
        if not mmproj_path or not os.path.isfile(mmproj_path):
            raise FileNotFoundError("MTMD projector file does not exist")
        mtmd = self._load_mtmd_module()
        self._free_mtmd_context()
        params = mtmd.mtmd_context_params_default()
        if hasattr(params, "use_gpu"):
            params.use_gpu = bool(use_gpu)
        if n_threads is not None and hasattr(params, "n_threads"):
            params.n_threads = int(n_threads)
        context = mtmd.mtmd_init_from_file(
            os.fspath(mmproj_path).encode("utf-8"), self._model.model, params,
        )
        if not context:
            raise RuntimeError("MTMD projector initialization failed")
        try:
            capabilities = {
                "vision": bool(mtmd.mtmd_support_vision(context)),
                "audio": bool(mtmd.mtmd_support_audio(context)),
            }
            if not capabilities["vision"]:
                raise RuntimeError("MTMD projector does not declare vision support")
        except Exception:
            try:
                mtmd.mtmd_free(context)
            except Exception:
                pass
            raise
        self._mtmd_context = context
        self._mmproj_path = os.path.abspath(mmproj_path)
        self._mtmd_capabilities = capabilities
        logger.info(
            "MTMD native capabilities registered: vision=%s audio=%s",
            capabilities["vision"], capabilities["audio"],
        )
        return self.get_capabilities()

    def get_capabilities(self) -> Dict[str, Any]:
        """Return a side-effect-free snapshot of registered capabilities."""
        return {
            "engine": "llama.cpp",
            "text": self.is_loaded,
            "native_mtmd": self._mtmd_context is not None,
            "vision": bool(self._mtmd_capabilities.get("vision")),
            "audio": bool(self._mtmd_capabilities.get("audio")),
            "mmproj_loaded": bool(self._mmproj_path),
        }

    @staticmethod
    def _mtmd_bitmap_pointer(wrapper):
        return getattr(wrapper, "bitmap", wrapper) if wrapper is not None else None

    def chat_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in one or two sentences. <__media__>",
        max_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        stop: List[str] = None,
        reasoning_channel_token_id: Optional[int] = 101,
        think_budget: int = 96,
        max_answer_tokens: int = 48,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the native MTMD image path for a single local image.

        Higher layers remain responsible for authorization, uploads and
        external-provider routing; this method only consumes a local image.
        """
        if not self.is_loaded:
            raise RuntimeError("model must be loaded before chat_image()")
        if not self._mtmd_context or not self._mtmd_capabilities.get("vision"):
            raise RuntimeError("native MTMD vision capability is not registered")
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError("image file does not exist")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if isinstance(max_answer_tokens, bool) or not isinstance(max_answer_tokens, int) or max_answer_tokens < 0:
            raise ValueError("max_answer_tokens must be non-negative")
        if isinstance(think_budget, bool) or not isinstance(think_budget, int) or think_budget < 0:
            raise ValueError("think_budget must be non-negative")
        if reasoning_channel_token_id is not None and (
            isinstance(reasoning_channel_token_id, bool)
            or not isinstance(reasoning_channel_token_id, int)
            or reasoning_channel_token_id < 0
        ):
            raise ValueError("reasoning_channel_token_id must be a non-negative integer or None")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        cancel_event = kwargs.pop("_cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
                "native_mtmd": True,
            }

        import llama_cpp
        mtmd = self._mtmd_module or self._load_mtmd_module()
        image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
        if os.path.splitext(image_path)[1].lower() not in image_extensions:
            raise ValueError("chat_image() accepts image files only")

        marker = mtmd.mtmd_default_marker()
        marker_text = marker.decode("utf-8", "replace") if isinstance(marker, bytes) else str(marker)
        prompt = prompt.replace("<__media__>", marker_text)
        if marker_text not in prompt:
            prompt = marker_text + "\n" + prompt
        if prompt.count(marker_text) != 1:
            raise ValueError("chat_image() requires exactly one MTMD media marker")

        wrapper = None
        bitmap = None
        chunks = None
        media_batch = None
        decode_batch = None
        sampler = None
        t0 = time.time()
        finish_reason = "length"
        channel_found = reasoning_channel_token_id is None
        generated_tokens: List[int] = []
        pre_channel_tokens: List[int] = []
        sampled_token_count = 0
        # G4.5：思考预算（等效 llama.cpp b10434 reasoning-budget sampler）——
        # 思考段超预算强制结束（注入 101），HF 独立工件思考段失控时正文仍稳定
        CHANNEL_START_TOKEN_ID = 100  # "<|channel>"
        THOUGHT_TEXT_TOKEN_ID = 45518  # "thought" 字面（HF 工件无 100 时的思考开始）
        in_thinking = False
        think_tokens = 0

        try:
            wrapper = mtmd.mtmd_helper_bitmap_init_from_file(
                self._mtmd_context, os.fspath(image_path).encode("utf-8"), False,
            )
            bitmap = self._mtmd_bitmap_pointer(wrapper)
            if not bitmap:
                raise RuntimeError("MTMD bitmap initialization failed")
            if getattr(wrapper, "video_ctx", None):
                raise ValueError("video inputs are not supported by chat_image()")

            chunks = mtmd.mtmd_input_chunks_init()
            text = mtmd.mtmd_input_text(prompt.encode("utf-8"), True, True)
            bitmaps = (mtmd.mtmd_bitmap_p_ctypes * 1)(bitmap)
            rc = mtmd.mtmd_tokenize(
                self._mtmd_context, chunks, byref(text), bitmaps, 1,
            )
            if rc != 0:
                raise RuntimeError(f"MTMD tokenize failed (rc={rc})")

            model_ctx = self._model._ctx
            clear_kv = getattr(model_ctx, "kv_cache_clear", None)
            if callable(clear_kv):
                clear_kv()
            if hasattr(self._model, "n_tokens"):
                self._model.n_tokens = 0
            n_past = llama_cpp.llama_pos(0)
            n_batch = kwargs.pop("n_batch", 64)
            if isinstance(n_batch, bool) or not isinstance(n_batch, int) or n_batch < 1:
                raise ValueError("n_batch must be positive")

            @mtmd.mtmd_helper_post_decode_callback
            def _post_decode_callback(_batch, _user_data):
                return 0

            for index in range(mtmd.mtmd_input_chunks_size(chunks)):
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                chunk = mtmd.mtmd_input_chunks_get(chunks, index)
                chunk_type = mtmd.mtmd_input_chunk_get_type(chunk)
                new_n_past = llama_cpp.llama_pos(n_past.value)
                if chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    rc = mtmd.mtmd_helper_eval_chunk_single(
                        self._mtmd_context, model_ctx.ctx, chunk, n_past, 0,
                        n_batch, False, byref(new_n_past),
                    )
                elif chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    if media_batch is None:
                        media_batch = mtmd.mtmd_batch_init(self._mtmd_context)
                    if not media_batch:
                        raise RuntimeError("MTMD batch initialization failed")
                    rc = mtmd.mtmd_batch_add_chunk(media_batch, chunk)
                    if rc == 0:
                        rc = mtmd.mtmd_batch_encode(media_batch)
                    if rc == 0:
                        embd = mtmd.mtmd_batch_get_output_embd(media_batch, chunk)
                        if not embd:
                            raise RuntimeError("MTMD image embedding is unavailable")
                        rc = mtmd.mtmd_helper_decode_image_chunk(
                            self._mtmd_context, model_ctx.ctx, chunk, embd, n_past, 0,
                            n_batch, byref(new_n_past), _post_decode_callback, None,
                        )
                else:
                    raise ValueError(f"unsupported MTMD input chunk type: {chunk_type}")
                if rc != 0:
                    raise RuntimeError(f"MTMD chunk evaluation failed (index={index}, rc={rc})")
                n_past = new_n_past

            if finish_reason == "cancelled":
                return {
                    "content": "",
                    "usage": {"prompt_tokens": int(n_past.value), "completion_tokens": 0,
                              "total_tokens": int(n_past.value)},
                    "model": os.path.basename(self._model_path),
                    "finish_reason": "cancelled",
                    "tokens_per_second": 0,
                    "native_mtmd": True,
                }

            from llama_cpp._internals import LlamaBatch, LlamaSampler

            decode_batch = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
            sampler = LlamaSampler()
            if temperature <= 0:
                sampler.add_greedy()
            else:
                sampler.add_top_k(int(kwargs.pop("top_k", 40)))
                sampler.add_top_p(max(0.0, min(float(top_p), 1.0)))
                sampler.add_temp(float(temperature))
                sampler.add_dist(int(kwargs.pop("seed", 0)))

            n_past_value = int(n_past.value)
            last_token = llama_cpp.llama_token_bos(model_ctx.ctx)
            eos_token = llama_cpp.llama_token_eos(model_ctx.ctx)
            for _ in range(int(max_tokens)):
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                decode_batch.set_batch([last_token], n_past_value, False)
                model_ctx.decode(decode_batch)
                n_past_value += 1
                token = sampler.sample(model_ctx, -1)
                sampler.accept(token)
                if token == eos_token:
                    finish_reason = "stop"
                    break
                sampled_token_count += 1
                if reasoning_channel_token_id is not None:
                    if (
                        token == CHANNEL_START_TOKEN_ID
                        or (
                            not channel_found and not in_thinking
                            and not generated_tokens and not pre_channel_tokens
                            and token == THOUGHT_TEXT_TOKEN_ID
                        )
                    ):
                        # 思考段开始（通道标记或生成开头 "thought" 字面）
                        in_thinking = True
                        think_tokens = 0
                        channel_found = False
                        pre_channel_tokens.clear()
                        last_token = token
                        continue
                    if in_thinking:
                        think_tokens += 1
                        if token == reasoning_channel_token_id:
                            # 思考段自然结束
                            in_thinking = False
                            channel_found = True
                            last_token = token
                            continue
                        if think_budget > 0 and think_tokens > think_budget:
                            # 思考超预算：注入结束 tag（等效 reasoning-budget FORCING）
                            in_thinking = False
                            channel_found = True
                            last_token = reasoning_channel_token_id
                            continue
                        # 思考内容：丢弃
                        last_token = token
                        continue
                    if token == reasoning_channel_token_id:
                        if generated_tokens:
                            # 正文已结束（模型以 101 开新段即循环）：停止
                            break
                        # 思考段自然结束（无 100/45518 触发路径）
                        channel_found = True
                        last_token = token
                        continue
                    if not channel_found:
                        pre_channel_tokens.append(token)
                        last_token = token
                        continue
                    generated_tokens.append(token)
                    if max_answer_tokens > 0 and len(generated_tokens) >= max_answer_tokens:
                        break
                    last_token = token
                    continue
                generated_tokens.append(token)
                last_token = token

            output_tokens = generated_tokens if channel_found else pre_channel_tokens
            # G4.5 裁剪：头尾思考段残余（101/thought/空行）
            while output_tokens and output_tokens[-1] in (101, 106, 107, 45518):
                output_tokens.pop()
            while output_tokens and output_tokens[0] in (101, 106, 107):
                output_tokens.pop(0)
            content = self._model.detokenize(output_tokens).decode("utf-8", "replace")
            for stop_value in _merge_stop_sequences(stop):
                if stop_value and stop_value in content:
                    content = content.split(stop_value, 1)[0]
                    finish_reason = "stop"
                    break
            elapsed = time.time() - t0
            completion_tokens = sampled_token_count
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": int(n_past.value),
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(n_past.value) + completion_tokens,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": finish_reason,
                "tokens_per_second": round(completion_tokens / elapsed if elapsed > 0 else 0, 1),
                "native_mtmd": True,
                "reasoning_channel_found": channel_found,
            }
        finally:
            if sampler is not None:
                try:
                    sampler.close()
                except Exception:
                    logger.debug("MTMD sampler release failed", exc_info=True)
            if decode_batch is not None:
                try:
                    decode_batch.close()
                except Exception:
                    logger.debug("MTMD decode batch release failed", exc_info=True)
            if media_batch is not None:
                try:
                    mtmd.mtmd_batch_free(media_batch)
                except Exception:
                    logger.debug("MTMD media batch release failed", exc_info=True)
            if chunks is not None:
                try:
                    mtmd.mtmd_input_chunks_free(chunks)
                except Exception:
                    logger.debug("MTMD input chunks release failed", exc_info=True)
            if bitmap is not None:
                try:
                    mtmd.mtmd_bitmap_free(bitmap)
                except Exception:
                    logger.debug("MTMD bitmap release failed", exc_info=True)

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
        cancel_event = kwargs.pop("_cancel_event", None)

        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
                "usage_estimated": True,
            }

        # llama-cpp-python's chat API has no stopping_criteria argument. For
        # cooperative cancellation, consume its stream and close at a token
        # boundary; the task coordinator then discards this partial output.
        if cancel_event is not None:
            stream = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=_merge_stop_sequences(stop),
                stream=True,
            )
            content_parts: List[str] = []
            finish_reason = "stop"
            try:
                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        choice = choices[0]
                        text = choice.get("delta", {}).get("content", "")
                        if text:
                            content_parts.append(text)
                        finish_reason = choice.get("finish_reason") or finish_reason
                    if cancel_event.is_set():
                        break
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

            content = "".join(content_parts)
            elapsed = time.time() - t0
            completion_tokens = len(
                self._model.tokenize(
                    content.encode("utf-8"), add_bos=False, special=True,
                )
            ) if content else 0
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
                "model": os.path.basename(self._model_path),
                "finish_reason": "cancelled" if cancel_event.is_set() else finish_reason,
                "tokens_per_second": round(
                    completion_tokens / elapsed if elapsed > 0 else 0, 1,
                ),
                "usage_estimated": True,
            }

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=_merge_stop_sequences(stop),
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
            stop=_merge_stop_sequences(stop),
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
            "capabilities": self.get_capabilities(),
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
