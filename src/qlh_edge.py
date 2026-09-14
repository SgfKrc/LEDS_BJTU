"""QLH Edge —— 边缘设备最小入口（edge edition 基线）。

基线定义见 local_docs/边缘基线框定-2026-09-14.md：
- 只依赖白名单：llama-cpp-python + fastapi + uvicorn + psutil（+ numpy 传递依赖）
- 不 import torch / transformers / accelerate / 多模态 / RAG / TUI / 实验链
- 目标：运行时 venv ≤300MB、冷启动 ≤15s（无 torch import 链）

启动（在 edge venv 中）：
    QLH_EDGE_MODEL=models/Qwen-1_8B-Chat.Q4_K_M.gguf \\
        .venv-edge/Scripts/python.exe -m uvicorn qlh_edge:app --port 8010

端点：
    GET  /health            —— 进程/内存/模型状态（不含路径泄漏）
    GET  /status            —— 边缘基线元信息 + 模型懒加载状态
    POST /generate          —— 单模型生成（llama.cpp，首次请求懒加载）
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import psutil  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

EDGE_BASELINE = {
    "edition": "edge",
    "runtime_whitelist": ["llama-cpp-python", "fastapi", "uvicorn", "psutil", "httpx"],
    "excluded": ["torch", "transformers", "accelerate", "multimodal", "rag", "tui"],
    "venv_budget_mb": 300,
    "cold_start_budget_s": 15,
}

app = FastAPI(title="QLH Edge", version="0.1.0")

_state_lock = threading.RLock()
_llm = None
_llm_path: str = ""


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16_384)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)


def _model_path() -> str:
    path = os.environ.get("QLH_EDGE_MODEL", "").strip()
    if not path:
        raise HTTPException(status_code=503, detail="QLH_EDGE_MODEL not configured")
    candidate = Path(path)
    if not candidate.is_file():
        raise HTTPException(status_code=503, detail="configured model file not found")
    return str(candidate)


def _get_llm():
    """Lazy-load the GGUF model once (single-model edge profile)."""
    global _llm, _llm_path
    with _state_lock:
        path = _model_path()
        if _llm is None or _llm_path != path:
            from llama_cpp import Llama  # deferred: keeps cold start free of the engine

            _llm = Llama(
                model_path=path,
                n_ctx=int(os.environ.get("QLH_EDGE_N_CTX", "2048")),
                n_threads=int(os.environ.get("QLH_EDGE_THREADS", "0")) or None,
                verbose=False,
            )
            _llm_path = path
        return _llm


@app.get("/health")
def health():
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "edition": "edge",
        "model_loaded": _llm is not None,
        "ram_total_mb": round(mem.total / 2**20),
        "ram_available_mb": round(mem.available / 2**20),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        "pid": os.getpid(),
    }


@app.get("/status")
def status():
    payload = dict(EDGE_BASELINE)
    payload["model_loaded"] = _llm is not None
    payload["model_configured"] = bool(os.environ.get("QLH_EDGE_MODEL", "").strip())
    payload["started_at_unix"] = _STARTED_AT
    return payload


@app.post("/generate")
def generate(req: GenerateRequest):
    started = time.perf_counter()
    llm = _get_llm()
    out = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        echo=False,
    )
    text = out["choices"][0]["text"] if out.get("choices") else ""
    usage = out.get("usage", {})
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


_STARTED_AT = time.time()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("QLH_EDGE_HOST", "127.0.0.1"), port=int(os.environ.get("QLH_EDGE_PORT", "8010")))
