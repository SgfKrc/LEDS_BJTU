"""
FastAPI 后端服务 — 模型管理 + 对话接口 + 性能监控 + 设备检测
===============================================================
启动: python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
      或在项目根目录: uvicorn src.api_server:app --host 0.0.0.0 --port 8000

功能:
- POST /api/models/load      — 加载/切换模型 (fp16 / int4 / int8)
- POST /api/chat             — 对话（多轮会话，自动维护 KV 缓存）
- POST /api/chat/clear       — 清空对话历史 + KV 缓存
- GET  /api/status           — 系统状态（模型信息、GPU、KV缓存、设备档位）
- GET  /api/models/current   — 当前模型信息
- GET  /api/device/profile   — 完整设备画像（CPU/RAM/GPU/Disk/OS）
- POST /api/device/auto-configure — 应用设备自适应配置
- POST /api/chat/upload       — 上传文本文件（txt/md/csv/py/json/log）
- GET  /api/presets           — 预设问题列表
"""

import hashlib
import json
import logging
import re
import time
import sys
import os
import threading
import uuid
from collections import Counter, deque
from contextvars import ContextVar
from typing import Optional

import torch

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paged_kv_cache import PagedKVCache
from device_profiler import DeviceProfiler, get_profile
from scheduler import Scheduler
import model_config as mc
from config import (
    MODEL_NAME, MODEL_PATH, QUANT_TYPE, USE_COMPILE,
    DEVICE, PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN, RUN_MODE,
    NODE_ROLE, NODE_ID, MAX_NODES, SERVER_IP, SERVER_PORT, API_PORT,
)

# 数据库模块（可选，未安装 psycopg2 时使用内存降级）
try:
    from db import init_db, close_db, db_health
    _db_importable = True
except ImportError:
    _db_importable = False
    init_db = lambda: None
    close_db = lambda: None
    db_health = lambda: {"status": "unavailable", "message": "psycopg2 未安装"}

# _db_available 动态追踪实际连接状态：启动时尝试连接，失败则标记 False
# _db_importable 仅表示 psycopg2 已安装（静态）
_db_available = _db_importable

# 本地文件储存（云数据库不可用时的降级方案）
import local_store as _local_store

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_LOG_BUFFER_MAXLEN = 5000
_log_buffer: deque[dict] = deque(maxlen=_LOG_BUFFER_MAXLEN)
_log_buffer_lock = threading.RLock()
_log_buffer_total_seen = 0


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        return True


_request_id_filter = RequestIdFilter()


class _LazyModelManager:
    """Delay importing model_module until the model manager is first used."""

    __slots__ = ("_instance", "_lock")

    def __init__(self):
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.RLock())

    def _get_instance(self):
        instance = self._instance
        if instance is not None:
            return instance
        with self._lock:
            instance = self._instance
            if instance is None:
                from model_module import ModelManager
                instance = ModelManager()
                object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._get_instance(), name, value)

    def __delattr__(self, name):
        if name in self.__slots__:
            raise AttributeError(name)
        delattr(self._get_instance(), name)

    def __repr__(self):
        instance = self._instance
        if instance is None:
            return "<_LazyModelManager unloaded>"
        return repr(instance)


def _current_node_id_safe() -> str:
    try:
        return scheduler.get_effective_node_id()
    except Exception:
        return NODE_ID


def _current_device_ip_safe() -> str:
    try:
        return getattr(scheduler, "_lan_ip", "") or SERVER_IP
    except Exception:
        return SERVER_IP


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _log_buffer_total_seen
        try:
            entry = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(record.created),
                ),
                "level": record.levelname,
                "levelno": record.levelno,
                "name": record.name,
                "message": record.getMessage(),
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "request_id": getattr(record, "request_id", _request_id_ctx.get("-")),
                "node_id": _current_node_id_safe(),
                "device_ip": _current_device_ip_safe(),
                "thread": record.threadName,
            }
            if record.exc_info:
                entry["exc_text"] = self.format(record)
            with _log_buffer_lock:
                _log_buffer_total_seen += 1
                entry["seq"] = _log_buffer_total_seen
                _log_buffer.append(entry)
        except Exception:
            self.handleError(record)


def _close_logging_handlers(keep_memory: bool = False):
    """
    关闭并移除 root logger 上的 handlers，避免 Windows 下日志文件被占用。

    Args:
        keep_memory: 若为 True，保留 MemoryLogHandler 和 StreamHandler，
                     仅关闭文件类 handler（RotatingFileHandler）。
                     用于日志文件删除操作期间维持内存缓冲和终端输出。
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        is_file_handler = isinstance(handler, logging.FileHandler)
        if keep_memory and not is_file_handler and isinstance(
            handler, (logging.StreamHandler, MemoryLogHandler)
        ):
            continue  # 保留终端和内存 handler，确保删除期间日志不丢失
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup_logging():
    """配置日志：控制台输出 + RotatingFileHandler（5MB×5 滚动）。"""
    import logging.handlers
    from datetime import datetime
    from config import LOG_DIR, LOG_LEVEL

    os.makedirs(LOG_DIR, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    _close_logging_handlers()

    # 控制台 handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.addFilter(_request_id_filter)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(message)s"))
    root.addHandler(ch)

    # 文件 handler（按日期 + 大小滚动）
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, f"qlh-{datetime.now():%Y-%m-%d}.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.addFilter(_request_id_filter)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # 内存环形缓冲 handler（用于 /api/logs/recent，不持有文件句柄）
    mh = MemoryLogHandler()
    mh.setLevel(level)
    mh.addFilter(_request_id_filter)
    mh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] request_id=%(request_id)s %(name)s: %(message)s"))
    root.addHandler(mh)

    # uvicorn 日志也走文件
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = True

    root.setLevel(level)


setup_logging()
logger = logging.getLogger("api_server")

# ============================================================
# FastAPI 应用初始化
# ============================================================

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan 上下文管理器（替代废弃的 @app.on_event）"""
    # ---- startup ----
    await _startup_device_detection()
    yield
    # ---- shutdown ----
    await _shutdown_resources()


app = FastAPI(
    title="轻量化大模型分布式边缘推理优化系统",
    version="0.1.7",
    description="北京交通大学 · 大学生创新创业训练计划",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173",
                   "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_request_id(value: str | None) -> str:
    if not value:
        return uuid.uuid4().hex
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "", value.strip())
    if not cleaned:
        return uuid.uuid4().hex
    return cleaned[:64]


@app.middleware("http")
async def request_id_logging_middleware(request: Request, call_next):
    request_id = _normalize_request_id(request.headers.get("X-Request-ID"))
    token = _request_id_ctx.set(request_id)
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.error(
            "event=http_request_error request_id=%s method=%s path=%s status=%s",
            request_id, request.method, request.url.path, status_code,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，请查看后端日志", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )
        _request_id_ctx.reset(token)


@app.exception_handler(HTTPException)
async def http_exception_with_request_id(request: Request, exc: HTTPException):
    request_id = _request_id_ctx.get("-")
    headers = dict(exc.headers or {})
    headers["X-Request-ID"] = request_id
    if exc.status_code >= 500:
        logger.error(
            "event=http_exception request_id=%s method=%s path=%s status=%s detail=%s",
            request_id, request.method, request.url.path, exc.status_code, exc.detail,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers=headers,
    )

# ============================================================
# 全局状态
# ============================================================

model_manager = _LazyModelManager()
kv_cache: Optional[PagedKVCache] = None
active_session_id: Optional[str] = None           # 当前活跃会话 ID
session_histories: dict[str, list[dict]] = {}     # session_id → 对话历史列表
conversation_stats: dict = {                    # 累计对话统计（实际消耗追踪）
    "total_prompt_tokens": 0,
    "total_generated_tokens": 0,
    "total_time_seconds": 0.0,
    "rounds": 0,
}
current_quant: str = QUANT_TYPE
model_loaded: bool = False
device_profile: Optional[dict] = None           # 设备画像缓存
_device_profile_ready = threading.Event()
_device_profile_started = False
generation_config: dict = {
    "max_new_tokens": 1024,          # laptop 档默认值
    "tier_max_new_tokens": 1024,     # 设备档位上限（auto_configure 后更新）
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# 调度器（单机 / 分布式模式共用）
scheduler: Scheduler = Scheduler()


def _refresh_pipeline_layer_config() -> None:
    """主节点模型变化后重新下发层配置，并使旧 ACK 失效。"""
    try:
        if scheduler._effective_role() == "master":
            scheduler.push_layer_config_to_clients()
    except Exception as e:
        # 模型本身已经加载成功；同步失败时保持流水线 not-ready，后续请求
        # 会安全回退到主节点本地推理。
        logger.warning(f"模型加载后刷新流水线层配置失败: {e}", exc_info=True)


def _run_exclusive_model_change(change):
    """Block inference, invalidate old worker ACKs, then refresh the new model."""
    with scheduler._inference_lock:
        with scheduler._layer_config_lock:
            scheduler._layer_config_pushed.clear()
            scheduler._layer_config_expected.clear()
            scheduler._layer_config_acks.clear()
        try:
            return change()
        finally:
            _refresh_pipeline_layer_config()


# ============================================================
# SIGTERM 优雅关闭（Linux systemd / 容器环境）
# 注意：uvicorn 内置了自己的 SIGTERM/SIGINT 处理器，在 uvicorn.run() 中
# 自动处理优雅关闭。不在此处注册模块级信号处理器，因为会被 uvicorn 覆盖。


# ============================================================
# 启动事件 — 设备检测（通过 lifespan 调用）
# ============================================================

async def _startup_device_detection():
    """Start core services immediately and detect slower hardware in background."""
    global device_profile, scheduler, _device_profile_started

    def _detect_device_profile() -> None:
        global device_profile
        try:
            profiler = get_profile()
            device_profile = profiler.to_dict()
            scheduler.update_local_device_profile(device_profile)
            logger.info(
                f"🚀 设备检测完成: tier={profiler.tier.value} "
                f"score={profiler.score:.1f}/100 | "
                f"CPU={profiler.cpu.physical_cores}核 RAM={profiler.ram.total_gb}GB "
                f"GPU={profiler.gpu.name}"
            )
            logger.info(f"   推荐配置: {profiler.recommend_config()['description']}")
            for warning in device_profile.get("warnings", []):
                logger.warning(f"   {warning}")
        except Exception as e:
            logger.error(f"设备检测失败: {e}")
            device_profile = None
        finally:
            _device_profile_ready.set()

    if not _device_profile_started:
        _device_profile_started = True
        threading.Thread(
            target=_detect_device_profile,
            name="device-profile",
            daemon=True,
        ).start()

    # 初始化调度器（单机模式下不启动 TCP 监听）
    try:
        scheduler.start()
        logger.info(f"调度器已初始化: mode={RUN_MODE}")
    except Exception as e:
        logger.error(f"调度器初始化失败: {e}")

    # L5: 启动日志保留策略清理线程
    try:
        _start_log_retention_thread()
    except Exception as e:
        logger.warning(f"日志保留线程启动失败: {e}")

    # 调度器启动时已经尝试过数据库；这里复用结果，避免重复阻塞初始化。
    try:
        import scheduler as _scheduler_module
        db_module = _scheduler_module._get_db()
        if db_module is None:
            raise RuntimeError(
                _scheduler_module.get_database_status().get("last_error")
                or "数据库未配置或暂不可用"
            )
        # ★ 设置数据隔离参数：conversations/sessions 将按 node_id 过滤
        from db import set_active_node_id
        # ★ MAC 不匹配自动切换：若 start() 时检测到 MAC 不匹配且有真实主节点，
        #    后台线程 _auto_switch_to_client 会在 2s 后完成切换。
        #    此处等待最多 5s，确保切换完成后再设置 node_id。
        if (getattr(scheduler, '_master_identity_reason', '') == 'mac_mismatch'
                and getattr(scheduler, '_role_override', None) != 'client'):
            # 自动切换尚未完成（后台线程延迟 2s），等待切换
            logger.info("⏳ 等待 MAC 不匹配自动切换到从节点模式...")
            import time as _time
            for _ in range(10):  # 最多等 5 秒 (10 × 0.5s)
                _time.sleep(0.5)
                if getattr(scheduler, '_role_override', None) == 'client':
                    break
        set_active_node_id(scheduler.get_effective_node_id())
        logger.info(f"数据库已连接，活跃节点: {scheduler.get_effective_node_id()}")
    except Exception as e:
        global _db_available
        _db_available = False
        runtime = _scheduler_module.get_database_status()
        if runtime.get("configured", True):
            logger.warning(f"数据库初始化失败（使用本地文件降级）: {e}")
        else:
            logger.info("数据库未配置，使用本地文件存储")

    # P3: 启动邮件投票轮询器（仅 master 节点，IMAP 轮询不需要 CUDA）
    try:
        if scheduler._effective_role() == "master":
            from email_notifier import start_mail_poller
            start_mail_poller(poll_interval=60)
            logger.info("📬 邮件投票轮询器已启动 (master)")
    except Exception as e:
        logger.warning(f"邮件投票轮询器启动失败: {e}")

    # P3: 启动审查工单过期检查后台线程（仅 master，每 5 分钟）
    try:
        if scheduler._effective_role() == "master":
            import threading as _th2
            def _review_expire_loop():
                import time as _time
                from review import ReviewManager
                _time.sleep(120)  # 启动后等 2 分钟再开始（避免空跑）
                while getattr(scheduler, '_running', True):
                    try:
                        ReviewManager().resolve_expired()
                    except Exception:
                        pass
                    _time.sleep(300)  # 每 5 分钟检查一次
            _t = _th2.Thread(target=_review_expire_loop, daemon=True, name="review-expire")
            _t.start()
            logger.info("⏳ 审查工单过期检查线程已启动 (master)")
    except Exception as e:
        logger.warning(f"审查工单过期检查启动失败: {e}")


# ============================================================
# 关闭事件 — 资源清理（通过 lifespan 调用）
# ============================================================

async def _shutdown_resources():
    """应用关闭时清理资源：数据库连接池 + 调度器 + TCP 服务"""
    # 1. 停止调度器（关闭 TCP 连接，注销从节点）
    try:
        scheduler.stop()
        logger.info("调度器已停止")
    except Exception as e:
        logger.warning(f"调度器停止异常: {e}")

    # 2. 关闭数据库连接池（线程超时防卡死）
    import threading as _th

    def _close_db_safe():
        try:
            close_db()
        except Exception:
            pass

    _t = _th.Thread(target=_close_db_safe, daemon=True)
    _t.start()
    _t.join(timeout=3.0)
    if _t.is_alive():
        logger.warning("数据库连接池关闭超时（3s），跳过")
    else:
        logger.info("数据库连接池已关闭")

    # 3. P3: 停止邮件投票轮询器
    try:
        from email_notifier import stop_mail_poller
        stop_mail_poller()
    except Exception:
        pass


# ============================================================
# Pydantic 模型
# ============================================================

class LoadModelRequest(BaseModel):
    engine: str = Field(
        default="llama_cpp",
        description="推理引擎: llama_cpp (GGUF, 推荐) | pytorch (Safetensors) | auto",
    )
    quant_type: str = Field(
        default="int4",
        description="PyTorch 量化精度: fp16 | int8 | int4（llama_cpp 引擎忽略此参数）",
    )
    use_compile: bool = Field(
        default=False,
        description="是否开启 torch.compile 算子融合（仅 PyTorch FP16 有效）",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="模型唯一标识（P3多模型支持）。不传则使用默认 Qwen-1.8B。",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: Optional[str] = Field(default=None, description="会话ID，为空时使用当前活跃会话")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    show_thinking: bool = Field(default=False, description="启用深度思考展示")
    streaming_mode: str = Field(
        default="full",
        description="流式模式（仅 /api/chat/stream 生效）: full=假流式完整功能（含历史/追问/持久化，默认） | fast=真流式逐token（低延迟，跳过持久化）",
    )
    client_node_id: Optional[str] = Field(default=None, description="请求来源节点 ID（Android/PC 客户端上报）")
    client_node_type: Optional[str] = Field(default=None, description="请求来源节点类型: pc | android")
    client_mode: Optional[str] = Field(default=None, description="请求来源模式: thin | full")
    client_app_variant: Optional[str] = Field(default=None, description="请求来源 App variant: full | lite")


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    thinking_content: Optional[str] = None
    metrics: dict = {}
    followups: list[str] = []


class NodeDetail(BaseModel):
    node_id: str
    role: str
    node_type: str = "pc"
    state: str
    address: str = ""
    hostname: str = ""
    device_info: dict = {}
    network_type: str = "unknown"
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    task_count: int = 0
    error_count: int = 0
    is_available: bool = False


class ClusterStatus(BaseModel):
    run_mode: str
    nodes_ready: bool
    nodes: dict[str, NodeDetail] = {}
    current_task: Optional[dict] = None
    tcp_server: Optional[dict] = None
    pipeline: Optional[dict] = None
    pipeline_queue: Optional[dict] = None


class UpdateMaxNodesRequest(BaseModel):
    max_nodes: int = Field(..., ge=1, le=64, description="新的最大节点数（包含 master）")


class ConnectToMasterRequest(BaseModel):
    master_host: str = Field(..., description="主节点 IP 地址", min_length=1)
    master_port: int = Field(8888, ge=1, le=65535, description="主节点端口")
    switch_to_client: bool = Field(
        False,
        description="待配置节点显式切换为从节点后加入现有集群",
    )


class FirstConnectBootstrapRequest(BaseModel):
    node_id: Optional[str] = Field(default=None, max_length=64, description="客户端稳定节点 ID")
    node_type: str = Field(default="pc", description="节点类型: pc | android")
    hostname: str = Field(default="", max_length=128, description="设备名")
    platform: str = Field(default="", max_length=64, description="平台: windows | linux | android")
    app_variant: str = Field(default="", max_length=32, description="Android full | lite")
    app_version: str = Field(default="", max_length=64, description="客户端版本")
    capabilities: dict = Field(default_factory=dict, description="设备画像/能力")


# ============================================================
# 辅助函数
# ============================================================

def _build_chat_prompt(messages: list[dict], system_prompt: Optional[str] = None,
                       assistant_prefill: Optional[str] = None) -> str:
    """
    使用 Qwen 的 chat template 构建对话 prompt。
    Qwen-1.8B-Chat 使用 <|im_start|>/<|im_end|> 格式。

    Args:
        messages: 对话历史列表
        system_prompt: 可选的系统提示，会插入在对话历史之前
        assistant_prefill: 可选的助手预填文本（强制模型从此处续写），
                           用于引导结构化输出，如深度思考的「【思考】\n」
    """
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    if assistant_prefill:
        parts.append(assistant_prefill)
    return "\n".join(parts)


def _build_model_chat_prompt(tokenizer, messages: list[dict],
                             system_prompt: Optional[str] = None,
                             assistant_prefill: Optional[str] = None) -> str:
    """Build a prompt with the active tokenizer's native chat template."""
    chat_messages = []
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    chat_messages.extend(messages)

    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # DeepSeek-R1-Distill templates already end with "<think>\n" when
        # add_generation_prompt=True. Appending the legacy Chinese prefill here
        # mixes two incompatible thinking protocols.
        native_thinking_prompt = "<think>" in prompt[-64:].lower()
        if assistant_prefill and not native_thinking_prompt:
            prompt += assistant_prefill
        return prompt
    except Exception:
        return _build_chat_prompt(
            messages,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )


# ================================================================
# 深度思考展示
# ================================================================

THINKING_START = "【思考】"
THINKING_END   = "【思考结束】"

THINKING_SYSTEM_PROMPT = (
    "你是一个善于深度思考的AI助手。回答前先进行推理分析，再给出答案。\n\n"
    "严格按以下格式输出：\n"
    "【思考】\n"
    "（你的推理过程，2-3句话即可）\n"
    "【思考结束】\n"
    "（你的最终回答）\n\n"
    "注意：\n"
    "- 必须在【思考结束】之后写回答内容\n"
    "- 回答部分不要写标记符号\n"
    "- 不要重复输出【思考】或【思考结束】"
)


def _strip_native_thinking_tags(text: str) -> str:
    """Remove native thinking/answer tags and leaked ChatML sentinels."""
    import re as _re

    if not text:
        return text

    result = _re.sub(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
        '',
        text,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    # DeepSeek templates put "<think>\n" in the prompt. The generated completion
    # can therefore start with "reasoning...</think>\nanswer" and contain only
    # the closing tag. In that case, drop everything up to the closing tag.
    result = _re.sub(
        r'^.*?<\s*/\s*think\s*>',
        '',
        result,
        count=1,
        flags=_re.DOTALL | _re.IGNORECASE,
    )

    response_match = _re.search(
        r'<\s*(?:answer|response)\s*>(.*?)(?:<\s*/\s*(?:answer|response)\s*>|$)',
        result,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    if response_match:
        result = response_match.group(1)

    result = _re.sub(r'<\s*/?\s*(?:think|answer|response)\s*>', '', result, flags=_re.IGNORECASE)
    result = result.replace('<|im_end|>', '').replace('<|im_start|>', '')
    result = _re.sub(r'<\s*\|im_(?:start|end)\|\s*>', '', result)
    result = _re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def _parse_thinking_response(text: str) -> tuple:
    """
    解析模型输出，分离思考内容和最终答案。

    当 show_thinking 启用时，模型应输出：

        【思考】
        (推理过程)
        【思考结束】
        (最终答案)

    本函数对各种格式错误具有容错能力：
    - 缺少结束标记 → 尝试智能分割
    - 答案为空 → 从思考中提取最后一段作为答案
    - 重复标记 → 使用第一次出现的有效标记对

    Args:
        text: 模型原始输出文本（已包含预填的【思考】前缀）

    Returns:
        (answer_content, thinking_content)
        - answer_content: 最终答案文本（绝不包含思考标记）
        - thinking_content: 思考过程文本，格式不匹配时为 None
    """
    import re as _re

    if not text:
        return "", None

    # ---- 查找标记位置 ----
    start_idx = text.find(THINKING_START)
    end_idx = text.find(THINKING_END)

    # ---- 情况1：标记成对且顺序正确 ----
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking = text[start_idx + len(THINKING_START):end_idx].strip()
        answer = text[end_idx + len(THINKING_END):].strip()

        # 清理思考中的标题前缀
        thinking = _re.sub(r'^分析思路[：:]\s*', '', thinking)

        # 清理答案开头的标题前缀
        answer = _re.sub(r'^【最终答案】[：:]?\s*', '', answer)
        answer = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', answer, flags=_re.IGNORECASE)
        for _pat in [r'^\[你的最终回答[^\]]*\]\s*', r'^\[你的推理过程[^\]]*\]\s*',
                     r'^（推理内容）\s*', r'^（答案内容）\s*',
                     r'^（给用户的答案[^）]*）\s*']:
            answer = _re.sub(_pat, '', answer)

        # 清理答案中残留的思考标记（模型可能在答案里又输出了标记）
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "").strip()

        # 开始标记之前的内容拼入答案
        prefix = text[:start_idx].strip()
        if prefix:
            answer = prefix + ("\n" + answer if answer else "")

        # 思考内容为空 → 格式未遵循，fallthrough 到情况2
        if thinking:
            # 如果答案为空但思考非空 → 尝试从思考中提取最后一段作为答案
            # 1.8B 模型常见失败模式：把所有内容都放在思考里，答案留空
            if not answer and thinking:
                paragraphs = thinking.split("\n")
                # 取最后一段非空内容作为答案
                for p in reversed(paragraphs):
                    p = p.strip()
                    if p and len(p) > 10:
                        answer = p
                        break
                # 如果还是空，用整个思考作为答案
                if not answer:
                    answer = thinking
            return answer, thinking

    # ---- 情况2：DeepSeek-R1 / Qwen3 本地  格式 ----
    # 这些模型通过 ChatML 原生输出  ...  包裹思考，
    # 不依赖 THINKING_SYSTEM_PROMPT 注入的【思考】标记。
    import re as _re2
    native_match = _re2.search(
        r'<\s*think\s*>(.*?)<\s*/\s*think\s*>',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if native_match:
        thinking = native_match.group(1).strip()
        # 取  之后、</think> 之前的内容作为思考
        answer = text[:native_match.start()].strip()
        after_think = text[native_match.end():].strip()
        # 去除  标记
        after_think = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', after_think, flags=_re2.IGNORECASE)
        if after_think:
            answer = (answer + '\n' + after_think).strip() if answer else after_think
        # 也尝试从  标记中提取回答
        response_match = _re2.search(
            r'<\s*(?:response|answer)\s*>(.*)',
            answer if answer else '',
            flags=_re2.DOTALL | _re2.IGNORECASE,
        )
        if response_match:
            answer = response_match.group(1).strip()
        # 清理残余标签
        answer = _re2.sub(r'<\s*/?\s*(?:think|response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if thinking:
            return answer, thinking

    closing_only_match = _re2.search(
        r'^(.*?)<\s*/\s*think\s*>(.*)$',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if closing_only_match:
        thinking = closing_only_match.group(1).strip()
        thinking = thinking.replace(THINKING_START, "").replace(THINKING_END, "").strip()
        answer = closing_only_match.group(2).strip()
        answer = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', answer, flags=_re2.IGNORECASE)
        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "")
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
        if answer or thinking:
            return answer, thinking or None

    # ---- 情况3：格式未遵循（缺少标记或标记顺序错误） ----
    # 清理所有思考标记，返回干净的文本作为答案
    cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
    # 也清理本地格式标记
    cleaned = _strip_native_thinking_tags(cleaned)
    # 清理常见的标题前缀
    cleaned = _re.sub(r'^分析思路[：:]\s*', '', cleaned)
    cleaned = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', cleaned, flags=_re.IGNORECASE)
    return cleaned, None


def _format_model_response(text: str, show_thinking: bool,
                           native_thinking_prompt: bool = False) -> tuple[str, Optional[str]]:
    """Format generated text without exposing unfinished native reasoning."""
    if show_thinking:
        return _parse_thinking_response(text)
    if native_thinking_prompt and "</think>" not in (text or "").lower():
        return "", None
    return _strip_native_thinking_tags(text), None


# ================================================================
# 多会话管理
# ================================================================

def _get_active_history() -> list[dict]:
    """
    获取当前活跃会话的对话历史列表。

    如果没有活跃会话，返回空列表（不自动创建会话）。
    返回的列表对象可被原地修改（append、clear 等）。
    """
    global active_session_id, session_histories
    if active_session_id is None:
        return []  # 不自动创建——由前端在首次发消息时显式创建
    if active_session_id not in session_histories:
        session_histories[active_session_id] = []
    return session_histories[active_session_id]


def _switch_session(target_id: str) -> None:
    """
    切换到目标会话：暂存当前历史 → 加载目标历史 → 清 KV Cache。

    如果目标会话不在内存中，首先尝试从 DB 加载；DB 不可用时初始化为空列表。
    """
    global active_session_id, kv_cache
    if active_session_id == target_id:
        return

    active_session_id = target_id

    # 如果目标会话不在内存中，尝试从 DB 或本地文件加载
    if target_id not in session_histories:
        messages = []
        if _db_available:
            try:
                import db as _db_mod
                rows = _db_mod.get_conversation(target_id)
                messages = [{"role": r["role"], "content": r["content"]} for r in rows]
            except Exception:
                pass
        if not messages:
            # DB 不可用或返回空 → 尝试本地文件
            try:
                local_rows = _local_store.load_local_conversation(target_id)
                messages = [{"role": r["role"], "content": r["content"]} for r in local_rows]
            except Exception:
                pass
        session_histories[target_id] = messages

    # 清 KV Cache（切换会话后 prompt 不同，必须重建）
    if kv_cache:
        kv_cache.clear()
    _init_kv_cache()
    logger.info(f"已切换到会话: {target_id}")


def _reset_runtime_conversation_state(clear_histories: bool = True) -> None:
    """Clear in-memory conversation/KV state after a model change."""
    global kv_cache, conversation_stats, session_histories

    if kv_cache:
        kv_cache.clear()
    kv_cache = None
    if clear_histories:
        session_histories = {}
    conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }


def _auto_title_session(session_id: str, first_message: str) -> None:
    """用首条用户消息自动生成会话标题（截取前30字）"""
    title = first_message.strip()[:30]
    if len(first_message.strip()) > 30:
        title += "..."
    if _db_available:
        try:
            import db as _db_mod
            _db_mod.update_session_title(session_id, title)
        except Exception:
            pass
    else:
        try:
            _local_store.update_local_session_title(session_id, title)
        except Exception:
            pass


def _is_question(text: str) -> bool:
    """
    判断文本是否为真正的疑问句，而非陈述句。

    Qwen-1.8B 小模型容易输出陈述句（如"机器学习有以下特点："），
    此函数用于过滤这类不合格输出。
    """
    text = text.strip()
    if not text:
        return False

    # 必须以问号结尾
    if not (text.endswith('？') or text.endswith('?')):
        return False

    # 必须包含疑问指示词
    question_indicators = [
        '吗', '呢',
        '什么', '怎么', '如何', '为何',
        '哪些', '哪个', '哪种', '哪位',
        '有没有', '能否', '是否', '可否',
        '能不能', '会不会', '可不可以',
        '多少', '几',
        '谁', '哪', '何时', '怎样',
        '可以', '能帮', '推荐', '介绍',
    ]
    has_indicator = any(ind in text for ind in question_indicators)
    if not has_indicator:
        return False

    # 拒绝陈述句式关键词
    statement_patterns = [
        '有以下', '包括以下', '如下',
        '例如', '比如',
        '这是', '以下是', '下面是',
        '区别在于', '不同之处', '特点有',
        '首先', '其次', '然后', '最后',
        '第一', '第二', '第三',
        '步骤', '流程', '方法有',
    ]
    if any(p in text for p in statement_patterns):
        return False

    # 拒绝看起来像列举的开头
    if re.match(r'^[\d]+[\.\、\)）]', text):
        return False

    return True


def _generate_followups(history: list[dict], tokenizer, model, device) -> list[str]:
    """
    根据对话上下文，让模型生成 2-3 个追问建议。

    类似豆包/千问 App 的追问推荐功能。
    使用 few-shot prompt + 问句质量验证 + 模板兜底，适配 1.8B 小模型。
    """
    if not history or len(history) < 2:
        return []

    # ---- Few-shot prompt：强调只输出疑问句，给出正确和错误示例 ----
    system_prompt = (
        "根据对话历史，生成3个用户可能追问的疑问句。\n"
        "严格规则：\n"
        "1. 每个输出必须以 Q: 开头，单独一行\n"
        "2. 每个输出必须是疑问句（以？结尾），严禁输出陈述句\n"
        "3. 不要输出解释、列举、定义等陈述性内容\n"
        "正确示例:\n"
        "Q: 深度学习与机器学习有什么区别？\n"
        "Q: 能推荐一些入门学习资源吗？\n"
        "Q: 这个概念在实际中有哪些应用？\n"
        "错误示例（严禁输出）:\n"
        "Q: 机器学习和深度学习有以下几点区别：\n"
        "Q: 深度学习是机器学习的一个分支\n"
        "Q: 1. 监督学习 2. 无监督学习"
    )
    followup_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    # 只取最近 3 轮对话
    recent = history[-6:]
    for msg in recent:
        followup_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    followup_prompt += "<|im_start|>assistant\n"

    questions = []

    try:
        inputs = tokenizer(followup_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # 解析 Q: 前缀的行，也兼容编号格式
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 匹配 Q: 前缀
            if line.upper().startswith("Q:") or line.upper().startswith("Q：") or line.startswith("问："):
                # 取第一个冒号后的内容
                q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            else:
                # 兼容编号格式: 1. xxx, 1、xxx, 1) xxx
                q = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
            # 长度过滤 + 问句验证：必须通过 _is_question() 检查
            if q and len(q) >= 5 and len(q) <= 80 and _is_question(q):
                questions.append(q)

        # ---- 质量过滤 ----
        # 过滤包含幻觉模型名称的追问（通义千问、ChatGPT、Claude 等）
        hallucination_patterns = [
            "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
            "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
            "百川", "智谱", "ChatGLM", "混元",
        ]
        questions = [
            q for q in questions
            if not any(p in q for p in hallucination_patterns)
        ]

        # 过滤高度重复的追问（如 "通义千问，通义千问，通义千问"）
        filtered = []
        seen_words = set()
        for q in questions:
            # 提取核心关键词
            words = frozenset(q[:10])  # 前 10 个字符作为特征
            if words not in seen_words:
                seen_words.add(words)
                filtered.append(q)
        questions = filtered

        logger.info(f"模型追问生成: {len(questions)} 条 → {questions}")

    except Exception as e:
        logger.warning(f"追问生成失败（非致命）: {e}")
        questions = []

    # ---- 模板兜底：如果模型输出不足 2 条，用规则补足 ----
    if len(questions) < 2:
        fallback = _fallback_followups(history, questions)
        questions = fallback

    return questions[:3]


def _generate_followups_llama(history: list[dict]) -> list[str]:
    """
    使用 llama.cpp 引擎生成追问建议。

    通过 model_manager.chat() 调用（llama.cpp 路径），
    使用简化的 few-shot prompt 适配小模型能力。
    失败时回退到关键词模板兜底。
    """
    if not history or len(history) < 2:
        return []

    # 简化版 prompt：直接要求输出问题，不需要 Q: 前缀格式
    system_prompt = (
        "根据对话内容，生成2-3个你会追问的问题。每个问题一行，以？结尾。"
    )
    followup_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"根据以下对话，生成我想追问的问题：\n"
         f"用户：{history[-2]['content'][:200]}\n"
         f"助手：{history[-1]['content'][:300]}"},
    ]

    questions = []
    try:
        result = model_manager.chat(
            messages=followup_messages,
            max_tokens=128,
            temperature=0.8,
            top_p=0.9,
        )
        text = result.get("content", "").strip()

        # 解析：每行一个追问
        for line in text.split("\n"):
            line = line.strip()
            # 清理编号前缀
            line = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
            # 清理 Q: 前缀
            if line.upper().startswith("Q:") or line.upper().startswith("Q："):
                line = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            if line and len(line) >= 5 and len(line) <= 80 and _is_question(line):
                questions.append(line)

        # 质量过滤（同 _generate_followups）
        hallucination_patterns = [
            "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
            "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
            "百川", "智谱", "ChatGLM", "混元",
        ]
        questions = [q for q in questions if not any(p in q for p in hallucination_patterns)]

        # 去重
        filtered = []
        seen = set()
        for q in questions:
            key = q[:15]
            if key not in seen:
                seen.add(key)
                filtered.append(q)
        questions = filtered

        logger.info(f"llama.cpp 追问生成: {len(questions)} 条 → {questions}")

    except Exception as e:
        logger.warning(f"llama.cpp 追问生成失败（非致命）: {e}")
        questions = []

    # 模板兜底
    if len(questions) < 2:
        fallback = _fallback_followups(history, questions)
        questions = fallback

    return questions[:3]


def _fallback_followups(history: list[dict], existing: list[str]) -> list[str]:
    """
    基于对话关键词匹配的追问模板兜底。

    当 1.8B 小模型无法生成合格追问时启用。
    """
    # 提取最后一轮问答的关键词
    last_assistant = ""
    last_user = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]

    combined = (last_user + " " + last_assistant).lower()

    # 关键词 → 追问模板映射（按优先级排序，更具体的匹配在前）
    templates = []

    if any(kw in combined for kw in ["量化", "quant", "int4", "int8", "fp16", "精度"]):
        templates.extend([
            "INT4和INT8量化在实际应用中如何选择？",
            "量化会对模型推理能力造成多大影响？",
            "除了量化还有哪些模型压缩方法？",
        ])

    if any(kw in combined for kw in ["边缘计算", "边缘", "edge", "分布式", "推理"]):
        templates.extend([
            "边缘推理和云端推理各有什么优缺点？",
            "分布式推理中的通信开销如何优化？",
            "边缘设备的算力瓶颈通常在哪里？",
        ])

    if any(kw in combined for kw in ["python", "代码", "编程", "写一个", "函数", "算法"]):
        templates.extend([
            "这段代码的时间复杂度是多少？",
            "有没有更高效的实现方式？",
            "能解释一下这段代码的核心逻辑吗？",
        ])

    if any(kw in combined for kw in ["模型", "训练", "微调", "lora", "参数"]):
        templates.extend([
            "这个模型的训练数据来源是什么？",
            "如何在特定领域数据上微调模型？",
            "LoRA微调相比全参数微调有哪些优势？",
        ])

    if any(kw in combined for kw in ["transformer", "注意力", "attention", "架构"]):
        templates.extend([
            "Transformer相比RNN有哪些优势？",
            "自注意力机制的计算复杂度如何？",
            "多头注意力的作用是什么？",
        ])

    if any(kw in combined for kw in ["token", "tokenizer", "分词", "词表"]):
        templates.extend([
            "不同的分词方法对模型性能有影响吗？",
            "中文分词和英文分词的主要区别是什么？",
            "BPE分词算法的原理是什么？",
        ])

    if any(kw in combined for kw in ["显存", "gpu", "内存", "oom", "优化", "加速"]):
        templates.extend([
            "还有哪些降低推理显存占用的方法？",
            "CPU推理在什么场景下比GPU更合适？",
            "KV Cache的显存占用如何估算？",
        ])

    if any(kw in combined for kw in ["应用", "场景", "实际", "落地", "工业"]):
        templates.extend([
            "当前这个技术还有哪些落地挑战？",
            "业界有哪些成功的应用案例可以参考？",
            "这项技术的商业化前景如何？",
        ])

    if any(kw in combined for kw in ["hello", "你好", "介绍", "你是谁", "能做什么"]):
        templates.extend([
            "你能帮我写代码吗？",
            "你的知识截止到什么时候？",
            "你擅长哪些类型的任务？",
        ])

    if any(kw in combined for kw in ["学习", "入门", "新手", "教程", "怎么学"]):
        templates.extend([
            "有哪些推荐的学习资源或课程？",
            "学习这个需要什么前置知识？",
            "从入门到精通大概需要多久？",
        ])

    if any(kw in combined for kw in ["区别", "对比", "比较", "不同", "差异", "选择"]):
        templates.extend([
            "在选择时应该考虑哪些关键因素？",
            "有没有具体的场景举例说明？",
            "未来哪个方向更有发展前景？",
        ])

    if any(kw in combined for kw in ["安全", "隐私", "加密", "攻击", "漏洞"]):
        templates.extend([
            "这种攻击的防御措施有哪些？",
            "业界有哪些典型的安全事件？",
            "如何在性能和安全性之间平衡？",
        ])

    if any(kw in combined for kw in ["数据", "dataset", "数据集", "预处理", "清洗"]):
        templates.extend([
            "数据质量对模型效果的影响有多大？",
            "有哪些常用的数据增强方法？",
            "如何处理数据中的类别不平衡问题？",
        ])

    # 默认通用追问（更智能的追问）
    default_templates = [
        "能再详细解释一下吗？",
        "这个结论有什么前提条件或局限性？",
        "有没有相关的参考资料或论文推荐？",
        "实际应用中需要注意哪些细节？",
        "能举一个具体的例子说明吗？",
    ]

    # 选择不重复的追问
    result = list(existing)
    candidate_pool = templates + default_templates
    for q in candidate_pool:
        if q not in result and len(result) < 3:
            result.append(q)

    if len(result) < 2:
        # 不可能到这一步，但也处理一下
        for q in default_templates:
            if q not in result and len(result) < 3:
                result.append(q)

    logger.info(f"追问兜底: 模型生成了 {len(existing)} 条，模板补充至 {len(result)} 条")
    return result


def _init_kv_cache():
    """初始化分页 KV 缓存（根据设备画像自适应大小）"""
    global kv_cache
    num_heads = 16      # Qwen-1.8B: 16 attention heads
    head_dim = 64       # 隐藏维度 2048 / 16 heads = 128, 但实际是 64 per head for K/V
    # 从模型获取实际的 head_dim
    if model_manager.model is not None:
        try:
            cfg = model_manager.model.config
            num_heads = cfg.num_attention_heads
            head_dim = cfg.hidden_size // num_heads
        except Exception:
            pass

    # 优先使用设备画像自适应大小
    if device_profile:
        kv_cache = PagedKVCache.from_profile(
            profile=device_profile,
            device=str(model_manager.get_device()),
            dtype=torch.float16,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        logger.info(
            f"🧠 KV 缓存已初始化 (profile): num_heads={num_heads}, "
            f"head_dim={head_dim}, device={model_manager.get_device()}"
        )
    else:
        kv_cache = PagedKVCache(
            page_size=PAGE_SIZE,
            max_pages=MAX_PAGE_NUM,
            device=str(model_manager.get_device()),
            dtype=torch.float16,
        )
        logger.info(
            f"🧠 KV 缓存已初始化 (default): page_size={PAGE_SIZE}, "
            f"max_pages={MAX_PAGE_NUM}, device={model_manager.get_device()}"
        )
    return kv_cache


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/presets")
async def get_presets():
    """
    返回预设问题列表，包含预估 Token 消耗和显存占用。

    类似豆包/千问 APP 的建议提问功能。
    Token 估算基于 Qwen-1.8B 的经验数据：
      - 中文约 1.5-2 tokens/字
      - 英文约 1-1.3 tokens/字
      - 回复通常为问题的 1-3 倍长度
    """
    # 根据当前加载的量化类型估算速度
    speed_map = {"fp16": 53, "int8": 10, "int4": 29}
    tok_s = speed_map.get(current_quant if model_loaded else "int4", 29)

    # 从设备画像获取档位，调整预估
    max_tokens = generation_config.get("max_new_tokens", 512)

    presets = [
        {
            "id": "intro",
            "icon": "👋",
            "label": "自我介绍",
            "question": "请简单介绍一下你自己，你能做什么？",
            "estimated_prompt_tokens": 25,
            "estimated_response_tokens": 120,
            "estimated_memory_mb": round(145 * 96 / 1024, 1),  # ~13.6 MB KV cache
            "estimated_seconds": round(120 / tok_s, 1),
        },
        {
            "id": "edge_computing",
            "icon": "🌐",
            "label": "边缘计算科普",
            "question": "什么是边缘计算？它和云计算有什么区别？",
            "estimated_prompt_tokens": 35,
            "estimated_response_tokens": 200,
            "estimated_memory_mb": round(235 * 96 / 1024, 1),  # ~22.0 MB
            "estimated_seconds": round(200 / tok_s, 1),
        },
        {
            "id": "model_quantization",
            "icon": "⚡",
            "label": "模型量化原理",
            "question": "大模型的INT4量化是怎么做到的？精度损失大吗？",
            "estimated_prompt_tokens": 40,
            "estimated_response_tokens": 250,
            "estimated_memory_mb": round(290 * 96 / 1024, 1),  # ~27.2 MB
            "estimated_seconds": round(250 / tok_s, 1),
        },
        {
            "id": "code_assist",
            "icon": "💻",
            "label": "Python 代码助手",
            "question": "用Python写一个函数，计算两个大文件的MD5哈希并比较是否相同",
            "estimated_prompt_tokens": 45,
            "estimated_response_tokens": 300,
            "estimated_memory_mb": round(345 * 96 / 1024, 1),  # ~32.3 MB
            "estimated_seconds": round(300 / tok_s, 1),
        },
        {
            "id": "creative",
            "icon": "✨",
            "label": "创意写作",
            "question": "以「边缘设备上的AI觉醒」为题，写一个300字的科幻微小说",
            "estimated_prompt_tokens": 50,
            "estimated_response_tokens": 400,
            "estimated_memory_mb": round(450 * 96 / 1024, 1),  # ~42.2 MB
            "estimated_seconds": round(400 / tok_s, 1),
        },
        {
            "id": "reasoning",
            "icon": "🧩",
            "label": "逻辑推理",
            "question": "A说B撒谎，B说C撒谎，C说A和B都在撒谎。请问谁说的是真话？",
            "estimated_prompt_tokens": 55,
            "estimated_response_tokens": 350,
            "estimated_memory_mb": round(405 * 96 / 1024, 1),  # ~38.0 MB
            "estimated_seconds": round(350 / tok_s, 1),
        },
    ]

    return {
        "presets": presets,
        "current_speed_tok_s": tok_s,
        "current_quant": current_quant if model_loaded else None,
        "max_new_tokens": max_tokens,
    }


# ---- 支持的文件类型 ----
ALLOWED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".py", ".json", ".log",
    ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".sh", ".bash", ".zsh", ".ps1",
    ".cpp", ".c", ".h", ".java", ".go", ".rs", ".rb",
    ".sql", ".r", ".m", ".swift", ".kt",
    ".toml", ".properties", ".env",
}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_UPLOAD_LINES = 5000              # 超过截断


@app.post("/api/chat/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文本文件，返回解析后的内容。

    支持 txt / md / csv / py / json / log 等纯文本格式。
    限制 5 MB，超过 5000 行自动截断（保留前 5000 行）。
    """
    import os as _os

    # 1. 校验扩展名
    filename = file.filename or "untitled"
    ext = _os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_TEXT_EXTENSIONS:
        raise HTTPException(
            400,
            f"不支持的文件类型: {ext}。"
            f"支持的格式: {', '.join(sorted(ALLOWED_TEXT_EXTENSIONS))}",
        )

    # 2. 读取内容
    try:
        raw = await file.read()
    except Exception as e:
        raise HTTPException(400, f"文件读取失败: {e}")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"文件过大 ({len(raw) / 1024 / 1024:.1f} MB)，"
            f"限制 {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
        )

    # 3. 解码（尝试 UTF-8 → GBK → latin-1）
    content = None
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise HTTPException(400, "无法解码文件内容，请确认文件编码为 UTF-8 或 GBK")

    # 4. 统计 + 截断
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines > MAX_UPLOAD_LINES:
        content = "\n".join(lines[:MAX_UPLOAD_LINES])
        truncated = True
    else:
        truncated = False

    # 统计字符数和词数近似值
    char_count = len(content)
    word_count = len(content.split())

    # 检测语言类型（用于前端代码高亮）
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html", ".css": "css",
        ".json": "json", ".md": "markdown", ".csv": "csv",
        ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
        ".cpp": "cpp", ".c": "c", ".h": "c", ".java": "java",
        ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".sql": "sql", ".r": "r", ".swift": "swift", ".kt": "kotlin",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    }
    language = lang_map.get(ext, "plaintext")

    logger.info(
        f"文件上传: {filename} ({ext}) {char_count} 字符 "
        f"{total_lines} 行{' (已截断)' if truncated else ''}"
    )

    return {
        "filename": filename,
        "extension": ext,
        "language": language,
        "char_count": char_count,
        "word_count": word_count,
        "line_count": total_lines if not truncated else MAX_UPLOAD_LINES,
        "total_lines": total_lines,
        "truncated": truncated,
        "truncated_lines": total_lines - MAX_UPLOAD_LINES if truncated else 0,
        "size_bytes": len(raw),
        "content": content,
    }


@app.get("/api/device/profile")
async def get_device_profile():
    """
    获取完整设备画像。

    包含 CPU / RAM / GPU / 磁盘 / OS 信息，
    设备档位、评分、推荐配置、警告。
    启动时自动检测一次，后续请求返回缓存。
    """
    global device_profile
    if device_profile is None:
        import asyncio

        await asyncio.to_thread(_device_profile_ready.wait, 15)
        if device_profile is None:
            raise HTTPException(503, "设备画像仍在检测中，请稍后重试")
    return device_profile


@app.post("/api/device/auto-configure")
async def auto_configure():
    """
    根据设备画像自动应用推荐配置。

    更新 KV 缓存大小、序列长度、生成参数等运行时配置。
    不重新加载模型（如需切换量化精度，请手动调用 /api/models/load）。
    """
    global kv_cache, device_profile, generation_config

    if device_profile is None:
        try:
            profiler = get_profile()
            device_profile = profiler.to_dict()
        except Exception as e:
            raise HTTPException(500, f"设备检测失败: {e}")
    scheduler.update_local_device_profile(device_profile)

    rec = device_profile.get("recommendations", [])
    warnings = device_profile.get("warnings", [])
    tier = device_profile.get("tier", "laptop")
    score = device_profile.get("score_total", 50)

    # 从 device_profiler 获取推荐配置
    from device_profiler import DeviceProfiler
    profiler = get_profile()
    config = profiler.recommend_config()

    # 应用 KV 缓存配置（如果尚未加载模型，则更新默认值）
    import config as cfg
    cfg.PAGE_SIZE = config["page_size"]
    cfg.MAX_PAGE_NUM = config["max_pages"]
    cfg.MAX_SEQ_LEN = config["max_seq_len"]

    # 更新生成配置（设置档位上限）
    generation_config["max_new_tokens"] = config["max_new_tokens"]
    generation_config["tier_max_new_tokens"] = config["max_new_tokens"]

    # 如果 KV 缓存已存在，重建
    if kv_cache and model_loaded:
        kv_cache.clear()
        from paged_kv_cache import PagedKVCache
        kv_cache = PagedKVCache(
            page_size=config["page_size"],
            max_pages=config["max_pages"],
            device=kv_cache.device,
            dtype=kv_cache.dtype,
        )
        logger.info(
            f"KV 缓存已重建: page_size={config['page_size']}, "
            f"max_pages={config['max_pages']}"
        )

    logger.info(f"自适应配置已应用: {config['description']}")

    return {
        "status": "configured",
        "tier": tier,
        "score": score,
        "applied_config": config,
        "recommendations": rec,
        "warnings": warnings,
    }


class SelectGpuRequest(BaseModel):
    gpu_index: int = Field(..., ge=0, description="GPU 列表中要切换到的序号")


@app.post("/api/device/select-gpu")
async def select_gpu(req: SelectGpuRequest):
    """
    切换推理 GPU。

    在集显（CPU 推理）和独显（CUDA）之间切换。
    切换后需要重新加载模型才能生效。

    游戏本默认使用独显（CUDA 加速），用户可手动切换到集显（低功耗）。
    """
    global device_profile, model_loaded

    if device_profile is None:
        raise HTTPException(400, "设备画像未就绪，请先调用 GET /api/device/profile")

    gpus = device_profile.get("gpus", [])
    if req.gpu_index < 0 or req.gpu_index >= len(gpus):
        raise HTTPException(
            400,
            f"无效的 GPU 序号: {req.gpu_index}。"
            f"可用范围: 0-{len(gpus) - 1}（共 {len(gpus)} 个 GPU）",
        )

    # 更新 profiler 中的选中 GPU
    from device_profiler import get_profile
    profiler = get_profile()
    if not profiler.select_gpu(req.gpu_index):
        raise HTTPException(500, "GPU 切换失败")

    # 更新缓存的 device_profile
    device_profile = profiler.to_dict()
    scheduler.update_local_device_profile(device_profile)

    selected = gpus[req.gpu_index]
    logger.info(
        f"GPU 已切换: [{req.gpu_index}] {selected['name']} "
        f"({selected['gpu_type']}, CUDA: {selected['cuda_available']})"
    )

    return {
        "status": "switched",
        "selected_gpu_index": req.gpu_index,
        "selected_gpu": {
            "name": selected["name"],
            "gpu_type": selected["gpu_type"],
            "cuda_available": selected["cuda_available"],
            "vram_total_gb": selected["vram_total_gb"],
        },
        "device": profiler.recommend_config()["device"],
        "warning": (
            "切换 GPU 后需要重新加载模型才能生效。"
            if model_loaded
            else None
        ),
    }


@app.get("/api/status")
async def get_status():
    """获取系统完整状态（含设备档位）"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2)),
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
            "utilization": round(
                torch.cuda.memory_allocated()
                / torch.cuda.get_device_properties(0).total_memory
                * 100,
                1,
            ),
        }

    # ---- KV 缓存统计（基于实际对话 token 消耗估算） ----
    # 注：单机模式下 model.generate() 使用内置 KV 缓存，PagedKVCache 未接入。
    # 这里根据实际对话 token 数估算 KV 缓存显存占用。
    num_heads = 16
    head_dim = 64
    num_layers = 24   # Qwen-1.8B
    dtype_bytes = 2   # fp16/bf16
    total_tokens = conversation_stats["total_prompt_tokens"] + conversation_stats["total_generated_tokens"]
    # KV cache per token = num_layers × 2(K+V) × num_heads × head_dim × dtype_bytes
    kv_bytes_per_token = num_layers * 2 * num_heads * head_dim * dtype_bytes
    kv_memory_mb = round(total_tokens * kv_bytes_per_token / (1024 ** 2), 2)

    # 已分配页估算（以当前 PAGE_SIZE 为基准）
    page_size = PAGE_SIZE
    estimated_pages = (total_tokens + page_size - 1) // page_size if total_tokens > 0 else 0
    max_pages = MAX_PAGE_NUM
    utilization = estimated_pages / max_pages if max_pages > 0 else 0.0

    kv_stats = {
        "total_tokens": total_tokens,
        "max_tokens": page_size * max_pages,
        "allocated_pages": estimated_pages,
        "free_pages": max_pages - estimated_pages,
        "max_pages": max_pages,
        "page_size": page_size,
        "utilization": round(utilization, 4),
        "estimated_memory_mb": kv_memory_mb,
        "rounds": conversation_stats["rounds"],
        "total_time_s": round(conversation_stats["total_time_seconds"], 1),
    }

    # 设备画像摘要
    device_summary = None
    if device_profile:
        device_summary = {
            "tier": device_profile.get("tier"),
            "tier_label": device_profile.get("tier_label"),
            "tier_icon": device_profile.get("tier_icon"),
            "score": device_profile.get("score_total"),
            "gpus": device_profile.get("gpus", []),
            "selected_gpu_index": device_profile.get("selected_gpu_index", 0),
            "recommendations": device_profile.get("recommendations", [])[:3],
            "warnings": device_profile.get("warnings", []),
        }

    active_info = {}
    if model_loaded and model_manager.is_loaded:
        try:
            active_info = model_manager.get_model_info()
        except Exception:
            active_info = {}

    return {
        "model_loaded": model_loaded,
        "current_quant": current_quant,
        "use_compile": USE_COMPILE if model_loaded else False,
        "model_name": active_info.get("model_name", MODEL_NAME),
        "model_path": active_info.get("model_path", MODEL_PATH),
        "active_model_id": active_info.get("model_id", model_manager.active_model_id if model_loaded else None),
        "run_mode": RUN_MODE,
        "node_role": scheduler._effective_role(),
        "node_id": scheduler.get_effective_node_id(),
        "max_nodes": scheduler._max_nodes,
        "gpu": gpu_info,
        "kv_cache": kv_stats,
        "conversation_turns": len(_get_active_history()),
        "generation_config": generation_config,
        "device": device_summary,
    }


@app.get("/api/models/current")
async def get_current_model():
    """当前模型信息"""
    if not model_loaded:
        return {"loaded": False, "quant_type": None, "model_id": None}

    info = model_manager.get_model_info()
    mem = model_manager.get_memory_usage()
    return {
        "loaded": True,
        "model_id": model_manager.active_model_id,
        "quant_type": current_quant,
        "model_name": info.get("model_name", MODEL_NAME),
        "model_path": info.get("model_path", ""),
        "engine": info.get("engine", ""),
        "total_params": info.get("total_params", "N/A"),
        "device": info.get("device", "N/A"),
        "gpu_allocated_gb": mem.get("gpu_allocated_gb", 0),
        "gpu_reserved_gb": mem.get("gpu_reserved_gb", 0),
    }


@app.post("/api/models/load")
async def load_model(req: LoadModelRequest):
    """
    加载/切换模型。

    耗时约 5-20 秒（取决于量化类型），期间会先卸载旧模型。
    使用 switch_model 获得失败时自动回滚到上一个模型的保护。
    """
    global model_loaded, current_quant, kv_cache, conversation_stats

    engine = req.engine.lower()
    if engine not in ("auto", "llama_cpp", "pytorch"):
        raise HTTPException(400, f"不支持的引擎: {engine}，可选: auto, llama_cpp, pytorch")

    _validate_model_load_request(req.model_id, engine)
    resolved_model_path = _resolve_model_path_for_engine(req.model_id, engine)
    effective_engine = _effective_engine_for_model(req.model_id, engine)
    quant = _normalize_quant_for_engine(req.quant_type, effective_engine)

    try:
        t0 = time.time()

        # 临时修改 config（引擎 + 量化 + compile）
        import config as cfg
        cfg.INFERENCE_ENGINE = effective_engine
        cfg.QUANT_TYPE = quant
        cfg.USE_COMPILE = req.use_compile

        # 清空内存会话/KV/统计（新模型不能复用旧模型的上下文）
        _reset_runtime_conversation_state(clear_histories=True)
        # 同步清空数据库对话历史
        if _db_available:
            try:
                import db as _db_mod
                _db_mod.clear_conversation("default")
            except Exception:
                pass

        # P3修复: 使用 switch_model 获得失败时自动回滚保护
        logger.info(f"加载模型: engine={effective_engine}, quant={quant}, compile={req.use_compile}")
        result = _run_exclusive_model_change(
            lambda: model_manager.switch_model(
                model_id=req.model_id or mc.DEFAULT_MODEL_ID,
                quant_type=quant,
                profile=device_profile,
                engine=effective_engine if effective_engine != "auto" else None,
                model_path=resolved_model_path,
                db_experimental_models=_get_db_experimental_models(),
            )
        )

        if result["success"]:
            model_loaded = True
            current_quant = quant
            generation_config["use_compile"] = req.use_compile

            # 初始化 KV 缓存
            _init_kv_cache()
            elapsed = time.time() - t0
            status = await get_status()
            status["load_time_seconds"] = round(elapsed, 1)
            status["model_name"] = result.get("model_name", "")

            logger.info(f"模型加载完成 ({elapsed:.1f}s): {quant}")
            return status
        else:
            # 切换失败 — 检查是否回滚成功
            if model_manager.is_loaded:
                model_loaded = True
                current_quant = model_manager.quant_type or (
                    "gguf" if model_manager._engine_type == "llama_cpp" else QUANT_TYPE
                )
                _init_kv_cache()
            else:
                model_loaded = False
                current_quant = QUANT_TYPE
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        model_loaded = False
        logger.error(f"模型加载失败: {e}", exc_info=True)
        raise HTTPException(500, f"模型加载失败: {str(e)}")


def _chat_origin(req: ChatRequest) -> str:
    """根据请求上报信息推断请求来源，用于 metrics 展示。"""
    if req.client_node_type == "android":
        return "android_http"
    if req.client_node_type == "pc":
        return "pc_http"
    return "web_http"


def _augment_chat_metrics(metrics: dict | None, req: ChatRequest, **defaults) -> dict:
    """补齐统一聊天 metrics 字段，不覆盖调度器已给出的真实执行信息。"""
    result = dict(metrics or {})
    for key, value in defaults.items():
        result.setdefault(key, value)
    origin = _chat_origin(req)
    result.setdefault("request_origin", origin)
    result.setdefault("request_origin_node_id", req.client_node_id or "")
    result.setdefault("request_origin_node_type", req.client_node_type or "")
    result.setdefault("client_mode", req.client_mode or "")
    result.setdefault("client_app_variant", req.client_app_variant or "")
    result.setdefault("serving_node_id", scheduler.get_effective_node_id())
    result.setdefault("distributed_requested", scheduler.get_distributed_inference_enabled())
    result.setdefault("distributed_used", False)
    result.setdefault("fallback", False)
    result.setdefault("fallback_reason", "")
    result.setdefault("workers_used", [])
    result.setdefault("layer_assignments", [])
    result.setdefault("request_id", _request_id_ctx.get("-"))
    return result


def _execute_chat_full(req: ChatRequest) -> dict:
    """
    执行完整聊天流程 — 从 /api/chat 提取的共用核心逻辑。

    处理: 会话切换、自动标题、客户端转发、流水线推理、
          llama.cpp、PyTorch、历史维护、DB 持久化、追问生成。

    Returns:
        {"content": str, "thinking_content": str|None,
         "metrics": dict, "followups": list[str]}

    Raises:
        HTTPException: 模型未加载、OOM、推理失败
    """
    global kv_cache, conversation_stats

    # ---- 多会话支持 ----
    target_session_id = req.session_id or active_session_id
    if target_session_id and target_session_id != active_session_id:
        _switch_session(target_session_id)

    # ---- 首条消息自动生成标题 ----
    history = _get_active_history()
    if target_session_id and len(history) == 0:
        _auto_title_session(target_session_id, req.message)

    # ---- 分布式推理路由：从节点转发给主节点 ----
    if (scheduler.get_distributed_inference_enabled()
            and RUN_MODE == "distributed"
            and scheduler._effective_role() == "client"):
        try:
            result = scheduler.forward_inference_to_master(
                message=req.message,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                show_thinking=req.show_thinking,
                session_id=req.session_id,
                messages=list(history) + [{"role": "user", "content": req.message}],
                request_id=_request_id_ctx.get("-"),   # L5: 链路追踪
            )
            if result.get("status") == "ok":
                history.append({"role": "user", "content": req.message})
                response_text = result.get("content", "")
                history.append({"role": "assistant", "content": response_text})
                forward_metrics = _augment_chat_metrics(
                    result.get("metrics", {}),
                    req,
                    engine="distributed_forward",
                    execution_mode="forwarded_to_master",
                    route="pc_client_forward_to_master",
                )

                db_session_id = target_session_id or "default"
                if _db_available:
                    try:
                        import db as _db_mod
                        if _db_mod.get_save_history():
                            _db_mod.save_message(db_session_id, "user", req.message)
                            _db_mod.save_message(db_session_id, "assistant", response_text,
                                                forward_metrics)
                            _db_mod.increment_session_message_count(db_session_id)
                    except Exception:
                        pass
                if not _db_available:
                    try:
                        _local_store.save_local_message(db_session_id, "user", req.message)
                        _local_store.save_local_message(db_session_id, "assistant", response_text,
                                                        forward_metrics)
                        _local_store.increment_local_session_message_count(db_session_id)
                    except Exception:
                        pass

                conversation_stats["rounds"] += 1
                try:
                    scheduler.record_task_complete(success=True)
                except Exception:
                    pass

                master_followups = result.get("followups", [])
                if master_followups:
                    followups = master_followups[:3]
                else:
                    followups = _fallback_followups(history, [])

                return {
                    "content": response_text,
                    "thinking_content": result.get("thinking_content"),
                    "metrics": forward_metrics,
                    "followups": followups,
                }
            elif result.get("status") == "disconnected":
                logger.warning("分布式推理转发失败（未连接主节点），回退到本地推理")
            elif result.get("status") == "timeout":
                logger.warning("分布式推理转发超时，回退到本地推理")
            else:
                logger.warning(f"分布式推理转发失败: {result.get('error', 'unknown')}，回退到本地推理")
        except Exception as e:
            logger.warning(f"分布式推理转发异常: {e}，回退到本地推理")

    # ---- 分布式流水线推理路径（主节点 + PyTorch 引擎 + 从节点可用）----
    if (scheduler.get_distributed_inference_enabled()
            and RUN_MODE == "distributed"
            and scheduler._effective_role() == "master"
            and model_manager._engine_type == "pytorch"):
        try:
            pipeline_result = scheduler.run_pipeline_safe(
                req.message,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                session_id=req.session_id,
                messages=list(history) + [{"role": "user", "content": req.message}],
                show_thinking=req.show_thinking,
            )
            if pipeline_result.get("error"):
                logger.warning(f"流水线推理失败: {pipeline_result['error']}，回退到本地推理")
            else:
                response_text = pipeline_result.get("response", "")
                if not response_text:
                    logger.warning("流水线返回空响应，回退到本地推理")
                else:
                    history.append({"role": "user", "content": req.message})
                    history.append({"role": "assistant", "content": response_text})

                    db_session_id = target_session_id or "default"
                    pipeline_metrics = _augment_chat_metrics(
                        pipeline_result.get("metrics", {}),
                        req,
                        engine="distributed_pipeline",
                        execution_mode="distributed_pipeline",
                        route="master_pipeline",
                    )
                    if _db_available:
                        try:
                            import db as _db_mod
                            if _db_mod.get_save_history():
                                _db_mod.save_message(db_session_id, "user", req.message)
                                _db_mod.save_message(db_session_id, "assistant", response_text,
                                                    pipeline_metrics)
                                _db_mod.increment_session_message_count(db_session_id)
                        except Exception:
                            pass
                    if not _db_available:
                        try:
                            _local_store.save_local_message(db_session_id, "user", req.message)
                            _local_store.save_local_message(db_session_id, "assistant",
                                                            response_text,
                                                            pipeline_metrics)
                            _local_store.increment_local_session_message_count(db_session_id)
                        except Exception:
                            pass

                    conversation_stats["rounds"] += 1
                    if not pipeline_metrics.get("distributed_used"):
                        try:
                            scheduler.record_task_complete(success=True)
                        except Exception:
                            pass

                    if model_manager._engine_type == "llama_cpp":
                        followups = _generate_followups_llama(history)
                    elif model_manager._engine_type == "pytorch":
                        # 流水线成功后主节点仍保留首段裁剪模型，不能拿它生成追问。
                        followups = _fallback_followups(history, [])
                    else:
                        followups = _fallback_followups(history, [])

                    return {
                        "content": response_text,
                        "thinking_content": pipeline_result.get("thinking"),
                        "metrics": pipeline_metrics,
                        "followups": followups,
                    }
        except Exception as e:
            logger.warning(f"流水线推理异常: {e}，回退到本地推理")

    # ---- llama.cpp 引擎路径（CPU/集显，GGUF）----
    if model_manager._engine_type == "llama_cpp":
        try:
            history.append({"role": "user", "content": req.message})
            result = model_manager.chat(
                messages=list(history),
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            response_text = result.get("content", "")
            # P3修复: llama.cpp 路径同样需要剥离本地思考标记
            if not req.show_thinking:
                response_text = _strip_native_thinking_tags(response_text)
            history.append({"role": "assistant", "content": response_text})
            tokens_per_sec = result.get("tokens_per_second", 0)
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            local_route = f"{_chat_origin(req)}_to_master_local_llama_cpp"
            fallback_reason = ""
            if scheduler.get_distributed_inference_enabled() and RUN_MODE == "distributed":
                fallback_reason = "llama.cpp engine does not support layer-split pipeline"
            metrics = _augment_chat_metrics(
                {
                    "engine": "llama_cpp",
                    "execution_mode": "local_llama_cpp",
                    "route": local_route,
                    "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                    "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                    "generated_tokens": completion_tokens,
                    "completion_tokens": completion_tokens,
                    "usage": usage,
                    "fallback": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                },
                req,
            )

            db_session_id = target_session_id or "default"
            followups = _generate_followups_llama(history)

            if _db_available:
                try:
                    import db as _db_mod
                    if _db_mod.get_save_history():
                        _db_mod.save_message(db_session_id, "user", req.message)
                        save_metrics = dict(metrics)
                        save_metrics["followups"] = followups
                        _db_mod.save_message(db_session_id, "assistant", response_text,
                                            save_metrics)
                        _db_mod.increment_session_message_count(db_session_id)
                except Exception:
                    pass
            if not _db_available:
                try:
                    _local_store.save_local_message(db_session_id, "user", req.message)
                    save_metrics = dict(metrics)
                    save_metrics["followups"] = followups
                    _local_store.save_local_message(db_session_id, "assistant", response_text,
                                                    save_metrics)
                    _local_store.increment_local_session_message_count(db_session_id)
                except Exception:
                    pass

            conversation_stats["total_generated_tokens"] += completion_tokens
            conversation_stats["rounds"] += 1
            try:
                scheduler.record_task_complete(success=True)
            except Exception:
                pass

            return {
                "content": response_text,
                "thinking_content": None,
                "metrics": metrics,
                "followups": followups,
            }
        except Exception as e:
            try:
                scheduler.record_task_error()
            except Exception:
                pass
            logger.error(f"llama.cpp 推理失败: {e}", exc_info=True)
            raise HTTPException(500, f"推理失败: {str(e)}")

    # ---- PyTorch 引擎路径（CUDA/独显）----
    try:
        model_manager.ensure_full_model()
        tier_max = generation_config.get("tier_max_new_tokens", generation_config["max_new_tokens"])
        thinking_budget = 384 if req.show_thinking else 0
        effective_max = min(req.max_new_tokens + thinking_budget,
                            tier_max + thinking_budget,
                            4096)
        generation_config["max_new_tokens"] = effective_max
        generation_config["temperature"] = req.temperature
        generation_config["top_p"] = req.top_p

        history.append({"role": "user", "content": req.message})

        tokenizer = model_manager.tokenizer
        thinking_prompt = THINKING_SYSTEM_PROMPT if req.show_thinking else None
        thinking_prefill = "【思考】\n" if req.show_thinking else None
        prompt = _build_model_chat_prompt(
            tokenizer,
            history,
            system_prompt=thinking_prompt,
            assistant_prefill=thinking_prefill,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_manager.get_device())
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_manager.get_device())
        prompt_len = input_ids.shape[1]
        stop_sequences = model_manager._merge_stop_sequences(None)
        generation_kwargs = {}
        eos_token_ids = model_manager._get_generation_eos_token_ids(stop_sequences)
        if eos_token_ids is not None:
            generation_kwargs["eos_token_id"] = eos_token_ids
        stop_criteria = model_manager._build_stop_criteria(stop_sequences, prompt_len)
        if stop_criteria is not None:
            generation_kwargs["stopping_criteria"] = stop_criteria

        t0 = time.time()
        with torch.no_grad():
            outputs = model_manager.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=effective_max,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_p=req.top_p,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        elapsed = time.time() - t0

        generated_ids = outputs[0][prompt_len:]
        raw_text = model_manager._decode_generated_ids(generated_ids, stop_sequences).strip()

        native_thinking_prompt = "<think>" in prompt[-128:].lower()
        parsed_text = raw_text
        if req.show_thinking and not native_thinking_prompt and "<think" not in raw_text.lower():
            parsed_text = "【思考】\n" + raw_text
        response_text, thinking_content = _format_model_response(
            parsed_text,
            req.show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )

        history.append({"role": "assistant", "content": response_text})

        new_tokens = len(generated_ids)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        metrics = _augment_chat_metrics(
            {
                "engine": "pytorch",
                "execution_mode": "local_pytorch",
                "route": f"{_chat_origin(req)}_to_master_local_pytorch",
                "prompt_tokens": prompt_len,
                "new_tokens": new_tokens,
                "generated_tokens": new_tokens,
                "total_tokens": prompt_len + new_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "tokens_per_second": round(tokens_per_sec, 1),
                "gpu_memory_mb": round(torch.cuda.memory_allocated() / (1024**2), 1)
                if torch.cuda.is_available()
                else 0,
            },
            req,
        )

        db_session_id = target_session_id or "default"

        followups = _generate_followups(
            history, tokenizer, model_manager.model, model_manager.get_device()
        )

        if _db_available:
            try:
                import db as _db_mod
                if _db_mod.get_save_history():
                    _db_mod.save_message(db_session_id, "user", req.message)
                    save_metrics = dict(metrics)
                    save_metrics["followups"] = followups
                    _db_mod.save_message(db_session_id, "assistant", response_text, save_metrics)
                    _db_mod.increment_session_message_count(db_session_id)
            except Exception:
                pass
        if not _db_available:
            try:
                save_metrics = dict(metrics)
                save_metrics["followups"] = followups
                _local_store.save_local_message(db_session_id, "user", req.message)
                _local_store.save_local_message(db_session_id, "assistant", response_text, save_metrics)
                _local_store.increment_local_session_message_count(db_session_id)
            except Exception:
                pass

        conversation_stats["total_prompt_tokens"] += prompt_len
        conversation_stats["total_generated_tokens"] += new_tokens
        conversation_stats["total_time_seconds"] += elapsed
        conversation_stats["rounds"] += 1
        try:
            scheduler.record_task_complete(success=True)
        except Exception:
            pass

        logger.info(
            f"推理完成: {new_tokens} tokens / {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s"
        )

        return {
            "content": response_text,
            "thinking_content": thinking_content,
            "metrics": metrics,
            "followups": followups,
        }

    except torch.cuda.OutOfMemoryError:
        try:
            scheduler.record_task_error()
        except Exception:
            pass
        if kv_cache:
            kv_cache.clear()
        _get_active_history().clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(507, "GPU 显存不足（OOM），已自动清空对话历史。请缩短消息后重试。")

    except Exception as e:
        try:
            scheduler.record_task_error()
        except Exception:
            pass
        logger.error(f"推理异常: {e}", exc_info=True)
        raise HTTPException(500, f"推理失败: {str(e)}")


def _auto_load_default_model():
    """自动加载默认模型（用于 thin client / Android 首次请求时服务端无模型的情况）。"""
    global model_loaded, current_quant, kv_cache, conversation_stats

    import config as cfg
    import glob

    # 1. 优先查找 GGUF 文件（llama.cpp 引擎，不依赖 transformers/bitsandbytes）
    gguf_candidates = []
    gguf_configured = cfg.GGUF_MODEL_PATH
    if os.path.isfile(gguf_configured):
        gguf_candidates.append(gguf_configured)
    # 搜索 models 目录下的所有 .gguf 文件
    models_dir = os.path.dirname(gguf_configured)
    if os.path.isdir(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.gguf"))):
            if f not in gguf_candidates:
                gguf_candidates.append(f)

    if gguf_candidates:
        gguf_path = gguf_candidates[0]
        engine = "llama_cpp"
        model_path = gguf_path
        quant = "int4"
        if len(gguf_candidates) > 1:
            logger.info(f"发现 {len(gguf_candidates)} 个 GGUF 文件，选择: {os.path.basename(gguf_path)}")
    elif os.path.isdir(cfg.MODEL_PATH):
        # 2. 回退：Safetensors 目录必须使用 PyTorch 后端。
        engine = "pytorch"
        model_path = cfg.MODEL_PATH
        quant = cfg.QUANT_TYPE
    else:
        raise FileNotFoundError(
            f"未找到可自动加载的模型文件。已检查:\n"
            f"  GGUF 配置路径: {gguf_configured}\n"
            f"  Safetensors 路径: {cfg.MODEL_PATH}\n"
            f"  models 目录: {models_dir}"
        )

    logger.info(f"自动加载默认模型: path={model_path}, engine={engine}")

    t0 = time.time()
    cfg.INFERENCE_ENGINE = engine
    cfg.QUANT_TYPE = quant
    cfg.USE_COMPILE = False

    _run_exclusive_model_change(
        lambda: model_manager.load_model(
            model_path=model_path,
            quant_type=quant,
            profile=device_profile,
            engine=engine,
        )
    )

    _init_kv_cache()
    conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    model_loaded = True
    current_quant = quant
    elapsed = time.time() - t0
    logger.info(f"默认模型自动加载完成 ({elapsed:.1f}s)")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    发送消息并获取模型回复（多轮对话）。

    自动维护对话历史 + KV 缓存。
    若模型未加载，自动尝试加载默认模型。
    """
    if not model_loaded or not model_manager.is_loaded:
        try:
            _auto_load_default_model()
        except FileNotFoundError:
            raise HTTPException(400, "模型未加载且未找到可自动加载的模型文件。请先在控制面板中加载模型。")
        except Exception as e:
            raise HTTPException(500, f"自动加载模型失败: {e}。请手动在控制面板中加载模型。")

    result = _execute_chat_full(req)
    return ChatResponse(
        content=result["content"],
        thinking_content=result.get("thinking_content"),
        metrics=result["metrics"],
        followups=result["followups"],
    )


# ================================================================
# SSE 流式输出
# ================================================================

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    流式聊天端点 (Server-Sent Events)。

    支持两种模式（通过 streaming_mode 参数切换）:

    fast（默认）— 真流式，逐 token 推送:
      - 路径1: 分布式流水线 → 逐 token SSE
      - 路径2: 单机 PyTorch → 逐 token SSE（TextIteratorStreamer）
      - 路径3: llama.cpp / 其他 → 假流式回退（单次 done 事件）
      - 注意: fast 模式跳过了历史/追问/DB持久化，专注低延迟

    full — 假流式，完整功能:
      - 走 /api/chat 全流程：会话管理、对话历史、追问生成、DB 持久化
      - 推理完成后一次性返回单个 done 事件（SSE 格式）
      - 功能与 /api/chat 完全一致，仅响应格式不同

    事件格式:
        data: {"token": "你"}
        data: {"done": true, "response": "...", "followups": [...], "metrics": {...}}
    """
    import json as _json
    request_id = _request_id_ctx.get("-")

    async def _generate():
        token = _request_id_ctx.set(request_id)
        try:
            async for chunk in _generate_events():
                yield chunk
        finally:
            _request_id_ctx.reset(token)

    async def _run_with_request_id(loop, func):
        def _runner():
            token = _request_id_ctx.set(request_id)
            try:
                return func()
            finally:
                _request_id_ctx.reset(token)

        return await loop.run_in_executor(None, _runner)

    def _error_event(message: str) -> str:
        return f"data: {_json.dumps({'done': True, 'error': message, 'request_id': request_id}, ensure_ascii=False)}\n\n"

    async def _generate_events():
        # ================================================================
        # ★ full 模式：完整 chat 流程，假流式（SSE 单事件）
        # ================================================================
        if req.streaming_mode == "full":
            if not model_loaded or not model_manager.is_loaded:
                try:
                    _auto_load_default_model()
                except Exception as e:
                    logger.error(f"full 模式自动加载模型失败: {e}", exc_info=True)
                    yield _error_event(f"自动加载模型失败: {e}")
                    return
            import asyncio
            loop = asyncio.get_event_loop()
            try:
                result = await _run_with_request_id(loop, lambda: _execute_chat_full(req))
                yield f"data: {_json.dumps({
                    'done': True,
                    'response': result['content'],
                    'thinking_content': result.get('thinking_content'),
                    'followups': result['followups'],
                    'metrics': result['metrics'],
                    'request_id': request_id,
                }, ensure_ascii=False)}\n\n"
            except HTTPException as e:
                yield _error_event(e.detail)
            except Exception as e:
                logger.error(f"full 模式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))
            return

        # ================================================================
        # fast 模式：真流式，跳过历史/追问/DB 持久化（低延迟）
        # ================================================================
        # ---- 路径 1: 分布式流水线流式 ----
        if (scheduler.get_distributed_inference_enabled()
                and RUN_MODE == "distributed"
                and scheduler._effective_role() == "master"
                and model_manager._engine_type == "pytorch"):
            try:
                for event in scheduler.run_pipeline_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=[{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                ):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))

        # ---- 路径 2: 单机 PyTorch 流式 ----
        elif (model_manager._engine_type == "pytorch"
                and model_manager.is_loaded):
            try:
                for event in scheduler._run_full_model_inference_stream(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=[{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                ):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"单机流式推理失败: {e}", exc_info=True)
                yield _error_event(str(e))

        # ---- 路径 3: llama.cpp / 从节点 / 模型未加载 → 假流式回退 ----
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            result = await _run_with_request_id(
                loop,
                lambda: scheduler.run_pipeline_safe(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                )
            )
            # 一次性返回完整结果（SSE 格式，单事件）
            metrics = result.get('metrics', {}) or {}
            metrics.setdefault("request_id", request_id)
            yield f"data: {_json.dumps({'done': True, 'response': result.get('response', ''), 'error': result.get('error'), 'metrics': metrics, 'request_id': request_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


@app.post("/api/chat/clear")
async def clear_chat():
    """清空当前活跃会话的对话历史与 KV 缓存"""
    global kv_cache, conversation_stats
    _get_active_history().clear()
    conversation_stats = {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    if kv_cache:
        kv_cache.clear()
    _init_kv_cache()
    logger.info("对话历史已清空")
    return {"status": "cleared", "conversation_turns": 0}


@app.get("/api/models/available")
async def list_available_models():
    """列出可选模型配置 + 可用引擎"""
    # 检测所有已注册模型（内置 + DB 注册）的实际落盘格式。
    # 旧逻辑只检查默认 Qwen 文件，DeepSeek/用户注册模型已下载时会漏报引擎。
    model_payloads = [_model_api_payload(m) for m in _get_all_model_configs()]
    engine_ids = {
        engine
        for payload in model_payloads
        for engine in payload.get("supported_engines", [])
    }
    available_engines = []

    if "llama_cpp" in engine_ids:
        available_engines.append({
            "id": "llama_cpp",
            "name": "llama.cpp + GGUF",
            "description": "GGUF 量化模型，适合 CPU/集显或轻量试水",
            "model_size_gb": None,
            "requires_cuda": False,
        })

    if "pytorch" in engine_ids:
        has_cuda = torch.cuda.is_available()
        available_engines.append({
            "id": "pytorch",
            "name": "PyTorch + Safetensors" + (" (CUDA)" if has_cuda else " (CPU)"),
            "description": "Safetensors 格式，支持 INT4/INT8/FP16 量化" + ("，GPU 加速" if has_cuda else "，CPU 模式较慢"),
            "model_size_gb": None,
            "requires_cuda": has_cuda,
        })
    # P3修复: 量化选项动态化 — 仅返回当前环境实际可用的量化精度
    pytorch_quants = []
    if "pytorch" in engine_ids:
        has_cuda = torch.cuda.is_available()
        pytorch_quants = [
            {
                "id": "int4",
                "name": "INT4 量化 ⭐",
                "description": "4-bit 量化，显存 ~1.8 GB，速度 ~29 tok/s（推荐边缘设备）",
                "memory_gb": 1.8,
                "speed_tok_s": 29,
                "compile_support": False,
                "engine": "pytorch",
                "is_available": True,
            },
            {
                "id": "int8",
                "name": "INT8 量化",
                "description": "8-bit 量化，显存 ~2.3 GB，速度 ~10 tok/s",
                "memory_gb": 2.3,
                "speed_tok_s": 10,
                "compile_support": False,
                "engine": "pytorch",
                "is_available": True,
            },
        ]
        if has_cuda:
            pytorch_quants.insert(0, {
                "id": "fp16",
                "name": "FP16 原版",
                "description": "原始精度，显存 ~3.5 GB，速度最快 (~53 tok/s)",
                "memory_gb": 3.5,
                "speed_tok_s": 53,
                "compile_support": True,
                "engine": "pytorch",
                "is_available": True,
            })

    gguf_quants = []
    if "llama_cpp" in engine_ids:
        gguf_quants = [
            {
                "id": "gguf",
                "name": "GGUF 量化",
                "description": "量化精度由 GGUF 文件决定（Q4_K_M / Q5_K_M 等），适合 CPU/集显",
                "memory_gb": None,
                "speed_tok_s": None,
                "compile_support": False,
                "engine": "llama_cpp",
                "is_available": True,
            },
        ]

    return {
        "models": pytorch_quants + gguf_quants,
        "current": current_quant if model_loaded else None,
        "current_engine": (
            model_manager._engine_type
            if model_loaded and model_manager.is_loaded
            else None
        ),
        "available_engines": available_engines,
    }


# ============================================================
# P3: 多模型实验支持 API
# ============================================================

class SwitchModelRequest(BaseModel):
    model_id: str = Field(..., description="目标模型唯一标识")
    quant_type: str = Field(default="int4", description="量化精度")
    engine: str = Field(default="auto", description="推理引擎")


class RegisterModelRequest(BaseModel):
    model_id: str = Field(..., description="模型唯一标识")
    name: str = Field(..., description="显示名称")
    model_type: str = Field(default="safetensors", description="safetensors | gguf | both")
    model_path: str = Field(default="", description="safetensors 目录路径")
    gguf_path: str = Field(default="", description="GGUF 文件路径")
    recommended_vram_gb: float = Field(default=8.0, description="推荐显存 (GB)")
    max_context: int = Field(default=4096, description="最大上下文长度")
    huggingface_id: str = Field(default="", description="HuggingFace 仓库 ID")
    description: str = Field(default="", description="简短说明")


def _get_db_experimental_models() -> list[dict]:
    """从 DB 读取用户注册的实验模型（安全包装）。"""
    try:
        if _db_available:
            from db import get_experimental_models
            return get_experimental_models()
    except Exception:
        pass
    return []


def _cuda_gate():
    """CUDA 门控：非 CUDA 环境拒绝请求。"""
    if not mc.is_cuda_available():
        raise HTTPException(
            status_code=403,
            detail="实验模型功能仅限 PC 独显版 (CUDA) 使用。当前环境未检测到 CUDA GPU。",
        )


def _db_entry_to_model_config(entry: dict) -> Optional[mc.ModelConfig]:
    """Convert a DB model entry into ModelConfig; invalid rows are ignored."""
    mid = entry.get("model_id", "")
    if not mid:
        return None
    try:
        return mc.ModelConfig(
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
    except (TypeError, ValueError):
        return None


def _get_all_model_configs() -> list[mc.ModelConfig]:
    """Return builtin + DB-registered models without hiding unavailable entries."""
    models = mc.get_builtin_models()
    seen = {m.model_id for m in models}
    for entry in _get_db_experimental_models():
        model = _db_entry_to_model_config(entry)
        if model and model.model_id not in seen:
            models.append(model)
            seen.add(model.model_id)
    return models


def _model_api_payload(model: mc.ModelConfig) -> dict:
    """Serialize a model config with local availability and loadability metadata."""
    file_status = mc.get_model_file_status(model)
    supported_engines: list[str] = []

    if file_status["has_gguf"]:
        supported_engines.append("llama_cpp")
    if file_status["has_safetensors"]:
        supported_engines.append("pytorch")

    is_available = bool(supported_engines)
    unavailable_reason = file_status["unavailable_reason"]
    if file_status["is_available"] and not supported_engines:
        unavailable_reason = "模型文件已存在，但当前设备缺少可用推理后端。"

    if "pytorch" in supported_engines:
        preferred_engine = "pytorch"
    elif "llama_cpp" in supported_engines:
        preferred_engine = "llama_cpp"
    else:
        preferred_engine = "auto"
    default_quant = "Q4_K_M" if preferred_engine == "llama_cpp" else "int4"

    return {
        "model_id": model.model_id,
        "name": model.name,
        "is_builtin": mc.get_builtin_model(model.model_id) is not None,
        "model_type": model.model_type,
        "is_experimental": model.is_experimental,
        "recommended_vram_gb": model.recommended_vram_gb,
        "max_context": model.max_context,
        "quant_types": model.quant_types,
        "description": model.description,
        "huggingface_id": model.huggingface_id,
        "location": model.location,
        "model_path": model.model_path,
        "gguf_path": model.gguf_path,
        "is_available": is_available,
        "unavailable_reason": unavailable_reason,
        "available_formats": file_status["available_formats"],
        "has_safetensors": file_status["has_safetensors"],
        "has_gguf": file_status["has_gguf"],
        "expected_paths": file_status["expected_paths"],
        "supported_engines": supported_engines,
        "preferred_engine": preferred_engine,
        "default_quant_type": default_quant,
        "requires_cuda": bool(
            model.is_experimental
            and file_status["has_safetensors"]
            and "pytorch" not in supported_engines
        ),
    }


def _normalize_quant_for_engine(quant_type: str, engine: str) -> str:
    """Return a safe quant value for the concrete engine or raise HTTP 400."""
    raw = str(quant_type or "").strip()
    if engine == "llama_cpp":
        return raw or "gguf"

    quant = raw.lower()
    if quant not in ("fp16", "int8", "int4"):
        raise HTTPException(400, f"不支持的量化类型: {quant}，可选: fp16, int8, int4")
    return quant


def _validate_model_load_request(model_id: Optional[str], engine: str) -> None:
    """Reject unavailable model loads before unloading the current model."""
    if not model_id:
        return

    model = mc.get_model_config(model_id, _get_db_experimental_models())
    if model is None:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 未在注册表中找到。")

    payload = _model_api_payload(model)
    if not payload["is_available"]:
        raise HTTPException(
            status_code=400,
            detail=f"模型 '{model.name}' 不可加载：{payload['unavailable_reason']}",
        )

    if engine == "llama_cpp" and not payload["has_gguf"]:
        raise HTTPException(status_code=400, detail=f"模型 '{model.name}' 未配置或未下载 GGUF 文件。")
    if engine == "pytorch":
        if not payload["has_safetensors"]:
            raise HTTPException(status_code=400, detail=f"模型 '{model.name}' 未配置或未下载 Safetensors 文件。")


def _resolve_model_path_for_engine(model_id: Optional[str], engine: str) -> Optional[str]:
    """Resolve a registered model path for the requested engine, including DB models."""
    if not model_id:
        return None
    model = mc.get_model_config(model_id, _get_db_experimental_models())
    if model is None:
        return None
    payload = _model_api_payload(model)
    selected_engine = engine if engine != "auto" else payload.get("preferred_engine", "auto")
    if selected_engine == "llama_cpp" and payload.get("has_gguf"):
        return mc.resolve_model_path(model.gguf_path)
    if selected_engine == "pytorch" and payload.get("has_safetensors"):
        return mc.resolve_model_path(model.model_path)
    return None


def _effective_engine_for_model(model_id: Optional[str], engine: str) -> str:
    """Return the concrete engine to pass into ModelManager."""
    if engine != "auto" or not model_id:
        return engine
    model = mc.get_model_config(model_id, _get_db_experimental_models())
    if model is None:
        return engine
    payload = _model_api_payload(model)
    return payload.get("preferred_engine") or engine


@app.get("/api/models")
async def list_models():
    """列出所有可用模型配置（内置 + 用户注册），含 active_model_id。

    - 实验模型仅在 CUDA 可用时返回
    - 始终包含默认 Qwen-1.8B 模型
    """
    models_data = [_model_api_payload(m) for m in _get_all_model_configs()]

    return {
        "models": models_data,
        "active_model_id": model_manager.active_model_id if model_loaded else None,
    }


@app.post("/api/models/switch")
async def switch_model(req: SwitchModelRequest):
    """
    切换到另一个模型（P3 多模型支持）。

    会卸载当前模型，然后加载新模型。
    仅 CUDA 环境可用（非 CUDA 返回 403）。
    """
    global model_loaded, current_quant, kv_cache, conversation_stats

    # 验证 engine 参数
    engine = req.engine.lower()
    if engine not in ("auto", "llama_cpp", "pytorch"):
        raise HTTPException(400, f"不支持的引擎: {engine}，可选: auto, llama_cpp, pytorch")
    _validate_model_load_request(req.model_id, engine)
    resolved_model_path = _resolve_model_path_for_engine(req.model_id, engine)
    effective_engine = _effective_engine_for_model(req.model_id, engine)
    quant = _normalize_quant_for_engine(req.quant_type, effective_engine)

    try:
        # 更新全局引擎配置（P3修复: switch_model 也需要更新 config）
        import config as cfg
        cfg.INFERENCE_ENGINE = effective_engine if effective_engine != "auto" else cfg.INFERENCE_ENGINE
        cfg.QUANT_TYPE = quant

        # 清空内存会话/KV/统计（新模型不能复用旧模型的上下文）
        _reset_runtime_conversation_state(clear_histories=True)

        result = _run_exclusive_model_change(
            lambda: model_manager.switch_model(
                model_id=req.model_id,
                quant_type=quant,
                profile=device_profile,
                engine=effective_engine if effective_engine != "auto" else None,
                model_path=resolved_model_path,
                db_experimental_models=_get_db_experimental_models(),
            )
        )

        if result["success"]:
            model_loaded = True
            current_quant = quant
            _init_kv_cache()
            return result
        else:
            # 切换失败 — 检查是否回滚成功
            if model_manager.is_loaded:
                model_loaded = True
                # P3修复: llama_cpp 引擎无 quant_type（GGUF 自带量化），回退到 "gguf"
                current_quant = model_manager.quant_type or (
                    "gguf" if model_manager._engine_type == "llama_cpp" else QUANT_TYPE
                )
            else:
                model_loaded = False
                current_quant = QUANT_TYPE
            raise HTTPException(status_code=500, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型切换异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型切换失败: {e}")


@app.get("/api/models/registry")
async def list_model_registry():
    """列出用户注册的实验模型配置。"""
    db_models = _get_db_experimental_models()
    return {"models": db_models}


@app.post("/api/models/registry")
async def register_model(req: RegisterModelRequest):
    """注册一个新的实验模型配置。

    模型文件需用户自行下载到指定路径。
    """
    if req.model_type not in {"safetensors", "gguf", "both"}:
        raise HTTPException(status_code=400, detail="model_type 必须是 safetensors | gguf | both")
    if not _db_available:
        raise HTTPException(status_code=503, detail="数据库不可用，无法注册模型。")

    config = {
        "model_id": req.model_id,
        "name": req.name,
        "model_type": req.model_type,
        "model_path": req.model_path,
        "gguf_path": req.gguf_path,
        "recommended_vram_gb": req.recommended_vram_gb,
        "max_context": req.max_context,
        "huggingface_id": req.huggingface_id,
        "description": req.description,
        "quant_types": ["fp16", "int8", "int4"] if req.model_type != "gguf" else ["Q4_K_M"],
    }

    try:
        from db import save_experimental_model
        ok = save_experimental_model(req.model_id, json.dumps(config, ensure_ascii=False))
        if not ok:
            raise HTTPException(status_code=500, detail="注册模型配置失败")
        return {"status": "registered", "model_id": req.model_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册模型失败: {e}")


@app.delete("/api/models/registry/{model_id}")
async def unregister_model(model_id: str):
    """删除一个用户注册的实验模型配置。

    不会删除磁盘上的模型文件，仅取消注册。
    """
    if not _db_available:
        raise HTTPException(status_code=503, detail="数据库不可用。")

    # 不允许删除内置模型
    if mc.get_builtin_model(model_id):
        raise HTTPException(status_code=400, detail=f"内置模型 '{model_id}' 不允许删除。")

    try:
        from db import delete_experimental_model
        deleted = delete_experimental_model(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 未注册")
        return {"status": "deleted", "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消注册失败: {e}")


# ============================================================
# 集群管理 API
# ============================================================

@app.get("/api/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """
    获取集群整体状态。

    包含所有节点状态、TCP 连接信息、当前任务等。
    单机模式下返回 3 个默认节点（均为 online）。
    """
    return scheduler.get_status()


@app.get("/api/cluster/nodes")
async def get_cluster_nodes():
    """
    获取所有节点详情列表。

    Returns:
        { nodes: [...], count: int, online_count: int }
    """
    nodes = scheduler.get_nodes()
    online_count = sum(1 for n in nodes if n["is_available"])
    return {
        "nodes": nodes,
        "count": len(nodes),
        "online_count": online_count,
        "offline_count": len(nodes) - online_count,
    }


@app.post("/api/cluster/nodes/{node_id}/deregister")
async def deregister_node(node_id: str):
    """
    强制注销一个从节点。

    仅在分布式模式下有效；master 节点不可注销。
    """
    if node_id == "master":
        raise HTTPException(400, "主节点不可注销")

    success = scheduler.deregister_node(node_id)
    if not success:
        raise HTTPException(404, f"节点 '{node_id}' 不存在")

    logger.info(f"节点 {node_id} 已被强制注销")
    return {
        "status": "deregistered",
        "node_id": node_id,
    }


@app.delete("/api/cluster/nodes/{node_id}")
async def delete_cluster_node(node_id: str):
    """
    删除离线节点记录（区别于 deregister：deregister 仅标记离线）。

    用于移除手动注册的 Android / 离线占位节点。
    """
    result = scheduler.delete_node(node_id)
    status = result.get("status")
    if status == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if status == "invalid":
        raise HTTPException(400, result.get("reason", "无效节点"))
    if status == "not_found":
        raise HTTPException(404, result.get("reason", "节点不存在"))
    if status == "online":
        raise HTTPException(409, result.get("reason", "节点在线，无法删除"))
    if status != "deleted":
        raise HTTPException(500, result.get("reason", "删除节点失败"))
    return result


@app.get("/api/cluster/config")
async def get_cluster_config():
    """
    获取分布式配置信息。

    包含网络配置、分层配置、模型配置、任务统计、当前节点角色。
    """
    return scheduler.get_config()


@app.get("/api/cluster/my-role")
async def get_my_role():
    """
    获取当前节点的角色信息。

    用于前端判断：
    - master 节点：后台管理 Tab 完全开放
    - client 节点：需在设置中开启"分布式推理优化"后才可见
    """
    return scheduler.get_my_role()


@app.put("/api/cluster/config/max-nodes")
async def update_max_nodes(req: UpdateMaxNodesRequest):
    """
    动态调整最大节点数量（仅主节点可调用）。

    仅修改容量上限，不预创建空槽位。从节点通过 TCP 注册动态加入。
    """
    result = scheduler.update_max_nodes(req.max_nodes)
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效参数"))
    return result


@app.get("/api/cluster/invite")
async def get_invite_info():
    """
    获取主节点的邀请/连接信息（供从节点连接使用）。

    主节点调用此接口获取自身监听地址和端口，
    用户将此信息提供给从节点，从节点在后台管理中输入并连接。
    """
    return scheduler.get_invite_info()


@app.post("/api/bootstrap/first-connect")
async def first_connect_bootstrap(req: FirstConnectBootstrapRequest, request: Request):
    """
    首次连接自动部署。

    安全边界：只接受 Tailscale / 受信 CIDR 来源。通过该接口下发集群密钥
    和主节点连接信息，客户端持久化后再走现有 TCP HMAC 注册。
    """
    if os.environ.get("QLH_BOOTSTRAP_ENABLED", "true").strip().lower() in {"0", "false", "no"}:
        raise HTTPException(403, "bootstrap disabled")

    peer_host = request.client.host if request.client else ""
    from bootstrap import is_trusted_bootstrap_source, normalize_node_id, normalize_node_type

    require_trusted = os.environ.get("QLH_BOOTSTRAP_REQUIRE_TAILSCALE", "true").strip().lower()
    if require_trusted not in {"0", "false", "no"}:
        if not is_trusted_bootstrap_source(peer_host):
            raise HTTPException(403, "source network is not trusted")

    if scheduler._effective_role() != "master":
        raise HTTPException(403, "only master can serve bootstrap")

    node_type = normalize_node_type(req.node_type)
    node_id = normalize_node_id(req.node_id, node_type)
    if node_id == "master":
        raise HTTPException(400, "reserved node_id")

    from node_config import ensure_local_cluster_secret
    cluster_secret = ensure_local_cluster_secret()
    try:
        import config as cfg
        cfg.CLUSTER_SECRET = cluster_secret
    except Exception:
        pass

    api_host = request.url.hostname or peer_host
    lan_ip = getattr(scheduler, "_lan_ip", "") or ""
    from bootstrap import select_advertised_master_host

    master_tcp_host = select_advertised_master_host(api_host, lan_ip)
    master_api_host = api_host or master_tcp_host
    master_api_port = request.url.port or API_PORT
    master_tcp_port = scheduler.tcp_server.port if scheduler.tcp_server else SERVER_PORT

    hostname = req.hostname or node_id
    address = f"{peer_host}" if peer_host else ""
    register_result = scheduler.manual_register_node(
        node_id=node_id,
        hostname=hostname,
        address=address,
        network_type="tailscale" if peer_host.startswith("100.") else "trusted",
        node_type=node_type,
    )
    if register_result.get("status") in {"denied", "invalid", "full"}:
        status_code = 403 if register_result.get("status") == "denied" else 400
        raise HTTPException(status_code, register_result.get("reason", "bootstrap registration failed"))

    pipeline_worker = node_type == "pc"
    response = {
        "status": "ok",
        "cluster": {
            "cluster_id": os.environ.get("QLH_CLUSTER_ID", "qlh-default"),
            "master_api_host": master_api_host,
            "master_api_port": master_api_port,
            "master_tcp_host": master_tcp_host,
            "master_tcp_port": master_tcp_port,
            "cluster_secret": cluster_secret,
        },
        "node": {
            "node_id": node_id,
            "role": "client",
            "node_type": node_type,
            "pipeline_worker": pipeline_worker,
        },
        "android": {
            "presence_interval_seconds": 45,
            "pipeline_worker": False,
            "model_manifest_url": f"http://{master_api_host}:{master_api_port}/api/models/downloadable",
        },
    }
    logger.info(
        "首次连接部署: node_id=%s type=%s peer=%s host=%s api=%s:%s tcp=%s:%s",
        node_id, node_type, peer_host, hostname,
        master_api_host, master_api_port, master_tcp_host, master_tcp_port,
    )
    return response


@app.get("/api/bootstrap/info")
async def bootstrap_info(request: Request):
    """Minimal discovery endpoint for peers already admitted to the Tailnet."""
    peer_host = request.client.host if request.client else ""
    from bootstrap import is_trusted_bootstrap_source

    if not is_trusted_bootstrap_source(peer_host):
        raise HTTPException(403, "source network is not trusted")
    role = scheduler.get_my_role()
    return {
        "status": "ok",
        "is_master": bool(role.get("is_master")),
        "node_id": role.get("node_id", ""),
        "master_api_port": API_PORT,
        "master_tcp_port": scheduler.tcp_server.port if scheduler.tcp_server else SERVER_PORT,
    }


@app.post("/api/cluster/connect")
async def connect_to_master(req: ConnectToMasterRequest):
    """
    从节点主动连接主节点（从节点的「连接主节点」按钮触发）。

    调用后本节点将通过 TCP 向指定主节点发起注册，
    注册成功后主节点的节点列表中将出现本节点。
    """
    force_bootstrap = False
    if scheduler._effective_role() == "master":
        if not req.switch_to_client or not scheduler.can_join_existing_master():
            raise HTTPException(403, "当前主节点已确认或已有从节点，不能切换为从节点")
        switch_result = scheduler.activate_client_mode()
        if switch_result.get("status") == "denied":
            raise HTTPException(409, switch_result.get("reason", "无法切换为从节点"))
        force_bootstrap = True

    result = scheduler.connect_to_master(
        req.master_host,
        req.master_port,
        force_bootstrap=force_bootstrap,
    )
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅从节点可连接主节点"))
    if result.get("status") == "bootstrap_failed":
        raise HTTPException(400, result.get("reason", "首次连接自动部署失败"))
    if result.get("status") == "failed":
        raise HTTPException(400, result.get("reason", "连接失败"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "连接异常"))
    return result


class ManualRegisterRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="节点标识")
    hostname: str = Field(default="", description="主机名")
    address: str = Field(default="", description="预留 IP:Port")
    network_type: str = Field(default="unknown", description="网络类型: wifi | ethernet | unknown")
    node_type: str = Field(default="pc", description="设备平台: pc | android")


class AndroidPresenceRequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64, description="Android 稳定节点标识")
    hostname: str = Field(default="", description="Android 设备名")
    address: str = Field(default="", description="HTTP 客户端地址（可选，仅展示）")
    network_type: str = Field(default="unknown", description="网络类型: wifi | mobile | ethernet | vpn | other | unknown")
    device_info: dict = Field(default_factory=dict, description="Android 设备画像/运行状态")
    client_mode: str = Field(default="thin", description="客户端模式: thin | full")
    app_variant: str = Field(default="full", description="Android flavor: full | lite")
    app_version: str = Field(default="", description="App 版本")


@app.post("/api/cluster/nodes/register")
async def manual_register_node(req: ManualRegisterRequest):
    """
    主节点手动注册一个从节点（无需 TCP 连接）。

    管理员可在后台管理页面提前录入从节点信息。
    手动注册的节点初始状态为 offline，待从节点通过 TCP 连接后自动变为 online。

    如果从节点主动通过「连接主节点」发起 TCP 注册，也会自动加入节点列表，
    无需手动注册。此接口用于管理员提前规划节点或预留槽位。
    """
    result = scheduler.manual_register_node(
        node_id=req.node_id,
        hostname=req.hostname,
        address=req.address,
        network_type=req.network_type,
        node_type=req.node_type,
    )
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅主节点可手动注册"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效参数"))
    if result.get("status") == "full":
        raise HTTPException(400, result.get("reason", "节点容量已满"))
    if result.get("status") == "exists":
        return result  # 已存在不报错，返回当前状态
    return result


@app.post("/api/cluster/android/register")
async def register_android_presence(req: AndroidPresenceRequest, request: Request):
    """Android Full 薄客户端在线登记/心跳（不是 TCP worker 注册）。"""
    http_peer = request.client.host if request.client else ""
    result = scheduler.register_android_client(
        node_id=req.node_id,
        hostname=req.hostname,
        address=req.address,
        network_type=req.network_type,
        device_info=req.device_info,
        client_mode=req.client_mode,
        app_variant=req.app_variant,
        app_version=req.app_version,
        http_peer=http_peer,
    )
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅主节点可登记 Android 客户端"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "无效 Android 节点"))
    return result


@app.post("/api/cluster/android/heartbeat")
async def heartbeat_android_presence(req: AndroidPresenceRequest, request: Request):
    """Android Full 薄客户端心跳；实现与 register 相同，重复调用会刷新 last_heartbeat。"""
    return await register_android_presence(req, request)


@app.get("/api/cluster/master-health")
async def check_master_health():
    """
    检查主节点是否在线（通过数据库心跳时间戳）。

    从节点前端周期性调用此接口（配合 5 秒轮询），
    当检测到主节点宕机时显示告警横幅。
    主节点自身调用时返回本地运行状态。

    Returns:
        { master_online, last_seen_seconds_ago, stale, master_host, master_port }
    """
    if scheduler._effective_role() == "master":
        # 主节点自身：直接返回在线
        return {
            "master_online": True,
            "last_seen_seconds_ago": 0,
            "stale": False,
            "master_host": getattr(scheduler, '_lan_ip', '') or SERVER_IP,
            "master_port": SERVER_PORT,
            "source": "self",
        }
    return scheduler.get_client_master_status()


@app.get("/api/cluster/discover")
async def discover_master():
    """
    从数据库查询主节点的连接信息（从节点自动发现）。

    从节点启动后调用此接口，尝试在数据库中查找已注册的主节点。
    如果找到且在 120 秒内有心跳，则返回主节点地址，
    前端可自动填充连接表单。

     Returns:
         {
             "found": bool,           # 是否在数据库中找到主节点
             "master_host": str,      # 主节点 IP
             "master_port": int,      # 主节点端口
             "master_mac_addresses": [str],  # 主节点 MAC 地址（身份标识）
             "stale": bool,           # 心跳是否过期 (>120s)
             "source": str,           # "database" | "config" | "none"
         }
    """
    return scheduler.discover_master()


class ResetIdentityRequest(BaseModel):
    confirm: str = Field(default="", description="输入 'reset' 确认重置")


@app.post("/api/cluster/reset-identity")
async def reset_master_identity(req: ResetIdentityRequest):
    """
    重置主节点身份标识（仅主节点可调用）。

    用于更换主节点机器或网卡后，清除数据库中旧的 MAC 地址记录。
    需要输入确认字符串 'reset' 以防止误操作。

    调用后需重启主节点后端服务，新的 MAC 地址将在下次启动时自动记录。
    """
    if req.confirm.strip().lower() != "reset":
        raise HTTPException(400, "请输入 'reset' 确认重置操作")
    result = scheduler.reset_master_identity()
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


@app.post("/api/cluster/email-test")
async def test_email_notification():
    """
    发送一封测试邮件，验证 SMTP 邮件告警配置是否正确。

    邮件将发送到 SMTP.md 中配置的目标邮箱。
    任何节点均可调用（主节点和从节点均可测试邮件发送）。
    """
    try:
        from email_notifier import send_test_email
        ok = send_test_email()
        if ok:
            return {"status": "ok", "message": "测试邮件已发送，请检查目标邮箱"}
        else:
            raise HTTPException(500, "邮件发送失败，请检查后端日志了解详情")
    except ImportError as e:
        raise HTTPException(500, f"邮件模块导入失败: {e}")
    except Exception as e:
        raise HTTPException(500, f"邮件发送异常: {e}")


# ============================================================
# 推理调度队列 API (Phase 3 — MLFQ 三级队列可视化与管理)
# ============================================================

class SetQueueStrategyRequest(BaseModel):
    strategy: str = Field(..., pattern="^(fifo|mlfq)$", description="调度策略: fifo | mlfq")


class CancelTaskResponse(BaseModel):
    success: bool
    task_id: str
    message: str = ""


@app.get("/api/cluster/queue")
async def get_queue_detail():
    """
    获取推理调度队列完整详情。

    返回三级队列（Q0/Q1/Q2）中每个任务的序列化信息，
    含优先级、等待时间、预估耗时、老化状态、抢占统计。
    仅主节点可用。
    """
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可查看请求队列")
    return scheduler.pipeline_queue.get_queue_detail()


@app.post("/api/cluster/queue/strategy")
async def set_queue_strategy(req: SetQueueStrategyRequest):
    """切换调度策略: fifo | mlfq。仅主节点。"""
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可切换调度策略")
    try:
        scheduler.pipeline_queue.set_strategy(req.strategy)
        return {"success": True, "strategy": req.strategy}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/cluster/queue/pause")
async def pause_queue():
    """暂停接受新请求。仅主节点。"""
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可暂停请求队列")
    scheduler.pipeline_queue.pause()
    return {"success": True, "paused": True}


@app.post("/api/cluster/queue/resume")
async def resume_queue():
    """恢复接受新请求。仅主节点。"""
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可恢复请求队列")
    scheduler.pipeline_queue.resume()
    return {"success": True, "paused": False}


@app.post("/api/cluster/queue/clear")
async def clear_queue():
    """清空所有排队任务（不影响执行中的任务）。仅主节点。"""
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可清空请求队列")
    count = scheduler.pipeline_queue.clear()
    return {"success": True, "cleared": count}


@app.delete("/api/cluster/queue/task/{task_id}")
async def cancel_queue_task(task_id: str):
    """
    取消指定排队任务。

    执行中的流水线任务会在当前 token step 完成后通过 PIPELINE_ABORT 中止。
    仅主节点。
    """
    if not scheduler._effective_role() == "master":
        raise HTTPException(403, "仅主节点可取消队列任务")
    ok = scheduler.pipeline_queue.cancel_task(task_id)
    if ok:
        return CancelTaskResponse(success=True, task_id=task_id, message="任务已取消")
    else:
        return CancelTaskResponse(
            success=False, task_id=task_id,
            message="任务不存在或已经完成，无法取消"
        )


# ============================================================
# 分布式推理开关 API
# ============================================================

@app.get("/api/cluster/config/distributed-inference")
async def get_distributed_inference_config():
    """
    获取分布式推理开关状态。
    """
    from config import DISTRIBUTED_INFERENCE_ENABLED
    return {
        "enabled": scheduler.get_distributed_inference_enabled(),
        "default": DISTRIBUTED_INFERENCE_ENABLED,
    }


class DistributedInferenceRequest(BaseModel):
    enabled: bool = Field(..., description="是否启用分布式推理")


@app.put("/api/cluster/config/distributed-inference")
async def set_distributed_inference_config(req: DistributedInferenceRequest):
    """
    设置分布式推理开关。

    - 主节点：控制是否接收从节点连接和协调分布式推理
    - 从节点：控制是否将推理请求转发给主节点
    """
    result = scheduler.set_distributed_inference_enabled(req.enabled)
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "设置失败"))
    return result


# ============================================================
# 动态模型分层 API
# ============================================================

@app.get("/api/cluster/layers")
async def get_layer_assignments():
    """
    获取当前模型分层配置。

    Returns:
        {
            "total": 24,
            "strategy": "dynamic" | "manual",
            "assignments": [{node_id, role, start_layer, end_layer,
                             has_embedding, has_lm_head, score}],
            "computed_at": timestamp | null,
        }
    """
    return scheduler.get_layer_assignments()


class LayerOverrideItem(BaseModel):
    node_id: str = Field(..., description="节点标识")
    start_layer: int = Field(..., ge=0, description="起始层（含）")
    end_layer: int = Field(..., ge=1, description="结束层（不含）")


class LayerOverrideRequest(BaseModel):
    assignments: list[LayerOverrideItem] = Field(..., min_length=1, description="分层覆盖列表")


@app.put("/api/cluster/layers")
async def override_layer_assignments(req: LayerOverrideRequest):
    """
    手动覆盖模型分层配置（仅主节点可调用）。

    验证规则:
      - 所有区间必须从 0 开始连续覆盖到 24
      - node_id 必须是已注册节点
      - 区间不能重叠
    """
    result = scheduler.override_layer_assignments([
        {"node_id": a.node_id, "start_layer": a.start_layer, "end_layer": a.end_layer}
        for a in req.assignments
    ])
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅主节点可修改"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "分层配置无效"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


@app.delete("/api/cluster/layers")
async def reset_layer_assignments():
    """
    重置分层配置，清除手动覆盖，恢复自动（dynamic）策略。

    仅主节点可调用。
    """
    if scheduler._effective_role() != "master":
        raise HTTPException(403, "仅主节点可重置分层配置")
    return scheduler.reset_layer_assignments()


# ============================================================
# 角色转让 API
# ============================================================

class TransferMasterRequest(BaseModel):
    target_node_id: str = Field(..., min_length=1, max_length=64,
                                 description="目标从节点 ID（将升级为新主节点）")


@app.post("/api/cluster/transfer-master")
async def transfer_master_role(req: TransferMasterRequest):
    """
    将主节点身份转让给指定从节点（仅主节点可调用）。

    流程:
      1. 主节点通过 TCP 向目标从节点发送 ROLE_TRANSFER 消息
      2. 从节点保存升级日志、返回 ACK
      3. 主节点保存降级日志、更新数据库中的主节点信息
      4. 建议双方重启以应用新角色

    注意: 转让后需要重启服务才能生效：
      - 原主节点重启后以从节点模式运行
      - 新主节点重启后以主节点模式运行
    """
    result = scheduler.transfer_master_role(req.target_node_id)
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "参数无效"))
    if result.get("status") == "timeout":
        raise HTTPException(408, result.get("reason", "超时"))
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


@app.get("/api/cluster/transfer-logs")
async def get_transfer_logs():
    """
    获取角色转让日志（降级 + 升级）。

    Returns:
        { logs: [{direction, from_role, to_role, related_node, timestamp, ...}] }
    """
    logs = scheduler.get_transfer_logs()
    return {"logs": logs, "count": len(logs)}


# ============================================================
# 备用主节点管理 API
# ============================================================

class SpareMasterRequest(BaseModel):
    target_node_id: str


@app.get("/api/cluster/spare-master")
async def get_spare_master():
    """
    获取当前备用主节点信息。

    Returns:
        { spare_master: {node_id, hostname, address, designated_at, is_online, state} | null }
    """
    spare = scheduler.get_spare_master()
    return {"spare_master": spare}


@app.post("/api/cluster/spare-master")
async def designate_spare_master(req: SpareMasterRequest):
    """
    指定一个在线从节点为备用主节点（仅主节点可调用）。

    规则:
      - 集群节点数 ≥ 2
      - 目标节点必须在线且为 client

    Returns:
        { status, message, spare_master, ... }
    """
    result = scheduler.designate_spare_master(req.target_node_id)
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    if result.get("status") == "invalid":
        raise HTTPException(400, result.get("reason", "参数无效"))
    if result.get("status") == "timeout":
        raise HTTPException(408, result.get("reason", "超时"))
    if result.get("status") == "duplicate":
        return result  # 不抛异常，返回已有信息
    if result.get("status") == "error":
        raise HTTPException(500, result.get("reason", "操作失败"))
    return result


@app.delete("/api/cluster/spare-master")
async def clear_spare_master():
    """
    清除备用主节点指定（仅主节点可调用）。

    Returns:
        { status, message }
    """
    result = scheduler.clear_spare_master()
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "权限不足"))
    return result


@app.get("/api/cluster/spare-master/logs")
async def get_spare_master_logs():
    """
    获取备用主节点操作日志。

    Returns:
        { logs: [{direction, timestamp, details, ...}] }
    """
    logs = scheduler.get_spare_master_logs()
    return {"logs": logs, "count": len(logs)}


# ============================================================
# P3: 主节点转让审查 API
# ============================================================

class CreateReviewRequest(BaseModel):
    target_node_id: str = Field(..., description="拟转让的目标从节点 ID")
    reason: str = Field(default="", description="转让原因")
    timeout_hours: float = Field(default=48.0, description="超时时间（小时）")


class CastVoteRequest(BaseModel):
    ticket_id: str = Field(..., description="工单 ID")
    vote: int = Field(..., description="-1（阻止）、0（弃权）、+1（赞同）")
    comment: str = Field(default="", description="投票附言")


@app.post("/api/cluster/review/create")
async def create_review_ticket(req: CreateReviewRequest):
    """
    创建主节点转让审查工单（仅 master 可用）。

    需要先指定备用主节点。
    创建成功后发送邮件通知管理员。
    """
    if scheduler._effective_role() != "master":
        raise HTTPException(status_code=403, detail="仅主节点可创建审查工单")

    # 检查 spare master
    spare = scheduler.get_spare_master()
    if not spare or not spare.get("node_id"):
        raise HTTPException(
            status_code=400,
            detail="未指定备用主节点。请先在「备用主节点」中指定后再创建审查工单。",
        )

    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = review_mgr.create_ticket(
            created_by=scheduler.get_effective_node_id(),
            target_node_id=req.target_node_id,
            reason=req.reason,
            timeout_hours=req.timeout_hours,
        )
        if ticket is None:
            raise HTTPException(status_code=503, detail="数据库不可用，无法创建审查工单")
        return ticket.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建审查工单失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建审查工单失败: {e}")


@app.post("/api/cluster/review/vote")
async def cast_review_vote(req: CastVoteRequest):
    """
    对审查工单投票（仅 PC 独显节点可投票）。

    投票值: -1（阻止）、0（弃权）、+1（赞同）。
    同一节点重复投票会更新之前的投票。

    阈值: >= +2 通过，<= -2 阻止。
    """
    node_id = scheduler.get_effective_node_id()

    # 验证投票资格
    can_vote, reason = scheduler.can_node_vote(node_id)
    if not can_vote:
        raise HTTPException(status_code=403, detail=reason)

    if req.vote not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="投票值必须为 -1、0 或 +1")

    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = review_mgr.cast_vote(
            ticket_id=req.ticket_id,
            voter_node_id=node_id,
            vote_value=req.vote,
            comment=req.comment,
        )
        if ticket is None:
            raise HTTPException(status_code=404, detail=f"工单 '{req.ticket_id}' 不存在或已关闭")
        return ticket.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"投票失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"投票失败: {e}")


@app.get("/api/cluster/review/tickets")
async def list_review_tickets(status: Optional[str] = None):
    """列出审查工单。可选过滤: ?status=pending"""
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        tickets = review_mgr.list_tickets(status)
        return {
            "tickets": [t.to_dict() for t in tickets],
            "count": len(tickets),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工单列表失败: {e}")


@app.get("/api/cluster/review/tickets/{ticket_id}")
async def get_review_ticket(ticket_id: str):
    """获取单个审查工单详情。"""
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        ticket = review_mgr.get_ticket(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail=f"工单 '{ticket_id}' 不存在")
        return ticket.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工单失败: {e}")


@app.get("/api/cluster/review/can-vote")
async def check_can_vote():
    """检查当前节点是否有审查投票资格。"""
    node_id = scheduler.get_effective_node_id()
    can_vote, reason = scheduler.can_node_vote(node_id)
    return {
        "node_id": node_id,
        "can_vote": can_vote,
        "reason": reason,
    }


@app.post("/api/cluster/review/expire-check")
async def trigger_expire_check():
    """手动触发审查工单过期检查。"""
    try:
        from review import ReviewManager
        review_mgr = ReviewManager()
        expired = review_mgr.resolve_expired()
        return {"expired": expired, "count": len(expired)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"过期检查失败: {e}")


@app.delete("/api/cluster/review/tickets/{ticket_id}")
async def delete_review_ticket(ticket_id: str):
    """删除单个审查工单（所有状态均可）。"""
    try:
        from review import ReviewManager
        ok = ReviewManager().delete_ticket(ticket_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在或删除失败")
        return {"status": "deleted", "ticket_id": ticket_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除工单失败: {e}")


@app.delete("/api/cluster/review/tickets")
async def delete_resolved_review_tickets():
    """批量删除所有已解决（approved/rejected/expired）的审查工单。"""
    try:
        from review import ReviewManager
        count = ReviewManager().delete_resolved()
        return {"status": "deleted", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除工单失败: {e}")


@app.post("/api/cluster/review/mail-poll")
async def trigger_mail_poll():
    """手动触发邮件投票轮询（用于测试 / 调试）。"""
    try:
        from email_notifier import poll_mail_once
        result = poll_mail_once()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"邮件轮询失败: {e}")


# ============================================================
# 用户偏好设置云同步 API
# ============================================================

@app.get("/api/user/settings")
async def get_user_settings():
    """
    从云数据库读取用户偏好设置。

    返回完整的 settings JSON，与 localStorage 格式一致。
    如果数据库不可用，返回空 dict（前端使用 localStorage 值）。
    """
    if not _db_available:
        return {"settings": {}, "source": "none"}
    try:
        import db as _db_mod
        settings = _db_mod.get_user_settings()
        return {"settings": settings, "source": "database"}
    except Exception as e:
        logger.warning(f"读取用户设置失败: {e}")
        return {"settings": {}, "source": "error"}


class UserSettingsRequest(BaseModel):
    settings: dict = Field(default={}, description="完整的用户设置 JSON")


@app.put("/api/user/settings")
async def update_user_settings(req: UserSettingsRequest):
    """
    将用户偏好设置存储到云数据库。

    前端在更新 localStorage 后调用此接口同步到云端。
    同时同步 save_history 和 distributed_inference 到各自的专用键。
    """
    if not _db_available:
        return {"status": "skipped", "reason": "数据库不可用"}
    try:
        import db as _db_mod
        settings = req.settings

        # 存储完整的 settings JSON
        _db_mod.set_user_settings(settings)

        # 同步 save_history 到专用键（后端推理保存逻辑依赖此键）
        if "saveHistory" in settings:
            _db_mod.set_save_history(bool(settings["saveHistory"]))

        # 同步 distributedInference 到专用键
        if "distributedInference" in settings:
            _db_mod.set_distributed_inference_enabled(bool(settings["distributedInference"]))

        return {"status": "ok", "synced_fields": list(settings.keys())}
    except Exception as e:
        logger.error(f"存储用户设置失败: {e}")
        raise HTTPException(500, f"存储失败: {e}")


# ============================================================
# 对话云同步状态 API
# ============================================================

@app.get("/api/conversations/sync-status")
async def get_conversation_sync_status():
    """
    获取对话历史云同步状态。
    """
    save_history = False
    if _db_available:
        try:
            import db as _db_mod
            save_history = _db_mod.get_save_history()
        except Exception:
            pass

    return {
        "save_history": save_history,
        "db_connected": _db_available,
        "local_save_enabled": True,  # localStorage 始终可用
        "local_store_enabled": True,  # 本地文件降级始终可用
        "cloud_sync_enabled": save_history and _db_available,
    }


# ============================================================
# 对话历史 API（数据库持久化）
# ============================================================

@app.get("/api/conversations")
async def get_conversations(session_id: str = "default", limit: int = 200):
    """
    从数据库加载指定会话的对话历史。

    如果数据库不可用，回退到内存中的对话历史（按 session_id 过滤）。
    """
    # ---- 加载顺序: 数据库 → 本地文件 → 内存 ----
    if _db_available:
        try:
            import db as _db_mod
            messages = _db_mod.get_conversation(session_id, limit)
            count = _db_mod.get_conversation_count(session_id)
            return {
                "messages": [
                    {"role": m["role"], "content": m["content"], "created_at": m.get("created_at")}
                    for m in messages
                ],
                "count": count,
                "source": "database",
            }
        except Exception as e:
            logger.warning(f"数据库读取对话历史失败: {e}")

    # 尝试本地文件
    try:
        local_messages = _local_store.load_local_conversation(session_id, limit)
        if local_messages:
            return {
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in local_messages
                ],
                "count": len(local_messages),
                "source": "local_store",
            }
    except Exception:
        pass

    # 最终降级：内存
    targeted_history = session_histories.get(session_id, [])
    return {
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in targeted_history
        ],
        "count": len(targeted_history),
        "source": "memory_fallback",
    }


@app.delete("/api/conversations")
async def delete_conversations(session_id: str = "default"):
    """
    清空指定会话的对话历史（数据库 + 内存同步）。

    单机模式下仅清空当前会话上下文；分布式模式下可跨节点同步。
    """
    deleted_count = 0

    if _db_available:
        try:
            import db as _db_mod
            deleted_count = _db_mod.clear_conversation(session_id)
            logger.info(f"数据库对话历史已清空: session={session_id}, {deleted_count} 条")
        except Exception as e:
            logger.warning(f"数据库清空对话历史失败: {e}")
    if not _db_available:
        try:
            deleted_count = _local_store.clear_local_conversation(session_id)
            logger.info(f"本地对话历史已清空: session={session_id}, {deleted_count} 条")
        except Exception as e:
            logger.warning(f"本地清空对话历史失败: {e}")

    if session_id == active_session_id or session_id == "default":
        _get_active_history().clear()
    logger.info(f"对话历史已清空 (内存)")
    return {
        "status": "cleared",
        "session_id": session_id,
        "deleted_count": deleted_count,
    }


# ============================================================
# 会话管理 API（多会话支持）
# ============================================================

class CreateSessionRequest(BaseModel):
    title: str = Field(default="新对话", description="会话标题")
    first_message: Optional[str] = Field(default=None, description="可选的首条消息用于自动生成标题")


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256, description="新标题")


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest = None):
    """
    创建新会话并自动激活。

    如果提供了 first_message，用它自动生成标题（截取前30字）；
    否则使用 req.title（默认"新对话"）。
    """
    import uuid
    session_id = str(uuid.uuid4())
    title = "新对话"

    if req and req.first_message:
        title = req.first_message.strip()[:30]
        if len(req.first_message.strip()) > 30:
            title += "..."
    elif req and req.title:
        title = req.title

    # 持久化到数据库（或本地文件降级）
    if _db_available:
        try:
            import db as _db_mod
            _db_mod.create_session(session_id, title)
        except Exception as e:
            logger.warning(f"数据库创建会话失败: {e}")
    if not _db_available:
        try:
            _local_store.create_local_session(session_id, title)
        except Exception as e:
            logger.warning(f"本地创建会话失败: {e}")

    # 注册到内存并激活
    session_histories[session_id] = []
    _switch_session(session_id)

    logger.info(f"会话已创建: {session_id} ({title})")
    return {
        "id": session_id,
        "title": title,
        "message_count": 0,
        "active": True,
    }


@app.get("/api/sessions")
async def list_sessions(limit: int = 50, offset: int = 0):
    """
    获取所有会话列表（按 updated_at DESC 排序）。
    """
    if _db_available:
        try:
            import db as _db_mod
            db_sessions = _db_mod.get_all_sessions(limit, offset)
            total = _db_mod.get_session_count()
            return {
                "sessions": db_sessions,
                "active_session_id": active_session_id,
                "total": total,
            }
        except Exception as e:
            logger.warning(f"数据库读取会话列表失败: {e}")

    # 降级：从本地文件 + 内存合并
    mem_sessions = []
    seen_ids = set()

    # 1. 本地文件中的会话（有持久化的元数据）
    try:
        local_sessions = _local_store.get_all_local_sessions(limit, offset)
        for ls in local_sessions:
            sid = ls["id"]
            seen_ids.add(sid)
            # 用内存中的实际消息数更新计数
            hist = session_histories.get(sid, [])
            mem_sessions.append({
                "id": sid,
                "title": ls.get("title", "新对话"),
                "message_count": len(hist) or ls.get("message_count", 0),
                "created_at": ls.get("created_at"),
                "updated_at": ls.get("updated_at"),
            })
    except Exception:
        pass

    # 2. 内存中有但本地文件中没有的会话
    for sid, hist in session_histories.items():
        if sid not in seen_ids:
            mem_sessions.append({
                "id": sid,
                "title": "会话" if not hist else (hist[0].get("content", "")[:30] if hist else "新对话"),
                "message_count": len(hist),
                "created_at": None,
                "updated_at": None,
            })

    return {
        "sessions": mem_sessions,
        "active_session_id": active_session_id,
        "total": len(mem_sessions),
    }


@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取单个会话的元数据"""
    if _db_available:
        try:
            import db as _db_mod
            session = _db_mod.get_session(session_id)
            if session:
                return session
        except Exception:
            pass
    # 尝试本地文件
    try:
        local_session = _local_store.get_local_session(session_id)
        if local_session:
            hist = session_histories.get(session_id, [])
            local_session["message_count"] = len(hist) or local_session.get("message_count", 0)
            local_session["active"] = session_id == active_session_id
            return local_session
    except Exception:
        pass
    # 最终降级
    hist = session_histories.get(session_id, [])
    return {
        "id": session_id,
        "title": "新对话",
        "message_count": len(hist),
        "active": session_id == active_session_id,
    }


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    """重命名会话"""
    if _db_available:
        try:
            import db as _db_mod
            updated = _db_mod.update_session_title(session_id, req.title)
            if updated:
                return updated
        except Exception as e:
            logger.warning(f"数据库重命名会话失败: {e}")
    if not _db_available:
        try:
            updated = _local_store.update_local_session_title(session_id, req.title)
            if updated:
                return updated
        except Exception as e:
            logger.warning(f"本地重命名会话失败: {e}")
    raise HTTPException(404, f"会话不存在: {session_id}")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话及其所有对话消息。

    如果删除的是当前活跃会话，自动切换到另一个会话（或清空状态）。
    """
    global active_session_id

    deleted = 0
    if _db_available:
        try:
            import db as _db_mod
            deleted = _db_mod.delete_session(session_id)
        except Exception as e:
            logger.warning(f"数据库删除会话失败: {e}")
    if not _db_available:
        try:
            deleted = _local_store.delete_local_session(session_id)
        except Exception as e:
            logger.warning(f"本地删除会话失败: {e}")

    # 从内存中移除
    session_histories.pop(session_id, None)

    # 如果删除的是活跃会话，清除状态
    if active_session_id == session_id:
        active_session_id = None
        if kv_cache:
            kv_cache.clear()
        _init_kv_cache()

    logger.info(f"会话已删除: {session_id} ({deleted} DB rows)")
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/sessions/{session_id}/activate")
async def activate_session(session_id: str):
    """
    切换到指定会话，返回该会话的消息历史。
    """
    _switch_session(session_id)

    # 返回该会话的消息历史
    history = _get_active_history()
    return {
        "session_id": session_id,
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ],
        "count": len(history),
    }


@app.delete("/api/sessions/{session_id}/turns/{turn_index}")
async def delete_turn(session_id: str, turn_index: int):
    """
    删除指定会话中的单轮对话（user + assistant 两条消息）。

    turn_index: 0-based 对话轮次索引。
    """
    global kv_cache

    # 验证 turn_index 范围
    history = session_histories.get(session_id, [])
    if not history:
        # 尝试从 DB 或本地文件加载
        if _db_available:
            try:
                import db as _db_mod
                rows = _db_mod.get_conversation(session_id)
                history = [{"role": r["role"], "content": r["content"]} for r in rows]
                session_histories[session_id] = history
            except Exception:
                pass
        if not history:
            try:
                local_rows = _local_store.load_local_conversation(session_id)
                history = [{"role": r["role"], "content": r["content"]} for r in local_rows]
                session_histories[session_id] = history
            except Exception:
                pass
        if not history:
            raise HTTPException(404, f"会话不存在或无消息: {session_id}")

    max_turn = (len(history) // 2) - 1
    if turn_index < 0 or turn_index > max_turn:
        raise HTTPException(400, f"无效的轮次索引: {turn_index}（有效范围: 0-{max_turn}）")

    # 从 DB 或本地文件删除
    deleted_count = 0
    if _db_available:
        try:
            import db as _db_mod
            deleted_count = _db_mod.delete_message_range(session_id, turn_index)
            _db_mod.decrement_session_message_count(session_id, 2)
        except Exception as e:
            logger.warning(f"数据库删除消息失败: {e}")
    if not _db_available:
        try:
            deleted_count = _local_store.delete_local_message_range(session_id, turn_index)
            _local_store.decrement_local_session_message_count(session_id, 2)
        except Exception as e:
            logger.warning(f"本地删除消息失败: {e}")

    # 从内存中移除这两条消息
    idx = turn_index * 2
    if idx + 1 < len(history):
        del history[idx:idx + 2]

    # 如果删除的是活跃会话的轮次，清 KV Cache（token 位置已变）
    if session_id == active_session_id and kv_cache:
        kv_cache.clear()
        _init_kv_cache()

    remaining_turns = len(history) // 2
    logger.info(f"已删除会话 {session_id} 第 {turn_index} 轮对话（{deleted_count} DB rows），剩余 {remaining_turns} 轮")
    return {
        "status": "deleted",
        "session_id": session_id,
        "turn_index": turn_index,
        "deleted_count": deleted_count,
        "remaining_turns": remaining_turns,
    }


# ============================================================
# 数据库健康检查
# ============================================================

@app.get("/api/db/health")
async def database_health():
    """数据库连接健康检查"""
    if not _db_importable:
        return {"status": "unavailable", "reason": "driver_missing", "message": "psycopg2 未安装"}
    import scheduler as _scheduler_module
    runtime = _scheduler_module.get_database_status()
    if not runtime["available"]:
        if not runtime.get("configured", True):
            return {
                "status": "unavailable",
                "reason": "not_configured",
                "message": "数据库未配置，正在使用本地文件存储",
                "retry_in_seconds": 0,
            }
        return {
            "status": "unavailable",
            "reason": "connection_failed",
            "message": runtime.get("last_error") or "数据库未配置或暂不可用",
            "retry_in_seconds": runtime.get("retry_in_seconds", 0),
        }
    return db_health()


# ============================================================
# 生产模式：挂载 React 前端静态文件
# ============================================================
# 构建前端: cd frontend && npm run build （输出到 frontend/dist/）
# 生产模式下 FastAPI 在 8000 端口直接提供全部服务（无需 Vite dev server）
# 开发模式下 dist 目录不存在，跳过挂载，使用 Vite proxy 模式

# ================================================================
# 模型文件下载（供 Android 等远程节点下载 GGUF 模型）
# ================================================================

# GGUF 模型存放目录
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
if not os.path.isdir(_MODELS_DIR):
    _MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _require_trusted_model_peer(request: Request) -> None:
    from bootstrap import is_trusted_bootstrap_source

    peer_host = request.client.host if request.client else ""
    if not is_trusted_bootstrap_source(peer_host):
        raise HTTPException(403, "source network is not trusted")


def _active_pytorch_model() -> dict:
    info = scheduler._get_active_pipeline_model_info()
    if not info:
        raise HTTPException(409, "主节点当前未加载可分层的 PyTorch 模型")
    return info


def _model_file_sha256(path: str) -> str:
    from model_sync import compute_file_sha256

    return compute_file_sha256(path)


@app.get("/api/models/downloadable")
async def downloadable_pytorch_model(request: Request, model_id: str = ""):
    """Return the exact active PyTorch model manifest to Tailnet workers."""
    _require_trusted_model_peer(request)
    info = _active_pytorch_model()
    if model_id and model_id != info["model_id"]:
        raise HTTPException(409, "请求模型已不是主节点当前流水线模型")

    root = os.path.realpath(info["model_path"])
    files = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in sorted(filenames):
            if filename in {"model.sha256", "model.sha256.meta.json"} or filename.endswith(".part"):
                continue
            if not filename.lower().endswith((
                ".safetensors", ".bin", ".json", ".py", ".tiktoken",
                ".model", ".txt", ".jinja", ".spm", ".vocab",
            )):
                continue
            path = os.path.realpath(os.path.join(directory, filename))
            try:
                if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
                    continue
            except ValueError:
                continue
            relative_path = os.path.relpath(path, root).replace(os.sep, "/")
            files.append({
                "path": relative_path,
                "size_bytes": os.path.getsize(path),
                "sha256": _model_file_sha256(path),
            })
    return {
        "model_id": info["model_id"],
        "sha256": info["model_sha256"],
        "total_layers": info["total_layers"],
        "files": files,
        "count": len(files),
    }


@app.get("/api/models/files/{model_id}/{relative_path:path}")
async def download_pytorch_model_file(
    model_id: str,
    relative_path: str,
    request: Request,
):
    """Stream one active-model file to an admitted Tailnet pipeline worker."""
    _require_trusted_model_peer(request)
    info = _active_pytorch_model()
    if model_id != info["model_id"]:
        raise HTTPException(409, "请求模型已不是主节点当前流水线模型")
    root = os.path.realpath(info["model_path"])
    path = os.path.realpath(os.path.join(root, relative_path.replace("/", os.sep)))
    try:
        inside_root = os.path.commonpath([root, path]) == root
    except ValueError:
        inside_root = False
    if not inside_root or not os.path.isfile(path):
        raise HTTPException(404, "模型文件不存在")
    if os.path.basename(path) in {"model.sha256", "model.sha256.meta.json"} or path.endswith(".part"):
        raise HTTPException(404, "模型文件不存在")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=os.path.basename(path),
    )


@app.get("/api/models/gguf")
async def list_gguf_models():
    """
    列出可下载的 GGUF 模型文件及其 SHA256 校验值。

    Android 全有模式调用此接口获取可下载的模型列表和下载 URL。
    """
    models = []
    if not os.path.isdir(_MODELS_DIR):
        return {"models": models, "directory": _MODELS_DIR, "exists": False}

    for fname in sorted(os.listdir(_MODELS_DIR)):
        if not fname.lower().endswith(".gguf"):
            continue
        fpath = os.path.join(_MODELS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        size = os.path.getsize(fpath)

        # 读取或计算 SHA256（优先读已有 .sha256 文件）
        sha256 = ""
        sha256_file = fpath + ".sha256"
        if os.path.isfile(sha256_file):
            try:
                with open(sha256_file, "r") as f:
                    sha256 = f.read().strip().split()[0]
            except Exception:
                pass
        if not sha256:
            try:
                sha256 = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                sha256 = sha256.hexdigest()
                # 缓存到 .sha256 文件
                with open(sha256_file, "w") as f:
                    f.write(f"{sha256}  {fname}\n")
            except Exception:
                sha256 = ""

        models.append({
            "filename": fname,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 1),
            "sha256": sha256,
            "download_url": f"/api/models/download/{fname}",
        })

    return {
        "models": models,
        "directory": os.path.abspath(_MODELS_DIR),
        "exists": True,
        "count": len(models),
    }


@app.get("/api/models/download/{filename}")
async def download_model_file(filename: str):
    """
    下载 GGUF 模型文件（支持 Range 断点续传）。

    Android ModelManager 调用此接口下载模型，支持分段下载和断点续传。
    """
    # 安全检查：防止路径穿越
    safe_name = os.path.basename(filename)
    if safe_name != filename or ".." in filename:
        raise HTTPException(400, "无效的文件名")

    if not safe_name.lower().endswith(".gguf"):
        raise HTTPException(400, "仅支持 .gguf 模型文件下载")

    file_path = os.path.join(_MODELS_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(404, f"模型文件不存在: {safe_name}")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=safe_name,
    )


# ============================================================
# 日志管理 API
# ============================================================


_LOG_FILE_RE = re.compile(r"^[^/\\]+\.log(?:\.\d+)?$")
_LOG_FILE_LOCK = threading.RLock()
_LOCAL_LOG_CLIENTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "testclient"}


def _get_request_client(request: Request) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client else ""
    return host or "unknown"


def _get_effective_role_safe() -> str:
    try:
        return scheduler._effective_role()
    except Exception:
        return "unknown"


def _require_log_api_access(request: Request) -> str:
    """
    L0 安全边界：日志可能包含隐私与调试细节，默认只允许本机访问。

    远程管理员访问需要显式配置 QLH_LOG_ADMIN_TOKEN，并在请求头中传入
    X-QLH-Log-Token。当前项目还没有 Web 登录态，所以不能仅凭 master
    进程角色放行 LAN 浏览器。
    """
    client_host = _get_request_client(request)
    if client_host in _LOCAL_LOG_CLIENTS:
        return client_host

    admin_token = os.environ.get("QLH_LOG_ADMIN_TOKEN", "").strip()
    request_token = request.headers.get("X-QLH-Log-Token", "").strip()
    if admin_token and request_token and request_token == admin_token:
        return f"{client_host}:admin-token"

    role = _get_effective_role_safe()
    logger.warning(
        "拒绝日志接口访问: client=%s role=%s path=%s",
        client_host, role, request.url.path,
    )
    raise HTTPException(403, "日志接口仅允许本机访问；远程访问需管理员授权")


def _log_admin_action(action: str, requester: str, target: str, status: str,
                      error: str = "") -> None:
    role = _get_effective_role_safe()
    if error:
        logger.warning(
            "日志管理操作: action=%s requester=%s role=%s target=%s status=%s error=%s",
            action, requester, role, target, status, error,
        )
    else:
        logger.info(
            "日志管理操作: action=%s requester=%s role=%s target=%s status=%s",
            action, requester, role, target, status,
        )


def _snapshot_recent_logs() -> tuple[list[dict], int]:
    with _log_buffer_lock:
        return [dict(item) for item in _log_buffer], _log_buffer_total_seen


def _normalize_log_limit(limit: int) -> int:
    return max(1, min(int(limit or 200), 1000))


def _filter_recent_logs(entries: list[dict], level: str = "", name: str = "",
                        node_id: str = "", request_id: str = "") -> list[dict]:
    level = (level or "").strip().upper()
    name = (name or "").strip()
    node_id = (node_id or "").strip()
    request_id = (request_id or "").strip()

    if level:
        levelno = logging._nameToLevel.get(level)
        if isinstance(levelno, int):
            entries = [item for item in entries if item.get("levelno", 0) >= levelno]
        else:
            entries = [item for item in entries if item.get("level", "").upper() == level]
    if name:
        entries = [item for item in entries if name in item.get("name", "")]
    if node_id:
        entries = [item for item in entries if item.get("node_id") == node_id]
    if request_id:
        entries = [item for item in entries if item.get("request_id") == request_id]
    return entries


def _is_log_filename(filename: str) -> bool:
    """允许普通 .log 和 RotatingFileHandler 生成的 .log.N 备份。"""
    return (
        filename == os.path.basename(filename)
        and ".." not in filename
        and _LOG_FILE_RE.fullmatch(filename) is not None
    )


def _validate_log_filename(filename: str) -> str:
    """只允许访问 LOG_DIR 下的单个日志文件。"""
    if not _is_log_filename(filename):
        raise HTTPException(400, "无效的日志文件名")
    return filename


@app.get("/api/logs")
async def list_log_files(request: Request):
    """列出 LOG_DIR 中所有 .log 文件，按修改时间降序。"""
    from config import LOG_DIR
    from datetime import datetime

    _require_log_api_access(request)

    files = []
    with _LOG_FILE_LOCK:
        if not os.path.isdir(LOG_DIR):
            return {"files": []}

        for fname in os.listdir(LOG_DIR):
            if not _is_log_filename(fname):
                continue
            fpath = os.path.join(LOG_DIR, fname)
            try:
                st = os.stat(fpath)
                files.append({
                    "name": fname,
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "_mtime": st.st_mtime,
                })
            except OSError:
                continue
    files.sort(key=lambda item: item["_mtime"], reverse=True)
    for item in files:
        item.pop("_mtime", None)
    return {"files": files}


@app.get("/api/logs/recent")
async def get_recent_logs(
    request: Request,
    limit: int = 200,
    level: str = "",
    name: str = "",
    node_id: str = "",
    request_id: str = "",
):
    """读取内存环形缓冲中的最近日志，不读取日志文件。"""
    _require_log_api_access(request)
    limit = _normalize_log_limit(limit)
    entries, total_seen = _snapshot_recent_logs()
    filtered = _filter_recent_logs(entries, level, name, node_id, request_id)
    result = filtered[-limit:]
    return {
        "logs": result,
        "count": len(result),
        "matched": len(filtered),
        "limit": limit,
        "buffer_size": len(entries),
        "buffer_capacity": _LOG_BUFFER_MAXLEN,
        "total_seen": total_seen,
        "truncated": len(filtered) > limit,
        "filters": {
            "level": level or None,
            "name": name or None,
            "node_id": node_id or None,
            "request_id": request_id or None,
        },
    }


@app.get("/api/logs/stats")
async def get_log_stats(request: Request):
    """返回日志文件与内存缓冲区统计信息。"""
    from config import LOG_DIR

    _require_log_api_access(request)
    entries, total_seen = _snapshot_recent_logs()
    level_counts = Counter(item.get("level", "UNKNOWN") for item in entries)
    logger_counts = Counter(item.get("name", "unknown") for item in entries)
    node_counts = Counter(item.get("node_id", "unknown") for item in entries)

    files = []
    total_file_bytes = 0
    with _LOG_FILE_LOCK:
        if os.path.isdir(LOG_DIR):
            for fname in os.listdir(LOG_DIR):
                if not _is_log_filename(fname):
                    continue
                try:
                    st = os.stat(os.path.join(LOG_DIR, fname))
                    total_file_bytes += st.st_size
                    files.append({
                        "name": fname,
                        "size": st.st_size,
                        "modified": st.st_mtime,
                    })
                except OSError:
                    continue

    return {
        "log_dir": LOG_DIR,
        "files_count": len(files),
        "files_total_bytes": total_file_bytes,
        "buffer_size": len(entries),
        "buffer_capacity": _LOG_BUFFER_MAXLEN,
        "buffer_total_seen": total_seen,
        "buffer_dropped_estimate": max(0, total_seen - len(entries)),
        "levels": dict(level_counts),
        "loggers": dict(logger_counts.most_common(20)),
        "nodes": dict(node_counts),
        "node_id": _current_node_id_safe(),
        "device_ip": _current_device_ip_safe(),
    }


@app.get("/api/logs/download")
async def download_log_file(request: Request, name: str):
    """下载单个日志文件。"""
    from config import LOG_DIR

    _require_log_api_access(request)
    safe_name = _validate_log_filename(name)
    file_path = os.path.join(LOG_DIR, safe_name)
    with _LOG_FILE_LOCK:
        if not os.path.isfile(file_path):
            raise HTTPException(404, "文件不存在")
    return FileResponse(
        file_path,
        media_type="text/plain; charset=utf-8",
        filename=safe_name,
    )


# ★ 通配路由 /api/logs/{filename:path} 移到最后，避免抢占 /api/logs/export、/api/logs/node/* 等特定路由


@app.delete("/api/logs")
async def delete_all_log_files(request: Request):
    """删除 LOG_DIR 中所有 .log 文件。"""
    from config import LOG_DIR

    requester = _require_log_api_access(request)

    deleted = []
    failed = []
    with _LOG_FILE_LOCK:
        if not os.path.isdir(LOG_DIR):
            return {"status": "ok", "deleted": [], "failed": []}

        # 仅关闭文件 handler，保留终端+内存 handler（避免删除期间日志丢失）
        _close_logging_handlers(keep_memory=True)
        try:
            for fname in os.listdir(LOG_DIR):
                if not _is_log_filename(fname):
                    continue
                try:
                    os.remove(os.path.join(LOG_DIR, fname))
                    deleted.append(fname)
                except OSError as e:
                    failed.append({"name": fname, "error": str(e)})
        finally:
            setup_logging()

    status = "ok" if not failed else "partial"
    _log_admin_action(
        "delete_all",
        requester,
        "*",
        status,
        "; ".join(f"{item['name']}: {item['error']}" for item in failed),
    )
    return {
        "status": status,
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
    }


# ============================================================
# L5: 日志压缩包导出
# ============================================================

@app.get("/api/logs/export")
async def export_logs_zip(request: Request):
    """将所有 .log 文件打包为 ZIP 并下载。"""
    import zipfile
    import io
    from config import LOG_DIR
    from datetime import datetime

    _require_log_api_access(request)

    # 在锁内收集文件列表，在锁外构建 ZIP（避免大 I/O 时阻塞其他日志操作）
    with _LOG_FILE_LOCK:
        if not os.path.isdir(LOG_DIR):
            raise HTTPException(404, "日志目录不存在")
        log_files = sorted(
            f for f in os.listdir(LOG_DIR) if _is_log_filename(f)
        )

    buf = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in log_files:
            fpath = os.path.join(LOG_DIR, fname)
            try:
                zf.write(fpath, fname)
                file_count += 1
            except OSError as e:
                logger.warning("日志导出跳过 %s: %s", fname, e)

    if file_count == 0:
        raise HTTPException(404, "没有可导出的日志文件")

    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    node_id = _current_node_id_safe() or "node"
    filename = f"qlh-logs-{node_id}-{timestamp}.zip"

    logger.info(
        "event=log_export files_count=%d requester=%s node_id=%s",
        file_count, _get_request_client(request), node_id,
    )
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# L5: 前端错误上报
# ============================================================

class ClientErrorReport(BaseModel):
    message: str = ""
    source: str = ""        # 错误来源: "window.onerror" | "unhandledrejection" | "manual"
    stack: str = ""          # 堆栈跟踪
    url: str = ""            # 发生错误的页面 URL
    line: int = 0            # 行号
    col: int = 0             # 列号
    user_agent: str = ""     # 浏览器 UA
    extra: dict = Field(default_factory=dict)  # 附加上下文（如 session_id、当前操作）


def _truncate_log_field(value, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


@app.post("/api/logs/client-error")
async def report_client_error(report: ClientErrorReport, request: Request):
    """接收前端错误报告并写入后端诊断日志。"""
    client_host = _get_request_client(request)
    request_id = str(_request_id_ctx.get("-") or "")
    logger.error(
        "event=client_error source=%s message=%s url=%s line=%d col=%d "
        "client=%s ua=%s stack=%s extra=%s request_id=%s",
        _truncate_log_field(report.source, 80),
        _truncate_log_field(report.message, 500),
        _truncate_log_field(report.url, 300),
        report.line,
        report.col,
        client_host,
        _truncate_log_field(report.user_agent or "-", 200),
        _truncate_log_field(report.stack, 2000),
        _truncate_log_field(
            json.dumps(report.extra or {}, ensure_ascii=False, default=str),
            500,
        ),
        request_id,
    )
    return {"status": "ok", "logged": True}


# ============================================================
# L5: 日志保留策略清理
# ============================================================

_log_retention_thread_started = False


def _run_log_retention_cleanup() -> None:
    """
    按 config.LOG_MAX_AGE_DAYS 和 LOG_MAX_TOTAL_SIZE_MB 清理旧日志。

    策略：
    1. 先按天数删除过期文件（mtime < now - LOG_MAX_AGE_DAYS）
    2. 再按总空间删除最旧文件（total > LOG_MAX_TOTAL_SIZE_MB）
    3. 不会删除当天日志文件
    """
    from config import LOG_DIR, LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB
    from datetime import datetime, timedelta

    if LOG_MAX_AGE_DAYS <= 0 and LOG_MAX_TOTAL_SIZE_MB <= 0:
        return

    with _LOG_FILE_LOCK:
        if not os.path.isdir(LOG_DIR):
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        cutoff_time = now - timedelta(days=LOG_MAX_AGE_DAYS) if LOG_MAX_AGE_DAYS > 0 else None
        max_bytes = LOG_MAX_TOTAL_SIZE_MB * 1024 * 1024 if LOG_MAX_TOTAL_SIZE_MB > 0 else 0

        # 收集所有日志文件信息
        files_info = []
        for fname in os.listdir(LOG_DIR):
            if not _is_log_filename(fname):
                continue
            fpath = os.path.join(LOG_DIR, fname)
            try:
                st = os.stat(fpath)
                files_info.append({
                    "name": fname,
                    "path": fpath,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
            except OSError:
                continue

        deleted_age = 0
        deleted_size = 0
        kept = []

        # 第一阶段：按天数清理
        for fi in files_info:
            mtime_dt = datetime.fromtimestamp(fi["mtime"])
            # 不删除当天日志
            if fi["name"].startswith(f"qlh-{today_str}"):
                kept.append(fi)
                continue
            if cutoff_time and mtime_dt < cutoff_time:
                try:
                    os.remove(fi["path"])
                    deleted_age += 1
                except OSError:
                    kept.append(fi)
            else:
                kept.append(fi)

        # 第二阶段：按总空间清理（最旧优先，但跳过当天日志）
        if max_bytes > 0:
            total_size = sum(f["size"] for f in kept)
            if total_size > max_bytes:
                # 按修改时间升序（最旧的在前）
                kept.sort(key=lambda f: f["mtime"])
                for fi in kept:
                    if total_size <= max_bytes:
                        break
                    if fi["name"].startswith(f"qlh-{today_str}"):
                        continue
                    try:
                        os.remove(fi["path"])
                        total_size -= fi["size"]
                        deleted_size += 1
                    except OSError:
                        pass

        if deleted_age > 0 or deleted_size > 0:
            remaining = len(kept) - deleted_size
            remaining_bytes = total_size if max_bytes > 0 else sum(f["size"] for f in kept)
            logger.info(
                "event=log_retention_cleanup deleted_age=%d deleted_size=%d "
                "remaining_files=%d total_size_mb=%.1f",
                deleted_age, deleted_size,
                remaining,
                remaining_bytes / (1024 * 1024),
            )


def _start_log_retention_thread() -> None:
    """启动日志保留策略后台线程（仅启动一次）。"""
    global _log_retention_thread_started
    if _log_retention_thread_started:
        return

    from config import (
        LOG_RETENTION_CHECK_INTERVAL, LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB,
    )

    # 双维度禁用时跳过启动（无清理任务可执行）
    if LOG_MAX_AGE_DAYS <= 0 and LOG_MAX_TOTAL_SIZE_MB <= 0:
        return

    _log_retention_thread_started = True

    def _retention_loop() -> None:
        # 启动后等待 5 分钟再首次清理（避免干扰初始化）
        time.sleep(300)
        while True:
            try:
                _run_log_retention_cleanup()
            except Exception:
                logger.warning("日志保留清理异常", exc_info=True)
            time.sleep(LOG_RETENTION_CHECK_INTERVAL)

    t = threading.Thread(target=_retention_loop, daemon=True, name="log-retention")
    t.start()
    logger.info(
        "日志保留策略已启动: max_age_days=%d max_total_mb=%d interval_s=%d",
        LOG_MAX_AGE_DAYS, LOG_MAX_TOTAL_SIZE_MB, LOG_RETENTION_CHECK_INTERVAL,
    )


# ============================================================
# L5: 多节点日志聚合 API
# ============================================================

@app.get("/api/logs/node/{node_id}/recent")
async def get_node_recent_logs(
    node_id: str,
    request: Request,
    limit: int = 100,
    level: str = "",
    name: str = "",
    timeout: float = 5.0,
):
    """
    从指定从节点拉取最近日志（主节点代理）。

    仅主节点可调用；向该节点发送 LOG_REQUEST TCP 消息并等待响应。
    """
    _require_log_api_access(request)

    role = _get_effective_role_safe()
    if role != "master":
        raise HTTPException(403, "仅主节点可拉取从节点日志")

    # 如果是本节点（或查询的是自己），直接返回本地 recent logs
    local_node_id = scheduler.get_effective_node_id()
    if node_id == local_node_id or node_id == "master":
        entries, _ = _snapshot_recent_logs()
        filtered = _filter_recent_logs(entries, level, name, node_id="", request_id="")
        result_slice = filtered[-limit:]
        return {
            "node_id": local_node_id,
            "source": "local",
            "logs": result_slice,
            "count": len(result_slice),
            "matched": len(filtered),
            "buffer_size": len(entries),
        }

    # 远程节点：通过 scheduler.request_node_logs 走 TCP
    result = scheduler.request_node_logs(
        node_id=node_id,
        limit=limit,
        level=level,
        name=name,
        timeout=timeout,
    )
    if result is None:
        raise HTTPException(
            504,
            f"无法从节点 {node_id} 获取日志：节点不在线或超时 ({timeout}s)",
        )

    result["source"] = "remote"
    return result


@app.get("/api/logs/nodes-summary")
async def get_nodes_log_summary(request: Request):
    """
    返回所有在线从节点的日志概要（文件数、大小、buffer 状态）。

    仅主节点可调用。不拉取完整日志内容，仅返回每个节点的统计摘要。
    """
    _require_log_api_access(request)

    role = _get_effective_role_safe()
    if role != "master":
        raise HTTPException(403, "仅主节点可查看集群日志概要")

    # 本地节点统计
    local_entries, _ = _snapshot_recent_logs()
    nodes_summary = {
        "local": {
            "node_id": scheduler.get_effective_node_id(),
            "buffer_size": len(local_entries),
            "buffer_capacity": _LOG_BUFFER_MAXLEN,
        },
        "workers": [],
    }

    from scheduler import NodeRole, NodeState

    # 对所有在线从节点拉取统计（快速超时）
    with scheduler._nodes_lock:
        online_workers = [
            (nid, info)
            for nid, info in scheduler.nodes.items()
            if info.role != NodeRole.MASTER
            and info.state == NodeState.ONLINE
        ]

    for nid, _info in online_workers:
        try:
            result = scheduler.request_node_logs(
                node_id=nid, limit=5, timeout=3.0,
            )
            if result:
                nodes_summary["workers"].append({
                    "node_id": nid,
                    "buffer_size": result.get("buffer_size", 0),
                    "sample_count": result.get("count", 0),
                })
            else:
                nodes_summary["workers"].append({
                    "node_id": nid,
                    "buffer_size": 0,
                    "error": "timeout",
                })
        except Exception as e:
            nodes_summary["workers"].append({
                "node_id": nid,
                "buffer_size": 0,
                "error": str(e)[:100],
            })

    nodes_summary["total_workers"] = len(online_workers)
    return nodes_summary


# ★ 通配路由必须放在所有特定 /api/logs/* 路由之后，避免抢占


@app.get("/api/logs/{filename:path}")
async def read_log_file(filename: str, request: Request):
    """读取指定日志文件的内容（最多返回末 1 MB）。"""
    from config import LOG_DIR

    _require_log_api_access(request)
    safe_name = _validate_log_filename(filename)
    max_bytes = 1024 * 1024  # 1 MB
    try:
        file_path = os.path.join(LOG_DIR, safe_name)
        # 锁内仅做存在性检查和大小获取，锁外读取文件内容（避免阻塞其他日志操作）
        with _LOG_FILE_LOCK:
            if not os.path.isfile(file_path):
                raise HTTPException(404, "文件不存在")
            file_size = os.path.getsize(file_path)

        with open(file_path, "rb") as f:
            # 重新获取实际文件大小（锁外可能已被轮转截断）
            f.seek(0, os.SEEK_END)
            actual_size = f.tell()
            truncated = actual_size > max_bytes
            if truncated:
                f.seek(max(0, actual_size - max_bytes))
                f.readline()  # 跳过不完整首行
            else:
                f.seek(0)
            content = f.read().decode("utf-8", errors="replace")
        return {"name": safe_name, "content": content, "truncated": truncated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"读取失败: {e}")


@app.delete("/api/logs/{filename:path}")
async def delete_log_file(filename: str, request: Request):
    """删除指定的 .log 文件。"""
    from config import LOG_DIR

    requester = _require_log_api_access(request)
    safe_name = _validate_log_filename(filename)
    error_msg = ""
    with _LOG_FILE_LOCK:
        file_path = os.path.join(LOG_DIR, safe_name)
        if not os.path.isfile(file_path):
            raise HTTPException(404, "文件不存在")

        # 仅关闭文件 handler，保留终端+内存 handler（避免删除期间日志丢失）
        _close_logging_handlers(keep_memory=True)
        try:
            os.remove(file_path)
        except Exception as e:
            error_msg = str(e)
        finally:
            setup_logging()

    if error_msg:
        _log_admin_action("delete", requester, safe_name, "failed", error_msg)
        raise HTTPException(500, f"删除失败: {error_msg}")

    _log_admin_action("delete", requester, safe_name, "ok")
    return {"status": "ok", "deleted": safe_name, "failed": []}


# PyInstaller 打包后前端文件在 sys._MEIPASS/frontend/dist/ 下
if getattr(sys, 'frozen', False):
    _frontend_dist = os.path.join(sys._MEIPASS, "frontend", "dist")
else:
    _frontend_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
    logger.info(f"前端静态文件已挂载: {_frontend_dist}")
else:
    logger.info("前端 dist 目录未找到，使用纯 API 模式（开发时由 Vite 提供前端）")

# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    # 启动前检查模型文件（静默模式，仅日志提示）
    from model_downloader import ensure_model_or_warn
    ensure_model_or_warn()

    logger.info("启动 API 服务器...")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")
