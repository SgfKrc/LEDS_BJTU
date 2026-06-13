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

import json
import logging
import re
import time
import sys
import os
from typing import Optional

import torch

# 确保 src 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_module import ModelManager
from paged_kv_cache import PagedKVCache
from device_profiler import DeviceProfiler, get_profile
from scheduler import Scheduler
from config import (
    MODEL_NAME, MODEL_PATH, QUANT_TYPE, USE_COMPILE,
    DEVICE, PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN, RUN_MODE,
    NODE_ROLE, NODE_ID, MAX_NODES,
)

# 数据库模块（可选，未安装 psycopg2 时使用内存降级）
try:
    from db import init_db, close_db, db_health
    _db_available = True
except ImportError:
    _db_available = False
    init_db = lambda: None
    close_db = lambda: None
    db_health = lambda: {"status": "unavailable", "message": "psycopg2 未安装"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("api_server")

# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="轻量化大模型分布式边缘推理优化系统",
    version="0.1.0",
    description="北京交通大学 · 大学生创新创业训练计划",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173",
                   "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 全局状态
# ============================================================

model_manager = ModelManager()
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
generation_config: dict = {
    "max_new_tokens": 1024,          # laptop 档默认值
    "tier_max_new_tokens": 1024,     # 设备档位上限（auto_configure 后更新）
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# 调度器（单机 / 分布式模式共用）
scheduler: Scheduler = Scheduler()


# ============================================================
# 启动事件 — 设备检测
# ============================================================

@app.on_event("startup")
async def startup_device_detection():
    """启动时自动检测设备能力并缓存画像，初始化调度器"""
    global device_profile, scheduler
    try:
        profiler = get_profile()
        device_profile = profiler.to_dict()
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

    # 初始化调度器（单机模式下不启动 TCP 监听）
    try:
        scheduler.start()
        logger.info(f"调度器已初始化: mode={RUN_MODE}")
    except Exception as e:
        logger.error(f"调度器初始化失败: {e}")

    # 初始化数据库连接
    try:
        init_db()
        # ★ 设置数据隔离参数：conversations/sessions 将按 node_id 过滤
        from db import set_active_node_id
        set_active_node_id(scheduler.get_effective_node_id())
        logger.info(f"数据库已连接，活跃节点: {scheduler.get_effective_node_id()}")
    except Exception as e:
        logger.warning(f"数据库初始化失败（使用内存降级）: {e}")


@app.on_event("shutdown")
async def shutdown_db():
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


# ============================================================
# Pydantic 模型
# ============================================================

class LoadModelRequest(BaseModel):
    quant_type: str = Field(
        default="int4",
        description="量化精度: fp16 | int8 | int4",
    )
    use_compile: bool = Field(
        default=False,
        description="是否开启 torch.compile 算子融合（仅 FP16 有效）",
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: Optional[str] = Field(default=None, description="会话ID，为空时使用当前活跃会话")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    show_thinking: bool = Field(default=False, description="启用深度思考展示")


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    thinking_content: Optional[str] = None
    metrics: dict = {}
    followups: list[str] = []


class NodeDetail(BaseModel):
    node_id: str
    role: str
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


class UpdateMaxNodesRequest(BaseModel):
    max_nodes: int = Field(..., ge=1, le=64, description="新的最大节点数（包含 master）")


class ConnectToMasterRequest(BaseModel):
    master_host: str = Field(..., description="主节点 IP 地址", min_length=1)
    master_port: int = Field(8888, ge=1, le=65535, description="主节点端口")


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

    # ---- 情况2：格式未遵循（缺少标记或标记顺序错误） ----
    # 清理所有思考标记，返回干净的文本作为答案
    cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
    # 清理常见的标题前缀
    cleaned = _re.sub(r'^分析思路[：:]\s*', '', cleaned)
    cleaned = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', cleaned, flags=_re.IGNORECASE)
    return cleaned, None


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

    # 如果目标会话不在内存中，尝试从 DB 加载
    if target_id not in session_histories:
        messages = []
        if _db_available:
            try:
                import db as _db_mod
                rows = _db_mod.get_conversation(target_id)
                messages = [{"role": r["role"], "content": r["content"]} for r in rows]
            except Exception:
                pass
        session_histories[target_id] = messages

    # 清 KV Cache（切换会话后 prompt 不同，必须重建）
    if kv_cache:
        kv_cache.clear()
    _init_kv_cache()
    logger.info(f"已切换到会话: {target_id}")


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

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
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
    else:
        kv_cache = PagedKVCache(
            page_size=PAGE_SIZE,
            max_pages=MAX_PAGE_NUM,
            device=str(model_manager.get_device()),
            dtype=torch.float16,
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
        try:
            profiler = get_profile()
            device_profile = profiler.to_dict()
        except Exception as e:
            raise HTTPException(500, f"设备检测失败: {e}")
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

    return {
        "model_loaded": model_loaded,
        "current_quant": current_quant,
        "use_compile": USE_COMPILE if model_loaded else False,
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "run_mode": RUN_MODE,
        "node_role": NODE_ROLE,
        "node_id": NODE_ID,
        "max_nodes": MAX_NODES,
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
        return {"loaded": False, "quant_type": None}

    info = model_manager.get_model_info()
    mem = model_manager.get_memory_usage()
    return {
        "loaded": True,
        "quant_type": current_quant,
        "model_name": MODEL_NAME,
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
    """
    global model_loaded, current_quant, kv_cache, conversation_stats

    quant = req.quant_type.lower()
    if quant not in ("fp16", "int8", "int4"):
        raise HTTPException(400, f"不支持的量化类型: {quant}，可选: fp16, int8, int4")

    try:
        t0 = time.time()

        # 卸载旧模型（双引擎兼容）
        if model_manager.is_loaded:
            logger.info("卸载旧模型...")
            if model_manager._engine_type == "llama_cpp" and model_manager._llama_engine:
                model_manager._llama_engine.unload()
                model_manager._llama_engine = None
            elif model_manager.model is not None:
                model_manager.model = None
                model_manager.tokenizer = None
            model_manager._engine_type = ""
            if kv_cache:
                kv_cache.clear()
            kv_cache = None
            conversation_stats = {
                "total_prompt_tokens": 0,
                "total_generated_tokens": 0,
                "total_time_seconds": 0.0,
                "rounds": 0,
            }
            # 同步清空数据库对话历史
            if _db_available:
                try:
                    import db as _db_mod
                    _db_mod.clear_conversation("default")
                except Exception:
                    pass
            model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # 临时修改 config
        import config as cfg
        cfg.QUANT_TYPE = quant
        cfg.USE_COMPILE = req.use_compile

        # 加载新模型（传入设备画像以启用自适应加载）
        logger.info(f"加载模型: quant={quant}, compile={req.use_compile}")
        model_manager.load_model(
            model_path=MODEL_PATH,
            quant_type=quant,
            profile=device_profile,
        )

        # 初始化 KV 缓存
        _init_kv_cache()
        conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }

        model_loaded = True
        current_quant = quant
        generation_config["use_compile"] = req.use_compile

        elapsed = time.time() - t0

        status = await get_status()
        status["load_time_seconds"] = round(elapsed, 1)

        logger.info(f"模型加载完成 ({elapsed:.1f}s): {quant}")
        return status

    except Exception as e:
        model_loaded = False
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(500, f"模型加载失败: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    发送消息并获取模型回复（多轮对话）。

    自动维护对话历史 + KV 缓存。
    """
    global kv_cache, conversation_stats

    if not model_loaded or not model_manager.is_loaded:
        raise HTTPException(400, "模型未加载，请先在控制面板中选择并加载模型")

    # ---- 多会话支持：确定目标会话并切换（分布式 & 本地共用） ----
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
            and NODE_ROLE == "client"):
        try:
            result = scheduler.forward_inference_to_master(
                message=req.message,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                show_thinking=req.show_thinking,
                session_id=req.session_id,
            )
            if result.get("status") == "ok":
                # 保存到本地对话历史
                history.append({"role": "user", "content": req.message})
                response_text = result.get("content", "")
                history.append({"role": "assistant", "content": response_text})

                # 持久化到数据库
                db_session_id = target_session_id or "default"
                if _db_available:
                    try:
                        import db as _db_mod
                        if _db_mod.get_save_history():
                            _db_mod.save_message(db_session_id, "user", req.message)
                            _db_mod.save_message(db_session_id, "assistant", response_text,
                                                {"engine": "distributed"})
                            _db_mod.increment_session_message_count(db_session_id)
                    except Exception:
                        pass

                # 累计对话统计
                conversation_stats["rounds"] += 1

                # 更新调度器任务计数（后台管理面板显示）
                try:
                    scheduler.record_task_complete(success=True)
                except Exception:
                    pass

                # 生成追问建议（优先主节点返回的，否则本地模型生成）
                master_followups = result.get("followups", [])
                if master_followups:
                    followups = master_followups[:3]
                elif model_manager._engine_type == "llama_cpp":
                    followups = _generate_followups_llama(history)
                elif model_manager._engine_type == "pytorch":
                    try:
                        followups = _generate_followups(
                            history, model_manager.tokenizer,
                            model_manager.model, model_manager.get_device()
                        )
                    except Exception:
                        followups = _fallback_followups(history, [])
                else:
                    followups = _fallback_followups(history, [])

                return ChatResponse(
                    content=response_text,
                    metrics={"engine": "distributed"},
                    followups=followups,
                )
            elif result.get("status") == "disconnected":
                logger.warning("分布式推理转发失败（未连接主节点），回退到本地推理")
            elif result.get("status") == "timeout":
                logger.warning("分布式推理转发超时，回退到本地推理")
            else:
                logger.warning(f"分布式推理转发失败: {result.get('error', 'unknown')}，回退到本地推理")
        except Exception as e:
            logger.warning(f"分布式推理转发异常: {e}，回退到本地推理")

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
            history.append({"role": "assistant", "content": response_text})
            tokens_per_sec = result.get("tokens_per_second", 0)
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            # 持久化到数据库（受 save_history 开关控制）
            db_session_id = target_session_id or "default"

            # 生成追问建议（在 DB 保存之前，以便持久化到 metrics 中）
            followups = _generate_followups_llama(history)

            if _db_available:
                try:
                    import db as _db_mod
                    if _db_mod.get_save_history():
                        _db_mod.save_message(db_session_id, "user", req.message)
                        _db_mod.save_message(db_session_id, "assistant", response_text,
                                            {"engine": "llama.cpp", "tokens_per_sec": round(tokens_per_sec, 1),
                                             "completion_tokens": completion_tokens,
                                             "followups": followups})
                        _db_mod.increment_session_message_count(db_session_id)
                except Exception:
                    pass

            # 累计对话统计
            conversation_stats["total_generated_tokens"] += completion_tokens
            conversation_stats["rounds"] += 1

            # 更新调度器任务计数（后台管理面板显示）
            try:
                scheduler.record_task_complete(success=True)
            except Exception:
                pass

            return ChatResponse(
                content=response_text,
                metrics={
                    "engine": "llama.cpp",
                    "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                    "completion_tokens": completion_tokens,
                },
                followups=followups,
            )
        except Exception as e:
            try:
                scheduler.record_task_error()
            except Exception:
                pass
            logger.error(f"llama.cpp 推理失败: {e}")
            raise HTTPException(500, f"推理失败: {str(e)}")

    # ---- PyTorch 引擎路径（CUDA/独显）----
    try:
        # 更新生成配置（应用设备档位上限，取请求值与档位值的较小者）
        tier_max = generation_config.get("tier_max_new_tokens", generation_config["max_new_tokens"])
        # 深度思考时需要额外 token 预算：思考+答案都需要空间，tier_max 也必须同步上调
        thinking_budget = 384 if req.show_thinking else 0
        effective_max = min(req.max_new_tokens + thinking_budget,
                            tier_max + thinking_budget,
                            4096)  # 模型硬上限
        generation_config["max_new_tokens"] = effective_max
        generation_config["temperature"] = req.temperature
        generation_config["top_p"] = req.top_p

        # 追加用户消息
        history.append({"role": "user", "content": req.message})

        # 构建 Qwen chat prompt（深度思考时注入思考系统提示 + 预填【思考】强制续写）
        thinking_prompt = THINKING_SYSTEM_PROMPT if req.show_thinking else None
        thinking_prefill = "【思考】\n" if req.show_thinking else None
        prompt = _build_chat_prompt(history,
                                    system_prompt=thinking_prompt,
                                    assistant_prefill=thinking_prefill)

        # Tokenize
        tokenizer = model_manager.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_manager.get_device())
        prompt_len = input_ids.shape[1]

        # 生成
        t0 = time.time()
        with torch.no_grad():
            outputs = model_manager.model.generate(
                input_ids,
                max_new_tokens=effective_max,
                temperature=req.temperature if req.temperature > 0 else 1.0,
                top_p=req.top_p,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        # 解码（仅提取新生成的部分）
        generated_ids = outputs[0][prompt_len:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 深度思考：分离思考内容和最终答案
        # prefill 的「【思考】\n」在 prompt 中，不在 generated_ids 里，
        # 需要手动补回以便解析器匹配 THINKING_START 标记
        thinking_content = None
        if req.show_thinking:
            parsed_text = "【思考】\n" + raw_text
            response_text, thinking_content = _parse_thinking_response(parsed_text)
            # 解析器已做清洗：answer 绝不包含思考标记；thinking_content=None 时 response_text 已清理
        else:
            response_text = raw_text

        # 追加助手回复到历史（仅存储干净的答案，不含思考标记）
        history.append({"role": "assistant", "content": response_text})

        # 性能指标
        new_tokens = len(generated_ids)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        metrics = {
            "prompt_tokens": prompt_len,
            "new_tokens": new_tokens,
            "total_tokens": prompt_len + new_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "gpu_memory_mb": round(torch.cuda.memory_allocated() / (1024**2), 1)
            if torch.cuda.is_available()
            else 0,
        }

        # 持久化到数据库（受 save_history 开关控制）
        db_session_id = target_session_id or "default"

        # 生成追问建议（在 DB 保存之前，以便持久化）
        followups = _generate_followups(
            history, tokenizer, model_manager.model, model_manager.get_device()
        )

        if _db_available:
            try:
                import db as _db_mod
                if _db_mod.get_save_history():
                    _db_mod.save_message(db_session_id, "user", req.message)
                    # 将 followups 嵌入 metrics，确保切换会话后能够恢复
                    save_metrics = dict(metrics)
                    save_metrics["followups"] = followups
                    _db_mod.save_message(db_session_id, "assistant", response_text, save_metrics)
                    _db_mod.increment_session_message_count(db_session_id)
            except Exception:
                pass

        # 累计对话统计
        conversation_stats["total_prompt_tokens"] += prompt_len
        conversation_stats["total_generated_tokens"] += new_tokens
        conversation_stats["total_time_seconds"] += elapsed
        conversation_stats["rounds"] += 1

        # 更新调度器任务计数（前端心跳/任务统计列显示）
        try:
            scheduler.record_task_complete(success=True)
        except Exception:
            pass

        logger.info(
            f"推理完成: {new_tokens} tokens / {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s"
        )

        return ChatResponse(
            content=response_text,
            thinking_content=thinking_content,
            metrics=metrics,
            followups=followups,
        )

    except torch.cuda.OutOfMemoryError:
        # OOM 恢复
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
        logger.error(f"推理异常: {e}")
        raise HTTPException(500, f"推理失败: {str(e)}")


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
    """列出可选模型配置"""
    return {
        "models": [
            {
                "id": "fp16",
                "name": "FP16 原版",
                "description": "原始精度，显存 ~3.5 GB，速度最快 (~53 tok/s)",
                "memory_gb": 3.5,
                "speed_tok_s": 53,
                "compile_support": True,
            },
            {
                "id": "int4",
                "name": "INT4 量化 ⭐",
                "description": "4-bit 量化，显存 ~1.8 GB，速度 ~29 tok/s（推荐边缘设备）",
                "memory_gb": 1.8,
                "speed_tok_s": 29,
                "compile_support": False,
            },
            {
                "id": "int8",
                "name": "INT8 量化",
                "description": "8-bit 量化，显存 ~2.3 GB，速度 ~10 tok/s",
                "memory_gb": 2.3,
                "speed_tok_s": 10,
                "compile_support": False,
            },
        ],
        "current": current_quant if model_loaded else None,
    }


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

    增加时自动创建新的 client 节点槽位；
    减少时仅移除离线且未注册的空位，保留在线节点。
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


@app.post("/api/cluster/connect")
async def connect_to_master(req: ConnectToMasterRequest):
    """
    从节点主动连接主节点（从节点的「连接主节点」按钮触发）。

    调用后本节点将通过 TCP 向指定主节点发起注册，
    注册成功后主节点的节点列表中将出现本节点。
    """
    result = scheduler.connect_to_master(req.master_host, req.master_port)
    if result.get("status") == "denied":
        raise HTTPException(403, result.get("reason", "仅从节点可连接主节点"))
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
    if NODE_ROLE == "master":
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
    start_layer: int = Field(..., ge=0, le=23, description="起始层（含）")
    end_layer: int = Field(..., ge=1, le=24, description="结束层（不含）")


class LayerOverrideRequest(BaseModel):
    assignments: list[LayerOverrideItem] = Field(..., min_items=1, description="分层覆盖列表")


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
    if not _db_available:
        targeted_history = session_histories.get(session_id, [])
        return {
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in targeted_history
            ],
            "count": len(targeted_history),
            "source": "memory",
        }

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

    # 持久化到数据库
    if _db_available:
        try:
            import db as _db_mod
            _db_mod.create_session(session_id, title)
        except Exception as e:
            logger.warning(f"数据库创建会话失败: {e}")

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

    # 降级：从内存构造
    mem_sessions = []
    for sid, hist in session_histories.items():
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
    # 降级
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
        # 尝试从 DB 加载
        if _db_available:
            try:
                import db as _db_mod
                rows = _db_mod.get_conversation(session_id)
                history = [{"role": r["role"], "content": r["content"]} for r in rows]
                session_histories[session_id] = history
            except Exception:
                pass
        if not history:
            raise HTTPException(404, f"会话不存在或无消息: {session_id}")

    max_turn = (len(history) // 2) - 1
    if turn_index < 0 or turn_index > max_turn:
        raise HTTPException(400, f"无效的轮次索引: {turn_index}（有效范围: 0-{max_turn}）")

    # 从 DB 删除
    deleted_count = 0
    if _db_available:
        try:
            import db as _db_mod
            deleted_count = _db_mod.delete_message_range(session_id, turn_index)
            _db_mod.decrement_session_message_count(session_id, 2)
        except Exception as e:
            logger.warning(f"数据库删除消息失败: {e}")

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
    if not _db_available:
        return {"status": "unavailable", "message": "psycopg2 未安装，使用内存降级模式"}
    return db_health()


# ============================================================
# 生产模式：挂载 React 前端静态文件
# ============================================================
# 构建前端: cd frontend && npm run build （输出到 frontend/dist/）
# 生产模式下 FastAPI 在 8000 端口直接提供全部服务（无需 Vite dev server）
# 开发模式下 dist 目录不存在，跳过挂载，使用 Vite proxy 模式

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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
