"""模型托管宿主（inference-svc 进程内）。

并行共存（§1.4）：本类在 inference-svc 独立进程内新建 ModelHost 实例，
与 api_server 进程内的 model_host 单例互不共享、互不影响。

1.1（本文件）：薄委托 ModelHost——其内部 _LazyModelManager 延迟 import
model_module，未加载模型时冷启动成本 <100ms（§2.3：model_module 子树
9.7s 全部发生在首次 load 请求时）。
1.2：把 api_server 数据面执行段（_execute_chat_full 等 8 函数）与
scheduler 流水线段（_run_pipeline 等）复制为本宿主方法，源文件保持
不动（复制迁移，禁改源）。

1.2a（已复制）：
  - _execute_task_worker_stage（含嵌套 run_model 闭包）→
    EngineHost.execute_task_worker_stage
  - 辅助纯函数：_format_model_response / _parse_thinking_response /
    _strip_native_thinking_tags（api_server.py:904-1088 复制，零改动）
"""
import json
import logging
import re
import threading
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# ---- 复制自 api_server.py:887-904（常量，保真不变） ----
THINKING_START = "【思考】"
THINKING_END = "【思考结束】"
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


# ---- 复制自 api_server.py:904-941（纯函数，保真不变） ----
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


# ---- 复制自 api_server.py:943-1074（纯函数，保真不变） ----
def _parse_thinking_response(text: str) -> tuple:
    """
    解析模型输出，分离思考内容和最终答案。

    （复制自 api_server._parse_thinking_response，含容错逻辑：
    缺少结束标记 → 智能分割；答案为空 → 提取思考最后一段；
    重复标记 → 使用第一次出现的有效标记对。）

    Returns:
        (answer_content, thinking_content)
    """
    import re as _re

    if not text:
        return "", None

    start_idx = text.find(THINKING_START)
    end_idx = text.find(THINKING_END)

    # ---- 情况1：标记成对且顺序正确 ----
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking = text[start_idx + len(THINKING_START):end_idx].strip()
        answer = text[end_idx + len(THINKING_END):].strip()

        thinking = _re.sub(r'^分析思路[：:]\s*', '', thinking)

        answer = _re.sub(r'^【最终答案】[：:]?\s*', '', answer)
        answer = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', answer, flags=_re.IGNORECASE)
        for _pat in [r'^\[你的最终回答[^\]]*\]\s*', r'^\[你的推理过程[^\]]*\]\s*',
                     r'^（推理内容）\s*', r'^（答案内容）\s*',
                     r'^（给用户的答案[^）]*）\s*']:
            answer = _re.sub(_pat, '', answer)

        answer = answer.replace(THINKING_START, "").replace(THINKING_END, "").strip()

        prefix = text[:start_idx].strip()
        if prefix:
            answer = prefix + ("\n" + answer if answer else "")

        if thinking:
            if not answer and thinking:
                paragraphs = thinking.split("\n")
                for p in reversed(paragraphs):
                    p = p.strip()
                    if p and len(p) > 10:
                        answer = p
                        break
                if not answer:
                    answer = thinking
            return answer, thinking

    # ---- 情况2：DeepSeek-R1 / Qwen3 本地格式 ----
    import re as _re2
    native_match = _re2.search(
        r'<\s*think\s*>(.*?)<\s*/\s*think\s*>',
        text,
        flags=_re2.DOTALL | _re2.IGNORECASE,
    )
    if native_match:
        thinking = native_match.group(1).strip()
        answer = text[:native_match.start()].strip()
        after_think = text[native_match.end():].strip()
        after_think = _re2.sub(r'<\s*/?\s*(?:response|answer)\s*>', '', after_think, flags=_re2.IGNORECASE)
        if after_think:
            answer = (answer + '\n' + after_think).strip() if answer else after_think
        response_match = _re2.search(
            r'<\s*(?:response|answer)\s*>(.*)',
            answer if answer else '',
            flags=_re2.DOTALL | _re2.IGNORECASE,
        )
        if response_match:
            answer = response_match.group(1).strip()
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

    # ---- 情况3：格式未遵循 ----
    cleaned = text.replace(THINKING_START, "").replace(THINKING_END, "").strip()
    cleaned = _strip_native_thinking_tags(cleaned)
    cleaned = _re.sub(r'^分析思路[：:]\s*', '', cleaned)
    cleaned = _re.sub(r'^(最终答案|回答|Answer)[：:]\s*', '', cleaned, flags=_re.IGNORECASE)
    return cleaned, None


# ---- 复制自 api_server.py:1076-1088（纯函数，保真不变） ----
def _format_model_response(text: str, show_thinking: bool,
                           native_thinking_prompt: bool = False) -> tuple[str, Optional[str]]:
    """Format generated text without exposing unfinished native reasoning."""
    if show_thinking:
        return _parse_thinking_response(text)
    if native_thinking_prompt and "</think>" not in (text or "").lower():
        return "", None
    return _strip_native_thinking_tags(text), None


# ---- 复制自 api_server.py:2225-2232（纯函数，保真不变） ----
def _chat_origin(req: "ChatRequest") -> str:
    """根据请求上报信息推断请求来源，用于 metrics 展示。"""
    if req.client_node_type == "android":
        return "android_http"
    if req.client_node_type == "pc":
        return "pc_http"
    return "web_http"


# ---- 复制自 api_server.py:2234-2259（宿主适配：scheduler 全局 → 参数注入） ----
def _augment_chat_metrics(
    metrics: Optional[dict],
    req: "ChatRequest",
    *,
    serving_node_id: str = "",
    distributed_enabled: bool = False,
    **defaults,
) -> dict:
    """补齐统一聊天 metrics 字段，不覆盖调度器已给出的真实执行信息。

    （api_server 版从 scheduler 全局取 serving_node_id / 分布式开关；
    本进程由 EngineHost 持有，经参数注入。）
    """
    result = dict(metrics or {})
    for key, value in defaults.items():
        result.setdefault(key, value)
    origin = _chat_origin(req)
    result.setdefault("request_origin", origin)
    result.setdefault("request_origin_node_id", req.client_node_id or "")
    result.setdefault("request_origin_node_type", req.client_node_type or "")
    result.setdefault("client_mode", req.client_mode or "")
    result.setdefault("client_app_variant", req.client_app_variant or "")
    result.setdefault("serving_node_id", serving_node_id)
    result.setdefault("distributed_requested", distributed_enabled)
    result.setdefault("distributed_used", False)
    result.setdefault("fallback", False)
    result.setdefault("fallback_reason", "")
    result.setdefault("workers_used", [])
    result.setdefault("layer_assignments", [])
    result.setdefault("request_id", _request_id_ctx.get("-"))
    result.setdefault("generation_id", req.generation_id or "")
    return result


# ---- 复制自 api_server.py:828-901（纯函数，保真不变） ----
def _build_chat_prompt(
    messages: list[dict],
    system_prompt: Optional[str] = None,
    assistant_prefill: Optional[str] = None,
) -> str:
    """
    使用 Qwen 的 chat template 构建对话 prompt。
    Qwen-1.8B-Chat 使用 <|im_start|>/<|im_end|> 格式。
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


def _build_model_chat_prompt(
    tokenizer,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    assistant_prefill: Optional[str] = None,
) -> str:
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


from .protocol import ChatRequest  # noqa: E402（置于辅助函数后保持文档头清晰）

logger = logging.getLogger("inference_service.engine_host")

# ---- 复制自 api_server.py:96（请求 ID 上下文，与 api_server 同机制） ----
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ---- 复制自 api_server.py:406-410（异常，保真不变） ----
class ChatGenerationCancelled(RuntimeError):
    def __init__(self, generation_id: str):
        self.generation_id = generation_id
        super().__init__(f"generation {generation_id} cancelled")


# ---- 复制自 api_server.py:459-464（纯函数，保真不变） ----
def _raise_if_generation_cancelled(
    cancel_event: Optional[threading.Event], generation_id: Optional[str],
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ChatGenerationCancelled(generation_id or "gen_unknown")


# ---- 复制自 api_server.py:1178-1226（纯函数，保真不变） ----
def _is_question(text: str) -> bool:
    """
    判断文本是否为真正的疑问句，而非陈述句。

    （复制自 api_server._is_question：必须以问号结尾 + 含疑问指示词
    + 拒绝陈述句式关键词 + 拒绝列举开头。）
    """
    text = text.strip()
    if not text:
        return False

    if not (text.endswith('？') or text.endswith('?')):
        return False

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

    if re.match(r'^[\d]+[\.\、\)）]', text):
        return False

    return True


# ---- 复制自 api_server.py:1438-1573（纯函数，保真不变） ----
def _fallback_followups(history: list, existing: list[str]) -> list[str]:
    """
    基于对话关键词匹配的追问模板兜底。

    （复制自 api_server._fallback_followups：关键词 → 追问模板映射，
    按优先级排序，去重后补足至 3 条。）
    """
    last_assistant = ""
    last_user = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]

    combined = (last_user + " " + last_assistant).lower()

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

    default_templates = [
        "能再详细解释一下吗？",
        "这个结论有什么前提条件或局限性？",
        "有没有相关的参考资料或论文推荐？",
        "实际应用中需要注意哪些细节？",
        "能举一个具体的例子说明吗？",
    ]

    result = list(existing)
    candidate_pool = templates + default_templates
    for q in candidate_pool:
        if q not in result and len(result) < 3:
            result.append(q)

    if len(result) < 2:
        for q in default_templates:
            if q not in result and len(result) < 3:
                result.append(q)

    logger.info(f"追问兜底: 模型生成了 {len(existing)} 条，模板补充至 {len(result)} 条")
    return result


class EngineHost:
    """推理宿主：进程内模型托管 + 执行段宿主 + generation 取消注册表。"""

    def __init__(self):
        # 延迟 import：保持模块顶层轻量（config 69ms 可接受，model_module 不在此触发）
        from model_host import ModelHost

        self._host: ModelHost = ModelHost()
        self._layers: List[str] = []  # 已加载层段（1.2 与 model 侧真实状态对齐）
        self._gen_lock = threading.RLock()
        self._generations: Dict[str, threading.Event] = {}
        # 1.2c 宿主适配状态（api_server 全局 → 实例属性）
        self._conversation_stats: Dict[str, int] = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }
        # 1.4 注入：scheduler-svc 侧任务完成回调（当前 no-op，进程内基线）
        self._on_task_complete = None
        # 1.4 注入：scheduler-svc 下发的节点身份/分布式开关（metrics 用）
        self._serving_node_id = ""
        self._distributed_enabled = False
        # 1.2c 会话管理状态（api_server 全局 → 实例属性）
        self._active_session_id: Optional[str] = None
        self._session_histories: Dict[str, list] = {}
        self._kv_cache: Any = None  # PagedKVCache 单例（惰性，_init_kv_cache）
        self._device_profile: Optional[dict] = None  # 惰性探测（_ensure_device_profile）
        self._device_profile_started = False
        # 1.4 注入：scheduler-svc 客户端接口（None = 单机基线：分布式/流水线禁用）
        self._scheduler: Any = None
        self._run_mode: str = "standalone"  # 对齐 api_server.RUN_MODE
        self._on_task_error = None  # 1.4 注入：任务失败回调
        self.role: str = "master"  # master / client（1.3 角色感知，1.5 peer 使用）
        # 1.2d task_graph 状态（api_server 全局 → 实例属性，惰性创建）
        self._task_graph_coordinator: Any = None
        self._task_graph_execution_slot = threading.BoundedSemaphore(1)

    # ------------------------------------------------------------------
    # 模型生命周期（委托 ModelHost / ModelManager）
    # ------------------------------------------------------------------
    def select_engine(self, profile=None) -> str:
        return self._host.select_engine(profile)

    def load_model(
        self,
        engine: Optional[str] = None,
        quant_type: Optional[str] = None,
        use_compile: bool = False,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self._host.load_model(
            engine=engine,
            quant_type=quant_type,
            use_compile=use_compile,
            model_id=model_id,
        )
        return result if isinstance(result, dict) else {"success": True, "data": result}

    def unload_model(self) -> Dict[str, Any]:
        self._host.unload_model()
        self._layers.clear()
        return {"success": True, "message": "模型已卸载"}

    def switch_model(self, model_id: str, engine: Optional[str] = None) -> Dict[str, Any]:
        result = self._host.switch_model(model_id=model_id, engine=engine)
        return result if isinstance(result, dict) else {"success": True, "data": result}

    def current_model(self) -> Dict[str, Any]:
        """当前模型信息（/v1/models/current）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        mgr = self._host
        if not loaded:
            return {"loaded": False}
        try:
            model_id = mgr.active_model_id() if callable(getattr(mgr, "active_model_id", None)) else None
        except Exception:
            model_id = None
        return {
            "loaded": True,
            "engine": getattr(mgr, "_engine_type", None),
            "model_id": model_id,
            "quant_type": getattr(self._host, "current_quant", None),
        }

    # ------------------------------------------------------------------
    # 层段接口（client 角色；master 角色本地层段同用）
    # ------------------------------------------------------------------
    def load_layer_range(
        self, layer_range: str, embed: bool = False, lm_head: bool = False
    ) -> Dict[str, Any]:
        """加载层段。layer_range 形如 "0-12"（[start, end)，对齐
        ModelManager.load_layer_range(start_layer, end_layer,
        has_embedding, has_lm_head) 语义。"""
        try:
            start_str, end_str = layer_range.split("-", 1)
            start_layer, end_layer = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"非法 layer_range: {layer_range!r}（期望如 '0-12'）")
        result = self._host.load_layer_range(
            start_layer=start_layer,
            end_layer=end_layer,
            has_embedding=embed,
            has_lm_head=lm_head,
        )
        if layer_range not in self._layers:
            self._layers.append(layer_range)
        return result if isinstance(result, dict) else {"success": True, "layer_range": layer_range}

    def unload_layer_range(self, layer_range: str) -> Dict[str, Any]:
        if layer_range in self._layers:
            self._layers.remove(layer_range)
        return {"success": True, "unloaded": layer_range}

    def forward_layers(
        self,
        layer_range: str,
        hidden,
        past_key_values=None,
        **kwargs,
    ):
        return self._host.forward_layers(
            layer_range=layer_range,
            hidden=hidden,
            past_key_values=past_key_values,
            **kwargs,
        )

    def embedding(self, input_ids):
        """Embedding 段（1.2 从 scheduler 流水线段复制真实实现）。"""
        raise NotImplementedError("embedding 段在 1.2 随流水线执行段复制接入")

    def lm_head(self, hidden):
        """LM Head 段（1.2 从 scheduler._run_master_lm_head 复制真实实现）。"""
        raise NotImplementedError("lm_head 段在 1.2 随流水线执行段复制接入")

    def ready(self) -> Dict[str, Any]:
        """/v1/ready：模型或层段就绪即 ready（冷启动方案 §5.4 语义）。"""
        loaded = bool(getattr(self._host, "model_loaded", False))
        return {
            "ready": loaded or bool(self._layers),
            "model_loaded": loaded,
            "layers": list(self._layers),
        }

    def status(self) -> Dict[str, Any]:
        """/v1/status：引擎、当前模型、显存、层段。"""
        mgr = self._host
        loaded = bool(getattr(self._host, "model_loaded", False))
        model = getattr(mgr, "model", None)
        device = str(getattr(model, "device", "")) if model is not None else None
        try:
            model_id = mgr.active_model_id() if callable(getattr(mgr, "active_model_id", None)) else None
        except Exception:
            model_id = None
        return {
            "engine": getattr(mgr, "_engine_type", None),
            "model_id": model_id,
            "device": device,
            "model_loaded": loaded,
            "quant_type": getattr(self._host, "current_quant", None),
            "layers": list(self._layers),
        }

    # ------------------------------------------------------------------
    # 对话（1.1 薄实现：本地模型 chat/chat_stream；
    # 1.2 替换为 _execute_chat_full / fast 模式副本，含历史/追问/持久化）
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 完整聊天流程（1.2c 复制自 api_server._execute_chat_full
    # api_server.py:3323-3872；宿主适配：kv_cache/conversation_stats/
    # 会话状态 → 实例属性、model_manager/model_host → self._host、
    # scheduler → self._scheduler 注入点（None=单机基线）、
    # RUN_MODE → self._run_mode）
    # ------------------------------------------------------------------
    def chat_full(
        self, req: ChatRequest, cancel_event: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
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
        import time as _time
        import torch as _torch
        from fastapi import HTTPException

        # ---- task_graph 分支（1.2d：对齐 api_server._execute_requested_chat）----
        if req.execution_mode == "task_graph":
            return self.execute_task_graph_chat(req, cancel_event)

        _raise_if_generation_cancelled(cancel_event, req.generation_id)

        # ---- 多会话支持 ----
        target_session_id = req.session_id or self._active_session_id
        if target_session_id and target_session_id != self._active_session_id:
            self._switch_session(target_session_id)

        # ---- 首条消息自动生成标题 ----
        history = self._get_active_history()
        if target_session_id and len(history) == 0:
            self._auto_title_session(target_session_id, req.message)

        # ---- 路线 B：外部推理服务整请求路由（数据作用域门控，默认不出集群）----
        external_fallback_reason = ""
        _ext_decision = self._external_route_decision(req)
        if _ext_decision.use_external:
            try:
                return self.execute_external_chat(
                    req, history, target_session_id, cancel_event,
                )
            except ChatGenerationCancelled:
                raise
            except Exception as exc:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if not self._host.model_loaded or not getattr(self._host, "is_loaded", False):
                    if req.prefer_external:
                        raise HTTPException(
                            502,
                            f"外部推理服务调用失败，且本地无可用推理引擎：{exc}",
                        ) from exc
                    try:
                        self._ensure_chat_model_or_forwarding()
                    except HTTPException:
                        raise
                    except Exception as load_exc:
                        raise HTTPException(
                            502,
                            f"外部推理服务调用失败（{exc}），"
                            f"且本地模型加载失败：{load_exc}",
                        ) from exc
                external_fallback_reason = f"external_api_failed: {exc}"
                logger.warning(f"外部推理服务调用失败: {exc}，回退到本地推理路径")

        # ---- 分布式推理路由：从节点转发给主节点 ----
        sched = self._scheduler
        distributed_enabled = bool(
            sched is not None and sched.get_distributed_inference_enabled()
        )
        if (distributed_enabled
                and self._run_mode == "distributed"
                and sched._effective_role() == "client"):
            try:
                result = sched.forward_inference_to_master(
                    message=req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    show_thinking=req.show_thinking,
                    session_id=req.session_id,
                    messages=list(history) + [{"role": "user", "content": req.message}],
                    request_id=_request_id_ctx.get("-"),
                    _cancel_event=cancel_event,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                if result.get("status") == "ok":
                    history.append({"role": "user", "content": req.message})
                    response_text = result.get("content", "")
                    history.append({"role": "assistant", "content": response_text})
                    forward_metrics = _augment_chat_metrics(
                        result.get("metrics", {}),
                        req,
                        serving_node_id=self._serving_node_id,
                        distributed_enabled=distributed_enabled,
                        engine="distributed_forward",
                        execution_mode="forwarded_to_master",
                        route="pc_client_forward_to_master",
                    )
                    if external_fallback_reason and not forward_metrics.get(
                        "fallback_reason",
                    ):
                        forward_metrics["fallback"] = True
                        forward_metrics["fallback_reason"] = external_fallback_reason

                    db_session_id = target_session_id or "default"
                    if getattr(self._host, "_db_available", False):
                        try:
                            import db as _db_mod
                            if _db_mod.get_save_history():
                                _db_mod.save_message(db_session_id, "user", req.message)
                                _db_mod.save_message(db_session_id, "assistant", response_text,
                                                    forward_metrics)
                                _db_mod.increment_session_message_count(db_session_id)
                        except Exception:
                            pass
                    else:
                        try:
                            import local_store as _local_store
                            _local_store.save_local_message(db_session_id, "user", req.message)
                            _local_store.save_local_message(db_session_id, "assistant", response_text,
                                                            forward_metrics)
                            _local_store.increment_local_session_message_count(db_session_id)
                        except Exception:
                            pass

                    self._conversation_stats["rounds"] += 1
                    self._record_task_complete()

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
            except ChatGenerationCancelled:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                logger.warning(f"分布式推理转发异常: {e}，回退到本地推理")

            if self._pipeline_worker_is_reserved():
                raise HTTPException(
                    503,
                    "本设备正作为 PyTorch 分层从节点，"
                    "当前无法转发到主节点，已拒绝覆盖分层模型。",
                )
            if not self._host.model_loaded or not getattr(self._host, "is_loaded", False):
                self._auto_load_default_model()

        if self._pipeline_worker_is_reserved():
            raise HTTPException(
                503,
                "本设备正作为 PyTorch 分层从节点，"
                "请先断开主节点或明确切换本地模型。",
            )

        # ---- 分布式流水线推理路径（主节点 + PyTorch 引擎 + 从节点可用）----
        if (distributed_enabled
                and self._run_mode == "distributed"
                and sched._effective_role() == "master"
                and getattr(self._host, "_engine_type", None) == "pytorch"):
            try:
                pipeline_result = sched.run_pipeline_safe(
                    req.message,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    session_id=req.session_id,
                    messages=list(history) + [{"role": "user", "content": req.message}],
                    show_thinking=req.show_thinking,
                    _cancel_event=cancel_event,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
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
                            serving_node_id=self._serving_node_id,
                            distributed_enabled=distributed_enabled,
                            engine="distributed_pipeline",
                            execution_mode="distributed_pipeline",
                            route="master_pipeline",
                        )
                        if external_fallback_reason and not pipeline_metrics.get(
                            "fallback_reason",
                        ):
                            pipeline_metrics["fallback"] = True
                            pipeline_metrics["fallback_reason"] = (
                                external_fallback_reason
                            )
                        if getattr(self._host, "_db_available", False):
                            try:
                                import db as _db_mod
                                if _db_mod.get_save_history():
                                    _db_mod.save_message(db_session_id, "user", req.message)
                                    _db_mod.save_message(db_session_id, "assistant", response_text,
                                                        pipeline_metrics)
                                    _db_mod.increment_session_message_count(db_session_id)
                            except Exception:
                                pass
                        else:
                            try:
                                import local_store as _local_store
                                _local_store.save_local_message(db_session_id, "user", req.message)
                                _local_store.save_local_message(db_session_id, "assistant",
                                                                response_text,
                                                                pipeline_metrics)
                                _local_store.increment_local_session_message_count(db_session_id)
                            except Exception:
                                pass

                        self._conversation_stats["rounds"] += 1
                        if not pipeline_metrics.get("distributed_used"):
                            self._record_task_complete()

                        if getattr(self._host, "_engine_type", None) in ("llama_cpp", "island"):
                            followups = self.generate_followups_llama(history)
                        elif getattr(self._host, "_engine_type", None) == "pytorch":
                            followups = _fallback_followups(history, [])
                        else:
                            followups = _fallback_followups(history, [])

                        return {
                            "content": response_text,
                            "thinking_content": pipeline_result.get("thinking"),
                            "metrics": pipeline_metrics,
                            "followups": followups,
                        }
            except ChatGenerationCancelled:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                logger.warning(f"流水线推理异常: {e}，回退到本地推理")

        model_manager = self._host
        # ---- llama.cpp / 孤岛引擎路径（整请求推理，不参与层拆分）----
        if getattr(model_manager, "_engine_type", None) in ("llama_cpp", "island"):
            try:
                engine_name = model_manager._engine_type
                request_history = [
                    *history,
                    {"role": "user", "content": req.message},
                ]
                result = model_manager.chat(
                    messages=request_history,
                    max_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    _cancel_event=cancel_event,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                response_text = result.get("content", "")
                if not req.show_thinking:
                    response_text = _strip_native_thinking_tags(response_text)
                completed_history = [
                    *request_history,
                    {"role": "assistant", "content": response_text},
                ]
                tokens_per_sec = result.get("tokens_per_second", 0)
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                local_route = f"{_chat_origin(req)}_to_master_local_{engine_name}"
                fallback_reason = ""
                if external_fallback_reason:
                    fallback_reason = external_fallback_reason
                elif distributed_enabled and self._run_mode == "distributed":
                    if engine_name == "island":
                        fallback_reason = "island engine delegates whole-request inference to the TP island"
                    else:
                        fallback_reason = "llama.cpp engine does not support layer-split pipeline"
                metrics = _augment_chat_metrics(
                    {
                        "engine": engine_name,
                        "execution_mode": f"local_{engine_name}",
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
                    serving_node_id=self._serving_node_id,
                    distributed_enabled=distributed_enabled,
                )

                db_session_id = target_session_id or "default"
                followups = self.generate_followups_llama(
                    completed_history, cancel_event,
                )
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                history.extend([
                    {"role": "user", "content": req.message},
                    {"role": "assistant", "content": response_text},
                ])

                if getattr(self._host, "_db_available", False):
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
                else:
                    try:
                        import local_store as _local_store
                        _local_store.save_local_message(db_session_id, "user", req.message)
                        save_metrics = dict(metrics)
                        save_metrics["followups"] = followups
                        _local_store.save_local_message(db_session_id, "assistant", response_text,
                                                        save_metrics)
                        _local_store.increment_local_session_message_count(db_session_id)
                    except Exception:
                        pass

                self._conversation_stats["total_generated_tokens"] += completion_tokens
                self._conversation_stats["rounds"] += 1
                self._record_task_complete()

                return {
                    "content": response_text,
                    "thinking_content": None,
                    "metrics": metrics,
                    "followups": followups,
                }
            except ChatGenerationCancelled:
                raise
            except Exception as e:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                self._record_task_error()
                _engine_label = (
                    "孤岛引擎"
                    if getattr(model_manager, "_engine_type", "") == "island"
                    else "llama.cpp"
                )
                logger.error(f"{_engine_label} 推理失败: {e}", exc_info=True)
                raise HTTPException(500, f"推理失败: {str(e)}")

        # ---- PyTorch 引擎路径（CUDA/独显）----
        try:
            model_manager.ensure_full_model()
            tier_max = self._host.generation_config.get("tier_max_new_tokens", self._host.generation_config["max_new_tokens"])
            thinking_budget = 384 if req.show_thinking else 0
            effective_max = min(req.max_new_tokens + thinking_budget,
                                tier_max + thinking_budget,
                                4096)
            self._host.generation_config["max_new_tokens"] = effective_max
            self._host.generation_config["temperature"] = req.temperature
            self._host.generation_config["top_p"] = req.top_p

            request_history = [
                *history,
                {"role": "user", "content": req.message},
            ]

            tokenizer = model_manager.tokenizer
            thinking_prompt = THINKING_SYSTEM_PROMPT if req.show_thinking else None
            thinking_prefill = "【思考】\n" if req.show_thinking else None
            prompt = _build_model_chat_prompt(
                tokenizer,
                request_history,
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
            stop_criteria_kwargs = (
                {"cancel_event": cancel_event} if cancel_event is not None else {}
            )
            stop_criteria = model_manager._build_stop_criteria(
                stop_sequences, prompt_len, **stop_criteria_kwargs,
            )
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            t0 = _time.time()
            with _torch.no_grad():
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
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            elapsed = _time.time() - t0

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

            completed_history = [
                *request_history,
                {"role": "assistant", "content": response_text},
            ]

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
                    "gpu_memory_mb": round(_torch.cuda.memory_allocated() / (1024**2), 1)
                    if _torch.cuda.is_available()
                    else 0,
                },
                req,
                serving_node_id=self._serving_node_id,
                distributed_enabled=distributed_enabled,
                fallback=bool(external_fallback_reason),
                fallback_reason=external_fallback_reason,
            )

            db_session_id = target_session_id or "default"

            followups = self._generate_followups(
                completed_history,
                tokenizer,
                model_manager.model,
                model_manager.get_device(),
                cancel_event,
            )
            _raise_if_generation_cancelled(cancel_event, req.generation_id)
            history.extend([
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": response_text},
            ])

            if getattr(self._host, "_db_available", False):
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
            else:
                try:
                    import local_store as _local_store
                    save_metrics = dict(metrics)
                    save_metrics["followups"] = followups
                    _local_store.save_local_message(db_session_id, "user", req.message)
                    _local_store.save_local_message(db_session_id, "assistant", response_text, save_metrics)
                    _local_store.increment_local_session_message_count(db_session_id)
                except Exception:
                    pass

            self._conversation_stats["total_prompt_tokens"] += prompt_len
            self._conversation_stats["total_generated_tokens"] += new_tokens
            self._conversation_stats["total_time_seconds"] += elapsed
            self._conversation_stats["rounds"] += 1
            self._record_task_complete()

            logger.info(
                f"推理完成: {new_tokens} tokens / {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s"
            )

            return {
                "content": response_text,
                "thinking_content": thinking_content,
                "metrics": metrics,
                "followups": followups,
            }

        except ChatGenerationCancelled:
            raise
        except _torch.cuda.OutOfMemoryError:
            self._record_task_error()
            if self._kv_cache:
                self._kv_cache.clear()
            self._get_active_history().clear()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            raise HTTPException(507, "GPU 显存不足（OOM），已自动清空对话历史。请缩短消息后重试。")

        except Exception as e:
            self._record_task_error()
            logger.error(f"推理异常: {e}", exc_info=True)
            raise HTTPException(500, f"推理失败: {str(e)}")

    def _record_task_complete(self) -> None:
        if self._on_task_complete is not None:
            try:
                self._on_task_complete(success=True)
            except Exception:
                pass

    def _record_task_error(self) -> None:
        if self._on_task_error is not None:
            try:
                self._on_task_error()
            except Exception:
                pass

    def chat_stream_events(self, req: ChatRequest, cancel_event: Optional[threading.Event]):
        """SSE 事件序列（1.1 薄实现；1.2 替换为 fast 模式副本）。

        Yields:
            dict 事件：{"token": ...} 或 {"done": True, "response": ..., "metrics": ...}
        """
        messages = [{"role": "user", "content": req.message}]
        chunks = self._host.chat_stream(
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        for chunk in chunks:
            if cancel_event is not None and cancel_event.is_set():
                break
            yield {"token": chunk}
        yield {"done": True, "response": "", "followups": [], "metrics": {}, "request_id": "-"}

    def speculative_run(self, req) -> Dict[str, Any]:
        """投机解码实验端点（1.2b 复制自 api_server._run_speculative_experiment
        api_server.py:2506-2519；SPEC_ENABLED 门控与异常映射在 routes 层）。"""
        from speculative import run_speculative_chat

        return run_speculative_chat(
            req.message,
            allow_external=bool(req.allow_external),
            max_new_tokens=int(req.max_new_tokens),
            gamma=(int(req.gamma) if req.gamma > 0 else None),
            max_rounds=(int(req.max_rounds) if req.max_rounds > 0 else None),
            temperature=(float(req.temperature) if req.temperature >= 0 else None),
            seed=int(req.seed),
            draft_hint=req.draft_hint or "",
        )

    # ------------------------------------------------------------------
    # 追问生成（1.2b 复制自 api_server._generate_followups_llama
    # api_server.py:1356-1435；宿主适配：model_manager → self._host）
    # ------------------------------------------------------------------
    def generate_followups_llama(
        self,
        history: list,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """
        使用 llama.cpp 引擎生成追问建议。

        通过 model_host.chat() 调用（llama.cpp 路径），失败时回退到
        关键词模板兜底（_fallback_followups）。
        """
        if not history or len(history) < 2:
            return []
        if cancel_event is not None and cancel_event.is_set():
            return []

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
            result = self._host.chat(
                messages=followup_messages,
                max_tokens=128,
                temperature=0.8,
                top_p=0.9,
                _cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                return []
            text = result.get("content", "").strip()

            for line in text.split("\n"):
                line = line.strip()
                line = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
                if line.upper().startswith("Q:") or line.upper().startswith("Q："):
                    line = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                if line and len(line) >= 5 and len(line) <= 80 and _is_question(line):
                    questions.append(line)

            hallucination_patterns = [
                "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
                "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
                "百川", "智谱", "ChatGLM", "混元",
            ]
            questions = [q for q in questions if not any(p in q for p in hallucination_patterns)]

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

        if len(questions) < 2:
            fallback = _fallback_followups(history, questions)
            questions = fallback

        return questions[:3]

    # ------------------------------------------------------------------
    # task-worker 数据面执行段（1.2a 复制自 api_server._execute_task_worker_stage
    # api_server.py:2734-2872，源文件保持不动；宿主适配：model_manager → self._host）
    # ------------------------------------------------------------------
    def execute_task_worker_stage(
        self,
        stage_request,
        provider_cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """Execute the shared local/remote Stage contract on a full model."""
        from task_graph import DEPENDENCY_FAILURES_KEY, TaskGraphError
        from task_provider import ProviderError, ProviderExecutionError

        root_input = stage_request.root_input
        options = root_input.get("task_options", {})
        if not isinstance(options, dict):
            raise TaskGraphError("任务 Stage 缺少有效执行参数")
        try:
            candidate_budget = max(
                1, min(int(options.get("candidate_max_tokens", 512)), 512)
            )
            final_budget = max(
                1, min(int(options.get("final_max_tokens", 1024)), 1024)
            )
            temperature = max(
                0.0, min(float(options.get("temperature", 0.7)), 2.0)
            )
            top_p = max(0.0, min(float(options.get("top_p", 0.9)), 1.0))
        except (TypeError, ValueError) as exc:
            raise TaskGraphError("任务 Stage 执行参数无效") from exc
        show_thinking = bool(options.get("show_thinking", False))
        model_manager = self._host

        def run_model(
            messages: list[dict],
            max_tokens: int,
            *,
            retry_empty_on_same_provider: bool = False,
        ) -> dict:
            if provider_cancel_event.is_set():
                return {
                    "content": "",
                    "usage": {},
                    "tokens_per_second": 0,
                    "model": model_manager.active_model_id,
                }
            result = model_manager.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                _cancel_event=provider_cancel_event,
            )
            raw_content = str(result.get("content", "") or "").strip()
            thinking_content = None
            if show_thinking:
                content, thinking_content = _format_model_response(
                    raw_content, show_thinking=True,
                )
            else:
                content = _strip_native_thinking_tags(raw_content)
            if not content and not provider_cancel_event.is_set():
                if retry_empty_on_same_provider:
                    raise ProviderExecutionError(
                        "complete model returned an empty aggregate result",
                        code="empty_provider_output",
                        provider_id=stage_request.provider_id,
                        same_provider_retryable=True,
                    )
                raise TaskGraphError("完整模型返回空 Stage 结果")
            return {
                "content": content,
                "thinking_content": thinking_content,
                "usage": dict(result.get("usage", {}) or {}),
                "tokens_per_second": result.get("tokens_per_second", 0),
                "model": result.get("model", model_manager.active_model_id),
                "usage_estimated": bool(result.get("usage_estimated", False)),
            }

        if stage_request.stage_type == "full_inference":
            candidate_instructions = {
                "candidate_a": (
                    "独立分析用户问题，给出准确、可验证且简洁的候选答案。"
                    "不要提及其他候选或任务链。"
                ),
                "candidate_b": (
                    "从不同角度独立解决用户问题，重点检查遗漏、反例和不确定性。"
                    "输出可直接供后续汇总的候选答案。"
                ),
            }
            instruction = candidate_instructions.get(stage_request.stage_id)
            messages = root_input.get("messages")
            if instruction is None or not isinstance(messages, list):
                raise TaskGraphError("完整推理 Stage 输入无效")
            if show_thinking:
                instruction = f"{instruction}\n\n{THINKING_SYSTEM_PROMPT}"
            return run_model(
                [{"role": "system", "content": instruction}, *messages],
                candidate_budget,
            )
        if stage_request.stage_type == "aggregate":
            message = str(root_input.get("message", "") or "")
            if not message:
                raise TaskGraphError("聚合 Stage 缺少原始问题")
            candidate_payload = {
                stage_id: value.get("content", "")
                for stage_id, value in stage_request.dependencies.items()
                if stage_id != DEPENDENCY_FAILURES_KEY
                and isinstance(value, dict)
                and str(value.get("content", "") or "").strip()
            }
            if not candidate_payload:
                raise TaskGraphError("聚合 Stage 没有可用候选")
            failure_payload = stage_request.dependencies.get(
                DEPENDENCY_FAILURES_KEY, {},
            )
            aggregation_prompt = (
                "请根据原始问题和可用的独立候选，输出一个最终答案。"
                "纠正冲突和明显错误；没有证据时明确不确定性。"
                "只输出最终答案，不描述内部任务链。\n\n"
                f"原始问题：{message}\n\n候选：\n"
                + json.dumps(candidate_payload, ensure_ascii=False)
                + (
                    "\n\n未完成候选摘要：\n"
                    + json.dumps(failure_payload, ensure_ascii=False)
                    if isinstance(failure_payload, dict) and failure_payload
                    else ""
                )
            )
            try:
                return run_model(
                    ([{"role": "system", "content": THINKING_SYSTEM_PROMPT}]
                     if show_thinking else [])
                    + [{"role": "user", "content": aggregation_prompt}],
                    final_budget,
                    retry_empty_on_same_provider=True,
                )
            except ProviderError:
                raise
            except (TimeoutError, ConnectionError) as exc:
                raise ProviderExecutionError(
                    "transient aggregate model execution failed",
                    code="provider_execution_failed",
                    provider_id=stage_request.provider_id,
                    same_provider_retryable=True,
                ) from exc
        raise TaskGraphError(f"不支持的 Stage 类型: {stage_request.stage_type}")

    # ------------------------------------------------------------------
    # 1.2d task_graph 执行段（复制自 api_server.py:2571-3321，源文件不动）
    # 入口：execute_task_graph_chat（slot/journal/角色门控）→
    #       execute_task_graph_chat_with_slot（主体 448 行）
    # ------------------------------------------------------------------
    def _ensure_task_graph_coordinator(self) -> Any:
        """1.2d 惰性创建任务图协调器（复制自 api_server._create_task_graph_coordinator
        api_server.py:354-395；全局实例 → 实例属性，首次访问时创建并缓存）。"""
        if self._task_graph_coordinator is not None:
            return self._task_graph_coordinator
        import config as _cfg
        from task_graph import TaskGraphCoordinator, TaskGraphUnavailable
        from task_journal import SQLiteTaskJournal, TaskJournalError

        if not _cfg.TASK_GRAPH_ENABLED:
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
            )
            self._task_graph_coordinator = coordinator
            return coordinator
        journal = None
        try:
            journal = SQLiteTaskJournal(_cfg.TASK_GRAPH_JOURNAL_PATH)
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                journal=journal,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
            )
            recovery = coordinator.recover_persisted_workflows()
            cleanup = coordinator.cleanup_journal(
                max_age_days=_cfg.TASK_GRAPH_RETENTION_DAYS,
                max_records=_cfg.TASK_GRAPH_RETENTION_MAX_RECORDS,
            )
            if recovery.get("recovered_workflows", 0):
                logger.warning("任务图启动恢复完成: %s", recovery)
            if cleanup.get("deleted_workflows", 0):
                logger.info("任务图 journal 保留清理完成: %s", cleanup)
            self._task_graph_coordinator = coordinator
            return coordinator
        except (TaskJournalError, TaskGraphUnavailable) as exc:
            if journal is not None:
                try:
                    journal.close()
                except Exception:
                    pass
            coordinator = TaskGraphCoordinator(
                max_records=_cfg.TASK_GRAPH_MAX_RECORDS,
                max_parallel_stages=_cfg.TASK_GRAPH_MAX_PARALLEL_STAGES,
                availability_error=f"task journal unavailable: {exc}",
            )
            logger.error(
                "任务图 journal 初始化失败，任务图已禁用: %s",
                exc,
                exc_info=True,
            )
            self._task_graph_coordinator = coordinator
            return coordinator


    def _dispatch_local_task_provider(
        self, request, cancel_event: threading.Event,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._dispatch_local_task_provider（api_server.py:2603-2612）。"""
        from task_graph import TaskGraphError
        from task_provider import ProviderExecutor
        from typing import cast
        executor = request.runtime_context.get("local_provider_executor")
        if not callable(executor):
            raise TaskGraphError("本地任务 Provider 缺少请求执行上下文")
        result = cast(ProviderExecutor, executor)(request, cancel_event)
        if not isinstance(result, dict):
            raise TaskGraphError("本地任务 Provider 返回值必须是 dict")
        return result



    def _ensure_local_task_provider(
        self,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._ensure_local_task_provider（api_server.py:2616-2629）；
        宿主适配：self._ensure_task_graph_coordinator() → 实例、scheduler → self._scheduler。"""
        from task_provider import LocalFullModelProvider, ProviderError
        if self._ensure_task_graph_coordinator().has_provider("local_full_model"):
            return
        try:
            self._ensure_task_graph_coordinator().register_provider(LocalFullModelProvider(
                self._dispatch_local_task_provider,
                provider_id="local_full_model",
                node_id=(self._scheduler.get_effective_node_id() if self._scheduler is not None else ""),
                max_concurrency=1,
            ))
        except ProviderError:
            if not self._ensure_task_graph_coordinator().has_provider("local_full_model"):
                raise



    def _active_task_graph_model_identity(
        self,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._active_task_graph_model_identity（api_server.py:2631-2679）；
        宿主适配：model_host/model_manager → self._host。"""
        import hashlib
        from task_provider import ModelIdentity
        if not self._host.model_loaded or not self._host.is_loaded:
            return None
        engine = str(getattr(self._host, "_engine_type", "") or "")
        model_path = str(getattr(self._host, "_model_path", "") or "")
        model_id = str(getattr(self._host, "active_model_id", "") or "")
        if engine not in {"pytorch", "llama_cpp", "island"} or not model_id or not model_path:
            return None
        if engine == "island":
            # 孤岛模型无本地 artifact：以"端点指纹 + 后端模型名"替代文件摘要，
            # 统计中如实标注为外部端点（不伪装成本地文件，见调研方案 §2.2）。
            island_engine = getattr(self._host, "_island_engine", None)
            backend_model = str(getattr(island_engine, "model_name", "") or "")
            masked_url = str(getattr(island_engine, "masked_base_url", "") or model_path)
            if not backend_model:
                return None
            digest = hashlib.sha256(
                f"{masked_url}::{backend_model}".encode("utf-8")
            ).hexdigest()
            return ModelIdentity(
                model_id=model_id,
                engine="island",
                format="openai_api",
                revision=f"island-{digest[:12]}",
                sha256=digest,
            )
        try:
            from model_sync import compute_file_sha256, compute_model_sha256

            if engine == "pytorch":
                digest = compute_model_sha256(model_path)
                model_format = "safetensors"
            else:
                digest = compute_file_sha256(model_path)
                model_format = "gguf"
        except Exception:
            logger.warning("无法计算任务链当前模型摘要", exc_info=True)
            return None
        if len(digest) != 64:
            return None
        return ModelIdentity(
            model_id=model_id,
            engine=engine,
            format=model_format,
            revision=f"local-{digest[:12]}",
            sha256=digest,
        )



    def _sync_remote_task_worker_providers(
        self,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._sync_remote_task_worker_providers（api_server.py:2680-2692）；
        宿主适配：scheduler → self._scheduler（None 时无远端 provider）。"""
        from task_provider import ProviderError
        """Register stable remote Provider objects without changing request policy."""
        registered = []
        if self._scheduler is None:
            return []
        for provider in self._scheduler.remote_task_worker_providers():
            if not self._ensure_task_graph_coordinator().has_provider(provider.provider_id):
                try:
                    self._ensure_task_graph_coordinator().register_provider(provider)
                except ProviderError:
                    if not self._ensure_task_graph_coordinator().has_provider(provider.provider_id):
                        raise
            registered.append(provider.provider_id)
        return sorted(registered)



    def _eligible_remote_task_worker_provider_ids(
        self, model_identity, stage_type: str, *, limit: int = 4,
    ) -> Dict[str, Any]:
        """1.2d 复制自 api_server._eligible_remote_task_worker_provider_ids
        （api_server.py:2694-2718）；宿主适配：_cfg.TASK_WORKER_EXPERIMENTAL_ENABLED
        → config、scheduler → self._scheduler。"""
        import config as _cfg
        """Return healthy exact-model Workers in deterministic least-loaded order."""
        if not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
            return []
        if self._scheduler is None:
            return []
        providers = {
            provider.provider_id: provider
            for provider in self._scheduler.remote_task_worker_providers()
        }
        self._sync_remote_task_worker_providers()
        statuses = {
            str(item.get("provider_id", "")): item
            for item in self._ensure_task_graph_coordinator().provider_status()
        }
        eligible = []
        for provider_id, provider in providers.items():
            status = statuses.get(provider_id, {})
            if (
                status.get("provider_kind") != "remote_full_worker"
                or not status.get("healthy")
                or not status.get("available")
                or stage_type not in status.get("supported_stage_types", [])
                or not provider.supports_model_identity(model_identity, stage_type)
            ):
                continue
            max_concurrency = max(1, int(status.get("max_concurrency", 1) or 1))
            active = max(0, int(status.get("active_reservations", 0) or 0))
            eligible.append((
                active / max_concurrency,
                active,
                provider_id,
            ))
        eligible.sort()
        return [item[2] for item in eligible[:max(0, int(limit))]]



    def execute_task_graph_chat(
        self, req: ChatRequest, cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Run the fixed local task graph without claiming multi-device execution.
        1.2d 复制自 api_server._execute_task_graph_chat（api_server.py:2571-2614）；
        宿主适配：TASK_GRAPH_ENABLED → config、task_graph_coordinator/
        self._task_graph_execution_slot/model_host → 实例状态、scheduler →
        self._scheduler（None = 单机基线视为 master）。"""
        import config as _cfg
        from fastapi import HTTPException
        if not _cfg.TASK_GRAPH_ENABLED:
            raise HTTPException(
                409,
                "任务链实验未启用。请设置 QLH_TASK_GRAPH_ENABLED=true 后重启。",
            )
        journal = self._ensure_task_graph_coordinator().journal_status()
        if not journal.get("available", False):
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 不可用，已拒绝不可恢复执行。",
                    "reason": journal.get("error", "journal health check failed"),
                },
            )
        if self._scheduler is not None and self._scheduler._effective_role() != "master":
            raise HTTPException(409, "任务链协调器当前只允许在主节点运行。")

        if not self._task_graph_execution_slot.acquire(blocking=False):
            raise HTTPException(429, "已有任务链正在执行，请稍后重试。")
        try:
            with self._host.full_chat_execution_lock:
                return self.execute_task_graph_chat_with_slot(req, cancel_event)
        finally:
            self._task_graph_execution_slot.release()



    def execute_task_graph_chat_with_slot(
        self, req: ChatRequest, cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Execute one workflow while the process-wide task-graph slot is held.
        1.2d 复制自 api_server._execute_task_graph_chat_with_slot
        （api_server.py:2874-3321）；宿主适配：scheduler → self._scheduler、
        task_graph_coordinator → self._ensure_task_graph_coordinator()、
        model_host/model_manager → self._host、self._gen_lock →
        self._gen_lock、conversation_stats/self._active_session_id → 实例属性、
        record_task_complete → self._record_task_complete()。"""
        import config as _cfg
        import local_store as _local_store
        from dataclasses import replace
        from fastapi import HTTPException
        from task_graph import (
            StageSpec,
            TaskGraphError,
            TaskGraphUnavailable,
            WorkflowCancelled,
            WorkflowExecutionError,
            dual_candidate_template,
        )
        from task_provider import ProviderError

        remote_stage_id = str(req.task_graph_remote_stage or "")
        remote_provider_id = str(req.task_graph_remote_provider_id or "")
        auto_remote = bool(req.task_graph_auto_remote)
        if bool(remote_stage_id) != bool(remote_provider_id):
            raise HTTPException(
                400,
                "N2.1 手动远端执行必须同时指定 Stage 和 Provider ID。",
            )
        if auto_remote and remote_stage_id:
            raise HTTPException(
                400,
                "N2.3 自动 Worker 选择不能与 N2.1 手动远端 Stage 同时启用。",
            )
        if remote_stage_id and not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
            raise HTTPException(
                409,
                "PC Full Worker 实验调度未启用。请设置 "
                "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED=true 后重启。",
            )

        target_session_id = req.session_id or self._active_session_id
        if target_session_id and target_session_id != self._active_session_id:
            self._switch_session(target_session_id)
        history = self._get_active_history()
        if target_session_id and len(history) == 0:
            self._auto_title_session(target_session_id, req.message)

        base_messages = list(history) + [{"role": "user", "content": req.message}]
        root_input = {
            "message": req.message,
            "messages": base_messages,
            "task_options": {
                "candidate_max_tokens": max(1, min(req.max_new_tokens, 512)),
                "final_max_tokens": max(1, min(req.max_new_tokens, 1024)),
                "temperature": req.temperature,
                "top_p": req.top_p,
                "show_thinking": req.show_thinking,
            },
        }

        try:
            self._ensure_local_task_provider()
        except ProviderError as exc:
            raise HTTPException(
                503,
                {
                    "message": "本地任务 Provider 注册失败。",
                    "reason": f"{exc.code}: {exc}",
                },
            ) from exc

        model_identity = None
        stages: Optional[list[StageSpec]] = None
        final_stage_id = ""
        auto_provider_ids: list[str] = []
        auto_fallback_reason = ""
        if remote_stage_id:
            remote_providers = {
                provider.provider_id: provider
                for provider in (self._scheduler.remote_task_worker_providers() if self._scheduler is not None else [])
            }
            self._sync_remote_task_worker_providers()
            remote_provider = remote_providers.get(remote_provider_id)
            if remote_provider is None:
                raise HTTPException(
                    404, "The selected remote PC Full Worker does not exist."
                )
            remote_status = next((
                item for item in self._ensure_task_graph_coordinator().provider_status()
                if item.get("provider_id") == remote_provider_id
                and item.get("provider_kind") == "remote_full_worker"
            ), None)
            if remote_status is None:
                raise HTTPException(404, "指定的远端 PC Full Worker Provider 不存在。")
            if not remote_status.get("healthy") or not remote_status.get("available"):
                raise HTTPException(503, "指定的远端 PC Full Worker 当前不可用或正忙。")
            model_identity = self._active_task_graph_model_identity()
            if model_identity is None:
                raise HTTPException(409, "手动远端 Stage 要求主节点先加载完整模型并生成精确身份。")
            template_stages, final_stage_id = dual_candidate_template()
            selected_stage = next((
                stage for stage in template_stages
                if stage.stage_id == remote_stage_id
            ), None)
            if (
                selected_stage is None
                or not remote_provider.supports_model_identity(
                    model_identity, selected_stage.stage_type,
                )
            ):
                raise HTTPException(
                    409,
                    {
                        "message": (
                            "The selected remote PC Full Worker does not have "
                            "the exact active model required by this Stage."
                        ),
                        "reason_code": "model_identity_mismatch",
                        "provider_id": remote_provider_id,
                        "stage_id": remote_stage_id,
                    },
                )
            stages = [
                replace(
                    stage,
                    provider=remote_provider_id,
                    fallback_providers=(),
                    pure=False,
                    max_same_provider_retries=0,
                )
                if stage.stage_id == remote_stage_id else stage
                for stage in template_stages
            ]
        elif auto_remote:
            if not _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
                auto_fallback_reason = "task_worker_experiment_disabled"
            else:
                model_identity = self._active_task_graph_model_identity()
            if _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED and model_identity is None:
                auto_fallback_reason = "model_identity_unavailable"
            elif _cfg.TASK_WORKER_EXPERIMENTAL_ENABLED:
                auto_provider_ids = self._eligible_remote_task_worker_provider_ids(
                    model_identity,
                    "full_inference",
                )
                if not auto_provider_ids:
                    auto_fallback_reason = "no_eligible_remote_provider"
                else:
                    template_stages, final_stage_id = dual_candidate_template()
                    candidate_index = 0
                    planned_stages = []
                    for stage in template_stages:
                        if stage.stage_type != "full_inference":
                            planned_stages.append(stage)
                            continue
                        if candidate_index >= len(auto_provider_ids):
                            planned_stages.append(replace(
                                stage,
                                provider="local_full_model",
                                fallback_providers=(),
                                pure=True,
                            ))
                            candidate_index += 1
                            continue
                        primary = auto_provider_ids[candidate_index]
                        other_remotes = [
                            provider_id for provider_id in auto_provider_ids
                            if provider_id != primary
                        ][:3]
                        planned_stages.append(replace(
                            stage,
                            provider=primary,
                            fallback_providers=tuple(
                                [*other_remotes, "local_full_model"]
                            ),
                            pure=True,
                        ))
                        candidate_index += 1
                    stages = planned_stages

        request_id = str(_request_id_ctx.get("-") or "-")
        runtime_context = {
            "local_provider_executor": self.execute_task_worker_stage,
            "task_graph_remote_policy": (
                "manual" if remote_stage_id else "auto" if auto_remote else "local"
            ),
        }
        try:
            if stages is None:
                final_output, workflow = self._ensure_task_graph_coordinator().run_template(
                    template=req.task_graph_template,
                    root_input=root_input,
                    request_id=request_id,
                    session_id=target_session_id or "default",
                    model_identity=model_identity,
                    runtime_context=runtime_context,
                    workflow_id=req.workflow_id,
                    cancel_event=cancel_event,
                )
            else:
                final_output, workflow = self._ensure_task_graph_coordinator().run(
                    stages=stages,
                    final_stage_id=final_stage_id,
                    template=req.task_graph_template,
                    root_input=root_input,
                    request_id=request_id,
                    session_id=target_session_id or "default",
                    model_identity=model_identity,
                    runtime_context=runtime_context,
                    workflow_id=req.workflow_id,
                    cancel_event=cancel_event,
                )
        except WorkflowCancelled as exc:
            raise HTTPException(
                409,
                {"message": "任务链已取消", "workflow_id": exc.workflow_id},
            ) from exc
        except WorkflowExecutionError as exc:
            raise HTTPException(
                500,
                {
                    "message": str(exc),
                    "workflow_id": exc.workflow_id,
                    "stage_id": exc.stage_id,
                },
            ) from exc
        except TaskGraphUnavailable as exc:
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 写入失败，执行已停止。",
                    "reason": str(exc),
                },
            ) from exc
        except TaskGraphError as exc:
            raise HTTPException(400, f"任务链请求无效: {exc}") from exc

        response_text = str(final_output.get("content", "") or "").strip()
        thinking_content = final_output.get("thinking_content")
        if not response_text:
            raise HTTPException(500, "任务链最终聚合结果为空。")

        history_start = len(history)
        try:
            with self._gen_lock:
                _raise_if_generation_cancelled(cancel_event, req.generation_id)
                history.extend([
                    {"role": "user", "content": req.message},
                    {"role": "assistant", "content": response_text},
                ])
                if cancel_event is not None and cancel_event.is_set():
                    del history[history_start:]
                    _raise_if_generation_cancelled(cancel_event, req.generation_id)
                try:
                    workflow = self._ensure_task_graph_coordinator().commit_result(
                        workflow["workflow_id"],
                    )
                except WorkflowCancelled as exc:
                    del history[history_start:]
                    raise ChatGenerationCancelled(
                        req.generation_id or "gen_unknown",
                    ) from exc
                except TaskGraphUnavailable:
                    del history[history_start:]
                    raise
        except ChatGenerationCancelled:
            try:
                self._ensure_task_graph_coordinator().discard_result(workflow["workflow_id"])
            except TaskGraphUnavailable as exc:
                raise HTTPException(
                    503,
                    {
                        "message": "任务链取消无法写入 journal。",
                        "reason": str(exc),
                    },
                ) from exc
            raise
        except TaskGraphUnavailable as exc:
            raise HTTPException(
                503,
                {
                    "message": "任务链 journal 写入失败，结果未提交。",
                    "reason": str(exc),
                },
            ) from exc

        attempts = [
            attempt
            for stage in workflow.get("stages", [])
            for attempt in stage.get("attempts", [])
            if attempt.get("state") == "completed"
        ]
        usages = [
            dict(attempt.get("result_metadata", {}).get("usage", {}) or {})
            for attempt in attempts
        ]
        prompt_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages)
        completion_tokens = sum(
            int(usage.get("completion_tokens", 0) or 0) for usage in usages
        )
        providers = sorted({
            str(attempt.get("provider", "") or "")
            for attempt in attempts
            if attempt.get("provider")
        })
        serving_node_id = (self._scheduler.get_effective_node_id() if self._scheduler is not None else "")
        participating_nodes = sorted({
            str(attempt.get("provider_node_id", "") or serving_node_id)
            for attempt in attempts
        }) or [serving_node_id]
        remote_attempts = [
            attempt for attempt in attempts
            if attempt.get("provider_kind") == "remote_full_worker"
        ]
        remote_nodes = sorted({
            str(attempt.get("provider_node_id", "") or "")
            for attempt in remote_attempts
            if attempt.get("provider_node_id")
        })
        remote_used = bool(remote_attempts)
        provider_status_by_id = {
            str(item.get("provider_id", "")): item
            for item in self._ensure_task_graph_coordinator().provider_status()
        }
        planned_remote_nodes = sorted({
            str(provider_status_by_id.get(provider_id, {}).get("node_id", "") or "")
            for provider_id in auto_provider_ids
            if provider_status_by_id.get(provider_id, {}).get("node_id")
        })
        retried_stages = [
            stage for stage in workflow.get("stages", [])
            if int(stage.get("retry_count", 0) or 0) > 0
        ]
        retry_error_codes = [
            str(stage.get("last_retry_error_code", "") or "")
            for stage in retried_stages
            if stage.get("last_retry_error_code")
        ]
        same_provider_retry_count = sum(
            int(stage.get("same_provider_retry_count", 0) or 0)
            for stage in retried_stages
        )
        total_retry_count = sum(
            int(stage.get("retry_count", 0) or 0)
            for stage in retried_stages
        )
        reassignment_count = max(
            0, total_retry_count - same_provider_retry_count,
        )
        fallback_used = reassignment_count > 0 or bool(
            auto_remote and not auto_provider_ids
        )
        retry_reason = retry_error_codes[0] if retry_error_codes else ""
        fallback_reason = (
            retry_reason if reassignment_count > 0 else auto_fallback_reason
        )
        metrics = _augment_chat_metrics(
            {
                "engine": self._host._engine_type,
                "execution_mode": "task_graph",
                "provider": (
                    providers[0] if len(providers) == 1 else "task_graph"
                ),
                "orchestrator": "task_graph",
                "subproviders": providers,
                "workflow_id": workflow["workflow_id"],
                "workflow_template": workflow["template"],
                "workflow_state": workflow["state"],
                "partial_result": bool(workflow.get("partial_result", False)),
                "stage_retry_count": total_retry_count,
                "same_provider_retry_count": same_provider_retry_count,
                "reassignment_count": reassignment_count,
                "retry_reason": retry_reason,
                "stage_count": workflow["stage_count"],
                "stage_attempt_count": workflow["attempt_count"],
                "nodes_planned": (
                    1 + len(planned_remote_nodes)
                    if auto_remote else len(participating_nodes)
                ),
                "nodes_participated": len(participating_nodes),
                "participating_nodes": participating_nodes,
                "distributed_requested": bool(remote_stage_id or auto_remote),
                "distributed_used": remote_used,
                "distributed_kind": (
                    "task_graph_remote_manual"
                    if remote_used and remote_stage_id
                    else "task_graph_remote_auto"
                    if remote_used and auto_remote
                    else "task_graph_local_fallback"
                    if auto_remote
                    else "task_graph_local_poc"
                ),
                "workers_used": remote_nodes,
                "manual_remote_stage": remote_stage_id,
                "manual_remote_provider": remote_provider_id,
                "auto_remote_enabled": auto_remote,
                "auto_remote_providers": auto_provider_ids,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "usage_estimated": any(
                    bool(attempt.get("result_metadata", {}).get(
                        "usage_estimated", False,
                    ))
                    for attempt in attempts
                ),
                "elapsed_seconds": workflow["duration_seconds"],
                "fallback": fallback_used,
                "fallback_reason": fallback_reason,
            },
            req,
            route=(
                f"{_chat_origin(req)}_to_task_graph_manual_remote"
                if remote_stage_id
                else f"{_chat_origin(req)}_to_task_graph_auto_remote"
                if auto_remote
                else f"{_chat_origin(req)}_to_local_task_graph"
            ),
        )
        followups = _fallback_followups(history, [])
        save_metrics = dict(metrics)
        save_metrics["followups"] = followups

        db_session_id = target_session_id or "default"
        if self._host._db_available:
            try:
                import db as _db_mod
                if _db_mod.get_save_history():
                    _db_mod.save_message(db_session_id, "user", req.message)
                    _db_mod.save_message(
                        db_session_id, "assistant", response_text, save_metrics,
                    )
                    _db_mod.increment_session_message_count(db_session_id)
            except Exception:
                pass
        if not self._host._db_available:
            try:
                _local_store.save_local_message(
                    db_session_id, "user", req.message,
                )
                _local_store.save_local_message(
                    db_session_id, "assistant", response_text, save_metrics,
                )
                _local_store.increment_local_session_message_count(db_session_id)
            except Exception:
                pass

        self._conversation_stats["total_prompt_tokens"] += prompt_tokens
        self._conversation_stats["total_generated_tokens"] += completion_tokens
        self._conversation_stats["total_time_seconds"] += workflow["duration_seconds"]
        self._conversation_stats["rounds"] += 1
        try:
            self._record_task_complete()
        except Exception:
            pass

        return {
            "content": response_text,
            "thinking_content": thinking_content,
            "metrics": metrics,
            "followups": followups,
        }



    # ------------------------------------------------------------------
    # 会话管理（1.2c 复制自 api_server.py:1090-1175，全局 → 实例属性）
    # ------------------------------------------------------------------
    def _get_active_history(self) -> list:
        """
        获取当前活跃会话的对话历史列表。

        如果没有活跃会话，返回空列表（不自动创建会话）。
        返回的列表对象可被原地修改（append、clear 等）。
        """
        if self._active_session_id is None:
            return []
        if self._active_session_id not in self._session_histories:
            self._session_histories[self._active_session_id] = []
        return self._session_histories[self._active_session_id]

    def _ensure_device_profile(self) -> Optional[dict]:
        """惰性设备画像探测（api_server 启动事件等价物，同步版）。"""
        if self._device_profile_started:
            return self._device_profile
        self._device_profile_started = True
        try:
            from device_profiler import get_profile
            profiler = get_profile()
            self._device_profile = profiler.to_dict()
            logger.info(
                f"🚀 设备检测完成: tier={profiler.tier.value} "
                f"score={profiler.score:.1f}/100"
            )
        except Exception as e:
            logger.error(f"设备检测失败: {e}")
            self._device_profile = None
        return self._device_profile

    def _init_kv_cache(self):
        """初始化分页 KV 缓存（根据设备画像自适应大小）。"""
        import torch as _torch
        from paged_kv_cache import PagedKVCache

        num_heads = 16      # Qwen-1.8B: 16 attention heads
        head_dim = 64
        mgr = self._host
        if getattr(mgr, "model", None) is not None:
            try:
                cfg = mgr.model.config
                num_heads = cfg.num_attention_heads
                head_dim = cfg.hidden_size // num_heads
            except Exception:
                pass

        device = str(mgr.get_device()) if callable(getattr(mgr, "get_device", None)) else "cpu"
        self._ensure_device_profile()
        if self._device_profile:
            self._kv_cache = PagedKVCache.from_profile(
                profile=self._device_profile,
                device=device,
                dtype=_torch.float16,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            logger.info(
                f"🧠 KV 缓存已初始化 (profile): num_heads={num_heads}, "
                f"head_dim={head_dim}, device={device}"
            )
        else:
            self._kv_cache = PagedKVCache(
                device=device,
                dtype=_torch.float16,
            )
            logger.info(
                f"🧠 KV 缓存已初始化 (default): device={device}"
            )
        return self._kv_cache

    def _switch_session(self, target_id: str) -> None:
        """
        切换到目标会话：暂存当前历史 → 加载目标历史 → 清 KV Cache。
        """
        if self._active_session_id == target_id:
            return
        self._active_session_id = target_id
        if target_id not in self._session_histories:
            messages = []
            if getattr(self._host, "_db_available", False):
                try:
                    import db as _db_mod
                    rows = _db_mod.get_conversation(target_id)
                    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
                except Exception:
                    pass
            if not messages:
                try:
                    import local_store as _local_store
                    local_rows = _local_store.load_local_conversation(target_id)
                    messages = [{"role": r["role"], "content": r["content"]} for r in local_rows]
                except Exception:
                    pass
            self._session_histories[target_id] = messages
        if self._kv_cache:
            self._kv_cache.clear()
        self._init_kv_cache()
        logger.info(f"已切换到会话: {target_id}")

    def _reset_runtime_conversation_state(self, clear_histories: bool = True) -> None:
        """Clear in-memory conversation/KV state after a model change."""
        if self._kv_cache:
            self._kv_cache.clear()
        self._kv_cache = None
        if clear_histories:
            self._session_histories = {}
        self._conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }

    def _auto_title_session(self, session_id: str, first_message: str) -> None:
        """用首条用户消息自动生成会话标题（截取前30字）"""
        title = first_message.strip()[:30]
        if len(first_message.strip()) > 30:
            title += "..."
        if getattr(self._host, "_db_available", False):
            try:
                import db as _db_mod
                _db_mod.update_session_title(session_id, title)
            except Exception:
                pass
        else:
            try:
                import local_store as _local_store
                _local_store.update_local_session_title(session_id, title)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 追问生成（1.2c 复制自 api_server.py:1228-1354，PyTorch 路径）
    # ------------------------------------------------------------------
    def _generate_followups(
        self,
        history: list,
        tokenizer,
        model,
        device,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """
        根据对话上下文，让模型生成 2-3 个追问建议。
        """
        import torch as _torch

        if not history or len(history) < 2:
            return []
        if cancel_event is not None and cancel_event.is_set():
            return []

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
            stop_criteria_kwargs = (
                {"cancel_event": cancel_event} if cancel_event is not None else {}
            )
            stop_criteria = self._host._build_stop_criteria(
                [], input_ids.shape[1], **stop_criteria_kwargs,
            )
            generation_kwargs = {}
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            with _torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )

            if cancel_event is not None and cancel_event.is_set():
                return []

            generated = outputs[0][input_ids.shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.upper().startswith("Q:") or line.upper().startswith("Q：") or line.startswith("问："):
                    q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                else:
                    q = re.sub(r'^[\d]+[\.\、\)）\s\-]+', '', line).strip()
                if q and len(q) >= 5 and len(q) <= 80 and _is_question(q):
                    questions.append(q)

            hallucination_patterns = [
                "通义千问", "千问", "ChatGPT", "Claude", "GPT-", "文心一言",
                "讯飞星火", "豆包", "Kimi", "Copilot", "Bard", "Gemini",
                "百川", "智谱", "ChatGLM", "混元",
            ]
            questions = [
                q for q in questions
                if not any(p in q for p in hallucination_patterns)
            ]

            filtered = []
            seen_words = set()
            for q in questions:
                words = frozenset(q[:10])
                if words not in seen_words:
                    seen_words.add(words)
                    filtered.append(q)
            questions = filtered

            logger.info(f"模型追问生成: {len(questions)} 条 → {questions}")

        except Exception as e:
            logger.warning(f"追问生成失败（非致命）: {e}")
            questions = []

        if len(questions) < 2:
            fallback = _fallback_followups(history, questions)
            questions = fallback

        return questions[:3]

    # ------------------------------------------------------------------
    # 路由决策与模型加载（1.2c 复制自 api_server.py:2261-2293 / 3882-3911 /
    # 3913-4011；scheduler 依赖 → self._scheduler 注入点，None=单机基线）
    # ------------------------------------------------------------------
    def _external_route_decision(self, req: "ChatRequest"):
        """按当前配置 + 请求 flag 计算外部路由决策（纯函数包装，读实时配置）。"""
        import config as _cfg
        from external_provider import decide_external_route

        return decide_external_route(
            enabled=bool(getattr(_cfg, "EXTERNAL_ENABLED", False)),
            base_url=str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or ""),
            data_scope=str(getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in")),
            allow_external=bool(req.allow_external),
            prefer_external=bool(req.prefer_external),
            prompt_chars=len(req.message or ""),
            min_prompt_chars=int(getattr(_cfg, "EXTERNAL_MIN_PROMPT_CHARS", 0) or 0),
        )

    def _should_forward_chat_to_master(self) -> bool:
        sched = self._scheduler
        return bool(
            sched is not None
            and sched.get_distributed_inference_enabled()
            and self._run_mode == "distributed"
            and sched._effective_role() == "client"
        )

    def _pipeline_worker_is_reserved(self) -> bool:
        sched = self._scheduler
        check = getattr(sched, "has_pipeline_worker_reservation", None) if sched else None
        return bool(callable(check) and check())

    def _run_exclusive_model_change(
        self, change, prepare=None, *, release_worker_reservation: bool = False,
    ):
        """Block inference, invalidate old worker ACKs, then refresh the new model.

        （api_server 版嵌套 scheduler 锁；本进程无 scheduler 时仅持
        ModelHost 执行锁，等价单机拓扑。）
        """
        with self._host.full_chat_execution_lock:
            if prepare is not None:
                prepare()
            sched = self._scheduler
            if sched is not None:
                with sched._inference_lock:
                    with sched._layer_execution_lock:
                        with sched._layer_config_lock:
                            sched._layer_config_pushed.clear()
                            sched._layer_config_expected.clear()
                            sched._layer_config_acks.clear()
                            sched._active_layer_config = None
                            sched._last_layer_config_ack_payload = None
                            sched._local_pipeline_steps.clear()
                        if release_worker_reservation:
                            release = getattr(
                                sched,
                                "release_pipeline_worker_for_local_model",
                                None,
                            )
                            if callable(release):
                                release()
                            else:
                                sched._pipeline_worker_reserved = False
            try:
                return change()
            finally:
                if sched is not None:
                    self._refresh_pipeline_layer_config(sched)

    def _refresh_pipeline_layer_config(self, sched) -> None:
        """主节点模型变化后重新下发层配置，并使旧 ACK 失效。"""
        try:
            if sched._effective_role() == "master":
                sched.push_layer_config_to_clients()
        except Exception:
            pass

    def _auto_load_default_model(self):
        """自动加载默认模型（thin client / 首次请求时服务端无模型的情况）。"""
        import config as cfg
        import glob
        import os as _os
        import time as _time

        # 0. TP 孤岛引擎优先（启用即为孤岛网关节点，无本地文件依赖）
        if getattr(cfg, "ISLAND_ENABLED", False) and getattr(cfg, "ISLAND_BASE_URL", ""):
            from island_engine import mask_island_url

            logger.info(
                f"自动加载孤岛引擎: endpoint={mask_island_url(cfg.ISLAND_BASE_URL)}"
            )
            t0 = _time.time()
            cfg.INFERENCE_ENGINE = "island"
            cfg.QUANT_TYPE = "island"
            cfg.USE_COMPILE = False
            self._run_exclusive_model_change(
                lambda: self._host.load_model(
                    profile=self._ensure_device_profile(),
                    engine="island",
                )
            )
            self._init_kv_cache()
            self._reset_runtime_conversation_state(clear_histories=False)
            self._conversation_stats = {
                "total_prompt_tokens": 0,
                "total_generated_tokens": 0,
                "total_time_seconds": 0.0,
                "rounds": 0,
            }
            self._host.model_loaded = True
            self._host.current_quant = "island"
            if self._scheduler is not None:
                try:
                    self._scheduler.refresh_task_worker_capabilities()
                except Exception:
                    pass
            logger.info(f"✅ 孤岛引擎自动连接完成 ({_time.time() - t0:.1f}s)")
            return

        # 1. 优先查找 GGUF 文件（llama.cpp 引擎，不依赖 transformers/bitsandbytes）
        gguf_candidates = []
        gguf_configured = cfg.GGUF_MODEL_PATH
        if _os.path.isfile(gguf_configured):
            gguf_candidates.append(gguf_configured)
        models_dir = _os.path.dirname(gguf_configured)
        if _os.path.isdir(models_dir):
            for f in sorted(glob.glob(_os.path.join(models_dir, "*.gguf"))):
                if f not in gguf_candidates:
                    gguf_candidates.append(f)

        if gguf_candidates:
            gguf_path = gguf_candidates[0]
            engine = "llama_cpp"
            model_path = gguf_path
            quant = "int4"
            if len(gguf_candidates) > 1:
                logger.info(f"发现 {len(gguf_candidates)} 个 GGUF 文件，选择: {_os.path.basename(gguf_path)}")
        elif _os.path.isdir(cfg.MODEL_PATH):
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

        t0 = _time.time()
        cfg.INFERENCE_ENGINE = engine
        cfg.QUANT_TYPE = quant
        cfg.USE_COMPILE = False

        self._run_exclusive_model_change(
            lambda: self._host.load_model(
                model_path=model_path,
                quant_type=quant,
                profile=self._ensure_device_profile(),
                engine=engine,
            )
        )

        self._init_kv_cache()
        self._reset_runtime_conversation_state(clear_histories=False)
        self._conversation_stats = {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        }
        self._host.model_loaded = True
        self._host.current_quant = getattr(self._host, "quant_type", None) or quant
        if self._scheduler is not None:
            try:
                self._scheduler.refresh_task_worker_capabilities()
            except Exception:
                pass
        elapsed = _time.time() - t0
        logger.info(f"默认模型自动加载完成 ({elapsed:.1f}s)")

    def _ensure_chat_model_or_forwarding(self, req: Optional["ChatRequest"] = None) -> None:
        """Load a local model only when this request cannot be master-forwarded."""
        if req is not None and self._external_route_decision(req).use_external:
            return
        if self._should_forward_chat_to_master():
            return
        if self._pipeline_worker_is_reserved():
            from fastapi import HTTPException
            raise HTTPException(
                503,
                "本设备正作为 PyTorch 分层从节点，不能加载本地完整模型。",
            )
        if self._host.model_loaded and getattr(self._host, "is_loaded", False):
            return
        self._auto_load_default_model()

    # ------------------------------------------------------------------
    # 外部推理整请求路由（1.2c 复制自 api_server._execute_external_chat
    # api_server.py:2294-2400；宿主适配：conversation_stats → 实例属性、
    # model_host._db_available → self._host、scheduler.record_task_complete
    # → self._on_task_complete 回调（1.4 注入））
    # ------------------------------------------------------------------
    def execute_external_chat(
        self,
        req: ChatRequest,
        history: list,
        target_session_id: Optional[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """整请求路由到外部推理服务（与 llama.cpp/孤岛整请求路径同构）。"""
        import config as _cfg
        from external_provider import get_external_chat_client

        client = get_external_chat_client()
        client.ensure_connected()
        request_history = [
            *history,
            {"role": "user", "content": req.message},
        ]
        result = client.chat(
            request_history,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            allow_external=req.allow_external,
            cancel_event=cancel_event,
        )
        _raise_if_generation_cancelled(cancel_event, req.generation_id)
        response_text = result.get("content", "")
        if not req.show_thinking:
            response_text = _strip_native_thinking_tags(response_text)
        completed_history = [
            *request_history,
            {"role": "assistant", "content": response_text},
        ]
        tokens_per_sec = result.get("tokens_per_second", 0)
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        metrics = _augment_chat_metrics(
            {
                "engine": "external_api",
                "execution_mode": "external_api",
                "route": f"{_chat_origin(req)}_to_external_api",
                "provider": "external_openai",
                "external_label": getattr(_cfg, "EXTERNAL_LABEL", ""),
                "external_base_url": client.masked_base_url,
                "data_scope": getattr(_cfg, "EXTERNAL_DATA_SCOPE", ""),
                "model": result.get("model", "") or client.model_name,
                "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else 0,
                "generated_tokens": completion_tokens,
                "completion_tokens": completion_tokens,
                "usage": usage,
                "usage_estimated": bool(result.get("usage_estimated", False)),
                "fallback": False,
                "fallback_reason": "",
            },
            req,
            serving_node_id=self._serving_node_id,
            distributed_enabled=self._distributed_enabled,
        )

        db_session_id = target_session_id or "default"
        followups = _fallback_followups(completed_history, [])
        history.extend([
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": response_text},
        ])

        if getattr(self._host, "_db_available", False):
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
        else:
            try:
                import local_store as _local_store
                _local_store.save_local_message(db_session_id, "user", req.message)
                save_metrics = dict(metrics)
                save_metrics["followups"] = followups
                _local_store.save_local_message(db_session_id, "assistant",
                                                response_text, save_metrics)
                _local_store.increment_local_session_message_count(db_session_id)
            except Exception:
                pass

        self._conversation_stats["total_generated_tokens"] += completion_tokens
        self._conversation_stats["rounds"] += 1
        if self._on_task_complete is not None:
            try:
                self._on_task_complete(success=True)
            except Exception:
                pass

        logger.info(
            f"外部推理完成: {completion_tokens} tokens, "
            f"endpoint={client.masked_base_url}"
        )
        return {
            "content": response_text,
            "thinking_content": None,
            "metrics": metrics,
            "followups": followups,
        }

    # ------------------------------------------------------------------
    # generation 注册表（取消语义，对齐 api_server._register_generation）
    # ------------------------------------------------------------------
    def register_generation(
        self, generation_id: Optional[str] = None
    ) -> Tuple[str, threading.Event]:
        with self._gen_lock:
            gid = generation_id or f"gen_{uuid4().hex[:12]}"
            ev = self._generations.get(gid)
            if ev is None:
                ev = threading.Event()
                self._generations[gid] = ev
            return gid, ev

    def unregister_generation(self, generation_id: str) -> None:
        with self._gen_lock:
            self._generations.pop(generation_id, None)

    def cancel_generation(self, generation_id: str) -> bool:
        """置取消事件；返回 False 表示 generation_id 未知（404 语义）。"""
        with self._gen_lock:
            ev = self._generations.get(generation_id)
            if ev is None:
                return False
            ev.set()
            return True
