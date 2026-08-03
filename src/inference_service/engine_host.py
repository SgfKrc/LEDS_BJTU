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
            "total_generated_tokens": 0,
            "rounds": 0,
        }
        # 1.4 注入：scheduler-svc 侧任务完成回调（当前 no-op，进程内基线）
        self._on_task_complete = None
        # 1.4 注入：scheduler-svc 下发的节点身份/分布式开关（metrics 用）
        self._serving_node_id = ""
        self._distributed_enabled = False

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
    def chat_full(self, req: ChatRequest) -> Dict[str, Any]:
        """完整对话响应（对齐 api_server /api/chat 响应形状）。"""
        messages = [{"role": "user", "content": req.message}]
        result = self._host.chat(
            messages=messages,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        if isinstance(result, dict):
            response = result.get("content") or result.get("response") or str(result)
            metrics = {
                k: result[k]
                for k in ("tokens_per_second", "usage")
                if k in result
            }
        else:
            response = str(result)
            metrics = {}
        return {
            "response": response,
            "followups": [],
            "metrics": metrics,
            "request_id": "-",
        }

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
