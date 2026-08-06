"""T9 共享层 — 聊天页与管理 TUI 共用的路径、格式化与命令注册。

设计（TUI 适配实施计划 §9.3）：
- 只含纯函数与常量，无第三方依赖（tui_admin.py 的标准库环境也可安全导入）；
- 端点字符串集中在 API_PATHS，聊天页与管理 TUI 不各自散落复制；
- metrics 展示只依据 done 事件实际字段，不推断分布式参与。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ============================================================
# API 端点路径（相对 /api 前缀；host 由调用方拼接）
# ============================================================

API_PATHS = {
    "chat_stream": "/chat/stream",
    "chat_cancel": "/chat/generations/{generation_id}/cancel",
    "sessions": "/sessions",
    "sessions_detail": "/sessions/{session_id}",
    "sessions_activate": "/sessions/{session_id}/activate",
    "conversations": "/conversations",
    "models_current": "/models/current",
    "distributed_config": "/cluster/config/distributed-inference",
}

# ============================================================
# 路由偏好
# ============================================================

ROUTING_PREFERENCES = (
    "auto",
    "local_only",
    "distributed_preferred",
    "distributed_required",
)

ROUTE_LABELS = {
    "auto": "route:auto",
    "local_only": "route:local",
    "distributed_preferred": "route:distributed",
    "distributed_required": "route:required",
}

ROUTE_SHORT_ARGS = {
    "auto": "auto",
    "local": "local_only",
    "distributed": "distributed_preferred",
    "required": "distributed_required",
}


# ============================================================
# interactive 请求体构造
# ============================================================

def build_interactive_request(
    message: str,
    *,
    session_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    routing_preference: str = "auto",
    show_thinking: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """构造 /api/chat/stream 的 interactive 请求体（T9 契约 §9.4.1）。"""
    if routing_preference not in ROUTING_PREFERENCES:
        routing_preference = "auto"
    return {
        "message": message,
        "streaming_mode": "interactive",
        "generation_id": generation_id,
        "session_id": session_id,
        "routing_preference": routing_preference,
        "show_thinking": show_thinking,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


# ============================================================
# metrics 格式化（done 事件）
# ============================================================

def format_metrics(
    metrics: Optional[Dict[str, Any]] = None,
    *,
    history_committed: Optional[bool] = None,
) -> str:
    """把 done 事件 metrics 格式化为状态行文本。

    展示规则（§9.5）：只读取完成事件中的实际字段；fallback 必须展示原因；
    history_committed=false 必须提示。
    """
    metrics = metrics or {}
    parts: List[str] = []
    engine = metrics.get("engine") or metrics.get("execution_mode") or "unknown"
    route = metrics.get("execution_mode") or "local"
    parts.append(f"{engine} · {route}")
    tokens = metrics.get("tokens_generated") or metrics.get("generated_tokens")
    if tokens is not None:
        parts.append(f"{tokens} tokens")
    tok_s = metrics.get("tok_per_sec") or metrics.get("tokens_per_second")
    if tok_s is not None:
        try:
            parts.append(f"{float(tok_s):.1f} tok/s")
        except (TypeError, ValueError):
            parts.append(f"{tok_s} tok/s")
    if metrics.get("fallback"):
        parts.append(f"⚠️ 回退: {metrics.get('fallback_reason', '未知')}")
    if metrics.get("distributed_requested") and not metrics.get("distributed_used"):
        parts.append("已请求分布式，实际本地")
    if history_committed is False:
        parts.append("历史未提交")
    return " · ".join(parts)


def parse_session_line(session: Dict[str, Any]) -> str:
    """会话 dict → 单行显示文本（兼容后端 id / session_id 两种字段）。"""
    session_id = session.get("session_id") or session.get("id") or ""
    title = session.get("title") or "(未命名)"
    count = session.get("message_count")
    suffix = f" · {count} 条" if count is not None else ""
    return f"{session_id}  {title}{suffix}"


# ============================================================
# T9 命令注册表（/help 与校验共用）
# ============================================================

COMMAND_SPECS: List[Dict[str, str]] = [
    {"name": "/new", "args": "", "desc": "创建并切换新会话"},
    {"name": "/resume", "args": "<session_id>", "desc": "恢复历史会话"},
    {"name": "/rename", "args": "<title>", "desc": "重命名当前会话"},
    {"name": "/delete-session", "args": "", "desc": "删除当前会话"},
    {"name": "/route", "args": "auto|local|distributed|required",
     "desc": "设置请求级路由偏好"},
    {"name": "/cancel", "args": "", "desc": "取消当前生成"},
    {"name": "/clear", "args": "", "desc": "清空当前会话（需确认）"},
    {"name": "/thinking", "args": "on|off", "desc": "思考内容展示"},
    {"name": "/sessions", "args": "", "desc": "列出最近会话"},
    {"name": "/help", "args": "", "desc": "显示本帮助"},
    {"name": "/quit", "args": "", "desc": "退出聊天页"},
]


def help_text() -> str:
    """生成 /help 的一行文本。"""
    items = []
    for spec in COMMAND_SPECS:
        parts = [spec["name"]]
        if spec["args"]:
            parts.append(spec["args"])
        items.append(" ".join(parts))
    return "可用命令：" + " ".join(items) + "；Enter 发送 · Alt+Enter 换行 · Ctrl+C 停止"


def resolve_route_arg(arg: str) -> Optional[str]:
    """把 /route 参数解析为完整 routing_preference；非法返回 None。"""
    normalized = (arg or "").strip().lower()
    if normalized in ROUTING_PREFERENCES:
        return normalized
    return ROUTE_SHORT_ARGS.get(normalized)
