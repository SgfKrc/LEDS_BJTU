"""T9 共享层 — 聊天页与管理 TUI 共用的路径、格式化与命令注册。

设计（TUI 适配实施计划 §9.3）：
- 只含纯函数与常量，无第三方依赖（tui_admin.py 的标准库环境也可安全导入）；
- 端点字符串集中在 API_PATHS，聊天页与管理 TUI 不各自散落复制；
- metrics 展示只依据 done 事件实际字段，不推断分布式参与。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from multimodal import (
        MAX_CHAT_IMAGE_BYTES,
        MAX_CHAT_IMAGES,
        MAX_CHAT_IMAGE_TOTAL_BYTES,
        validate_image_data_urls,
    )
except ImportError:  # pragma: no cover - package import path
    from .multimodal import (  # type: ignore
        MAX_CHAT_IMAGE_BYTES,
        MAX_CHAT_IMAGES,
        MAX_CHAT_IMAGE_TOTAL_BYTES,
        validate_image_data_urls,
    )

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
    "cluster_resources": "/cluster/resources",
    "cluster_control_plane": "/cluster/control-plane",
    "cluster_management_score": "/cluster/management-score",
    "cluster_layers": "/cluster/layers",
    "cluster_pipeline_capacity": "/cluster/pipeline-capacity",
    "cluster_pipeline_reshard": "/cluster/pipeline-reshard",
    # 只读运维面（Textual 外壳的节点/队列/日志屏）
    "cluster_nodes": "/cluster/nodes",
    "cluster_queue": "/cluster/queue",
    "cluster_log_aggregate": "/cluster/nodes/log-aggregate",
    # 设备画像（后端所在机器）
    "device_profile": "/device/profile",
    # 2026-09-17 按后端真实返回对齐（此前沿用旧 TUI 假设，导致多屏显示"无数据"）
    "health": "/health",
    "readiness": "/ready",
    "system_status": "/status",
    "models_list": "/models",

    # ---- 写操作面（用户 2026-09-17 裁决：模型控制 + 会话管理 + 队列控制）----
    # 权限：后端 model_api_access.require_model_api_source() 对 **loopback 默认放行**，
    # 故本机 TUI 可直接调用；从节点远程控制主节点需显式 QLH_MODEL_API_TRUSTED_CIDRS。
    "models_load": "/models/load",          # POST {engine, quant_type, use_compile, model_id}
    "models_unload": "/models/unload",      # POST（无请求体）
    "chat_clear": "/chat/clear",            # POST 清空后端当前会话历史 + KV 缓存
    "session_detail": "/sessions/{session_id}",            # PUT 重命名 / DELETE 删除
    "session_activate": "/sessions/{session_id}/activate",  # POST 恢复会话
    "cluster_queue_pause": "/cluster/queue/pause",          # POST（仅主节点）
    "cluster_queue_resume": "/cluster/queue/resume",        # POST（仅主节点）
    "cluster_queue_strategy": "/cluster/queue/strategy",    # POST {strategy: fifo|mlfq}
    "cluster_queue_clear": "/cluster/queue/clear",          # POST（仅主节点）
    # ---- 2026-09-19 补缺口 A：队列「单任务」取消（此前只能整体 clear）----
    "cluster_queue_task_cancel": "/cluster/queue/task/{task_id}",  # DELETE（仅主节点）
    # ---- 2026-09-19 补缺口 B：日志细粒度（此前只有 recent/stats/export/整体清理）----
    "logs_list": "/logs",                         # GET 日志文件列表
    "logs_download": "/logs/download",            # GET 下载
    "logs_file": "/logs/{filename}",              # GET 读单文件 / DELETE 删单文件
    "logs_nodes_summary": "/logs/nodes-summary",  # GET 各节点日志汇总
    # ---- 2026-09-19 补缺口 E：集群高可用（备用主节点 / 主节点转让 / 身份重置）----
    "cluster_master_health": "/cluster/master-health",       # GET 主节点健康
    "cluster_transfer_logs": "/cluster/transfer-logs",       # GET 角色转让日志
    "cluster_spare_master": "/cluster/spare-master",         # GET 查询 / POST 指定 / DELETE 清除
    "cluster_spare_master_logs": "/cluster/spare-master/logs",  # GET 备用主节点操作日志
    "cluster_transfer_master": "/cluster/transfer-master",   # POST ⚠️ 高危（需重启）
    "cluster_reset_identity": "/cluster/reset-identity",     # POST ⚠️ 高危（需 confirm="reset"）
    # ---- 2026-09-19 补缺口 C：模型资产浏览（只读）----
    "models_available": "/models/available",       # GET 可选模型配置 + 可用引擎
    "models_registry": "/models/registry",         # GET 用户注册的实验模型
    "models_downloadable": "/models/downloadable",  # GET 可下载清单
    "models_gguf": "/models/gguf",                 # GET 本地 GGUF 文件
    # ---- 2026-09-19 补缺口 D：存储健康（只读）----
    "db_health": "/db/health",                     # GET SQLite 健康
    "storage_health": "/storage/health",           # GET 存储健康
    # ---- 2026-09-19 补缺口 F：会话细粒度（读为主）----
    "session_info": "/sessions/{session_id}",              # GET 会话元数据
    "session_turn": "/sessions/{session_id}/turns/{turn_index}",  # DELETE 删单轮
    "conversation_sync_status": "/conversations/sync-status",  # GET 持久化状态
    # ---- 2026-09-19 补缺口 G-⑤：认证与账户（monolith 内实现）----
    "auth_capability": "/auth/capability",           # GET 能力（required / bootstrap_open）
    "auth_login": "/auth/login",                     # POST {username,password,totp_code?}
    "auth_logout": "/auth/logout",                   # POST
    "auth_me": "/auth/me",                           # GET
    "auth_totp_provision": "/auth/totp/provision",   # POST
    "auth_totp_verify": "/auth/totp/verify",         # POST {code}
    "auth_users": "/users",                          # GET 列表 / POST 创建
    "auth_user": "/users/{username}",                # PATCH 修改 / DELETE 删除
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
    enable_thinking: Optional[bool] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    image_data_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """构造 /api/chat/stream 的 interactive 请求体（T9 契约 §9.4.1）。"""
    if routing_preference not in ROUTING_PREFERENCES:
        routing_preference = "auto"
    images = validate_image_data_urls(image_data_urls or [])
    local_image = bool(images and routing_preference == "local_only")
    if local_image and len(images) != 1:
        raise ValueError("本地 MTMD 图像请求仅支持一张图片")
    body = {
        "message": message,
        "streaming_mode": "full" if local_image else "interactive",
        "generation_id": generation_id,
        "session_id": session_id,
        "routing_preference": routing_preference,
        "show_thinking": show_thinking,
        "enable_thinking": enable_thinking,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if images:
        body.update({
            "image_data_urls": images,
            "execution_mode": "auto",
            "allow_external": not local_image,
            "prefer_external": not local_image,
        })
    return body


def load_local_chat_image(path_value: str) -> Dict[str, Any]:
    """读取 TUI 用户明确指定的本地图片并构造一次性 data URL。"""
    candidate = (path_value or "").strip().strip('"')
    if not candidate:
        raise ValueError("用法: /image <PNG/JPEG/WebP 本地路径>")
    path = Path(candidate).expanduser()
    try:
        if not path.is_file():
            raise ValueError("图片路径不存在或不是普通文件")
        size = path.stat().st_size
        if size > MAX_CHAT_IMAGE_BYTES:
            raise ValueError("图片超过单张 8 MiB 上限")
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取图片: {exc}") from exc

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("只支持内容正确的 PNG、JPEG 或 WebP 图片")

    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    validate_image_data_urls([data_url])
    return {
        "name": path.name,
        "size": size,
        "data_url": data_url,
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

#: 命令注册表 —— **只登记已实现的命令**（ChatPane.on_input_submitted 分支与之一一对应）；
#: 未实现的旧规范条目（/image、/images、/image-clear）已移除，避免 /help "写着却不能用"。
COMMAND_SPECS: List[Dict[str, str]] = [
    {"name": "/new", "args": "[title]", "desc": "新建并切换会话（POST /sessions）"},
    {"name": "/resume", "args": "<session_id>", "desc": "恢复历史会话并渲染其消息"},
    {"name": "/rename", "args": "<title>", "desc": "重命名当前会话"},
    {"name": "/sessions", "args": "", "desc": "列出最近会话"},
    {"name": "/delete-session", "args": "", "desc": "删除当前会话及其全部消息（需确认）"},
    {"name": "/reset", "args": "", "desc": "清空后端会话历史与 KV 缓存（需确认）"},
    {"name": "/model", "args": "load <id> [engine] [quant] | unload",
     "desc": "加载/卸载模型（需确认；仅 loopback 后端可调用）"},
    {"name": "/queue",
     "args": "pause | resume | strategy <fifo|mlfq> | clear | cancel <task_id>",
     "desc": "队列控制（clear 需确认）"},
    {"name": "/route", "args": "auto|local|distributed|required",
     "desc": "设置请求级路由偏好"},
    {"name": "/thinking", "args": "on|off", "desc": "思考内容**展示**（仅 UI 显隐，不改变模型行为）"},
    {"name": "/reasoning", "args": "on|off|auto",
     "desc": "深度思考**开关**（改变模型行为）：on=强制思考 / off=强制不思考（省算力，"
             "可避免 Qwen3 等模型输出超长 `<think>`）/ auto=沿用模型模板默认"},
    {"name": "/cancel", "args": "", "desc": "取消当前生成"},
    {"name": "/logs", "args": "list | download <file> | read <file> | delete <file> | nodes",
     "desc": "日志细粒度：文件列表 / 下载 / 查看 / 删除（需确认）/ 各节点汇总"},
    {"name": "/login", "args": "<username> <password> [totp_code]",
     "desc": "登录（若账户已绑定 Auth App，需附 6 位验证码）——凭据仅本进程内存持有，不落盘"},
    {"name": "/logout", "args": "", "desc": "注销（服务端吊销当前登录态）"},
    {"name": "/whoami", "args": "", "desc": "显示当前登录主体与认证能力"},
    {"name": "/users", "args": "list | add <name> <pass> [role] | role <name> <role> "
                               "| disable|enable <name> | passwd <name> <pass> | del <name>",
     "desc": "账户管理（需 admin）：列表 / 创建 / 改角色 / 启用禁用 / 重置口令 / 删除"},
    {"name": "/totp", "args": "provision | verify <code>",
     "desc": "Auth App 绑定：provision 生成密钥与 otpauth URI；verify 校验一次"},
    {"name": "/history", "args": "[<session_id>] [limit] | sync-status | info <session_id> "
                              "| drop-turn <session_id> <turn_index>",
     "desc": "会话历史：查看对话（默认当前会话）/ 本地持久化状态 / 会话详情 / "
             "删单轮（需确认，删 user+assistant 两条）"},
    {"name": "/assets", "args": "available | registry | downloadable | gguf",
     "desc": "模型资产浏览（只读）：可选模型与引擎 / 已注册实验模型 / 可下载清单 / 本地 GGUF"},
    {"name": "/storage", "args": "", "desc": "存储与数据库健康（只读）"},
    {"name": "/ha", "args": "health | transfer-logs | spare | spare-logs | designate <node> "
                            "| clear-spare | transfer <node> | reset-identity",
     "desc": "集群高可用：健康 / 转让日志 / 备用主节点；⚠️ transfer 与 reset-identity 为高危"
             "（转让后需重启，身份重置不可撤销）"},
    {"name": "/clear", "args": "", "desc": "清空本地显示（不动后端；清后端用 /reset）"},
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
