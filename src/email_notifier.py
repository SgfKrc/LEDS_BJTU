"""
邮件通知模块 — 主节点宕机/恢复时向管理员发送告警邮件
======================================================
使用 QQ 邮箱 SMTP SSL 发送，无需额外依赖（仅使用标准库 smtplib + email）。

配置来源: SMTP.md（发信邮箱 studyp4ct@qq.com，授权码 vfcrzzlxbpwxcafb）
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# SMTP 配置（QQ 邮箱）
# ============================================================
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465                     # SSL
SMTP_SENDER = "studyp4ct@qq.com"
SMTP_PASSWORD = "vfcrzzlxbpwxcafb"  # QQ 邮箱授权码（非密码）
SMTP_RECIPIENT = "2743631775@qq.com"


def send_master_down_alert(
    master_host: str,
    master_port: int,
    downtime_seconds: float,
    last_seen_seconds_ago: float = None,
    client_node_id: str = "",
) -> bool:
    """
    发送主节点宕机告警邮件。

    Args:
        master_host: 主节点 IP 地址
        master_port: 主节点监听端口
        downtime_seconds: 宕机持续秒数
        last_seen_seconds_ago: 上次心跳距今秒数
        client_node_id: 发送告警的从节点 ID

    Returns:
        发送是否成功
    """
    subject = f"⚠️ 分布式推理系统告警：主节点 {master_host}:{master_port} 宕机"

    downtime_min = downtime_seconds / 60
    last_seen_str = f"{last_seen_seconds_ago:.0f} 秒前" if last_seen_seconds_ago else "未知"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""\
分布式推理系统 — 主节点宕机告警

══════════════════════════════════════
  告警时间:     {now_str}
  主节点地址:   {master_host}:{master_port}
  上报节点:     {client_node_id or '从节点'}
  上次心跳:     {last_seen_str}
  宕机持续:     {downtime_min:.1f} 分钟
══════════════════════════════════════

主节点已停止发送数据库心跳，从节点无法正常执行分布式推理任务。

建议操作:
  1. 检查主节点设备是否正常运行
  2. 检查主节点后端进程 (uvicorn) 是否存活
  3. 检查主节点与云数据库的网络连接是否正常
  4. 如主节点已恢复，从节点将自动重连并发送恢复通知

此邮件由分布式推理系统自动发送，请勿回复。
"""

    return _send_email(subject, body)


def send_master_recovery_alert(
    master_host: str,
    master_port: int,
    total_downtime_seconds: float,
    client_node_id: str = "",
) -> bool:
    """
    发送主节点恢复通知邮件。

    Args:
        master_host: 主节点 IP 地址
        master_port: 主节点监听端口
        total_downtime_seconds: 总宕机时长秒数
        client_node_id: 发送通知的从节点 ID

    Returns:
        发送是否成功
    """
    subject = f"✅ 分布式推理系统恢复：主节点 {master_host}:{master_port} 已上线"

    downtime_min = total_downtime_seconds / 60
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""\
分布式推理系统 — 主节点恢复通知

══════════════════════════════════════
  恢复时间:     {now_str}
  主节点地址:   {master_host}:{master_port}
  上报节点:     {client_node_id or '从节点'}
  宕机总时长:   {downtime_min:.1f} 分钟
══════════════════════════════════════

主节点已恢复在线，数据库心跳已恢复。
从节点将自动重连至主节点，分布式推理任务可正常执行。

此邮件由分布式推理系统自动发送，请勿回复。
"""

    return _send_email(subject, body)


def send_test_email() -> bool:
    """
    发送一封测试邮件，验证 SMTP 配置是否正确。

    Returns:
        发送是否成功
    """
    subject = "🧪 分布式推理系统 — SMTP 邮件测试"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""\
分布式推理系统 — SMTP 邮件测试

发送时间: {now_str}

如果您收到此邮件，说明 SMTP 邮件告警配置正确，
主节点宕机时将自动向此邮箱发送告警通知。

此邮件由分布式推理系统自动发送，请勿回复。
"""

    return _send_email(subject, body)


def _send_email(subject: str, body: str) -> bool:
    """
    通过 QQ 邮箱 SMTP SSL 发送邮件。

    使用标准库 smtplib，无额外依赖。
    发送失败时记录错误日志并返回 False，不抛出异常。
    """
    msg = MIMEMultipart()
    msg["From"] = SMTP_SENDER
    msg["To"] = SMTP_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_SENDER, SMTP_PASSWORD)
            server.sendmail(SMTP_SENDER, [SMTP_RECIPIENT], msg.as_string())
        logger.info(f"📧 邮件已发送: {subject}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"邮件发送失败（SMTP 认证错误，请检查授权码是否正确）: {e}")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"邮件发送失败（SMTP 协议错误）: {e}")
        return False
    except Exception as e:
        logger.error(f"邮件发送失败（未知错误）: {e}")
        return False
