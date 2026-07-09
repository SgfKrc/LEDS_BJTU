"""
邮件通知模块 — 主节点宕机/恢复时向管理员发送告警邮件 + 邮件回复投票
==================================================================
SMTP 发送: QQ 邮箱 SMTP SSL (smtp.qq.com:465)
IMAP 轮询: QQ 邮箱 IMAP SSL (imap.qq.com:993) — 用于管理员回复 Y/N 投票

配置来源: SMTP.md（发信邮箱 studyp4ct@qq.com，授权码 vfcrzzlxbpwxcafb）
"""

import imaplib
import logging
import os
import re
import smtplib
import time
import threading
from email import message_from_bytes, policy
from email.header import decode_header
from email.mime.text import MIMEText
from datetime import datetime
from typing import Optional, Set

logger = logging.getLogger(__name__)

# ============================================================
# SMTP / IMAP 配置（QQ 邮箱）
# ============================================================
SMTP_SERVER = os.environ.get("QLH_SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("QLH_SMTP_PORT", "465"))
IMAP_SERVER = os.environ.get("QLH_IMAP_SERVER", "imap.qq.com")
IMAP_PORT = int(os.environ.get("QLH_IMAP_PORT", "993"))
SMTP_SENDER = os.environ.get("QLH_SMTP_SENDER", "")
SMTP_PASSWORD = os.environ.get("QLH_SMTP_PASSWORD", "")  # QQ 邮箱授权码
SMTP_RECIPIENT = os.environ.get("QLH_SMTP_RECIPIENT", "")

# 邮件投票轮询间隔（秒）
MAIL_POLL_INTERVAL = 60


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


# ================================================================
# P3: 审查票邮件通知
# ================================================================

def send_review_created_alert(
    ticket_id: str,
    created_by: str,
    target_node_id: str,
    reason: str = "",
    expires_at: float = 0.0,
) -> bool:
    """审查工单创建时发送邮件通知管理员。

    Args:
        ticket_id: 工单 ID
        created_by: 发起者 node_id
        target_node_id: 拟转让目标
        reason: 转让原因
        expires_at: 过期时间（Unix timestamp）

    Returns:
        发送是否成功
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    expire_str = (
        datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S")
        if expires_at > 0 else "未知"
    )

    subject = f"[REVIEW-VOTE] {ticket_id} (reply Y/N)"

    body = f"""\
分布式推理系统 — 主节点转让审查工单

══════════════════════════════════════
  工单 ID:      {ticket_id}
  创建时间:     {now_str}
  发起者:       {created_by}
  拟转让目标:   {target_node_id}
  转让原因:     {reason or '(未填写)'}
  过期时间:     {expire_str}
══════════════════════════════════════

主节点 '{created_by}' 请求将主节点角色转让给从节点 '{target_node_id}'。

📧 邮件快速投票（管理员）：
  直接回复此邮件，正文以 Y 或 N 开头：
    Y / Yes  →  赞同 (+1)
    N / No   →  阻止 (-1)
  其他任何内容（空、中文等）均视为无效，系统会回复提示。

🖥️ Web 投票（PC 独显节点）：
  登录管理面板 →「管理」→「主节点转让审查」→ 选择 +1 / 0 / -1

通过条件: 合计 >= +2 票  |  阻止条件: 合计 <= -2 票

此邮件由分布式推理系统自动发送。
"""

    return _send_email(subject, body)


def send_review_resolved_alert(
    ticket_id: str,
    status: str,
    score: int,
    target_node_id: str,
) -> bool:
    """审查工单已解决时发送邮件通知。

    Args:
        ticket_id: 工单 ID
        status: "approved" | "rejected" | "expired"
        score: 最终得分
        target_node_id: 转让目标

    Returns:
        发送是否成功
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_labels = {
        "approved": "✅ 审查通过",
        "rejected": "❌ 审查被阻止",
        "expired": "⏰ 审查已过期",
    }
    status_label = status_labels.get(status, status)

    subject = f"审查结果: {ticket_id} — {status_label}"

    body = f"""\
分布式推理系统 — 审查工单结果

══════════════════════════════════════
  工单 ID:      {ticket_id}
  结果:         {status_label}
  最终得分:     {score}
  拟转让目标:   {target_node_id}
  完成时间:     {now_str}
══════════════════════════════════════

"""

    if status == "approved":
        body += f"""\
审查已通过（得分 {score} >= +2）。管理员可以执行主节点转让操作。

转让操作:
  1. 登录主节点管理面板
  2. 进入「管理」→「角色转让」
  3. 选择目标节点 '{target_node_id}' 执行转让
"""
    elif status == "rejected":
        body += f"""\
审查被阻止（得分 {score} <= -2）。此转让请求已被集群拒绝。
如需重新发起转让，请创建新的审查工单。
"""
    elif status == "expired":
        body += f"""\
审查工单已超过 48 小时有效期，自动关闭。
如需继续转让，请创建新的审查工单。
"""

    body += "\n此邮件由分布式推理系统自动发送，请勿回复。"

    return _send_email(subject, body)


# ================================================================
# 邮件回复投票轮询器 (IMAP)
# ================================================================

# 已处理的邮件 Message-ID 集合（防止重复投票）
_processed_mail_ids: Set[str] = set()
_processed_mail_ids_lock = threading.Lock()
_MAX_PROCESSED_IDS = 500  # 内存上限


class MailPoller:
    """
    IMAP 邮件轮询器 — 监听管理员对审查工单的 Y/N 回复投票。

    工作流程:
      1. 通过 IMAP SSL 连接邮箱
      2. 搜索未读邮件，subject 匹配 "[审查投票]"
      3. 提取 ticket_id + 解析 Y/N
      4. 调用 ReviewManager.cast_vote() 执行投票
      5. 发送确认/错误邮件给回复者
      6. 标记邮件为已读

    线程安全: 单线程轮询（轮询在后台线程中串行执行）。
    """

    # Y/N 投票解析: 正文 trim 后第一个非空字符
    _VOTE_RE = re.compile(r'^\s*([yn])', re.IGNORECASE)

    # 从 subject 提取 ticket_id
    _TICKET_ID_RE = re.compile(r'review_[a-f0-9]{12}')

    def __init__(
        self,
        imap_server: str = IMAP_SERVER,
        imap_port: int = IMAP_PORT,
        username: str = SMTP_SENDER,
        password: str = SMTP_PASSWORD,
        admin_email: str = SMTP_RECIPIENT,
    ):
        self._imap_server = imap_server
        self._imap_port = imap_port
        self._username = username
        self._password = password
        self._admin_email = admin_email.lower()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- 公开 API ----

    def start(self, poll_interval: float = MAIL_POLL_INTERVAL) -> None:
        """在后台线程中启动轮询。

        Args:
            poll_interval: 轮询间隔（秒），默认 60。
        """
        if self._running:
            logger.warning("MailPoller 已在运行中")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(poll_interval,),
            daemon=True,
            name="mail-poller",
        )
        self._thread.start()
        logger.info(
            f"📬 MailPoller 已启动 (interval={poll_interval}s, "
            f"imap={self._imap_server}:{self._imap_port})"
        )

    def stop(self) -> None:
        """停止轮询。"""
        self._running = False
        self._stop_event.set()  # 立即唤醒睡眠中的轮询线程
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("📬 MailPoller 已停止")

    def poll_once(self) -> dict:
        """执行一次手动轮询（同步，用于测试或手动触发）。

        Returns:
            {"checked": int, "voted": int, "errors": int, "details": [...]}
        """
        return self._poll_inbox()

    # ---- 内部方法 ----

    def _poll_loop(self, interval: float) -> None:
        """后台轮询循环 — 仅在有待处理工单时连接 IMAP。
        使用 threading.Event 实现可中断的睡眠，避免关闭时延迟过长。
        """
        # 无工单时的休眠间隔（5 分钟检查一次是否有新工单）
        IDLE_SLEEP = 300.0

        while self._running:
            try:
                if self._has_pending_tickets():
                    self._poll_inbox()
                    self._stop_event.wait(timeout=interval)
                else:
                    # 无待处理工单，长休眠（可被 stop() 立即中断）
                    self._stop_event.wait(timeout=IDLE_SLEEP)
            except Exception as e:
                logger.error(f"MailPoller 轮询异常: {e}", exc_info=True)
                self._stop_event.wait(timeout=interval)

    def _has_pending_tickets(self) -> bool:
        """检查是否有待处理的审查工单（避免无意义的 IMAP 连接）。"""
        try:
            from db import list_review_tickets
            rows = list_review_tickets("pending")
            return bool(rows)
        except Exception:
            # DB 不可用时假定有工单（宁可多连一次 IMAP 也不错失投票）
            return True

    def _poll_inbox(self) -> dict:
        """连接 IMAP，查找审查回复邮件，执行投票。"""
        stats = {"checked": 0, "voted": 0, "errors": 0, "details": []}

        try:
            conn = self._connect_imap()
            if conn is None:
                return stats

            try:
                # 搜索未读邮件（UNSEEN），subject 包含审查标记
                status, msg_ids = conn.search(None, '(UNSEEN SUBJECT "[REVIEW-VOTE]")')
                if status != "OK":
                    return stats

                id_list = msg_ids[0].split() if msg_ids[0] else []
                stats["checked"] = len(id_list)

                for msg_id_bytes in id_list:
                    result = self._process_message(conn, msg_id_bytes)
                    if result:
                        stats["details"].append(result)
                        if result.get("voted"):
                            stats["voted"] += 1
                        if result.get("error"):
                            stats["errors"] += 1

            finally:
                try:
                    conn.close()
                    conn.logout()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"IMAP 轮询失败: {e}")
            stats["errors"] += 1

        return stats

    def _connect_imap(self):
        """建立 IMAP SSL 连接并登录。

        连接失败时确保关闭已创建的 socket，防止资源泄漏。
        """
        conn = None
        try:
            conn = imaplib.IMAP4_SSL(self._imap_server, self._imap_port, timeout=30)
            conn.login(self._username, self._password)
            conn.select("INBOX", readonly=False)
            return conn
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP 登录失败: {e}")
            if conn is not None:
                try:
                    conn.logout()
                except Exception:
                    try:
                        conn.shutdown()
                    except Exception:
                        pass
            return None
        except Exception as e:
            logger.error(f"IMAP 连接失败: {e}")
            if conn is not None:
                try:
                    conn.shutdown()
                except Exception:
                    pass
            return None

    def _process_message(self, conn, msg_id_bytes) -> Optional[dict]:
        """处理单封邮件：解析、投票、回复确认。

        Returns:
            dict 或 None（已处理过/跳过）。
        """
        try:
            # 获取邮件内容
            status, msg_data = conn.fetch(msg_id_bytes, "(RFC822)")
            if status != "OK":
                return None

            raw_email = msg_data[0][1]
            msg = message_from_bytes(raw_email, policy=policy.default)

            # 去重：检查 Message-ID（线程安全）
            message_id = msg.get("Message-ID", "")
            with _processed_mail_ids_lock:
                already_processed = message_id and message_id in _processed_mail_ids
            if already_processed:
                # 已处理过，标记为已读并跳过
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                return None

            # 验证发件人（只接受管理员邮箱）
            from_addr = self._extract_email(msg.get("From", ""))
            if from_addr != self._admin_email:
                logger.info(f"忽略非管理员邮件: {from_addr}")
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)
                return {"from": from_addr, "skipped": True, "reason": "非管理员邮箱"}

            # 跳过自动回复/退信
            auto_submitted = msg.get("Auto-Submitted", "")
            if auto_submitted and auto_submitted.lower() != "no":
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)
                return {"from": from_addr, "skipped": True, "reason": "自动回复邮件"}

            # 提取 ticket_id
            subject_raw = msg.get("Subject", "")
            subject = _decode_mime_header(subject_raw)
            ticket_match = self._TICKET_ID_RE.search(subject)
            if not ticket_match:
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)
                return {"from": from_addr, "skipped": True,
                        "reason": f"subject 中无 ticket_id: {subject}"}

            ticket_id = ticket_match.group(0)

            # 解析投票
            body_text = _get_email_body(msg)
            vote_value = self._parse_vote(body_text)

            if vote_value is None:
                # 无效回复 → 发送提示并标记已读
                _send_vote_error(from_addr, ticket_id)
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)
                return {"from": from_addr, "ticket_id": ticket_id,
                        "voted": False, "error": f"无效回复内容: {body_text[:80]}"}

            # 执行投票（注意：邮件投票路径是管理员专用通道，
            # 不经过 can_node_vote GPU 硬件检查——管理员通过邮箱授权即可投票）
            try:
                from review import ReviewManager
                voter_id = f"mail:{from_addr}"
                review_mgr = ReviewManager()
                ticket = review_mgr.cast_vote(
                    ticket_id=ticket_id,
                    voter_node_id=voter_id,
                    vote_value=vote_value,
                    comment=f"邮件投票: {from_addr}",
                )

                if ticket is None:
                    _send_vote_error(
                        from_addr, ticket_id,
                        "工单不存在或已关闭（可能已过期）"
                    )
                else:
                    _send_vote_confirmation(from_addr, ticket_id, vote_value)

                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)

                return {
                    "from": from_addr, "ticket_id": ticket_id,
                    "voted": True, "vote": vote_value,
                    "ticket_status": ticket.status.value if ticket else "not_found",
                }

            except Exception as e:
                logger.error(f"邮件投票执行失败: {e}", exc_info=True)
                _send_vote_error(from_addr, ticket_id, f"系统错误: {e}")
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
                if message_id:
                    _add_processed(message_id)
                return {"from": from_addr, "ticket_id": ticket_id,
                        "voted": False, "error": str(e)}

        except Exception as e:
            logger.error(f"处理邮件失败: {e}", exc_info=True)
            try:
                conn.store(msg_id_bytes, "+FLAGS", "\\Seen")
            except Exception:
                pass
            return {"error": str(e)}

    def _parse_vote(self, body_text: str) -> Optional[int]:
        """从邮件正文解析投票值。

        规则（严格）:
          - trim 后第一个非空字符 Y/y → +1
          - trim 后第一个非空字符 N/n → -1
          - 其他任何内容 → None（无效）

        Args:
            body_text: 纯文本正文。

        Returns:
            +1、-1 或 None（无效）。
        """
        if not body_text:
            return None
        # 剥离回复引用行 (以 > 开头)，然后取第一行非空内容
        lines = body_text.split('\n')
        stripped_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('>')]
        clean_text = '\n'.join(stripped_lines) if stripped_lines else body_text
        m = self._VOTE_RE.match(clean_text)
        if not m:
            return None
        char = m.group(1).lower()
        return 1 if char == 'y' else -1

    @staticmethod
    def _extract_email(from_header: str) -> str:
        """从 From header 提取纯邮箱地址。

        'Admin <admin@qq.com>' → 'admin@qq.com'
        """
        match = re.search(r'<([^>]+)>', from_header)
        if match:
            return match.group(1).strip().lower()
        return from_header.strip().lower()


# ---- 模块级 MailPoller 单例 ----

_mail_poller: Optional[MailPoller] = None


_mail_poller_lock = threading.Lock()

def get_mail_poller() -> MailPoller:
    """获取全局 MailPoller 单例（线程安全）。"""
    global _mail_poller
    if _mail_poller is None:
        with _mail_poller_lock:
            if _mail_poller is None:
                _mail_poller = MailPoller()
    return _mail_poller


def start_mail_poller(poll_interval: float = MAIL_POLL_INTERVAL) -> None:
    """启动全局邮件轮询器（后台线程）。"""
    get_mail_poller().start(poll_interval)


def stop_mail_poller() -> None:
    """停止全局邮件轮询器。"""
    if _mail_poller:
        _mail_poller.stop()


def poll_mail_once() -> dict:
    """手动触发一次轮询（用于测试）。"""
    return get_mail_poller().poll_once()


# ---- 内部辅助 ----

def _add_processed(message_id: str) -> None:
    """记录已处理的 Message-ID（去重，线程安全）。"""
    global _processed_mail_ids
    with _processed_mail_ids_lock:
        _processed_mail_ids.add(message_id)
        # 防止无限增长
        if len(_processed_mail_ids) > _MAX_PROCESSED_IDS:
            # 清理前半部分（FIFO）
            to_remove = list(_processed_mail_ids)[:len(_processed_mail_ids) // 2]
            for mid in to_remove:
                _processed_mail_ids.discard(mid)


def _decode_mime_header(header: str) -> str:
    """解码 MIME 编码的邮件头（如 =?UTF-8?B?...?=）。"""
    if not header:
        return ""
    parts = decode_header(header)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            try:
                result.append(part.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(str(part))
    return "".join(result)


def _get_email_body(msg) -> str:
    """从 email Message 对象提取纯文本正文。

    优先 text/plain，回退到 text/html 的简单标签剥离。
    """
    # 优先纯文本
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    payload = part.get_content()
                    return str(payload) if payload else ""
                except Exception:
                    pass

    # 回退：text/html → 简单标签剥离
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    payload = part.get_content()
                    text = str(payload) if payload else ""
                    # 简单剥离 HTML 标签
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text.strip()
                except Exception:
                    pass

    # 非 multipart → 直接取 payload
    try:
        payload = msg.get_content()
        return str(payload).strip() if payload else ""
    except Exception:
        return ""


def _send_vote_confirmation(to_addr: str, ticket_id: str, vote: int) -> None:
    """发送投票确认邮件。"""
    vote_label = "赞同 (+1)" if vote == 1 else "阻止 (-1)"
    subject = f"✅ 投票已记录: {ticket_id}"
    body = f"""\
分布式推理系统 — 投票确认

══════════════════════════════════════
  工单 ID:      {ticket_id}
  您的投票:     {vote_label}
  投票时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
══════════════════════════════════════

您的投票已成功记录。合计 >= +2 票通过，<= -2 票阻止。
审查结果将另行邮件通知。

此邮件由分布式推理系统自动发送。
"""
    _send_email_to(subject, body, to_addr)


def _send_vote_error(to_addr: str, ticket_id: str, extra: str = "") -> None:
    """发送投票错误提示邮件。"""
    subject = f"❌ 投票无效: {ticket_id}"
    body = f"""\
分布式推理系统 — 投票无效

工单: {ticket_id}

您的回复格式无效。请直接回复审查通知邮件，正文只写一个字母:
  Y  →  赞同 (+1)
  N  →  阻止 (-1)

其他任何内容（空行、中文、"yes"、"同意" 等）均不会被视为有效投票。
{extra if extra else ''}
如有疑问，请登录管理面板操作。

此邮件由分布式推理系统自动发送。
"""
    _send_email_to(subject, body, to_addr)


def _send_email_to(subject: str, body: str, to_addr: str) -> bool:
    """发送邮件到指定地址。"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = SMTP_SENDER
    msg["To"] = to_addr
    msg["Subject"] = subject

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.login(SMTP_SENDER, SMTP_PASSWORD)
            server.sendmail(SMTP_SENDER, [to_addr], msg.as_string())
        logger.info(f"📧 邮件已发送: {subject} → {to_addr}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败 ({to_addr}): {e}")
        return False


def _send_email(subject: str, body: str) -> bool:
    """向默认管理员发送邮件。"""
    return _send_email_to(subject, body, SMTP_RECIPIENT)
