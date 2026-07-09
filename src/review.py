"""
审查票状态机 — Gerrit 风格主节点转让审查投票 (P3)
=====================================================

功能:
  - ReviewTicket / Vote 数据结构
  - ReviewManager: 创建工单、投票、过期处理、阈值判定
  - 投票资格检查: 仅 PC 独显 (node_type="pc" + cuda_available=True) 可投票

门控:
  - 仅 CUDA PC 节点可投票 — CPU/集显/Android 节点被拒绝
  - 通过阈值: score >= +2  → APPROVED
  - 阻止阈值: score <= -2  → REJECTED
  - 超时: 48h 默认 → EXPIRED

持久化:
  - 通过 db.py 的 review_tickets 表存储
  - 依赖 db.py 的 create_review_ticket / get_review_ticket / update_review_ticket / list_review_tickets

不依赖 scheduler.py（避免循环引用）。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ================================================================
# 枚举与数据结构
# ================================================================

class TicketStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Vote:
    """单票记录。"""
    voter_node_id: str
    value: int            # -1, 0, +1
    timestamp: float
    comment: str = ""


@dataclass
class ReviewTicket:
    """审查工单的完整状态。"""
    ticket_id: str
    created_at: float
    created_by: str               # 发起者 node_id
    target_node_id: str           # 拟转让目标
    transfer_reason: str          # 转让原因
    status: TicketStatus          # pending → approved/rejected/expired
    votes: list = field(default_factory=list)   # list[Vote]
    score: int = 0
    expires_at: float = 0.0
    resolved_at: Optional[float] = None
    notification_sent: bool = False

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "target_node_id": self.target_node_id,
            "transfer_reason": self.transfer_reason,
            "votes": [
                {
                    "voter_node_id": v.voter_node_id,
                    "value": v.value,
                    "timestamp": v.timestamp,
                    "comment": v.comment,
                }
                for v in self.votes
            ],
            "score": self.score,
            "expires_at": self.expires_at,
            "resolved_at": self.resolved_at,
            "notification_sent": self.notification_sent,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewTicket":
        raw_votes = d.get("votes", [])
        if isinstance(raw_votes, str):
            try:
                raw_votes = json.loads(raw_votes)
            except (json.JSONDecodeError, TypeError):
                raw_votes = []

        votes = []
        for v in raw_votes:
            votes.append(Vote(
                voter_node_id=v.get("voter_node_id", ""),
                value=v.get("value", 0),
                timestamp=v.get("timestamp", 0.0),
                comment=v.get("comment", ""),
            ))

        status_str = d.get("status", "pending")
        try:
            status = TicketStatus(status_str)
        except ValueError:
            status = TicketStatus.PENDING

        return cls(
            ticket_id=d.get("ticket_id", ""),
            created_at=d.get("created_at", 0.0),
            created_by=d.get("created_by", ""),
            target_node_id=d.get("target_node_id", ""),
            transfer_reason=d.get("transfer_reason", ""),
            status=status,
            votes=votes,
            score=d.get("score", 0),
            expires_at=d.get("expires_at", 0.0),
            resolved_at=d.get("resolved_at"),
            notification_sent=d.get("notification_sent", False),
        )


# ================================================================
# ReviewManager
# ================================================================

class ReviewManager:
    """审查票状态机 + 持久化。

    线程安全：所有 DB 操作通过 db.py 的连接池，psycopg2 自带线程安全。
    """

    APPROVE_THRESHOLD: int = 2     # score >= +2 → APPROVED
    REJECT_THRESHOLD: int = -2    # score <= -2 → REJECTED
    DEFAULT_TIMEOUT_HOURS: float = 48.0

    # ---- 创建工单 ----

    def create_ticket(
        self,
        created_by: str,
        target_node_id: str,
        reason: str = "",
        timeout_hours: float = None,
    ) -> Optional[ReviewTicket]:
        """创建一个新的待审查工单。

        Args:
            created_by: 发起者 node_id（通常是 master 的 node_id）。
            target_node_id: 拟转让的目标从节点 ID。
            reason: 转让原因说明。
            timeout_hours: 超时时间（小时），默认 48。

        Returns:
            ReviewTicket 或 None（DB 不可用时）。
        """
        timeout = timeout_hours or self.DEFAULT_TIMEOUT_HOURS
        now = time.time()

        ticket = ReviewTicket(
            ticket_id=f"review_{uuid.uuid4().hex[:12]}",
            created_at=now,
            created_by=created_by,
            target_node_id=target_node_id,
            transfer_reason=reason,
            status=TicketStatus.PENDING,
            votes=[],
            score=0,
            expires_at=now + timeout * 3600,
        )

        if not self._persist_ticket(ticket):
            return None

        logger.info(
            f"审查工单已创建: {ticket.ticket_id} "
            f"({created_by} -> {target_node_id}, "
            f"score={ticket.score}, "
            f"expires_in={timeout}h)"
        )

        # 触发邮件通知（延迟导入，避免循环）
        try:
            from email_notifier import send_review_created_alert
            send_review_created_alert(
                ticket_id=ticket.ticket_id,
                created_by=created_by,
                target_node_id=target_node_id,
                reason=reason,
                expires_at=ticket.expires_at,
            )
            ticket.notification_sent = True
            self._persist_ticket(ticket)
        except Exception as e:
            logger.warning(f"审查邮件通知发送失败: {e}")

        return ticket

    # ---- 投票 ----

    def cast_vote(
        self,
        ticket_id: str,
        voter_node_id: str,
        vote_value: int,
        comment: str = "",
    ) -> Optional[ReviewTicket]:
        """为一个工单投票。

        Args:
            ticket_id: 工单 ID。
            voter_node_id: 投票者 node_id。
            vote_value: -1（阻止）、0（弃权）、+1（赞同）。
            comment: 投票附言。

        Returns:
            更新后的 ReviewTicket，或 None（工单不存在/DB 不可用）。

        Raises:
            ValueError: vote_value 不在 [-1, 0, 1] 范围内。
        """
        if vote_value not in (-1, 0, 1):
            raise ValueError(f"无效的投票值: {vote_value}，需为 -1, 0 或 +1")

        # P3修复: 重试机制防止并发投票竞态（读-改-写冲突）
        max_retries = 3
        for attempt in range(max_retries):
            ticket = self.get_ticket(ticket_id)
            if ticket is None:
                logger.warning(f"工单不存在: {ticket_id}")
                return None

            if ticket.status != TicketStatus.PENDING:
                logger.warning(
                    f"工单 {ticket_id} 状态为 {ticket.status.value}，不接受投票"
                )
                return None

            # 检查自投票（P3修复: 创建者和目标节点不可投票）
            if voter_node_id == ticket.created_by:
                logger.warning(f"创建者不可投票: {voter_node_id}")
                return None
            if voter_node_id == ticket.target_node_id:
                logger.warning(f"目标节点不可自我批准: {voter_node_id}")
                return None

            # 更新已有投票 或 追加新投票
            now = time.time()
            existing = False
            for v in ticket.votes:
                if v.voter_node_id == voter_node_id:
                    v.value = vote_value
                    v.timestamp = now
                    v.comment = comment
                    existing = True
                    logger.info(
                        f"投票已更新: {ticket_id} <- {voter_node_id}: {vote_value:+d}"
                    )
                    break

            if not existing:
                ticket.votes.append(Vote(
                    voter_node_id=voter_node_id,
                    value=vote_value,
                    timestamp=now,
                    comment=comment,
                ))
                logger.info(
                    f"投票已记录: {ticket_id} <- {voter_node_id}: {vote_value:+d}"
                )

            # 重算得分并检查阈值
            ticket.score = sum(v.value for v in ticket.votes)

            if ticket.score >= self.APPROVE_THRESHOLD:
                ticket.status = TicketStatus.APPROVED
                ticket.resolved_at = now
                logger.info(
                    f"审查通过: {ticket_id} score={ticket.score} >= {self.APPROVE_THRESHOLD}"
                )
                self._send_resolved_alert(ticket)
            elif ticket.score <= self.REJECT_THRESHOLD:
                ticket.status = TicketStatus.REJECTED
                ticket.resolved_at = now
                logger.info(
                    f"审查被阻止: {ticket_id} score={ticket.score} <= {self.REJECT_THRESHOLD}"
                )
                self._send_resolved_alert(ticket)

            if self._persist_ticket(ticket):
                return ticket

            # 持久化失败（可能被并发更新覆盖），重试
            if attempt < max_retries - 1:
                logger.warning(
                    f"投票持久化冲突，重试 ({attempt + 1}/{max_retries}): {ticket_id}"
                )
                time.sleep(0.1 * (attempt + 1))  # 递增退避

        logger.error(f"投票持久化失败（已达最大重试次数）: {ticket_id}")
        return None

    # ---- 查询 ----

    def get_ticket(self, ticket_id: str) -> Optional[ReviewTicket]:
        """获取单个工单。"""
        try:
            from db import get_review_ticket
            row = get_review_ticket(ticket_id)
            if row is None:
                return None
            return ReviewTicket.from_dict(row)
        except Exception as e:
            logger.warning(f"读取工单失败 ({ticket_id}): {e}")
            return None

    def list_tickets(self, status: Optional[str] = None) -> list[ReviewTicket]:
        """列出工单，可按状态过滤。

        Args:
            status: None（全部）、"pending"、"approved"、"rejected"、"expired"。

        Returns:
            ReviewTicket 列表，按 created_at 降序。
        """
        try:
            from db import list_review_tickets
            rows = list_review_tickets(status)
            tickets = []
            for row in (rows or []):
                try:
                    tickets.append(ReviewTicket.from_dict(row))
                except Exception:
                    continue
            tickets.sort(key=lambda t: t.created_at, reverse=True)
            return tickets
        except Exception as e:
            logger.warning(f"列出工单失败: {e}")
            return []

    def find_approved_ticket(self, target_node_id: str) -> Optional[ReviewTicket]:
        """查找针对某个目标节点的已批准（且未过期）工单。

        用于 transfer_master_role() 的审查门控检查。
        批准后的有效期与工单原始超时时间一致。
        """
        try:
            from db import list_review_tickets
            rows = list_review_tickets("approved")
            for row in (rows or []):
                if row.get("target_node_id") == target_node_id:
                    ticket = ReviewTicket.from_dict(row)
                    if not ticket.resolved_at:
                        continue
                    # 使用工单自身的超时窗口（从 expires_at - created_at 计算）
                    validity_window = ticket.expires_at - ticket.created_at
                    if validity_window <= 0:
                        validity_window = self.DEFAULT_TIMEOUT_HOURS * 3600
                    if (time.time() - ticket.resolved_at) < validity_window:
                        return ticket
            return None
        except Exception as e:
            logger.warning(f"查找已批准工单失败: {e}")
            return None

    # ---- 删除 ----

    def delete_ticket(self, ticket_id: str) -> bool:
        """删除单个工单（所有状态均可）。"""
        try:
            from db import delete_review_ticket
            result = delete_review_ticket(ticket_id)
            if result:
                logger.info(f"审查工单已删除: {ticket_id}")
            return result
        except Exception as e:
            logger.warning(f"删除工单失败 ({ticket_id}): {e}")
            return False

    def delete_resolved(self) -> int:
        """删除所有已解决（approved/rejected/expired）的工单。"""
        try:
            from db import delete_resolved_review_tickets
            count = delete_resolved_review_tickets()
            if count > 0:
                logger.info(f"已清理 {count} 个已解决审查工单")
            return count
        except Exception as e:
            logger.warning(f"批量清理工单失败: {e}")
            return 0

    # ---- 过期处理 ----

    def resolve_expired(self) -> list[str]:
        """查找并标记所有过期的 PENDING 工单为 EXPIRED。

        Returns:
            新过期的工单 ID 列表。
        """
        now = time.time()
        expired_ids = []

        try:
            from db import list_review_tickets, update_review_ticket
            rows = list_review_tickets("pending")
            for row in (rows or []):
                expires_at = row.get("expires_at", 0)
                if expires_at > 0 and now > expires_at:
                    tid = row["ticket_id"]
                    update_review_ticket(tid, {
                        "status": "expired",
                        "resolved_at": now,
                    })
                    expired_ids.append(tid)
                    logger.info(f"审查工单已过期: {tid}")
                    self._send_resolved_alert(ReviewTicket.from_dict({
                        **row, "status": "expired", "resolved_at": now,
                    }))
        except Exception as e:
            logger.warning(f"过期检查失败: {e}")

        return expired_ids

    # ---- 投票资格检查（静态方法，不依赖 scheduler） ----

    @staticmethod
    def can_node_vote(node_type: str, device_info: dict = None) -> tuple[bool, str]:
        """检查节点是否有投票资格。

        规则:
          - node_type 必须为 "pc"
          - device_info.gpu.cuda_available 必须为 True
          - Android / CPU / 集显节点不可投票

        Args:
            node_type: "pc" | "android"
            device_info: 节点注册时上报的 device_info dict

        Returns:
            (can_vote: bool, reason: str)
        """
        if node_type != "pc":
            return False, "仅 PC 节点可参与审查投票"

        if device_info is None:
            return False, "缺少设备信息，无法验证 GPU 类型"

        gpu = device_info.get("gpu", {})
        if not isinstance(gpu, dict):
            gpu = {}

        cuda_available = gpu.get("cuda_available", False)
        if not cuda_available:
            return False, "仅 NVIDIA CUDA 独显节点可参与审查投票"

        return True, "ok"

    # ---- 内部方法 ----

    def _persist_ticket(self, ticket: ReviewTicket) -> bool:
        """将工单写入 DB（先更新，后创建）。

        Returns:
            True 如果持久化成功（包括 UPDATE 和 INSERT）。
            False 如果 DB 不可用或写入失败。
        """
        try:
            from db import update_review_ticket, create_review_ticket
            data = ticket.to_dict()
            data["votes"] = json.dumps(data["votes"], ensure_ascii=False)
            # 先尝试 update（投票后工单已存在）
            result = update_review_ticket(ticket.ticket_id, data)
            if result is not None:
                return True
            # ticket 不存在 → create（使用 ON CONFLICT DO UPDATE 确保不丢数据）
            created = create_review_ticket(data)
            return created is not None
        except Exception as e:
            logger.error(f"持久化工单失败 ({ticket.ticket_id}): {e}")
            return False

    def _send_resolved_alert(self, ticket: ReviewTicket) -> None:
        """工单已解决时发送邮件通知。"""
        try:
            from email_notifier import send_review_resolved_alert
            send_review_resolved_alert(
                ticket_id=ticket.ticket_id,
                status=ticket.status.value,
                score=ticket.score,
                target_node_id=ticket.target_node_id,
            )
        except Exception as e:
            logger.warning(f"审查结果通知邮件发送失败: {e}")
