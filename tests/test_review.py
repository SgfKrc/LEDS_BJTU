"""
单元测试 — 审查票状态机 (P3)
===========================
测试 ReviewManager: 创建工单、投票、阈值判定、过期处理、投票资格检查。

持久化测试基于主节点本地 SQLite（local_store），通过 monkeypatch 隔离到临时目录，
不依赖远端数据库。
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


# ================================================================
# 投票资格检查（纯逻辑，不需要 DB）
# ================================================================

class TestCanNodeVote:
    """测试投票资格检查（静态方法，不需要 DB）"""

    def test_pc_cuda_can_vote(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote(
            "pc", {"gpu": {"cuda_available": True}}
        )
        assert ok is True
        assert reason == "ok"

    def test_pc_multi_gpu_can_vote_with_discrete_cuda(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote(
            "pc",
            {
                "gpu": {"name": "Intel Iris Xe", "cuda_available": False, "is_integrated": True},
                "gpus": [
                    {"name": "Intel Iris Xe", "cuda_available": False, "is_integrated": True},
                    {"name": "NVIDIA GeForce RTX 4060", "cuda_available": True, "is_integrated": False},
                ],
            },
        )
        assert ok is True
        assert reason == "ok"

    def test_android_cannot_vote(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote("android", {})
        assert ok is False
        assert "PC" in reason

    def test_pc_without_cuda_cannot_vote(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote(
            "pc", {"gpu": {"cuda_available": False}}
        )
        assert ok is False
        assert "CUDA" in reason or "cuda" in reason.lower()

    def test_pc_no_device_info_cannot_vote(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote("pc", None)
        assert ok is False

    def test_empty_device_info_cannot_vote(self):
        from review import ReviewManager
        ok, reason = ReviewManager.can_node_vote("pc", {})
        assert ok is False


# ================================================================
# Ticket 序列化测试（不需要 DB）
# ================================================================

class TestTicketSerialization:
    """测试 ReviewTicket <-> dict 转换"""

    def test_to_dict_and_back(self):
        from review import ReviewTicket, TicketStatus, Vote
        now = time.time()
        ticket = ReviewTicket(
            ticket_id="review_test123456",
            created_at=now,
            created_by="master",
            target_node_id="client1",
            transfer_reason="测试转让",
            status=TicketStatus.PENDING,
            votes=[
                Vote(voter_node_id="client2", value=1, timestamp=now, comment="ok"),
                Vote(voter_node_id="client3", value=-1, timestamp=now + 1, comment="bad"),
            ],
            score=0,
            expires_at=now + 48 * 3600,
        )

        d = ticket.to_dict()
        restored = ReviewTicket.from_dict(d)

        assert restored.ticket_id == ticket.ticket_id
        assert restored.status == ticket.status
        assert restored.created_by == ticket.created_by
        assert restored.target_node_id == ticket.target_node_id
        assert restored.score == 0
        assert len(restored.votes) == 2
        assert restored.votes[0].voter_node_id == "client2"
        assert restored.votes[1].voter_node_id == "client3"

    def test_from_dict_with_json_votes_string(self):
        """测试 votes 字段为 JSON 字符串时的反序列化"""
        from review import ReviewTicket, TicketStatus
        import json

        d = {
            "ticket_id": "review_test123456",
            "status": "pending",
            "created_at": time.time(),
            "created_by": "master",
            "target_node_id": "client1",
            "transfer_reason": "",
            "votes": json.dumps([
                {"voter_node_id": "n1", "value": 1,
                 "timestamp": 1.0, "comment": ""},
            ]),
            "score": 1,
            "expires_at": time.time() + 172800,
            "resolved_at": None,
            "notification_sent": False,
        }

        ticket = ReviewTicket.from_dict(d)
        assert len(ticket.votes) == 1
        assert ticket.votes[0].voter_node_id == "n1"

    def test_from_dict_with_invalid_status(self):
        """测试无效 status 字符串 → 回退到 PENDING"""
        from review import ReviewTicket
        d = {
            "ticket_id": "review_test",
            "status": "invalid_status",
            "created_at": time.time(),
            "created_by": "m",
            "target_node_id": "c",
            "transfer_reason": "",
            "votes": [],
            "score": 0,
            "expires_at": time.time() + 3600,
            "resolved_at": None,
            "notification_sent": False,
        }
        ticket = ReviewTicket.from_dict(d)
        assert ticket.status.value == "pending"


# ================================================================
# 状态机逻辑测试（不需要 DB）
# ================================================================

class TestStateMachine:
    """测试审查票状态机逻辑"""

    def test_approve_threshold(self):
        """score >= +2 → APPROVED"""
        from review import ReviewTicket, TicketStatus, Vote
        now = time.time()
        ticket = ReviewTicket(
            ticket_id="t1", created_at=now, created_by="m",
            target_node_id="c1", transfer_reason="",
            status=TicketStatus.PENDING,
            votes=[
                Vote("a", 1, now), Vote("b", 1, now),
            ],
            score=2, expires_at=now + 3600,
        )
        from review import ReviewManager
        rm = ReviewManager()
        # 手动模拟阈值检查
        assert ticket.score >= rm.APPROVE_THRESHOLD

    def test_reject_threshold(self):
        """score <= -2 → REJECTED"""
        from review import ReviewTicket, TicketStatus, Vote
        now = time.time()
        ticket = ReviewTicket(
            ticket_id="t1", created_at=now, created_by="m",
            target_node_id="c1", transfer_reason="",
            status=TicketStatus.PENDING,
            votes=[
                Vote("a", -1, now), Vote("b", -1, now),
            ],
            score=-2, expires_at=now + 3600,
        )
        from review import ReviewManager
        rm = ReviewManager()
        assert ticket.score <= rm.REJECT_THRESHOLD

    def test_pending_below_threshold(self):
        """score in (-1, 0, +1) → stays PENDING"""
        from review import ReviewTicket, TicketStatus, Vote
        now = time.time()
        for score in (-1, 0, 1):
            ticket = ReviewTicket(
                ticket_id=f"t_{score}", created_at=now, created_by="m",
                target_node_id="c1", transfer_reason="",
                status=TicketStatus.PENDING,
                votes=[
                    Vote("a", score, now),
                ],
                score=score, expires_at=now + 3600,
            )
            from review import ReviewManager
            rm = ReviewManager()
            assert ticket.score < rm.APPROVE_THRESHOLD
            assert ticket.score > rm.REJECT_THRESHOLD

    def test_score_calculation(self):
        """测试手动评分计算"""
        from review import Vote
        votes = [
            Vote("a", 1, 0), Vote("b", 1, 0),
            Vote("c", -1, 0), Vote("d", 0, 0),
        ]
        score = sum(v.value for v in votes)
        assert score == 1


# ================================================================
# 本地 SQLite 持久化测试（M1.2 后 review 已迁主节点 local_store）
# ================================================================

@pytest.fixture
def review_store(monkeypatch, tmp_path):
    """把 local_store 指向临时目录并初始化，返回隔离的 ReviewManager。

    屏蔽邮件通知（无 SMTP 凭据时 _send_email 最长阻塞 30s）。
    """
    import local_store
    sqlite_path = tmp_path / "qlh-local-store.sqlite3"
    legacy_dir = tmp_path / "legacy-json"
    legacy_dir.mkdir()
    monkeypatch.setattr(local_store, '_store_dir', str(legacy_dir))
    monkeypatch.setattr(local_store, '_get_store_dir', lambda: str(legacy_dir))
    monkeypatch.setattr(local_store, '_legacy_store_dir', lambda: str(legacy_dir))
    monkeypatch.setattr(local_store, '_sqlite_path', lambda: str(sqlite_path))
    monkeypatch.setattr('email_notifier.send_review_created_alert', lambda **kwargs: None)
    local_store.initialize_local_store()

    from review import ReviewManager
    return ReviewManager()


class TestReviewPersistence:
    """ReviewManager 在本地 SQLite 上的真实持久化测试（不再跳过）。"""

    def test_create_and_get_ticket(self, review_store):
        """创建工单并读取"""
        ticket = review_store.create_ticket(
            created_by="master_test",
            target_node_id="client_test",
            reason="自动化测试",
            timeout_hours=1,
        )

        assert ticket is not None
        assert ticket.status.value == "pending"
        assert ticket.score == 0
        assert ticket.created_by == "master_test"
        assert ticket.target_node_id == "client_test"

        # 读取验证
        loaded = review_store.get_ticket(ticket.ticket_id)
        assert loaded is not None
        assert loaded.ticket_id == ticket.ticket_id

    def test_cast_vote_and_threshold(self, review_store):
        """投票并验证阈值触发"""
        ticket = review_store.create_ticket(
            created_by="master_test",
            target_node_id="client_test2",
            reason="阈值测试",
            timeout_hours=1,
        )
        assert ticket is not None

        tid = ticket.ticket_id

        # 投 2 票 +1 → 应 APPROVED
        t = review_store.cast_vote(tid, "node_a", 1)
        assert t is not None
        assert t.score == 1
        assert t.status.value == "pending"

        t = review_store.cast_vote(tid, "node_b", 1)
        assert t is not None
        assert t.score == 2
        assert t.status.value == "approved"

    def test_cast_vote_rejected(self, review_store):
        """投票达到 -2 → REJECTED"""
        ticket = review_store.create_ticket(
            created_by="master_test",
            target_node_id="client_test3",
            reason="拒绝测试",
            timeout_hours=1,
        )
        assert ticket is not None

        tid = ticket.ticket_id

        review_store.cast_vote(tid, "node_a", -1)
        t = review_store.cast_vote(tid, "node_b", -1)
        assert t.status.value == "rejected"
        assert t.score == -2

    def test_duplicate_vote_updates(self, review_store):
        """同一节点重复投票 → 更新旧票而非追加"""
        ticket = review_store.create_ticket(
            created_by="master_test",
            target_node_id="client_test4",
            reason="重复投票测试",
            timeout_hours=1,
        )
        assert ticket is not None

        tid = ticket.ticket_id

        review_store.cast_vote(tid, "node_x", 1)
        assert review_store.get_ticket(tid).score == 1

        # 同一节点改投 -1
        t = review_store.cast_vote(tid, "node_x", -1)
        assert t.score == -1
        assert len(t.votes) == 1  # 不是 2 票

    def test_list_tickets(self, review_store):
        """列出工单"""
        tickets = review_store.list_tickets()
        assert isinstance(tickets, list)

    def test_find_approved_ticket(self, review_store):
        """查找已批准工单"""
        ticket = review_store.find_approved_ticket("nonexistent_node_xyz")
        assert ticket is None  # 不存在的节点无批准票

    def test_invalid_vote_value(self, review_store):
        """无效投票值 → ValueError"""
        with pytest.raises(ValueError, match="无效的投票值"):
            review_store.cast_vote("t1", "n1", 2)
        with pytest.raises(ValueError, match="无效的投票值"):
            review_store.cast_vote("t1", "n1", -2)
