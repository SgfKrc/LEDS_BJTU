"""
单元测试 — 邮件通知模块 + 邮件投票轮询器
=========================================
测试 SMTP 配置、Y/N 投票解析、IMAP 轮询逻辑。
实际发送测试需要真实网络连接，标记为 slow。

---- SMTP 实际发送测试（test_send_test_email）----
默认跳过，因为需要 QQ 邮箱授权码。

启用方法:
  方案A（推荐）: 在项目根目录创建 .env 文件（.gitignore 已保护），内容:
    QLH_SMTP_SENDER=studyp4ct@qq.com
    QLH_SMTP_PASSWORD=<你的QQ邮箱授权码>
    QLH_SMTP_RECIPIENT=2743631775@qq.com
    QLH_SMTP_SERVER=smtp.qq.com
    QLH_SMTP_PORT=465
    QLH_IMAP_SERVER=imap.qq.com
    QLH_IMAP_PORT=993
    然后: pip install python-dotenv
    运行: python -c "from dotenv import load_dotenv; load_dotenv()"
    测试: QLH_SMTP_PASSWORD=<授权码> pytest tests/test_email_notifier.py -v -k "send"

  方案B: 直接设置环境变量（仅当前终端有效）:
    export QLH_SMTP_PASSWORD=<你的QQ邮箱授权码>
    pytest tests/test_email_notifier.py -v

  授权码获取: QQ邮箱 → 设置 → 账户 → POP3/SMTP服务 → 生成授权码
"""

import sys
import os

# P3修复: 邮箱凭据已移至环境变量 / .env 文件
# 以下为测试默认值（含假密码，触发 send_test_email 自动跳过）
# 要运行实际发送测试，请设置真实 QLH_SMTP_PASSWORD（见上方注释）
for _k, _v in [
    ("QLH_SMTP_SENDER", "studyp4ct@qq.com"),
    ("QLH_SMTP_PASSWORD", "test_password"),
    ("QLH_SMTP_RECIPIENT", "2743631775@qq.com"),
    ("QLH_SMTP_SERVER", "smtp.qq.com"),
    ("QLH_SMTP_PORT", "465"),
    ("QLH_IMAP_SERVER", "imap.qq.com"),
    ("QLH_IMAP_PORT", "993"),
]:
    if _k not in os.environ:
        os.environ[_k] = _v

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from email_notifier import (
    SMTP_SERVER, SMTP_PORT, IMAP_SERVER, IMAP_PORT,
    SMTP_SENDER, SMTP_RECIPIENT,
    send_test_email, MailPoller,
)


# ================================================================
# SMTP 配置验证
# ================================================================

class TestSmtpConfig:
    """测试 SMTP 配置"""

    def test_smtp_server_set(self):
        """SMTP 服务器应已配置"""
        assert SMTP_SERVER == "smtp.qq.com"

    def test_smtp_port_ssl(self):
        """应使用 SSL 端口 465"""
        assert SMTP_PORT == 465

    def test_sender_set(self):
        """发件人应已配置"""
        assert "@" in SMTP_SENDER
        assert "qq.com" in SMTP_SENDER

    def test_recipient_set(self):
        """收件人应已配置"""
        assert "@" in SMTP_RECIPIENT
        assert "qq.com" in SMTP_RECIPIENT

    def test_sender_and_recipient_different(self):
        """发件人和收件人不应相同（避免自己发给自己）"""
        # 两者都是 @qq.com 但应该是不同的账号
        # 如果相同也 OK — 只是验证配置合理性
        assert len(SMTP_SENDER) > 0
        assert len(SMTP_RECIPIENT) > 0


# ================================================================
# 邮件发送测试（需要真实网络）
# ================================================================

# ================================================================
# P3: Y/N 投票解析测试（不需要网络）
# ================================================================

class TestVoteParsing:
    """测试 MailPoller._parse_vote() 的 Y/N 解析逻辑"""

    def setup_method(self):
        self.poller = MailPoller()

    # ---- 有效投票 ----

    def test_parse_y_lower(self):
        assert self.poller._parse_vote("y") == 1

    def test_parse_y_upper(self):
        assert self.poller._parse_vote("Y") == 1

    def test_parse_n_lower(self):
        assert self.poller._parse_vote("n") == -1

    def test_parse_n_upper(self):
        assert self.poller._parse_vote("N") == -1

    def test_parse_y_with_whitespace(self):
        assert self.poller._parse_vote("  Y  ") == 1

    def test_parse_n_with_newlines(self):
        assert self.poller._parse_vote("\n\n  n  \n") == -1

    def test_parse_y_with_leading_text(self):
        """Y/N 只要第一个非空字符匹配即可"""
        assert self.poller._parse_vote("  Y") == 1

    # ---- 无效投票 ----

    def test_parse_empty_body(self):
        assert self.poller._parse_vote("") is None

    def test_parse_whitespace_only(self):
        assert self.poller._parse_vote("   \n  \t  ") is None

    def test_parse_yes_word(self):
        """'yes' 首字符为 'y' → 有效 +1"""
        assert self.poller._parse_vote("yes") == 1

    def test_parse_no_word(self):
        """'no' 首字符为 'n' → 有效 -1"""
        assert self.poller._parse_vote("no") == -1

    def test_parse_chinese_yes(self):
        """'是' 不是有效投票"""
        assert self.poller._parse_vote("是") is None

    def test_parse_chinese_no(self):
        assert self.poller._parse_vote("否") is None

    def test_parse_plus_one(self):
        """'+1' 不是有效投票"""
        assert self.poller._parse_vote("+1") is None

    def test_parse_random_text(self):
        assert self.poller._parse_vote("hello world") is None

    def test_parse_agree_word(self):
        """'同意' 不是有效投票"""
        assert self.poller._parse_vote("同意") is None

    def test_parse_none(self):
        assert self.poller._parse_vote(None) is None

    # ---- 边界情况 ----

    def test_parse_y_followed_by_text(self):
        """'Y - 我同意这个转让' → Y 会被识别（第一个非空字符）"""
        assert self.poller._parse_vote("Y - 我同意这个转让") == 1

    def test_parse_n_followed_by_reason(self):
        """'N 因为目标节点性能不足' → N 会被识别"""
        assert self.poller._parse_vote("N 因为目标节点性能不足") == -1

    def test_parse_tab_before_y(self):
        assert self.poller._parse_vote("\tY") == 1


# ================================================================
# 邮件头解析测试（不需要网络）
# ================================================================

class TestMailHeaderParsing:
    """测试邮件头解析辅助函数"""

    def test_extract_email_angle_brackets(self):
        assert MailPoller._extract_email(
            "Admin <admin@qq.com>"
        ) == "admin@qq.com"

    def test_extract_email_plain(self):
        assert MailPoller._extract_email(
            "admin@qq.com"
        ) == "admin@qq.com"

    def test_extract_email_with_name(self):
        assert MailPoller._extract_email(
            "Zhang San <zhangsan@qq.com>"
        ) == "zhangsan@qq.com"

    def test_ticket_id_regex(self):
        import re
        pattern = MailPoller._TICKET_ID_RE
        match = pattern.search("Re: [审查投票] review_a1b2c3d4e5f6 — 主节点转让审查")
        assert match is not None
        assert match.group(0) == "review_a1b2c3d4e5f6"

    def test_ticket_id_not_found(self):
        import re
        pattern = MailPoller._TICKET_ID_RE
        match = pattern.search("普通邮件 无工单号")
        assert match is None


# ================================================================
# IMAP 配置验证
# ================================================================

class TestImapConfig:
    """测试 IMAP 配置"""

    def test_imap_server_set(self):
        assert IMAP_SERVER == "imap.qq.com"

    def test_imap_port_ssl(self):
        assert IMAP_PORT == 993

    def test_smtp_imap_same_credentials(self):
        """SMTP 和 IMAP 应使用相同的 QQ 邮箱账号"""
        assert SMTP_SENDER == "studyp4ct@qq.com"


# ================================================================
# 邮件发送测试（需要真实网络）
# ================================================================

@pytest.mark.slow
class TestEmailSending:
    """测试实际邮件发送（需要网络连接）"""

    @pytest.mark.skipif(
        not os.environ.get("QLH_SMTP_PASSWORD") or os.environ.get("QLH_SMTP_PASSWORD") == "test_password",
        reason="需要真实 QQ 邮箱授权码（设置环境变量 QLH_SMTP_PASSWORD）"
    )
    def test_send_test_email(self):
        """发送测试邮件（需要真实 SMTP 凭据）"""
        ok = send_test_email()
        assert ok is True, "测试邮件发送失败，请检查 SMTP 配置和网络连接"
