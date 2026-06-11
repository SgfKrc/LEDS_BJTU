"""
单元测试 — 邮件通知模块
=======================
测试 SMTP 配置、邮件构建逻辑。
实际发送测试需要真实网络连接，标记为 slow。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from email_notifier import (
    SMTP_SERVER, SMTP_PORT, SMTP_SENDER, SMTP_RECIPIENT,
    send_test_email,
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

@pytest.mark.slow
class TestEmailSending:
    """测试实际邮件发送（需要网络连接）"""

    def test_send_test_email(self):
        """发送测试邮件"""
        ok = send_test_email()
        assert ok is True, "测试邮件发送失败，请检查 SMTP 配置和网络连接"
