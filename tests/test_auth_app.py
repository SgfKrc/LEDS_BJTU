"""★ `auth_app.py` 测试（2026-09-19，票 AUTH-TOTP-01）。

判据优先级：
1. **RFC 6238 附录 B 官方测试向量**（SHA1）——用 8 位码验证，再取后 6 位核对 6 位实现；
2. **RFC 4226 附录 D 的 HOTP 向量**（同一密钥）；
3. 重放去重、失败限速、格式校验、密钥规范化。
"""

from __future__ import annotations

import base64

import pytest

import auth_app
from auth_app import (
    TotpError,
    TotpRateLimitedError,
    TotpReplayError,
    TotpVerifier,
    generate_secret,
    hotp,
    provisioning_uri,
    totp,
)

# RFC 6238 附录 B：ASCII "12345678901234567890" 的 base32 编码
RFC_SECRET = base64.b32encode(b"12345678901234567890").decode("ascii").rstrip("=")


class TestRfc6238Vectors:
    """RFC 6238 附录 B（SHA1 列）。8 位码；6 位实现取其后 6 位。"""

    CASES = [
        (59, "94287082"),
        (1111111109, "07081804"),
        (1111111111, "14050471"),
        (1234567890, "89005924"),
        (2000000000, "69279037"),
        (20000000000, "65353130"),
    ]

    @pytest.mark.parametrize("t,expected8", CASES)
    def test_eight_digit_matches_rfc(self, t, expected8):
        assert hotp(RFC_SECRET, t // 30, digits=8) == expected8

    @pytest.mark.parametrize("t,expected8", CASES)
    def test_six_digit_is_last_six(self, t, expected8):
        assert totp(RFC_SECRET, at=t) == expected8[-6:]


class TestRfc4226HotpVectors:
    """RFC 4226 附录 D：同一密钥，counter 0..9 的 6 位 HOTP。"""

    EXPECTED = [
        "755224", "287082", "359152", "969429", "338314",
        "254676", "287922", "162583", "399871", "520489",
    ]

    def test_counters_0_to_9(self):
        got = [hotp(RFC_SECRET, c) for c in range(10)]
        assert got == self.EXPECTED


class TestSecretHandling:
    def test_generate_secret_is_base32_and_unique(self):
        s1, s2 = generate_secret(), generate_secret()
        assert s1 != s2
        assert isinstance(s1, str) and len(s1) >= 16
        # 能解码回字节（去掉填充后仍需合法）
        base64.b32decode(s1 + "=" * ((-len(s1)) % 8), casefold=True)

    def test_empty_secret_rejected(self):
        with pytest.raises(TotpError) as ei:
            hotp("", 0)
        assert ei.value.code == "empty_secret"

    def test_invalid_base32_rejected(self):
        with pytest.raises(TotpError) as ei:
            hotp("!!!not-base32!!!", 0)
        assert ei.value.code == "invalid_secret"

    def test_secret_is_normalized(self):
        """小写 + 空格应被规范化（便于手动录入）。"""
        spaced = " ".join(RFC_SECRET[i:i + 4] for i in range(0, len(RFC_SECRET), 4)).lower()
        assert hotp(spaced, 1) == hotp(RFC_SECRET, 1)


class TestProvisioningUri:
    def test_uri_shape(self):
        uri = provisioning_uri(RFC_SECRET, account="koakuma", issuer="QLH")
        assert uri.startswith("otpauth://totp/")
        assert f"secret={RFC_SECRET}" in uri
        assert "digits=6" in uri and "period=30" in uri and "algorithm=SHA1" in uri


class TestVerifier:
    def test_accepts_current_code(self):
        v = TotpVerifier()
        assert v.verify(RFC_SECRET, totp(RFC_SECRET, at=1000.0), at=1000.0) is True

    def test_honours_one_step_window(self):
        v = TotpVerifier()
        prev = totp(RFC_SECRET, at=1000.0 - 30)
        assert v.verify(RFC_SECRET, prev, at=1000.0) is True

    def test_rejects_beyond_window(self):
        v = TotpVerifier()
        old = totp(RFC_SECRET, at=1000.0 - 300)
        assert v.verify(RFC_SECRET, old, at=1000.0) is False

    def test_replay_of_same_step_rejected(self):
        v = TotpVerifier()
        code = totp(RFC_SECRET, at=5000.0)
        assert v.verify(RFC_SECRET, code, at=5000.0) is True
        with pytest.raises(TotpReplayError) as ei:
            v.verify(RFC_SECRET, code, at=5000.0)
        assert ei.value.code == "replayed"

    def test_bad_format_raises(self):
        v = TotpVerifier()
        for bad in ("abc", "12345", "1234567", ""):
            with pytest.raises(TotpError) as ei:
                v.verify(RFC_SECRET, bad, at=1000.0)
            assert ei.value.code == "invalid_format"

    def test_rate_limit_after_failures(self):
        v = TotpVerifier(max_failures=3, lockout_seconds=60)
        for _ in range(3):
            assert v.verify(RFC_SECRET, "000000", at=1000.0) is False
        # 达到阈值 ⇒ 冷却期内一律拒绝（即使是正确码）
        good = totp(RFC_SECRET, at=1000.0)
        with pytest.raises(TotpRateLimitedError) as ei:
            v.verify(RFC_SECRET, good, at=1000.0)
        assert ei.value.code == "rate_limited"
        # 冷却期结束后恢复（⚠️ 必须用**该时刻**的码：旧码已超出 ±1 时间步窗口）
        later = 1000.0 + 61
        assert v.verify(RFC_SECRET, totp(RFC_SECRET, at=later), at=later) is True

    def test_success_resets_failure_counter(self):
        v = TotpVerifier(max_failures=3, lockout_seconds=60)
        assert v.verify(RFC_SECRET, "000000", at=1000.0) is False
        assert v.verify(RFC_SECRET, totp(RFC_SECRET, at=1000.0), at=1000.0) is True
        # 计数已清零 ⇒ 再失败两次不会触发冷却
        assert v.verify(RFC_SECRET, "000000", at=2000.0) is False
        assert v.verify(RFC_SECRET, "000000", at=2000.0) is False
        assert v.verify(RFC_SECRET, totp(RFC_SECRET, at=2000.0), at=2000.0) is True

    def test_reset_clears_state(self):
        v = TotpVerifier()
        code = totp(RFC_SECRET, at=9000.0)
        assert v.verify(RFC_SECRET, code, at=9000.0) is True
        v.reset()
        # reset 后同一时间步的码可再次通过（重放记录已清空）
        assert v.verify(RFC_SECRET, code, at=9000.0) is True


class TestModuleContract:
    def test_constants_match_doc(self):
        """与调研文档写定的参数一致（30s / 6 位 / ±1 步 / 限速）。"""
        assert auth_app.TOTP_INTERVAL_SECONDS == 30
        assert auth_app.TOTP_DIGITS == 6
        assert auth_app.TOTP_WINDOW_STEPS == 1
        assert auth_app.TOTP_MAX_FAILURES > 0 and auth_app.TOTP_LOCKOUT_SECONDS > 0

    def test_no_external_dependency(self):
        """纯标准库：模块命名空间里不应出现 pyotp / otpauth 之类第三方包。"""
        import sys

        # ⚠️ 只检查 **import 语句** —— 模块 docstring 里引用了调研原文（含 "pyotp" 一词），
        #    全文匹配会误报。
        import re as _re

        src = open(auth_app.__file__, encoding="utf-8").read()
        imported = _re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", src, _re.M)
        top = {name.split(".")[0] for name in imported}
        allowed = {"base64", "hashlib", "hmac", "secrets", "struct", "threading",
                   "time", "dataclasses", "typing", "urllib", "__future__"}
        assert top <= allowed, f"auth_app 应只用标准库，实得: {sorted(top - allowed)}"
        assert "pyotp" not in sys.modules
