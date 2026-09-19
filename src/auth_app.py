"""★ TOTP / Auth App 的一步确认审核（2026-09-19，票 `AUTH-TOTP-01` 的实做）。

## 为什么纯标准库
仓内调研（`docs/分布式组网审核与可用性重构调研-2026-09-15.md:48`）明确：
「**全仓无 TOTP 实现**（`grep pyotp|hotp|base32|otpauth` 零命中），依赖清单亦无 OTP 库 —— 需要新增」。
为**不引入新依赖**（L 档 / Edge 的轻量定位），本模块只用 `hmac` / `hashlib` / `base64` /
`struct` / `time` / `secrets` 实现 **RFC 6238**。

## 设计（与调研文档一致）
* 共享密钥 + 时间步（默认 **30s**）+ **HMAC-SHA1** → **6 位码**；
* 校验 **±1 步容差**（容忍轻微时钟漂移）；
* **已用码去重**（同一时间步的码只能成功一次）防**重放**；
* **失败限速**（连续失败达阈值后在冷却期内一律拒绝）。

## 与集群入群的关系
`/api/cluster/join/grant` 的 `auth_verified` 此前是**布尔审批接缝**（fail-closed 501）。
本模块提供**真正的校验器**：主节点持有人在 Auth App 里输入一次 6 位码即视为审核通过。
**本模块不接收也不存储明文种子**之外的东西；种子由调用方（本地存储）持有。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "TOTP_ALGORITHM",
    "TOTP_DIGITS",
    "TOTP_INTERVAL_SECONDS",
    "TOTP_WINDOW_STEPS",
    "TOTP_MAX_FAILURES",
    "TOTP_LOCKOUT_SECONDS",
    "TotpError",
    "TotpReplayError",
    "TotpRateLimitedError",
    "generate_secret",
    "provisioning_uri",
    "hotp",
    "totp",
    "TotpVerifier",
]

TOTP_ALGORITHM = "SHA1"
TOTP_DIGITS = 6
TOTP_INTERVAL_SECONDS = 30
TOTP_WINDOW_STEPS = 1              # ±1 步容差
TOTP_MAX_FAILURES = 5              # 连续失败阈值
TOTP_LOCKOUT_SECONDS = 300         # 冷却期（秒）

_SECRET_BYTES = 20                 # SHA1 的推荐密钥长度（160 bit）


class TotpError(ValueError):
    """TOTP 校验失败的基类。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class TotpReplayError(TotpError):
    """同一时间步的验证码已被使用过（防重放）。"""


class TotpRateLimitedError(TotpError):
    """连续失败过多，处于冷却期。"""


def generate_secret(num_bytes: int = _SECRET_BYTES) -> str:
    """生成 base32 编码的共享密钥（无填充，便于手动录入）。"""
    return base64.b32encode(secrets.token_bytes(num_bytes)).decode("ascii").rstrip("=")


def _normalize_secret(secret: str) -> bytes:
    raw = (secret or "").strip().replace(" ", "").upper()
    if not raw:
        raise TotpError("empty_secret", "TOTP 密钥不能为空")
    # 补回 base32 填充
    padding = (-len(raw)) % 8
    try:
        return base64.b32decode(raw + "=" * padding, casefold=True)
    except Exception as exc:  # noqa: BLE001
        raise TotpError("invalid_secret", f"TOTP 密钥不是合法 base32: {exc}") from exc


def hotp(secret: str, counter: int, *, digits: int = TOTP_DIGITS) -> str:
    """RFC 4226 HOTP：`HMAC-SHA1(counter) → 动态截断 → digits 位十进制`。"""
    key = _normalize_secret(secret)
    digest = hmac.new(key, struct.pack(">Q", int(counter)), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(binary % (10 ** digits)).zfill(digits)


def totp(secret: str, *, at: Optional[float] = None,
         interval: int = TOTP_INTERVAL_SECONDS, digits: int = TOTP_DIGITS) -> str:
    """RFC 6238 TOTP（时间步默认 30s）。`at` 可注入固定时间，便于测试。"""
    now = time.time() if at is None else float(at)
    return hotp(secret, int(now // interval), digits=digits)


def provisioning_uri(secret: str, *, account: str, issuer: str = "QLH") -> str:
    """`otpauth://` URI（供 Auth App 扫码/手动录入）。"""
    from urllib.parse import quote

    label = quote(f"{issuer}:{account}", safe="")
    return (
        f"otpauth://totp/{label}"
        f"?secret={secret}&issuer={quote(issuer, safe='')}"
        f"&algorithm={TOTP_ALGORITHM}&digits={TOTP_DIGITS}&period={TOTP_INTERVAL_SECONDS}"
    )


@dataclass
class _RateState:
    failures: int = 0
    locked_until: float = 0.0


@dataclass
class TotpVerifier:
    """一次性验证码校验器（**有状态**：负责重放去重与失败限速）。

    ⚠️ 线程安全：内部一把 `RLock`。失败计数与冷却对所有账号共享一份
    （本机单用户场景足够；如需按账号细分，实例化多个即可）。
    """

    interval: int = TOTP_INTERVAL_SECONDS
    digits: int = TOTP_DIGITS
    window: int = TOTP_WINDOW_STEPS
    max_failures: int = TOTP_MAX_FAILURES
    lockout_seconds: int = TOTP_LOCKOUT_SECONDS
    _used: dict[str, int] = field(default_factory=dict)     # secret -> 最后成功的时间步
    _rate: _RateState = field(default_factory=_RateState)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ---------------------------------------------------------------- 内部
    def _check_not_locked(self, now: float) -> None:
        if self._rate.locked_until and now < self._rate.locked_until:
            remain = int(self._rate.locked_until - now) + 1
            raise TotpRateLimitedError(
                "rate_limited", f"失败次数过多，请在 {remain} 秒后重试")

    def _note_failure(self, now: float) -> None:
        self._rate.failures += 1
        if self._rate.failures >= self.max_failures:
            self._rate.locked_until = now + self.lockout_seconds
            self._rate.failures = 0

    def _note_success(self) -> None:
        self._rate.failures = 0
        self._rate.locked_until = 0.0

    # ---------------------------------------------------------------- 公开
    def verify(self, secret: str, code: str, *, at: Optional[float] = None) -> bool:
        """校验一次性验证码。

        Args:
            secret: base32 共享密钥。
            code: 用户从 Auth App 读到的 6 位码。
            at: 注入当前时间（测试用）。

        Returns:
            True 表示通过。

        Raises:
            TotpReplayError: 该码（时间步）已被成功使用过 ⇒ 防重放。
            TotpRateLimitedError: 处于失败冷却期。
            TotpError: 码格式非法（非 digits 位数字）。
        """
        now = time.time() if at is None else float(at)
        text = (code or "").strip().replace(" ", "")
        if not text.isdigit() or len(text) != self.digits:
            raise TotpError("invalid_format", f"验证码必须是 {self.digits} 位数字")

        with self._lock:
            self._check_not_locked(now)

            step = int(now // self.interval)
            for delta in range(-self.window, self.window + 1):
                candidate_step = step + delta
                if hmac.compare_digest(hotp(secret, candidate_step, digits=self.digits), text):
                    last = self._used.get(secret)
                    if last is not None and candidate_step <= last:
                        raise TotpReplayError(
                            "replayed", "该验证码已被使用，请等待下一个时间步")
                    self._used[secret] = candidate_step
                    self._note_success()
                    return True

            self._note_failure(now)
            return False

    def reset(self) -> None:
        """清空重放记录与限速状态（测试/重置身份时用）。"""
        with self._lock:
            self._used.clear()
            self._rate = _RateState()
