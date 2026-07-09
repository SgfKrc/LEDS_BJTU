"""
网络通信模块 — TCP 主从通信、长连接、粘包处理、张量序列化
==========================================================
功能职责:
1. 主节点（服务端）监听、从节点（客户端）连接
2. 长连接维持、心跳检测、断线重连
3. 封包/解包（长度头 + 数据体）
4. 张量序列化 / 反序列化
5. 控制指令、特征张量收发

协议格式:
  数据包: [4字节长度头(大端序)] + [数据体(JSON字符串 或 二进制张量)]
  控制数据: JSON 字符串（指令、心跳、状态）
  推理数据: torch.save 序列化的二进制张量

依赖: socket, threading, json, torch
"""

import json
import logging
import socket
import struct
import threading
import time
import hashlib
import hmac
import io
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Callable

import torch

from config import (
    SERVER_IP, SERVER_PORT, HEARTBEAT_INTERVAL,
    RECONNECT_MAX_RETRIES, RECONNECT_DELAY,
    CLUSTER_SECRET, AUTH_TIMESTAMP_WINDOW,
)

# 尝试导入 psutil 用于网络类型检测
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class _RegistrationRejected(Exception):
    """客户端注册被拒绝，调用方应关闭连接且不分发给上层。"""


# ================================================================
# 数据包格式常量
# ================================================================
HEADER_LEN = 4          # 长度头字节数（大端序 uint32）
MAX_PACKET_SIZE = 256 * 1024 * 1024  # 最大包大小 256MB（特征张量可能较大）


# ================================================================
# HMAC 集群认证（阶段 7 — 跨节点安全）
# ================================================================

def build_auth_signature(node_id: str, timestamp: float = None) -> dict:
    """
    为注册消息构建 HMAC 认证签名。

    签名算法: HMAC-SHA256(cluster_secret, "{node_id}:{timestamp}")
    防重放: timestamp 用于服务端时间窗口校验

    Returns:
        {"auth_timestamp": float, "auth_signature": str}
    """
    ts = timestamp or time.time()
    message = f"{node_id}:{ts:.6f}".encode("utf-8")
    sig = hmac.new(
        CLUSTER_SECRET.encode("utf-8"), message, hashlib.sha256
    ).hexdigest()
    return {"auth_timestamp": ts, "auth_signature": sig}


def verify_auth_signature(node_id: str, auth_data: dict) -> tuple:
    """
    验证 HMAC 认证签名。

    检查:
    1. 时间戳在允许窗口内（防重放攻击）
    2. HMAC-SHA256 签名匹配

    Args:
        node_id: 注册节点 ID
        auth_data: {"auth_timestamp": float, "auth_signature": str}

    Returns:
        (ok: bool, reason: str)
    """
    if not auth_data:
        return False, "缺少认证签名（auth_timestamp/auth_signature）"

    ts = auth_data.get("auth_timestamp")
    sig = auth_data.get("auth_signature", "")

    if ts is None:
        return False, "缺少时间戳（auth_timestamp）"
    if not sig:
        return False, "缺少签名（auth_signature）"

    # 1. 时间窗口校验（±5 分钟）
    now = time.time()
    drift = abs(now - ts)
    if drift > AUTH_TIMESTAMP_WINDOW:
        return False, (
            f"时间戳偏差过大（{drift:.0f}s > {AUTH_TIMESTAMP_WINDOW}s），"
            f"请检查系统时钟同步"
        )

    # 2. HMAC 签名校验
    message = f"{node_id}:{ts:.6f}".encode("utf-8")
    expected = hmac.new(
        CLUSTER_SECRET.encode("utf-8"), message, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, sig):
        return False, "HMAC 签名不匹配 — 集群密钥不一致"

    return True, "ok"


class MessageType(str, Enum):
    """消息类型枚举"""
    # 控制指令
    REGISTER = "register"           # 从节点注册
    HEARTBEAT = "heartbeat"         # 心跳
    HEARTBEAT_ACK = "heartbeat_ack" # 心跳应答
    TASK_START = "task_start"       # 推理任务开始
    TASK_STOP = "task_stop"         # 推理任务停止
    TASK_DONE = "task_done"         # 推理任务完成
    ERROR = "error"                 # 错误上报
    # 推理数据
    TENSOR = "tensor"               # 中间特征张量
    RESULT = "result"               # 最终推理结果（文本）
    # 状态
    STATUS_REQ = "status_req"       # 状态查询
    STATUS_RES = "status_res"       # 状态应答
    # 分布式推理调度
    INFER_FORWARD = "infer_forward"  # 从节点 → 主节点：转发推理请求
    INFER_RESULT = "infer_result"    # 主节点 → 从节点：推理结果回传
    LAYER_CONFIG = "layer_config"    # 主节点 → 从节点：推送分层配置
    # 角色转让
    ROLE_TRANSFER = "role_transfer"          # 主节点 → 从节点：转让主节点身份
    ROLE_TRANSFER_ACK = "role_transfer_ack"  # 从节点 → 主节点：确认接收转让
    # 备用主节点
    SPARE_MASTER_DESIGNATE = "spare_master_designate"          # 主节点 → 从节点：指定为备用主节点
    SPARE_MASTER_DESIGNATE_ACK = "spare_master_designate_ack"  # 从节点 → 主节点：确认接收备用身份
    SPARE_MASTER_ACTIVATE = "spare_master_activate"            # 主节点 → 备用：激活暂代主节点职责
    SPARE_MASTER_ACTIVATE_ACK = "spare_master_activate_ack"    # 备用 → 主节点：确认激活
    SPARE_MASTER_DEACTIVATE = "spare_master_deactivate"        # 新主节点 → 备用：接管完成，退出暂代
    # 节点列表同步
    NODE_LIST_SYNC = "node_list_sync"            # 主节点 → 从节点：全量节点列表同步
    NODE_UPDATE = "node_update"                  # 主节点 → 从节点：单节点变更 (add/update/remove)
    # 分布式流水线推理
    LAYER_FORWARD = "layer_forward"              # 主→从：执行层前向传播
    LAYER_RESULT = "layer_result"                # 从→主：层前向传播结果
    CHAIN_FORWARD = "chain_forward"              # 从→从：链式直连层前向转发（P2 优化）
    PIPELINE_DONE = "pipeline_done"              # 主→从：流水线任务完成（清理 KV 缓存）
    PIPELINE_ABORT = "pipeline_abort"            # 主→从：取消流水线任务
    PIPELINE_PAUSE = "pipeline_pause"            # 主→从：暂停流水线（二期协同抢占，协议预留）
    PIPELINE_RESUME = "pipeline_resume"          # 主→从：恢复流水线（二期协同抢占，协议预留）


# ================================================================
# 网络类型检测
# ================================================================

def detect_network_type() -> str:
    """
    检测当前节点的主要网络连接类型。

    通过枚举活跃网络接口名称判断是 WiFi 还是以太网。
    同时也会检测实际的网络连接情况作为辅助判断。

    Returns:
        "wifi" | "ethernet" | "unknown"
    """
    detected = "unknown"

    if _HAS_PSUTIL:
        try:
            stats = psutil.net_if_stats()
            addrs = psutil.net_if_addrs()

            wifi_keywords = ['wi-fi', 'wlan', '无线', 'wifi', 'wireless']
            eth_keywords = ['eth', '以太', 'ethernet', 'en0', 'en']

            has_wifi = False
            has_eth = False

            for iface, stat in stats.items():
                iface_lower = iface.lower()
                if stat.isup:
                    # 检查接口名称
                    if any(kw in iface_lower for kw in wifi_keywords):
                        has_wifi = True
                    elif any(kw in iface_lower for kw in eth_keywords):
                        has_eth = True
                    # macOS: en0/en1 通常是以太网/WiFi
                    if iface_lower.startswith('en0'):
                        has_eth = True
                    elif iface_lower.startswith('en1'):
                        has_wifi = True

            # 尝试通过连接的默认路由接口判断
            if has_wifi and not has_eth:
                detected = "wifi"
            elif has_eth and not has_wifi:
                detected = "ethernet"
            elif has_wifi and has_eth:
                # 两者都有，尝试判断哪个是主连接
                try:
                    # 通过连接外部地址来判断使用的接口
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    # 查找该 IP 对应的接口
                    for iface, addr_list in addrs.items():
                        for addr in addr_list:
                            if addr.address == local_ip:
                                iface_lower = iface.lower()
                                if any(kw in iface_lower for kw in wifi_keywords):
                                    detected = "wifi"
                                    break
                                elif any(kw in iface_lower for kw in eth_keywords):
                                    detected = "ethernet"
                                    break
                        if detected != "unknown":
                            break
                except Exception:
                    logger.debug("默认路由接口检测失败，保持 unknown", exc_info=True)
            else:
                # 无法根据名称判断，通过实际接口类型判断
                for iface, stat in stats.items():
                    if stat.isup:
                        # en0 通常是以太网，en1 通常是WiFi (macOS)
                        # wlan 通常是WiFi (Linux), eth 通常是以太网
                        pass  # 已在上面处理

        except Exception:
            logger.debug("psutil 网络类型检测失败，回退到基本判断", exc_info=True)

    # 如果 psutil 不可用或检测失败，尝试通过系统命令检测
    if detected == "unknown":
        try:
            import subprocess
            import platform
            if platform.system() == "Windows":
                result = subprocess.run(
                    ['netsh', 'interface', 'show', 'interface'],
                    capture_output=True, text=True, timeout=5
                )
                output = result.stdout.lower()
                if 'wi-fi' in output or 'wlan' in output:
                    detected = "wifi"
                if 'ethernet' in output or '以太网' in output:
                    detected = "ethernet"  # 以太网优先（更稳定）
            elif platform.system() == "Linux":
                # 通过 /sys/class/net 检测接口类型
                try:
                    net_dir = "/sys/class/net"
                    if os.path.isdir(net_dir):
                        for iface in os.listdir(net_dir):
                            iface_lower = iface.lower()
                            if iface_lower.startswith("wl"):
                                detected = "wifi"
                                break
                            elif (iface_lower.startswith("en")
                                  or iface_lower.startswith("eth")):
                                detected = "ethernet"
                                break
                except OSError:
                    pass
        except Exception:
            logger.debug("系统网络类型检测失败", exc_info=True)

    logger.debug(f"检测到网络类型: {detected}")
    return detected


def detect_lan_ip() -> str:
    """
    检测本机最优可达 IP 地址（供其他节点连接使用）。

    优先级（从高到低）:
    1. Tailscale / ZeroTier 虚拟组网 IP（100.x.y.z）— 最可靠，跨子网直连
    2. psutil 枚举物理网卡（以太网 > WiFi）— 同局域网直连
    3. UDP 默认路由 IP — 出网网卡 IP
    4. socket.gethostbyname(hostname) — 最后兜底

    Returns:
        IP 地址字符串，检测失败返回 "127.0.0.1"
    """

    # ---- 策略 1: 优先检测 Tailscale / ZeroTier 虚拟组网 IP ----
    #   Tailscale 使用 100.64.0.0/10 (CGNAT) 地址段，接口名含 "tailscale"
    #   ZeroTier 接口名含 "zerotier"
    #   这些 IP 在组网内是全局可达的，不受物理子网隔离影响
    if _HAS_PSUTIL:
        try:
            import socket as _sock
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            # 检测 Tailscale / ZeroTier 接口
            overlay_keywords = ('tailscale', 'zerotier', 'zt')
            # 100.64.0.0/10 范围: 100.64.0.0 ~ 100.127.255.255
            _TAILSCALE_LO = 0x64400000   # 100.64.0.0
            _TAILSCALE_HI = 0x647FFFFF   # 100.127.255.255

            overlay_candidates = []  # [(ip, iface, is_up)]

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                # Tailscale / ZeroTier 接口名匹配
                is_overlay = any(kw in iface_lower for kw in overlay_keywords)

                stat = stats.get(iface)
                is_up = stat.isup if stat else False

                for addr in addr_list:
                    if addr.family != _sock.AF_INET:
                        continue
                    ip = addr.address
                    if ip.startswith("127."):
                        continue
                    # 100.64.0.0/10 范围的 IP 也视为 overlay（即使接口名不匹配）
                    try:
                        octets = [int(o) for o in ip.split(".")]
                        ip_int = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
                        in_tailscale_range = _TAILSCALE_LO <= ip_int <= _TAILSCALE_HI
                    except (ValueError, IndexError):
                        in_tailscale_range = False

                    if is_overlay or in_tailscale_range:
                        overlay_candidates.append((ip, iface, is_up))

            if overlay_candidates:
                # 优先启用的接口
                overlay_candidates.sort(key=lambda c: (0 if c[2] else 1))
                ip = overlay_candidates[0][0]
                logger.info(f"检测到组网 IP (Tailscale/ZeroTier): {ip} (接口: {overlay_candidates[0][1]})")
                return ip
        except Exception:
            logger.debug("组网 IP (Tailscale/ZeroTier) 检测失败，尝试物理网卡", exc_info=True)

    # ---- 策略 2: psutil 枚举物理网络接口 ----
    if _HAS_PSUTIL:
        try:
            import socket as _sock
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            # 排除的接口前缀（虚拟 / 容器 / 回环 / 组网）
            skip_prefixes = ('lo', 'veth', 'docker', 'br-', 'vmnet', 'virbr',
                             'tun', 'tap', 'wg', 'utun', 'anpi', 'bluetooth',
                             'tailscale', 'zerotier', 'zt')

            candidates = []  # [(ip, iface, is_up, is_wifi)]

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                if any(iface_lower.startswith(p) for p in skip_prefixes):
                    continue

                stat = stats.get(iface)
                is_up = stat.isup if stat else False

                is_wifi = any(kw in iface_lower for kw in
                              ('wi-fi', 'wlan', '无线', 'wifi', 'wireless'))

                for addr in addr_list:
                    if addr.family == _sock.AF_INET and not addr.address.startswith("127."):
                        candidates.append((addr.address, iface, is_up, is_wifi))

            if candidates:
                # 优先级: 启用 + 以太网 > 启用 + WiFi > 启用 > 其他
                def _priority(c):
                    _, _, up, wifi = c
                    if up and not wifi:
                        return 0
                    if up and wifi:
                        return 1
                    return 2

                candidates.sort(key=_priority)
                ip = candidates[0][0]
                logger.info(f"检测到局域网 IP (psutil): {ip} (接口: {candidates[0][1]})")
                return ip
        except Exception:
            logger.debug("psutil 局域网 IP 检测失败，尝试 UDP 默认路由", exc_info=True)

    # ---- 策略 3: UDP 默认路由 IP ----
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 53))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            logger.info(f"检测到局域网 IP (UDP): {ip}")
            return ip
    except Exception:
        logger.debug("UDP 默认路由 IP 检测失败，尝试 hostname", exc_info=True)

    # ---- 策略 4: gethostbyname 兜底 ----
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            logger.info(f"检测到局域网 IP (hostname): {ip} ({hostname})")
            return ip
    except Exception:
        logger.debug("hostname 局域网 IP 检测失败", exc_info=True)

    logger.warning("无法检测局域网 IP，回退到 127.0.0.1")
    return "127.0.0.1"


def get_mac_addresses() -> list[str]:
    """
    获取本机所有物理网卡的 MAC 地址列表。

    用于主节点身份验证：主节点首次启动时将 MAC 地址写入数据库，
    后续启动时验证本机 MAC 与数据库记录是否匹配，防止 IP 变化
    导致身份混淆或其他机器冒充主节点。

    过滤策略:
    - 排除回环接口 (lo, Loopback)
    - 排除虚拟接口 (Docker, VMware, WSL, Hyper-V, VirtualBox)
    - 排除隧道/蓝牙/PPP 接口
    - 仅保留物理网卡 (WiFi + 以太网)

    Returns:
        MAC 地址字符串列表，如 ["aa:bb:cc:dd:ee:ff", "11:22:33:44:55:66"]
        检测失败返回空列表
    """
    macs: list[str] = []

    # 排除的接口关键字（虚拟 / 容器 / 隧道 / 蓝牙 / 回环）
    skip_keywords = (
        'loopback', 'pseudo', 'lookback',
        'docker', 'vmware', 'virtualbox', 'hyper-v', 'wsl',
        'veth', 'br-', 'virbr', 'vbox', 'vmnet',
        'tunnel', 'teredo', 'isatap', '6to4',
        'bluetooth', 'bluetooth',
        'usb', 'ndis', 'wan', 'ppp', 'pppoe',
    )

    # 策略 1: psutil 枚举（最可靠）
    if _HAS_PSUTIL:
        try:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()

                # 排除虚拟/回环接口
                if iface_lower.startswith('lo'):
                    continue
                if any(kw in iface_lower for kw in skip_keywords):
                    continue

                # 检查接口是否真实存在（不是纯软件接口）
                stat = stats.get(iface)
                if stat is None:
                    continue

                for addr in addr_list:
                    # AF_LINK 族（平台相关）— 在 Linux 上是 17，macOS 是 18
                    # psutil 中 MAC 地址的 family 可能是 AF_LINK 或 -1
                    if hasattr(addr, 'family') and str(addr.family) in ('17', '18', '-1', 'AF_LINK'):
                        mac = addr.address
                        if mac and mac != '00:00:00:00:00:00' and ':' in mac:
                            # psutil 有时将 MAC 放在 family=-1 的 address 中
                            pass
                    # 更稳健的方式：psutil 的 snicaddr 有 address 属性
                    # MAC 地址通常伴随 family == psutil.AF_LINK

            # 重新用更可靠的方式：遍历所有地址，取 family == AF_LINK 的
            import ctypes
            # AF_LINK 在不同平台的值不同，但 psutil 会将 MAC 地址的 family 设为 AF_LINK
            # 在 Windows 上 AF_LINK 通常不存在，MAC 地址的 family 可能是 -1

            for iface, addr_list in addrs.items():
                iface_lower = iface.lower()
                if iface_lower.startswith('lo'):
                    continue
                if any(kw in iface_lower for kw in skip_keywords):
                    continue

                stat = stats.get(iface)
                if stat is None:
                    continue

                for addr in addr_list:
                    family_val = getattr(addr, 'family', None)
                    # psutil.AF_LINK 通常用于 MAC 地址
                    # 在不同 OS 上值不同: Linux=17, macOS=18, Windows 可能没有
                    is_mac_addr = False
                    if hasattr(psutil, 'AF_LINK') and family_val == psutil.AF_LINK:
                        is_mac_addr = True
                    elif family_val == -1 or family_val == 17 or family_val == 18:
                        # Windows 或未知平台
                        is_mac_addr = True

                    if is_mac_addr:
                        mac = addr.address.strip().lower()
                        # 验证 MAC 格式 (6 组十六进制，用 : 或 - 分隔)
                        if mac and mac != '00:00:00:00:00:00':
                            import re
                            if re.match(r'^([0-9a-f]{2}[:-]){5}[0-9a-f]{2}$', mac):
                                macs.append(mac)

            if macs:
                logger.info(f"检测到物理网卡 MAC: {macs}")
                return macs
        except Exception as e:
            logger.debug(f"psutil MAC 检测失败: {e}", exc_info=True)

    # 策略 2: uuid.getnode() 兜底（返回单一 MAC）
    try:
        import uuid
        mac_int = uuid.getnode()
        if mac_int and mac_int != 0:
            mac = ':'.join(f'{(mac_int >> (i * 8)) & 0xff:02x}' for i in range(5, -1, -1))
            if mac != '00:00:00:00:00:00':
                logger.info(f"检测到 MAC (uuid): {mac}")
                return [mac]
    except Exception:
        logger.debug("uuid.getnode MAC 检测失败，尝试 Windows getmac", exc_info=True)

    # 策略 3: Windows subprocess 兜底
    try:
        import subprocess
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(
                ['getmac', '/fo', 'csv', '/v'],
                capture_output=True, text=True, timeout=5,
            )
            import re
            for line in result.stdout.strip().split('\n')[1:]:  # skip header
                mac_match = re.findall(r'([0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2})', line)
                for mac in mac_match:
                    mac_lower = mac.lower()
                    if mac_lower != '00:00:00:00:00:00':
                        # 排除蓝牙、断开连接的设备（getmac 会显示 "Media disconnected"）
                        if 'media disconnected' not in line.lower():
                            macs.append(mac_lower)
            if macs:
                logger.info(f"检测到 MAC (getmac): {macs}")
                return macs
    except Exception:
        logger.debug("Windows getmac MAC 检测失败", exc_info=True)

    logger.warning("无法检测到任何有效 MAC 地址")
    return []


# ================================================================
# 封包 / 解包
# ================================================================

def pack_data(payload: bytes) -> bytes:
    """
    封包：4字节长度头（大端序）+ 数据体。

    Args:
        payload: 数据体字节流

    Returns:
        完整数据包字节流
    """
    header = struct.pack(">I", len(payload))
    return header + payload


def unpack_header(header: bytes) -> int:
    """
    解包长度头。

    Args:
        header: 4字节长度头

    Returns:
        数据体长度（字节数）
    """
    return struct.unpack(">I", header)[0]


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    精确接收 n 字节数据（处理 TCP 粘包/拆包）。

    Args:
        sock: socket 对象
        n: 需要接收的字节数

    Returns:
        接收到的字节流，连接断开时返回 None
    """
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


# ================================================================
# 张量序列化 / 反序列化
# ================================================================

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """将张量序列化为字节流（torch.save 到内存 buffer）"""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """从字节流反序列化为张量（torch.load 从内存 buffer）"""
    buffer = io.BytesIO(data)
    return torch.load(buffer)


# ---- 高速序列化路径（流水线隐藏状态传输优化） ----

def serialize_tensor_fast(tensor: torch.Tensor) -> bytes:
    """
    高速张量序列化，自动选择最优路径。

    小张量 (< 1MB): 使用 torch.save（兼容性好、含完整元数据）
    大张量 (≥ 1MB): 使用 numpy tobytes + 自定义头（零拷贝、速度快 ~3x）

    自定义头格式（大张量路径）:
        [4B  magic: b'TNR0' ]
        [1B  dtype_code      ]  # 0=float16, 1=float32, 2=int64, 3=int32
        [1B  ndim             ]
        [4B  dim0             ]
        [4B  dim1             ]
        [4B  dim2 (optional)  ]
        [4B  dim3 (optional)  ]
        [NB  raw_bytes        ]
    总开销: 10-22 字节

    小张量路径:
        [4B  magic: b'TNR1' ]
        [NB  torch.save 输出]
    """
    tensor_cpu = tensor.detach().cpu()
    nbytes = tensor_cpu.numel() * tensor_cpu.element_size()

    if nbytes < 1_000_000:
        # 小张量：torch.save（兼容性好）
        buf = io.BytesIO()
        torch.save(tensor_cpu, buf)
        payload = buf.getvalue()
        return b'TNR1' + payload
    else:
        # 大张量：numpy tobytes（零拷贝）
        import numpy as np
        # Phase 5 review H3: 确保连续内存布局，避免非连续张量 .numpy() 崩溃
        arr = tensor_cpu.contiguous().numpy()
        dtype_map = {
            np.float16: 0, np.float32: 1,
            np.int64: 2, np.int32: 3,
        }
        dtype_code = dtype_map.get(arr.dtype.type)
        if dtype_code is None:
            # Phase 5.1: 拒绝不支持的 dtype，避免静默数据损坏
            raise ValueError(
                f"serialize_tensor_fast 不支持的 dtype: {arr.dtype} "
                f"(仅支持 float16/float32/int32/int64)"
            )
        ndim = arr.ndim
        shape = arr.shape

        header = struct.pack(
            f'>4s B B {"I" * ndim}',
            b'TNR0', dtype_code, ndim, *shape,
        )
        return header + arr.tobytes()


def deserialize_tensor_fast(data: bytes) -> torch.Tensor:
    """
    反序列化 serialize_tensor_fast 的输出。

    magic b'TNR0' → 大张量路径（numpy frombuffer）
    magic b'TNR1' → 小张量路径（torch.load）
    """
    if not data or len(data) < 4:
        raise ValueError("序列化数据为空或长度不足")

    magic = data[:4]
    if magic == b'TNR1':
        # 小张量路径
        buf = io.BytesIO(data[4:])
        return torch.load(buf)
    elif magic == b'TNR0':
        # 大张量路径
        import numpy as np
        dtype_map = {
            0: np.float16, 1: np.float32,
            2: np.int64, 3: np.int32,
        }
        dtype_code = data[4]
        ndim = data[5]
        dtype = dtype_map.get(dtype_code)
        if dtype is None:
            # Phase 5.1: 服务端拒绝未知 dtype code
            raise ValueError(
                f"deserialize_tensor_fast 收到未知 dtype_code: {dtype_code}"
            )
        # 解析 shape（每个维度 4 字节）
        shape = []
        offset = 6
        for _ in range(ndim):
            dim = struct.unpack_from('>I', data, offset)[0]
            shape.append(dim)
            offset += 4
        # 重建张量
        raw = data[offset:]
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return torch.from_numpy(arr.copy())  # copy 确保内存连续
    else:
        # 未知格式，回退到 torch.load（weights_only 防 pickle 注入）
        buf = io.BytesIO(data)
        return torch.load(buf, weights_only=True)


# ================================================================
# 消息构建与解析
# ================================================================

def build_message(msg_type: MessageType, data: Any = None) -> bytes:
    """
    构建通信消息。

    Args:
        msg_type: 消息类型
        data: 消息载荷（dict/str 用于JSON，torch.Tensor 用于张量）

    Returns:
        封包后的字节流
    """
    meta = {"type": msg_type.value}

    if isinstance(data, torch.Tensor):
        meta["format"] = "tensor"
        tensor_bytes = serialize_tensor(data)
        meta_bytes = json.dumps(meta).encode("utf-8")
        # 组合: [meta长度(4B)] + [meta JSON] + [张量数据]
        meta_packed = pack_data(meta_bytes)
        tensor_packed = pack_data(tensor_bytes)
        return meta_packed + tensor_packed
    else:
        meta["format"] = "json"
        if data is not None:
            meta["data"] = data
        return pack_data(json.dumps(meta).encode("utf-8"))


def parse_message(raw: bytes) -> dict:
    """
    解析通信消息。

    Args:
        raw: 原始数据包

    Returns:
        包含 type 和 data 的字典。张量消息额外包含 tensor 字段。
    """
    meta = json.loads(raw.decode("utf-8"))

    if meta.get("format") == "tensor":
        # 张量数据在后续包中，由调用方继续接收
        meta["_needs_tensor"] = True
    return meta


# ================================================================
# TCP 服务端（主节点）
# ================================================================

@dataclass
class ClientConn:
    """已连接客户端信息"""
    client_id: str               # 节点标识 "client1" / "client2"
    sock: socket.socket
    addr: tuple                  # TCP peer endpoint: (ip, ephemeral_port)
    advertised_host: str = ""    # 客户端自身可被其他节点连接的服务 host
    advertised_port: int = SERVER_PORT  # 客户端自身 TCP 服务监听端口
    advertised_address: str = "" # "host:port"，用于节点列表和链式转发
    role: str = ""               # 节点角色
    node_type: str = "pc"        # 设备平台: "pc" | "android"
    hostname: str = ""           # 客户端主机名
    device_info: dict = None     # 客户端设备信息
    network_type: str = "unknown"  # 网络连接类型: wifi | ethernet | unknown
    connected_at: float = 0.0    # 连接时间
    last_heartbeat: float = 0.0  # 上次心跳时间
    heartbeat_missed: int = 0    # 连续心跳丢失次数

    def __post_init__(self):
        if self.device_info is None:
            self.device_info = {}
        if not self.advertised_host and self.addr:
            self.advertised_host = self.addr[0]
        if not self.advertised_port:
            self.advertised_port = SERVER_PORT
        if not self.advertised_address and self.advertised_host:
            self.advertised_address = f"{self.advertised_host}:{self.advertised_port}"
        if self.connected_at == 0.0:
            self.connected_at = time.time()
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


class TCPServer:
    """TCP 服务端：主节点使用，监听从节点连接"""

    MAX_HEARTBEAT_MISSED = 3     # 连续丢失 N 次心跳视为离线

    def __init__(self, host: str = None, port: int = None):
        self.host = host or SERVER_IP
        self.port = port or SERVER_PORT
        self.sock: Optional[socket.socket] = None
        self.clients: dict[str, ClientConn] = {}     # client_id -> ClientConn
        self._clients_lock = threading.RLock()       # 保护 clients 的跨线程访问
        self._running = False
        self._accept_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_threads: dict[str, threading.Thread] = {}
        self.on_message: Optional[Callable] = None    # 消息回调
        self.on_disconnect: Optional[Callable] = None # 断连回调

    def start(self, on_message: Callable = None,
              on_disconnect: Callable = None) -> None:
        """
        启动 TCP 服务端，开始监听。

        Args:
            on_message: 收到消息时的回调函数
                        签名: (client_id: str, msg: dict, raw: bytes) -> None
            on_disconnect: 客户端断连时的回调函数
                          签名: (client_id: str) -> None
        """
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(2)  # 两台从节点
        self.sock.settimeout(1.0)  # accept 超时 1s，以便检查 _running
        self._running = True
        logger.info(f"TCP服务端启动: {self.host}:{self.port}")

        # 启动 accept 循环线程
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # 启动心跳检测线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    # ---- clients 线程安全访问 ----

    def _get_client(self, client_id: str) -> Optional[ClientConn]:
        """线程安全获取客户端连接。"""
        with self._clients_lock:
            return self.clients.get(client_id)

    def _set_client(self, client_id: str, conn: ClientConn) -> None:
        """线程安全设置客户端连接。"""
        with self._clients_lock:
            self.clients[client_id] = conn

    def _pop_client(self, client_id: str) -> Optional[ClientConn]:
        """线程安全移除客户端连接；不存在时返回 None。"""
        with self._clients_lock:
            return self.clients.pop(client_id, None)

    def _pop_client_if_same(self, client_id: str,
                            expected: ClientConn) -> Optional[ClientConn]:
        """仅当当前连接对象仍是 expected 时才移除，避免误删新连接。"""
        with self._clients_lock:
            if self.clients.get(client_id) is expected:
                return self.clients.pop(client_id, None)
            return None

    def _client_ids_snapshot(self) -> list[str]:
        """线程安全获取客户端 ID 快照。"""
        with self._clients_lock:
            return list(self.clients.keys())

    def _client_items_snapshot(self) -> list[tuple[str, ClientConn]]:
        """线程安全获取客户端连接快照。"""
        with self._clients_lock:
            return list(self.clients.items())

    # ---- Accept 循环 ----

    def _accept_loop(self) -> None:
        """在独立线程中循环 accept 客户端连接"""
        logger.info("Accept 循环已启动，等待客户端连接...")
        while self._running:
            try:
                conn, addr = self.sock.accept()
                logger.info(f"新连接: {addr}")
                # 启动消息接收线程（暂时以 addr 为临时 ID，注册后更新）
                temp_id = f"pending_{addr[1]}"
                t = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr, temp_id),
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue  # 正常超时，继续循环
            except OSError as e:
                if self._running:
                    logger.error(f"Accept 异常: {e}", exc_info=True)
                break
        logger.info("Accept 循环已退出")

    # ---- 客户端消息处理 ----

    def _handle_client(self, conn: socket.socket, addr: tuple, temp_id: str) -> None:
        """
        接收并分发客户端消息。

        在独立线程中运行，持续接收消息直到连接断开。
        """
        conn.settimeout(HEARTBEAT_INTERVAL + 2)
        client_id = temp_id

        try:
            while self._running:
                # 接收长度头
                header = recv_exact(conn, HEADER_LEN)
                if header is None:
                    break  # 连接断开

                payload_len = unpack_header(header)
                if payload_len <= 0 or payload_len > MAX_PACKET_SIZE:
                    logger.warning(f"非法包长度: {payload_len}，断开 {addr}")
                    break

                payload = recv_exact(conn, payload_len)
                if payload is None:
                    break

                msg = parse_message(payload)

                # 处理张量附加数据
                if msg.get("_needs_tensor"):
                    tensor_header = recv_exact(conn, HEADER_LEN)
                    if tensor_header is None:
                        break
                    tensor_len = unpack_header(tensor_header)
                    tensor_data = recv_exact(conn, tensor_len)
                    if tensor_data is None:
                        break
                    msg["tensor"] = deserialize_tensor(tensor_data)
                    del msg["_needs_tensor"]

                msg_type = msg.get("type", "")

                # ---- 消息分发 ----
                if msg_type == MessageType.REGISTER.value:
                    # 注册消息：提取客户端身份信息
                    try:
                        client_id = self._handle_registration(conn, addr, temp_id, msg)
                    except _RegistrationRejected as e:
                        logger.info(f"注册被拒，关闭连接: {addr[0]}:{addr[1]} — {e}")
                        try:
                            conn.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        break
                    # 更新 recv 线程映射
                    if temp_id in self._recv_threads:
                        del self._recv_threads[temp_id]
                    self._recv_threads[client_id] = threading.current_thread()

                elif msg_type == MessageType.HEARTBEAT.value:
                    # 心跳：回显客户端时间戳并回复 ACK
                    self._handle_heartbeat(client_id, msg)

                # 回调上层（scheduler）
                if self.on_message:
                    try:
                        self.on_message(client_id, msg)
                    except Exception as e:
                        logger.error(f"消息回调异常: {e}", exc_info=True)

        except socket.timeout:
            logger.info(f"客户端 {client_id} 接收超时")
        except (ConnectionError, OSError) as e:
            logger.warning(f"客户端 {client_id} 连接异常: {e}", exc_info=True)
        finally:
            # 清理
            try:
                conn.close()
            except OSError:
                pass

            removed = None
            if client_id != temp_id:
                current = self._get_client(client_id)
                if current is not None and current.sock is conn:
                    removed = self._pop_client_if_same(client_id, current)

            if removed is not None:
                logger.info(f"客户端 {client_id} 已断开: {removed.addr}")
            else:
                logger.debug(f"未注册或已清理连接已断开: {client_id} addr={addr}")

            if client_id in self._recv_threads:
                del self._recv_threads[client_id]
            # 通知上层断连（仅对本线程实际移除的已注册连接通知一次）
            if self.on_disconnect and removed is not None:
                try:
                    self.on_disconnect(client_id)
                except Exception as e:
                    logger.error(f"断连回调异常: {e}", exc_info=True)

    def _handle_registration(self, conn: socket.socket, addr: tuple,
                             temp_id: str, msg: dict) -> str:
        """
        处理客户端注册消息。

        Returns:
            解析出的 client_id
        """
        data = msg.get("data", {})
        client_id = data.get("client_id", temp_id)
        role = data.get("role", "unknown")
        node_type = data.get("node_type", "pc")
        hostname = data.get("hostname", "unknown")
        device_info = data.get("device_info", {})
        network_type = data.get("network_type", "unknown")

        advertised_host = str(data.get("advertised_host") or "").strip()
        advertised_port_raw = data.get("advertised_port")
        advertised_address = str(data.get("advertised_address") or "").strip()
        if advertised_address and (not advertised_host or advertised_port_raw is None):
            if ":" in advertised_address:
                h, p = advertised_address.rsplit(":", 1)
                advertised_host = advertised_host or h.strip()
                if advertised_port_raw is None:
                    advertised_port_raw = p.strip()
        if advertised_host in ("", "0.0.0.0", "127.0.0.1", "localhost"):
            advertised_host = addr[0]
        try:
            advertised_port = int(advertised_port_raw) if advertised_port_raw is not None else SERVER_PORT
        except (TypeError, ValueError):
            advertised_port = SERVER_PORT
        if not (1 <= advertised_port <= 65535):
            advertised_port = SERVER_PORT
        advertised_address = f"{advertised_host}:{advertised_port}"

        # ★ 阶段 7：HMAC 集群认证
        auth_ok, auth_reason = verify_auth_signature(
            client_id, data.get("auth", {})
        )
        if not auth_ok:
            logger.error(
                f"⛔ 认证失败: {client_id}@{addr[0]}:{addr[1]} — {auth_reason}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": f"认证失败: {auth_reason}",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"认证失败 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected(f"认证失败: {auth_reason}")

        # ★ 安全：拒绝 client_id="master" 或 role="master" 的注册
        if client_id == "master":
            logger.error(
                f"⛔ 拒绝注册: client_id='master' 是保留字，"
                f"来源 {addr[0]}:{addr[1]} hostname={hostname}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "client_id 'master' 是保留字，请使用其他 ID",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("client_id 'master' 是保留字")
        if role == "master":
            logger.error(
                f"⛔ 拒绝注册: 从节点不能声明 role='master'，"
                f"来源 {addr[0]}:{addr[1]} hostname={hostname}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "role 'master' 仅限于主节点自身",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("从节点不能声明 role='master'")
        # Android 节点只能作为 client
        if node_type == "android" and role == "master":
            logger.error(
                f"⛔ 拒绝注册: Android 节点不能担任 master 角色，"
                f"来源 {addr[0]}:{addr[1]}"
            )
            ack = build_message(MessageType.REGISTER, {
                "status": "rejected",
                "reason": "Android 节点不能担任 master 角色",
            })
            try:
                conn.sendall(ack)
            except OSError as e:
                logger.debug(f"注册拒绝 ACK 发送失败: {e}", exc_info=True)
            raise _RegistrationRejected("Android 节点不能担任 master 角色")

        client_conn = ClientConn(
            client_id=client_id,
            sock=conn,
            addr=addr,
            advertised_host=advertised_host,
            advertised_port=advertised_port,
            advertised_address=advertised_address,
            role=role,
            node_type=node_type,
            hostname=hostname,
            device_info=device_info,
            network_type=network_type,
        )
        self._set_client(client_id, client_conn)
        logger.info(
            f"✅ 节点注册成功: {client_id} role={role} "
            f"hostname={hostname} advertised={advertised_address} "
            f"peer={addr[0]}:{addr[1]}"
        )

        # 发送注册确认
        ack = build_message(MessageType.REGISTER, {
            "status": "registered",
            "client_id": client_id,
        })
        try:
            conn.sendall(ack)
        except OSError as e:
            logger.warning(f"注册确认发送失败: {e}", exc_info=True)

        return client_id

    def _handle_heartbeat(self, client_id: str, msg: dict = None) -> None:
        """处理心跳消息：更新时间戳并回复 ACK（回显客户端时间戳用于 RTT 测量）"""
        conn = self._get_client(client_id)
        if conn:
            conn.last_heartbeat = time.time()
            conn.heartbeat_missed = 0
            # 提取客户端发送时间戳，原样回显
            echo_data = None
            if msg and isinstance(msg.get("data"), dict):
                t_send = msg["data"].get("t_send", 0)
                if t_send:
                    echo_data = {"t_send": t_send}
            # 回复 ACK
            try:
                ack = build_message(MessageType.HEARTBEAT_ACK, echo_data)
                conn.sock.sendall(ack)
            except OSError as e:
                logger.debug(f"心跳 ACK 发送失败: client={client_id}, error={e}", exc_info=True)

    def send_to_client(self, client_id: str, data: Any,
                       msg_type: MessageType = MessageType.TENSOR) -> None:
        """
        向指定从节点发送数据。

        Args:
            client_id: 从节点ID（"client1" / "client2"）
            data: 发送数据（dict / torch.Tensor）
            msg_type: 消息类型
        """
        conn = self._get_client(client_id)
        if conn is None:
            raise ConnectionError(f"从节点 {client_id} 未连接")
        packet = build_message(msg_type, data)
        try:
            conn.sock.sendall(packet)
        except OSError as e:
            raise ConnectionError(f"向 {client_id} 发送失败: {e}")

    def broadcast(self, data: Any,
                  msg_type: MessageType = MessageType.TASK_START) -> None:
        """向所有从节点广播消息"""
        for cid in self._client_ids_snapshot():
            try:
                self.send_to_client(cid, data, msg_type)
            except ConnectionError as e:
                logger.warning(f"广播跳过 {cid}: {e}", exc_info=True)

    def send_layer_config(self, client_id: str, assignment: dict) -> None:
        """向指定从节点推送分层配置"""
        self.send_to_client(client_id, assignment, MessageType.LAYER_CONFIG)

    def broadcast_layer_config(self, assignments: dict) -> None:
        """
        向所有已连接从节点广播分层配置。

        Args:
            assignments: {client_id: {start_layer, end_layer, has_embedding, has_lm_head}}
        """
        for cid in self._client_ids_snapshot():
            if cid in assignments:
                try:
                    self.send_layer_config(cid, assignments[cid])
                except ConnectionError as e:
                    logger.warning(f"分层配置推送跳过 {cid}: {e}", exc_info=True)

    # ---- 心跳检测 ----

    def _heartbeat_loop(self) -> None:
        """心跳检测循环：定期检查所有客户端心跳状态"""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            now = time.time()
            for cid, conn in self._client_items_snapshot():
                elapsed = now - conn.last_heartbeat
                if elapsed > HEARTBEAT_INTERVAL * (self.MAX_HEARTBEAT_MISSED + 1):
                    conn.heartbeat_missed = self.MAX_HEARTBEAT_MISSED + 1
                    logger.warning(
                        f"⚠️ 节点 {cid} 心跳超时 ({elapsed:.0f}s)，"
                        f"连续丢失 {conn.heartbeat_missed} 次"
                    )
                    # 关闭连接，触发清理
                    try:
                        conn.sock.close()
                    except OSError:
                        pass
                    removed = self._pop_client_if_same(cid, conn)
                    # 通知上层（仅本线程实际移除连接时通知一次）
                    if removed is not None and self.on_disconnect:
                        try:
                            self.on_disconnect(cid)
                        except Exception as e:
                            logger.error(f"断连回调异常: {e}", exc_info=True)

    def get_client_ids(self) -> list:
        """获取所有已连接客户端 ID 列表"""
        return self._client_ids_snapshot()

    def get_client_info(self, client_id: str) -> Optional[dict]:
        """获取指定客户端的连接信息"""
        c = self._get_client(client_id)
        if c is None:
            return None
        peer_addr = f"{c.addr[0]}:{c.addr[1]}"
        advertised_addr = c.advertised_address or f"{c.advertised_host}:{c.advertised_port}"
        return {
            "client_id": c.client_id,
            # 兼容旧调用：addr 现在表示节点服务端点，不再是 TCP 临时源端口
            "addr": advertised_addr,
            "advertised_addr": advertised_addr,
            "advertised_host": c.advertised_host,
            "advertised_port": c.advertised_port,
            "peer_addr": peer_addr,
            "role": c.role,
            "node_type": c.node_type,
            "hostname": c.hostname,
            "device_info": c.device_info,
            "network_type": c.network_type,
            "connected_at": c.connected_at,
            "last_heartbeat": c.last_heartbeat,
            "heartbeat_missed": c.heartbeat_missed,
        }

    def stop(self) -> None:
        """停止服务端"""
        self._running = False
        with self._clients_lock:
            conns = list(self.clients.values())
            self.clients.clear()
        for conn in conns:
            try:
                conn.sock.close()
            except OSError:
                pass
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        logger.info("TCP服务端已停止")


# ================================================================
# TCP 客户端（从节点）
# ================================================================

class TCPClient:
    """TCP 客户端：从节点使用，连接主节点"""

    def __init__(self, server_host: str = None, server_port: int = None,
                 client_id: str = None, role: str = None,
                 node_type: str = "pc",
                 advertise_host: str = None,
                 advertise_port: int = None):
        self.server_host = server_host or SERVER_IP
        self.server_port = server_port or SERVER_PORT
        self.client_id = client_id or f"client_{socket.gethostname()}"
        self.role = role or "client"
        self.node_type = node_type
        self.advertise_host = advertise_host
        self.advertise_port = advertise_port
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._recv_thread: Optional[threading.Thread] = None
        self.on_message: Optional[Callable] = None
        self.on_heartbeat: Optional[Callable] = None  # 心跳发送后回调（无参数）
        self._registered = False
        self.avg_rtt_ms: float = 0.0            # 滑动平均 RTT（指数加权）
        self._last_heartbeat_send: float = 0.0  # 最近一次心跳发送时间
        self._connect_lock = threading.Lock()   # Phase 5.4: 防止并发 connect()

    @staticmethod
    def _compute_local_model_sha256() -> str:
        """
        计算本地模型文件的 SHA256（用于注册时上报）。

        优先级:
        1. 读取已有的 .sha256 缓存文件
        2. 计算实际模型文件的 SHA256
        3. 无法获取时返回空字符串
        """
        import config as cfg

        # 尝试 GGUF 模型路径
        gguf_path = getattr(cfg, 'GGUF_MODEL_PATH', '')
        if gguf_path and os.path.isfile(gguf_path):
            sha256_file = gguf_path + ".sha256"
            if os.path.isfile(sha256_file):
                try:
                    with open(sha256_file, "r") as f:
                        return f.read().strip().split()[0]
                except Exception:
                    pass
            # 计算并缓存
            try:
                h = hashlib.sha256()
                with open(gguf_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                result = h.hexdigest()
                with open(sha256_file, "w") as f:
                    f.write(f"{result}  {os.path.basename(gguf_path)}\n")
                return result
            except Exception:
                pass

        # 尝试 Safetensors 模型目录
        model_path = getattr(cfg, 'MODEL_PATH', '')
        if model_path and os.path.isdir(model_path):
            # 计算目录下所有 .safetensors 文件的组合 SHA256
            try:
                files = sorted([
                    f for f in os.listdir(model_path)
                    if f.endswith('.safetensors') or f.endswith('.bin')
                ])
                if files:
                    h = hashlib.sha256()
                    for fname in files:
                        fpath = os.path.join(model_path, fname)
                        with open(fpath, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                h.update(chunk)
                    return h.hexdigest()
            except Exception:
                pass

        return ""

    def connect(self, on_message: Callable = None) -> bool:
        """
        连接主节点并完成注册。

        Args:
            on_message: 收到消息时的回调函数

        Returns:
            连接并注册是否成功
        """
        # Phase 5.4: 互斥锁防止并发连接导致多个 socket + 多组线程竞争
        if not self._connect_lock.acquire(blocking=False):
            logger.warning("connect() 已在执行中，忽略并发调用")
            return False
        try:
            self.on_message = on_message
            for attempt in range(RECONNECT_MAX_RETRIES):
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(HEARTBEAT_INTERVAL + 5)
                    self.sock.connect((self.server_host, self.server_port))
                    self._running = True
                    logger.info(f"已连接主节点: {self.server_host}:{self.server_port}")

                    # 发送注册消息（携带身份、网络类型、服务端点和完整设备画像）
                    import platform as _platform
                    network_type = detect_network_type()
                    self._last_network_type = network_type  # 存储供外部查询
                    logger.info(f"检测到网络类型: {network_type}，准备向主节点注册")

                    advertised_host = (self.advertise_host or "").strip()
                    if not advertised_host:
                        try:
                            advertised_host = self.sock.getsockname()[0]
                        except Exception:
                            advertised_host = ""
                    if advertised_host in ("", "0.0.0.0", "127.0.0.1", "localhost"):
                        try:
                            advertised_host = detect_lan_ip()
                        except Exception:
                            advertised_host = advertised_host or "127.0.0.1"
                    advertised_port = self.advertise_port or SERVER_PORT
                    try:
                        advertised_port = int(advertised_port)
                    except (TypeError, ValueError):
                        advertised_port = SERVER_PORT
                    if not (1 <= advertised_port <= 65535):
                        advertised_port = SERVER_PORT
                    advertised_address = f"{advertised_host}:{advertised_port}"
                    logger.info(f"注册服务端点: {advertised_address}")

                    # 收集完整设备画像（延迟导入避免循环依赖）
                    try:
                        from device_profiler import get_profile
                        profiler = get_profile()
                        _device_info = profiler.to_dict()
                    except Exception as e:
                        logger.warning(f"设备画像收集失败，使用基础信息: {e}", exc_info=True)
                        _device_info = {
                            "platform": _platform.system(),
                            "machine": _platform.machine(),
                            "python_version": _platform.python_version(),
                        }

                    reg_data = {
                        "client_id": self.client_id,
                        "role": self.role,
                        "node_type": self.node_type,
                        "hostname": _platform.node(),
                        "network_type": network_type,
                        "advertised_host": advertised_host,
                        "advertised_port": advertised_port,
                        "advertised_address": advertised_address,
                        "device_info": _device_info,
                        "model_sha256": self._compute_local_model_sha256(),
                        "auth": build_auth_signature(self.client_id),
                    }
                    self.send_data(reg_data, MessageType.REGISTER)

                    # 等待注册确认
                    try:
                        ack_msg = self.recv_data()
                        if ack_msg and ack_msg.get("type") == "register":
                            ack_data = ack_msg.get("data", {})
                            if ack_data.get("status") == "registered":
                                self._registered = True
                                logger.info(f"✅ 注册确认: {self.client_id} → {self.server_host}:{self.server_port}")
                    except (socket.timeout, OSError) as e:
                        logger.warning(f"注册确认等待超时: {e}", exc_info=True)

                    # 启动心跳线程
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop, daemon=True
                    )
                    self._heartbeat_thread.start()

                    # 启动接收线程
                    self._recv_thread = threading.Thread(
                        target=self._recv_loop, daemon=True
                    )
                    self._recv_thread.start()

                    return True
                except ConnectionRefusedError:
                    logger.warning(
                        f"连接被拒绝 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}) → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试..."
                    )
                    time.sleep(RECONNECT_DELAY)
                except socket.timeout:
                    logger.warning(
                        f"连接超时 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}) → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试..."
                    )
                    time.sleep(RECONNECT_DELAY)
                except OSError as e:
                    logger.warning(
                        f"连接异常 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES}): {e} → "
                        f"{self.server_host}:{self.server_port}，{RECONNECT_DELAY}s 后重试...",
                        exc_info=True,
                    )
                    time.sleep(RECONNECT_DELAY)

            logger.error(f"无法连接主节点 {self.server_host}:{self.server_port}，已重试 {RECONNECT_MAX_RETRIES} 次")
            return False
        finally:
            self._connect_lock.release()

    def _recv_loop(self) -> None:
        """接收消息循环（在独立线程中运行）"""
        while self._running and self.sock:
            try:
                msg = self.recv_data()
                if msg is None:
                    break  # 连接断开
                # 内部处理：HEARTBEAT_ACK → 计算 RTT
                if msg.get("type") == MessageType.HEARTBEAT_ACK.value:
                    self._handle_heartbeat_ack(msg)
                    continue  # 不向上层转发
                if self.on_message:
                    try:
                        self.on_message(msg)
                    except Exception as e:
                        logger.error(f"消息回调异常: {e}", exc_info=True)
            except socket.timeout:
                continue
            except (ConnectionError, OSError) as e:
                logger.warning(
                    f"客户端接收循环连接异常: {self.client_id} "
                    f"→ {self.server_host}:{self.server_port}, error={e}",
                    exc_info=True,
                )
                break
        logger.info(f"接收循环已退出: {self.client_id}")

    def send_data(self, data: Any, msg_type: MessageType = MessageType.TENSOR) -> None:
        """向主节点发送数据"""
        if not self.sock:
            raise ConnectionError("未连接到主节点")
        packet = build_message(msg_type, data)
        self.sock.sendall(packet)

    def recv_data(self) -> Optional[dict]:
        """
        接收主节点发来的数据（自动解包、还原张量/字符串）。

        Returns:
            解析后的消息字典，连接断开时返回 None
        """
        if not self.sock:
            return None

        # 接收长度头
        header = recv_exact(self.sock, HEADER_LEN)
        if header is None:
            return None
        payload_len = unpack_header(header)

        # 接收数据体
        payload = recv_exact(self.sock, payload_len)
        if payload is None:
            return None

        msg = parse_message(payload)

        # 如果消息携带张量，继续接收张量数据
        if msg.get("_needs_tensor"):
            tensor_header = recv_exact(self.sock, HEADER_LEN)
            if tensor_header is None:
                return None
            tensor_len = unpack_header(tensor_header)
            tensor_data = recv_exact(self.sock, tensor_len)
            if tensor_data is None:
                return None
            msg["tensor"] = deserialize_tensor(tensor_data)
            del msg["_needs_tensor"]

        return msg

    def _handle_heartbeat_ack(self, msg: dict) -> None:
        """
        处理心跳应答：计算往返延迟 RTT。

        服务端在 HEARTBEAT_ACK 中回显客户端发送的时间戳，
        客户端收到后计算 RTT = now - t_send，并用指数加权滑动平均平滑。
        """
        data = msg.get("data", {})
        t_echo = data.get("t_send", 0) if isinstance(data, dict) else 0
        t_now = time.time()
        if t_echo > 0:
            rtt_ms = (t_now - t_echo) * 1000
        elif self._last_heartbeat_send > 0:
            rtt_ms = (t_now - self._last_heartbeat_send) * 1000
        else:
            return  # 无法计算

        # 指数加权滑动平均（α=0.1），平滑网络抖动
        if self.avg_rtt_ms > 0:
            self.avg_rtt_ms = 0.9 * self.avg_rtt_ms + 0.1 * rtt_ms
        else:
            self.avg_rtt_ms = rtt_ms

    def _heartbeat_loop(self) -> None:
        """心跳发送循环（含时间戳用于 RTT 测量）"""
        while self._running and self.sock:
            try:
                self._last_heartbeat_send = time.time()
                self.send_data(
                    {"t_send": self._last_heartbeat_send},
                    MessageType.HEARTBEAT,
                )
                if self.on_heartbeat:
                    try:
                        self.on_heartbeat()
                    except Exception as e:
                        logger.debug(f"on_heartbeat 回调异常: {e}", exc_info=True)
                time.sleep(HEARTBEAT_INTERVAL)
            except (ConnectionError, OSError) as e:
                logger.warning(f"心跳发送失败，尝试重连: {e}", exc_info=True)
                self._reconnect()

    def _reconnect(self) -> None:
        """断线重连"""
        logger.info(
            f"开始重连主节点: client={self.client_id}, "
            f"target={self.server_host}:{self.server_port}"
        )
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        ok = self.connect(self.on_message)
        if ok:
            logger.info(f"重连主节点成功: client={self.client_id}")
        else:
            logger.error(f"重连主节点失败: client={self.client_id}")

    @property
    def is_registered(self) -> bool:
        """是否已完成注册"""
        return self._registered

    def disconnect(self) -> None:
        """断开连接"""
        self._running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        logger.info("已断开主节点连接")
