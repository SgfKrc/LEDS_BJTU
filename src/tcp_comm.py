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
import io
from enum import Enum
from typing import Any, Optional, Callable

import torch

from config import (
    SERVER_IP, SERVER_PORT, HEARTBEAT_INTERVAL,
    RECONNECT_MAX_RETRIES, RECONNECT_DELAY,
)

logger = logging.getLogger(__name__)


# ================================================================
# 数据包格式常量
# ================================================================
HEADER_LEN = 4          # 长度头字节数（大端序 uint32）
MAX_PACKET_SIZE = 256 * 1024 * 1024  # 最大包大小 256MB（特征张量可能较大）


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

class TCPServer:
    """TCP 服务端：主节点使用，监听从节点连接"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or SERVER_IP
        self.port = port or SERVER_PORT
        self.sock: Optional[socket.socket] = None
        self.clients: dict[int, socket.socket] = {}  # client_id -> socket
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.on_message: Optional[Callable] = None  # 消息回调

    def start(self, on_message: Callable = None) -> None:
        """
        启动 TCP 服务端，开始监听。

        Args:
            on_message: 收到消息时的回调函数
        """
        self.on_message = on_message
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(2)  # 两台从节点
        self._running = True
        logger.info(f"TCP服务端启动: {self.host}:{self.port}")

        # 启动心跳检测线程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # 接受连接循环（在实际使用中应在独立线程运行）
        # TODO: 在独立线程中 accept 客户端连接

    def send_to_client(self, client_id: int, data: Any, msg_type: MessageType = MessageType.TENSOR) -> None:
        """
        向指定从节点发送数据。

        Args:
            client_id: 从节点ID
            data: 发送数据（dict / torch.Tensor）
            msg_type: 消息类型
        """
        if client_id not in self.clients:
            raise ConnectionError(f"从节点 {client_id} 未连接")
        packet = build_message(msg_type, data)
        self.clients[client_id].sendall(packet)

    def broadcast(self, data: Any, msg_type: MessageType = MessageType.TASK_START) -> None:
        """向所有从节点广播消息"""
        for cid in self.clients:
            self.send_to_client(cid, data, msg_type)

    def _heartbeat_loop(self) -> None:
        """心跳检测循环"""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            # TODO: 向所有从节点发送心跳，检测超时应答
            # 标记超时节点为离线，触发断线重连

    def stop(self) -> None:
        """停止服务端"""
        self._running = False
        for sock in self.clients.values():
            sock.close()
        if self.sock:
            self.sock.close()
        logger.info("TCP服务端已停止")


# ================================================================
# TCP 客户端（从节点）
# ================================================================

class TCPClient:
    """TCP 客户端：从节点使用，连接主节点"""

    def __init__(self, server_host: str = None, server_port: int = None):
        self.server_host = server_host or SERVER_IP
        self.server_port = server_port or SERVER_PORT
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.on_message: Optional[Callable] = None

    def connect(self, on_message: Callable = None) -> bool:
        """
        连接主节点。

        Args:
            on_message: 收到消息时的回调函数

        Returns:
            连接是否成功
        """
        self.on_message = on_message
        for attempt in range(RECONNECT_MAX_RETRIES):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.server_host, self.server_port))
                self._running = True
                logger.info(f"已连接主节点: {self.server_host}:{self.server_port}")

                # 发送注册消息
                self.send_data({"status": "online"}, MessageType.REGISTER)

                # 启动心跳线程
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True
                )
                self._heartbeat_thread.start()

                return True
            except ConnectionRefusedError:
                logger.warning(
                    f"连接失败 (尝试 {attempt+1}/{RECONNECT_MAX_RETRIES})，"
                    f"{RECONNECT_DELAY}s 后重试..."
                )
                time.sleep(RECONNECT_DELAY)

        logger.error(f"无法连接主节点，已重试 {RECONNECT_MAX_RETRIES} 次")
        return False

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

    def _heartbeat_loop(self) -> None:
        """心跳发送循环"""
        while self._running and self.sock:
            try:
                self.send_data(None, MessageType.HEARTBEAT)
                time.sleep(HEARTBEAT_INTERVAL)
            except (ConnectionError, OSError):
                logger.warning("心跳发送失败，尝试重连...")
                self._reconnect()

    def _reconnect(self) -> None:
        """断线重连"""
        self._running = False
        if self.sock:
            self.sock.close()
            self.sock = None
        self.connect(self.on_message)

    def disconnect(self) -> None:
        """断开连接"""
        self._running = False
        if self.sock:
            self.sock.close()
            self.sock = None
        logger.info("已断开主节点连接")
