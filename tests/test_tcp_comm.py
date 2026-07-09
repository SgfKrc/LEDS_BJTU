"""
单元测试 — TCP 通信协议
=======================
测试消息打包/解包、MessageType 枚举、消息构建。
纯逻辑测试，无需网络连接。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import socket
import struct
import threading
import pytest
import torch

import tcp_comm as tcp_comm_mod
from tcp_comm import (
    MessageType, pack_data, unpack_header,
    build_message, parse_message, recv_exact,
    HEADER_LEN, MAX_PACKET_SIZE,
    serialize_tensor, deserialize_tensor,
    serialize_tensor_fast, deserialize_tensor_fast,
    TCPServer, TCPClient, ClientConn,
)


# ================================================================
# pack_data / unpack_header 测试
# ================================================================

class TestPackUnpack:
    """测试封包/解包"""

    def test_pack_data_format(self):
        """封包格式: 4字节长度头 + 数据体"""
        payload = b"hello world"
        packed = pack_data(payload)

        # 长度头 = 4 字节
        header = packed[:HEADER_LEN]
        body = packed[HEADER_LEN:]

        assert len(header) == HEADER_LEN
        assert body == payload
        assert unpack_header(header) == len(payload)

    def test_pack_empty_payload(self):
        """空 payload"""
        packed = pack_data(b"")
        assert len(packed) == HEADER_LEN
        assert unpack_header(packed[:HEADER_LEN]) == 0

    def test_pack_large_payload(self):
        """较大 payload（模拟中间特征张量）"""
        payload = b"x" * 100000  # 100KB
        packed = pack_data(payload)
        assert len(packed) == HEADER_LEN + 100000
        assert unpack_header(packed[:HEADER_LEN]) == 100000

    def test_unpack_header_value(self):
        """解包长度头应返回正确的整数值"""
        for size in [0, 1, 255, 256, 65535, 1048576]:
            header = struct.pack(">I", size)
            assert unpack_header(header) == size

    def test_header_big_endian(self):
        """长度头应为大端序"""
        header = struct.pack(">I", 0x12345678)
        # 大端序: 12 34 56 78
        assert header[0] == 0x12
        assert header[1] == 0x34
        assert header[2] == 0x56
        assert header[3] == 0x78
        assert unpack_header(header) == 0x12345678


# ================================================================
# MessageType 枚举测试
# ================================================================

class TestMessageType:
    """测试消息类型枚举"""

    def test_all_types_defined(self):
        """所有消息类型应定义完整"""
        expected = {
            "register", "heartbeat", "heartbeat_ack",
            "task_start", "task_stop", "task_done", "error",
            "tensor", "result",
            "status_req", "status_res",
            "infer_forward", "infer_result", "layer_config",
            "role_transfer", "role_transfer_ack",
            "spare_master_designate", "spare_master_designate_ack",
            "spare_master_activate", "spare_master_activate_ack",
            "spare_master_deactivate",
        }
        actual = set(MessageType.__members__.keys())
        for t in expected:
            assert t.upper() in actual, f"缺少消息类型: {t}"

    def test_string_enum(self):
        """MessageType 应为字符串枚举"""
        assert MessageType.REGISTER.value == "register"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert isinstance(MessageType.REGISTER, str)

    def test_infer_forward_type(self):
        """分布式推理转发类型"""
        assert MessageType.INFER_FORWARD.value == "infer_forward"
        assert MessageType.INFER_RESULT.value == "infer_result"

    def test_layer_config_type(self):
        """分层配置推送类型"""
        assert MessageType.LAYER_CONFIG.value == "layer_config"

    def test_chain_forward_type(self):
        """链式直连转发类型（P2 优化）"""
        assert MessageType.CHAIN_FORWARD.value == "chain_forward"


# ================================================================
# build_message 测试
# ================================================================

class TestBuildMessage:
    """测试消息构建"""

    def test_build_json_message(self):
        """构建 JSON 控制消息"""
        data = {"action": "register", "node_id": "client1"}
        packet = build_message(MessageType.REGISTER, data)

        # 解包验证
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "register"
        # build_message 对 dict 数据会设置 format: "json"
        assert meta.get("format") == "json"

    def test_build_tensor_message(self):
        """构建张量消息（双层封包）"""
        tensor = torch.randn(4, 128)
        packet = build_message(MessageType.TENSOR, tensor)

        # 第一层: meta
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "tensor"
        assert meta["format"] == "tensor"

        # 第二层: tensor data
        offset = HEADER_LEN + meta_len
        tensor_header = packet[offset:offset + HEADER_LEN]
        tensor_len = unpack_header(tensor_header)
        tensor_bytes = packet[offset + HEADER_LEN:offset + HEADER_LEN + tensor_len]

        # 反序列化
        restored = deserialize_tensor(tensor_bytes)
        assert torch.equal(restored, tensor)

    def test_build_message_no_data(self):
        """无数据的消息（如心跳）"""
        packet = build_message(MessageType.HEARTBEAT)
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        meta_bytes = packet[HEADER_LEN:HEADER_LEN + meta_len]
        meta = json.loads(meta_bytes.decode("utf-8"))

        assert meta["type"] == "heartbeat"

    def test_build_message_str_data(self):
        """字符串数据"""
        packet = build_message(MessageType.RESULT, "推理完成")
        header = packet[:HEADER_LEN]
        meta_len = unpack_header(header)
        # 字符串被当作普通 JSON 数据处理（嵌入在 meta 中或作为单独字段）

        # build_message 对 str 的处理：与 dict 相同路径
        assert meta_len > 0


# ================================================================
# serialize_tensor / deserialize_tensor 测试
# ================================================================

class TestTensorSerialization:
    """测试张量序列化"""

    def test_roundtrip_float32(self):
        """FP32 张量序列化往返"""
        tensor = torch.randn(3, 64, 128)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_float16(self):
        """FP16 张量序列化往返"""
        tensor = torch.randn(2, 32, 64, dtype=torch.float16)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_int64(self):
        """INT64 张量（input_ids）序列化往返"""
        tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int64)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_roundtrip_scalar(self):
        """标量张量"""
        tensor = torch.tensor(42.0)
        restored = deserialize_tensor(serialize_tensor(tensor))
        assert torch.equal(restored, tensor)

    def test_serialize_returns_bytes(self):
        """序列化应返回 bytes"""
        tensor = torch.randn(4, 4)
        result = serialize_tensor(tensor)
        assert isinstance(result, bytes)

    def test_large_tensor(self):
        """较大张量（模拟中间特征）"""
        tensor = torch.randn(1, 2048, 2048)  # ~16MB FP32
        serialized = serialize_tensor(tensor)
        restored = deserialize_tensor(serialized)
        assert torch.equal(restored, tensor)
        # 大小应在 MAX_PACKET_SIZE 范围内
        assert len(serialized) < MAX_PACKET_SIZE


# ================================================================
# MAX_PACKET_SIZE 验证
# ================================================================

class TestPacketSizeLimit:
    """测试包大小限制"""

    def test_max_packet_size_reasonable(self):
        """MAX_PACKET_SIZE 应足够大"""
        # 256 MB 应能容纳最大的中间特征张量
        assert MAX_PACKET_SIZE == 256 * 1024 * 1024

    def test_header_len_is_4(self):
        """长度头固定 4 字节"""
        assert HEADER_LEN == 4


# ================================================================
# serialize_tensor_fast / deserialize_tensor_fast 测试
# ================================================================

class TestFastTensorSerialization:
    """测试高速序列化路径（流水线隐藏状态传输）"""

    def test_fast_small_roundtrip_fp32(self):
        """小张量 (<1MB) FP32 往返 — 走 TNR1 路径"""
        tensor = torch.randn(3, 64, 128)  # ~98KB
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR1', f"小张量应走 TNR1 路径，实际 magic: {data[:4]!r}"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_small_roundtrip_fp16(self):
        """小张量 FP16 往返"""
        tensor = torch.randn(2, 32, 64, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_small_roundtrip_int64(self):
        """小张量 INT64（input_ids）往返"""
        tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int64)
        data = serialize_tensor_fast(tensor)
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_large_roundtrip(self):
        """大张量 (≥1MB) 往返 — 走 TNR0 路径（numpy 零拷贝）"""
        # (1, 512, 2048) FP16 ≈ 2MB — 模拟 prefill 阶段隐藏状态
        tensor = torch.randn(1, 512, 2048, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR0', f"大张量应走 TNR0 路径，实际 magic: {data[:4]!r}"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_large_decode_phase(self):
        """解码阶段隐藏状态 (1, 1, 2048) FP16 ≈ 4KB — 走小张量路径"""
        tensor = torch.randn(1, 1, 2048, dtype=torch.float16)
        data = serialize_tensor_fast(tensor)
        assert data[:4] == b'TNR1', "decode 阶段张量小，应走 TNR1 路径"
        restored = deserialize_tensor_fast(data)
        assert torch.equal(restored, tensor)

    def test_fast_equals_slow(self):
        """fast 序列化/反序列化结果应与 slow 一致"""
        tensors = [
            torch.randn(4, 4),
            torch.randn(1, 512, 2048, dtype=torch.float16),
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            torch.tensor(42.0),
        ]
        for t in tensors:
            fast_restored = deserialize_tensor_fast(serialize_tensor_fast(t))
            slow_restored = deserialize_tensor(serialize_tensor(t))
            assert torch.equal(fast_restored, slow_restored), \
                f"fast ≠ slow for shape {t.shape}, dtype {t.dtype}"

    def test_fast_returns_bytes(self):
        """高速序列化应返回 bytes"""
        tensor = torch.randn(8, 8)
        result = serialize_tensor_fast(tensor)
        assert isinstance(result, bytes)
        assert len(result) > 4  # 至少有 magic header

    def test_fast_large_under_max_packet(self):
        """大张量序列化后应在 MAX_PACKET_SIZE 范围内"""
        tensor = torch.randn(1, 2048, 2048, dtype=torch.float16)  # ~8MB
        data = serialize_tensor_fast(tensor)
        assert len(data) < MAX_PACKET_SIZE, \
            f"序列化大小 {len(data)} 超出 MAX_PACKET_SIZE {MAX_PACKET_SIZE}"

    def test_fast_shape_preserved(self):
        """往返后 shape 和 dtype 应完整保留"""
        shapes = [
            (1, 1, 2048),
            (1, 512, 2048),
            (3, 64, 128),
            (1, 24, 32, 64),  # 4D 张量
        ]
        for shape in shapes:
            tensor = torch.randn(*shape, dtype=torch.float16)
            restored = deserialize_tensor_fast(serialize_tensor_fast(tensor))
            assert restored.shape == tensor.shape, \
                f"shape 不一致: {restored.shape} ≠ {tensor.shape}"
            assert restored.dtype == tensor.dtype, \
                f"dtype 不一致: {restored.dtype} ≠ {tensor.dtype}"


# ================================================================
# TCPServer 连接管理与日志测试
# ================================================================

class TestTCPServerConnectionManagement:
    """测试 TCPServer 注册拒绝、连接表线程安全与回调日志。"""

    def test_registration_uses_advertised_endpoint_not_peer_port(self):
        """节点服务地址应使用 advertised_address，peer_addr 保留临时源端口。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client1",
                    "role": "client",
                    "hostname": "worker",
                    "advertised_host": "10.0.0.9",
                    "advertised_port": 8888,
                    "advertised_address": "10.0.0.9:8888",
                    "auth": tcp_comm_mod.build_auth_signature("client1"),
                },
            }
            client_id = server._handle_registration(
                srv_sock, ("10.0.0.9", 54321), "pending_54321", msg
            )
            assert client_id == "client1"
            info = server.get_client_info("client1")
            assert info["advertised_addr"] == "10.0.0.9:8888"
            assert info["addr"] == "10.0.0.9:8888"
            assert info["peer_addr"] == "10.0.0.9:54321"
        finally:
            server.stop()
            srv_sock.close()
            cli_sock.close()

    def test_legacy_registration_falls_back_to_server_port(self):
        """旧客户端未上报 advertised_port 时应使用 peer_ip:SERVER_PORT，而不是临时端口。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client_legacy",
                    "role": "client",
                    "auth": tcp_comm_mod.build_auth_signature("client_legacy"),
                },
            }
            server._handle_registration(
                srv_sock, ("10.0.0.8", 51234), "pending_51234", msg
            )
            info = server.get_client_info("client_legacy")
            assert info["advertised_addr"] == f"10.0.0.8:{tcp_comm_mod.SERVER_PORT}"
            assert info["peer_addr"] == "10.0.0.8:51234"
        finally:
            server.stop()
            srv_sock.close()
            cli_sock.close()

    def test_rejected_registration_returns_ack_and_raises(self):
        """认证失败的 REGISTER 应返回 rejected ACK 并抛内部拒绝异常。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            msg = {
                "type": "register",
                "data": {
                    "client_id": "client_bad",
                    "role": "client",
                    "auth": {"auth_timestamp": 1.0, "auth_signature": "bad"},
                },
            }
            with pytest.raises(tcp_comm_mod._RegistrationRejected):
                server._handle_registration(srv_sock, ("127.0.0.1", 54321), "pending_54321", msg)

            header = recv_exact(cli_sock, HEADER_LEN)
            assert header is not None
            payload = recv_exact(cli_sock, unpack_header(header))
            ack = parse_message(payload)
            assert ack["type"] == "register"
            assert ack["data"]["status"] == "rejected"
            assert server.get_client_ids() == []
        finally:
            srv_sock.close()
            cli_sock.close()

    def test_rejected_registration_does_not_call_on_message_or_leak_client(self):
        """_handle_client 应拦截注册拒绝，不进入上层回调且不泄露 pending_*。"""
        server = TCPServer(host="127.0.0.1", port=0)
        called = []
        server.on_message = lambda client_id, msg: called.append((client_id, msg))

        srv_sock, cli_sock = socket.socketpair()
        try:
            t = threading.Thread(
                target=server._handle_client,
                args=(srv_sock, ("127.0.0.1", 54321), "pending_54321"),
                daemon=True,
            )
            t.start()

            cli_sock.sendall(build_message(MessageType.REGISTER, {
                "client_id": "client_bad",
                "role": "client",
                "auth": {"auth_timestamp": 1.0, "auth_signature": "bad"},
            }))

            t.join(timeout=2)
            assert not t.is_alive()
            assert called == []
            assert server.get_client_ids() == []
            assert not any(cid.startswith("pending_") for cid in server.get_client_ids())
        finally:
            cli_sock.close()

    def test_pop_client_is_idempotent_and_send_missing_raises_connection_error(self):
        """重复清理连接不应 KeyError，不存在客户端发送应抛 ConnectionError。"""
        server = TCPServer(host="127.0.0.1", port=0)
        srv_sock, cli_sock = socket.socketpair()
        try:
            conn = ClientConn(client_id="client1", sock=srv_sock, addr=("127.0.0.1", 1))
            server._set_client("client1", conn)
            assert server._pop_client("client1") is conn
            assert server._pop_client("client1") is None
            with pytest.raises(ConnectionError):
                server.send_to_client("client1", {"x": 1}, MessageType.STATUS_REQ)
        finally:
            srv_sock.close()
            cli_sock.close()

    def test_on_heartbeat_exception_is_logged(self, monkeypatch, caplog):
        """on_heartbeat 回调异常应进入 DEBUG 日志且携带 exc_info。"""
        client = TCPClient(server_host="127.0.0.1", server_port=1, client_id="client1")
        client.sock = object()
        client._running = True

        def fake_send_data(data, msg_type):
            client._running = False

        def bad_callback():
            raise ValueError("boom")

        monkeypatch.setattr(client, "send_data", fake_send_data)
        monkeypatch.setattr(tcp_comm_mod.time, "sleep", lambda _s: None)
        client.on_heartbeat = bad_callback

        with caplog.at_level(logging.DEBUG, logger="tcp_comm"):
            client._heartbeat_loop()

        records = [r for r in caplog.records if "on_heartbeat 回调异常" in r.getMessage()]
        assert records
        assert records[0].exc_info is not None


# ================================================================
# Phase 7 P1: recv_exact 超时测试
# ================================================================

class TestRecvExactTimeout:
    """测试 recv_exact 在 socket 超时和对方关闭时的行为。"""

    def test_recv_exact_returns_none_on_peer_close(self):
        """对方关闭连接时 recv_exact 应返回 None。"""
        from tcp_comm import recv_exact

        srv, cli = socket.socketpair()
        try:
            cli.close()  # 写端关闭
            srv.settimeout(1.0)
            result = recv_exact(srv, 1024)
            assert result is None, "对方关闭时应返回 None"
        finally:
            srv.close()

    def test_recv_exact_timeout_raises_on_blocking(self):
        """超时时 socket 应抛出 socket.timeout（无数据到达）。"""
        from tcp_comm import recv_exact

        srv, cli = socket.socketpair()
        try:
            srv.settimeout(0.1)
            # 对方不发数据 → recv 超时
            with pytest.raises((socket.timeout, OSError)):
                recv_exact(srv, 1024)
        finally:
            srv.close()
            cli.close()

    def test_serialize_deserialize_large_tensor_roundtrip(self):
        """接近 16MB 的张量序列化→反序列化往返应正确。"""
        from tcp_comm import serialize_tensor_fast, deserialize_tensor_fast

        # 创建一个 ~4M 元素的 float32 张量 ≈ 16MB
        shape = (1024, 1024, 4)
        t = torch.randn(*shape, dtype=torch.float32)
        data = serialize_tensor_fast(t)
        assert len(data) > 0
        t2 = deserialize_tensor_fast(data)
        assert t2.shape == shape
        assert torch.allclose(t, t2, atol=1e-6)

    def test_serialize_tensor_fast_rejects_unsupported_dtype(self):
        """Phase 5.1: 不支持的 dtype 应抛 ValueError 而非静默转 float32。"""
        from tcp_comm import serialize_tensor_fast

        # 需要 >= 1MB 触发 TNR0（大张量）路径；float64 不在支持的 dtype map 中
        # float64: element_size=8, nbytes=8*250k=2MB → TNR0 路径
        t = torch.zeros(250000, dtype=torch.float64)
        with pytest.raises(ValueError, match="不支持"):
            serialize_tensor_fast(t)
