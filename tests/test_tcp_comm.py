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
import struct
import pytest
import torch

from tcp_comm import (
    MessageType, pack_data, unpack_header,
    build_message, HEADER_LEN, MAX_PACKET_SIZE,
    serialize_tensor, deserialize_tensor,
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
