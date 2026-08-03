"""张量跨进程传输（loopback 策略，微服务架构改造计划 §4.1）。

复用 tcp_comm.serialize_tensor_fast / deserialize_tensor_fast
（tcp_comm.py:685/742）：每 token 传输量 = hidden 向量
（Qwen-1.8B：2048 float16 ≈ 4KB），loopback 上开销 ~0.1ms 级。

KV 缓存永不跨进程（§4.1 决策 2）：本模块只负责 hidden states / logits
等张量的序列化；KV 只传任务级引用（task_id）。
"""
from tcp_comm import deserialize_tensor_fast, serialize_tensor_fast

__all__ = ["serialize_tensor", "deserialize_tensor"]


def serialize_tensor(tensor) -> bytes:
    """将 torch.Tensor 序列化为字节（复用 tcp_comm 成熟实现）。"""
    return serialize_tensor_fast(tensor)


def deserialize_tensor(data: bytes):
    """将 serialize_tensor 的字节还原为 torch.Tensor。"""
    return deserialize_tensor_fast(data)
