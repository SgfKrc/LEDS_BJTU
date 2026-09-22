"""`scripts/relay_l2l_net_probe.py` 的轻量判据测试（不需要 GPU / 模型）。

只覆盖不依赖推理的部分：endpoint 解析与 prompt token 构造；真正的数值判据由探针在实际设备上
运行时给出（记录见 `build/relay-records/l2l-net-*.json`）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relay_l2l_net_probe.py"


def _load():
    spec = importlib.util.spec_from_file_location("l2l_probe_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_endpoint_accepts_host_port() -> None:
    module = _load()
    assert module._split("127.0.0.1:50184") == ("127.0.0.1", 50184)


def test_split_endpoint_rejects_bad_shape() -> None:
    module = _load()
    for bad in ("127.0.0.1", "127.0.0.1:abc", ":50184"):
        with pytest.raises(SystemExit):
            module._split(bad)


class _FakeTokenizer:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, _text: str) -> list[int]:
        return list(self._ids)


def test_build_prompt_tokens_pads_to_prefill(tmp_path: Path) -> None:
    """短文本要重复到 prefill 长度（否则不同切点的输入长度不一致，比较就无效）。"""
    module = _load()
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello world", encoding="utf-8")
    tokens = module._build_prompt_tokens(_FakeTokenizer([1, 2, 3]), prompt, 8)
    assert len(tokens) == 8
    assert tokens[:3] == [1, 2, 3]


def test_build_prompt_tokens_truncates_long_text(tmp_path: Path) -> None:
    module = _load()
    prompt = tmp_path / "p.txt"
    prompt.write_text("x", encoding="utf-8")
    tokens = module._build_prompt_tokens(_FakeTokenizer(list(range(50))), prompt, 8)
    assert tokens == list(range(8))


def test_preflight_reports_unreachable_endpoint() -> None:
    """★ P4.5 健康检查：对**无人监听**的端点必须给出明确诊断，而不是模糊的连接错误。

    对应一个真实踩坑：远端段若已退出（典型是随 ssh 会话被网络抖动静默带走），隧道端口仍在
    本机监听，第一次收发才会失败，错误形如连接被 reset —— **与"模型算错"几乎无法区分**。
    """
    module = _load()
    with pytest.raises(SystemExit) as excinfo:
        module._preflight("127.0.0.1:50999", role="末段(tail)", timeout=1.0)
    message = str(excinfo.value)
    assert "不可达" in message
    assert "已退出" in message          # 指向正确的排查方向
    assert "alive_at" in message        # 给出可执行的下一步


def test_preflight_accepts_listening_endpoint() -> None:
    """有服务监听时必须**通过**（否则健康检查会误杀正常链路）。"""
    import socket

    module = _load()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        module._preflight(f"127.0.0.1:{port}", role="末段(tail)", timeout=1.0)
    finally:
        listener.close()


def test_preflight_rejects_bad_endpoint_shape() -> None:
    module = _load()
    with pytest.raises(SystemExit):
        module._preflight("127.0.0.1", role="末段(tail)", timeout=1.0)


def test_mid_service_exposes_heartbeat_option() -> None:
    """服务端必须提供 `--heartbeat-interval`（健康检查的服务端一半）。"""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "relay_mid_service.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert "--heartbeat-interval" in done.stdout
