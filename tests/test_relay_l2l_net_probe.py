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
    # ★ 2026-09-23（§10.2）：ready 文件要能带构建标识 ⇒ 服务端必须有该开关
    assert "--digest-artifacts" in done.stdout


def test_segment_builds_records_local_head_and_unknown_remote() -> None:
    """★ 2026-09-23（§10.2）：记录里逐段写 runner/构建；远端没给 ready 文件时**显式** unknown。"""
    import argparse

    module = _load()
    args = argparse.Namespace(
        head_endpoint=None,
        head_model="head16.gguf",
        shim="build/keephead/build-cpu/bin/qlh_keep_head.dll",
        head_ready_file=None,
        middle_endpoint=None,
        middle_ready_file=None,
        tail_endpoint="127.0.0.1:50188",
        tail_ready_file=None,
        digest_artifacts=False,
    )
    segments = module._segment_builds(args)

    assert segments["head"]["runner"] == "llama_keep_head.KeepHeadUpstream"
    assert segments["head"]["mode"] == "nextn"
    assert segments["head"]["build"]["schema_version"] == "qlh.relay_segment_info.v1"
    # 没有中间段 ⇒ 该段不出现（而不是给一个空对象）
    assert "middle" not in segments
    # 远端末段未提供 ready 文件 ⇒ 显式 remote_unknown（记录要能区分「未知」与「没写」）
    assert segments["tail"]["build"]["source"] == "remote_unknown"
    assert segments["tail"]["endpoint"] == "127.0.0.1:50188"


def test_segment_builds_reads_ready_file_for_remote_segments(tmp_path: Path) -> None:
    import argparse
    import json

    module = _load()
    ready = tmp_path / "mid.ready"
    ready.write_text(json.dumps({"role": "middle", "build": {"llama_cpp_version": "x"}}),
                     encoding="utf-8")
    args = argparse.Namespace(
        head_endpoint="127.0.0.1:50185",
        head_model=None,
        shim="build/keephead/build-cpu/bin/qlh_keep_head.dll",
        head_ready_file=None,
        middle_endpoint="127.0.0.1:50190",
        middle_ready_file=str(ready),
        tail_endpoint="127.0.0.1:50188",
        tail_ready_file=None,
        digest_artifacts=False,
    )
    segments = module._segment_builds(args)

    assert segments["head"]["build"]["source"] == "remote_unknown"     # head 也没给 ready 文件
    assert segments["middle"]["build"]["source"] == "ready_file"
    assert segments["middle"]["build"]["llama_cpp_version"] == "x"


def _coverage_args(**overrides):
    """★ #30 的层覆盖参数默认全空，逐个用例按需覆盖（`argparse` 按本文件既有风格就近 import）。"""
    import argparse

    base = {
        "head_layers": None, "mid_layers": None, "tail_start": None,
        "total_layers": None, "middle_endpoint": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_layer_coverage_accepts_exact_three_segment_tiling() -> None:
    """三段恰好铺满 0..23 ⇒ verified（`head8` + `mid8-16` + `cut-k16`）。"""
    module = _load()
    args = _coverage_args(head_layers="8", mid_layers="8-16", tail_start=16,
                          total_layers=24, middle_endpoint="127.0.0.1:50161")
    result = module._layer_coverage(args)

    assert result["status"] == "verified"
    assert result["segments"] == {"head": [0, 8], "middle": [8, 16], "tail": [16, 24]}


def test_layer_coverage_rejects_layers_missing_in_the_middle() -> None:
    """★ 这正是 2026-09-26 误判的形态：`head4` + `mid8-16` ⇒ 缺 4..7，**必须** invalid。

    当时端点实际载的是 `mid8-16`（而记录写成 `mid4-16`）⇒ 覆盖缺 4..7 ⇒ 逐 token 不一致被误判成
    "代码缺陷"。本用例把这个形态钉成回归（见 `docs/已知问题记录.md` #30）。
    """
    module = _load()
    args = _coverage_args(head_layers="4", mid_layers="8-16", tail_start=16,
                          total_layers=24, middle_endpoint="127.0.0.1:50161")
    result = module._layer_coverage(args)

    assert result["status"] == "invalid"
    assert "缺口或重叠" in result["detail"]


def test_layer_coverage_rejects_overlap() -> None:
    """中段起点早于 head 终点 ⇒ 重叠，**必须** invalid。"""
    module = _load()
    args = _coverage_args(head_layers="8", mid_layers="4-16", tail_start=16,
                          total_layers=24, middle_endpoint="127.0.0.1:50161")
    assert module._layer_coverage(args)["status"] == "invalid"


def test_layer_coverage_accepts_exact_two_segment_tiling() -> None:
    """两段恰好铺满（`head8` + `tail8`）⇒ verified；`head8` + 起点 12 ⇒ invalid。"""
    module = _load()
    ok = _coverage_args(head_layers="8", tail_start=8, total_layers=24)
    assert module._layer_coverage(ok)["status"] == "verified"

    gap = _coverage_args(head_layers="8", tail_start=12, total_layers=24)
    assert module._layer_coverage(gap)["status"] == "invalid"


def test_layer_coverage_is_unverified_without_full_arguments() -> None:
    """参数不全 ⇒ unverified（**默认不拦**）；三段缺 `--mid-layers` 同样是 unverified。"""
    module = _load()
    assert module._layer_coverage(_coverage_args(head_layers="8"))["status"] == "unverified"
    assert module._layer_coverage(_coverage_args(
        head_layers="8", tail_start=16, total_layers=24,
        middle_endpoint="127.0.0.1:50161"))["status"] == "unverified"
    assert module._layer_coverage(_coverage_args(
        head_layers="8", tail_start=16, total_layers=24,
        middle_endpoint="127.0.0.1:50161"))["status"] == "unverified"
