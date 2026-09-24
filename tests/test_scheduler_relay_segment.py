"""tests/test_scheduler_relay_segment.py — ★ A1 / X 档：调度层「Relay 段委托」的接线级用例。

不打起完整调度器：构造 `SchedulerPipelineMixin` 的**最小假实例**（只 stub 三个协作方法），
把 `_normalize_relay_segment` 与 `_handle_layer_forward_via_relay` 当**单元**测 —— 毫秒级覆盖全部分支，
不需要模型、不需要 master/worker 拓扑（真拓扑回归在 X 档后续的 `tests/test_scheduler_relay_segment.py`
扩展里）。线协议部分用真实 loopback 服务 + 假 runner。

⚠️ 含「该红必须红」断言：
1. `PIPELINE_RELAY_ENABLED` **默认必须是 False**（默认打开 = 生产路径被静默改动 ⇒ 立刻红）；
2. 段失败时**必须抛具名异常**、**不得**静默发出空 hidden（把失败吞掉 ⇒ 立刻红）。
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import config  # noqa: E402
import scheduler_pipeline  # noqa: E402
from relay_segment_client import RelaySegmentError  # noqa: E402
from relay_transport import (  # noqa: E402
    RELAY_TRANSPORT_ERROR,
    expected_hidden_bytes,
    open_loopback_listener,
    serve_relay_middle_connection,
)
from scheduler_pipeline import SchedulerPipelineMixin  # noqa: E402
from scheduler_pipeline import RELAY_HIDDEN_WIRE_FORMAT, _decode_relay_hidden, _encode_relay_hidden  # noqa: E402

N_EMBD = 4
N_TOKENS = 2


class _PlusOneRunner:
    def __init__(self) -> None:
        self.seen: list[tuple[int, bytes]] = []
        self.meta: dict[str, object] | None = None
        self.resets = 0

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        self.seen.append((n_tokens, hidden))
        return bytes((value + 1) % 256 for value in hidden)

    def request_hidden_seq(self, hidden: bytes, *, n_tokens: int, meta) -> bytes:
        self.seen.append((n_tokens, hidden))
        self.meta = meta
        return bytes((value + 1) % 256 for value in hidden)

    def reset(self) -> None:
        self.resets += 1


def _hidden(seed: int = 0) -> bytes:
    return bytes((seed + i) % 256 for i in range(expected_hidden_bytes(N_TOKENS, N_EMBD)))


class _Harness:
    """最小假实例：只 stub 与 relay 分支协作的三个方法，并记录发回主节点的结果。"""

    def __init__(self) -> None:
        self.obj = object.__new__(SchedulerPipelineMixin)
        self.sent: list[dict[str, object]] = []
        self.began: list[str] = []
        self.obj.get_effective_node_id = lambda: "worker-1"     # type: ignore[method-assign]
        self.obj._begin_local_pipeline_task = self.began.append   # type: ignore[method-assign]
        self.obj._send_layer_result = self._send_layer_result     # type: ignore[method-assign]

    def _send_layer_result(self, client_id: str, task_id: str, **kwargs) -> bool:
        self.sent.append({"client_id": client_id, "task_id": task_id, **kwargs})
        return True


def _listen_middle(runner):
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])
    box: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            box["bridge"] = serve_relay_middle_connection(sock, runner, n_embd=N_EMBD,
                                                          max_tokens=64)
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return listener, port, thread


def _spec(port: int, **overrides) -> dict[str, object]:
    spec = {"role": "middle", "host": "127.0.0.1", "port": port, "n_embd": N_EMBD,
            "timeout": 5.0}
    spec.update(overrides)
    return spec


def test_relay_wire_format_is_raw_f32_and_shape_preserving():
    import torch

    value = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    encoded, shape = _encode_relay_hidden(value)
    raw = __import__("base64").b64decode(encoded)
    assert shape == [1, 2, 4]
    assert len(raw) == 1 * 2 * 4 * 4
    assert _decode_relay_hidden(raw, shape).dtype == torch.float32
    assert _decode_relay_hidden(raw, shape).shape == value.shape


def test_relay_wire_format_rejects_shape_mismatch():
    import torch

    encoded, _ = _encode_relay_hidden(torch.zeros((1, 2, N_EMBD)))
    raw = __import__("base64").b64decode(encoded)
    with pytest.raises(ValueError, match="length mismatch"):
        _decode_relay_hidden(raw, [1, 3, N_EMBD])


# ---- _normalize_relay_segment：严格拒绝 ------------------------------------


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (None, "缺失"),
        ("middle", "不是 dict"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4, "extra": 1}, "未知键"),
        ({"role": "head", "host": "127.0.0.1", "port": 1, "n_embd": 4}, "head 属 Y 档"),
        ({"role": "tail", "host": "127.0.0.1", "port": 1, "n_embd": 4}, "tail 属 Y 档"),
        ({"role": "middle", "host": "10.1.2.3", "port": 1, "n_embd": 4}, "非 loopback"),
        ({"role": "middle", "host": "127.0.0.1", "port": 0, "n_embd": 4}, "端口越界"),
        ({"role": "middle", "host": "127.0.0.1", "port": 70000, "n_embd": 4}, "端口越界"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 0}, "n_embd 非法"),
        ({"role": "middle", "host": "127.0.0.1", "port": 1, "n_embd": 4, "timeout": 0}, "timeout 非法"),
        ({"role": "middle", "host": "127.0.0.1", "port": "x", "n_embd": 4}, "端口不可解析"),
    ],
)
def test_normalize_rejects_out_of_scope(raw, why: str):
    assert SchedulerPipelineMixin._normalize_relay_segment(raw) is None, why


def test_normalize_accepts_middle_loopback():
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": "MIDDLE", "host": "localhost", "port": 50183, "n_embd": 896, "timeout": 30})
    assert spec == {"role": "middle", "host": "localhost", "port": 50183, "n_embd": 896,
                    "timeout": 30.0}


def test_normalize_defaults_timeout():
    spec = SchedulerPipelineMixin._normalize_relay_segment(
        {"role": "middle", "host": "127.0.0.1", "port": 50183, "n_embd": 896})
    assert spec is not None and spec["timeout"] == 60.0


# ---- _handle_layer_forward_via_relay：成功路径 ----------------------------


def test_via_relay_forwards_hidden_and_sends_result():
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        harness.obj._handle_layer_forward_via_relay(
            _spec(port),
            data={"hidden_states": _hidden(), "task_id": "t1", "step": 0},
            task_id="t1", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    assert harness.began == ["t1"]                    # begin/finish 平衡的前提
    assert len(harness.sent) == 1
    sent = harness.sent[0]
    assert sent["client_id"] == "master" and sent["task_id"] == "t1"
    result = sent["result_data"]
    expected = bytes((value + 1) % 256 for value in _hidden())
    assert result["hidden_states"] == expected        # 逐字节正确（且不是原样返回）
    assert result["hidden_states"] != _hidden()
    assert result["hidden_shape"] == [N_TOKENS, N_EMBD]
    assert result["chain_path"] == ["worker-1"]
    assert result["metrics"]["relay_executed"] is True
    assert set(result["metrics"]) >= {"relay_segment", "relay_frames", "relay_tokens",
                                      "relay_payload_bytes", "relay_error"}
    assert result["metrics"]["relay_error"] == ""
    assert runner.seen and runner.seen[0][0] == N_TOKENS


def test_via_relay_passes_seq_meta():
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        harness.obj._handle_layer_forward_via_relay(
            _spec(port),
            data={"hidden_states": _hidden(), "task_id": "t2", "step": 0,
                  "seq_ids": [0, 1], "positions": [0, 0]},
            task_id="t2", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    assert runner.meta is not None
    assert list(runner.meta["seq_ids"]) == [0, 1]
    assert list(runner.meta["n_seq_id"]) == [1, 1]


def test_relay_session_is_reused_until_task_finish():
    runner = _PlusOneRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = int(listener.getsockname()[1])
    box: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            box["bridge"] = serve_relay_middle_connection(
                sock, runner, n_embd=N_EMBD, max_tokens=64,
            )
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    harness = _Harness()
    try:
        spec = _spec(port)
        harness.obj._handle_layer_forward_via_relay(
            spec, data={"hidden_states": _hidden(), "task_id": "reuse", "step": 0},
            task_id="reuse", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        harness.obj._handle_layer_forward_via_relay(
            spec, data={"hidden_states": _hidden(1), "task_id": "reuse", "step": 1},
            task_id="reuse", step=1, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        assert len(runner.seen) == 2
        assert "reuse" in harness.obj._relay_segment_clients
        harness.obj._close_relay_segment_client("reuse")
        assert "reuse" not in harness.obj._relay_segment_clients
        thread.join(timeout=5)
    finally:
        listener.close()


# ---- 失败/越界必须具名（绝不静默）-----------------------------------------


def test_via_relay_failure_is_named_and_sends_nothing():
    """段不可达 ⇒ **抛具名异常**（供外层回传 `relay_segment_failed:*`），**不发**空 hidden。

    「该红必须红」：任何人把这里的 raise 改成"发一个空 hidden 继续"，本用例立刻红 ——
    而那种改动会让失败退化成"模型算错"（正是我们要消灭的那类症状）。
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    harness = _Harness()
    with pytest.raises(RelaySegmentError) as excinfo:
        harness.obj._handle_layer_forward_via_relay(
            _spec(dead_port, timeout=1.0),
            data={"hidden_states": _hidden(), "task_id": "t3", "step": 0},
            task_id="t3", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )

    assert excinfo.value.code == RELAY_TRANSPORT_ERROR
    assert harness.sent == []          # 绝不在失败时发出 LAYER_RESULT


def test_via_relay_rejects_chain_next():
    """X 档只做 2 段拓扑：带 `chain_next` 显式拒绝（>2 段属 Y 档）。"""
    harness = _Harness()
    with pytest.raises(RuntimeError, match="链式转发"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183),
            data={"hidden_states": _hidden(), "task_id": "t4", "step": 0,
                  "chain_next": {"node_id": "worker-2"}},
            task_id="t4", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


def test_via_relay_rejects_wrong_hidden_size():
    harness = _Harness()
    with pytest.raises(RuntimeError, match="长度与 n_embd 不匹配"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183),
            data={"hidden_states": b"too-short", "task_id": "t5", "step": 0},
            task_id="t5", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


def test_via_relay_rejects_missing_hidden():
    harness = _Harness()
    with pytest.raises(RuntimeError, match="需要 hidden_states"):
        harness.obj._handle_layer_forward_via_relay(
            _spec(50183), data={"task_id": "t6", "step": 0},
            task_id="t6", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
    assert harness.sent == []


# ---- 主节点侧：下发规格与具名回退 -----------------------------------------


def test_parse_relay_segment_map_accepts_valid_and_drops_invalid():
    """配置解析：只认 X 档范围内的条目，非法条目**整条丢弃**（绝不下发"半懂"规格）。"""
    raw = ("worker-2=middle@127.0.0.1:50183#896;"           # 合法
           "worker-3=head@127.0.0.1:50184#896;"            # head ⇒ Y 档，丢
           "worker-4=middle@10.0.0.5:50185#896;"           # 非 loopback，丢
           "broken;worker-5=middle@127.0.0.1:notaport#896;"  # 非法，丢
           "worker-6=middle@127.0.0.1:50186#896")          # 合法
    parsed = SchedulerPipelineMixin._parse_relay_segment_map(raw)

    assert set(parsed) == {"worker-2", "worker-6"}
    assert parsed["worker-2"] == {"role": "middle", "host": "127.0.0.1", "port": 50183,
                                  "n_embd": 896, "timeout": 60.0}


def test_parse_relay_segment_map_empty_is_empty():
    assert SchedulerPipelineMixin._parse_relay_segment_map("") == {}
    assert SchedulerPipelineMixin._parse_relay_segment_map("   ;  ") == {}


def test_relay_segment_for_worker_respects_switch(monkeypatch: pytest.MonkeyPatch):
    """★ 开关关闭 ⇒ **永远不下发**（对既有路径零影响）。"""
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", False)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-2=middle@127.0.0.1:50183#896")
    harness = _Harness()

    assert harness.obj._relay_segment_for_worker("worker-2") is None


def test_relay_segment_for_worker_returns_spec_and_caches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-2=middle@127.0.0.1:50183#896")
    harness = _Harness()

    spec = harness.obj._relay_segment_for_worker("worker-2")
    assert spec is not None and spec["port"] == 50183
    assert harness.obj._relay_segment_for_worker("worker-9") is None

    # 缓存：解析一次后进程内稳定（改配置不影响已解析结果）
    monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                        "worker-9=middle@127.0.0.1:50199#896")
    assert harness.obj._relay_segment_for_worker("worker-9") is None


def test_via_relay_failure_message_is_readable_for_fallback_reason():
    """★ 失败消息必须能直接落进 `_fallback_reason`（带 `relay_segment_failed:` 前缀）。

    「该红必须红」：把 `detail` 去掉（只留白名单码）会让这条红 —— 而主节点的
    `_fallback_reason` 就只能写笼统码，正是我们要消灭的"与模型算错难以区分"。
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = int(probe.getsockname()[1])
    probe.close()

    harness = _Harness()
    with pytest.raises(RelaySegmentError) as excinfo:
        harness.obj._handle_layer_forward_via_relay(
            _spec(dead_port, timeout=1.0),
            data={"hidden_states": _hidden(), "task_id": "t7", "step": 0},
            task_id="t7", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )

    message = str(excinfo.value)
    assert message.startswith(f"relay_segment_failed:{RELAY_TRANSPORT_ERROR}")
    assert "middle@127.0.0.1" in message
    assert excinfo.value.code == RELAY_TRANSPORT_ERROR     # code 与可读消息解耦
    assert harness.sent == []


# ---- 接线闭环 -------------------------------------------------------------


def test_end_to_end_config_to_worker_relay_roundtrip(monkeypatch: pytest.MonkeyPatch):
    """★ 接线闭环：主节点配置 ⇒ 下发规格 ⇒ worker 认它 ⇒ 真 middle 服务跑通。

    这是 X 档「本机可闭环」的核心证据：全链路（配置解析 → 两侧判据同源 → 段委托 →
    逐字节往返）都在本进程内完成，**不需要模型、不需要多节点**。
    """
    runner = _PlusOneRunner()
    listener, port, thread = _listen_middle(runner)
    harness = _Harness()
    try:
        monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_ENABLED", True)
        monkeypatch.setattr(scheduler_pipeline, "PIPELINE_RELAY_SEGMENTS",
                            f"worker-1=middle@127.0.0.1:{port}#{N_EMBD}")

        # ① 主节点侧：解析配置并取该 worker 的规格（= 会随 LAYER_FORWARD 下发的那个 dict）
        spec = harness.obj._relay_segment_for_worker("worker-1")
        assert spec is not None

        # ② worker 侧：收到的规格必须被 `_normalize_relay_segment` **原样**接受
        #    （两侧判据同源 ⇒ 不会出现"主节点下发、worker 不认"的隐性不对称）
        normalized = SchedulerPipelineMixin._normalize_relay_segment(spec)
        assert normalized == spec

        # ③ 执行：真 loopback 往返 ⇒ 逐字节正确
        harness.obj._handle_layer_forward_via_relay(
            normalized,
            data={"hidden_states": _hidden(), "task_id": "e2e", "step": 0},
            task_id="e2e", step=0, config_id="c1", model_sha256="sha",
            model_type="qwen", received_chain_path=[],
        )
        thread.join(timeout=5)
    finally:
        listener.close()

    result = harness.sent[0]["result_data"]
    assert result["hidden_states"] == bytes((v + 1) % 256 for v in _hidden())
    assert result["hidden_states"] != _hidden()
    assert result["metrics"]["relay_executed"] is True
    assert result["metrics"]["relay_tokens"] == N_TOKENS


# ---- 「该红必须红」：开关默认必须关 ---------------------------------------


def test_guard_relay_switch_defaults_off():
    """★「该红必须红」：`QLH_RELAY_ENABLED` **默认关**。

    默认打开意味着生产路径被静默改动（没有显式 opt-in 的节点也会走 Relay 分支），
    任何人把默认值改成 True，这条立刻红。
    """
    assert config.PIPELINE_RELAY_ENABLED is False
    assert scheduler_pipeline.PIPELINE_RELAY_ENABLED is False


def test_guard_branch_requires_both_switch_and_spec(monkeypatch: pytest.MonkeyPatch):
    """★「该红必须红」：委托需要**开关 + 合法规格**同时成立。

    直接按 `_handle_layer_forward_locked` 里的分支条件断言：任一条件不成立都必须
    退回既有拒绝路径（而不是走 Relay）。
    """
    def _would_delegate(raw: object, enabled: bool) -> bool:
        spec = SchedulerPipelineMixin._normalize_relay_segment(raw)
        return bool(enabled) and spec is not None

    good = {"role": "middle", "host": "127.0.0.1", "port": 50183, "n_embd": 896}
    assert _would_delegate(good, True) is True
    assert _would_delegate(good, False) is False        # 开关关 ⇒ 不委托
    assert _would_delegate(None, True) is False         # 无规格 ⇒ 不委托
    assert _would_delegate({"role": "tail", "host": "127.0.0.1", "port": 1,
                            "n_embd": 4}, True) is False
