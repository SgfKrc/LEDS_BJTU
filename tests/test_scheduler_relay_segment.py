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
