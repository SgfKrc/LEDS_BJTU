"""发行 profile 的上报契约（最小发行包 P0-1 的"能力广告"切片）。

覆盖三件事：

1. `detect_runtime_profile()` 的取值规则：只认 `RUNTIME_PROFILES`，其余一律 `unspecified`；
2. src 侧与打包侧（`packaging/packaging/runtime_guard.py`）的 profile 集合一致，防两端漂移；
3. 上报链路：设备画像 → `device_info` → 主节点节点表（`NodeInfo.to_dict`）全程可见。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging" / "packaging"))

import device_profiler as dp  # noqa: E402
from scheduler_types import NodeInfo, NodeState  # noqa: E402


def test_detect_runtime_profile_accepts_each_declared_value(monkeypatch):
    for profile in dp.RUNTIME_PROFILES:
        monkeypatch.setenv("QLH_RUNTIME_PROFILE", profile)
        assert dp.detect_runtime_profile() == profile


def test_detect_runtime_profile_normalizes_case_and_spaces(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "  LLAMA_CPP_ONLY  ")
    assert dp.detect_runtime_profile() == "llama_cpp_only"


def test_detect_runtime_profile_falls_back_to_unspecified(monkeypatch):
    monkeypatch.delenv("QLH_RUNTIME_PROFILE", raising=False)
    assert dp.detect_runtime_profile() == dp.RUNTIME_PROFILE_UNSPECIFIED

    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "gpu_only")
    assert dp.detect_runtime_profile() == dp.RUNTIME_PROFILE_UNSPECIFIED


def test_runtime_profile_set_matches_packaging_contract():
    """src 与打包侧的 profile 集合是同一份合同，任一侧新增必须同时改另一侧。"""
    import runtime_guard

    assert dp.RUNTIME_PROFILES == runtime_guard.RUNTIME_PROFILES


def test_device_profile_dict_carries_runtime_profile(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    payload = dp.DeviceProfiler.mock_mobile().to_dict()

    assert payload["runtime_profile"] == "llama_cpp_only"


def test_peer_report_carries_runtime_profile(monkeypatch):
    """从节点上报设备画像时把 profile 写进 device_info 与客户端对象。"""
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    from inference_service.peer import PeerClient

    peer = PeerClient.__new__(PeerClient)
    peer._client = SimpleNamespace(device_info={})
    peer._report_device_profile()

    assert peer._device_info["runtime_profile"] == "llama_cpp_only"
    assert peer._client.device_info["runtime_profile"] == "llama_cpp_only"


def test_master_node_table_exposes_runtime_profile():
    """主节点 `/api/cluster/nodes` 走的 `NodeInfo.to_dict` 必须透出该字段。"""
    info = NodeInfo(
        node_id="edge-1",
        role="client",
        node_type="pc",
        state=NodeState.ONLINE,
        device_info={"runtime_profile": "llama_cpp_only"},
    )

    assert info.to_dict()["device_info"]["runtime_profile"] == "llama_cpp_only"


def test_pc_task_worker_hello_exposes_runtime_profile(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    from scheduler import Scheduler

    scheduler = Scheduler()
    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["runtime_profile"] == "llama_cpp_only"
