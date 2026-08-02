"""阶段 0.3 测试：node_runtime 运行时状态单例。

验证：
  - 初始值来自 config（NODE_ID/NODE_ROLE）
  - set/get 与 update 语义（等价原 scheduler._sync_runtime_node_config）
  - 并发读写安全
  - scheduler 的运行时身份读写改走 node_runtime（cfg 不再被写回）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import threading

from node_runtime import NodeRuntime, node_runtime


class TestNodeRuntimeBasics:
    def test_initial_from_config(self):
        # 用独立实例验证初始值（全局单例可能被其他测试污染）
        rt = NodeRuntime()
        import config as cfg
        assert rt.get_node_id() == str(cfg.NODE_ID)
        assert rt.get_node_role() == str(cfg.NODE_ROLE)

    def test_set_get(self):
        rt = NodeRuntime()
        rt.set_node_id("node-7")
        rt.set_node_role("client")
        assert rt.get_node_id() == "node-7"
        assert rt.get_node_role() == "client"

    def test_update_partial(self):
        rt = NodeRuntime()
        rt.update(node_id="n1")
        assert rt.get_node_id() == "n1"
        assert rt.get_node_role() != ""  # 未动 role
        rt.update(node_role="slave")
        assert rt.get_node_role() == "slave"
        assert rt.get_node_id() == "n1"

    def test_type_coercion(self):
        rt = NodeRuntime()
        rt.set_node_id(123)
        assert rt.get_node_id() == "123"


class TestNodeRuntimeConcurrency:
    def test_concurrent_set(self):
        rt = NodeRuntime()
        errors = []

        def writer(i):
            try:
                for _ in range(200):
                    rt.set_node_id(f"node-{i}")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # 最终值必为某个 writer 的合法值
        assert rt.get_node_id().startswith("node-")


class TestSchedulerIntegration:
    def test_scheduler_sync_uses_node_runtime(self):
        # scheduler._sync_runtime_node_config 不再写回 cfg
        import scheduler as sched_mod

        import config as cfg
        before_cfg = cfg.NODE_ID
        try:
            sched_mod._sync_runtime_node_config(node_id="sync-test-node")
            assert node_runtime.get_node_id() == "sync-test-node"
            # cfg.NODE_ID 未被写回（原实现会写）
            assert cfg.NODE_ID == before_cfg
        finally:
            # 恢复
            sched_mod._sync_runtime_node_config(node_id=before_cfg)

    def test_configured_node_id_reads_runtime(self):
        import scheduler as sched_mod
        old = node_runtime.get_node_id()
        try:
            node_runtime.set_node_id("runtime-1")
            assert sched_mod._configured_node_id() == "runtime-1"
        finally:
            node_runtime.set_node_id(old)

    def test_node_config_apply_uses_node_runtime(self):
        # apply_runtime_config 写 node_id/role 时更新 node_runtime 而非 cfg
        import node_config
        import config as cfg
        before = cfg.NODE_ID
        node_config.apply_runtime_config({
            "node": {"node_id": "apply-node", "role": "master"},
        })
        assert node_runtime.get_node_id() == "apply-node"
        assert cfg.NODE_ID == before  # config 不再被写回
        node_config.apply_runtime_config({
            "node": {"node_id": before},
        })
