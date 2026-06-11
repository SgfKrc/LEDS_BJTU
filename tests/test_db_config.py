"""
单元测试 — 数据库配置持久化
===========================
测试 cluster_config 表的读写操作（需要 PostgreSQL 连接）。
使用 pytest.mark.requires_db 标记，CI 中可跳过。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pytest


# ================================================================
# 辅助：检查数据库是否可用
# ================================================================

def _db_connected():
    """检查是否可以连接到 PostgreSQL"""
    try:
        from db import get_pool, db_health
        get_pool()
        return db_health()["status"] == "ok"
    except Exception:
        return False


DB_AVAILABLE = _db_connected()
requires_db = pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL 不可用")


# ================================================================
# cluster_config 键值存储测试
# ================================================================

@pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL 不可用")
class TestClusterConfig:
    """测试 cluster_config 表的 CRUD 操作"""

    TEST_KEY = "__pytest_test_key__"
    TEST_VALUE = "pytest_value_12345"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """每个测试后清理"""
        yield
        try:
            from db import set_config
            set_config(self.TEST_KEY, "")
        except Exception:
            pass

    def test_set_and_get_config(self):
        """基本写入和读取"""
        from db import set_config, get_config

        set_config(self.TEST_KEY, self.TEST_VALUE)
        result = get_config(self.TEST_KEY, "default")
        assert result == self.TEST_VALUE

    def test_get_config_default(self):
        """读取不存在的键应返回默认值"""
        from db import get_config

        result = get_config("__nonexistent_key_xyz__", "fallback")
        assert result == "fallback"

    def test_overwrite_config(self):
        """重复写入应覆盖旧值"""
        from db import set_config, get_config

        set_config(self.TEST_KEY, "v1")
        set_config(self.TEST_KEY, "v2")
        assert get_config(self.TEST_KEY) == "v2"

    def test_get_all_configs(self):
        """批量读取应包含刚写入的键"""
        from db import set_config, get_all_configs

        set_config(self.TEST_KEY, self.TEST_VALUE)
        configs = get_all_configs()
        assert self.TEST_KEY in configs
        assert configs[self.TEST_KEY] == self.TEST_VALUE

    def test_set_configs_batch(self):
        """批量写入应全部生效"""
        from db import set_configs_batch, get_config, set_config

        keys = {
            f"{self.TEST_KEY}_a": "alpha",
            f"{self.TEST_KEY}_b": "beta",
        }
        set_configs_batch(keys)

        for k, v in keys.items():
            assert get_config(k) == v
            set_config(k, "")  # cleanup


# ================================================================
# 用户偏好设置 JSON 存储测试
# ================================================================

@pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL 不可用")
class TestUserSettings:
    """测试用户设置的 JSON 存储"""

    def test_set_and_get_user_settings(self):
        """完整设置 JSON 的写入和读取"""
        from db import set_user_settings, get_user_settings, set_config

        settings = {
            "maxNewTokens": 512,
            "temperature": 0.7,
            "topP": 0.9,
            "saveHistory": True,
            "distributedInference": True,
            "theme": "dark",
        }
        set_user_settings(settings)
        result = get_user_settings()

        assert result["maxNewTokens"] == 512
        assert result["temperature"] == 0.7
        assert result["topP"] == 0.9
        assert result["saveHistory"] is True
        assert result["distributedInference"] is True
        assert result["theme"] == "dark"

        # cleanup
        set_config("user_settings", "")

    def test_get_empty_settings(self):
        """无记录时应返回空 dict"""
        from db import get_user_settings

        # 确保键不存在
        from db import set_config
        set_config("user_settings", "")

        result = get_user_settings()
        assert result == {}

    def test_overwrite_settings(self):
        """覆盖写入后应返回新值"""
        from db import set_user_settings, get_user_settings, set_config

        set_user_settings({"a": 1, "b": 2})
        set_user_settings({"c": 3})

        result = get_user_settings()
        assert result == {"c": 3}

        # cleanup
        set_config("user_settings", "")

    def test_settings_persist_across_reads(self):
        """同一连接多次读取应一致"""
        from db import set_user_settings, get_user_settings, set_config

        settings = {"token": 1024, "temp": 0.5}
        set_user_settings(settings)

        r1 = get_user_settings()
        r2 = get_user_settings()
        assert r1 == r2 == settings

        # cleanup
        set_config("user_settings", "")


# ================================================================
# save_history / distributed_inference 专用键测试
# ================================================================

@pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL 不可用")
class TestDedicatedKeys:
    """测试 save_history 和 distributed_inference_enabled 专用键"""

    def test_save_history_default_false(self):
        """默认应返回 False（无记录时）"""
        from db import get_save_history, set_config
        # 清空已有记录，确保测试默认行为
        set_config("save_history", "")
        result = get_save_history()
        assert result is False

    def test_save_history_set_true(self):
        """设置为 True 后应返回 True"""
        from db import set_save_history, get_save_history

        set_save_history(True)
        assert get_save_history() is True
        set_save_history(False)  # restore

    def test_distributed_inference_default_true(self):
        """默认应返回 True（主节点默认开启）"""
        from db import get_distributed_inference_enabled

        result = get_distributed_inference_enabled()
        assert result is True  # 默认值

    def test_distributed_inference_toggle(self):
        """开关切换应正确持久化"""
        from db import set_distributed_inference_enabled, get_distributed_inference_enabled

        set_distributed_inference_enabled(False)
        assert get_distributed_inference_enabled() is False

        set_distributed_inference_enabled(True)
        assert get_distributed_inference_enabled() is True


# ================================================================
# 离线测试（不依赖数据库）
# ================================================================

class TestConfigLogicOffline:
    """测试配置逻辑本身（无需 DB 连接）"""

    def test_user_settings_json_roundtrip(self):
        """验证 JSON 序列化/反序列化正确性"""
        original = {
            "maxNewTokens": 2048,
            "temperature": 0.3,
            "topP": 0.95,
            "saveHistory": True,
            "distributedInference": False,
        }
        serialized = json.dumps(original, ensure_ascii=False)
        restored = json.loads(serialized)
        assert restored == original

    def test_settings_merge_priority(self):
        """验证设置合并逻辑：localStorage 优先，云端补充"""
        cloud = {"maxNewTokens": 512, "temperature": 0.7, "topP": 0.9}
        local = {"maxNewTokens": 1024, "temperature": 0.5}  # 用户改了这两个

        # merge: cloud 基础 + local 覆盖
        merged = {**cloud, **local}
        assert merged["maxNewTokens"] == 1024  # local wins
        assert merged["temperature"] == 0.5    # local wins
        assert merged["topP"] == 0.9           # cloud supplement

    def test_empty_cloud_fallback(self):
        """云端为空时完全使用本地设置"""
        cloud = {}
        local = {"maxNewTokens": 512, "temperature": 0.7}

        merged = {**cloud, **local}
        assert merged == local
