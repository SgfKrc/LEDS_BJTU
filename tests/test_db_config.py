"""
单元测试 — 数据库配置离线测试
===========================
测试配置解析、环境变量处理与线程安全（无需任何数据库连接）。

原 13 个 requires_db 用例（cluster_config 键值 / user_settings /
save_history / distributed_inference）已随 PostgreSQL 退场（M1.3）改写为
本地 SQLite 版，见 tests/test_local_store.py 的 TestLocalSettings。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pytest



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


# ================================================================
# S1/S2 修复验证：DB 凭据环境变量配置测试（离线）
# ================================================================

class TestDbEnvVarConfig:
    """测试数据库凭据从环境变量读取（BUG S1/S2 修复验证）"""

    def test_db_host_from_env(self, monkeypatch):
        """DB_HOST 应从 QLH_DB_HOST 环境变量读取"""
        monkeypatch.setenv("QLH_DB_HOST", "10.20.30.40")
        # 重新加载 db 模块以读取新的环境变量
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_HOST == "10.20.30.40"

    def test_db_password_from_env(self, monkeypatch):
        """DB_PASSWORD 应从 QLH_DB_PASSWORD 环境变量读取，而非硬编码"""
        monkeypatch.setenv("QLH_DB_PASSWORD", "my_secret_password")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PASSWORD == "my_secret_password"
        # 不应包含旧硬编码值
        assert db_mod.DB_PASSWORD != "WUTqw6bLkK3Hn5Va"

    def test_db_password_empty_by_default(self, monkeypatch):
        """未设置环境变量时 DB_PASSWORD 应为空字符串"""
        real_exists = os.path.exists

        def no_dotenv(path):
            if os.path.basename(str(path)) == ".env":
                return False
            return real_exists(path)

        monkeypatch.delenv("QLH_DB_PASSWORD", raising=False)
        monkeypatch.setattr(os.path, "exists", no_dotenv)
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PASSWORD == ""

    def test_db_port_from_env(self, monkeypatch):
        """DB_PORT 应从 QLH_DB_PORT 环境变量读取并转为 int"""
        monkeypatch.setenv("QLH_DB_PORT", "6543")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PORT == 6543
        assert isinstance(db_mod.DB_PORT, int)

    def test_no_hardcoded_password_in_source(self):
        """db.py 源码中不应包含旧的硬编码密码"""
        import db as db_mod
        import inspect
        source = inspect.getsource(db_mod)
        # 旧的硬编码密码不应出现在源码中
        assert "WUTqw6bLkK3Hn5Va" not in source, "源码中仍包含旧的硬编码数据库密码"
        # 旧的硬编码IP不应出现在源码中
        assert "8.160.161.53" not in source, "源码中仍包含旧的硬编码数据库IP"

    def test_no_password_in_log_message(self, monkeypatch):
        """连接池创建日志不应包含密码片段（BUG S2 修复验证）"""
        import importlib
        import db as db_mod
        monkeypatch.setenv("QLH_DB_PASSWORD", "test_password_123")
        importlib.reload(db_mod)
        # 检查 logger.info 的格式字符串不包含密码
        # get_pool() 函数的日志格式应不包含 pwd= 字段
        import inspect
        source = inspect.getsource(db_mod.get_pool)
        assert "pwd=" not in source, "get_pool() 源码中仍包含密码日志输出"
        assert "DB_PASSWORD[:4]" not in source, "源码中仍包含密码截取逻辑"


# ================================================================
# P1 修复验证：环境变量解析错误处理测试
# ================================================================

class TestDbEnvVarErrorHandling:
    """测试环境变量解析错误时的安全处理（BUG P1 修复验证）"""

    def test_db_port_invalid_string(self, monkeypatch):
        """QLH_DB_PORT 设置为非数字字符串时应使用默认值 5432"""
        monkeypatch.setenv("QLH_DB_PORT", "not_a_number")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PORT == 5432, "无效端口字符串应回退到默认值 5432"

    def test_db_port_out_of_range(self, monkeypatch):
        """QLH_DB_PORT 设置超出范围时应使用默认值"""
        monkeypatch.setenv("QLH_DB_PORT", "99999")  # 超出 65535
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PORT == 5432, "超出范围的端口应回退到默认值"

    def test_db_min_conn_invalid_string(self, monkeypatch):
        """QLH_DB_MIN_CONN 设置为非数字字符串时应使用默认值 2"""
        monkeypatch.setenv("QLH_DB_MIN_CONN", "invalid")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_MIN_CONN == 2, "无效连接数字符串应回退到默认值"

    def test_db_max_conn_invalid_string(self, monkeypatch):
        """QLH_DB_MAX_CONN 设置为非数字字符串时应使用默认值 8"""
        monkeypatch.setenv("QLH_DB_MAX_CONN", "not_int")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_MAX_CONN == 8, "无效最大连接数字符串应回退到默认值"

    def test_db_min_greater_than_max_auto_swap(self, monkeypatch):
        """DB_MIN_CONN > DB_MAX_CONN 时应自动交换"""
        monkeypatch.setenv("QLH_DB_MIN_CONN", "10")
        monkeypatch.setenv("QLH_DB_MAX_CONN", "3")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_MIN_CONN <= db_mod.DB_MAX_CONN, \
            f"MIN_CONN({db_mod.DB_MIN_CONN}) 不应大于 MAX_CONN({db_mod.DB_MAX_CONN})"

    def test_db_port_negative_value(self, monkeypatch):
        """QLH_DB_PORT 设置为负数时应使用默认值"""
        monkeypatch.setenv("QLH_DB_PORT", "-1")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_PORT == 5432, "负数端口应回退到默认值"

    def test_db_conn_zero_value(self, monkeypatch):
        """QLH_DB_MIN_CONN 设置为 0 时应使用默认值"""
        monkeypatch.setenv("QLH_DB_MIN_CONN", "0")
        import importlib
        import db as db_mod
        importlib.reload(db_mod)
        assert db_mod.DB_MIN_CONN >= 1, "最小连接数不应小于 1"


# ================================================================
# P2 修复验证：close_db() 线程安全测试
# ================================================================

class TestDbCloseThreadSafety:
    """测试 close_db() 的线程安全性（BUG P2 修复验证）"""

    def test_close_db_has_lock_protection(self):
        """close_db() 应使用 _init_lock 保护"""
        import db as db_mod
        import inspect
        source = inspect.getsource(db_mod.close_db)
        assert "_init_lock" in source, "close_db() 应使用 _init_lock 进行线程安全保护"

    def test_close_db_idempotent(self):
        """多次调用 close_db() 不应报错"""
        import db as db_mod
        # 多次调用不应抛出异常
        db_mod.close_db()
        db_mod.close_db()
        db_mod.close_db()
        # 如果执行到这里说明 close_db 是幂等的
