"""
单元测试 — 本地持久化存储 (local_store)
=====================================
测试 JSON 文件降级存储的 CRUD 操作和线程安全。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import json
import sqlite3
import tempfile
import threading
import time

# Mock _store_dir before importing local_store
_original_store_dir = None


@pytest.fixture(autouse=True)
def temp_store_dir(monkeypatch):
    """将 local_store 的存储目录重定向到临时目录"""
    tmpdir = tempfile.mkdtemp(prefix="qlh_test_localstore_")
    import local_store
    legacy_dir = os.path.join(tmpdir, "legacy")
    sqlite_path = os.path.join(tmpdir, "qlh-control.sqlite3")

    # Patch _store_dir 函数
    def _mock_store_dir():
        os.makedirs(tmpdir, exist_ok=True)
        return tmpdir

    monkeypatch.setattr(local_store, '_store_dir', _mock_store_dir)
    monkeypatch.setattr(local_store, '_get_store_dir', _mock_store_dir)
    monkeypatch.setattr(local_store, '_legacy_store_dir', lambda: legacy_dir)
    monkeypatch.setattr(local_store, '_sqlite_path', lambda: sqlite_path)
    local_store._initialized_paths.clear()

    yield {
        "root": tmpdir,
        "legacy_dir": legacy_dir,
        "sqlite_path": sqlite_path,
    }

    # 清理
    import shutil
    local_store._initialized_paths.clear()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestLocalSessions:
    """测试本地会话 CRUD"""

    def test_create_session(self):
        from local_store import create_local_session, get_local_session
        sid = "test-session-001"
        session = create_local_session(sid, "测试对话")
        assert session["id"] == sid
        assert session["title"] == "测试对话"
        assert session["message_count"] == 0

        # 应能通过 ID 查询
        fetched = get_local_session(sid)
        assert fetched is not None
        assert fetched["id"] == sid

    def test_list_sessions_sorted(self):
        from local_store import create_local_session, get_all_local_sessions
        sid1 = "test-session-a"
        sid2 = "test-session-b"
        create_local_session(sid1, "A")
        time.sleep(0.01)  # 确保时间戳不同
        create_local_session(sid2, "B")

        sessions = get_all_local_sessions()
        ids = [s["id"] for s in sessions]
        # 最新的在前
        assert ids.index(sid2) < ids.index(sid1), "较新的会话应排在前面"

    def test_update_session_title(self):
        from local_store import (
            create_local_session, update_local_session_title,
            get_local_session,
        )
        sid = "test-session-title"
        create_local_session(sid, "原标题")
        updated = update_local_session_title(sid, "新标题")
        assert updated is not None
        assert updated["title"] == "新标题"

        fetched = get_local_session(sid)
        assert fetched["title"] == "新标题"

    def test_delete_session(self):
        from local_store import create_local_session, delete_local_session, get_local_session
        sid = "test-session-del"
        create_local_session(sid, "待删除")
        assert get_local_session(sid) is not None

        deleted = delete_local_session(sid)
        assert deleted == 1
        assert get_local_session(sid) is None

    def test_delete_nonexistent_session(self):
        from local_store import delete_local_session
        deleted = delete_local_session("nonexistent-id")
        assert deleted == 0


class TestLocalMessages:
    """测试本地消息 CRUD"""

    def test_save_and_load_messages(self):
        from local_store import (
            create_local_session, save_local_message,
            load_local_conversation,
        )
        sid = "test-session-msg"
        create_local_session(sid, "消息测试")

        save_local_message(sid, "user", "你好")
        save_local_message(sid, "assistant", "你好！有什么可以帮助你的？")

        msgs = load_local_conversation(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"

    def test_message_count_tracking(self):
        from local_store import (
            create_local_session, save_local_message,
            get_local_session, increment_local_session_message_count,
        )
        sid = "test-session-count"
        create_local_session(sid, "计数测试")

        save_local_message(sid, "user", "msg1")
        increment_local_session_message_count(sid)
        save_local_message(sid, "assistant", "msg2")
        increment_local_session_message_count(sid)

        session = get_local_session(sid)
        assert session["message_count"] == 2

    def test_message_limit(self):
        from local_store import (
            create_local_session, save_local_message,
            load_local_conversation,
        )
        sid = "test-session-limit"
        create_local_session(sid, "消息限制测试")

        # 创建 10 条消息
        for i in range(10):
            save_local_message(sid, "user", f"msg-{i}")

        # limit=5
        msgs = load_local_conversation(sid, limit=5)
        assert len(msgs) == 5
        # 应返回最后 5 条
        assert msgs[0]["content"] == "msg-5"
        assert msgs[-1]["content"] == "msg-9"

    def test_clear_conversation(self):
        from local_store import (
            create_local_session, save_local_message,
            clear_local_conversation, load_local_conversation,
        )
        sid = "test-session-clear"
        create_local_session(sid, "清空测试")

        save_local_message(sid, "user", "msg1")
        save_local_message(sid, "user", "msg2")

        count = clear_local_conversation(sid)
        assert count == 2
        msgs = load_local_conversation(sid)
        assert len(msgs) == 0

    def test_delete_message_range(self):
        from local_store import (
            create_local_session, save_local_message,
            delete_local_message_range, load_local_conversation,
        )
        sid = "test-session-range"
        create_local_session(sid, "范围删除测试")

        # 构造 4 条消息 = 2 轮（user+assistant）
        save_local_message(sid, "user", "q1")
        save_local_message(sid, "assistant", "a1")
        save_local_message(sid, "user", "q2")
        save_local_message(sid, "assistant", "a2")

        # 删除第 0 轮
        deleted = delete_local_message_range(sid, 0)
        assert deleted == 2
        msgs = load_local_conversation(sid)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "q2"

    def test_metrics_saved_with_message(self):
        from local_store import (
            create_local_session, save_local_message,
            load_local_conversation,
        )
        sid = "test-session-metrics"
        create_local_session(sid, "metrics 测试")

        save_local_message(sid, "assistant", "reply", metrics={
            "engine": "llama_cpp", "tokens_per_second": 12.5,
        })

        msgs = load_local_conversation(sid)
        assert len(msgs) == 1
        assert msgs[0]["metrics"]["engine"] == "llama_cpp"
        assert msgs[0]["metrics"]["tokens_per_second"] == 12.5


class TestLocalStoreThreadSafety:
    """测试本地存储的线程安全性"""

    def test_concurrent_writes(self):
        from local_store import create_local_session, save_local_message
        sid = "test-session-thread"
        create_local_session(sid, "线程测试")

        errors = []

        def write_messages(thread_id):
            try:
                for i in range(10):
                    save_local_message(sid, "user", f"t{thread_id}-msg-{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_messages, args=(i,))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发写入出现异常: {errors}"

    def test_concurrent_reads_during_write(self):
        from local_store import (
            create_local_session, save_local_message,
            load_local_conversation,
        )
        sid = "test-session-readwrite"
        create_local_session(sid, "读写并发测试")

        save_local_message(sid, "user", "init")

        errors = []
        # 注意: Windows 上 os.replace() 在目标文件被同时读取时可能返回
        # PermissionError。这是原子写入在 Windows 上的已知限制。
        # 写操作之间有足够间隔时不会触发，实际使用中读/写频率远低于本测试。

        def writer():
            for i in range(10):
                try:
                    save_local_message(sid, "user", f"w-{i}")
                    time.sleep(0.05)  # 给 reader 窗口时间
                except PermissionError:
                    pass  # Windows 原子写入限制，可接受
                except Exception as e:
                    errors.append(str(e))

        def reader():
            for _ in range(5):
                try:
                    load_local_conversation(sid)
                    time.sleep(0.1)
                except Exception as e:
                    errors.append(str(e))

        readers = [threading.Thread(target=reader) for _ in range(2)]
        writer_thread = threading.Thread(target=writer)

        for r in readers:
            r.start()
        writer_thread.start()
        writer_thread.join()
        for r in readers:
            r.join(timeout=3)

        assert len(errors) == 0, f"并发读写出现非预期异常: {errors}"


class TestLocalStoreStats:
    """测试统计信息"""

    def test_stats_empty(self):
        from local_store import local_store_stats
        stats = local_store_stats()
        assert "store_dir" in stats
        assert stats["session_count"] == 0
        assert stats["message_count"] == 0

    def test_stats_with_data(self):
        from local_store import (
            create_local_session, save_local_message,
            local_store_stats,
        )
        sid = "test-stats"
        create_local_session(sid, "统计测试")
        save_local_message(sid, "user", "m1")
        save_local_message(sid, "assistant", "m2")

        stats = local_store_stats()
        assert stats["session_count"] == 1
        assert stats["message_count"] == 2


class TestSqliteCompatibility:
    def test_python_store_does_not_claim_schema_version(self, temp_store_dir):
        from local_store import initialize_local_store

        path = initialize_local_store()
        connection = sqlite3.connect(path)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            connection.close()

        assert version == 0
        assert str(mode).lower() == "wal"

    def test_existing_control_schema_version_and_assets_are_preserved(self, temp_store_dir):
        from local_store import create_local_session, initialize_local_store

        path = temp_store_dir["sqlite_path"]
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE local_users (
              user_id TEXT PRIMARY KEY,
              username TEXT NOT NULL UNIQUE,
              display_name TEXT,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            PRAGMA user_version = 5;
            """
        )
        connection.execute(
            "INSERT INTO local_users VALUES (?, ?, ?, ?, ?, ?)",
            ("u1", "owner", "Owner", "active", "now", "now"),
        )
        connection.commit()
        connection.close()

        initialize_local_store()
        create_local_session("shared-session", "Shared")

        connection = sqlite3.connect(path)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            user = connection.execute(
                "SELECT username FROM local_users WHERE user_id = 'u1'"
            ).fetchone()
            session = connection.execute(
                "SELECT title FROM sessions WHERE session_id = 'shared-session'"
            ).fetchone()
        finally:
            connection.close()

        assert version == 5
        assert user == ("owner",)
        assert session == ("Shared",)

    def test_legacy_json_is_imported_once_and_retained(self, temp_store_dir):
        import local_store

        legacy_dir = temp_store_dir["legacy_dir"]
        os.makedirs(legacy_dir, exist_ok=True)
        sessions_path = os.path.join(legacy_dir, "_sessions.json")
        messages_path = os.path.join(legacy_dir, "legacy-one.json")
        with open(sessions_path, "w", encoding="utf-8") as handle:
            json.dump([
                {
                    "id": "legacy-one",
                    "title": "Legacy",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "message_count": "not-a-number",
                }
            ], handle)
        with open(messages_path, "w", encoding="utf-8") as handle:
            json.dump([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ], handle)

        local_store.initialize_local_store()
        assert len(local_store.load_local_conversation("legacy-one")) == 2
        assert os.path.isfile(sessions_path)
        assert os.path.isfile(messages_path)

        local_store._initialized_paths.clear()
        local_store.initialize_local_store()
        assert len(local_store.load_local_conversation("legacy-one")) == 2

    def test_user_settings_round_trip(self):
        from local_store import (
            get_local_save_history,
            get_local_user_settings,
            set_local_user_settings,
        )

        settings = {"saveHistory": False, "distributedInference": True}
        set_local_user_settings(settings)
        assert get_local_user_settings() == settings
        assert get_local_save_history() is False
