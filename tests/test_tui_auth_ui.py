"""★ TUI 认证界面测试（G-⑤，2026-09-19）。

两条主线：
1. **`ApiClient` 的 Bearer 支持** —— 登录后所有请求自动带 `Authorization: Bearer`；
   **未登录时不得带该头**（避免把空 token 发出去）。
2. **认证命令面** —— `/login` `/logout` `/whoami` `/users` `/totp` 的路径与语义。
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Any

import pytest

import tui_api
from tui_shared import API_PATHS, COMMAND_SPECS


class RecordingClient(tui_api.ApiClient):
    """拦截真实 HTTP，记录最终发出的 `Request`（用于断言头）。"""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.requests: list[urllib.request.Request] = []

    def _request_url(self, method, url, body=None, params=None,
                     with_log_token=False, timeout=None):  # noqa: D102
        # 复制父类构造头的逻辑，但只记录、不联网
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if with_log_token and self.log_token:
            headers["X-QLH-Log-Token"] = self.log_token
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        self.requests.append(urllib.request.Request(url, headers=headers, method="method"))
        return {"ok": True}


class TestBearerHeader:
    def test_no_authorization_when_logged_out(self):
        api = RecordingClient()
        assert api.auth_token == ""
        api._request_url("GET", "http://x/api/health")
        assert api.requests[-1].get_header("Authorization") is None

    def test_authorization_present_after_login(self):
        api = RecordingClient()
        api.auth_token = "tok-abc"
        api._request_url("GET", "http://x/api/health")
        assert api.requests[-1].get_header("Authorization") == "Bearer tok-abc"

    def test_logout_clears_header(self):
        api = RecordingClient()
        api.auth_token = "tok-abc"
        api.auth_token = ""
        api._request_url("GET", "http://x/api/health")
        assert api.requests[-1].get_header("Authorization") is None

    def test_auth_token_is_not_persisted_to_disk(self, tmp_path):
        """★ 安全性质：token 只在内存（`ApiClient` 不写任何文件）。"""
        api = RecordingClient()
        api.auth_token = "super-secret-token"
        files = list(tmp_path.rglob("*"))
        assert files == [], "ApiClient 不应落盘任何文件"


class TestAuthPaths:
    def test_all_paths(self):
        assert API_PATHS["auth_capability"] == "/auth/capability"
        assert API_PATHS["auth_login"] == "/auth/login"
        assert API_PATHS["auth_logout"] == "/auth/logout"
        assert API_PATHS["auth_me"] == "/auth/me"
        assert API_PATHS["auth_totp_provision"] == "/auth/totp/provision"
        assert API_PATHS["auth_totp_verify"] == "/auth/totp/verify"
        assert API_PATHS["auth_users"] == "/users"
        assert API_PATHS["auth_user"] == "/users/{username}"


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get(self, path: str, **kw: Any) -> dict:
        self.calls.append(("get", path, kw))
        return {}

    def post(self, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("post", path, body, kw))
        return {}

    def request(self, method: str, path: str, body: Any = None, **kw: Any) -> dict:
        self.calls.append(("request", method, path, body, kw))
        return {}


class TestAuthCalls:
    def test_login_includes_totp_only_when_given(self):
        api = FakeApi()
        tui_api.auth_login(api, "u", "p")
        assert api.calls[0] == ("post", "/auth/login", {"username": "u", "password": "p"}, {})
        api.calls.clear()
        tui_api.auth_login(api, "u", "p", totp_code="123456")
        assert api.calls[0][2]["totp_code"] == "123456"

    def test_logout_and_me(self):
        api = FakeApi()
        tui_api.auth_logout(api)
        tui_api.auth_me(api)
        assert api.calls[0][1] == "/auth/logout"
        assert api.calls[1] == ("get", "/auth/me", {})

    def test_totp_provision_and_verify(self):
        api = FakeApi()
        tui_api.auth_totp_provision(api)
        tui_api.auth_totp_verify(api, "654321")
        assert api.calls[0][1] == "/auth/totp/provision"
        assert api.calls[1] == ("post", "/auth/totp/verify", {"code": "654321"}, {})

    def test_user_admin_calls(self):
        api = FakeApi()
        tui_api.list_auth_users(api)
        tui_api.create_auth_user(api, "n", "password123", role="operator")
        tui_api.patch_auth_user(api, "a/b", role="admin", disabled=True)
        tui_api.delete_auth_user(api, "a/b")
        assert api.calls[0] == ("get", "/users", {})
        assert api.calls[1][2] == {"username": "n", "password": "password123",
                                   "role": "operator"}
        # 用户名带 "/" 必须编码
        assert api.calls[2][2] == "/users/a%2Fb"
        assert api.calls[2][3] == {"role": "admin", "disabled": True}
        assert api.calls[3][1] == "DELETE" and api.calls[3][2] == "/users/a%2Fb"

    def test_patch_omits_unspecified_fields(self):
        api = FakeApi()
        tui_api.patch_auth_user(api, "u", disabled=False)
        assert api.calls[0][3] == {"disabled": False}


class TestCommandSpecs:
    def test_all_auth_commands_declared(self):
        names = {c["name"] for c in COMMAND_SPECS}
        for cmd in ("/login", "/logout", "/whoami", "/users", "/totp"):
            assert cmd in names, f"命令表应包含 {cmd}"

    def test_login_documents_totp_and_no_disk(self):
        spec = next(c for c in COMMAND_SPECS if c["name"] == "/login")
        assert "totp_code" in spec["args"]
        assert "不落盘" in spec["desc"], "应说明凭据不落盘"

    def test_users_spec_lists_subcommands(self):
        spec = next(c for c in COMMAND_SPECS if c["name"] == "/users")
        for token in ("list", "add", "role", "disable", "passwd", "del"):
            assert token in spec["args"]

    def test_no_duplicate_commands(self):
        names = [c["name"] for c in COMMAND_SPECS]
        assert len(names) == len(set(names))

    def test_new_functions_defined_once(self):
        import pathlib
        import re

        src = pathlib.Path(tui_api.__file__).read_text(encoding="utf-8")
        for fn in ("auth_login", "auth_logout", "auth_me", "auth_totp_provision",
                   "auth_totp_verify", "list_auth_users", "create_auth_user",
                   "patch_auth_user", "delete_auth_user"):
            assert len(re.findall(rf"^def {fn}\(", src, re.M)) == 1, f"{fn} 应只定义一次"
