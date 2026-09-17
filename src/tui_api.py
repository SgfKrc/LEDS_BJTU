"""QLH 后端的最小 REST / SSE 客户端（**纯标准库**）。

从旧的 ``tui_admin.py`` 抽出的 ``ApiClient``，用途：

* 供 Textual 外壳（``src/tui_textual.py``）与后续的单命令薄层共用一份协议实现；
* 只依赖标准库（``urllib``/``json``），SSE 解析复用 ``src/tui_sse.py``（同样零依赖），
  因此**无 UI 依赖的环境**（Edge、CI）也能 import 本模块。

历史：``ApiClient`` 原在 ``tui_admin.py`` 内，随该文件的界面代码一起被视为可归档对象；
协议层不应与界面同生共死，故独立成模块。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, Optional

try:  # 同目录导入（python src/xxx.py）
    from tui_sse import SSEDecoder, decode_json_event
    from tui_shared import API_PATHS, build_interactive_request
except ImportError:  # pragma: no cover - 包导入路径
    from .tui_sse import SSEDecoder, decode_json_event  # type: ignore
    from .tui_shared import API_PATHS, build_interactive_request  # type: ignore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
BACKEND_HINT = "请先启动本机后端（python src/api_server.py），或用 qlh 自动启动。"


class ApiError(Exception):
    """后端返回的错误（含 HTTP 状态码，便于界面区分 4xx/5xx）。"""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class ApiClient:
    """与 FastAPI 后端通信的极简 REST 客户端（纯标准库）。"""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 5.0, log_token: str = "") -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.log_token = log_token

    @property
    def base_url(self) -> str:
        from network_address import build_url

        return build_url("http", self.host, self.port)

    # ------------------------------------------------------------ 底层请求

    def request(self, method: str, path: str, body=None, params=None,
                with_log_token: bool = False):
        url = self.base_url + "/api" + path
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
            if qs:
                url = url + "?" + qs
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if with_log_token and self.log_token:
            headers["X-QLH-Log-Token"] = self.log_token
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw) if raw else {}
                detail = parsed.get("detail", raw) if isinstance(parsed, dict) else raw
                if isinstance(detail, dict):
                    detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
            except Exception:  # noqa: BLE001 - 响应体不是 JSON 时保留状态码
                detail = ""
            raise ApiError("HTTP %d: %s" % (e.code, detail or e.reason), status=e.code)
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (TimeoutError, OSError)) and "timed out" in str(reason):
                raise ApiError("请求超时（>%.0fs）: %s" % (self.timeout, url))
            raise ApiError("无法连接后端 %s（%s）。%s" % (self.base_url, reason, BACKEND_HINT))
        except TimeoutError:
            raise ApiError("请求超时（>%.0fs）: %s" % (self.timeout, url))
        except OSError as e:
            raise ApiError("网络错误 %s: %s。%s" % (self.base_url, e, BACKEND_HINT))
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            return {"detail": text}

    def get(self, path, params=None, with_log_token: bool = False):
        return self.request("GET", path, params=params, with_log_token=with_log_token)

    def post(self, path, body=None, params=None):
        return self.request("POST", path, body=body, params=params)

    def put(self, path, body=None):
        return self.request("PUT", path, body=body)


def cancel_generation(api: ApiClient, generation_id: str) -> None:
    """取消某次生成；``generation_id`` 含 ``/`` 时必须 safe="" 编码。"""
    path = API_PATHS["chat_cancel"].format(
        generation_id=urllib.parse.quote(generation_id, safe=""))
    api.post(path)


def iter_chat_payloads(
    api: ApiClient,
    message: str,
    *,
    session_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    routing_preference: str = "auto",
    show_thinking: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    read_timeout: float = 60.0,
) -> Iterator[Dict[str, Any]]:
    """POST ``/api/chat/stream`` 并逐事件产出 JSON payload（供界面 worker 消费）。

    同步生成器：调用方负责放进线程/worker，避免阻塞 UI 事件循环。
    """
    body = build_interactive_request(
        message,
        session_id=session_id,
        generation_id=generation_id,
        routing_preference=routing_preference,
        show_thinking=show_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    url = api.base_url + "/api" + API_PATHS["chat_stream"]
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    decoder = SSEDecoder()
    with urllib.request.urlopen(req, timeout=read_timeout) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            for event in decoder.feed(chunk):
                payload = decode_json_event(event)
                if payload:
                    yield payload
