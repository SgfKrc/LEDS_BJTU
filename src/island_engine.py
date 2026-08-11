"""
TP 孤岛推理引擎 — OpenAI 兼容端点的网关侧适配层（路线 A PoC）
==============================================================
职责:
1. 将同构 GPU 子集群（vLLM / SGLang / llama.cpp rpc-server 等张量并行孤岛）
   暴露的 OpenAI 兼容端点封装为本地推理引擎
2. ChatML 语义对话补全（对话模板由孤岛后端自行套用）
3. 流式输出（SSE）+ 非流式补全
4. 与 LlamaCppEngine 接口对齐，上游调用者（ModelManager / api_server）无需修改

设计要点（对应《张量并行外部辅助与混合拆分调研方案》§2.1 路线 A）:
  - 孤岛内部的张量并行对 QLH 完全透明——本引擎只做整请求转发
  - 网关节点行为与 llama.cpp 全模型推理节点一致：不参与 PyTorch 层拆分
  - "加载模型" = 健康检查（GET /v1/models）+ 解析后端模型名，无本地文件
  - 凭据（api_key、URL 内嵌账号）在日志与状态上报中一律脱敏

依赖: httpx（同步客户端，已在 requirements.txt）

配置（config.py / 环境变量）:
  QLH_ISLAND_BASE_URL  孤岛 OpenAI 兼容端点，如 http://10.0.0.2:8000
  QLH_ISLAND_API_KEY   可选 Bearer 凭据
  QLH_ISLAND_MODEL     可选模型名；为空时自动取 /v1/models 首个模型
  QLH_ISLAND_TIMEOUT   请求超时（秒）
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


# ================================================================
# 错误分类 — 全部继承 RuntimeError，消息为面向用户的中文描述
# ================================================================

class IslandEngineError(RuntimeError):
    """孤岛引擎统一错误基类。"""


class IslandUnreachableError(IslandEngineError):
    """孤岛后端不可达（连接被拒绝 / DNS 失败 / 网络中断）。"""


class IslandTimeoutError(IslandEngineError):
    """孤岛后端响应超时。"""


class IslandHTTPError(IslandEngineError):
    """孤岛后端返回 HTTP 错误状态码。"""


class IslandStreamInterruptedError(IslandEngineError):
    """孤岛流式响应在传输中途中断。"""


def mask_island_url(url: str) -> str:
    """脱敏孤岛 URL：去掉内嵌账号密码与查询串，仅保留 scheme://host:port/path。"""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            # 去掉 user:pass@ 前缀
            netloc = netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))
    except (ValueError, AttributeError):
        return "<无效 URL>"


def _classify_httpx_error(exc: Exception, base_url_masked: str,
                          streaming: bool = False) -> IslandEngineError:
    """将 httpx 异常映射为孤岛错误分类（消息不泄露凭据与原始堆栈）。"""
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        if streaming:
            return IslandStreamInterruptedError(
                f"孤岛流式响应中断：等待后端 {base_url_masked} 的下一段数据超时。"
                f"请检查孤岛内网负载或调大 QLH_ISLAND_TIMEOUT。"
            )
        return IslandTimeoutError(
            f"孤岛后端超时：{base_url_masked} 未在限定时间内响应。"
            f"请确认孤岛推理服务负载正常，或调大 QLH_ISLAND_TIMEOUT。"
        )
    if isinstance(exc, httpx.ConnectError):
        return IslandUnreachableError(
            f"孤岛后端不可达：无法连接 {base_url_masked}。"
            f"请确认孤岛推理服务已启动且网关与孤岛网络互通。"
        )
    if isinstance(exc, httpx.RequestError):
        if streaming:
            return IslandStreamInterruptedError(
                f"孤岛流式响应中断：与 {base_url_masked} 的连接在传输中途断开。"
            )
        return IslandUnreachableError(
            f"孤岛后端通信失败：{base_url_masked}（{type(exc).__name__}）。"
        )
    return IslandEngineError(
        f"孤岛后端未知错误：{base_url_masked}（{type(exc).__name__}）。"
    )


class IslandEngine:
    """
    TP 孤岛推理引擎 — 面向 OpenAI 兼容端点的整请求转发。

    使用方式:

        engine = IslandEngine()
        engine.load_model()          # 读取 config.QLH_ISLAND_* 并做健康检查

        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好"},
        ]
        result = engine.chat(messages, max_tokens=512, temperature=0.7)
        print(result["content"])

        for chunk in engine.chat_stream(messages):
            print(chunk, end="", flush=True)
    """

    def __init__(self):
        self._client = None               # httpx.Client 实例
        self._base_url: str = ""          # 已剥离内嵌凭据的端点 URL
        self._basic_auth = None           # URL 内嵌账号密码 → httpx.BasicAuth
        self._api_key: str = ""
        self._model_name: str = ""        # 解析后的后端模型名
        self._timeout: float = 120.0      # 读/写超时（秒）
        self._connect_timeout: float = 5.0
        self._loaded: bool = False

    # ================================================================
    # 加载（= 健康检查 + 模型名解析，无本地文件）
    # ================================================================

    def load_model(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        timeout: float = None,
        connect_timeout: float = None,
        **kwargs,
    ) -> None:
        """
        "加载"孤岛引擎：建立 HTTP 客户端并对孤岛端点做健康检查。

        Args:
            base_url: 孤岛 OpenAI 兼容端点，默认 config.ISLAND_BASE_URL
            api_key: 可选 Bearer 凭据，默认 config.ISLAND_API_KEY
            model: 后端模型名；为空时自动取 GET /v1/models 的首个模型
            timeout: 请求超时（秒），默认 config.ISLAND_TIMEOUT
            connect_timeout: 连接超时（秒），默认 config.ISLAND_CONNECT_TIMEOUT
            **kwargs: 预留（与 LlamaCppEngine.load_model 的 **kwargs 对齐）
        """
        import httpx
        import config as _cfg

        self._base_url = (base_url or getattr(_cfg, "ISLAND_BASE_URL", "")).rstrip("/")
        # URL 内嵌账号密码（http://user:pass@host）必须在此剥离：
        # httpx 的 INFO 请求日志会原样打印请求 URL，凭据留在 URL 里会泄露到
        # 网关日志。剥离后转成 httpx.BasicAuth 传给客户端，线上行为不变
        # （httpx 对 userinfo 同样是发 Basic 认证头）。
        self._basic_auth = None
        try:
            _parts = urlsplit(self._base_url)
        except ValueError:
            _parts = None
        if _parts is not None and "@" in _parts.netloc:
            from urllib.parse import unquote as _unquote
            _userinfo, _host = _parts.netloc.rsplit("@", 1)
            _user, _, _password = _userinfo.partition(":")
            self._basic_auth = (_unquote(_user), _unquote(_password))
            self._base_url = urlunsplit((
                _parts.scheme, _host, _parts.path, _parts.query, _parts.fragment,
            )).rstrip("/")
        self._api_key = api_key if api_key is not None else getattr(_cfg, "ISLAND_API_KEY", "")
        self._model_name = model if model is not None else getattr(_cfg, "ISLAND_MODEL", "")
        self._timeout = float(timeout or getattr(_cfg, "ISLAND_TIMEOUT", 120))
        self._connect_timeout = float(
            connect_timeout or getattr(_cfg, "ISLAND_CONNECT_TIMEOUT", 5)
        )

        if not self._base_url:
            raise IslandEngineError(
                "孤岛端点未配置：请设置 QLH_ISLAND_BASE_URL "
                "（如 http://10.0.0.2:8000）后重试。"
            )

        masked = self.masked_base_url
        logger.info(f"连接 TP 孤岛后端: {masked}")
        logger.info(f"  请求超时: {self._timeout:.0f}s | 连接超时: {self._connect_timeout:.0f}s")
        if self._basic_auth is not None and self._api_key:
            # HTTP 只允许一个 Authorization 头：URL 内嵌凭据会被 httpx 用
            # Basic 覆盖掉 _headers() 设置的 Bearer，API Key 实际发不出去。
            # 两者不可兼得，这里显式告警，避免后端 401 时无从排查。
            logger.warning(
                "同时配置了 URL 内嵌凭据和 API Key，但 HTTP 仅允许一个 "
                "Authorization 头：本次将发送 Basic 凭据，API Key 不会发送。"
                "若后端校验的是 API Key，请从 base_url 中去掉 user:pass。"
            )

        t0 = time.perf_counter()
        self._client = httpx.Client(
            timeout=httpx.Timeout(self._timeout, connect=self._connect_timeout),
            auth=(
                httpx.BasicAuth(*self._basic_auth)
                if self._basic_auth is not None else None
            ),
        )

        try:
            models = self._list_backend_models()
        except IslandEngineError:
            self._close_client()
            self._loaded = False
            raise

        if not self._model_name:
            if not models:
                self._close_client()
                self._loaded = False
                raise IslandEngineError(
                    f"孤岛后端 {masked} 的 /v1/models 未返回任何模型，"
                    f"且未通过 QLH_ISLAND_MODEL 显式指定模型名。"
                )
            self._model_name = models[0]
            logger.info(f"  自动发现孤岛模型: {self._model_name}")
        elif models and self._model_name not in models:
            # 不中断：部分后端 /v1/models 返回别名，以显式配置为准
            logger.warning(
                f"配置的孤岛模型 '{self._model_name}' 不在后端模型列表 {models} 中，"
                f"仍按配置值转发"
            )

        self._loaded = True
        logger.info(
            f"孤岛引擎就绪 ({time.perf_counter() - t0:.1f}s): "
            f"endpoint={masked}, model={self._model_name}"
        )

    def _list_backend_models(self) -> List[str]:
        """GET /v1/models — 健康检查兼模型发现。"""
        import httpx

        masked = self.masked_base_url
        try:
            response = self._client.get(
                f"{self._base_url}/v1/models", headers=self._headers(),
            )
        except Exception as exc:  # httpx 异常统一分类
            raise _classify_httpx_error(exc, masked) from None
        if response.status_code != 200:
            raise IslandHTTPError(
                f"孤岛后端 HTTP 错误：GET {masked}/v1/models "
                f"返回状态码 {response.status_code}。"
            )
        try:
            payload = response.json()
            data = payload.get("data", [])
            return [
                str(item.get("id", ""))
                for item in data
                if isinstance(item, dict) and item.get("id")
            ]
        except (ValueError, AttributeError):
            raise IslandHTTPError(
                f"孤岛后端响应格式异常：GET {masked}/v1/models 返回的不是有效 JSON。"
            ) from None

    def unload(self) -> None:
        """断开孤岛连接，释放 HTTP 客户端。"""
        self._close_client()
        self._loaded = False
        logger.info("孤岛引擎已断开")

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._client is not None

    @property
    def masked_base_url(self) -> str:
        """脱敏后的孤岛端点（用于日志与状态上报）。"""
        return mask_island_url(self._base_url)

    @property
    def model_name(self) -> str:
        """解析后的孤岛后端模型名。"""
        return self._model_name

    # ================================================================
    # 对话补全（对齐 LlamaCppEngine 接口）
    # ================================================================

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request_payload(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str] = None,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        # 注意：不注入 ChatML 停止词——孤岛后端自行套用其对话模板；
        # 仅透传调用方显式给出的 stop（OpenAI 语义上限 4 个）。
        if stop:
            payload["stop"] = list(stop)[:4]
        if reasoning_effort in {"none", "minimal", "low", "medium", "high"}:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        对话补全，返回完整结果（与 LlamaCppEngine.chat 返回结构一致）。

        Args:
            messages: [{"role": "user/assistant/system", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度 (0-2)
            top_p: nucleus sampling
            stop: 停止词列表

        Returns:
            {
                "content": "模型回复文本",
                "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": T},
                "model": "孤岛后端模型名",
                "finish_reason": "stop" | "length",
                "tokens_per_second": float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("孤岛引擎未连接，请先调用 load_model()")

        t0 = time.perf_counter()
        cancel_event = kwargs.pop("_cancel_event", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)

        if cancel_event is not None and cancel_event.is_set():
            return {
                "content": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "model": self._model_name,
                "finish_reason": "cancelled",
                "tokens_per_second": 0,
                "usage_estimated": True,
            }

        # 协作取消：OpenAI 端点无服务端取消原语，改走流式消费，
        # 在 chunk 边界断开；孤岛后端可能继续算完（best-effort，与
        # 调研方案 §2.2 的 cancel 缺口结论一致）。
        if cancel_event is not None:
            content_parts: List[str] = []
            finish_reason = "stop"
            chunk_count = 0
            cancelled = False
            usage: Dict[str, int] = {}
            try:
                for event in self._iter_stream_events(
                    messages, max_tokens, temperature, top_p, stop,
                    reasoning_effort=reasoning_effort,
                ):
                    text = event.get("content", "")
                    if text:
                        content_parts.append(text)
                        chunk_count += 1
                    if event.get("finish_reason"):
                        finish_reason = event["finish_reason"]
                    if event.get("usage"):
                        usage = event["usage"]
                    if cancel_event.is_set():
                        cancelled = True
                        break
            except IslandStreamInterruptedError:
                if not cancel_event.is_set():
                    raise
                cancelled = True

            content = "".join(content_parts)
            elapsed = time.perf_counter() - t0
            # 无本地 tokenizer：取消场景用 chunk 数估算生成 token 数
            completion_tokens = int(usage.get("completion_tokens", 0) or chunk_count)
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(
                        usage.get("total_tokens", 0) or completion_tokens
                    ),
                },
                "model": self._model_name,
                "finish_reason": "cancelled" if cancelled else finish_reason,
                "tokens_per_second": round(
                    completion_tokens / elapsed if elapsed > 0 else 0, 1,
                ),
                "usage_estimated": not bool(usage),
            }

        # ---- 非流式路径 ----
        masked = self.masked_base_url
        payload = self._request_payload(
            messages, max_tokens, temperature, top_p, stop, stream=False,
            reasoning_effort=reasoning_effort,
        )
        try:
            response = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            )
        except Exception as exc:
            raise _classify_httpx_error(exc, masked) from None

        if response.status_code != 200:
            detail = ""
            try:
                detail = str(response.json().get("error", {}).get("message", ""))[:200]
            except Exception:
                pass
            raise IslandHTTPError(
                f"孤岛后端 HTTP 错误：POST {masked}/v1/chat/completions "
                f"返回状态码 {response.status_code}"
                + (f"（{detail}）" if detail else "。")
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice.get("message", {}).get("content", "") or ""
            finish_reason = choice.get("finish_reason") or "stop"
            usage = body.get("usage", {}) or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise IslandHTTPError(
                f"孤岛后端响应格式异常：{masked}/v1/chat/completions "
                f"返回内容无法按 OpenAI 格式解析。"
            ) from None

        elapsed = time.perf_counter() - t0
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        # tok/s 优先取后端计时（llama.cpp server 的 timings.predicted_per_second），
        # 否则用 usage / 本地耗时计算
        timings = body.get("timings", {}) or {}
        tok_per_sec = float(timings.get("predicted_per_second", 0) or 0)
        if tok_per_sec <= 0:
            tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"孤岛推理完成: {completion_tokens} tokens / {elapsed:.1f}s "
            f"= {tok_per_sec:.1f} tok/s"
        )

        return {
            "content": content,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "model": str(body.get("model", "") or self._model_name),
            "finish_reason": finish_reason,
            "tokens_per_second": round(tok_per_sec, 1),
        }

    def post_json(
        self, path: str, payload: Dict[str, Any],
    ) -> tuple:
        """
        POST 任意 OpenAI 兼容子路径并返回 (解析后的 JSON, 响应体字节数)。

        存在的理由：路线 C-1 投机解码需要 POST /v1/completions 取
        per-token logprobs（见 speculative.py）。让它复用本引擎的 HTTP
        客户端，凭据脱敏、URL 内嵌账号→BasicAuth、超时与错误分类就全部
        免费继承，不必另起一套 HTTP 栈——这与 external_provider 组合复用
        本引擎的思路一致。响应体字节数用于投机解码的每轮通信量指标。

        Args:
            path: 以 "/" 开头的子路径，如 "/v1/completions"
            payload: 请求 JSON

        Returns:
            (响应 JSON dict, 响应体字节数)
        """
        if not self.is_loaded:
            raise RuntimeError("孤岛引擎未连接，请先调用 load_model()")
        masked = self.masked_base_url
        url_path = path if str(path).startswith("/") else f"/{path}"
        try:
            response = self._client.post(
                f"{self._base_url}{url_path}",
                json=payload,
                headers=self._headers(),
            )
        except Exception as exc:
            raise _classify_httpx_error(exc, masked) from None
        if response.status_code != 200:
            raise IslandHTTPError(
                f"孤岛后端 HTTP 错误：POST {masked}{url_path} "
                f"返回状态码 {response.status_code}。"
            )
        try:
            return response.json(), len(response.content or b"")
        except (ValueError, AttributeError):
            raise IslandHTTPError(
                f"孤岛后端响应格式异常：POST {masked}{url_path} "
                f"返回的不是有效 JSON。"
            ) from None

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: List[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式对话补全，逐步 yield 增量文本 chunk（与 LlamaCppEngine 一致）。

        Yields:
            str: 增量文本 chunk
        """
        if not self.is_loaded:
            raise RuntimeError("孤岛引擎未连接，请先调用 load_model()")

        reasoning_effort = kwargs.pop("reasoning_effort", None)

        for event in self._iter_stream_events(
            messages, max_tokens, temperature, top_p, stop,
            reasoning_effort=reasoning_effort,
        ):
            content = event.get("content", "")
            if content:
                yield content

    def _iter_stream_events(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        内部 SSE 解析器：POST stream=true 并逐事件产出
        {"content": str, "finish_reason": str | None, "usage": dict | None}。
        """
        masked = self.masked_base_url
        payload = self._request_payload(
            messages, max_tokens, temperature, top_p, stop, stream=True,
            reasoning_effort=reasoning_effort,
        )

        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            )
        except Exception as exc:
            raise _classify_httpx_error(exc, masked) from None

        done = False
        try:
            with stream_ctx as response:
                if response.status_code != 200:
                    try:
                        response.read()
                    except Exception:
                        pass
                    raise IslandHTTPError(
                        f"孤岛后端 HTTP 错误：流式请求 {masked}/v1/chat/completions "
                        f"返回状态码 {response.status_code}。"
                    )
                for line in response.iter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        raise IslandStreamInterruptedError(
                            f"孤岛流式响应中断：{masked} 返回了无法解析的 SSE 数据块。"
                        ) from None
                    choices = chunk.get("choices", [])
                    delta: Dict[str, Any] = {}
                    finish_reason = None
                    if choices:
                        delta = choices[0].get("delta", {}) or {}
                        finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        done = True
                    yield {
                        "content": delta.get("content", "") or "",
                        "finish_reason": finish_reason,
                        "usage": chunk.get("usage") or None,
                    }
        except IslandEngineError:
            raise
        except Exception as exc:
            raise _classify_httpx_error(exc, masked, streaming=True) from None

        if not done:
            raise IslandStreamInterruptedError(
                f"孤岛流式响应中断：{masked} 未发送结束标记（[DONE]）即关闭了连接。"
            )

    # ================================================================
    # 工具方法（对齐 LlamaCppEngine）
    # ================================================================

    def get_memory_usage(self) -> dict:
        """获取网关本机内存占用估算（孤岛显存不在此统计）。"""
        import psutil
        mem = psutil.virtual_memory()
        process = psutil.Process()
        proc_mem = process.memory_info().rss / (1024 ** 3)
        return {
            "process_gb": round(proc_mem, 2),
            "system_available_gb": round(mem.available / (1024 ** 3), 1),
            "system_percent": mem.percent,
        }

    def get_model_info(self) -> dict:
        """获取孤岛引擎基本信息（凭据已脱敏）。"""
        import config as _cfg

        info = {
            "engine": "island",
            "base_url": self.masked_base_url,
            "model": self._model_name,
            "backend": getattr(_cfg, "ISLAND_BACKEND", ""),
            "tp_size": getattr(_cfg, "ISLAND_TP_SIZE", 1),
            "gpu_count": getattr(_cfg, "ISLAND_GPU_COUNT", 1),
            "vram_gb": getattr(_cfg, "ISLAND_VRAM_GB", 0.0),
            "timeout": self._timeout,
            "loaded": self._loaded,
        }
        if self._loaded:
            info["memory"] = self.get_memory_usage()
        return info

    def reset_kv_cache(self) -> None:
        """
        清空 KV 缓存（接口兼容预留）。

        OpenAI 兼容端点无跨请求 KV 状态承诺（vLLM/SGLang 的前缀缓存
        由后端自行管理），此方法为 no-op。
        """
        logger.debug("KV cache reset (island — 后端自管理，no-op)")

    def tokenize(self, text: str) -> List[int]:
        """孤岛引擎无本地 tokenizer，不支持本地 tokenize。"""
        raise RuntimeError(
            "孤岛引擎不支持本地 tokenize：tokenizer 位于孤岛后端。"
        )

    def detokenize(self, tokens: List[int]) -> str:
        """孤岛引擎无本地 tokenizer，不支持本地 detokenize。"""
        raise RuntimeError(
            "孤岛引擎不支持本地 detokenize：tokenizer 位于孤岛后端。"
        )


# ================================================================
# 便捷函数
# ================================================================

def check_island_available() -> bool:
    """检测孤岛引擎是否已启用且配置了端点（不发起网络请求）。"""
    try:
        import config as _cfg
        return bool(
            getattr(_cfg, "ISLAND_ENABLED", False)
            and getattr(_cfg, "ISLAND_BASE_URL", "")
        )
    except Exception:
        return False
