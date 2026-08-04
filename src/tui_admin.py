#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLH 轻量化大模型分布式边缘推理优化系统 — TUI 管理菜单（终端版后台管理）
============================================================

与 Web 管理面板（AdminPanel）功能对应的跨平台终端管理界面。
仅使用 Python 标准库，无任何第三方依赖：

  - Windows 10+ : 通过 ctypes 启用 VT 转义序列（ANSI），键盘输入用 msvcrt
  - Linux/macOS : termios/tty + select 读取按键
  - 任何不支持 ANSI/非交互终端（如管道、CI）自动降级为纯文本编号菜单

用法:
    python src/tui_admin.py [--host 127.0.0.1] [--port 8000] [--plain]

    --host/--port  后端 API 地址（可管理 Tailscale 远程主节点，如 100.x.x.x）
    --plain        强制纯文本编号菜单模式（无 ANSI）
    --interval     仪表盘/节点/队列自动刷新间隔秒数（默认 3）
    --timeout      HTTP 请求超时秒数（默认 5）
    --log-token    远程访问日志接口所需的 X-QLH-Log-Token
    --no-color     关闭彩色输出

后端未启动时会给出提示：请先运行 `python src/api_server.py`。
"""

import argparse
import json
import os
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import ctypes
    try:
        import msvcrt
    except ImportError:      # 理论上 Windows 一定有
        msvcrt = None
else:
    import select
    import termios
    import tty

APP_TITLE = "QLH 分布式边缘推理 · TUI 管理菜单"
TUI_VERSION = "1.1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
BACKEND_HINT = "请先在项目根目录启动后端: python src/api_server.py （或在「设置」中修改后端地址）"

# 项目根目录（src/ 的上一级），用于本地日志目录定位
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ============================================================
# 一、CJK 显示宽度处理（East Asian Width）
# ============================================================

def char_width(ch: str) -> int:
    """单字符终端显示宽度：全角/宽字符=2，组合字符=0，其余=1。"""
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def disp_width(text: str) -> int:
    """字符串终端显示宽度（按列数）。"""
    return sum(char_width(ch) for ch in text)


def truncate_display(text: str, width: int) -> str:
    """按显示宽度截断，超长时以 … 结尾。"""
    if width <= 0:
        return ""
    if disp_width(text) <= width:
        return text
    take = max(width - 1, 0)
    out = []
    used = 0
    for ch in text:
        cw = char_width(ch)
        if used + cw > take:
            break
        out.append(ch)
        used += cw
    return "".join(out) + "…"


def pad_display(text: str, width: int, align: str = "left") -> str:
    """按显示宽度填充/截断到指定列宽（CJK 安全）。"""
    t = truncate_display(text, width)
    gap = width - disp_width(t)
    if gap <= 0:
        return t
    if align == "right":
        return " " * gap + t
    if align == "center":
        left = gap // 2
        return " " * left + t + " " * (gap - left)
    return t + " " * gap


def make_table(headers, rows, max_width: int = 0, sep: str = "  ") -> list:
    """生成对齐表格文本行（CJK 宽度安全）。返回 list[str]。"""
    cols = len(headers)
    srows = []
    for r in rows:
        cells = ["" if c is None else str(c) for c in r]
        while len(cells) < cols:
            cells.append("")
        srows.append(cells[:cols])
    widths = [disp_width(str(h)) for h in headers]
    for r in srows:
        for i in range(cols):
            widths[i] = max(widths[i], disp_width(r[i]))
    sep_w = disp_width(sep)
    if max_width and max_width > 10:
        total = sum(widths) + sep_w * (cols - 1)
        # 超宽时反复削减最宽列
        while total > max_width:
            i = widths.index(max(widths))
            if widths[i] <= 4:
                break
            widths[i] -= 1
            total -= 1
    lines = [sep.join(pad_display(str(headers[i]), widths[i]) for i in range(cols))]
    lines.append(sep.join("─" * widths[i] for i in range(cols)))
    for r in srows:
        lines.append(sep.join(pad_display(r[i], widths[i]) for i in range(cols)))
    return lines


# ============================================================
# 二、格式化小工具
# ============================================================

def fmt_age(ts) -> str:
    """时间戳 → 距今多久。"""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return "—"
    if ts <= 0:
        return "—"
    d = max(time.time() - ts, 0)
    if d < 60:
        return "%.0f秒前" % d
    if d < 3600:
        return "%.0f分前" % (d / 60)
    return "%.1f时前" % (d / 3600)


def fmt_bytes(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%.0f%s" if unit == "B" else "%.1f%s") % (n, unit)
        n /= 1024
    return "—"


def onoff(b) -> str:
    return "已启用" if b else "已停用"


def state_cn(state: str) -> str:
    return {
        "online": "在线", "offline": "离线", "busy": "忙碌",
        "error": "错误", "connecting": "连接中",
    }.get(str(state), str(state))


def role_cn(role: str) -> str:
    return {"master": "主节点", "client": "从节点", "unknown": "未确认"}.get(str(role), str(role))


# ============================================================
# 三、HTTP API 客户端（urllib 标准库）
# ============================================================

class ApiError(Exception):
    """后端 API 调用失败（含连接失败/超时/HTTP 错误）。"""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class ApiClient:
    """与 FastAPI 后端通信的极简 REST 客户端。"""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 5.0, log_token: str = ""):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.log_token = log_token

    @property
    def base_url(self) -> str:
        return "http://%s:%s" % (self.host, self.port)

    # ---- 底层请求 ----
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
            except Exception:
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

    def get(self, path, params=None, with_log_token=False):
        return self.request("GET", path, params=params, with_log_token=with_log_token)

    def post(self, path, body=None, params=None):
        return self.request("POST", path, body=body, params=params)

    def put(self, path, body=None):
        return self.request("PUT", path, body=body)

    def delete(self, path, with_log_token=False):
        return self.request("DELETE", path, with_log_token=with_log_token)


# ============================================================
# 四、终端层（ANSI + 跨平台按键读取）
# ============================================================

class TermNotCapable(Exception):
    """终端不支持交互式 ANSI 模式，需要降级为纯文本模式。"""


def enable_windows_vt() -> bool:
    """Windows 10+: 启用 VT 转义序列处理。成功返回 True。"""
    if not IS_WINDOWS:
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        STD_OUTPUT_HANDLE = -11
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return bool(kernel32.SetConsoleMode(
            handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


class AnsiTerm:
    """交互式终端：备用屏幕缓冲、原始按键、光标控制。"""

    def __init__(self, color: bool = True):
        self.color = color
        self._unix_saved = None
        self._active = False
        self._pending_keys = []

    # ---- 生命周期 ----
    def __enter__(self):
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise TermNotCapable("stdin/stdout 不是交互终端")
        if IS_WINDOWS:
            if msvcrt is None or not enable_windows_vt():
                raise TermNotCapable("无法启用 Windows VT 模式")
        else:
            try:
                self._unix_saved = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            except termios.error as e:
                raise TermNotCapable("termios 初始化失败: %s" % e)
        self._active = True
        self.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")   # 备用屏 + 隐藏光标
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.write("\x1b[?25h\x1b[?1049l")            # 恢复光标 + 主屏幕
        finally:
            self._restore_input()
            self._active = False
        return False

    def _restore_input(self):
        if not IS_WINDOWS and self._unix_saved is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._unix_saved)
            except termios.error:
                pass

    def _raw_input_mode(self):
        if not IS_WINDOWS:
            try:
                tty.setcbreak(sys.stdin.fileno())
            except termios.error:
                pass

    # ---- 输出 ----
    @staticmethod
    def write(s: str):
        sys.stdout.write(s)
        sys.stdout.flush()

    @staticmethod
    def size():
        try:
            sz = shutil.get_terminal_size(fallback=(100, 30))
            return max(sz.columns, 40), max(sz.lines, 10)
        except Exception:
            return 100, 30

    def paint(self, text: str, style: str) -> str:
        if not self.color or not style:
            return text
        codes = {
            "title": "1;36", "ok": "32", "err": "1;31", "warn": "33",
            "dim": "2", "sel": "7", "head": "1;37", "key": "36",
            "input": "36", "cmd": "1;33",
        }
        code = codes.get(style)
        if not code:
            return text
        return "\x1b[" + code + "m" + text + "\x1b[0m"

    # ---- 按键读取（带超时，None=超时无按键） ----
    def get_key(self, timeout: float = 0.3):
        if IS_WINDOWS:
            return self._get_key_windows(timeout)
        return self._get_key_unix(timeout)

    @staticmethod
    def _get_key_windows(timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    ch2 = msvcrt.getwch()
                    return {
                        "H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                        "I": "PGUP", "Q": "PGDN", "G": "HOME", "O": "END",
                    }.get(ch2)
                if ch in ("\r", "\n"):
                    return "ENTER"
                if ch == "\x1b":
                    return "ESC"
                if ch == "\x03":
                    raise KeyboardInterrupt
                return ch
            time.sleep(0.02)
        return None

    _ESC_MAP = {
        "[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT",
        "OA": "UP", "OB": "DOWN", "OC": "RIGHT", "OD": "LEFT",
        "[5~": "PGUP", "[6~": "PGDN", "[H": "HOME", "[F": "END",
        "[1~": "HOME", "[4~": "END",
    }

    def _get_key_unix(self, timeout: float):
        if self._pending_keys:
            return self._pending_keys.pop(0)
        fd = sys.stdin.fileno()
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return None
        try:
            data = os.read(fd, 128)
        except OSError:
            return "EOF"
        if not data:
            return "EOF"          # 终端关闭/输入流结束
        # 结尾恰好是 ESC 时稍等，避免转义序列被拆断
        if data.endswith(b"\x1b"):
            r2, _, _ = select.select([fd], [], [], 0.02)
            if r2:
                data += os.read(fd, 64)
        keys = self._parse_input_bytes(data)
        if not keys:
            return None
        first = keys.pop(0)
        self._pending_keys.extend(keys)
        return first

    @classmethod
    def _parse_input_bytes(cls, data: bytes) -> list:
        """把原始输入字节流解析为按键名列表（支持批量到达的按键）。"""
        keys = []
        i = 0
        n = len(data)
        while i < n:
            b = data[i]
            if b == 0x03:
                raise KeyboardInterrupt
            if b in (0x0D, 0x0A):
                keys.append("ENTER")
                i += 1
                continue
            if b == 0x1B:
                nxt = data[i + 1:i + 2]
                if nxt in (b"[", b"O"):
                    j = i + 2
                    while j < n and not (0x40 <= data[j] <= 0x7E):
                        j += 1
                    seq = data[i + 1:j + 1].decode("ascii", errors="ignore")
                    i = j + 1
                    mapped = cls._ESC_MAP.get(seq)
                    if mapped:
                        keys.append(mapped)
                else:
                    keys.append("ESC")
                    i += 1
                continue
            if b < 0x80:
                keys.append(chr(b))
                i += 1
                continue
            # UTF-8 多字节（用户误输入中文等）
            need = 2 if b < 0xE0 else (3 if b < 0xF0 else 4)
            ch = data[i:i + need].decode("utf-8", errors="ignore")
            if ch:
                keys.append(ch)
            i += need
        return keys

    # ---- 底部行输入（临时恢复正常行编辑） ----
    def prompt_line(self, label: str, default: str = ""):
        _, h = self.size()
        self.write("\x1b[%d;1H\x1b[K\x1b[?25h" % h)
        self._restore_input()
        try:
            tip = label
            if default:
                tip = "%s(默认 %s) " % (label, default)
            try:
                value = input(tip)
            except (EOFError, KeyboardInterrupt):
                return None
            value = value.strip()
            return value if value else (default if default else "")
        finally:
            self._raw_input_mode()
            self.write("\x1b[?25l")

    # ---- 分页文本输出（命令大结果 / 帮助，恢复行模式打印） ----
    def show_lines(self, lines: list, title: str = ""):
        """临时退出全屏输出多行文本，超出屏幕分页；q/ESC 提前返回。"""
        if not lines:
            lines = ["（无输出）"]
        w, h = self.size()
        page = max(h - 4, 5)
        self._restore_input()
        self.write("\x1b[?25h")
        try:
            pos = 0
            total = len(lines)
            while True:
                self.write("\x1b[2J\x1b[H")
                if title:
                    sys.stdout.write(title + "\n")
                chunk = lines[pos:pos + page]
                sys.stdout.write("\n".join(chunk))
                if chunk:
                    sys.stdout.write("\n")
                if pos + page < total:
                    sys.stdout.write(
                        "\n-- 第 %d-%d / %d 行：任意键下一页，q/ESC 返回 --\n"
                        % (pos + 1, pos + len(chunk), total))
                    sys.stdout.flush()
                    key = self._wait_any_key()
                    if key in ("q", "Q", "ESC"):
                        break
                    pos += page
                else:
                    sys.stdout.write("\n-- 已到末尾，按任意键返回 --\n")
                    sys.stdout.flush()
                    self._wait_any_key()
                    break
        finally:
            self._raw_input_mode()
            self.write("\x1b[?25l")

    def _wait_any_key(self) -> str:
        """阻塞等待任意按键（分页提示用，无操作超时后继续等待）。"""
        while True:
            key = self.get_key(300)
            if key is not None:
                return key


# ============================================================
# 五、交互抽象（供业务动作在两种模式下复用）
# ============================================================

class BaseUI:
    def prompt(self, label: str, default: str = ""):
        raise NotImplementedError

    def confirm(self, label: str) -> bool:
        ans = self.prompt(label + " [y/N] ")
        return (ans or "").strip().lower() in ("y", "yes", "是")


class TermUI(BaseUI):
    def __init__(self, term: AnsiTerm):
        self.term = term

    def prompt(self, label: str, default: str = ""):
        return self.term.prompt_line(label, default)


class PlainUI(BaseUI):
    def prompt(self, label: str, default: str = ""):
        tip = label
        if default:
            tip = "%s(默认 %s) " % (label, default)
        try:
            value = input(tip)
        except (EOFError, KeyboardInterrupt):
            raise
        value = value.strip()
        return value if value else (default if default else "")


# ============================================================
# 六、屏幕（Screen）框架与各管理屏幕
# ============================================================

class Screen:
    """一个管理屏幕：数据获取 + 文本渲染 + 快捷键动作。"""

    name = "屏幕"
    auto_refresh = False        # True 时使用 app.interval 周期刷新

    def __init__(self, app):
        self.app = app
        self.api = app.api
        self.data = None
        self.error = None
        self.last_fetch = 0.0

    def refresh(self, force: bool = False):
        if not force:
            if not self.auto_refresh and self.data is not None:
                return
            if self.auto_refresh and (time.monotonic() - self.last_fetch) < self.app.interval:
                return
        self.last_fetch = time.monotonic()
        try:
            self.data = self.fetch()
            self.error = None
        except ApiError as e:
            self.error = str(e)
        except Exception as e:                      # 防御式：绝不让界面崩溃
            self.error = "内部错误: %s" % e

    def fetch(self):
        return None

    def lines(self, width: int) -> list:
        """返回 [(style, text)] 或 [str]。"""
        return []

    def actions(self) -> list:
        """返回 [(key, label, handler(ui) -> str|None)]。"""
        return []

    # ---- 工具 ----
    def _sub(self, path, params=None, with_log_token=False):
        """屏幕内的附属请求：失败不致命，返回 (data, err)。"""
        try:
            return self.api.get(path, params=params, with_log_token=with_log_token), None
        except ApiError as e:
            return None, str(e)


def _norm_lines(raw) -> list:
    out = []
    for item in raw:
        if isinstance(item, tuple):
            out.append(item)
        else:
            out.append(("", str(item)))
    return out


# ------------------------------------------------------------
# 1. 系统状态总览
# ------------------------------------------------------------

class DashboardScreen(Screen):
    name = "系统状态总览"
    auto_refresh = True

    def fetch(self):
        health = self.api.get("/health")
        status = self.api.get("/status")
        model, model_err = self._sub("/models/current")
        role, role_err = self._sub("/cluster/my-role")
        return {"health": health, "status": status, "model": model,
                "model_err": model_err, "role": role, "role_err": role_err}

    def lines(self, width: int) -> list:
        d = self.data or {}
        st = d.get("status") or {}
        health = d.get("health") or {}
        model = d.get("model") or {}
        role = d.get("role") or {}
        out = []
        out.append(("head", "后端: %s    健康: %s    TUI v%s" % (
            self.api.base_url,
            "正常" if health.get("status") == "ok" else str(health.get("status", "未知")),
            TUI_VERSION)))
        out.append(("", ""))
        out.append(("title", "◆ 运行状态"))
        out.append(("", "  运行模式 RUN_MODE : %s" % st.get("run_mode", "—")))
        out.append(("", "  节点角色          : %s (%s)" % (
            role_cn(st.get("node_role", "—")), st.get("node_id", "—"))))
        out.append(("", "  最大节点数        : %s" % st.get("max_nodes", "—")))
        if role.get("is_provisional"):
            out.append(("warn", "  角色状态          : 待确认主节点（可切换为从节点加入现有集群）"))
        out.append(("", ""))
        out.append(("title", "◆ 模型"))
        if st.get("model_loaded"):
            out.append(("ok", "  模型已加载        : %s" % st.get("model_name", "—")))
            out.append(("", "  模型标识          : %s" % st.get("active_model_id", "—")))
            out.append(("", "  量化精度          : %s    推理引擎: %s    设备: %s" % (
                st.get("current_quant", "—"),
                model.get("engine", "—") or "—",
                model.get("device", "—"))))
            out.append(("", "  参数量            : %s    GPU 显存: %.2f GB" % (
                model.get("total_params", "—"),
                float(model.get("gpu_allocated_gb") or 0))))
        else:
            out.append(("warn", "  模型未加载（可在 Web 面板或 API 加载模型）"))
        gpu = st.get("gpu") or {}
        if gpu:
            out.append(("", "  GPU               : %s  %s/%s MB (%.1f%%)" % (
                gpu.get("name", "—"), gpu.get("allocated_mb", 0),
                gpu.get("total_mb", 0), float(gpu.get("utilization") or 0))))
        kv = st.get("kv_cache") or {}
        if kv:
            out.append(("", "  KV 缓存           : tokens=%s  pages=%s/%s  约 %s MB  对话轮次=%s" % (
                kv.get("total_tokens", 0), kv.get("allocated_pages", 0),
                kv.get("max_pages", 0), kv.get("estimated_memory_mb", 0),
                kv.get("rounds", 0))))
        dev = st.get("device") or {}
        if dev:
            out.append(("", ""))
            out.append(("title", "◆ 设备档位"))
            out.append(("", "  档位: %s (%s)    评分: %s" % (
                dev.get("tier_label", "—"), dev.get("tier", "—"), dev.get("score", "—"))))
            for w in (dev.get("warnings") or [])[:3]:
                out.append(("warn", "  警告: %s" % w))
        out.append(("", ""))
        out.append(("dim", "此屏幕每 %.0f 秒自动刷新（r 立即刷新）" % self.app.interval))
        return out

    def actions(self):
        return []


# ------------------------------------------------------------
# 2. 节点管理
# ------------------------------------------------------------

class NodesScreen(Screen):
    name = "节点管理"
    auto_refresh = True

    def fetch(self):
        nodes = self.api.get("/cluster/nodes")
        role, _ = self._sub("/cluster/my-role")
        role = role or {}
        result = {"nodes": nodes, "role": role}
        if role.get("is_master"):
            result["invite"], _ = self._sub("/cluster/invite")
            result["spare"], _ = self._sub("/cluster/spare-master")
        else:
            result["master_health"], _ = self._sub("/cluster/master-health")
        return result

    def lines(self, width: int) -> list:
        d = self.data or {}
        nd = d.get("nodes") or {}
        role = d.get("role") or {}
        out = []
        out.append(("head", "本机角色: %s (%s)    节点 %s 个 / 在线 %s / 离线 %s" % (
            role_cn(role.get("node_role", "—")), role.get("node_id", "—"),
            nd.get("count", 0), nd.get("online_count", 0), nd.get("offline_count", 0))))
        invite = d.get("invite") or {}
        if invite:
            out.append(("", "邀请信息(供从节点连接): %s:%s    容量 %s/%s    身份校验: %s" % (
                invite.get("master_host", "—"), invite.get("master_port", "—"),
                invite.get("node_count", "—"), invite.get("max_nodes", "—"),
                "通过" if invite.get("identity_verified", True) else "异常")))
        spare = (d.get("spare") or {}).get("spare_master")
        if spare:
            out.append(("", "备用主节点: %s (%s)  %s" % (
                spare.get("node_id", "—"), spare.get("hostname", "—"),
                "在线" if spare.get("is_online") else "离线")))
        mh = d.get("master_health")
        if mh is not None:
            style = "ok" if mh.get("master_online") else "err"
            out.append((style, "主节点健康: %s    地址 %s:%s    最近心跳 %s 秒前%s" % (
                "在线" if mh.get("master_online") else "离线/失联",
                mh.get("master_host", "—"), mh.get("master_port", "—"),
                mh.get("last_seen_seconds_ago", "—"),
                "（已过期）" if mh.get("stale") else "")))
        out.append(("", ""))
        rows = []
        for n in (nd.get("nodes") or []):
            rows.append([
                n.get("node_id", ""), role_cn(n.get("role", "")),
                n.get("node_type", ""), state_cn(n.get("state", "")),
                n.get("address", "") or "—", n.get("network_type", ""),
                "%.0f" % float(n.get("avg_rtt_ms") or 0),
                n.get("task_count", 0), n.get("error_count", 0),
                fmt_age(n.get("last_heartbeat")),
            ])
        if rows:
            for ln in make_table(
                    ["节点ID", "角色", "类型", "状态", "地址", "网络", "RTT", "任务", "错误", "心跳"],
                    rows, max_width=width - 2):
                out.append(("", ln))
        else:
            out.append(("dim", "（暂无节点记录）"))
        out.append(("", ""))
        out.append(("dim", "此屏幕每 %.0f 秒自动刷新" % self.app.interval))
        return out

    # ---- 动作 ----
    def actions(self):
        return [
            ("c", "连接主节点", self.act_connect),
            ("a", "手动注册节点", self.act_register),
            ("d", "注销节点", self.act_deregister),
            ("x", "删除节点记录", self.act_delete),
            ("t", "转让主节点", self.act_transfer),
            ("s", "设为备用主节点", self.act_spare_set),
            ("u", "移除备用主节点", self.act_spare_clear),
            ("m", "最大节点数", self.act_max_nodes),
            ("v", "自动发现主节点", self.act_discover),
            ("l", "转让日志", self.act_transfer_logs),
            ("e", "邮件告警测试", self.act_email_test),
            ("z", "重置主节点身份", self.act_reset_identity),
        ]

    def act_connect(self, ui: BaseUI):
        default_host = ""
        try:
            found = self.api.get("/cluster/discover")
            if found.get("found"):
                default_host = str(found.get("master_host") or "")
        except ApiError:
            pass
        host = ui.prompt("主节点 Tailscale IP: ", default_host)
        if not host:
            return "已取消"
        port = ui.prompt("主节点端口: ", "8888")
        try:
            port_i = int(port or "8888")
        except ValueError:
            return "端口无效: %s" % port
        switch = False
        role = (self.data or {}).get("role") or {}
        if role.get("runtime_node_role") == "master" or role.get("is_provisional") or role.get("is_master"):
            switch = ui.confirm("本机当前为主节点，切换为从节点后加入 %s:%s ?" % (host, port_i))
            if not switch:
                return "已取消（保持主节点身份）"
        r = self.api.post("/cluster/connect", {
            "master_host": host, "master_port": port_i, "switch_to_client": switch})
        return "连接结果: %s %s" % (r.get("status", "ok"), r.get("message", ""))

    def act_register(self, ui: BaseUI):
        node_id = ui.prompt("新节点 ID: ")
        if not node_id:
            return "已取消"
        hostname = ui.prompt("主机名(可空): ") or ""
        address = ui.prompt("预留地址 IP:Port (可空): ") or ""
        network = ui.prompt("网络类型 wifi/ethernet/unknown: ", "unknown") or "unknown"
        ntype = ui.prompt("平台 pc/android: ", "pc") or "pc"
        r = self.api.post("/cluster/nodes/register", {
            "node_id": node_id, "hostname": hostname, "address": address,
            "network_type": network, "node_type": ntype})
        return "注册结果: %s %s" % (r.get("status", "ok"), r.get("reason", "") or r.get("message", ""))

    def act_deregister(self, ui: BaseUI):
        node_id = ui.prompt("要注销的节点 ID: ")
        if not node_id:
            return "已取消"
        if not ui.confirm("确认强制注销节点 %s ?" % node_id):
            return "已取消"
        r = self.api.post("/cluster/nodes/%s/deregister" % urllib.parse.quote(node_id))
        return "节点 %s 已注销 (%s)" % (node_id, r.get("status", "ok"))

    def act_delete(self, ui: BaseUI):
        node_id = ui.prompt("要删除记录的节点 ID (仅限离线节点): ")
        if not node_id:
            return "已取消"
        if not ui.confirm("确认删除节点记录 %s ?" % node_id):
            return "已取消"
        r = self.api.delete("/cluster/nodes/%s" % urllib.parse.quote(node_id))
        return "删除结果: %s" % r.get("status", "ok")

    def act_transfer(self, ui: BaseUI):
        target = ui.prompt("转让给从节点 ID: ")
        if not target:
            return "已取消"
        if not ui.confirm("确认将主节点身份转让给 %s ? 转让后双方需重启生效" % target):
            return "已取消"
        r = self.api.post("/cluster/transfer-master", {"target_node_id": target})
        return "转让结果: %s %s" % (r.get("status", "ok"), r.get("message", ""))

    def act_spare_set(self, ui: BaseUI):
        target = ui.prompt("指定为备用主节点的从节点 ID: ")
        if not target:
            return "已取消"
        if not ui.confirm("确认指定 %s 为备用主节点?" % target):
            return "已取消"
        r = self.api.post("/cluster/spare-master", {"target_node_id": target})
        return "指定结果: %s %s" % (r.get("status", "ok"), r.get("message", ""))

    def act_spare_clear(self, ui: BaseUI):
        if not ui.confirm("确认移除当前备用主节点?"):
            return "已取消"
        r = self.api.delete("/cluster/spare-master")
        return "移除结果: %s %s" % (r.get("status", "ok"), r.get("message", ""))

    def act_max_nodes(self, ui: BaseUI):
        v = ui.prompt("新的最大节点数(1-64): ")
        if not v:
            return "已取消"
        try:
            n = int(v)
        except ValueError:
            return "无效数字: %s" % v
        r = self.api.put("/cluster/config/max-nodes", {"max_nodes": n})
        return "容量更新: %s" % r.get("status", "ok")

    def act_discover(self, ui: BaseUI):
        r = self.api.get("/cluster/discover")
        if r.get("found"):
            return "发现主节点 %s:%s (来源 %s%s)" % (
                r.get("master_host"), r.get("master_port"), r.get("source", "—"),
                ", 心跳过期" if r.get("stale") else "")
        return "数据库中未发现主节点记录"

    def act_transfer_logs(self, ui: BaseUI):
        r = self.api.get("/cluster/transfer-logs")
        logs = r.get("logs") or []
        if not logs:
            return "暂无角色转让日志"
        latest = logs[0] if isinstance(logs[0], dict) else {}
        return "共 %d 条转让日志，最近: %s %s→%s 关联 %s" % (
            r.get("count", len(logs)), latest.get("direction", "—"),
            latest.get("from_role", "—"), latest.get("to_role", "—"),
            latest.get("related_node", "—"))

    def act_email_test(self, ui: BaseUI):
        if not ui.confirm("发送一封 SMTP 测试邮件?"):
            return "已取消"
        r = self.api.post("/cluster/email-test")
        return "邮件测试: %s" % (r.get("message") or r.get("status", "已发送"))

    def act_reset_identity(self, ui: BaseUI):
        word = ui.prompt("危险操作！输入 reset 确认重置主节点 MAC 身份: ")
        if (word or "").strip().lower() != "reset":
            return "未确认，已取消"
        r = self.api.post("/cluster/reset-identity", {"confirm": "reset"})
        return "重置结果: %s（需重启后端生效）" % r.get("status", "ok")


# ------------------------------------------------------------
# 3. 分布式开关与模型分层
# ------------------------------------------------------------

class DistributedScreen(Screen):
    name = "分布式与分层"
    auto_refresh = False

    def fetch(self):
        di = self.api.get("/cluster/config/distributed-inference")
        layers, layers_err = self._sub("/cluster/layers")
        cfg, cfg_err = self._sub("/cluster/config")
        return {"di": di, "layers": layers, "layers_err": layers_err,
                "cfg": cfg, "cfg_err": cfg_err}

    def lines(self, width: int) -> list:
        d = self.data or {}
        di = d.get("di") or {}
        layers = d.get("layers") or {}
        cfg = d.get("cfg") or {}
        out = []
        style = "ok" if di.get("enabled") else "warn"
        out.append(("title", "◆ 分布式推理开关"))
        out.append((style, "  当前状态: %s    （配置默认值: %s）" % (
            onoff(di.get("enabled")), onoff(di.get("default")))))
        out.append(("", ""))
        out.append(("title", "◆ 模型分层"))
        out.append(("", "  总层数: %s    策略: %s    计算时间: %s" % (
            layers.get("total", "—"), layers.get("strategy", "—"),
            fmt_age(layers.get("computed_at")))))
        rows = []
        for a in (layers.get("assignments") or []):
            rows.append([
                a.get("node_id", ""), role_cn(a.get("role", "")),
                "%s-%s" % (a.get("start_layer", "?"), a.get("end_layer", "?")),
                "是" if a.get("has_embedding") else "否",
                "是" if a.get("has_lm_head") else "否",
                a.get("score", "—"),
            ])
        if rows:
            for ln in make_table(["节点", "角色", "层区间", "Embedding", "LM Head", "评分"],
                                 rows, max_width=width - 2):
                out.append(("", "  " + ln))
        else:
            out.append(("dim", "  （暂无分层分配，单机模式或未启用分布式）"))
        if d.get("layers_err"):
            out.append(("err", "  分层查询失败: %s" % d["layers_err"]))
        net = cfg.get("network") or {}
        model = cfg.get("model") or {}
        if cfg:
            out.append(("", ""))
            out.append(("title", "◆ 分布式配置"))
            out.append(("", "  主节点 TCP: %s:%s    心跳间隔: %ss" % (
                net.get("server_ip", "—"), net.get("server_port", "—"),
                net.get("heartbeat_interval_s", "—"))))
            out.append(("", "  量化: %s    KV 页大小: %s    最大页数: %s    最大序列: %s" % (
                model.get("quant_type", "—"), model.get("page_size", "—"),
                model.get("max_page_num", "—"), model.get("max_seq_len", "—"))))
        return out

    def actions(self):
        return [
            ("t", "切换分布式开关", self.act_toggle),
            ("o", "手动覆盖分层", self.act_override),
            ("R", "重置分层为自动", self.act_reset_layers),
        ]

    def act_toggle(self, ui: BaseUI):
        di = (self.data or {}).get("di") or {}
        target = not di.get("enabled")
        verb = "启用" if target else "停用"
        if not ui.confirm("确认%s分布式推理?" % verb):
            return "已取消"
        r = self.api.put("/cluster/config/distributed-inference", {"enabled": target})
        return "分布式推理已%s (%s)" % (verb, r.get("status", "ok"))

    def act_override(self, ui: BaseUI):
        total = ((self.data or {}).get("layers") or {}).get("total", 24)
        items = []
        while True:
            node_id = ui.prompt("第 %d 段 节点 ID (留空结束): " % (len(items) + 1))
            if not node_id:
                break
            start = ui.prompt("  起始层(含, 0-%s): " % total)
            end = ui.prompt("  结束层(不含, 1-%s): " % total)
            try:
                items.append({"node_id": node_id,
                              "start_layer": int(start), "end_layer": int(end)})
            except (TypeError, ValueError):
                return "层号无效，已取消"
        if not items:
            return "已取消（未输入分层）"
        desc = ", ".join("%s:%s-%s" % (i["node_id"], i["start_layer"], i["end_layer"])
                         for i in items)
        if not ui.confirm("确认覆盖分层为 [%s] ? 区间需从 0 连续覆盖到 %s" % (desc, total)):
            return "已取消"
        r = self.api.put("/cluster/layers", {"assignments": items})
        return "分层覆盖: %s %s" % (r.get("status", "ok"), r.get("message", ""))

    def act_reset_layers(self, ui: BaseUI):
        if not ui.confirm("确认清除手动分层，恢复自动(dynamic)策略?"):
            return "已取消"
        r = self.api.delete("/cluster/layers")
        return "分层已重置: %s" % r.get("status", "ok")


# ------------------------------------------------------------
# 4. 请求队列 / MLFQ
# ------------------------------------------------------------

class QueueScreen(Screen):
    name = "请求队列(MLFQ)"
    auto_refresh = True

    def fetch(self):
        return self.api.get("/cluster/queue")

    def lines(self, width: int) -> list:
        q = self.data or {}
        out = []
        paused = q.get("paused")
        out.append(("head", "调度策略: %s    队列状态: %s    执行中任务: %s" % (
            str(q.get("strategy", "—")).upper(),
            "已暂停" if paused else "接收中",
            q.get("current_task") or "无")))
        out.append(("", "排队总数: %s (上限 %s)    Q0交互:%s  Q1普通:%s  Q2批量:%s    已完成: %s" % (
            q.get("queue_size", 0), q.get("max_size", "—"),
            q.get("q0_depth", 0), q.get("q1_depth", 0), q.get("q2_depth", 0),
            q.get("completed_count", 0))))
        aging = q.get("aging_params") or {}
        out.append(("dim", "分级: Q0≤%stk  Q1≤%stk  Q2>%stk    老化: Q1→Q0 %ss, Q2→Q1 %ss" % (
            aging.get("q0_max_tokens", 128), aging.get("q1_max_tokens", 512),
            aging.get("q1_max_tokens", 512),
            aging.get("q1_to_q0_s", "—"), aging.get("q2_to_q1_s", "—"))))
        pre = q.get("preempt_stats") or {}
        out.append(("dim", "抢占: %s 次    累计开销: %s ms" % (
            pre.get("count", 0), pre.get("total_overhead_ms", 0))))
        out.append(("", ""))
        rows = []
        for level in ("q0", "q1", "q2"):
            for t in (q.get(level) or []):
                rows.append([
                    t.get("task_id", ""), level.upper(),
                    "Q%s" % t.get("original_level", "?"),
                    "%.0fs" % float(t.get("wait_seconds") or 0),
                    t.get("max_new_tokens", "—"),
                    "是" if t.get("is_aged") else "",
                    (t.get("session_id") or "")[:12],
                ])
        if rows:
            for ln in make_table(["任务ID", "级别", "初始", "等待", "tokens", "老化", "会话"],
                                 rows, max_width=width - 2):
                out.append(("", ln))
        else:
            out.append(("dim", "（队列为空）"))
        out.append(("", ""))
        out.append(("dim", "此屏幕每 %.0f 秒自动刷新（仅主节点可见/可管理队列）" % self.app.interval))
        return out

    def actions(self):
        return [
            ("s", "切换调度策略", self.act_strategy),
            ("p", "暂停队列", self.act_pause),
            ("u", "恢复队列", self.act_resume),
            ("C", "清空队列", self.act_clear),
            ("k", "取消任务", self.act_cancel),
        ]

    def act_strategy(self, ui: BaseUI):
        cur = (self.data or {}).get("strategy", "mlfq")
        s = ui.prompt("调度策略 fifo/mlfq: ", "fifo" if cur == "mlfq" else "mlfq")
        if s not in ("fifo", "mlfq"):
            return "无效策略: %s" % s
        r = self.api.post("/cluster/queue/strategy", {"strategy": s})
        return "策略已切换为 %s" % r.get("strategy", s)

    def act_pause(self, ui: BaseUI):
        self.api.post("/cluster/queue/pause")
        return "队列已暂停（不再接收新请求）"

    def act_resume(self, ui: BaseUI):
        self.api.post("/cluster/queue/resume")
        return "队列已恢复"

    def act_clear(self, ui: BaseUI):
        if not ui.confirm("确认清空所有排队任务(不影响执行中任务)?"):
            return "已取消"
        r = self.api.post("/cluster/queue/clear")
        return "已清空 %s 个排队任务" % r.get("cleared", 0)

    def act_cancel(self, ui: BaseUI):
        task_id = ui.prompt("要取消的任务 ID: ")
        if not task_id:
            return "已取消"
        if not ui.confirm("确认取消任务 %s ?" % task_id):
            return "已取消"
        r = self.api.delete("/cluster/queue/task/%s" % urllib.parse.quote(task_id))
        return r.get("message") or ("取消%s" % ("成功" if r.get("success") else "失败"))


# ------------------------------------------------------------
# 5. 设备画像
# ------------------------------------------------------------

class DeviceScreen(Screen):
    name = "设备画像"
    auto_refresh = False

    def fetch(self):
        return self.api.get("/device/profile")

    def lines(self, width: int) -> list:
        p = self.data or {}
        out = []
        out.append(("title", "◆ 本机设备画像（后端所在机器）"))
        os_info = p.get("os") or {}
        if isinstance(os_info, dict):
            os_text = "%s %s" % (os_info.get("system", ""), os_info.get("release", ""))
        else:
            os_text = str(os_info)
        out.append(("", "  操作系统: %s    主机名: %s" % (
            os_text.strip() or "—", p.get("hostname", "—"))))
        cpu = p.get("cpu") or {}
        out.append(("", "  CPU     : %s    物理核: %s  逻辑核: %s" % (
            cpu.get("model", cpu.get("brand", "—")),
            cpu.get("physical_cores", "—"), cpu.get("logical_cores", "—"))))
        ram = p.get("ram") or p.get("memory") or {}
        out.append(("", "  内存    : 总量 %s GB    可用 %s GB" % (
            ram.get("total_gb", "—"), ram.get("available_gb", "—"))))
        disk = p.get("disk") or {}
        if disk:
            out.append(("", "  磁盘    : 剩余 %s GB / 总 %s GB" % (
                disk.get("free_gb", "—"), disk.get("total_gb", "—"))))
        gpus = p.get("gpus") or []
        sel = p.get("selected_gpu_index", 0)
        out.append(("", ""))
        out.append(("title", "◆ GPU 列表（当前选中 #%s）" % sel))
        rows = []
        for i, g in enumerate(gpus):
            rows.append([
                ("*%d" % i) if i == sel else str(i),
                g.get("name", ""), g.get("gpu_type", ""),
                "支持" if g.get("cuda_available") else "不支持",
                g.get("vram_total_gb", "—"),
            ])
        if rows:
            for ln in make_table(["#", "名称", "类型", "CUDA", "显存GB"], rows,
                                 max_width=width - 2):
                out.append(("", "  " + ln))
        else:
            out.append(("dim", "  （未检测到 GPU）"))
        out.append(("", ""))
        out.append(("title", "◆ 档位评估"))
        out.append(("", "  档位: %s (%s)    总评分: %s" % (
            p.get("tier_label", "—"), p.get("tier", "—"), p.get("score_total", "—"))))
        for r in (p.get("recommendations") or [])[:5]:
            out.append(("ok", "  建议: %s" % r))
        for w in (p.get("warnings") or [])[:5]:
            out.append(("warn", "  警告: %s" % w))
        return out

    def actions(self):
        return [
            ("g", "切换推理GPU", self.act_select_gpu),
            ("A", "应用自动配置", self.act_auto_configure),
        ]

    def act_select_gpu(self, ui: BaseUI):
        idx = ui.prompt("切换到 GPU 序号: ")
        if not idx:
            return "已取消"
        try:
            n = int(idx)
        except ValueError:
            return "无效序号: %s" % idx
        if not ui.confirm("确认切换到 GPU #%d ? 切换后需重新加载模型" % n):
            return "已取消"
        r = self.api.post("/device/select-gpu", {"gpu_index": n})
        g = r.get("selected_gpu") or {}
        return "已切换到 GPU #%s %s%s" % (
            r.get("selected_gpu_index", n), g.get("name", ""),
            "（" + r["warning"] + "）" if r.get("warning") else "")

    def act_auto_configure(self, ui: BaseUI):
        if not ui.confirm("按设备画像自动应用推荐配置(KV缓存/生成参数)?"):
            return "已取消"
        r = self.api.post("/device/auto-configure")
        cfg = r.get("applied_config") or {}
        return "自动配置完成: 档位 %s 评分 %s %s" % (
            r.get("tier", "—"), r.get("score", "—"), cfg.get("description", ""))


# ------------------------------------------------------------
# 6. 日志查看
# ------------------------------------------------------------

class LogsScreen(Screen):
    name = "日志查看"
    auto_refresh = False

    MODES = {"local": "本地文件尾部", "recent": "后端最近日志",
             "files": "后端日志文件", "stats": "后端日志统计"}

    def __init__(self, app):
        super().__init__(app)
        self.mode = "local"
        self.tail_lines = 30
        self.level_filter = ""

    def fetch(self):
        if self.mode == "local":
            return {"local": self._read_local_tail()}
        if self.mode == "recent":
            return {"recent": self.api.get(
                "/logs/recent",
                params={"limit": self.tail_lines, "level": self.level_filter},
                with_log_token=True)}
        if self.mode == "files":
            return {"files": self.api.get("/logs", with_log_token=True)}
        return {"stats": self.api.get("/logs/stats", with_log_token=True)}

    def _read_local_tail(self):
        if not os.path.isdir(LOCAL_LOG_DIR):
            return {"error": "本地日志目录不存在: %s" % LOCAL_LOG_DIR}
        candidates = []
        try:
            for fname in os.listdir(LOCAL_LOG_DIR):
                if not fname.endswith(".log"):
                    continue
                fpath = os.path.join(LOCAL_LOG_DIR, fname)
                try:
                    candidates.append((os.path.getmtime(fpath), fname, fpath))
                except OSError:
                    continue
        except OSError as e:
            return {"error": "读取日志目录失败: %s" % e}
        if not candidates:
            return {"error": "目录 %s 下没有 .log 文件" % LOCAL_LOG_DIR}
        candidates.sort(reverse=True)
        _, fname, fpath = candidates[0]
        try:
            size = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                if size > 131072:
                    f.seek(-131072, os.SEEK_END)
                raw = f.read()
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()[-self.tail_lines:]
        except OSError as e:
            return {"error": "读取文件失败: %s" % e}
        return {"file": fname, "size": size, "lines": lines}

    def lines(self, width: int) -> list:
        d = self.data or {}
        out = []
        out.append(("head", "查看模式: %s    [1]本地尾部 [2]后端最近 [3]后端文件 [4]后端统计" %
                    self.MODES.get(self.mode, self.mode)))
        out.append(("dim", "行数: %d (+/- 调整)    级别过滤: %s    远程访问需日志Token（t 设置）" % (
            self.tail_lines, self.level_filter or "全部")))
        out.append(("", ""))
        if self.mode == "local":
            local = d.get("local") or {}
            if local.get("error"):
                out.append(("warn", local["error"]))
            else:
                out.append(("title", "◆ %s (%s) 最后 %d 行" % (
                    local.get("file", "—"), fmt_bytes(local.get("size")),
                    len(local.get("lines") or []))))
                for ln in (local.get("lines") or []):
                    style = ""
                    if " ERROR" in ln or "[ERROR]" in ln:
                        style = "err"
                    elif " WARNING" in ln or "[WARNING]" in ln:
                        style = "warn"
                    out.append((style, ln))
        elif self.mode == "recent":
            rec = d.get("recent") or {}
            out.append(("title", "◆ 后端内存环形缓冲最近日志 (%s/%s 条, 缓冲 %s/%s)" % (
                rec.get("count", 0), rec.get("matched", 0),
                rec.get("buffer_size", 0), rec.get("buffer_capacity", 0))))
            for item in (rec.get("logs") or []):
                level = item.get("level", "")
                style = "err" if level == "ERROR" else ("warn" if level == "WARNING" else "")
                out.append((style, "%s %-7s %s %s" % (
                    item.get("time", item.get("timestamp", "")), level,
                    item.get("name", ""), item.get("message", ""))))
        elif self.mode == "files":
            files = (d.get("files") or {}).get("files") or []
            out.append(("title", "◆ 后端日志文件（按修改时间降序）"))
            rows = [[f.get("name", ""), fmt_bytes(f.get("size")), f.get("modified", "")]
                    for f in files]
            if rows:
                for ln in make_table(["文件名", "大小", "修改时间"], rows, max_width=width - 2):
                    out.append(("", ln))
            else:
                out.append(("dim", "（后端无日志文件）"))
        else:
            st = d.get("stats") or {}
            out.append(("title", "◆ 后端日志统计"))
            out.append(("", "日志目录: %s" % st.get("log_dir", "—")))
            out.append(("", "文件数: %s    总大小: %s" % (
                st.get("files_count", 0), fmt_bytes(st.get("files_total_bytes")))))
            out.append(("", "内存缓冲: %s/%s 条    累计: %s    估算丢弃: %s" % (
                st.get("buffer_size", 0), st.get("buffer_capacity", 0),
                st.get("buffer_total_seen", 0), st.get("buffer_dropped_estimate", 0))))
            levels = st.get("levels") or {}
            if levels:
                out.append(("", "级别分布: " + "  ".join(
                    "%s=%s" % (k, v) for k, v in sorted(levels.items()))))
            nodes = st.get("nodes") or {}
            if nodes:
                out.append(("", "节点分布: " + "  ".join(
                    "%s=%s" % (k, v) for k, v in sorted(nodes.items()))))
        return out

    def actions(self):
        return [
            ("1", "本地文件尾部", lambda ui: self._switch("local")),
            ("2", "后端最近日志", lambda ui: self._switch("recent")),
            ("3", "后端文件列表", lambda ui: self._switch("files")),
            ("4", "后端日志统计", lambda ui: self._switch("stats")),
            ("+", "增加行数", lambda ui: self._adjust(20)),
            ("-", "减少行数", lambda ui: self._adjust(-20)),
            ("f", "级别过滤", self.act_filter),
            ("t", "设置日志Token", self.act_token),
        ]

    def _switch(self, mode):
        self.mode = mode
        self.refresh(force=True)
        return "已切换到: %s" % self.MODES[mode]

    def _adjust(self, delta):
        self.tail_lines = max(10, min(500, self.tail_lines + delta))
        self.refresh(force=True)
        return "显示行数: %d" % self.tail_lines

    def act_filter(self, ui: BaseUI):
        v = ui.prompt("级别过滤 ERROR/WARNING/INFO/DEBUG (留空取消过滤): ")
        self.level_filter = (v or "").strip().upper()
        self.refresh(force=True)
        return "级别过滤: %s" % (self.level_filter or "全部")

    def act_token(self, ui: BaseUI):
        v = ui.prompt("X-QLH-Log-Token (远程访问日志接口需要, 留空清除): ")
        self.api.log_token = (v or "").strip()
        self.refresh(force=True)
        return "日志 Token 已" + ("设置" if self.api.log_token else "清除")


# ------------------------------------------------------------
# 7. 设置
# ------------------------------------------------------------

class SettingsScreen(Screen):
    name = "设置"
    auto_refresh = False

    def fetch(self):
        return {}

    def lines(self, width: int) -> list:
        out = []
        out.append(("title", "◆ TUI 设置（当前会话生效，可用 --host/--port 等启动参数固化）"))
        out.append(("", ""))
        out.append(("", "  后端主机 host     : %s   （支持 Tailscale IP，如 100.x.x.x）" % self.api.host))
        out.append(("", "  后端端口 port     : %s" % self.api.port))
        out.append(("", "  完整地址          : %s" % self.api.base_url))
        out.append(("", "  请求超时          : %.0f 秒" % self.api.timeout))
        out.append(("", "  自动刷新间隔      : %.0f 秒（总览/节点/队列）" % self.app.interval))
        out.append(("", "  日志 Token        : %s" % ("已设置" if self.api.log_token else "未设置")))
        out.append(("", "  界面模式          : %s" % ("纯文本菜单" if self.app.is_plain else "ANSI 交互")))
        out.append(("", ""))
        out.append(("dim", "  提示: 修改地址后各屏幕会重新连接新的后端。"))
        return out

    def actions(self):
        return [
            ("h", "修改主机", self.act_host),
            ("p", "修改端口", self.act_port),
            ("o", "修改超时", self.act_timeout),
            ("i", "修改刷新间隔", self.act_interval),
            ("k", "设置日志Token", self.act_token),
            ("T", "测试后端连通", self.act_test),
        ]

    def act_host(self, ui: BaseUI):
        v = ui.prompt("后端主机/IP: ", self.api.host)
        if not v:
            return "已取消"
        self.api.host = v.strip()
        self.app.reset_screens()
        return "后端地址已更新: %s" % self.api.base_url

    def act_port(self, ui: BaseUI):
        v = ui.prompt("后端端口: ", str(self.api.port))
        try:
            self.api.port = int(v)
        except (TypeError, ValueError):
            return "无效端口: %s" % v
        self.app.reset_screens()
        return "后端地址已更新: %s" % self.api.base_url

    def act_timeout(self, ui: BaseUI):
        v = ui.prompt("请求超时秒数: ", "%.0f" % self.api.timeout)
        try:
            t = float(v)
        except (TypeError, ValueError):
            return "无效数字: %s" % v
        self.api.timeout = max(1.0, min(t, 120.0))
        return "请求超时: %.0f 秒" % self.api.timeout

    def act_interval(self, ui: BaseUI):
        v = ui.prompt("自动刷新间隔秒数: ", "%.0f" % self.app.interval)
        try:
            t = float(v)
        except (TypeError, ValueError):
            return "无效数字: %s" % v
        self.app.interval = max(1.0, min(t, 60.0))
        return "自动刷新间隔: %.0f 秒" % self.app.interval

    def act_token(self, ui: BaseUI):
        v = ui.prompt("X-QLH-Log-Token (留空清除): ")
        self.api.log_token = (v or "").strip()
        return "日志 Token 已" + ("设置" if self.api.log_token else "清除")

    def act_test(self, ui: BaseUI):
        r = self.api.get("/health")
        if r.get("status") == "ok":
            return "后端 %s 连接正常" % self.api.base_url
        return "后端响应异常: %s" % r


SCREEN_CLASSES = [
    DashboardScreen, NodesScreen, DistributedScreen, QueueScreen,
    DeviceScreen, LogsScreen, SettingsScreen,
]


# ============================================================
# 六·五、TUI 命令系统（/ 开头，任意界面可用）
#
# 每条命令: name / aliases / usage / summary / handler(app, args, opts)
# handler 返回 (消息, 样式)；样式 ∈ ok|err|warn（用于底部消息行）。
# 大输出用 app.show_output(lines, title=...) 分页展示。
# ============================================================

def _split_cmd_args(argv: list) -> tuple:
    """拆分命令参数：位置参数 + --flag value / --flag=value / --flag。"""
    positional, opts = [], {}
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a.startswith("--"):
            name = a[2:].strip()
            if "=" in name:
                k, v = name.split("=", 1)
                opts[k] = v
            elif i + 1 < n and not argv[i + 1].startswith("--"):
                opts[name] = argv[i + 1]
                i += 1
            else:
                opts[name] = True
        else:
            positional.append(a)
        i += 1
    return positional, opts


def _model_summary(resp: dict) -> str:
    """从模型接口响应中提取 model/quant/engine 摘要。"""
    parts = []
    for key in ("model_name", "model_id", "model"):
        if resp.get(key):
            parts.append(str(resp[key]))
            break
    for key in ("quant_type", "current_quant", "quant"):
        if resp.get(key):
            parts.append(str(resp[key]))
            break
    for key in ("engine", "engine_type"):
        if resp.get(key):
            parts.append(str(resp[key]))
            break
    return " ".join(parts) or "—"


# ---- 命令实现 ----

def cmd_help(app, args, opts):
    app.show_output(COMMAND_HELP_LINES, title="QLH TUI 命令集（%d 条命令）" % len(COMMANDS))
    return ("共 %d 条命令（详见上方列表）" % len(COMMANDS), "ok")


def cmd_quit(app, args, opts):
    app.exit_requested = True
    app.shutdown_backend = False
    return ("TUI 正在退出（后端保持运行）…", "ok")


def cmd_shutdown(app, args, opts):
    reason = " ".join(args) or "TUI /shutdown"
    try:
        app.api.post("/system/shutdown", {"reason": reason})
    except ApiError as e:
        return ("后端关闭请求失败（%s）。若后端已停止，请用 /quit 退出 TUI。" % e, "err")
    app.exit_requested = True
    app.shutdown_backend = True
    return ("已请求后端优雅退出（保存状态并清理资源）…", "ok")


def cmd_status(app, args, opts):
    return (app.open_screen("1"), "ok")


def cmd_screen(app, args, opts):
    if not args:
        names = "  ".join("%d.%s" % (i + 1, s.name) for i, s in enumerate(app.screens))
        return ("可用屏幕: %s" % names, "ok")
    msg = app.open_screen(args[0])
    if msg is None:
        return ("未找到屏幕: %s（数字 1-%d 或名称关键字）" % (args[0], len(app.screens)), "err")
    return (msg, "ok")


def cmd_refresh(app, args, opts):
    if app.current is not None:
        app.current.refresh(force=True)
        return ("已刷新: %s" % app.current.name, "ok")
    return ("当前不在任何屏幕中", "warn")


def cmd_model(app, args, opts):
    cur = app.api.get("/models/current")
    if not cur.get("loaded"):
        return ("模型未加载。可用: /models 查看列表，/load <模型ID> 加载", "warn")
    lines = [
        "模型标识    : %s" % cur.get("model_id", "—"),
        "名称        : %s" % cur.get("model_name", "—"),
        "量化精度    : %s" % cur.get("quant_type", "—"),
        "推理引擎    : %s" % cur.get("engine", "—"),
        "设备        : %s" % cur.get("device", "—"),
        "参数量      : %s" % cur.get("total_params", "—"),
        "GPU 显存    : %.2f GB" % float(cur.get("gpu_allocated_gb") or 0),
        "模型路径    : %s" % cur.get("model_path", "—"),
    ]
    app.show_output(lines, title="◆ 当前模型")
    return ("当前模型: %s (%s)" % (cur.get("model_name", "—"), cur.get("quant_type", "—")), "ok")


def cmd_models(app, args, opts):
    try:
        avail = app.api.get("/models/available")
    except ApiError as e:
        return (str(e), "err")
    models = []
    try:
        models = (app.api.get("/models") or {}).get("models") or []
    except ApiError:
        pass
    lines = ["可用模型:"]
    for m in models:
        engines = ",".join(m.get("supported_engines") or [])
        lines.append("  %-16s %s  [%s]  %s" % (
            m.get("model_id", "—"), m.get("name", ""), engines,
            m.get("description", "")))
    if not models:
        lines.append("  （无可用模型）")
    lines.append("")
    lines.append("量化/引擎选项:")
    for q in (avail.get("models") or []):
        lines.append("  %-10s %-22s %s" % (
            q.get("id", ""), q.get("name", ""), q.get("description", "")))
    lines.append("")
    lines.append("当前: quant=%s  engine=%s" % (
        avail.get("current") or "—", avail.get("current_engine") or "—"))
    app.show_output(lines, title="◆ 模型 / 量化 / 引擎")
    return ("共 %d 个模型配置" % len(models), "ok")


def _do_model_change(app, path, args, opts):
    model_id = args[0] if args else None
    body = {
        "model_id": model_id,
        "quant_type": opts.get("quant", "int4"),
        "engine": opts.get("engine", "auto"),
        "use_compile": bool(opts.get("compile")),
    }
    verb = "切换" if path.endswith("/switch") else "加载"
    resp = app.api.post(path, body)
    summary = _model_summary(resp)
    return ("模型%s完成: %s" % (verb, summary or "OK"), "ok")


def cmd_switch(app, args, opts):
    return _do_model_change(app, "/models/switch", args, opts)


def cmd_load(app, args, opts):
    return _do_model_change(app, "/models/load", args, opts)


def cmd_quant(app, args, opts):
    q = args[0].lower()
    cur = app.api.get("/models/current")
    if not cur.get("loaded"):
        return ("当前未加载模型，请先 /load <模型ID> 或 /switch <模型ID>", "warn")
    mid = cur.get("model_id") or ""
    engine = cur.get("engine") or "auto"
    if engine == "llama_cpp":
        engine = "auto"          # 交给后端按文件类型解析
    resp = app.api.post("/models/switch", {
        "model_id": mid, "quant_type": q, "engine": engine})
    return ("量化已切换为 %s: %s" % (q, _model_summary(resp)), "ok")


def cmd_engine(app, args, opts):
    engine = args[0].lower()
    if engine not in ("auto", "llama_cpp", "pytorch", "island"):
        return ("引擎无效: %s（可选 auto|llama_cpp|pytorch|island）" % engine, "err")
    cur = app.api.get("/models/current")
    if not cur.get("loaded"):
        return ("当前未加载模型，请先 /load <模型ID> 或 /switch <模型ID>", "warn")
    mid = cur.get("model_id") or ""
    quant = opts.get("quant") or cur.get("quant_type") or "int4"
    resp = app.api.post("/models/switch", {
        "model_id": mid, "quant_type": quant, "engine": engine})
    return ("引擎已切换为 %s: %s" % (engine, _model_summary(resp)), "ok")


def cmd_presets(app, args, opts):
    try:
        data = app.api.get("/presets")
    except ApiError as e:
        return (str(e), "err")
    lines = ["当前量化: %s    速度估算: %s tok/s    max_tokens: %s" % (
        data.get("current_quant") or "—", data.get("current_speed_tok_s", "—"),
        data.get("max_new_tokens", "—"))]
    lines.append("")
    for p in (data.get("presets") or []):
        lines.append("  %s %s" % (p.get("icon", ""), p.get("label", "")))
        lines.append("      %s" % p.get("question", ""))
        lines.append("      预估: prompt %s tok / 回复 %s tok / 显存 %s MB / %s 秒" % (
            p.get("estimated_prompt_tokens", "—"), p.get("estimated_response_tokens", "—"),
            p.get("estimated_memory_mb", "—"), p.get("estimated_seconds", "—")))
    app.show_output(lines, title="◆ 预设问题")
    return ("共 %d 个预设" % len(data.get("presets") or []), "ok")


def cmd_gpu(app, args, opts):
    if not args:
        prof = app.api.get("/device/profile")
        gpus = prof.get("gpus") or []
        sel = prof.get("selected_gpu_index", 0)
        lines = []
        for i, g in enumerate(gpus):
            mark = "»" if i == sel else " "
            lines.append("  %s #%d  %s  已用 %s/%s MB (%.0f%%)" % (
                mark, i, g.get("name", "—"),
                g.get("used_mb", g.get("allocated_mb", 0)),
                g.get("total_mb", 0), float(g.get("utilization") or 0)))
        if not gpus:
            lines.append("  （未检测到 GPU）")
        app.show_output(lines, title="◆ GPU 列表（当前 #%s）" % sel)
        return ("共 %d 块 GPU（/gpu <序号> 切换）" % len(gpus), "ok")
    try:
        n = int(args[0])
    except ValueError:
        return ("无效 GPU 序号: %s" % args[0], "err")
    r = app.api.post("/device/select-gpu", {"gpu_index": n})
    g = r.get("selected_gpu") or {}
    warn = "（" + r["warning"] + "）" if r.get("warning") else ""
    return ("已切换到 GPU #%s %s%s" % (r.get("selected_gpu_index", n), g.get("name", ""), warn), "ok")


def cmd_device(app, args, opts):
    sub = args[0].lower() if args else "profile"
    if sub == "auto":
        r = app.api.post("/device/auto-configure")
        cfg = r.get("applied_config") or {}
        return ("自动配置完成: 档位 %s 评分 %s %s" % (
            r.get("tier", "—"), r.get("score", "—"), cfg.get("description", "")), "ok")
    if sub == "profile":
        prof = app.api.get("/device/profile")
        lines = [
            "档位    : %s (%s)" % (prof.get("tier_label", "—"), prof.get("tier", "—")),
            "评分    : %s/100" % prof.get("score_total", "—"),
            "推荐配置: %s" % ((prof.get("recommendations") or [{}])[0].get("description", "—")),
        ]
        for w in (prof.get("warnings") or []):
            lines.append("警告    : %s" % w)
        app.show_output(lines, title="◆ 设备画像")
        return ("设备档位: %s" % prof.get("tier_label", "—"), "ok")
    return ("用法: /device auto | /device profile", "err")


def cmd_nodes(app, args, opts):
    nd = app.api.get("/cluster/nodes") or {}
    role = {}
    try:
        role = app.api.get("/cluster/my-role") or {}
    except ApiError:
        pass
    lines = ["本机角色: %s (%s)    节点 %s / 在线 %s / 离线 %s" % (
        role_cn(role.get("node_role", "—")), role.get("node_id", "—"),
        nd.get("count", 0), nd.get("online_count", 0), nd.get("offline_count", 0))]
    lines.append("")
    rows = []
    for n in (nd.get("nodes") or []):
        rows.append([
            n.get("node_id", ""), role_cn(n.get("role", "")), n.get("node_type", ""),
            state_cn(n.get("state", "")), n.get("address", "") or "—",
            n.get("network_type", ""), n.get("task_count", 0),
            n.get("error_count", 0), fmt_age(n.get("last_heartbeat")),
        ])
    if rows:
        lines.extend(make_table(["节点ID", "角色", "类型", "状态", "地址", "网络", "任务", "错误", "心跳"],
                                rows, max_width=shutil.get_terminal_size(fallback=(100, 30)).columns - 2))
    else:
        lines.append("（暂无节点记录）")
    app.show_output(lines, title="◆ 节点列表")
    return ("节点 %s 个（在线 %s）" % (nd.get("count", 0), nd.get("online_count", 0)), "ok")


def cmd_connect(app, args, opts):
    host = args[0]
    port = 8888
    if len(args) > 1:
        try:
            port = int(args[1])
        except ValueError:
            return ("端口无效: %s" % args[1], "err")
    switch = bool(opts.get("switch"))
    role = {}
    try:
        role = app.api.get("/cluster/my-role") or {}
    except ApiError:
        pass
    is_master = (role.get("runtime_node_role") == "master"
                 or role.get("is_provisional") or role.get("is_master"))
    if is_master and not switch:
        return ("本机当前为主节点：确认放弃主节点身份加入 %s:%s 请加 --switch" % (host, port), "warn")
    r = app.api.post("/cluster/connect", {
        "master_host": host, "master_port": port, "switch_to_client": switch})
    return ("连接结果: %s %s" % (r.get("status", "ok"), r.get("message", "")), "ok")


def cmd_dist(app, args, opts):
    act = args[0].lower() if args else "status"
    try:
        cur = app.api.get("/cluster/config/distributed-inference") or {}
    except ApiError as e:
        return (str(e), "err")
    enabled = bool(cur.get("enabled"))
    if act == "status":
        return ("分布式推理: %s" % ("开启" if enabled else "关闭"), "ok")
    if act == "on":
        target = True
    elif act == "off":
        target = False
    elif act == "toggle":
        target = not enabled
    else:
        return ("用法: /dist on|off|toggle|status", "err")
    r = app.api.put("/cluster/config/distributed-inference", {"enabled": target})
    return ("分布式推理已%s (%s)" % ("启用" if target else "停用", r.get("status", "ok")), "ok")


def cmd_queue(app, args, opts):
    sub = args[0].lower() if args else "status"
    if sub == "status":
        q = app.api.get("/cluster/queue") or {}
        lines = ["策略: %s    状态: %s    执行中: %s    排队: %s/%s" % (
            str(q.get("strategy", "—")).upper(),
            "已暂停" if q.get("paused") else "接收中",
            q.get("current_task") or "无",
            q.get("queue_size", 0), q.get("max_size", "—"))]
        lines.append("Q0交互: %s   Q1普通: %s   Q2批量: %s   已完成: %s" % (
            q.get("q0_depth", 0), q.get("q1_depth", 0), q.get("q2_depth", 0),
            q.get("completed_count", 0)))
        rows = []
        for level in ("q0", "q1", "q2"):
            for t in (q.get(level) or []):
                rows.append([
                    t.get("task_id", ""), level.upper(),
                    "%.0fs" % float(t.get("wait_seconds") or 0),
                    t.get("max_new_tokens", "—"),
                    "是" if t.get("is_aged") else "",
                    (t.get("session_id") or "")[:12],
                ])
        lines.append("")
        if rows:
            lines.extend(make_table(["任务ID", "级别", "等待", "tokens", "老化", "会话"], rows,
                                    max_width=shutil.get_terminal_size(fallback=(100, 30)).columns - 2))
        else:
            lines.append("（队列为空）")
        app.show_output(lines, title="◆ 请求队列 (MLFQ)")
        return ("队列: %s 个任务" % q.get("queue_size", 0), "ok")
    if sub == "strategy":
        s = args[1].lower() if len(args) > 1 else ""
        if s not in ("fifo", "mlfq"):
            return ("用法: /queue strategy <fifo|mlfq>", "err")
        r = app.api.post("/cluster/queue/strategy", {"strategy": s})
        return ("策略已切换为 %s" % r.get("strategy", s), "ok")
    if sub == "pause":
        app.api.post("/cluster/queue/pause")
        return ("队列已暂停（不再接收新请求）", "ok")
    if sub == "resume":
        app.api.post("/cluster/queue/resume")
        return ("队列已恢复", "ok")
    if sub == "clear":
        r = app.api.post("/cluster/queue/clear")
        return ("已清空 %s 个排队任务" % r.get("cleared", 0), "ok")
    if sub == "cancel":
        if len(args) < 2:
            return ("用法: /queue cancel <任务ID>", "err")
        tid = urllib.parse.quote(args[1])
        r = app.api.delete("/cluster/queue/task/%s" % tid)
        return ("已取消任务 %s: %s" % (args[1], r.get("status", "ok")), "ok")
    return ("用法: /queue [status|strategy|pause|resume|clear|cancel]", "err")


def cmd_logs(app, args, opts):
    scr = app.find_screen("日志")
    if scr is None:
        return ("日志屏幕不可用", "err")
    if args:
        try:
            scr.tail_lines = max(10, min(500, int(args[0])))
        except ValueError:
            return ("行数无效: %s" % args[0], "err")
    if not app.is_plain and opts.get("remote"):
        scr.mode = "recent"
    return (app.open_screen("日志"), "ok")


def cmd_log(app, args, opts):
    sub = args[0].lower() if args else ""
    scr = app.find_screen("日志")
    if sub == "filter":
        level = args[1].upper() if len(args) > 1 else ""
        if level and level not in ("ERROR", "WARNING", "INFO", "DEBUG"):
            return ("级别无效: %s（ERROR/WARNING/INFO/DEBUG）" % level, "err")
        if scr is not None:
            scr.level_filter = level
            scr.refresh(force=True)
        return ("日志级别过滤: %s" % (level or "全部"), "ok")
    if sub == "token":
        token = args[1] if len(args) > 1 else ""
        app.api.log_token = token.strip()
        return ("日志 Token 已" + ("设置" if app.api.log_token else "清除"), "ok")
    return ("用法: /log filter <级别> | /log token <令牌>", "err")


def cmd_host(app, args, opts):
    host = args[0]
    port = getattr(app.api, "port", DEFAULT_PORT)
    if len(args) > 1:
        try:
            port = int(args[1])
        except ValueError:
            return ("端口无效: %s" % args[1], "err")
    app.api.host = host
    app.api.port = port
    app.reset_screens()
    return ("后端地址已更新: %s" % app.api.base_url, "ok")


def cmd_interval(app, args, opts):
    try:
        t = float(args[0])
    except (TypeError, ValueError, IndexError):
        return ("用法: /interval <秒>", "err")
    app.interval = max(1.0, min(t, 60.0))
    return ("自动刷新间隔: %.0f 秒" % app.interval, "ok")


def cmd_timeout(app, args, opts):
    try:
        t = float(args[0])
    except (TypeError, ValueError, IndexError):
        return ("用法: /timeout <秒>", "err")
    app.api.timeout = max(1.0, min(t, 120.0))
    return ("请求超时: %.0f 秒" % app.api.timeout, "ok")


def cmd_token(app, args, opts):
    token = args[0] if args else ""
    app.api.log_token = token.strip()
    return ("日志 Token 已" + ("设置" if app.api.log_token else "清除"), "ok")


def cmd_chat(app, args, opts):
    sub = args[0].lower() if args else ""
    if sub == "clear":
        app.api.post("/chat/clear")
        return ("对话历史已清空", "ok")
    return ("用法: /chat clear", "err")


def cmd_cancel(app, args, opts):
    tid = args[0] if args else ""
    if not tid:
        return ("用法: /cancel <任务ID>", "err")
    from urllib.parse import quote
    try:
        r = app.api.post("/chat/generations/%s/cancel" % quote(tid))
        return ("已取消生成任务 %s" % tid, "ok")
    except ApiError:
        pass
    try:
        r = app.api.post("/workflows/%s/cancel" % quote(tid))
        return ("已取消工作流 %s" % tid, "ok")
    except ApiError as e:
        return ("取消失败（生成与工作流均未找到）: %s" % e, "err")


# ---- 注册表 ----

COMMANDS = [
    # (name, aliases, usage, summary, handler, min_args, max_args)
    {"name": "/help", "aliases": ["/h"], "usage": "/help",
     "summary": "显示命令集帮助", "handler": cmd_help},
    {"name": "/quit", "aliases": ["/q", "/exit"], "usage": "/quit",
     "summary": "退出 TUI（后端保持运行）", "handler": cmd_quit},
    {"name": "/shutdown", "aliases": ["/halt"], "usage": "/shutdown [原因]",
     "summary": "优雅退出：后端保存/清理资源后退出，随后 TUI 退出", "handler": cmd_shutdown},
    {"name": "/status", "aliases": ["/st"], "usage": "/status",
     "summary": "打开系统状态总览", "handler": cmd_status},
    {"name": "/screen", "aliases": ["/goto"], "usage": "/screen <编号|名称>",
     "summary": "跳转管理屏幕（1-7 或 名称关键字）", "handler": cmd_screen, "min_args": 1},
    {"name": "/refresh", "aliases": ["/r"], "usage": "/refresh",
     "summary": "立即刷新当前屏幕", "handler": cmd_refresh},
    {"name": "/model", "aliases": [], "usage": "/model",
     "summary": "当前模型详情", "handler": cmd_model},
    {"name": "/models", "aliases": [], "usage": "/models",
     "summary": "列出可用模型 / 量化 / 引擎", "handler": cmd_models},
    {"name": "/switch", "aliases": [], "usage": "/switch <模型ID> [--quant 精度] [--engine 引擎] [--compile]",
     "summary": "切换模型（失败自动回滚）", "handler": cmd_switch, "min_args": 1, "max_args": 1},
    {"name": "/load", "aliases": [], "usage": "/load <模型ID> [--quant 精度] [--engine 引擎] [--compile]",
     "summary": "加载模型（缺省模型ID 使用默认 Qwen）", "handler": cmd_load, "max_args": 1},
    {"name": "/quant", "aliases": [], "usage": "/quant <int4|int8|fp16|gguf>",
     "summary": "切换量化精度（重载当前模型）", "handler": cmd_quant, "min_args": 1, "max_args": 1},
    {"name": "/engine", "aliases": [], "usage": "/engine <auto|llama_cpp|pytorch|island>",
     "summary": "切换推理引擎（重载当前模型）", "handler": cmd_engine, "min_args": 1, "max_args": 1},
    {"name": "/presets", "aliases": [], "usage": "/presets",
     "summary": "预设问题与 Token/显存估算", "handler": cmd_presets},
    {"name": "/gpu", "aliases": [], "usage": "/gpu [序号]",
     "summary": "列出 GPU；带序号则切换推理 GPU", "handler": cmd_gpu, "max_args": 1},
    {"name": "/device", "aliases": [], "usage": "/device <auto|profile>",
     "summary": "设备自动配置 / 查看设备画像", "handler": cmd_device, "min_args": 1, "max_args": 1},
    {"name": "/nodes", "aliases": [], "usage": "/nodes",
     "summary": "节点列表与状态", "handler": cmd_nodes},
    {"name": "/connect", "aliases": [], "usage": "/connect <IP> [端口] [--switch]",
     "summary": "连接主节点（--switch 放弃本机主节点身份）", "handler": cmd_connect, "min_args": 1, "max_args": 2},
    {"name": "/dist", "aliases": [], "usage": "/dist <on|off|toggle|status>",
     "summary": "分布式推理开关", "handler": cmd_dist, "max_args": 1},
    {"name": "/queue", "aliases": [], "usage": "/queue [status|strategy <fifo|mlfq>|pause|resume|clear|cancel <任务ID>]",
     "summary": "请求队列状态与控制", "handler": cmd_queue},
    {"name": "/logs", "aliases": [], "usage": "/logs [行数] [--remote]",
     "summary": "打开日志查看（可指定行数）", "handler": cmd_logs, "max_args": 1},
    {"name": "/log", "aliases": [], "usage": "/log <filter <级别>|token <令牌>>",
     "summary": "日志级别过滤 / 设置日志 Token", "handler": cmd_log},
    {"name": "/host", "aliases": [], "usage": "/host <主机> [端口]",
     "summary": "切换后端地址", "handler": cmd_host, "min_args": 1, "max_args": 2},
    {"name": "/interval", "aliases": [], "usage": "/interval <秒>",
     "summary": "自动刷新间隔", "handler": cmd_interval, "min_args": 1, "max_args": 1},
    {"name": "/timeout", "aliases": [], "usage": "/timeout <秒>",
     "summary": "HTTP 请求超时", "handler": cmd_timeout, "min_args": 1, "max_args": 1},
    {"name": "/token", "aliases": [], "usage": "/token <令牌>",
     "summary": "设置日志访问 Token（留空清除）", "handler": cmd_token, "max_args": 1},
    {"name": "/chat", "aliases": [], "usage": "/chat <clear>",
     "summary": "清空对话历史", "handler": cmd_chat, "min_args": 1, "max_args": 1},
    {"name": "/cancel", "aliases": [], "usage": "/cancel <任务ID>",
     "summary": "取消生成 / 工作流任务", "handler": cmd_cancel, "min_args": 1, "max_args": 1},
]


def _build_command_help_lines() -> list:
    """生成命令集帮助文本行（/help 与 --help 共用）。"""
    lines = []
    groups = [
        ("系统", ["/help", "/status", "/screen", "/refresh", "/quit", "/shutdown"]),
        ("模型 / 量化 / 引擎", ["/model", "/models", "/switch", "/load", "/quant", "/engine", "/presets"]),
        ("设备", ["/gpu", "/device"]),
        ("集群 / 队列", ["/nodes", "/connect", "/dist", "/queue"]),
        ("日志", ["/logs", "/log"]),
        ("设置", ["/host", "/interval", "/timeout", "/token"]),
        ("会话", ["/chat", "/cancel"]),
    ]
    by_name = {c["name"]: c for c in COMMANDS}
    for title, names in groups:
        lines.append("── %s ──" % title)
        for name in names:
            c = by_name[name]
            alias = ("  别名: %s" % " ".join(c["aliases"])) if c.get("aliases") else ""
            lines.append("  %-58s %s%s" % (c["usage"], c["summary"], alias))
        lines.append("")
    return lines


COMMAND_HELP_LINES = _build_command_help_lines()
COMMAND_HELP_TEXT = (
    "TUI 命令集：任意界面输入 / 开头命令后按 Enter 执行，ESC 取消。\n"
    "模型切换 / 量化切换 / 引擎切换等操作无需进入菜单，直接输入命令即可。\n\n"
    + "\n".join(COMMAND_HELP_LINES))


# ============================================================
# 七、应用主体（ANSI 交互模式）
# ============================================================

class BaseApp:
    def __init__(self, api: ApiClient, interval: float, is_plain: bool):
        self.api = api
        self.interval = interval
        self.is_plain = is_plain
        self.screens = [cls(self) for cls in SCREEN_CLASSES]
        self.exit_requested = False     # True 时主循环退出
        self.shutdown_backend = False   # 退出前是否已请求后端优雅关闭

    def reset_screens(self):
        """后端地址变化后清除各屏幕缓存数据。"""
        for s in self.screens:
            s.data = None
            s.error = None
            s.last_fetch = 0.0

    # ---- 命令系统 ----
    def find_screen(self, key: str):
        """按编号（1-N）或名称关键字查找屏幕。"""
        key = (key or "").strip()
        if key.isdigit():
            n = int(key)
            if 1 <= n <= len(self.screens):
                return self.screens[n - 1]
            return None
        low = key.lower()
        for s in self.screens:
            if low in s.name.lower():
                return s
        return None

    def open_screen(self, key: str):
        """打开屏幕：交互模式切换当前屏，纯文本模式打印内容。失败返回 None。"""
        scr = self.find_screen(key)
        if scr is None:
            return None
        scr.refresh(force=True)
        if self.is_plain:
            w = shutil.get_terminal_size(fallback=(100, 30)).columns
            out = []
            if scr.error:
                out.append("[错误] %s" % scr.error)
            else:
                for style, text in _norm_lines(scr.lines(w)):
                    prefix = "[错误] " if style == "err" else ("[注意] " if style == "warn" else "")
                    out.append(prefix + text)
            self.show_output(out, title="==== %s ====" % scr.name)
            return "已打开: %s" % scr.name
        self.current = scr
        self.offset = 0
        return "已打开: %s" % scr.name

    def exec_command(self, line: str) -> tuple:
        """解析并执行一条 / 命令，返回 (消息, 样式)。"""
        text = (line or "").strip()
        if not text.startswith("/"):
            return ("命令必须以 / 开头（输入 /help 查看命令集）", "err")
        parts = text[1:].split()
        if not parts:
            return ("输入 /help 查看命令集", "warn")
        name = parts[0].lower()
        args, opts = _split_cmd_args(parts[1:])
        cmd = None
        for c in COMMANDS:
            if name == c["name"].lstrip("/") or name in (a.lstrip("/") for a in c.get("aliases", [])):
                cmd = c
                break
        if cmd is None:
            return ("未知命令: /%s（输入 /help 查看命令集）" % name, "err")
        if len(args) < cmd.get("min_args", 0):
            return ("参数不足。用法: %s" % cmd["usage"], "warn")
        if cmd.get("max_args") is not None and len(args) > cmd["max_args"]:
            return ("参数过多。用法: %s" % cmd["usage"], "warn")
        return cmd["handler"](self, args, opts)

    def show_output(self, lines: list, title: str = ""):
        """大结果输出（子类实现：交互分页 / 纯文本直接打印）。"""
        for ln in lines:
            print(ln)


class InteractiveApp(BaseApp):
    """全屏 ANSI 交互模式。"""

    def __init__(self, api, interval, term: AnsiTerm):
        super().__init__(api, interval, is_plain=False)
        self.term = term
        self.menu_idx = 0
        self.current = None       # None=主菜单
        self.offset = 0
        self.message = None
        self.message_style = "ok"
        self.message_at = 0.0
        self.cmd_mode = False     # True=正在输入 / 命令
        self.cmd_buf = ""

    # ---- 主循环 ----
    def run(self) -> int:
        while not self.exit_requested:
            if self.current is not None:
                self.current.refresh()
            self.render()
            try:
                key = self.term.get_key(0.3)
            except KeyboardInterrupt:
                return 0
            if key is None:
                continue
            if key == "EOF":
                return 0
            if self.cmd_mode:
                self._handle_cmd_key(key)
            elif key == "/":
                self.cmd_mode = True
                self.cmd_buf = ""
            elif self.current is None:
                if not self.handle_menu_key(key):
                    return 0
            else:
                self.handle_screen_key(key)
        return 0

    # ---- 命令输入 ----
    def _handle_cmd_key(self, key):
        if key == "ESC":
            self.cmd_mode = False
            self.cmd_buf = ""
            return
        if key == "ENTER":
            line = "/" + self.cmd_buf
            self.cmd_mode = False
            self.cmd_buf = ""
            self._run_command(line)
            return
        if key in ("\x08", "\x7f", "BACKSPACE"):
            self.cmd_buf = self.cmd_buf[:-1]
            return
        if isinstance(key, str) and len(key) == 1:
            self.cmd_buf += key

    def _run_command(self, line: str):
        self._say("正在执行: %s …" % line, "cmd")
        self.render()
        try:
            msg, style = self.exec_command(line)
        except ApiError as e:
            msg, style = str(e), "err"
        except (EOFError, KeyboardInterrupt):
            msg, style = "命令已取消", "warn"
        except Exception as e:
            msg, style = "命令失败: %s" % e, "err"
        if self.current is not None:
            self.current.refresh(force=True)
        if msg:
            self._say(msg, style)
        self.term.write("\x1b[2J")     # 命令输出可能弄脏屏幕，整屏重绘

    def show_output(self, lines: list, title: str = ""):
        self.term.show_lines(lines, title=title)

    # ---- 渲染 ----
    def render(self):
        w, h = self.term.size()
        rows = []
        title = " %s  v%s   后端 %s" % (APP_TITLE, TUI_VERSION, self.api.base_url)
        rows.append(("title", truncate_display(title, w)))
        rows.append(("dim", "─" * w))
        body_h = max(h - 4, 3)
        if self.current is None:
            body = self._menu_lines()
            self.offset = 0
        else:
            raw = []
            if self.current.error:
                raw.append(("err", "错误: %s" % self.current.error))
                raw.append(("", ""))
            raw.extend(_norm_lines(self.current.lines(w)))
            max_off = max(len(raw) - body_h, 0)
            self.offset = max(0, min(self.offset, max_off))
            body = raw[self.offset:self.offset + body_h]
            if max_off > 0 and body:
                pos = "── [第 %d-%d 行 / 共 %d 行, ↑↓/PgUp/PgDn 翻看] ──" % (
                    self.offset + 1, min(self.offset + body_h, len(raw)), len(raw))
                body[-1] = ("dim", pos)
        while len(body) < body_h:
            body.append(("", ""))
        rows.extend(body[:body_h])
        # 消息行 / 命令输入行
        if self.cmd_mode:
            rows.append(("input", truncate_display("命令: /%s▌" % self.cmd_buf, w)))
        elif self.message and time.monotonic() - self.message_at < 8:
            rows.append((self.message_style, truncate_display("» " + self.message, w)))
        else:
            rows.append(("", ""))
        rows.append(("dim", truncate_display(self._hints(), w)))
        out = ["\x1b[H"]
        for style, text in rows[:h]:
            padded = pad_display(text, w)
            out.append(self.term.paint(padded, style))
            out.append("\x1b[K\r\n")
        frame = "".join(out)
        if frame.endswith("\r\n"):
            frame = frame[:-2]
        self.term.write(frame)

    def _menu_lines(self):
        lines = [("", ""), ("head", "  请选择管理功能（↑↓ 移动，Enter 进入，数字直达，q 退出）"), ("", "")]
        for i, s in enumerate(self.screens):
            marker = "»" if i == self.menu_idx else " "
            text = "  %s %d. %s" % (marker, i + 1, s.name)
            lines.append(("sel" if i == self.menu_idx else "", text))
        lines.append(("", ""))
        lines.append(("dim", "  提示: 若后端未启动，请先运行 python src/api_server.py"))
        return lines

    def _hints(self):
        if self.cmd_mode:
            return " 输入命令: Enter 执行   ESC 取消   Backspace 删除   按 /help 查看命令集"
        if self.current is None:
            return " ↑/↓ 选择   Enter 进入   1-%d 直达   / 命令   q 退出" % len(self.screens)
        acts = self.current.actions()
        parts = ["Esc 返回", "r 刷新", "↑/↓ 滚动", "/ 命令"]
        parts.extend("%s %s" % (k, label) for k, label, _ in acts)
        return " " + "   ".join(parts)

    # ---- 按键 ----
    def handle_menu_key(self, key) -> bool:
        if key in ("q", "Q", "ESC"):
            return False
        if key == "UP":
            self.menu_idx = (self.menu_idx - 1) % len(self.screens)
        elif key == "DOWN":
            self.menu_idx = (self.menu_idx + 1) % len(self.screens)
        elif key == "ENTER":
            self._open(self.menu_idx)
        elif isinstance(key, str) and key.isdigit():
            n = int(key)
            if 1 <= n <= len(self.screens):
                self._open(n - 1)
        return True

    def _open(self, idx):
        self.current = self.screens[idx]
        self.offset = 0
        self.message = None
        self.current.refresh(force=True)

    def handle_screen_key(self, key):
        scr = self.current
        if key in ("ESC", "q", "Q"):
            self.current = None
            self.offset = 0
            return
        if key == "UP":
            self.offset = max(self.offset - 1, 0)
            return
        if key == "DOWN":
            self.offset += 1
            return
        if key == "PGUP":
            self.offset = max(self.offset - 15, 0)
            return
        if key == "PGDN":
            self.offset += 15
            return
        if key == "HOME":
            self.offset = 0
            return
        if key in ("r", "R") and all(k != key for k, _, _ in scr.actions()):
            scr.refresh(force=True)
            self._say("已刷新", "ok")
            return
        for k, label, handler in scr.actions():
            if key == k:
                self._run_action(label, handler)
                return

    def _run_action(self, label, handler):
        ui = TermUI(self.term)
        try:
            msg = handler(ui)
            style = "ok"
        except ApiError as e:
            msg = str(e)
            style = "err"
        except (EOFError, KeyboardInterrupt):
            msg = "已取消"
            style = "warn"
        except Exception as e:
            msg = "操作失败: %s" % e
            style = "err"
        self.current.refresh(force=True)
        if msg:
            self._say(msg, style)
        self.term.write("\x1b[2J")     # 输入行可能弄脏屏幕，整屏重绘

    def _say(self, msg, style="ok"):
        self.message = msg
        self.message_style = style
        self.message_at = time.monotonic()


# ============================================================
# 八、纯文本编号菜单模式（--plain / 非交互终端降级）
# ============================================================

class PlainApp(BaseApp):
    """无 ANSI 的编号菜单模式：适配管道、旧终端、脚本化操作。"""

    def __init__(self, api, interval):
        super().__init__(api, interval, is_plain=True)

    def run(self) -> int:
        print("=" * 64)
        print(APP_TITLE + "  v" + TUI_VERSION + "  (纯文本模式)")
        print("后端: " + self.api.base_url)
        print("输入 / 开头的命令（如 /help /quant int4）可直接操作")
        print("=" * 64)
        try:
            while True:
                if not self._main_menu():
                    break
        except (EOFError, KeyboardInterrupt):
            print()
        if self.shutdown_backend:
            print("后端已请求优雅退出，本进程即将结束。")
        print("再见！")
        return 0

    def show_output(self, lines: list, title: str = ""):
        if title:
            print(title)
        for ln in lines:
            print(ln)

    def _main_menu(self) -> bool:
        print()
        print("---- 主菜单 ----")
        for i, s in enumerate(self.screens):
            print("  %d. %s" % (i + 1, s.name))
        print("  q. 退出")
        choice = input("选择> ").strip().lower()
        if choice in ("q", "0", "exit", "quit"):
            return False
        if not choice:
            return True
        if choice.startswith("/"):
            try:
                msg, style = self.exec_command(choice)
            except ApiError as e:
                msg, style = str(e), "err"
            except Exception as e:
                msg, style = "命令失败: %s" % e, "err"
            if msg:
                print(("[错误] " if style == "err" else "» ") + msg)
            if self.exit_requested:
                return False
            return True
        if choice.isdigit() and 1 <= int(choice) <= len(self.screens):
            self._screen_loop(self.screens[int(choice) - 1])
        else:
            print("无效选择: %s" % choice)
        return True

    def _screen_loop(self, scr: Screen):
        while True:
            scr.refresh(force=True)
            width = shutil.get_terminal_size(fallback=(100, 30)).columns
            print()
            print("==== %s ====" % scr.name)
            if scr.error:
                print("[错误] %s" % scr.error)
            else:
                for style, text in _norm_lines(scr.lines(width)):
                    prefix = ""
                    if style == "err":
                        prefix = "[错误] "
                    elif style == "warn":
                        prefix = "[注意] "
                    print(prefix + text)
            acts = scr.actions()
            print()
            parts = ["r=刷新"] + ["%s=%s" % (k, label) for k, label, _ in acts] + ["回车/b=返回", "q=退出"]
            print("操作: " + "  ".join(parts))
            choice = input("> ").strip()
            if choice in ("", "b", "B", "0"):
                return
            if choice in ("q", "Q"):
                raise EOFError
            if choice.startswith("/"):
                try:
                    msg, style = self.exec_command(choice)
                except ApiError as e:
                    msg, style = str(e), "err"
                except Exception as e:
                    msg, style = "命令失败: %s" % e, "err"
                if msg:
                    print(("[错误] " if style == "err" else "» ") + msg)
                if self.exit_requested:
                    raise EOFError
                continue
            if choice in ("r", "R") and not any(k == choice for k, _, _ in acts):
                continue
            matched = False
            for k, label, handler in acts:
                if choice == k:
                    matched = True
                    try:
                        msg = handler(PlainUI())
                        if msg:
                            print("» " + msg)
                    except ApiError as e:
                        print("[错误] %s" % e)
                    except (EOFError, KeyboardInterrupt):
                        raise
                    except Exception as e:
                        print("[错误] 操作失败: %s" % e)
                    break
            if not matched:
                print("无效操作: %s" % choice)


# ============================================================
# 九、入口
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bjtu",
        description="QLH 分布式边缘推理 TUI 管理菜单（跨平台终端版后台管理，纯标准库实现）\n"
                    "任意界面输入 / 开头命令即可操作：模型/量化/引擎切换、GPU 选择、\n"
                    "分布式开关、队列控制、日志、优雅退出等，无需进入菜单。",
        epilog=COMMAND_HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="后端 API 主机（默认 %s，可填 Tailscale IP）" % DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="后端 API 端口（默认 %d）" % DEFAULT_PORT)
    p.add_argument("--plain", action="store_true",
                   help="强制纯文本编号菜单模式（无 ANSI 全屏界面）")
    p.add_argument("--interval", type=float, default=3.0,
                   help="自动刷新间隔秒数（默认 3）")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="HTTP 请求超时秒数（默认 5）")
    p.add_argument("--log-token", default="",
                   help="远程访问日志接口的 X-QLH-Log-Token")
    p.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    p.add_argument("--version", action="version",
                   version="qlh-tui-admin %s" % TUI_VERSION)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    api = ApiClient(host=args.host, port=args.port,
                    timeout=max(1.0, args.timeout), log_token=args.log_token)
    interval = max(1.0, args.interval)

    if not args.plain:
        term = AnsiTerm(color=not args.no_color)
        try:
            with term:
                app = InteractiveApp(api, interval, term)
                rc = app.run()
                if app.shutdown_backend:
                    print("后端已优雅退出，所有资源已清理。")
                else:
                    print("TUI 已退出（后端保持运行，随时可重新运行 bjtu 进入）。")
                return rc
        except TermNotCapable as e:
            print("[提示] 交互模式不可用（%s），自动切换纯文本模式。" % e)
        except KeyboardInterrupt:
            return 0

    try:
        return PlainApp(api, interval).run()
    except (EOFError, KeyboardInterrupt):
        print()
        print("再见！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
