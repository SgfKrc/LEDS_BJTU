"""
QLH 边缘推理系统 — 打包版启动器
================================
跨平台（Windows / Linux）安装包的主入口点。

双引擎架构:
  1. llama.cpp + GGUF  — CPU/集显默认引擎（Q4_K_M ~1.16 GB）
  2. PyTorch + bitsandbytes — CUDA 引擎（INT4 ~1.75 GB 显存）

启动流程:
  0. Tailscale 组网检查（首次启动引导加入）
  1. 检测 CUDA 可用性 + 模型文件
  2. 若缺失 → 弹出系统对话框引导下载（智能推荐格式）
  3. 模型就绪 → 自动选择最优引擎 → 后台启动 FastAPI（端口 8000）
  4. 自动打开系统浏览器加载 React 前端（Linux）或 pywebview 原生窗口（Windows）
"""

from __future__ import annotations

import os
import sys

# ═══════════════════════════════════════════════════════════════════
# ★ 平台检测（影响后续分支逻辑）
# ═══════════════════════════════════════════════════════════════════
IS_LINUX = sys.platform == "linux"
IS_WINDOWS = sys.platform == "win32"

# ═══════════════════════════════════════════════════════════════════
# ★ 静默模式：PyInstaller console=False 时，重定向 stdout/stderr 到日志文件
# 必须在 import logging 之前执行，否则 basicConfig 的 StreamHandler 已绑定到原始 stderr
# ═══════════════════════════════════════════════════════════════════
if getattr(sys, 'frozen', False):
    import datetime as _dt
    _redirect_dir = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "logs")
    try:
        os.makedirs(_redirect_dir, exist_ok=True)
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sys.stdout = open(os.path.join(_redirect_dir, f"stdout_{_ts}.log"), "w", encoding="utf-8")
        sys.stderr = open(os.path.join(_redirect_dir, f"stderr_{_ts}.log"), "w", encoding="utf-8")
    except Exception:
        pass

import logging
import threading
import time
import subprocess as _sp

# ★ Windows PyInstaller: 强制在 psycopg2 之前加载 ssl，避免 OpenSSL DLL 冲突
# psycopg2-binary 捆绑了自己的 libssl-3-x64-{hash}.dll，通过 add_dll_directory 注册后
# 可能干扰 Python _ssl.pyd 加载 libssl-3.dll，导致"内存位置访问无效"。
# Linux 上无此问题——OpenSSL 由系统包管理器管理。
if IS_WINDOWS and getattr(sys, 'frozen', False):
    import ctypes as _ctypes
    import os as _os
    _internal_dir = _os.path.join(_os.path.dirname(sys.executable), '_internal')
    if _os.path.isdir(_internal_dir):
        try:
            _os.add_dll_directory(_internal_dir)
        except Exception:
            pass
        # 按依赖顺序加载：libcrypto 先，libssl 后
        for _dll_name in ('libcrypto-3.dll', 'libssl-3.dll'):
            _dll_path = _os.path.join(_internal_dir, _dll_name)
            if _os.path.isfile(_dll_path):
                try:
                    _ctypes.CDLL(_dll_path)
                except Exception:
                    pass

import ssl  # noqa: E402, F401

# 确保 src 目录在 path 中（开发模式：launcher.py 在 packaging/ 子目录下）
# PyInstaller 打包后所有模块由 bootloader 加载，无需手动添加 path
if getattr(sys, 'frozen', False):
    # PyInstaller 模式：模块在 PYZ 归档中，Python 可正常导入
    _launcher_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    _launcher_dir = os.path.dirname(os.path.abspath(__file__))
    _src_dir = os.path.abspath(os.path.join(_launcher_dir, "..", "src"))
    if os.path.isdir(_src_dir):
        sys.path.insert(0, _src_dir)
    # 同目录也加入 path
    sys.path.insert(0, _launcher_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("launcher")

# ================================================================
# Tailscale 组网配置
# ================================================================
TAILSCALE_INVITE_URL = "https://login.tailscale.com/uinv/iWAME6zVuB11wUxixU2Z611"
if IS_LINUX:
    TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/linux"
else:
    TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/windows"

# 配置目录（跨平台：Linux 用 XDG，Windows 用 LOCALAPPDATA）
if IS_LINUX:
    _CONFIG_DIR = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "qlh",
    )
else:
    _CONFIG_DIR = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "QLH-Edge-Inference",
    )
_TAILSCALE_FLAG_DIR = _CONFIG_DIR
_TAILSCALE_FLAG_FILE = os.path.join(_TAILSCALE_FLAG_DIR, ".tailscale_joined")


# 对话框返回值常量（保持与 Win32 MessageBox 兼容）
_IDYES = 6
_IDNO = 7
_IDCANCEL = 2
_MB_OK = 0x00000000
_MB_OKCANCEL = 0x00000001
_MB_YESNO = 0x00000004
_MB_YESNOCANCEL = 0x00000003
_MB_ICONINFORMATION = 0x00000040
_MB_ICONQUESTION = 0x00000020
_MB_ICONWARNING = 0x00000030


def _has_interactive_stdin() -> bool:
    """判断当前进程是否有可交互 stdin。PyInstaller windowed 模式没有控制台。"""
    try:
        return sys.stdin is not None and not sys.stdin.closed and sys.stdin.isatty()
    except Exception:
        return False


def _safe_input(prompt: str = "", default: str | None = None) -> str | None:
    """input() 的安全包装，避免 windowed 打包版触发 lost sys.stdin 崩溃。"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt, RuntimeError, OSError) as e:
        logger.warning(f"无法读取控制台输入: {e}")
        return default


def _safe_pause(message: str = "按 Enter 键退出..."):
    """仅在有控制台时暂停；无控制台时不阻塞、不崩溃。"""
    print(message)
    _safe_input(default="")


def _show_windows_messagebox(title: str, message: str,
                              flags: int = _MB_OK | _MB_ICONINFORMATION) -> int:
    """Windows: ctypes.windll MessageBox。"""
    try:
        import ctypes
        return ctypes.windll.user32.MessageBoxW(0, message, title, flags)
    except Exception as e:
        logger.warning(f"MessageBox 显示失败: {e}")
        return _IDCANCEL


def _show_linux_dialog(title: str, message: str,
                        buttons: str = "ok") -> int:
    """
    Linux: 使用 zenity 显示对话框，返回 Win32 兼容的返回值。

    buttons:
      "ok"       → zenity --info       → _IDCANCEL (zenity info 无返回值区分)
      "okcancel" → zenity --question   → OK=0→_IDYES, Cancel=1→_IDNO
      "yesno"    → zenity --question   → Yes=0→_IDYES, No=1→_IDNO
      "yesnocancel" → zenity --question --extra-button ... 不支持三层完美映射
    """
    import shutil
    if shutil.which("zenity"):
        try:
            if buttons == "ok":
                _sp.run(["zenity", "--info", "--title", title,
                         "--text", message, "--width=450"],
                        timeout=30)
                return _IDYES
            elif buttons in ("okcancel", "yesno"):
                rc = _sp.run(["zenity", "--question", "--title", title,
                              "--text", message, "--width=450",
                              "--ok-label=是(Y)", "--cancel-label=否(N)"],
                             timeout=30).returncode
                return _IDYES if rc == 0 else _IDNO
            elif buttons == "yesnocancel":
                # zenity 不支持三按钮；用 --extra-button 模拟
                rc = _sp.run(["zenity", "--question", "--title", title,
                              "--text", message, "--width=450",
                              "--ok-label=是(Y)", "--cancel-label=否(N)",
                              "--extra-button=取消"],
                             capture_output=True, text=True, timeout=30)
                if rc.returncode == 0:
                    # extra-button 被点击时 zenity 退出码仍为 0，
                    # 但按钮标签会输出到 stdout
                    stdout_out = (rc.stdout or "").strip()
                    if "取消" in stdout_out:
                        return _IDCANCEL
                    return _IDYES
                elif rc.returncode == 1:
                    return _IDNO  # 关闭窗口 / Esc
                return _IDCANCEL
        except Exception as e:
            logger.debug(f"zenity 对话框失败: {e}")
    # 回退：终端 CLI
    return _cli_dialog(title, message, buttons)


def _cli_dialog(title: str, message: str, buttons: str = "ok") -> int:
    """终端 CLI 回退对话框。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")
    if buttons == "ok":
        _safe_input("按 Enter 继续...", default="")
        return _IDYES
    elif buttons in ("okcancel", "yesno"):
        choice = _safe_input("输入 y/yes 继续，n/no 取消: ", default="n")
        return _IDYES if choice and choice.lower() in ("y", "yes") else _IDNO
    else:
        choice = _safe_input("输入 y/yes=是, n/no=否, c/cancel=取消: ", default="c")
        c = (choice or "c").lower()
        if c in ("y", "yes"):
            return _IDYES
        elif c in ("n", "no"):
            return _IDNO
        return _IDCANCEL


def _show_dialog(title: str, message: str,
                 buttons: str = "ok") -> int:
    """跨平台对话框：Linux→zenity→CLI, Windows→MessageBox, 其他→CLI。"""
    if IS_LINUX:
        return _show_linux_dialog(title, message, buttons)
    elif IS_WINDOWS:
        flags_map = {
            "ok": _MB_OK | _MB_ICONINFORMATION,
            "okcancel": _MB_OKCANCEL | _MB_ICONQUESTION,
            "yesno": _MB_YESNO | _MB_ICONQUESTION,
            "yesnocancel": _MB_YESNOCANCEL | _MB_ICONQUESTION,
        }
        flags = flags_map.get(buttons, _MB_OK | _MB_ICONINFORMATION)
        return _show_windows_messagebox(title, message, flags)
    else:
        return _cli_dialog(title, message, buttons)


def _open_url(url: str):
    """用系统默认浏览器打开 URL（跨平台）。"""
    if IS_LINUX:
        try:
            _sp.run(["xdg-open", url], timeout=10)
            return
        except Exception:
            pass
    else:
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"无法打开链接 {url}: {e}")


def _find_tailscale_exe() -> str | None:
    """查找 tailscale 可执行文件的完整路径，未安装返回 None（跨平台）。"""
    import shutil
    if IS_WINDOWS:
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                        "Tailscale", "tailscale.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)",
                        "C:\\Program Files (x86)"), "Tailscale", "tailscale.exe"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    # Linux / macOS / Windows PATH fallback
    exe = shutil.which("tailscale")
    if exe:
        return exe
    return None


def _is_tailscale_ip(ip: str | None) -> bool:
    """判断是否是 Tailscale CGNAT 地址（100.64.0.0/10，兼容 100.x 项目约定）。"""
    if not ip or not isinstance(ip, str):
        return False
    ip = ip.strip()
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False
    if nums[0] != 100:
        return False
    # Tailscale 官方地址池是 100.64.0.0/10；当前项目文档也用 100.x.x.x 表述。
    return 0 <= nums[1] <= 255 and all(0 <= n <= 255 for n in nums[2:])


def _detect_tailscale_ip_from_interfaces() -> str | None:
    """从网卡接口中检测 Tailscale IP，作为 tailscale CLI 瞬时失败时的兜底。"""
    try:
        import psutil
        import socket
        addrs = psutil.net_if_addrs()
        candidates = []
        for iface, addr_list in addrs.items():
            iface_l = (iface or "").lower()
            for addr in addr_list:
                if addr.family != socket.AF_INET:
                    continue
                ip = getattr(addr, "address", "")
                if not _is_tailscale_ip(ip):
                    continue
                # Tailscale 接口优先，其次接受 100.x 兜底。
                priority = 0 if "tailscale" in iface_l else 1
                candidates.append((priority, ip))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    except Exception as e:
        logger.debug(f"Tailscale 网卡扫描失败: {e}", exc_info=True)
    return None


def _check_tailscale_status() -> dict:
    """
    检查本机 Tailscale 状态。

    检测采用多来源兜底：
      1. tailscale status --json（短重试，获取 hostname/self 信息）
      2. tailscale ip -4（CLI JSON 失败时仍可返回 IP）
      3. 网卡接口 100.x / Tailscale IP 扫描

    只要检测到可用 Tailscale IP，就认为可以继续启动，避免打包版启动时
    因 tailscale 服务刚启动、CLI status 暂时非 0 等瞬时状态误弹“未连接”。
    """
    result = {
        "installed": False,
        "running": False,
        "logged_in": False,
        "tailscale_ip": None,
        "hostname": None,
        "source": "none",
        "error_detail": "",
    }
    exe = _find_tailscale_exe()
    if not exe:
        result["error_detail"] = "tailscale.exe not found"
        return result
    result["installed"] = True

    errors = []

    # 1) status --json：短重试，规避服务刚恢复/刚登录后的瞬时失败。
    for attempt in range(3):
        try:
            r = _sp.run(
                [exe, "status", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                import json
                data = json.loads(r.stdout)
                self_node = data.get("Self", {}) or {}
                ips = self_node.get("TailscaleIPs", []) or []
                ts_ip = next((ip for ip in ips if _is_tailscale_ip(ip)), ips[0] if ips else None)
                result["running"] = True
                if self_node:
                    result["logged_in"] = True
                    result["hostname"] = self_node.get("HostName")
                if ts_ip:
                    result["tailscale_ip"] = ts_ip
                    result["source"] = "status_json"
                    return result
                errors.append("status json has no Tailscale IP")
            else:
                stderr = (r.stderr or "").strip()
                errors.append(f"status --json rc={r.returncode}: {stderr[:160]}")
        except Exception as e:
            errors.append(f"status --json attempt {attempt + 1}: {e}")
        time.sleep(0.8)

    # 2) tailscale ip -4：很多情况下 status JSON 不可用但 IP 命令可用。
    try:
        r = _sp.run(
            [exe, "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            for line in r.stdout.splitlines():
                ip = line.strip()
                if _is_tailscale_ip(ip):
                    result.update({
                        "running": True,
                        "logged_in": True,
                        "tailscale_ip": ip,
                        "source": "tailscale_ip",
                    })
                    logger.info(f"Tailscale status JSON 不可用，已通过 `tailscale ip -4` 确认: {ip}")
                    return result
            errors.append("tailscale ip -4 returned no 100.x IP")
        else:
            stderr = (r.stderr or "").strip()
            errors.append(f"tailscale ip -4 rc={r.returncode}: {stderr[:160]}")
    except Exception as e:
        errors.append(f"tailscale ip -4: {e}")

    # 3) 网卡扫描：最终兜底。
    ip = _detect_tailscale_ip_from_interfaces()
    if ip:
        result.update({
            "running": True,
            "logged_in": True,
            "tailscale_ip": ip,
            "source": "interface",
        })
        logger.info(f"Tailscale CLI 暂不可用，已通过网卡/IP 检测确认可用: {ip}")
        return result

    result["error_detail"] = " | ".join(errors[-4:])
    if result["error_detail"]:
        logger.warning(f"Tailscale 状态检测未获取到 IP: {result['error_detail']}")
    return result


def _prompt_tailscale_setup(status: dict) -> bool:
    """
    首次启动时引导用户安装/加入 Tailscale 组网。

    打包版为 console=False，没有 stdin；因此优先使用 Windows 消息框，避免
    input() 在 windowed 进程中触发 "lost sys.stdin" 崩溃。

    Args:
        status: _check_tailscale_status() 的返回值

    Returns:
        True = 用户确认继续（已加入或选择暂时跳过时也返回 True）
    """
    installed = status["installed"]
    logged_in = status["logged_in"]
    ts_ip = status.get("tailscale_ip")

    # 已安装且已登录且有 IP → 静默通过
    if installed and logged_in and ts_ip:
        return True

    if not installed:
        problem = "未检测到 Tailscale。"
        steps = (
            "1. 下载并安装 Tailscale\n"
            f"   {TAILSCALE_DOWNLOAD_URL}\n\n"
            "2. 安装完成后，通过邀请链接加入组网\n"
            f"   {TAILSCALE_INVITE_URL}\n\n"
            "3. 登录后，确认系统托盘 Tailscale 图标显示 Connected。"
        )
    elif not logged_in:
        problem = "Tailscale 已安装，但未登录/未加入组网。"
        steps = (
            "请通过邀请链接完成加入：\n"
            f"{TAILSCALE_INVITE_URL}\n\n"
            "加入后确认系统托盘 Tailscale 图标显示 Connected。"
        )
    else:
        problem = "Tailscale 已登录，但未获取到 100.x IP。"
        steps = "请检查 Tailscale 网络状态，确认已 Connected。"

    message = (
        "QLH 分布式推理建议使用 Tailscale 实现跨子网互联。\n\n"
        f"当前状态：{problem}\n\n"
        f"{steps}\n\n"
        "选择“是”：打开相关链接，稍后请重新启动程序。\n"
        "选择“否”：本次先跳过检查，继续启动单机/本机 Web 服务。\n"
        "选择“取消”：退出程序。"
    )

    # 无控制台的安装包路径：使用系统对话框交互。
    if not _has_interactive_stdin():
        result = _show_dialog(
            "Tailscale 组网检查",
            message,
            "yesnocancel",
        )
        if result == _IDYES:
            if not installed:
                _open_url(TAILSCALE_DOWNLOAD_URL)
            _open_url(TAILSCALE_INVITE_URL)
            return False
        if result == _IDNO:
            logger.warning("用户选择暂时跳过 Tailscale 检查，继续启动。")
            return True
        return False

    # ---- 控制台开发模式：显示完整引导界面 ----
    print()
    print("=" * 60)
    print("  🔗 Tailscale 组网检查")
    print("=" * 60)
    print()
    print("  QLH 分布式推理需要节点间直接通信。")
    print("  由于校园网不同子网之间相互隔离，")
    print("  系统采用 Tailscale 虚拟组网实现跨子网互联。")
    print()
    print(f"  ⚠️  {problem}")
    print()
    for line in steps.splitlines():
        print(f"  {line}")
    print()
    print("─" * 60)
    print()
    print("  加入组网后，请在下面输入 yes 继续。")
    print("  输入 skip 可本次跳过检查，仅用于单机/本机 Web 服务。")
    print("  输入 no 将退出程序。")
    print()

    while True:
        choice_raw = _safe_input("  >>> 是否已加入 Tailscale 组网？(yes/skip/no): ", default="no")
        choice = (choice_raw or "no").strip().lower()

        if choice in ("yes", "y"):
            # 再次检查状态，确认用户真的加入了
            new_status = _check_tailscale_status()
            if new_status.get("logged_in") and new_status.get("tailscale_ip"):
                # 写入跳过标志，下次启动不再提示
                try:
                    os.makedirs(_TAILSCALE_FLAG_DIR, exist_ok=True)
                    with open(_TAILSCALE_FLAG_FILE, "w") as f:
                        f.write(f"joined=1\n")
                        f.write(f"tailscale_ip={new_status['tailscale_ip']}\n")
                except Exception:
                    pass
                logger.info(f"Tailscale 组网已就绪: {new_status['tailscale_ip']}")
                return True
            else:
                print()
                print("  ⚠️  仍检测不到 Tailscale 连接。请确认：")
                print("     1. Tailscale 已安装并运行")
                print("     2. 已通过邀请链接加入组网")
                print("     3. 系统托盘图标显示 Connected")
                print()
        elif choice in ("skip", "s"):
            logger.warning("用户选择暂时跳过 Tailscale 检查，继续启动。")
            return True
        elif choice in ("no", "n"):
            print()
            print("  已取消。请加入 Tailscale 组网后重新启动程序。")
            return False
        else:
            print("  请输入 yes、skip 或 no。")


def _check_tailscale_requirement() -> bool:
    """
    检查 Tailscale 组网要求。

    逻辑:
    - 若标记文件存在 → 静默快速检查（已登录 → 跳过；未登录 → 提示）
    - 若标记文件不存在 → 首次启动，显示完整引导流程

    Returns:
        True = 可以继续启动
        False = 应退出程序
    """
    # 已标记为加入 → 快速检查
    if os.path.isfile(_TAILSCALE_FLAG_FILE):
        status = _check_tailscale_status()
        if status.get("logged_in") and status.get("tailscale_ip"):
            logger.info(f"Tailscale 在线: {status['tailscale_ip']}")
            return True
        else:
            # 标记存在但 Tailscale 未运行（可能未开机自启）
            logger.warning("Tailscale 已标记但当前未连接，提示重新连接")
            print()
            print("  ⚠️  Tailscale 当前未连接。")
            print("  请确保 Tailscale 正在运行且已连接到组网。")
            print(f"  如有问题，请重新访问邀请链接: {TAILSCALE_INVITE_URL}")
            print()
            return _prompt_tailscale_setup(status)

    # 首次启动 → 完整引导
    status = _check_tailscale_status()
    return _prompt_tailscale_setup(status)


def _detect_cuda() -> bool:
    """静默检测 CUDA 可用性。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_engine_preference() -> str:
    """
    检测推荐的推理引擎。

    Returns:
        "llama_cpp" 或 "pytorch"
    """
    from config import INFERENCE_ENGINE

    if INFERENCE_ENGINE == "llama_cpp":
        return "llama_cpp"
    if INFERENCE_ENGINE == "pytorch":
        return "pytorch"

    # "auto" 模式
    if _detect_cuda():
        return "pytorch"
    return "llama_cpp"


def _has_webview() -> bool:
    """检测 pywebview 是否可用（仅 Windows 支持原生窗口）。"""
    if not IS_WINDOWS:
        return False
    try:
        import webview  # noqa: F401
        return True
    except ImportError:
        return False


def _run_ui(url: str, title: str):
    """
    启动用户界面（跨平台）。

    Linux: 使用 xdg-open 打开系统浏览器。
    Windows: 优先使用 pywebview 原生窗口，不可用时回退到外部浏览器。
    """
    if IS_LINUX or not _has_webview():
        _launch_browser(url)
    else:
        _run_pywebview(url, title)


def _run_pywebview(url: str, title: str):
    """
    Windows pywebview 原生窗口。

    如果 pywebview 不可用或启动失败，回退到外部浏览器。
    """
    try:
        import webview
    except ImportError:
        logger.warning("pywebview 未安装，回退到外部浏览器")
        _launch_browser(url)
        return

    try:
        window = webview.create_window(
            title=title,
            url=url,
            width=1200,
            height=800,
            min_size=(800, 600),
            resizable=True,
            confirm_close=False,
        )

        def on_closed():
            logger.info("窗口已关闭，程序退出。")

        window.events.closed += on_closed
        webview.start(gui='edgechromium', debug=False)
    except Exception as e:
        logger.warning(f"pywebview 启动失败 ({e})，回退到外部浏览器")
        _launch_browser(url)


def _launch_browser(url: str):
    """跨平台打开系统浏览器。Linux 用 xdg-open，Windows 用 startfile，通用回退 webbrowser。"""
    import webbrowser

    methods = []
    if IS_LINUX:
        methods.append(lambda: _sp.run(["xdg-open", url], timeout=10))
    else:
        methods.append(lambda: os.startfile(url))
        methods.append(lambda: _sp.run(["cmd", "/c", "start", url],
                                       capture_output=True, timeout=5))
    methods.append(lambda: webbrowser.open(url))

    for method in methods:
        try:
            method()
            logger.info("浏览器已打开: " + url)
            return
        except Exception:
            continue

    logger.info("请手动打开浏览器访问: " + url)
    print(f"\n{'='*60}")
    print(f"  请手动打开浏览器访问:")
    print(f"  >>> {url} <<<")
    print(f"{'='*60}\n")
    _safe_input("按 Enter 键退出...", default="")


def _kill_port_8000(port: int = 8000):
    """
    检查 API 端口。若被占用则尝试温和释放（旧实例退出后自行释放），
    不做强行杀进程操作以避免杀软误报（netstat + taskkill /F 会触发 BITS 行为检测）。
    """
    import socket

    def _port_in_use() -> bool:
        """检测 API 端口是否被占用。"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
            return True
        except (OSError, ConnectionRefusedError):
            return False
        finally:
            s.close()

    if not _port_in_use():
        return  # 端口空闲，一切正常

    # 端口已被占用 — 可能是旧实例尚未退出
    logger.warning("端口 %s 已被占用，等待旧实例释放...", port)
    print(f"  ⚠️  端口 {port} 被占用，可能是旧实例仍在运行。")
    print("     请关闭旧窗口或等待 5 秒后自动重试。")
    for i in range(5, 0, -1):
        print(f"     {i}...")
        time.sleep(1)
        if not _port_in_use():
            logger.info("端口 %s 已释放，继续启动。", port)
            return
    logger.error("端口 %s 仍被占用，启动可能失败。", port)
    print(f"  ❌ 端口 {port} 仍被占用。请手动关闭占用程序后重试。")


def main():
    """启动器主入口（跨平台）。

    CLI 参数:
      --headless    跳过浏览器/窗口，仅后台运行 API 服务器（适合 systemd / 无头部署）
      --check-only  仅检查环境，打印状态后退出（CI/测试用）
    """
    headless = "--headless" in sys.argv
    check_only = "--check-only" in sys.argv
    from config import API_PORT

    # ---- 启动前清理 ----
    _kill_port_8000(API_PORT)

    # ---- 确定引擎 ----
    engine = _detect_engine_preference()
    has_cuda = _detect_cuda()

    print("=" * 60)
    print("  轻量化大模型分布式边缘推理优化系统")
    if has_cuda:
        print("  独显版本 — PyTorch + bitsandbytes INT4")
    else:
        print("  集显版本 (CPU-only) — llama.cpp + GGUF Q4_K_M")
    print("  北京交通大学 · 大学生创新创业训练计划")
    print(f"  平台: {'Linux' if IS_LINUX else 'Windows'}")
    print("=" * 60)

    if engine == "llama_cpp":
        print("  🚀 推理引擎: llama.cpp (CPU/集显 优化)")
        print("     模型: GGUF Q4_K_M (~1.16 GB)")
        print("     预计速度: 10-15 tok/s (4核 CPU)")
    else:
        print("  🚀 推理引擎: PyTorch + bitsandbytes")
        print("     模型: Safetensors (~3.6 GB)")
        print("     量化: INT4 (~1.75 GB 显存)")
    if headless:
        print("  🤖 无头模式: API 服务器 + 无浏览器")
    print()

    # ---- 第 0 步：Tailscale 组网检查 ----
    if not _check_tailscale_requirement():
        print()
        print("按 Enter 键退出...")
        _safe_input(default="")
        sys.exit(1)
    print()

    # ---- 第 1 步：检查模型文件 ----
    from model_downloader import (
        check_and_prompt_model,
        model_exists,
        gguf_model_exists,
        safetensors_model_exists,
    )

    model_ready = check_and_prompt_model()
    if not model_ready:
        print()
        print("模型文件未就绪，程序将退出。")
        if engine == "llama_cpp":
            print("请下载 GGUF 格式模型后重新启动。")
            print("推荐: Qwen-1_8B-Chat-Q4_K_M.gguf (~1.16 GB)")
            print("下载: https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
        else:
            print("请下载 Safetensors 格式模型后重新启动。")
            print("下载: https://huggingface.co/Qwen/Qwen-1.8B-Chat")
        print()
        print("按 Enter 键退出...")
        _safe_input(default="")
        sys.exit(1)

    # ---- 报告检测结果 ----
    has_gguf = gguf_model_exists()
    has_safetensors = safetensors_model_exists()

    if has_gguf:
        logger.info("✅ GGUF 模型就绪 (llama.cpp)")
    if has_safetensors:
        logger.info("✅ Safetensors 模型就绪 (PyTorch)")

    # ---- 确认引擎选择 ----
    from model_module import ModelManager
    actual_engine = ModelManager.select_engine()
    logger.info(f"推理引擎: {actual_engine}")

    if check_only:
        print()
        print("✅ 环境检查通过。模型就绪，引擎已选择。")
        print(f"   引擎: {actual_engine}")
        print(f"   平台: {'Linux' if IS_LINUX else 'Windows'}")
        print(f"   CUDA: {'可用' if has_cuda else '不可用'}")
        return

    # ---- 第 2 步：后台启动 API 服务器 ----
    print("正在加载 API 服务...")
    _server_error = []

    def run_server():
        """在后台线程中启动 uvicorn。"""
        try:
            import uvicorn
            from api_server import app
            uvicorn.run(
                app, host="0.0.0.0", port=API_PORT,
                log_level="info",
                log_config=None,
                timeout_graceful_shutdown=10,
            )
        except Exception as e:
            _server_error.append(str(e))
            import traceback
            _server_error.append(traceback.format_exc())
            print(f"\n[ERROR] 服务器启动失败: {e}\n", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    server_thread = threading.Thread(target=run_server, daemon=True, name="uvicorn")
    server_thread.start()

    # 等待服务器就绪（最多 15 秒）
    print("等待 API 服务器就绪...")
    import urllib.request
    server_ready = False
    for i in range(150):
        time.sleep(0.1)
        if _server_error:
            print("\n服务器线程崩溃，错误信息:\n")
            print(_server_error[-1])
            print("\n按 Enter 键退出...")
            _safe_input(default="")
            sys.exit(1)
        try:
            resp = urllib.request.urlopen(f"http://localhost:{API_PORT}", timeout=0.5)
            if resp.status < 500:
                server_ready = True
                break
        except Exception:
            if i % 20 == 19:
                print(f"  等待中... ({int(i * 0.1 + 1)}s)")
    if not server_ready:
        if _server_error:
            print(f"\n服务器启动失败: {_server_error[0]}")
        else:
            print("\n服务器启动超时（15秒）。")
        print("按 Enter 键退出...")
        _safe_input(default="")
        sys.exit(1)

    print(f"API 服务器已就绪: http://localhost:{API_PORT}")

    # ---- 第 3 步：启动用户界面 ----
    if headless:
        print()
        print("🤖 无头模式: API 服务器运行中，按 Ctrl+C 退出。")
        print(f"   访问 http://localhost:{API_PORT} 打开 Web 界面。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在退出...")
    else:
        print()
        print("启动用户界面...")
        _run_ui(
            url=f"http://localhost:{API_PORT}",
            title="轻量化大模型分布式边缘推理系统",
        )

    # 窗口关闭后强制退出，避免 DB 连接池 / TCP socket 清理卡死
    print("程序已退出。")
    os._exit(0)


if __name__ == "__main__":
    main()
