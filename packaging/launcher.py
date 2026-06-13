"""
QLH 边缘推理系统 — 打包版启动器
================================
Windows 安装包的主入口点。

双引擎架构:
  1. llama.cpp + GGUF  — CPU/集显默认引擎（Q4_K_M ~1.16 GB）
  2. PyTorch + bitsandbytes — CUDA 引擎（INT4 ~1.75 GB 显存）

启动流程:
  0. Tailscale 组网检查（首次启动引导加入）
  1. 检测 CUDA 可用性 + 模型文件
  2. 若缺失 → 弹出 Windows 消息框引导下载（智能推荐格式）
  3. 模型就绪 → 自动选择最优引擎 → 后台启动 FastAPI（端口 8000）
  4. pywebview 原生窗口加载 React 前端（不依赖外部浏览器）
"""

import os
import sys
import logging
import threading
import time
import subprocess as _sp

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
TAILSCALE_DOWNLOAD_URL = "https://tailscale.com/download/windows"

# 跳过标志文件（位于应用数据目录下，首次完成后创建）
_TAILSCALE_FLAG_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "QLH-Edge-Inference")
_TAILSCALE_FLAG_FILE = os.path.join(_TAILSCALE_FLAG_DIR, ".tailscale_joined")


def _find_tailscale_exe() -> str | None:
    """查找 tailscale.exe 的完整路径，未安装返回 None。"""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Tailscale", "tailscale.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Tailscale", "tailscale.exe"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # 尝试 PATH 中查找（使用 shutil.which，避免 subprocess 触发杀软误报）
    import shutil
    exe = shutil.which("tailscale")
    if exe:
        return exe
    return None


def _check_tailscale_status() -> dict:
    """
    检查本机 Tailscale 状态。

    Returns:
        {
            "installed": bool,          # 是否已安装
            "running": bool,            # Tailscale 服务是否运行
            "logged_in": bool,          # 是否已登录
            "tailscale_ip": str | None, # Tailscale IP (100.x.y.z)
            "hostname": str | None,     # Tailscale 主机名
        }
    """
    result = {
        "installed": False,
        "running": False,
        "logged_in": False,
        "tailscale_ip": None,
        "hostname": None,
    }
    exe = _find_tailscale_exe()
    if not exe:
        return result
    result["installed"] = True

    try:
        # tailscale status — 返回 JSON（需要 --json）
        r = _sp.run(
            [exe, "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            import json
            data = json.loads(r.stdout)
            result["running"] = True
            # 检查是否已登录（有 Self 节点信息）
            self_node = data.get("Self", {})
            if self_node:
                result["logged_in"] = True
                result["tailscale_ip"] = self_node.get("TailscaleIPs", [None])[0]
                result["hostname"] = self_node.get("HostName")
    except Exception:
        pass

    return result


def _prompt_tailscale_setup(status: dict) -> bool:
    """
    首次启动时引导用户安装/加入 Tailscale 组网。

    显示 ASCII 引导界面，用户必须明确输入 yes 才能继续。
    仅在用户确认后返回 True。

    Args:
        status: _check_tailscale_status() 的返回值

    Returns:
        True = 用户确认继续（已加入或跳过时也返回 True）
    """
    installed = status["installed"]
    logged_in = status["logged_in"]
    ts_ip = status.get("tailscale_ip")

    # 已安装且已登录且有 IP → 静默通过
    if installed and logged_in and ts_ip:
        return True

    # ---- 显示引导界面 ----
    print()
    print("=" * 60)
    print("  🔗 Tailscale 组网检查")
    print("=" * 60)
    print()
    print("  QLH 分布式推理需要节点间直接通信。")
    print("  由于校园网不同子网之间相互隔离，")
    print("  系统采用 Tailscale 虚拟组网实现跨子网互联。")
    print()

    if not installed:
        print("  ⚠️  未检测到 Tailscale。请按以下步骤操作：")
        print()
        print("  第 1 步：下载并安装 Tailscale")
        print(f"    {TAILSCALE_DOWNLOAD_URL}")
        print()
        print("  第 2 步：安装完成后，打开浏览器访问邀请链接加入组网：")
        print(f"    {TAILSCALE_INVITE_URL}")
        print()
        print("  第 3 步：登录后，系统托盘会出现 Tailscale 图标。")
        print("    确保图标显示「Connected」状态。")
    elif not logged_in:
        print("  ⚠️  Tailscale 已安装但未登录/未加入组网。")
        print()
        print("  请打开浏览器访问以下邀请链接完成加入：")
        print(f"    {TAILSCALE_INVITE_URL}")
        print()
        print("  加入后确保系统托盘 Tailscale 图标显示「Connected」。")
    else:
        print("  ⚠️  Tailscale 已登录但未获取到 IP，请检查网络状态。")

    print()
    print("─" * 60)
    print()
    print("  加入组网后，请在下面输入 yes 继续。")
    print("  输入 no 将退出程序。")
    print()

    while True:
        try:
            choice = input("  >>> 是否已加入 Tailscale 组网？(yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

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
                print("     3. 系统托盘图标显示「Connected」")
                print()
        elif choice in ("no", "n"):
            print()
            print("  已取消。请加入 Tailscale 组网后重新启动程序。")
            return False
        else:
            print("  请输入 yes 或 no。")


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


def _run_pywebview(url: str, title: str):
    """
    启动 pywebview 原生窗口。

    如果 pywebview 不可用，回退到外部浏览器。
    """
    try:
        import webview
    except ImportError:
        logger.warning("pywebview 未安装，回退到外部浏览器")
        _fallback_browser(url)
        return

    # 部分系统缺少 WebView2 Runtime → 回退
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

        # 窗口关闭后清理
        def on_closed():
            logger.info("窗口已关闭，程序退出。")

        window.events.closed += on_closed
        webview.start(gui='edgechromium', debug=False)
    except Exception as e:
        logger.warning(f"pywebview 启动失败 ({e})，回退到外部浏览器")
        _fallback_browser(url)


def _fallback_browser(url: str):
    """回退方案：尝试打开外部浏览器。"""
    import subprocess as _sp
    import webbrowser

    methods = [
        lambda: os.startfile(url),                              # Windows 原生
        lambda: _sp.run(["cmd", "/c", "start", url],             # cmd start
                        capture_output=True, timeout=5),
        lambda: webbrowser.open(url),                            # 标准库
    ]
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
    # 阻塞，让用户看到提示
    try:
        input("按 Enter 键退出...")
    except (EOFError, KeyboardInterrupt):
        pass


def _kill_port_8000():
    """
    检查 8000 端口。若被占用则尝试温和释放（旧实例退出后自行释放），
    不做强行杀进程操作以避免杀软误报（netstat + taskkill /F 会触发 BITS 行为检测）。
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.bind(("127.0.0.1", 8000))
        # 端口空闲，一切正常
    except OSError:
        # 端口已被占用 — 可能是旧实例尚未退出
        logger.warning("端口 8000 已被占用，等待旧实例释放...")
        print("  ⚠️  端口 8000 被占用，可能是旧实例仍在运行。")
        print("     请关闭旧窗口或等待 5 秒后自动重试。")
        for i in range(5, 0, -1):
            print(f"     {i}...")
            time.sleep(1)
            try:
                sock.bind(("127.0.0.1", 8000))
                logger.info("端口 8000 已释放，继续启动。")
                break
            except OSError:
                if i == 1:
                    logger.error("端口 8000 仍被占用，启动可能失败。")
                    print("  ❌ 端口 8000 仍被占用。请手动关闭占用程序后重试。")
    finally:
        sock.close()


def main():
    """启动器主入口。"""

    # ---- 启动前清理 ----
    _kill_port_8000()

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
    print("=" * 60)

    if engine == "llama_cpp":
        print("  🚀 推理引擎: llama.cpp (CPU/集显 优化)")
        print("     模型: GGUF Q4_K_M (~1.16 GB)")
        print("     预计速度: 10-15 tok/s (4核 CPU)")
    else:
        print("  🚀 推理引擎: PyTorch + bitsandbytes")
        print("     模型: Safetensors (~3.6 GB)")
        print("     量化: INT4 (~1.75 GB 显存)")
    print()

    # ---- 第 0 步：Tailscale 组网检查 ----
    if not _check_tailscale_requirement():
        print()
        print("按 Enter 键退出...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
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
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
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

    # ---- 第 2 步：后台启动 API 服务器 ----
    print("正在加载 API 服务...")
    _server_error = []

    def run_server():
        """在后台线程中启动 uvicorn。"""
        try:
            import uvicorn
            from api_server import app
            # 强制 stdout/stderr 输出到控制台
            uvicorn.run(
                app, host="0.0.0.0", port=8000,
                log_level="info",
                log_config=None,  # 使用默认 logging，直接输出到 stderr
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
    for i in range(150):  # 15 秒
        time.sleep(0.1)
        # 检查服务线程是否崩了
        if _server_error:
            print("\n服务器线程崩溃，错误信息:\n")
            print(_server_error[-1])
            print("\n按 Enter 键退出...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            sys.exit(1)
        try:
            resp = urllib.request.urlopen("http://localhost:8000", timeout=0.5)
            # 任何 HTTP 响应（200/307/404/...）都说明服务器在运行
            if resp.status < 500:
                server_ready = True
                break
        except Exception:
            if i % 20 == 19:  # 每 2 秒报告一次
                print(f"  等待中... ({int(i * 0.1 + 1)}s)")
    if not server_ready:
        # 超时
        if _server_error:
            print(f"\n服务器启动失败: {_server_error[0]}")
        else:
            print("\n服务器启动超时（15秒）。")
        print("按 Enter 键退出...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)

    print("API 服务器已就绪: http://localhost:8000")

    # ---- 第 3 步：启动 pywebview 原生窗口 ----
    print()
    print("启动原生窗口...")
    _run_pywebview(
        url="http://localhost:8000",
        title="轻量化大模型分布式边缘推理系统",
    )

    # 窗口关闭后强制退出，避免 DB 连接池 / TCP socket 清理卡死
    print("程序已退出。")
    os._exit(0)


if __name__ == "__main__":
    main()
