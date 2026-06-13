"""
QLH 边缘推理系统 — 打包版启动器
================================
Windows 安装包的主入口点。

双引擎架构:
  1. llama.cpp + GGUF  — CPU/集显默认引擎（Q4_K_M ~1.16 GB）
  2. PyTorch + bitsandbytes — CUDA 引擎（INT4 ~1.75 GB 显存）

启动流程:
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
    """清理占用 8000 端口的旧进程（防止重复启动冲突）。"""
    import subprocess as _sp
    try:
        result = _sp.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if ":8000" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = parts[-1]
                try:
                    _sp.run(["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5)
                    logger.info(f"已清理占用 8000 端口的旧进程 (PID {pid})")
                except Exception:
                    pass
    except Exception:
        pass


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
