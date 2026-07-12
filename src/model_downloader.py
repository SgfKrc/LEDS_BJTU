"""
模型下载引导程序
================
首次启动时检测模型文件，引导用户通过网盘或命令行下载。

支持双格式:
  - GGUF (llama.cpp)    → 推荐集显/CPU设备（Q4_K_M ~1.16 GB）
  - Safetensors (PyTorch) → CUDA设备（~3.6 GB）

流程:
1. 检测 models/ 目录是否包含模型文件（GGUF 或 Safetensors）
2. 若缺失 → 弹出系统对话框引导下载（Linux: zenity, Windows: MessageBox）
3. 引擎选择:
   - 有 CUDA → 引导下载 Safetensors
   - 无 CUDA → 引导下载 GGUF（推荐）
4. 用户拒绝网盘 → 命令行交互式下载（ModelScope / HuggingFace）

集成方式: api_server.py 启动时调用 check_and_prompt_model()
"""

import os
import sys
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)

IS_LINUX = sys.platform == "linux"
IS_WINDOWS = sys.platform == "win32"


def _safe_input(prompt: str = "", default: str | None = None) -> str | None:
    """input() 的安全包装，兼容 PyInstaller windowed 模式无 stdin 的场景。"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt, RuntimeError, OSError) as e:
        logger.warning(f"无法读取控制台输入: {e}")
        return default


# ================================================================
# 路径解析（兼容开发模式 + PyInstaller 打包）
# ================================================================

def _get_project_root() -> str:
    """
    获取项目根目录（models/ 的父目录）。

    开发模式: 此文件在 src/ 下 → 返回项目根目录
    PyInstaller 打包: 返回 exe 所在目录（models/ 与 exe 同级）
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller one-dir 模式：models/ 与 .exe 同级
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 开发模式：此文件在 src/ 下 → 项目根目录 = src/../
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# 模型目录和关键文件
# ================================================================

# Safetensors 格式（PyTorch）
SAFETENSORS_DIR = os.path.join(_get_project_root(), "models", "qwen-1_8b-chat")
SAFETENSORS_REQUIRED = ["config.json", "tokenizer_config.json", "qwen.tiktoken"]

# GGUF 格式（llama.cpp）
GGUF_DIR = os.path.join(_get_project_root(), "models")
GGUF_FILENAMES = [
    "qwen-1_8b-chat-Q4_K_M.gguf",     # 推荐 (小写)
    "qwen-1_8b-chat-Q5_K_M.gguf",
    "qwen-1_8b-chat-Q8_0.gguf",
    "qwen-1_8b-chat-Q4_K_S.gguf",
    "qwen-1_8b-chat-Q3_K_M.gguf",
    "Qwen-1_8B-Chat-Q4_K_M.gguf",      # HuggingFace 短横线命名
    "Qwen-1_8B-Chat-Q5_K_M.gguf",
    "Qwen-1_8B-Chat.Q4_K_M.gguf",      # HuggingFace 点号命名（实际下载）
    "Qwen-1_8B-Chat.Q5_K_M.gguf",
    "Qwen-1_8B-Chat.Q8_0.gguf",
    "Qwen-1_8B-Chat.Q4_K_S.gguf",
    "Qwen-1_8B-Chat.Q3_K_M.gguf",
]

# ================================================================
# 网盘链接
# ================================================================

PAN_LINKS = {
    "baidu": "https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3 （提取码: vtp3）",
    "baidu_gguf": "https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3 （提取码: vtp3，含 GGUF）",
    "aliyun": "https://www.alipan.com/  （请自行搜索 Qwen-1.8B-Chat 网盘资源）",
}

# ================================================================
# HuggingFace GGUF 仓库
# ================================================================

GGUF_REPOS = [
    "RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf",
    "Lemmih/Qwen-GGUF",            # 文件名: qwen-1_8b-chat-q4_k_m.gguf
]

GGUF_FILE_RECOMMENDED = "qwen-1_8b-chat-q4_k_m.gguf"

# ================================================================
# 下载命令
# ================================================================

# ModelScope — Safetensors
MODELSCOPE_CMD_SAFETENSORS = (
    'python -c "from modelscope import snapshot_download; '
    "snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')\""
)

# HuggingFace — Safetensors
HUGGINGFACE_CMD_SAFETENSORS = (
    "huggingface-cli download Qwen/Qwen-1.8B-Chat --local-dir models/qwen-1_8b-chat"
)

# HuggingFace — GGUF（推荐仓库）
HUGGINGFACE_CMD_GGUF = (
    f"huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf "
    f"Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/"
)

# ModelScope — GGUF（国内镜像，速度快）
# ModelScope 上有部分 GGUF 镜像仓库
MODELSCOPE_CMD_GGUF = (
    'python -c "from modelscope import snapshot_download; '
    "snapshot_download('RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf', "
    "local_dir='models/')\""
)


# ================================================================
# 模型检测
# ================================================================

def safetensors_model_exists() -> bool:
    """检查 Safetensors 格式模型是否存在。"""
    safetensors_dir = os.path.abspath(SAFETENSORS_DIR)
    if not os.path.isdir(safetensors_dir):
        return False

    for fname in SAFETENSORS_REQUIRED:
        if not os.path.isfile(os.path.join(safetensors_dir, fname)):
            return False

    has_weights = any(
        fname.endswith(".safetensors")
        for fname in os.listdir(safetensors_dir)
    )
    return has_weights


def gguf_model_exists() -> bool:
    """检查 GGUF 格式模型是否存在。"""
    gguf_dir = os.path.abspath(GGUF_DIR)
    if not os.path.isdir(gguf_dir):
        return False

    for fname in GGUF_FILENAMES:
        # 支持多种命名变体
        path = os.path.join(gguf_dir, fname)
        if os.path.isfile(path):
            return True

    # 另外检查 models/ 目录下任意 .gguf 文件
    try:
        for fname in os.listdir(gguf_dir):
            if fname.lower().endswith(".gguf"):
                return True
    except Exception:
        pass

    return False


def model_exists() -> bool:
    """
    检查任意格式的模型是否存在。

    优先级: GGUF > Safetensors（GGUF 更小更快，适用场景更广）
    实际引擎选择由 config.INFERENCE_ENGINE 和 model_module.select_engine() 决定。
    """
    return gguf_model_exists() or safetensors_model_exists()


def detect_cuda_available() -> bool:
    """检测 CUDA 是否可用（用于推荐下载格式）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ================================================================
# 交互界面
# ================================================================

def _windows_messagebox(title: str, message: str) -> int:
    """Windows MessageBox。返回: 6=是, 7=否, 2=取消。"""
    import ctypes
    MB_YESNOCANCEL = 0x00000003
    MB_ICONQUESTION = 0x00000020
    try:
        return ctypes.windll.user32.MessageBoxW(
            0, message, title,
            MB_YESNOCANCEL | MB_ICONQUESTION,
        )
    except Exception:
        return _cli_fallback(title, message)


def _linux_zenity_dialog(title: str, message: str) -> int:
    """Linux zenity 三按钮对话框。返回: 6=是, 7=否, 2=取消。"""
    # 简化: zenity --question 无法表达三按钮语义，用 --info + --question 组合
    # 先显示 info 告知内容，再用 question 询问是否打开网盘
    try:
        subprocess.run(
            ["zenity", "--info", "--title", title, "--text", message,
             "--width=500"],
            timeout=30,
        )
    except Exception:
        pass
    # 询问: 是否打开网盘
    try:
        rc = subprocess.run(
            ["zenity", "--question", "--title", "下载方式",
             "--text=是否打开网盘链接？\n\n选择“是”= 打开网盘链接\n选择“否”= 命令行下载",
             "--ok-label=是(网盘)", "--cancel-label=否(命令行)",
             "--width=400"],
            timeout=30,
        ).returncode
        return 6 if rc == 0 else 7  # Yes → IDYES, No → IDNO
    except Exception:
        pass
    return _cli_fallback(title, message)


def show_model_dialog(title: str, message: str) -> int:
    """
    跨平台模型下载对话框。

    返回值:
        6 = 是 (IDYES)     → 打开网盘
        7 = 否 (IDNO)      → 命令行下载
        2 = 取消 (IDCANCEL) → 退出
    """
    if IS_LINUX and shutil.which("zenity"):
        return _linux_zenity_dialog(title, message)
    elif IS_WINDOWS:
        return _windows_messagebox(title, message)
    else:
        return _cli_fallback(title, message)


def _cli_fallback(title: str, message: str) -> int:
    """命令行回退交互（无 GUI 环境）。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    clean_msg = message.replace("⚠️", "[!]").replace("📦", "[*]").replace("🔗", "[>]")
    print(clean_msg)
    print(f"{'=' * 60}")
    print("  [1] 打开网盘链接（需要手动复制到浏览器）")
    print("  [2] 命令行下载")
    print("  [3] 退出，稍后手动下载")
    print(f"{'=' * 60}")

    while True:
        choice = (_safe_input("请输入选项 [1/2/3]: ", default="3") or "3").strip()
        if choice == "1":
            return 6  # IDYES
        elif choice == "2":
            return 7  # IDNO
        elif choice == "3":
            return 2  # IDCANCEL
        print("无效选项，请输入 1、2 或 3")


def _build_message_text() -> str:
    """根据 CUDA 检测结果构建引导消息文本。"""
    has_cuda = detect_cuda_available()

    if has_cuda:
        recommended = "Safetensors 格式（~3.6 GB，支持 INT4 量化加速）"
    else:
        recommended = "GGUF Q4_K_M 格式（~1.16 GB，CPU 优化，速度 10-15 tok/s）"

    return (
        "⚠️ 未检测到模型文件\n\n"
        f"推荐下载: {recommended}\n\n"
        "📦 百度网盘（含全部格式）:\n"
        "   链接: https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3\n"
        "   提取码: vtp3\n\n"
        "🔗 HuggingFace GGUF (Q4_K_M ~1.16GB):\n"
        "   https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf\n\n"
        "点击 [是] 打开网盘链接\n"
        "点击 [否] 使用命令行下载\n"
        "点击 [取消] 退出安装"
    )


def open_pan_links():
    """打印网盘 + HuggingFace 链接到控制台，并尝试用浏览器打开。"""
    logger.info("=" * 60)
    logger.info("  模型下载 — 网盘 / 在线方式")
    logger.info("=" * 60)
    logger.info("  [百度网盘]")
    logger.info(f"    Safetensors + GGUF: {PAN_LINKS['baidu']}")
    logger.info("")
    logger.info("  [HuggingFace GGUF (推荐 CPU/集显)]")
    logger.info(f"    RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
    logger.info(f"    https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
    logger.info(f"    推荐文件: Qwen-1_8B-Chat-Q4_K_M.gguf (~1.16 GB)")
    logger.info("")
    logger.info("  [HuggingFace Safetensors (CUDA 设备)]")
    logger.info(f"    https://huggingface.co/Qwen/Qwen-1.8B-Chat")
    logger.info("")
    logger.info("=" * 60)
    logger.info("  模型文件请放入 models/ 目录。")
    logger.info("=" * 60)

    try:
        import webbrowser
        webbrowser.open("https://pan.baidu.com/s/1hAAaIN1Og-ZdeEHzxU-o4g?pwd=vtp3")
    except Exception:
        pass


def download_via_cli():
    """
    命令行下载模型。

    智能推荐:
      - 无 CUDA → 优先 GGUF（小、快）
      - 有 CUDA → 可选 Safetensors（支持 INT4）
    """
    has_cuda = detect_cuda_available()

    print("\n" + "=" * 60)
    print("  模型下载 — 命令行方式")
    print("=" * 60)

    if has_cuda:
        print("  ✅ 检测到 CUDA，可使用 INT4 量化加速")
        print("  [1] GGUF Q4_K_M (~1.16 GB) — HuggingFace")
        print("  [2] Safetensors (~3.6 GB) — HuggingFace")
        print("  [3] Safetensors — ModelScope (国内更快)")
        print("  [Q] 返回")
    else:
        print("  💡 CPU / 集显设备，推荐 GGUF 格式（速度 3-5x 快于 PyTorch CPU）")
        print("  [1] GGUF Q4_K_M (~1.16 GB) — HuggingFace (推荐)")
        print("  [2] Safetensors (~3.6 GB) — HuggingFace (不推荐：慢，需 3.5GB 内存)")
        print("  [3] Safetensors — ModelScope")
        print("  [Q] 返回")

    print("=" * 60)

    while True:
        choice = (_safe_input("请选择下载方式 [1/2/3/Q]: ", default="q") or "q").strip().lower()

        if choice == "q":
            return
        elif choice == "1":
            _download_gguf_huggingface()
            return
        elif choice == "2":
            _download_safetensors_huggingface()
            return
        elif choice == "3":
            _download_safetensors_modelscope()
            return
        else:
            print("无效选项，请输入 1、2、3 或 Q")


# ================================================================
# GGUF 下载
# ================================================================

def _download_gguf_huggingface():
    """从 HuggingFace 下载 GGUF 模型（Q4_K_M）。"""
    logger.info("正在检查 HuggingFace Hub CLI...")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.info("正在安装 HuggingFace Hub CLI...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            logger.error("安装失败，请手动执行: pip install huggingface_hub")
            return

    gguf_dir = os.path.abspath(GGUF_DIR)
    os.makedirs(gguf_dir, exist_ok=True)

    repo = "RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf"
    filename = "Qwen-1_8B-Chat-Q4_K_M.gguf"

    logger.info(f"正在从 HuggingFace 下载 GGUF 模型...")
    logger.info(f"  仓库: {repo}")
    logger.info(f"  文件: {filename} (~1.16 GB)")
    logger.info(f"  存放: {gguf_dir}/")
    logger.info("下载进度将显示在下方（可能需要 5-15 分钟）")
    logger.info("-" * 60)

    try:
        subprocess.check_call(
            [
                "huggingface-cli", "download",
                repo,
                filename,
                "--local-dir", gguf_dir,
            ],
        )
        # 重命名为更统一的命名
        dst = os.path.join(gguf_dir, "qwen-1_8b-chat-Q4_K_M.gguf")
        src = os.path.join(gguf_dir, filename)
        if os.path.isfile(src) and src != dst:
            try:
                shutil.move(src, dst)
                logger.info(f"已移动: {filename} → qwen-1_8b-chat-Q4_K_M.gguf")
            except OSError:
                pass

        logger.info("✅ GGUF 模型下载完成！")
    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败 (退出码: {e.returncode})")
        logger.error("请尝试:")
        logger.error(f"  1. 网盘下载: {PAN_LINKS['baidu']}")
        logger.error(f"  2. 手动执行: {HUGGINGFACE_CMD_GGUF}")
        logger.error(f"  3. 浏览器访问: https://huggingface.co/{repo}")


# ================================================================
# Safetensors 下载
# ================================================================

def _download_safetensors_huggingface():
    """从 HuggingFace 下载 Safetensors 模型。"""
    logger.info("正在检查 HuggingFace Hub CLI...")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.info("正在安装 HuggingFace Hub CLI...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]"],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            logger.error("安装失败，请手动执行: pip install huggingface_hub")
            return

    safetensors_dir = os.path.abspath(SAFETENSORS_DIR)
    os.makedirs(safetensors_dir, exist_ok=True)

    logger.info("正在从 HuggingFace 下载 Qwen-1.8B-Chat (~3.6GB)...")
    logger.info("国内用户可能较慢，建议使用 ModelScope 或网盘。")
    logger.info("下载进度将显示在下方（可能需要 20-60 分钟）")
    logger.info("-" * 60)

    try:
        subprocess.check_call(
            [
                "huggingface-cli", "download",
                "Qwen/Qwen-1.8B-Chat",
                "--local-dir", safetensors_dir,
            ],
        )
        logger.info("✅ Safetensors 模型下载完成！")
    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败 (退出码: {e.returncode})")
        logger.error("请尝试使用 ModelScope 或网盘下载。")


def _download_safetensors_modelscope():
    """从 ModelScope 下载 Safetensors 模型（国内推荐）。"""
    logger.info("正在检查 ModelScope SDK...")
    try:
        import modelscope  # noqa: F401
    except ImportError:
        logger.info("正在安装 ModelScope SDK...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "modelscope"],
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            logger.error("安装失败，请手动执行: pip install modelscope")
            return

    safetensors_dir = os.path.abspath(SAFETENSORS_DIR)
    os.makedirs(safetensors_dir, exist_ok=True)

    logger.info("正在从 ModelScope 下载 Qwen-1.8B-Chat (~3.6GB)...")
    logger.info("下载进度将显示在下方（可能需要 10-30 分钟）")
    logger.info("-" * 60)

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "from modelscope import snapshot_download; "
                f"snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='{safetensors_dir}')",
            ],
        )
        logger.info("✅ Safetensors 模型下载完成！")
    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败 (退出码: {e.returncode})")
        logger.error(f"手动命令: {MODELSCOPE_CMD_SAFETENSORS}")


# ================================================================
# 主入口
# ================================================================

def check_and_prompt_model() -> bool:
    """
    检查模型是否存在，若缺失则弹窗引导下载。

    应在 api_server 启动时调用（在所有模型加载操作之前）。

    Returns:
        True  = 模型已就绪或用户完成了下载
        False = 用户选择退出，不应继续启动
    """
    if model_exists():
        # 报告检测到的模型格式
        has_gguf = gguf_model_exists()
        has_safetensors = safetensors_model_exists()
        formats = []
        if has_gguf:
            formats.append("GGUF (llama.cpp)")
        if has_safetensors:
            formats.append("Safetensors (PyTorch)")
        logger.info(f"✅ 模型文件检测通过: {', '.join(formats)}")
        return True

    title = "未检测到模型文件"
    message = _build_message_text()

    logger.warning("模型文件未找到，弹出下载引导...")

    result = show_model_dialog(title, message)

    if result == 6:  # 是 → 网盘
        open_pan_links()
        logger.info("\n请在下载完成后重新启动程序。")
        logger.info("GGUF 文件请放入 models/ 目录。")
        logger.info(f"Safetensors 文件请放入: {os.path.abspath(SAFETENSORS_DIR)}")
        return False

    elif result == 7:  # 否 → CLI 下载
        download_via_cli()
        if model_exists():
            return True
        else:
            logger.warning("下载未完成或失败，请手动下载后重新启动。")
            return False

    else:  # 取消 → 退出
        logger.info("用户选择退出。请手动下载模型后重新启动。")
        logger.info(f"GGUF 目录: {os.path.abspath(GGUF_DIR)}")
        logger.info(f"Safetensors 目录: {os.path.abspath(SAFETENSORS_DIR)}")
        return False


# ================================================================
# 便捷函数：静默检查（用于非交互场景，如 API 服务内部检查）
# ================================================================

def ensure_model_or_warn() -> bool:
    """
    静默检查模型是否存在，仅打印日志警告，不弹窗。

    用于 api_server 启动时的非阻塞检查。
    """
    if model_exists():
        return True
    logger.warning("=" * 60)
    logger.warning("  ⚠️ 模型文件未找到！")
    logger.warning(f"  GGUF 目录: {os.path.abspath(GGUF_DIR)}")
    logger.warning(f"  Safetensors 目录: {os.path.abspath(SAFETENSORS_DIR)}")
    logger.warning("  请参考 README.md 的「模型下载」章节获取模型。")
    logger.warning("=" * 60)
    return False
