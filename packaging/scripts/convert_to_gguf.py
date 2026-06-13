#!/usr/bin/env python3
"""
Qwen-1.8B-Chat Safetensors → GGUF 模型转换工具
================================================
将 HuggingFace 格式的 Qwen 模型转换为 llama.cpp 兼容的 GGUF 格式。

依赖:
  - pip install transformers torch
  - pip install llama-cpp-python  (提供 convert-hf-to-gguf 命令)

或直接使用 llama.cpp 官方转换脚本:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  pip install -r requirements/requirements-convert_hf_to_gguf.txt
  python convert_hf_to_gguf.py /path/to/qwen-1_8b-chat --outtype q4_k_m

用法:
  # 1. 检测工具是否可用
  python convert_to_gguf.py --check

  # 2. 转换模型（默认 Q4_K_M）
  python convert_to_gguf.py --model models/qwen-1_8b-chat --outtype q4_k_m

  # 3. 转换 + 列出可用量化类型
  python convert_to_gguf.py --list-types

量化类型说明:
  Q2_K     (~0.78 GB) — 极低质量，不推荐
  Q3_K_M   (~0.94 GB) — 低质量，仅极限内存使用
  Q4_K_S   (~1.04 GB) — 较小，轻微质量损失
  Q4_K_M   (~1.16 GB) — 推荐！速度/质量最佳平衡
  Q5_K_M   (~1.31 GB) — 更高质量
  Q8_0     (~1.82 GB) — 近无损
  F16      (~3.47 GB) — 无损

参考:
  - HuggingFace 已有现成 GGUF: https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf
  - llama.cpp 转换指南: https://github.com/ggerganov/llama.cpp
"""

import os
import sys
import subprocess
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


QUANT_TYPES = {
    "q2_k":     {"size_gb": 0.78, "desc": "极低质量，不推荐"},
    "q3_k_m":   {"size_gb": 0.94, "desc": "低质量，极限内存设备"},
    "q4_k_s":   {"size_gb": 1.04, "desc": "较小，轻微质量损失"},
    "q4_k_m":   {"size_gb": 1.16, "desc": "[推荐] 速度/质量最佳平衡"},
    "q5_k_m":   {"size_gb": 1.31, "desc": "更高质量"},
    "q8_0":     {"size_gb": 1.82, "desc": "近无损"},
    "f16":      {"size_gb": 3.47, "desc": "无损（FP16 原版）"},
}

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "qwen-1_8b-chat")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")


def check_converter_available() -> bool:
    """检测是否安装了 GGUF 转换工具。"""
    # 方法 1: llama-cpp-python 内置的 convert 脚本
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llama_cpp.convert", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 or "usage" in result.stdout.lower() + result.stderr.lower():
            logger.info("[OK] llama_cpp.convert 可用")
            return True
    except Exception:
        pass

    # 方法 2: convert-hf-to-gguf.py（llama.cpp 官方脚本）
    convert_script = shutil.which("convert-hf-to-gguf.py")
    if convert_script:
        logger.info(f"[OK]  convert-hf-to-gguf.py 可用: {convert_script}")
        return True

    # 方法 3: 当前目录或项目中的 convert-hf-to-gguf.py
    for search_dir in [
        os.path.dirname(__file__),
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, "packaging"),
    ]:
        script = os.path.join(search_dir, "convert-hf-to-gguf.py")
        if os.path.isfile(script):
            logger.info(f"[OK]  本地脚本可用: {script}")
            return True

    logger.warning("[FAIL]  未检测到 GGUF 转换工具")
    return False


def list_quant_types():
    """列出所有可用量化类型。"""
    print("\n" + "=" * 60)
    print("  GGUF 量化类型")
    print("=" * 60)
    print(f"  {'类型':<12} {'大小':>8}   说明")
    print("  " + "-" * 56)
    for qtype, info in QUANT_TYPES.items():
        print(f"  {qtype:<12} {info['size_gb']:>5.1f} GB   {info['desc']}")
    print("=" * 60)
    print()


def convert_model(model_dir: str = None, outtype: str = "q4_k_m",
                  output_dir: str = None) -> bool:
    """
    将 HuggingFace Safetensors 模型转换为 GGUF。

    Args:
        model_dir: 源模型目录（含 config.json, model.safetensors 等）
        outtype: 输出量化类型 (q4_k_m / q5_k_m / q8_0 / f16)
        output_dir: 输出目录（默认 models/）

    Returns:
        转换是否成功
    """
    model_dir = model_dir or DEFAULT_MODEL_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    if not os.path.isdir(model_dir):
        logger.error(f"源模型目录不存在: {model_dir}")
        return False

    # 检查必要文件
    required = ["config.json", "tokenizer_config.json"]
    for fname in required:
        if not os.path.isfile(os.path.join(model_dir, fname)):
            logger.error(f"缺少必要文件: {fname}")
            return False

    # 检查是否有模型权重
    has_weights = any(
        fname.endswith(".safetensors") or fname.endswith(".bin")
        for fname in os.listdir(model_dir)
    )
    if not has_weights:
        logger.error(f"模型目录未找到权重文件 (.safetensors / .bin): {model_dir}")
        return False

    outtype = outtype.lower().replace("-", "_")
    if outtype not in QUANT_TYPES:
        logger.error(f"未知量化类型: {outtype}，可用类型:")
        list_quant_types()
        return False

    output_name = f"qwen-1_8b-chat-{outtype.upper()}.gguf"

    logger.info(f"源模型: {model_dir}")
    logger.info(f"量化类型: {outtype} ({QUANT_TYPES[outtype]['desc']})")
    logger.info(f"预计大小: ~{QUANT_TYPES[outtype]['size_gb']:.2f} GB")
    logger.info(f"输出文件: {os.path.join(output_dir, output_name)}")

    os.makedirs(output_dir, exist_ok=True)

    # 方法 1: llama-cpp-python 内置转换
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "llama_cpp.convert",
                model_dir,
                "--outtype", outtype,
                "--outfile", os.path.join(output_dir, output_name),
            ],
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            logger.info(f"[OK]  转换完成: {os.path.join(output_dir, output_name)}")
            return True
    except Exception as e:
        logger.debug(f"llama_cpp.convert 方法失败: {e}")

    # 方法 2: convert-hf-to-gguf.py 命令行
    convert_script = shutil.which("convert-hf-to-gguf.py")
    if not convert_script:
        # 尝试常见位置
        for search_dir in [
            os.path.dirname(__file__),
            PROJECT_ROOT,
        ]:
            candidate = os.path.join(search_dir, "convert-hf-to-gguf.py")
            if os.path.isfile(candidate):
                convert_script = candidate
                break

    if convert_script:
        logger.info(f"使用转换脚本: {convert_script}")
        result = subprocess.run(
            [
                sys.executable, convert_script,
                model_dir,
                "--outtype", outtype,
                "--outfile", os.path.join(output_dir, output_name),
            ],
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            logger.info(f"[OK]  转换完成: {os.path.join(output_dir, output_name)}")
            return True
        else:
            logger.error(f"转换失败 (退出码: {result.returncode})")
            return False

    # 无可用工具
    logger.error("=" * 60)
    logger.error("未找到可用的 GGUF 转换工具。请选择以下方案之一:")
    logger.error("")
    logger.error("方案 1: 安装 llama-cpp-python（推荐）")
    logger.error("  pip install llama-cpp-python")
    logger.error(f"  python -m llama_cpp.convert {model_dir} --outtype {outtype}")
    logger.error("")
    logger.error("方案 2: 从 HuggingFace 直接下载已转换的 GGUF")
    logger.error("  https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
    logger.error(f"  下载 Qwen-1_8B-Chat-Q4_K_M.gguf → {output_dir}/")
    logger.error("")
    logger.error("方案 3: 使用 llama.cpp 官方转换脚本")
    logger.error("  git clone https://github.com/ggerganov/llama.cpp")
    logger.error("  cd llama.cpp && pip install -r requirements/requirements-convert_hf_to_gguf.txt")
    logger.error(f"  python convert_hf_to_gguf.py {model_dir} --outtype {outtype}")
    logger.error("=" * 60)
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen-1.8B-Chat → GGUF 模型转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --check              检测转换工具是否可用
  %(prog)s --list-types         列出所有量化类型
  %(prog)s --model models/qwen-1_8b-chat --outtype q4_k_m
  %(prog)s --model models/qwen-1_8b-chat --outtype q5_k_m --output models/
        """,
    )
    parser.add_argument("--check", action="store_true", help="检测转换工具是否可用")
    parser.add_argument("--list-types", action="store_true", help="列出所有量化类型")
    parser.add_argument("--model", type=str, default=None,
                        help=f"源模型目录（默认: {DEFAULT_MODEL_DIR}）")
    parser.add_argument("--outtype", type=str, default="q4_k_m",
                        help="量化类型 (q4_k_m / q5_k_m / q8_0 / f16)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认: models/）")

    args = parser.parse_args()

    if args.check:
        ok = check_converter_available()
        if ok:
            list_quant_types()
        else:
            print("\n[提示] 可以从 HuggingFace 直接下载已转换的 GGUF 文件:\n"
                  "   https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
        sys.exit(0 if ok else 1)

    if args.list_types:
        list_quant_types()
        sys.exit(0)

    # 转换模式
    if not args.model and not os.path.isdir(DEFAULT_MODEL_DIR):
        logger.error("请指定源模型目录 (--model) 或将模型放入 models/qwen-1_8b-chat/")
        logger.info("也可以直接下载已转换的 GGUF:")
        logger.info("  https://huggingface.co/RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf")
        sys.exit(1)

    success = convert_model(
        model_dir=args.model,
        outtype=args.outtype,
        output_dir=args.output,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
