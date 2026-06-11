"""
模型准备脚本 — 拷贝 Qwen-1.8B 到项目 + 验证各量化模式
========================================================
用法:
    # 拷贝模型 + 验证 INT4 加载（默认）
    python scripts/quantize_model.py

    # 验证所有量化模式（fp16, int8, int4）
    python scripts/quantize_model.py --all

    # 仅拷贝，跳过验证
    python scripts/quantize_model.py --skip-verify

说明:
    bitsandbytes 的量化是"加载时量化"——权重在磁盘保持 FP16，
    加载到 GPU 时由 BitsAndBytesConfig 实时转换为 INT4/INT8。
    因此只需在项目中存一份模型权重，不同量化模式通过代码切换。
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
MODELSCOPE_CACHE = os.path.expandvars(
    r"C:\Users\Koakuma\.cache\modelscope\hub\models\Qwen\Qwen-1_8B-Chat"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "qwen-1_8b-chat")  # 统一目录，不含点号


# ============================================================
# 量化配置工厂
# ============================================================

def get_bnb_config(quant_type: str) -> BitsAndBytesConfig | None:
    """获取 bitsandbytes 量化配置（传给 from_pretrained 的 quantization_config 参数）"""
    if quant_type == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_type == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None


# ============================================================
# 模型文件拷贝
# ============================================================

def copy_model_files(src_dir: str, dst_dir: str) -> None:
    """拷贝推理所需模型文件到项目 models/ 目录"""
    os.makedirs(dst_dir, exist_ok=True)

    include_patterns = [
        ".json", ".safetensors", ".tiktoken", ".py",
        "LICENSE", "NOTICE", "README.md",
    ]

    copied, skipped = 0, 0
    for item in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, item)
        if os.path.isdir(src_path):
            skipped += 1
            continue

        should_copy = any(pat in item for pat in include_patterns)
        if should_copy:
            dst_path = os.path.join(dst_dir, item)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                size_mb = os.path.getsize(dst_path) / (1024 * 1024)
                logger.info(f"  复制: {item} ({size_mb:.1f} MB)")
                copied += 1
            else:
                skipped += 1
        else:
            skipped += 1

    logger.info(f"文件复制完成: {copied} 个, 跳过 {skipped} 个")


def ensure_no_quantization_config(model_dir: str) -> None:
    """确保 config.json 中不含 quantization_config（避免与显式传参冲突）"""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("已移除 config.json 中的旧 quantization_config")
    else:
        logger.debug("config.json 无量化配置，无需清理")


# ============================================================
# 模型加载验证
# ============================================================

def verify_model_loading(model_dir: str, quant_type: str) -> bool:
    """
    验证模型能以指定量化模式正确加载并执行推理。

    核心：通过 quantization_config 参数显式传给 from_pretrained，
    不依赖 config.json 中的配置。
    """
    logger.info(f"验证模型加载: {model_dir}  [量化={quant_type}]")

    try:
        bnb_config = get_bnb_config(quant_type)
        t0 = time.time()

        # 构建加载参数
        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        load_time = time.time() - t0

        # 模型信息
        total_params = sum(p.numel() for p in model.parameters())
        param_dtype = next(model.parameters()).dtype

        logger.info(f"  ✅ 加载成功! 耗时: {load_time:.1f}s")
        logger.info(f"  参数量: {total_params/1e9:.2f}B  |  参数类型: {param_dtype}  |  设备: {model.device}")

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"  GPU 显存: 已分配 {gpu_mem:.2f} GB / 预留 {gpu_reserved:.2f} GB")

        # 推理测试
        logger.info("  执行推理测试...")
        test_prompt = "你好，请用一句话介绍你自己。"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        t1 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        infer_time = time.time() - t1

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tok_per_sec = new_tokens / infer_time if infer_time > 0 else 0

        logger.info(f"  Prompt: {test_prompt}")
        logger.info(f"  回复: {response}")
        logger.info(f"  推理耗时: {infer_time:.2f}s  |  生成 {new_tokens} tokens  |  速度: {tok_per_sec:.1f} tok/s")

        del model, tokenizer
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        logger.error(f"  ❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen-1.8B 模型准备 & 量化验证")
    parser.add_argument("--quant", default="int4", choices=["fp16", "int8", "int4"],
                        help="验证的量化精度 (默认: int4)")
    parser.add_argument("--source", default=MODELSCOPE_CACHE,
                        help="原始模型路径")
    parser.add_argument("--output", default=MODEL_OUTPUT_DIR,
                        help="模型输出目录")
    parser.add_argument("--skip-copy", action="store_true",
                        help="跳过文件拷贝（仅验证）")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过加载验证")
    parser.add_argument("--all", action="store_true",
                        help="验证全部三种量化模式")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        logger.error(f"源模型目录不存在: {args.source}")
        sys.exit(1)

    # ---- Step 1: 拷贝模型文件 ----
    if not args.skip_copy:
        logger.info(f"{'='*60}")
        logger.info(f"[Step 1] 拷贝模型文件 → {args.output}")
        logger.info(f"{'='*60}")
        copy_model_files(args.source, args.output)
        ensure_no_quantization_config(args.output)
    else:
        logger.info("[Step 1] 跳过拷贝 (--skip-copy)")

    # ---- Step 2: 验证量化加载 ----
    if args.skip_verify:
        logger.info("[Step 2] 跳过验证 (--skip-verify)")
        return

    quant_types = ["fp16", "int8", "int4"] if args.all else [args.quant]
    results = {}

    for qt in quant_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"[Step 2] 验证: {qt}")
        logger.info(f"{'='*60}")
        results[qt] = verify_model_loading(args.output, qt)

    # ---- 总结 ----
    logger.info(f"\n{'='*60}")
    logger.info("完成! 结果摘要:")
    for qt, ok in results.items():
        logger.info(f"  {'✅' if ok else '❌'} {qt}")

    logger.info(f"\n模型目录: {args.output}")
    logger.info(f"在 config.py 中配置:")
    logger.info(f'  MODEL_PATH = "./models/qwen-1_8b-chat"')
    logger.info(f'  QUANT_TYPE = "int4"   # fp16 | int8 | int4')


if __name__ == "__main__":
    main()
