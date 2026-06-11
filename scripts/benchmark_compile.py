"""
算子融合基准测试 — 对比 torch.compile 前后的推理速度
======================================================
用法:
    python scripts/benchmark_compile.py              # INT4 模型
    python scripts/benchmark_compile.py --quant fp16  # FP16 模型
"""

import argparse
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen-1_8b-chat")
WARMUP_RUNS = 2       # 预热轮数（JIT 编译）
BENCH_RUNS = 5         # 正式测试轮数
PROMPT = "请用中文详细介绍一下北京交通大学的办学特色和优势专业。"  # 较长 prompt


def get_bnb_config(quant_type: str):
    if quant_type == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif quant_type == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(model_path: str, quant_type: str, use_compile: bool):
    """加载模型，可选开启 torch.compile"""
    logger.info(f"加载模型: quant={quant_type}, compile={use_compile}")
    kwargs = dict(device_map="auto", trust_remote_code=True)
    bnb = get_bnb_config(quant_type)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_compile:
        logger.info("  启用 torch.compile (mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("  compile 成功")
        except Exception as e:
            logger.warning(f"  compile 失败: {e}")

    return model, tokenizer


def benchmark(model, tokenizer, label: str) -> dict:
    """运行基准测试，返回性能指标"""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # 预热
    logger.info(f"  [{label}] 预热中 ({WARMUP_RUNS} 轮)...")
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # 正式测试
    logger.info(f"  [{label}] 基准测试中 ({BENCH_RUNS} 轮)...")
    times = []
    new_tokens = 0
    for i in range(BENCH_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        new_tokens = out.shape[1] - input_len
        logger.info(f"    第{i+1}轮: {elapsed:.2f}s → {new_tokens} tokens ({new_tokens/elapsed:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    avg_tok_per_s = new_tokens / avg_time
    gpu_mem = torch.cuda.memory_allocated() / (1024**3)

    return {
        "label": label,
        "avg_time_s": round(avg_time, 2),
        "std_time_s": round(torch.tensor(times).std().item(), 3),
        "tok_per_s": round(avg_tok_per_s, 1),
        "gpu_mem_gb": round(gpu_mem, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", default="int4", choices=["fp16", "int8", "int4"])
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    logger.info(f"{'='*60}")
    logger.info(f"算子融合基准测试 — {args.quant}")
    logger.info(f"{'='*60}")

    # ---- 无融合 ----
    model_no, tok = load_model(args.model, args.quant, use_compile=False)
    result_no = benchmark(model_no, tok, "无融合")
    del model_no
    torch.cuda.empty_cache()

    # ---- torch.compile ----
    model_comp, tok = load_model(args.model, args.quant, use_compile=True)
    result_comp = benchmark(model_comp, tok, "compile融合")
    del model_comp
    torch.cuda.empty_cache()

    # ---- 对比 ----
    logger.info(f"\n{'='*60}")
    logger.info("对比结果:")
    logger.info(f"{'='*60}")

    for r in [result_no, result_comp]:
        logger.info(f"  {r['label']:12s} | 平均 {r['avg_time_s']}s | "
                    f"速度 {r['tok_per_s']} tok/s | 显存 {r['gpu_mem_gb']} GB")

    speedup = result_comp["tok_per_s"] / result_no["tok_per_s"]
    direction = "🚀 加速" if speedup > 1.01 else ("🐌 变慢" if speedup < 0.99 else "➡️  持平")
    logger.info(f"\n  compile 效果: {speedup:.2f}x {direction}")


if __name__ == "__main__":
    main()
