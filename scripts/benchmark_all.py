"""
统一基准测试 — 采集所有量化+融合组合的性能数据
================================================
用法:
    python scripts/benchmark_all.py
输出:
    一张完整的性能对照表，可直接用于 README 和答辩材料
"""

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
PROMPT = "请用中文详细介绍一下北京交通大学的办学特色和优势专业。"  # 中等长度，贴近真实对话
MAX_NEW = 50
WARMUP = 1
RUNS = 3


def get_bnb(qt):
    if qt == "int4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    if qt == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def bench_one(quant_type: str, use_compile: bool) -> dict:
    label = f"{quant_type}" + (" + compile" if use_compile else "")
    logger.info(f"--- {label} ---")

    kwargs = dict(device_map="auto", trust_remote_code=True)
    bnb = get_bnb(quant_type)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"compile failed: {e}")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]

    # warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    # benchmark
    times, out_tokens = [], 0
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out_tokens = out.shape[1] - in_len

    avg_t = sum(times) / len(times)
    mem = torch.cuda.memory_allocated() / (1024**3)
    del model
    torch.cuda.empty_cache()

    return {"quant": quant_type, "compile": use_compile, "label": label,
            "avg_s": round(avg_t, 2), "tok_s": round(out_tokens / avg_t, 1),
            "mem_gb": round(mem, 2)}


def main():
    logger.info(f"{'='*70}")
    logger.info("统一基准测试 — 全部量化 × 融合组合")
    logger.info(f"{'='*70}")

    configs = [
        ("fp16", False), ("fp16", True),
        ("int8", False),
        ("int4", False),
    ]

    results = []
    for qt, comp in configs:
        # 跳过无效组合
        if qt != "fp16" and comp:
            logger.info(f"跳过 {qt}+compile (已知不兼容)")
            continue
        r = bench_one(qt, comp)
        results.append(r)
        logger.info(f"  → {r['label']:20s} | {r['avg_s']:5.2f}s | {r['tok_s']:5.1f} tok/s | {r['mem_gb']:.2f} GB")

    # 汇总表
    logger.info(f"\n{'='*70}")
    logger.info("性能汇总表")
    logger.info(f"{'='*70}")
    logger.info(f"{'配置':25s} {'显存':>8s} {'速度':>10s} {'备注'}")
    logger.info("-" * 60)

    for r in results:
        note = ""
        if r["quant"] == "int4":
            note = "⭐ 推荐边缘设备"
        elif r["quant"] == "fp16" and r["compile"]:
            note = "compile +8%"
        logger.info(f"{r['label']:25s} {r['mem_gb']:6.2f} GB {r['tok_s']:8.1f} tok/s  {note}")

    # 保存为 markdown 表格
    md = "\n## 性能基准数据\n\n"
    md += "| 配置 | 显存 | 推理速度 | 备注 |\n"
    md += "|------|------|----------|------|\n"
    for r in results:
        note = ""
        if r["quant"] == "int4":
            note = "推荐边缘设备"
        elif r["quant"] == "fp16" and r["compile"]:
            note = "compile 融合 +8%"
        md += f"| {r['label']} | {r['mem_gb']:.2f} GB | {r['tok_s']:.1f} tok/s | {note} |\n"

    logger.info(f"\n{md}")


if __name__ == "__main__":
    main()
