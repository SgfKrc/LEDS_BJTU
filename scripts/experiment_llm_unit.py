"""LLM 量化/编译性能单单元 runner（EX-N2，供 run_experiments.py 调用）。

按《自动化优化实验与报告方案》口径：固定提示词集、固定 seed/贪心解码、
warmup 1 次、每组 ≥5 轮取中位数；加载后强制卸载并 empty_cache。

用法（由实验计划 manifest 的 command 引用）：
    python scripts/experiment_llm_unit.py --quant fp16 --compile 0 \
        --prompt-set-dir fixtures/prompt_sets/ps-v1-zh-en-code \
        --result-file build/experiments/<run>/exp-0001.result.json \
        --runs 5 --max-prompts 6 --max-new-tokens 50
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "qwen-1_8b-chat"
WARMUP = 1


def get_bnb(qt: str):
    import torch
    from transformers import BitsAndBytesConfig
    if qt == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    if qt == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_prompts(prompt_set_dir: Path, max_prompts: int) -> list[dict]:
    prompts_file = prompt_set_dir / "prompts.jsonl"
    rows = []
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if max_prompts and len(rows) > max_prompts:
        # 均匀采样子集：保证类别覆盖
        step = len(rows) / max_prompts
        rows = [rows[int(i * step)] for i in range(max_prompts)]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=("fp16", "int8", "int4"), required=True)
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--prompt-set-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-prompts", type=int, default=0, help="0=全部")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(Path(args.prompt_set_dir), args.max_prompts)
    logger.info("quant=%s compile=%s prompts=%d runs=%d",
                args.quant, bool(args.compile), len(prompts), args.runs)

    load_t0 = time.perf_counter()
    kwargs = dict(device_map="auto", trust_remote_code=True)
    bnb = get_bnb(args.quant)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    load_s = time.perf_counter() - load_t0
    logger.info("loaded in %.1fs", load_s)

    e2e_samples: list[float] = []
    tok_samples: list[float] = []
    vram_peak = 0.0
    success = 0
    failed = 0
    total_prompt_tokens = 0

    try:
        # warmup（固定第一条）
        inputs = tokenizer(prompts[0]["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()

        for prompt in prompts:
            inputs = tokenizer(prompt["prompt"], return_tensors="pt").to(model.device)
            in_len = inputs.input_ids.shape[1]
            total_prompt_tokens += in_len
            for _ in range(args.runs):
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(
                            **inputs, max_new_tokens=args.max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id,
                        )
                    torch.cuda.synchronize()
                    e2e_ms = (time.perf_counter() - t0) * 1000.0
                    out_tokens = out.shape[1] - in_len
                    e2e_samples.append(e2e_ms)
                    tok_samples.append(out_tokens / (e2e_ms / 1000.0))
                    vram_peak = max(
                        vram_peak,
                        torch.cuda.memory_allocated() / (1024 ** 3),
                    )
                    success += 1
                except Exception as exc:  # 单元失败留痕，不中断整个单元
                    logger.warning("run failed: %s", exc)
                    failed += 1
    finally:
        del model
        torch.cuda.empty_cache()

    result = {
        "ttft_ms": None,
        "decode_tok_s": round(statistics.median(tok_samples), 2) if tok_samples else None,
        "e2e_ms": round(statistics.median(e2e_samples), 2) if e2e_samples else None,
        "peak_vram_gb": round(vram_peak, 2),
        "load_s": round(load_s, 2),
        "success": success,
        "failed": failed,
        "runs": args.runs,
        "prompt_count": len(prompts),
        "total_prompt_tokens": total_prompt_tokens,
        "independent_variable": {"quant_family": args.quant, "compile": bool(args.compile)},
    }
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    logger.info("result: %s", result)
    return 0 if success > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
