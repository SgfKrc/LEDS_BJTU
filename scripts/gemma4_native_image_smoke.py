"""G4.3.2B 原生图片语义验证：MTMD 图像管线（参照 mtmd-cli.cpp）。

用法（在 .venv-gemma4-native 下运行，需 PATH 含 C:/msys64/ucrt64/bin）：
  python verify_image_semantics.py <gguf> <mmproj> <image> [--prompt ...]
"""
import argparse
import os
import sys
import time
from ctypes import byref, c_void_p

os.add_dll_directory(r"C:\msys64\ucrt64\bin")

from llama_cpp import Llama, llama_cpp
import llama_cpp.mtmd_cpp as m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("mmproj")
    ap.add_argument("image")
    ap.add_argument("--prompt", default="Describe this image in one or two sentences. <__media__>")
    ap.add_argument("--n-ctx", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    t0 = time.time()
    model = Llama(
        model_path=args.gguf,
        n_ctx=args.n_ctx,
        n_batch=64,
        n_gpu_layers=0,
        verbose=False,
    )
    print(f"[1] 主模型加载完成 {time.time()-t0:.1f}s", flush=True)

    mtmd_ctx = m.mtmd_init_from_file(
        args.mmproj.encode("utf-8"), model.model, m.mtmd_context_params_default()
    )
    if not mtmd_ctx:
        print("[x] mtmd_init_from_file 返回空", file=sys.stderr)
        return 1
    print(f"[2] MTMD 上下文初始化完成（vision={m.mtmd_support_vision(mtmd_ctx)}）", flush=True)

    marker = m.mtmd_default_marker()
    prompt = args.prompt.replace("<__media__>", marker.decode("utf-8", "replace"))
    print(f"[3] 提示词: {prompt!r}", flush=True)

    wrapper = m.mtmd_helper_bitmap_init_from_file(mtmd_ctx, args.image.encode("utf-8"), False)
    bitmap = wrapper.bitmap if wrapper else None
    if not bitmap:
        print("[x] bitmap 构造失败", file=sys.stderr)
        return 1
    print(f"[4] bitmap 构造完成（{os.path.getsize(args.image)} bytes）", flush=True)

    chunks = m.mtmd_input_chunks_init()
    text = m.mtmd_input_text(prompt.encode("utf-8"), True, True)
    bitmaps_arr = (m.mtmd_bitmap_p_ctypes * 1)(bitmap)
    rc = m.mtmd_tokenize(mtmd_ctx, chunks, byref(text), bitmaps_arr, 1)
    if rc != 0:
        print(f"[x] mtmd_tokenize 失败 rc={rc}", file=sys.stderr)
        return 1
    n_chunks = m.mtmd_input_chunks_size(chunks)
    print(f"[5] tokenize 完成：{n_chunks} 个 chunk", flush=True)

    n_past = llama_cpp.llama_pos(0)
    n_batch = 64
    batch = None

    @m.mtmd_helper_post_decode_callback
    def _noop_callback(_batch, _user_data):  # type: ignore[no-untyped-def]
        return 0
    for i in range(n_chunks):
        chunk = m.mtmd_input_chunks_get(chunks, i)
        ctype = m.mtmd_input_chunk_get_type(chunk)
        new_n_past = llama_cpp.llama_pos(n_past.value)
        if ctype == m.MTMD_INPUT_CHUNK_TYPE_TEXT:
            rc = m.mtmd_helper_eval_chunk_single(
                mtmd_ctx, model._ctx.ctx, chunk, n_past, 0, n_batch, False, byref(new_n_past)
            )
            if rc != 0:
                print(f"[x] 文本 chunk {i} eval 失败 rc={rc}", file=sys.stderr)
                return 1
        else:
            if batch is None:
                batch = m.mtmd_batch_init(mtmd_ctx)
            rc = m.mtmd_batch_add_chunk(batch, chunk)
            if rc != 0:
                print(f"[x] batch add chunk {i} 失败 rc={rc}", file=sys.stderr)
                return 1
            rc = m.mtmd_batch_encode(batch)
            if rc != 0:
                print(f"[x] batch encode 失败 rc={rc}", file=sys.stderr)
                return 1
            embd = m.mtmd_batch_get_output_embd(batch, chunk)
            if not embd:
                print("[x] 未取得 media embd", file=sys.stderr)
                return 1
            rc = m.mtmd_helper_decode_image_chunk(
                mtmd_ctx, model._ctx.ctx, chunk, embd, n_past, 0, n_batch,
                byref(new_n_past), _noop_callback, None,
            )
            if rc != 0:
                print(f"[x] 图像 chunk {i} decode 失败 rc={rc}", file=sys.stderr)
                return 1
        n_past = new_n_past
        print(f"    chunk {i} (type={ctype}) 已处理，n_past={n_past}", flush=True)

    print(f"[6] 图像编码+解码完成，开始生成（n_past={n_past}）", flush=True)
    from llama_cpp._internals import LlamaBatch
    CHANNEL_TOKEN_ID = 101  # "<channel|>"：思考段与正文的分隔特殊 token（gemma4）
    tokens: list[int] = []
    collecting = False
    last = llama_cpp.llama_token_bos(model._ctx.ctx)
    n_past = int(n_past.value)  # chunk 循环结束的实际 KV 位置
    batch = LlamaBatch(n_tokens=1, embd=0, n_seq_max=1, verbose=False)
    for _ in range(args.max_tokens):
        batch.set_batch([last], n_past, False)
        rc = model._ctx.decode(batch)
        if rc not in (None, 0):
            print(f"[x] decode 失败 rc={rc}（n_past={n_past}，token={last}）", file=sys.stderr)
            break
        n_past += 1
        smpl = llama_cpp.llama_sampler_init_greedy()
        tok = llama_cpp.llama_sampler_sample(smpl, model._ctx.ctx, -1)
        llama_cpp.llama_sampler_free(smpl)
        if tok == llama_cpp.llama_token_eos(model._ctx.ctx):
            break
        if tok == CHANNEL_TOKEN_ID:
            # 思考段结束标记：丢弃此前全部内容（含 "thought" 字面），开始收集正文
            collecting = True
            continue
        if collecting:
            tokens.append(tok)
        last = tok
    if not collecting:
        print(
            "[!] 未检测到 <channel|>（思考段未结束或模型未输出正文）："
            "原生路径无 reasoning_effort 等效控制，思考段长度不可控；"
            "产品图像路径请使用 Ollama external_api（reasoning_effort=none）",
            file=sys.stderr,
        )

    text_out = model.detokenize(tokens).decode("utf-8", "replace")
    print(f"[7] 生成完成（{len(tokens)} tokens，{time.time()-t0:.1f}s）", flush=True)
    print("=== 原生描述输出 ===")
    print(text_out)
    print("====================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
