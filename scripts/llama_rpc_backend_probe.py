#!/usr/bin/env python3
"""RPC backend 注入探针：用 ctypes 把 ggml-rpc-server 的 device 交给 llama.cpp 使用。

机理（源码级，见 local_docs/CORE-LLAMA-PC-RPC-01-rpc-backend-注册与-ctypes-装配-2026-09-17.md）：

  1. RPC backend **不参与** `ggml_backend_dev_count()` 枚举
     （`ggml-rpc.cpp:1958`：`GGML_ABORT("The RPC backend does not have enumerated devices
     - use ggml_backend_rpc_add_server instead")`，其 reg->context == NULL）。
  2. 必须先 `ggml_backend_rpc_add_server(endpoint)` 得到一个**带 devices 的 reg**，
     再把它 `ggml_backend_register()` 进全局注册表 —— 这是 `common/arg.cpp:1174-1181`
     的正确流程，也是上一轮 `dev_count` 恒为 1（只有 CPU）的原因。
  3. 之后 `ggml_backend_dev_get()` 里就能拿到 `RPC0`，把它写成 **NULL 结尾的数组**
     塞进 `llama_model_params.devices`，模型即经 RPC 设备加载/计算。

用法（必须用带 GGML_RPC 的 venv 解释器）：

    .venv-llama-rpc/Scripts/python.exe scripts/llama_rpc_backend_probe.py \
        --model models/minicpm4-0.5b-q4_k_m.gguf --n-predict 8 \
        --report-json local_docs/CORE-LLAMA-PC-RPC-01-ctypes-device-2026-09-17.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

GGML_BACKEND_DEVICE_TYPE = {0: "CPU", 1: "GPU", 2: "ACCEL", 3: "META"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lib-dir", default=str(REPO / ".venv-llama-rpc/Lib/site-packages/llama_cpp/lib"),
                   help="含 llama.dll / ggml*.dll 的目录（RPC 版 llama-cpp-python 的 lib/）")
    p.add_argument("--server-exe", default=str(REPO / "runtime/llama-cpp/b_rpc/ggml-rpc-server.exe"),
                   help="与 llama.dll 协议匹配的 ggml-rpc-server 可执行文件")
    p.add_argument("--model", default=str(REPO / "models/minicpm4-0.5b-q4_k_m.gguf"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=50163)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-predict", type=int, default=8)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--server-log", default=str(REPO / "logs/llama-rpc-worker.log"))
    p.add_argument("--report-json", default=None)
    p.add_argument("--keep-server", action="store_true", help="跑完不杀 worker（调试用）")
    p.add_argument("--skip-decode", action="store_true", help="只验证 device 注入，不加载模型")
    p.add_argument("--no-inject", action="store_true",
                   help="对照组：不注入 RPC device，走本机 CPU（用于数值对照）")
    p.add_argument("--via-llama", action="store_true",
                   help="走高层 llama_cpp.Llama（patch 加载调用注入 devices），验证产品路径")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="仅 --via-llama 用：Llama 的 n_gpu_layers（-1=全部层交给 devices）")
    p.add_argument("--client-threads", type=int, default=0,
                   help="client 侧 n_threads（0=库默认）")
    p.add_argument("--flash-attn", type=int, default=None, choices=[-1, 0, 1],
                   help="flash attention：-1 AUTO / 0 DISABLED / 1 ENABLED（默认 None=库默认）")
    p.add_argument("--no-extra-bufts", action="store_true",
                   help="关闭 use_extra_bufts（权重 repack kernel），用于定位数值差异来源")
    p.add_argument("--local-device", action="store_true",
                   help="把本机 CPU device 与 RPC device 一起放进 devices（分片/部分驻留）")
    p.add_argument("--tensor-split", default=None,
                   help="逗号分隔的层分配比例，顺序对应 devices（如 0.5,0.5）")
    return p.parse_args(argv)


# ---------------------------------------------------------------- dll helpers

def proc_cpu_seconds(pid: int) -> float | None:
    """取进程累计 CPU 时间（user+kernel，秒）—— 用于证明「算力真的在 worker 上」。

    仅 Windows：GetProcessTimes 返回 100ns 单位的 FILETIME。
    """
    if pid is None:
        return None
    try:
        import ctypes.wintypes as wt
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            creation, exit_t = wt.FILETIME(), wt.FILETIME()
            kernel, user = wt.FILETIME(), wt.FILETIME()
            ok = k32.GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(exit_t),
                                     ctypes.byref(kernel), ctypes.byref(user))
            if not ok:
                return None
            ticks = (kernel.dwHighDateTime << 32 | kernel.dwLowDateTime) + \
                    (user.dwHighDateTime << 32 | user.dwLowDateTime)
            return ticks / 1e7
        finally:
            k32.CloseHandle(h)
    except Exception:
        return None


def prepare_dll_search(lib_dir: pathlib.Path) -> None:
    """修掉 exit 127：让 ctypes 与 python 都能在 lib_dir 找到同目录依赖。"""
    os.add_dll_directory(str(lib_dir))
    os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")


def bind(lib_dir: pathlib.Path) -> dict:
    """加载 ggml 三件套并按签名绑定 RPC 注入所需的符号。"""
    base = ctypes.CDLL(str(lib_dir / "ggml-base.dll"))
    ggml = ctypes.CDLL(str(lib_dir / "ggml.dll"))
    rpc = ctypes.CDLL(str(lib_dir / "ggml-rpc.dll"))

    ggml.ggml_backend_load_all.argtypes = []
    ggml.ggml_backend_load_all.restype = None
    ggml.ggml_backend_reg_by_name.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_reg_by_name.restype = ctypes.c_void_p
    ggml.ggml_backend_register.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_register.restype = None
    ggml.ggml_backend_dev_count.argtypes = []
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p

    base.ggml_backend_reg_get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    base.ggml_backend_reg_get_proc_address.restype = ctypes.c_void_p
    base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_name.restype = ctypes.c_char_p
    base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_description.restype = ctypes.c_char_p
    base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
    base.ggml_backend_dev_type.restype = ctypes.c_int
    base.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
    base.ggml_backend_dev_memory.restype = None

    rpc.ggml_backend_rpc_reg.argtypes = []
    rpc.ggml_backend_rpc_reg.restype = ctypes.c_void_p

    return {"base": base, "ggml": ggml, "rpc": rpc}


def enumerate_devices(api: dict) -> list[dict]:
    ggml, base = api["ggml"], api["base"]
    out: list[dict] = []
    for i in range(ggml.ggml_backend_dev_count()):
        dev = ggml.ggml_backend_dev_get(i)
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        base.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
        out.append({
            "index": i,
            "ptr": hex(dev),
            "name": (base.ggml_backend_dev_name(dev) or b"").decode("utf-8", "replace"),
            "description": (base.ggml_backend_dev_description(dev) or b"").decode("utf-8", "replace"),
            "type": GGML_BACKEND_DEVICE_TYPE.get(base.ggml_backend_dev_type(dev), "?"),
            "total_mib": round(total.value / 2**20, 1),
            "free_mib": round(free.value / 2**20, 1),
        })
    return out


# ---------------------------------------------------------------- rpc worker

def start_worker(args: argparse.Namespace) -> tuple[subprocess.Popen, pathlib.Path]:
    exe = pathlib.Path(args.server_exe)
    if not exe.exists():
        raise SystemExit(f"ggml-rpc-server 不存在：{exe}")
    log_path = pathlib.Path(args.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    proc = subprocess.Popen(
        [str(exe), "--host", args.host, "--port", str(args.port),
         "--device", "CPU", "--threads", str(args.threads)],
        stdout=log, stderr=subprocess.STDOUT, cwd=str(exe.parent),
    )
    return proc, log_path


def wait_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def stop_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------- injection

def inject_rpc_device(api: dict, endpoint: str) -> dict:
    """严格按 common/arg.cpp:add_rpc_devices() 的顺序注入 RPC device。"""
    ggml, base, rpc = api["ggml"], api["base"], api["rpc"]
    steps: dict = {"endpoint": endpoint}

    ggml.ggml_backend_load_all()
    rpc_reg = ggml.ggml_backend_reg_by_name(b"RPC")
    steps["reg_by_name_RPC"] = hex(rpc_reg) if rpc_reg else None
    if not rpc_reg:  # 兜底：直接注册 backend 自带的 reg（无 device 的那个）
        rpc_reg = rpc.ggml_backend_rpc_reg()
        ggml.ggml_backend_register(rpc_reg)
        steps["reg_by_name_RPC"] = hex(rpc_reg)
        steps["fallback_register"] = True

    addr = base.ggml_backend_reg_get_proc_address(rpc_reg, b"ggml_backend_rpc_add_server")
    steps["add_server_fn"] = hex(addr) if addr else None
    if not addr:
        raise SystemExit("RPC backend 未导出 ggml_backend_rpc_add_server")
    add_server = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(addr)

    srv_reg = add_server(endpoint.encode())
    steps["add_server_reg"] = hex(srv_reg) if srv_reg else None
    if not srv_reg:
        raise SystemExit(f"add_server({endpoint}) 返回 NULL（worker 未就绪或协议不匹配）")

    # ★ 上一轮缺的就是这一步：add_server 返回的 reg 必须注册进全局表，
    #   否则 ggml_backend_dev_count() 看不到 RPC0。
    ggml.ggml_backend_register(srv_reg)
    steps["register_srv_reg"] = True
    return steps


def pick_rpc_device(devices: list[dict], endpoint: str) -> dict | None:
    for d in devices:
        if d["description"] == endpoint or d["name"].startswith("RPC"):
            return d
    return None


# ---------------------------------------------------------------- model run

def run_via_llama(llama_cpp, model_path: pathlib.Path, dev_ptr: int, args: argparse.Namespace) -> dict:
    """走高层 Llama 类：patch llama_model_load_from_file，让 Llama 用上 RPC device。

    llama-cpp-python 0.3.35 的 `Llama.__init__` 不接受 devices（`llama_model_params.devices`
    在绑定里被标为 "unused"），所以只能在加载调用处注入 —— `_internals.LlamaModel.__init__`
    是 `llama_cpp.llama_model_load_from_file(...)` 的属性访问，patch 模块属性即生效。
    """
    import llama_cpp.llama_cpp as lc

    dev_array = (ctypes.c_void_p * 2)(ctypes.c_void_p(dev_ptr), None)
    orig = lc.llama_model_load_from_file
    hits = {"n": 0}

    def patched(path_model, params):
        hits["n"] += 1
        params.devices = ctypes.cast(dev_array, ctypes.c_void_p).value
        return orig(path_model, params)

    lc.llama_model_load_from_file = patched
    try:
        extra: dict = {}
        if args.client_threads:
            extra["n_threads"] = args.client_threads
        if args.flash_attn is not None:
            extra["flash_attn"] = bool(args.flash_attn)
        llm = llama_cpp.Llama(model_path=str(model_path), n_ctx=args.n_ctx,
                              n_batch=min(args.n_ctx, 64), n_gpu_layers=args.n_gpu_layers,
                              verbose=True, logits_all=False, **extra)
        try:
            out = llm(args.prompt, max_tokens=args.n_predict, temperature=0.0)
            text = out["choices"][0]["text"]
            usage = out.get("usage", {})
        finally:
            llm.close()
    finally:
        lc.llama_model_load_from_file = orig
    return {"model": str(model_path), "via": "llama_cpp.Llama(patched devices)",
            "patch_hits": hits["n"], "n_gpu_layers": args.n_gpu_layers,
            "prompt": args.prompt, "completion": text, "usage": usage}


def run_model(lc, model_path: pathlib.Path, dev_ptr: int, args: argparse.Namespace,
              api: dict | None = None) -> dict:
    """把 RPC device（可与本地 CPU 组合）塞进 llama_model_params.devices 并贪心生成。"""
    import llama_cpp._ctypes_extensions  # noqa: F401  (确保 DLL 已就绪)

    # NULL 结尾的 device 数组必须保活到模型加载结束（空 -> NULL，走本机 CPU）
    dev_ptrs: list[int] = []
    if args.local_device:
        if api is None:
            raise SystemExit("--local-device 需要 ggml API 句柄")
        for index in range(api["ggml"].ggml_backend_dev_count()):
            dev = api["ggml"].ggml_backend_dev_get(index)
            if (api["base"].ggml_backend_dev_name(dev) or b"") == b"CPU":
                dev_ptrs.append(int(dev))
                break
        else:
            raise SystemExit("枚举里找不到 CPU device")
    if dev_ptr:
        dev_ptrs.append(int(dev_ptr))
    dev_array = None
    if dev_ptrs:
        dev_array = (ctypes.c_void_p * (len(dev_ptrs) + 1))()
        for i, ptr in enumerate(dev_ptrs):
            dev_array[i] = ctypes.c_void_p(ptr)
        dev_array[len(dev_ptrs)] = None

    params = lc.llama_model_default_params()
    # llama_model_params 只有 devices（NULL 结尾数组），n_ctx 属于 context params
    if dev_array is not None:
        params.devices = ctypes.cast(dev_array, ctypes.c_void_p).value
        params.n_gpu_layers = -1          # 显式 devices 时把层交给这些 device
    if args.tensor_split:
        splits = [float(x) for x in args.tensor_split.split(",")]
        ts = (ctypes.c_float * len(splits))(*splits)
        params.tensor_split = ctypes.cast(ts, ctypes.POINTER(ctypes.c_float))
    if args.no_extra_bufts:
        params.use_extra_bufts = False

    t0 = time.time()
    model = lc.llama_model_load_from_file(str(model_path).encode(), params)
    if not model:
        raise SystemExit("llama_model_load_from_file 返回 NULL")
    load_s = time.time() - t0
    desc_buf = ctypes.create_string_buffer(256)
    lc.llama_model_desc(model, desc_buf, len(desc_buf))
    result: dict = {
        "model": str(model_path),
        "model_desc": desc_buf.value.decode("utf-8", "replace"),
        "n_params": lc.llama_model_n_params(model),
        "load_s": round(load_s, 2),
    }

    vocab = lc.llama_model_get_vocab(model)
    ctx_params = lc.llama_context_default_params()
    ctx_params.n_ctx = args.n_ctx
    ctx_params.n_batch = min(args.n_ctx, 64)
    if args.client_threads:
        ctx_params.n_threads = args.client_threads
    if args.flash_attn is not None:
        ctx_params.flash_attn_type = args.flash_attn
    result["ctx"] = {
        "n_threads": int(ctx_params.n_threads),
        "n_threads_batch": int(ctx_params.n_threads_batch),
        "flash_attn_type": int(ctx_params.flash_attn_type),
    }
    ctx = lc.llama_init_from_model(model, ctx_params)
    if not ctx:
        raise SystemExit("llama_init_from_model 返回 NULL")

    def to_piece(tok: int) -> str:
        buf = ctypes.create_string_buffer(64)
        n = lc.llama_token_to_piece(vocab, tok, buf, len(buf), 0, False)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    raw = args.prompt.encode()
    n_max = len(raw) + 8
    toks = (lc.llama_token * n_max)()
    n = lc.llama_tokenize(vocab, raw, len(raw), toks, n_max, True, True)
    if n < 0:
        raise SystemExit(f"llama_tokenize 失败：需要 {abs(n)} 个 token 缓冲")
    t1 = time.time()
    if lc.llama_decode(ctx, lc.llama_batch_get_one(toks, n)) != 0:
        raise SystemExit("prefill llama_decode 失败")
    prefill_s = time.time() - t1

    sampler = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
    lc.llama_sampler_chain_add(sampler, lc.llama_sampler_init_greedy())
    pieces: list[str] = []
    token_ids: list[int] = []          # 逐 token 记录：跨路径比较用（字符比较会掩盖差异）
    t2 = time.time()
    for _ in range(args.n_predict):
        tok = lc.llama_sampler_sample(sampler, ctx, -1)
        if lc.llama_token_is_eog(vocab, tok):
            break
        lc.llama_sampler_accept(sampler, tok)
        pieces.append(to_piece(tok))
        token_ids.append(int(tok))
        one = (lc.llama_token * 1)(tok)
        if lc.llama_decode(ctx, lc.llama_batch_get_one(one, 1)) != 0:
            result["decode_error_after"] = len(pieces)
            break
    gen_s = time.time() - t2

    lc.llama_sampler_free(sampler)
    lc.llama_free(ctx)
    lc.llama_model_free(model)

    result.update({
        "prompt": args.prompt,
        "completion": "".join(pieces),
        "token_ids": token_ids,
        "n_generated": len(pieces),
        "prefill_s": round(prefill_s, 3),
        "gen_s": round(gen_s, 3),
    })
    return result


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lib_dir = pathlib.Path(args.lib_dir).resolve()
    model_path = pathlib.Path(args.model).resolve()
    if not (lib_dir / "llama.dll").exists():
        raise SystemExit(f"lib 目录里没有 llama.dll：{lib_dir}")
    if not args.skip_decode and not model_path.exists():
        raise SystemExit(f"模型不存在：{model_path}")

    # 必须在 import llama_cpp 之前修好 DLL 搜索路径（上一轮 exit 127 的根因）
    prepare_dll_search(lib_dir)
    import llama_cpp
    from llama_cpp import llama_cpp as lc

    import logging
    logging.basicConfig(level=logging.INFO, format="[llama.cpp] %(message)s")
    logging.getLogger("llama_cpp").setLevel(logging.INFO)

    report: dict = {
        "schema": "llama-rpc-ctypes-inject-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lib_dir": str(lib_dir),
        "server_exe": str(pathlib.Path(args.server_exe).resolve()),
        "endpoint": f"{args.host}:{args.port}",
    }
    api = bind(lib_dir)
    lc.llama_backend_init()
    report["supports_rpc"] = bool(lc.llama_supports_rpc())
    report["devices_before"] = enumerate_devices(api)
    print("[probe] devices before:", json.dumps(report["devices_before"], ensure_ascii=False))

    proc, log_path = (None, None) if args.no_inject else start_worker(args)
    if log_path is not None:
        report["worker_log"] = str(log_path)
    try:
        if proc is not None:
            if not wait_port(args.host, args.port):
                raise SystemExit(f"worker 未在 {args.endpoint} 监听（见 {log_path}）")
            print(f"[probe] worker ready pid={proc.pid} log={log_path}")
        else:
            print("[probe] --no-inject：不启动 worker，走本机 CPU")

        report["inject"] = {"skipped": True, "reason": "--no-inject"} if args.no_inject \
            else inject_rpc_device(api, report["endpoint"])
        print("[probe] inject:", json.dumps(report["inject"], ensure_ascii=False))

        report["devices_after"] = enumerate_devices(api)
        print("[probe] devices after :", json.dumps(report["devices_after"], ensure_ascii=False))
        rpc_dev = pick_rpc_device(report["devices_after"], report["endpoint"])
        report["rpc_device"] = rpc_dev
        if not args.no_inject and not rpc_dev:
            raise SystemExit("注入后仍未枚举到 RPC device")

        if not args.skip_decode:
            # --no-inject 时 dev_ptr=0 -> devices 保持 NULL，走本机 CPU（对照组）
            dev_ptr = int(rpc_dev["ptr"], 16) if rpc_dev else 0
            client_cpu0 = time.process_time()
            worker_cpu0 = proc_cpu_seconds(proc.pid) if proc is not None else None
            wall0 = time.perf_counter()
            if args.via_llama:
                report["run"] = run_via_llama(llama_cpp, model_path, dev_ptr, args)
            else:
                report["run"] = run_model(lc, model_path, dev_ptr, args, api)
            wall = time.perf_counter() - wall0
            client_cpu = time.process_time() - client_cpu0
            worker_cpu1 = proc_cpu_seconds(proc.pid) if proc is not None else None
            report["cpu"] = {
                "client_process_cpu_s": round(client_cpu, 3),
                "worker_process_cpu_s": None if worker_cpu0 is None or worker_cpu1 is None
                                        else round(worker_cpu1 - worker_cpu0, 3),
                "wall_s": round(wall, 3),
                "note": "worker_process_cpu_s 显著 >0 即证明算力真的落在 RPC worker 上",
            }
            print("[probe] cpu:", json.dumps(report["cpu"], ensure_ascii=False))
            print("[probe] run:", json.dumps(report["run"], ensure_ascii=False))
        report["ok"] = True
    finally:
        if proc is not None and not args.keep_server:
            stop_worker(proc)
        if args.report_json:
            out = pathlib.Path(args.report_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[probe] report -> {out}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
