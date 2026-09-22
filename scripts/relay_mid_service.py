#!/usr/bin/env python
"""relay_mid_service.py — 跨机接力的**远端段**服务（P2 多段拓扑的网络侧）。

链路：`torch 上游 → [网络] → 远端段 → [网络] → 下一段/末段`。
本服务提供两种远端角色，正好是 `src/relay_transport.py` 的两种往返：

| `--role` | 语义 | 协议 | 本地等价物 |
|---|---|---|---|
| `middle` | 吃 hidden → 吐 **hidden**（末层输出，`output_norm` 之前） | HIDDEN → HIDDEN | `llama_keep_head.KeepHeadUpstream.forward_hidden_to_hidden` |
| `tail` | 吃 hidden → 吐 **token**（末位 argmax） | HIDDEN → TOKEN | `llama_engine.LlamaCppEngine.forward_layers_from_hidden` |

⚠️ 语义纪律：`middle` 必须用 **keep-head 通道**（补丁导出的 nextn / 层输入），
不能用 `llama_get_embeddings_ith`（那是 `output_norm(H)`，多一次归一化 ⇒ 下游分叉）。
Android 段用同一语义的 JNI 入口：`nativeLayerForwardHiddenKeepHead`。

本服务只允许 **loopback 绑定**（`open_loopback_listener` 强制）—— 跨机时用 SSH 隧道把
远端端口映射到本机 loopback（与本仓既有 relay 纪律一致），不直接把端口暴露到 LAN/tailnet。

用法::

    # 远端（或本机另一进程）：中间段
    python scripts/relay_mid_service.py --role middle --listen 127.0.0.1:50161 \
        --keep-head-shim build/keephead/build-cpu/bin/qlh_keep_head.dll \
        --model build/cross-framework-layer-poc/out/qwen25-05b-f16-mid8-16.gguf \
        --threads 8 --ready-file build/relay-records/mid.ready
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.relay_transport import (  # noqa: E402
    RELAY_DEFAULT_MAX_TOKENS,
    open_loopback_listener,
    serve_relay_connection,
    serve_relay_middle_connection,
)


class KeepHeadMiddleRunner:
    """中间段：吃 hidden → 吐 hidden（keep-head 语义）。模型只加载一次，多次连接复用。"""

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 n_seq_max: int, n_batch: int, mode: str = "nextn",
                 cut_layer: int | None = None, extra_dll_dirs: list[str]) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx,
                                          n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self.n_seq_max = max(1, int(n_seq_max))
        self.mode = mode
        self.cut_layer = cut_layer
        self._pos = 0
        print(json.dumps({"role": "middle", "n_embd": self.n_embd,
                          "n_layer": self.n_layer, "n_seq_max": self.n_seq_max,
                          "n_batch": max(512, int(n_batch)), "mode": mode,
                          "cut_layer": cut_layer,
                          "channel": "keep_head_nextn" if mode == "nextn"
                                     else "layer_inp"}), flush=True)

    def reset(self) -> None:
        """每条连接从干净状态开始：位置归零 + **清 KV/recurrent 记忆**（同进程多连接必需）。"""
        self._pos = 0
        self._upstream.reset()

    def request_hidden(self, hidden_bytes: bytes, *, n_tokens: int) -> bytes:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        produced = self._upstream.forward_hidden_to_hidden(incoming, n_past=self._pos)
        self._pos += count
        return produced.astype(np.float32).tobytes()

    def request_hidden_seq(self, hidden_bytes: bytes, *, n_tokens: int,
                           meta: dict[str, object]) -> bytes:
        """★ P3 多序列：按帧里的 `seq_ids` / `positions` 显式绑定（不依赖本段位置累加）。

        远端每步都拿到**完整位置**，因此不需要（也不应该）维护 `self._pos`。
        """
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        produced = self._upstream.forward_hidden_to_hidden(
            incoming,
            seq_ids=meta.get("seq_ids"),
            positions=meta.get("positions"))
        return produced.astype(np.float32).tobytes()

    def close(self) -> None:
        """连接级清理：**不卸载模型**（下一个连接继续复用）。"""
        self.reset()


class TailRunner:
    """末段：吃 hidden → 吐 token（主仓 llama_engine）。"""

    def __init__(self, *, model: str, n_ctx: int, n_threads: int) -> None:
        from llama_engine import LlamaCppEngine  # noqa: PLC0415

        self._engine = LlamaCppEngine()
        self._engine.load_model(model_path=str(model), n_ctx=n_ctx, n_threads=n_threads,
                               n_seq_max=1)
        if not self._engine.is_loaded:
            raise RuntimeError(f"下游模型加载失败：{model}")
        import llama_cpp.llama_cpp as M  # noqa: PLC0415

        native = self._engine._model._model.model
        self.n_embd = int(M.llama_model_n_embd_inp(native))
        self.n_layer = int(M.llama_model_n_layer(native))
        self._pos = 0
        print(json.dumps({"role": "tail", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "llama_engine.forward_layers_from_hidden"}), flush=True)

    def reset(self) -> None:
        self._pos = 0

    def request_token(self, hidden_bytes: bytes, *, n_tokens: int) -> int:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        logits = self._engine.forward_layers_from_hidden(incoming, n_past=self._pos,
                                                         all_logits=True)
        if logits is None:
            return -1
        self._pos += count
        return int(np.asarray(logits)[-1].argmax())

    def close(self) -> None:
        self.reset()


class HeadRunner:
    """★ P4.5 **上游段**（无 PC 集群）：吃 **token** → 吐 hidden（本节点自己的 head 段）。

    为什么需要：层接力要能**完全不依赖本机**运行（集群里可能没有任何 PC / 能跑 torch 的节点）。
    上游段只需 keep-head 的 token 入口（`forward_tokens_to_hidden`，shim 早已支持），
    因此设备侧无需 pip `llama_cpp` —— 与 `middle` / `ShimTailRunner` 共用同一份 shim。

    走 `serve_relay_middle_connection` 的会话循环（它支持 `TOKENS → HIDDEN`）。
    """

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 mode: str = "nextn", cut_layer: int | None = None,
                 extra_dll_dirs: list[str], n_seq_max: int = 1,
                 n_batch: int = 512) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx, n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self._pos = 0
        print(json.dumps({"role": "head", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "keep_head.forward_tokens_to_hidden",
                          "mode": mode, "cut_layer": cut_layer}), flush=True)

    def reset(self) -> None:
        self._pos = 0
        self._upstream.reset()

    def request_hidden_from_tokens(self, tokens: list[int]) -> bytes:
        import numpy as np  # noqa: PLC0415

        hidden = self._upstream.forward_tokens_to_hidden([int(t) for t in tokens],
                                                         n_past=self._pos)
        self._pos += len(tokens)
        return np.ascontiguousarray(hidden, dtype=np.float32).tobytes()

    def close(self) -> None:
        self.reset()
        self._upstream.close()


class ShimTailRunner:
    """末段（**无 pip `llama_cpp` 的设备**）：吃 hidden → 吐 token，走 **keep-head shim**。

    为什么需要它（而不是"在设备上装 llama-cpp-python"）：

    1. `TailRunner` 依赖 pip `llama_cpp`，而 head/middle 需要的
       `llama_get_embeddings_nextn_ith` / `llama_get_embeddings_layer_inp` /
       `llama_set_embeddings_layer_inp` 这些**补丁符号在 pip 的绑定层里不存在**（实测：三个符号
       在 `llama_cpp.llama_cpp` 里全部 MISSING，而标准符号都在）⇒ 用 pip 版做 head/middle
       不仅要换它 vendor 的 llama.cpp，**还要改 `llama_cpp.py` 补声明并长期跟进上游**；
    2. shim 的 C 源与 NDK 交叉编译流程本仓已有且**已在真机验证过**，编译发生在**开发机**，
       设备只收产物 —— 边缘设备不该承担编译；
    3. 末段走同一份 shim ⇒ **同一份 C 源、同一份 llama.cpp**，与 head/middle 的数值口径天然对齐，
       不会在同一链路里混入**第二套构建配置**的 llama.cpp。

    因此设备侧统一走 shim：本类的 `request_token` 与 `TailRunner.request_token` 同语义
    （吃 hidden、吐末位 argmax），可被 `serve_relay_middle_connection` 直接替换。
    """

    def __init__(self, *, shim: str, model: str, n_ctx: int, n_threads: int,
                 mode: str = "nextn", cut_layer: int | None = None,
                 extra_dll_dirs: list[str], n_seq_max: int = 1,
                 n_batch: int = 512) -> None:
        import os  # noqa: PLC0415

        from llama_keep_head import KeepHeadUpstream  # noqa: PLC0415

        dirs = list(extra_dll_dirs)
        dirs.extend(d for d in (os.environ.get("QLH_KEEP_HEAD_DLL_DIRS") or "")
                    .split(os.pathsep) if d)
        self._upstream = KeepHeadUpstream(shim, model, mode=mode, cut_layer=cut_layer,
                                          n_ctx=n_ctx, n_threads=n_threads,
                                          n_seq_max=max(1, int(n_seq_max)),
                                          n_batch=max(512, int(n_batch)),
                                          extra_dll_dirs=dirs)
        self.n_embd = self._upstream.n_embd
        self.n_layer = self._upstream.n_layer
        self._pos = 0
        print(json.dumps({"role": "tail", "n_embd": self.n_embd, "n_layer": self.n_layer,
                          "channel": "keep_head.forward_hidden_to_token",
                          "mode": mode, "cut_layer": cut_layer}), flush=True)

    def reset(self) -> None:
        self._pos = 0
        self._upstream.reset()

    def request_token(self, hidden_bytes: bytes, *, n_tokens: int) -> int:
        import numpy as np  # noqa: PLC0415

        count = int(n_tokens)
        expected = count * int(self.n_embd) * 4
        if os.environ.get("QLH_TAIL_DIAG") == "1":
            # 诊断用（默认关闭）：定位 A 组（上游在 y700）通、B 组（上游在本机）报 rc=-5 的差异。
            # rc=-5 的语义是"embd/n_tokens 到了非法值"，所以要看的是**帧的字节数**而不是数值。
            print(json.dumps({"diag": "tail_frame", "n_tokens": count, "n_embd": self.n_embd,
                              "got_bytes": len(hidden_bytes), "expected_bytes": expected,
                              "n_past": self._pos, "ok": len(hidden_bytes) == expected}),
                  flush=True)
        if len(hidden_bytes) != expected:
            raise ValueError(
                f"tail 段收到 {len(hidden_bytes)} 字节，但 n_tokens={count} × n_embd={self.n_embd}"
                f" × 4 应为 {expected} 字节（帧与段划分不匹配）")
        incoming = np.frombuffer(hidden_bytes, dtype=np.float32).reshape(count, self.n_embd)
        token = self._upstream.forward_hidden_to_token(incoming, n_past=self._pos)
        self._pos += count
        return int(token)

    def close(self) -> None:
        self.reset()
        self._upstream.close()


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="跨机接力的远端段服务（keep-head / 末段）")
    ap.add_argument("--role", choices=("head", "middle", "tail"), default="middle")
    ap.add_argument("--listen", required=True, help="loopback 端点，host:port（跨机用 SSH 隧道）")
    ap.add_argument("--keep-head-shim", default=None, help="--role middle 必需")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-embd", type=int, default=None, help="可选：与本地期望宽度核对")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-seq-max", type=int, default=8,
                    help="★ P3：允许的并行序列上限（跨机多序列要求 ≥ 调用方 batch）")
    ap.add_argument("--n-batch", type=int, default=1024,
                    help="★ P3：batch 容量下限（≥ 调用方 batch × prefill 长度）")
    ap.add_argument("--mode", choices=("nextn", "layer_inp"), default="nextn",
                    help="中间段的 keep-head 通道：nextn（配 head 裁层工件，只跑本段层）/"
                         "layer_inp（配整模工件 + --cut-layer，取第 cut-layer 层输入；"
                         "会跑满全部层，只适合验证或没有裁层工件时）")
    ap.add_argument("--cut-layer", type=int, default=None,
                    help="--mode layer_inp 必需：切点 K（取第 K 层输入 = 前 K 层输出）")
    ap.add_argument("--max-tokens", type=int, default=RELAY_DEFAULT_MAX_TOKENS)
    ap.add_argument("--dll-dir", action="append", default=[])
    ap.add_argument("--ready-file", default=None, help="写就绪标记（含实际端点），供驱动等待")
    ap.add_argument("--max-connections", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--heartbeat-interval", type=float, default=5.0,
                    help="★ P4.5 健康检查：定期刷新 ready 文件的时间戳（秒；0 = 关闭）。"
                         "外部据此判断服务是否还活着 —— 服务跑在 ssh 会话里时会被网络抖动静默带走，"
                         "没有心跳就分不清'服务已退出'与'模型算错'")
    return ap.parse_args(argv)


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    host, _, port = str(endpoint).rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"FAIL: --listen 需要 host:port，实得 {endpoint!r}")
    return host, int(port)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    host, port = _split_endpoint(args.listen)

    if args.role == "middle":
        if not args.keep_head_shim:
            raise SystemExit("FAIL: --role middle 需要 --keep-head-shim")
        if args.mode == "layer_inp" and args.cut_layer is None:
            raise SystemExit("FAIL: --mode layer_inp 需要 --cut-layer（整模 + 切点 K）")
        runner: Any = KeepHeadMiddleRunner(shim=args.keep_head_shim, model=args.model,
                                           n_ctx=args.n_ctx, n_threads=args.threads,
                                           n_seq_max=args.n_seq_max, n_batch=args.n_batch,
                                           mode=args.mode, cut_layer=args.cut_layer,
                                           extra_dll_dirs=list(args.dll_dir))
        serve = serve_relay_middle_connection
    elif args.role == "head":
        # ★ P4.5：上游段（吃 token 吐 hidden）—— 同样走 shim，设备侧无需 pip llama_cpp。
        # 会话循环复用 middle 的那套（它支持 TOKENS → HIDDEN）。
        if not args.keep_head_shim:
            raise SystemExit("FAIL: --role head 需要 --keep-head-shim")
        runner = HeadRunner(shim=args.keep_head_shim, model=args.model,
                            n_ctx=args.n_ctx, n_threads=args.threads,
                            mode=args.mode, cut_layer=args.cut_layer,
                            extra_dll_dirs=list(args.dll_dir),
                            n_seq_max=args.n_seq_max, n_batch=args.n_batch)
        serve = serve_relay_middle_connection
    else:
        if args.keep_head_shim:
            # ★ P4.5：末段走 shim（设备侧既没有 pip llama_cpp，也不该为它维护打补丁的 fork）
            runner = ShimTailRunner(shim=args.keep_head_shim, model=args.model,
                                    n_ctx=args.n_ctx, n_threads=args.threads,
                                    mode=args.mode, cut_layer=args.cut_layer,
                                    extra_dll_dirs=list(args.dll_dir),
                                    n_seq_max=args.n_seq_max, n_batch=args.n_batch)
        else:
            runner = TailRunner(model=args.model, n_ctx=args.n_ctx, n_threads=args.threads)
        serve = serve_relay_connection

    if args.n_embd is not None and int(args.n_embd) != int(runner.n_embd):
        raise SystemExit(f"FAIL: 模型 n_embd={runner.n_embd} 与期望 {args.n_embd} 不一致")

    listener = open_loopback_listener(host, port)
    ready = {"role": args.role, "host": host, "port": port, "n_embd": runner.n_embd,
             "ready_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.ready_file:
        target = Path(args.ready_file)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _write_ready() -> None:
            payload = dict(ready)
            payload["alive_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            payload["pid"] = os.getpid()
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        _write_ready()
        if args.heartbeat_interval and float(args.heartbeat_interval) > 0:
            # ★ P4.5 健康检查：定期刷新 ready 文件的时间戳，外部据此判断"服务是否还活着"。
            # 教训：服务跑在 ssh 会话里时，一次网络抖动就会把它**静默带走**，而客户端只会
            # 看到一个与"模型层错误"难以区分的连接错误（实测踩到，见文档 §8.4）。
            # 有了心跳，"服务已退出"与"模型算错"就能被分开。
            def _beat() -> None:
                while True:
                    time.sleep(float(args.heartbeat_interval))
                    try:
                        _write_ready()
                    except Exception:  # noqa: BLE001 - 心跳写失败不应终止服务
                        pass

            threading.Thread(target=_beat, daemon=True).start()
    print(f"[ready] role={args.role} listening {host}:{port} n_embd={runner.n_embd} "
          f"(heartbeat={args.heartbeat_interval}s)", flush=True)

    served = 0
    try:
        while True:
            if args.max_connections and served >= int(args.max_connections):
                break
            sock, _addr = listener.accept()
            try:
                runner.reset()
                result = serve(sock, runner, n_embd=int(runner.n_embd),
                               max_tokens=int(args.max_tokens))
                print(f"[session] frames={result.frames} tokens={result.tokens} "
                      f"closed_cleanly={result.closed_cleanly} error={result.error or '-'}",
                      flush=True)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
            served += 1
    except KeyboardInterrupt:
        print("[stop] interrupted", flush=True)
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
