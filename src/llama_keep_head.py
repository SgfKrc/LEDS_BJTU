"""keep-head 上游：用**带补丁**的 llama.cpp 取「前 K 层输出」（`output_norm` 之前）。

为什么需要（P1 的否证 + 用户裁定 C 路线）：
`pip` 绑定的 `llama_get_embeddings_ith` 返回的是 **`output_norm(H)`**
（证据 `scripts/relay_diag_head_norm.py`：`rel_err=0.0018` / `cos=0.999998`），
比层接力上游所需的 hidden 多一次归一化 ⇒ L 段无法当上游/中间段。

项目已有补丁 `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch`
（登记在 `scripts/model_tools/llama_quantize.lock.json`，marker `★ QLH 2026-09-20`），
把 `src/llama-context.cpp` 里**早已实现但未导出**的 6 个 API 补进 `include/llama.h`：

* `llama_set_embeddings_layer_inp(ctx, lid, enable)` / `llama_get_embeddings_layer_inp(ctx, lid)`
  —— **第 lid 层的输入**（＝第 lid-1 层的输出，残差流，未经任何 norm）；
* `llama_set_embeddings_nextn(ctx, masked)` / `llama_get_embeddings_nextn_ith(ctx, i)`
  —— **末层输出（`output_norm` 之前）**，按 token 稠密存放（`masked=false`）。

Python 侧**不直接 ctypes 调 libllama**：`llama_context_params` 是按值传参的大结构体，
用 PyPI 绑定的声明去调自建 DLL 会字段错位（实测 `Unsupported ctx type`）。
因此走 C shim `scripts/model_tools/keep_head_shim/qlh_keep_head.c`（与被调 DLL 同一份
`include/llama.h` 编译，Python 侧只剩 4 个平凡签名）。

两种模式：

* ``mode="nextn"``（默认）：配 **head 裁层工件**（保留 `blk.0..K-1`）⇒ 只跑 K 层，
  末层输出即 relay 上游该交的 hidden；
* ``mode="layer_inp"``：配**整模工件** + `cut_layer=K`，取第 K 层输入（语义等价，
  但会跑满全部层，只适合数值对照）。

⚠️ 运行时依赖：shim 由 MinGW 构建 ⇒ 除同目录的 `libllama.dll` / `ggml*.dll` 外还需要
`libgcc_s_seh-1.dll` / `libstdc++-6.dll` / `libwinpthread-1.dll` 与 MSYS 的
`api-ms-win-crt-*` 副本。本模块会依次把「shim 所在目录」与
`QLH_KEEP_HEAD_DLL_DIRS`（`os.pathsep` 分隔）加入 DLL 搜索路径。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence

#: 允许额外 DLL 搜索目录（例如 `C:\msys64\ucrt64\bin`），`os.pathsep` 分隔。
EXTRA_DLL_DIRS_ENV = "QLH_KEEP_HEAD_DLL_DIRS"

#: shim 必须导出的符号（缺任何一个都说明编译产物不对）。
SHIM_SYMBOLS = ("qlh_kh_load", "qlh_kh_forward", "qlh_kh_forward_embd",
                "qlh_kh_n_embd", "qlh_kh_n_layer", "qlh_kh_close")

#: `qlh_kh_forward*` 的错误码 → 说明。
FORWARD_ERRORS = {
    -1: "参数非法（句柄 / tokens / 输出缓冲）",
    -2: "llama_decode 失败",
    -3: "llama_get_embeddings_layer_inp 返回空（该层导出没打开？）",
    -4: "llama_get_embeddings_nextn_ith 返回空（nextn 导出没打开，或该架构没把末层输出挂上？）",
    -5: "参数非法（句柄 / embd / 输出缓冲）",
}

MODE_CODES = {"nextn": 0, "layer_inp": 1}


class KeepHeadUnavailable(RuntimeError):
    """shim 缺失、符号不对或加载/初始化失败（都属于「不能用」，绝不降级成 embeddings 通道）。"""


def _add_dll_dirs(shim_dir: Path, extra: Sequence[str] = ()) -> list[str]:
    dirs: list[str] = [str(shim_dir)]
    dirs.extend(extra)
    dirs.extend(d for d in os.environ.get(EXTRA_DLL_DIRS_ENV, "").split(os.pathsep) if d)
    added: list[str] = []
    for candidate in dirs:
        if not candidate or candidate in added or not Path(candidate).is_dir():
            continue
        try:
            os.add_dll_directory(candidate)
        except (AttributeError, OSError):
            continue
        added.append(candidate)
    return added


class KeepHeadUpstream:
    """最小 keep-head 上游：吃 token → 吐「前 K 层输出」`[n_tokens, n_embd]` f32。

    只做上游该做的事：加载模型、按 token decode、导出 hidden。不提供 logits、不管采样，
    避免把实验路径膨胀成第二个引擎。
    """

    def __init__(
        self,
        shim_path: str | Path,
        model_path: str | Path,
        *,
        mode: str = "nextn",
        cut_layer: int | None = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
        extra_dll_dirs: Sequence[str] = (),
    ) -> None:
        if mode not in MODE_CODES:
            raise KeepHeadUnavailable(f"mode 必须是 {sorted(MODE_CODES)}，实得 {mode!r}")
        if mode == "layer_inp" and cut_layer is None:
            raise KeepHeadUnavailable("layer_inp 模式必须给 cut_layer")
        self.mode = mode
        self.shim_path = Path(shim_path)
        self.model_path = Path(model_path)
        if not self.shim_path.is_file():
            raise KeepHeadUnavailable(
                f"找不到 keep-head shim：{self.shim_path}（用 "
                "scripts/model_tools/build_keep_head_shim.ps1 生成）")
        if not self.model_path.is_file():
            raise KeepHeadUnavailable(f"找不到模型：{self.model_path}")

        self._dll_dirs = _add_dll_dirs(self.shim_path.parent, extra_dll_dirs)
        try:
            self._lib = ctypes.CDLL(str(self.shim_path))
        except OSError as exc:  # noqa: PERF203
            raise KeepHeadUnavailable(
                f"加载 {self.shim_path} 失败：{exc}；若缺 MinGW 运行时，请把 MSYS 的 bin "
                f"目录放进 {EXTRA_DLL_DIRS_ENV}") from exc

        missing = [name for name in SHIM_SYMBOLS if not hasattr(self._lib, name)]
        if missing:
            raise KeepHeadUnavailable(f"{self.shim_path} 缺符号 {missing}")

        lib = self._lib
        lib.qlh_kh_load.argtypes = [
            ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        lib.qlh_kh_load.restype = ctypes.c_void_p
        lib.qlh_kh_forward.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.qlh_kh_forward.restype = ctypes.c_int32
        lib.qlh_kh_forward_embd.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.qlh_kh_forward_embd.restype = ctypes.c_int32
        lib.qlh_kh_n_embd.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_n_embd.restype = ctypes.c_int32
        lib.qlh_kh_n_layer.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_n_layer.restype = ctypes.c_int32
        lib.qlh_kh_close.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_close.restype = None

        n_embd_out = ctypes.c_int32(0)
        n_layer_out = ctypes.c_int32(0)
        err = ctypes.create_string_buffer(512)
        handle = lib.qlh_kh_load(
            str(self.model_path).encode("utf-8"), int(n_ctx), int(n_threads), int(n_batch),
            MODE_CODES[mode], int(cut_layer or 0),
            ctypes.byref(n_embd_out), ctypes.byref(n_layer_out), err, len(err))
        if not handle:
            message = err.value.decode("utf-8", "replace") or "未知错误"
            raise KeepHeadUnavailable(f"keep-head 初始化失败：{message}")
        self._handle = handle
        self.n_embd = int(n_embd_out.value)
        self.n_layer = int(n_layer_out.value)
        self.cut_layer = None if cut_layer is None else int(cut_layer)
        if self.n_embd <= 0:
            self.close()
            raise KeepHeadUnavailable("shim 报告 n_embd <= 0")

    # ------------------------------------------------------------------ 前向
    def forward_tokens_to_hidden(self, tokens: Sequence[int], *, n_past: int = 0):
        """跑模型（`nextn` 模式即前 K 层），返回 `[n_tokens, n_embd]` 的 f32 hidden。"""
        import numpy as np

        toks = [int(t) for t in tokens]
        n_tokens = len(toks)
        if n_tokens == 0:
            raise ValueError("tokens 不能为空")
        tokens_arr = (ctypes.c_int32 * n_tokens)(*toks)
        out = np.zeros((n_tokens, self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward(
            self._handle, tokens_arr, n_tokens, int(n_past),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise KeepHeadUnavailable(
                f"keep-head 前向失败 rc={rc}：{FORWARD_ERRORS.get(rc, '未知错误码')}")
        return out

    def forward_hidden_to_hidden(self, hidden, *, n_past: int = 0):
        """★ **中间段能力**：吃上游 hidden（`embd` 注入）→ 吐本段的 hidden。

        这是「1 个 torch 上游 + n 个 llama 下游」链式拼接的关键：中间的 llama 段必须能
        既接受上游 hidden 又交出 hidden。当前只支持**单序列、位置连续**（`[n_past, ...)`）；
        多序列交错需要显式 positions，尚未接（写进文档的未覆盖项）。
        """
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32))
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_embd:
            raise ValueError(f"hidden 形状应为 [n_tokens, {self.n_embd}]，实得 {arr.shape}")
        n_tokens = int(arr.shape[0])
        if n_tokens == 0:
            raise ValueError("hidden 的 token 数不能为 0")

        out = np.zeros((n_tokens, self.n_embd), dtype=np.float32)
        rc = self._lib.qlh_kh_forward_embd(
            self._handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tokens, int(n_past), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise KeepHeadUnavailable(
                f"keep-head embd 前向失败 rc={rc}：{FORWARD_ERRORS.get(rc, '未知错误码')}")
        return out

    # ------------------------------------------------------------------ 资源
    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and getattr(self, "_lib", None) is not None:
            self._lib.qlh_kh_close(handle)
            self._handle = None

    def __enter__(self) -> "KeepHeadUpstream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - 解释器退出时的兜底
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
