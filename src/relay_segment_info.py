"""relay_segment_info.py — 段构建标识（供 relay 记录与对账；§10.2 待办）。

**为什么要它**：relay 记录里的 `engines` 过去是**固定字符串**（"纯 llama.cpp（无 torch / 无 D 档
组件参与推理）"）、`ifaces` 只写模块名 ⇒ 从记录本身**无法回答**「这一段实际用哪套 llama.cpp 构建、
段工件是哪一份」。这不是理论问题：tail 段走 **pip 绑定**还是**自建 shim**，直接决定同一 head 段下
是 1/32 还是 32/32（见 `docs/跨框架接力…§10.7`），而记录里当时分辨不出来。

三个入口：

* :func:`collect_local_build` —— **本进程段**：shim 与同目录 `libllama`/`ggml*` 的 sha256
  （小文件，很快）、pip `llama_cpp` 版本与其 `lib/llama.dll` 摘要、段工件的大小/名字；
  **GB 级段工件默认不算 sha256**（会明显拖慢每轮实验），需要时用 `digest_artifacts=True` 显式开启；
* :func:`load_ready_build` —— **远端段**：读服务端 ready 文件里的 `build` 字段（`relay_mid_service`
  已在写）；读不到/字段缺失时**如实标注**原因，不抛异常；
* :func:`unknown_remote_build` —— 远端既没给 ready 文件也没法读时**显式标注** `remote_unknown`：
  记录必须能区分「未知」与「没写这个字段」。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "RELAY_SEGMENT_INFO_SCHEMA",
    "collect_local_build",
    "load_ready_build",
    "unknown_remote_build",
]

#: 段构建标识的 schema 版本（记录里逐段出现）。
RELAY_SEGMENT_INFO_SCHEMA = "qlh.relay_segment_info.v1"

#: 与 llama.cpp 构建同源的动态库（这些才是真正决定数值的构建产物）。
_LLAMA_LIBRARIES = ("libllama.dll", "llama.dll", "libllama.so")
_GGML_LIBRARIES = ("ggml-base.dll", "ggml.dll", "libggml-base.so", "libggml.so")

#: 超过这个大小就不默认算 sha256（段工件事常是 GB 级）。
_DIGEST_SIZE_LIMIT = 64 * 1024 * 1024


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_info(path: Path, *, digest: bool) -> dict[str, Any]:
    """单个文件的标识：名字/大小/mtime；`digest=True` 且文件不大时附 sha256。"""
    info: dict[str, Any] = {"path": str(path), "name": path.name, "exists": path.is_file()}
    if not info["exists"]:
        return info
    stat = path.stat()
    info["size"] = int(stat.st_size)
    info["mtime"] = int(stat.st_mtime)
    if digest and stat.st_size <= _DIGEST_SIZE_LIMIT:
        info["sha256"] = _sha256_file(path)
    else:
        # 大工件默认不算：GB 级 sha256 会拖慢每轮实验；要算请显式开 digest_artifacts。
        info["sha256"] = None
    return info


def collect_local_build(
    *,
    shim: str | Path | None = None,
    model: str | Path | None = None,
    llama_cpp_module: Any = None,
    digest_artifacts: bool = False,
) -> dict[str, Any]:
    """收集**本进程段**的构建标识。

    Args:
        shim: keep-head shim 路径；给了就同时记录同目录的 `libllama` / `ggml*`（构建同源）。
        model: 段工件（裁层 GGUF / 整模）路径。
        llama_cpp_module: 已导入的 `llama_cpp` 模块（可选）。**本函数不会自行 import 它** ——
            隐式导入会污染 `sys.modules`、进而改变 keep-head 的进程隔离判据（实测踩到）。
        digest_artifacts: 对**段工件**也算 sha256（GB 级会明显变慢）。
    """
    build: dict[str, Any] = {
        "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
        "collected_at": _utc_now(),
    }
    if shim is not None:
        shim_path = Path(shim)
        build["shim"] = _file_info(shim_path, digest=True)
        siblings = {
            name: _file_info(shim_path.parent / name, digest=True)
            for name in (*_LLAMA_LIBRARIES, *_GGML_LIBRARIES)
            if (shim_path.parent / name).is_file()
        }
        if siblings:
            build["llama_cpp_build"] = siblings

    # ⚠️ **不做隐式 import**：调用方可能只是想记 shim 摘要，而在主进程里 `import llama_cpp`
    #    会**改变 keep-head 的隔离判据**（一旦 `llama_cpp` 进了 `sys.modules`，后续
    #    `KeepHeadUpstream` 会改走 worker 子进程）⇒ 这种副作用不可接受（实测踩到：
    #    `relay_mid_service._runner_build` 曾因此把 tail 段打成 `runner_failed`）。
    module = llama_cpp_module
    if module is not None:
        build["llama_cpp_version"] = getattr(module, "__version__", None)
        module_file = getattr(module, "__file__", None)
        lib_dir = Path(module_file).parent / "lib" if module_file else None
        if lib_dir is not None and lib_dir.is_dir():
            libs = {
                name: _file_info(lib_dir / name, digest=True)
                for name in (*_LLAMA_LIBRARIES, *_GGML_LIBRARIES)
                if (lib_dir / name).is_file()
            }
            if libs:
                build["llama_cpp_libs"] = libs

    if model is not None:
        build["model"] = _file_info(Path(model), digest=bool(digest_artifacts))
    return build


def load_ready_build(path: str | Path) -> dict[str, Any]:
    """读服务端 ready 文件里的 `build` 段；读不到就**如实标注原因**（不抛）。"""
    ready_path = Path(path)
    try:
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 读不到就标 unknown
        return {
            "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
            "source": "ready_file_unreadable",
            "ready_file": str(ready_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    build = payload.get("build")
    if not isinstance(build, dict):
        return {
            "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
            "source": "ready_file_without_build",
            "ready_file": str(ready_path),
            "role": payload.get("role"),
        }
    result = dict(build)
    result.setdefault("schema_version", RELAY_SEGMENT_INFO_SCHEMA)
    result["source"] = "ready_file"
    result["ready_file"] = str(ready_path)
    result["role"] = payload.get("role")
    return result


def unknown_remote_build(endpoint: str | None) -> dict[str, Any]:
    """远端段未提供构建标识时的**显式**占位（`remote_unknown`）。"""
    return {
        "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
        "source": "remote_unknown",
        "endpoint": endpoint,
    }
