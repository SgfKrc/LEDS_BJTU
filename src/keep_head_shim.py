"""keep-head shim 的**唯一定位入口**（`#53`）。

背景
----
master 侧（`llama_engine`）在缺 `QLH_KEEP_HEAD_SHIM` 时会**按仓库默认推导**
（`build/keephead/build-cpu/bin/qlh_keep_head.dll`），而 worker 侧
（`inference_service.engine_host`）此前**只读环境变量** ⇒ 同一台机器上会出现
"master 能用、worker 报缺环境变量"，排查成本高且文案不具体。

本模块把该推导收敛为**单一事实源**，两侧共用：

    解析顺序 = 显式入参 > 环境变量 `QLH_KEEP_HEAD_SHIM` > 仓库内默认构建产物

找不到时返回 `""`，由调用方给出**具名**错误（不要笼统地说"需要两个环境变量"）。
"""
from __future__ import annotations

import os
from pathlib import Path

#: 环境变量名（两侧共用同一常量，避免字面量漂移）。
ENV_VAR = "QLH_KEEP_HEAD_SHIM"

#: 仓库内默认构建产物。shim 本体是 `qlh_keep_head.dll`（`qlh_kh_*` 入口在它里面）；
#: 同目录的 `libllama.dll` 只是它依赖的**带补丁** llama.cpp，不是 shim。
DEFAULT_SHIM_RELATIVE = (
    Path("build") / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
)


def default_shim_path() -> Path:
    """仓库内默认 shim 路径（`<repo>/build/keephead/build-cpu/bin/qlh_keep_head.dll`）。"""
    return Path(__file__).resolve().parent.parent / DEFAULT_SHIM_RELATIVE


def resolve_keep_head_shim(explicit: str | None = None) -> str:
    """解析 shim 路径：显式入参 > 环境变量 > 仓库默认产物；都没有则返回 `""`。"""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env
    candidate = default_shim_path()
    if candidate.is_file():
        return str(candidate)
    return ""


def missing_shim_hint() -> str:
    """缺 shim 时的**具名**提示（列出已尝试的路径，便于现场定位）。"""
    return (
        f"未找到 keep-head shim：已尝试 环境变量 {ENV_VAR} 与默认构建产物 "
        f"{default_shim_path()}；请构建 shim 或显式设置该环境变量"
    )
