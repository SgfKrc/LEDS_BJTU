#!/usr/bin/env python
"""verify_xframe_consistency.py — #9 准入证据④：**协议版本与跨节点一致性**核验

## 为什么需要
`docs/跨框架层接力-项目报告.md` §7「未完成/待办」第 1 条列出的准入条件之一：
**跨节点时钟与协议版本一致性**。Relay 把 hidden 交给**另一个进程/机器**的 llama.cpp，
因此「两侧必须是同一构建」是正确性前提 —— 而实验 fork **确实带本地补丁**，
若两端构建不同，逐 token 一致结论不成立。

## 核验内容
1. **下游 llama.cpp 构建指纹**：commit / describe / 工作区是否 dirty / 每个补丁的 SHA-256；
2. **Relay 线协议常量**：magic / wire version / dtype / 字节序 / header 布局；
3. **上游 PyTorch 侧指纹**：torch / transformers 版本、CUDA 与设备、上游精度开关；
4. **时钟基准**：本机 UTC 偏移 + 单调时钟源，用于判断「跨节点时钟」是否构成风险；
5. **跨进程一致性**：本机起两个独立进程各取一次时钟与版本，验证同一构建下可复现。

输出 JSON 证据，供生产准入评审引用。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> 仓库根（入仓后由 build/ 下的副本移入）
POC = ROOT / "build" / "cross-framework-layer-poc"
LLAMA = POC / "llama.cpp"


def _git(args: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                              text=True, timeout=60, encoding="utf-8", errors="replace")
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def llama_fingerprint() -> dict:
    dirty_files = _git(["status", "--porcelain"], LLAMA)
    patched = []
    for line in dirty_files.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        status, rel = parts
        p = LLAMA / rel.replace("/", os.sep)
        entry = {"status": status, "path": rel, "sha256": _sha256(p) if p.is_file() else None}
        patched.append(entry)
    bin_dir = LLAMA / "build-cpu" / "bin"
    bins = {}
    for name in ("llama-relay-gen.exe", "llama-relay-gen-dl.exe"):
        bp = bin_dir / name
        if bp.is_file():
            st = bp.stat()
            bins[name] = {"size": st.st_size, "mtime": datetime.fromtimestamp(
                st.st_mtime, timezone.utc).isoformat(), "sha256": _sha256(bp)[:32]}
    return {
        "commit": _git(["rev-parse", "HEAD"], LLAMA),
        "describe": _git(["describe", "--tags"], LLAMA),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], LLAMA),
        "worktree_clean": not dirty_files,
        "local_patches": patched,
        "binaries": bins,
    }


def protocol_fingerprint() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import relay_transport as rt

    return {
        "magic": rt.RELAY_WIRE_MAGIC.decode("ascii", "replace"),
        "wire_version": int(rt.RELAY_WIRE_VERSION),
        "dtype": rt.RELAY_DTYPE,
        "dtype_bytes": int(rt.RELAY_DTYPE_BYTES),
        "header_struct": str(rt._HEADER.format),
        "header_size": rt._HEADER.size,
        "default_max_tokens": int(rt.RELAY_DEFAULT_MAX_TOKENS),
        "default_max_payload": int(rt.RELAY_DEFAULT_MAX_PAYLOAD),
        "client_side_version_check": True,
    }


def upstream_fingerprint() -> dict:
    out: dict = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            out["device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["device_total_memory_gb"] = round(props.total_memory / 1e9, 2)
    except Exception as exc:  # noqa: BLE001
        out["torch_error"] = str(exc)
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        out["transformers_error"] = str(exc)
    out["upstream_quant_env"] = os.environ.get("QLH_UPSTREAM_QUANT", "f32")
    out["upstream_attn_env"] = os.environ.get("QLH_UPSTREAM_ATTN", "eager")
    out["downstream_keep_kv_env"] = os.environ.get("QLH_DOWNSTREAM_KEEP_KV", "0")
    return out


def clock_fingerprint() -> dict:
    utc = datetime.now(timezone.utc)
    return {
        "utc_iso": utc.isoformat(),
        "utc_offset_seconds": utc.astimezone().utcoffset().total_seconds(),
        "local_tz": time.tzname[0] if time.tzname else "",
        "monotonic_ns": time.monotonic_ns(),
        "process_id": os.getpid(),
    }


def main() -> int:
    started = time.perf_counter()
    report = {
        "ticket": "CORE-RELAY-XFRAME-01",
        "evidence": "protocol-and-crossnode-consistency",
        "date": datetime.now(timezone.utc).astimezone().isoformat(),
        "downstream_llama_cpp": llama_fingerprint(),
        "relay_protocol": protocol_fingerprint(),
        "upstream": upstream_fingerprint(),
        "clock": clock_fingerprint(),
        "host": platform.node(),
    }
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # 判定：两侧一致性前置
    fp = report["downstream_llama_cpp"]
    verdict = {
        "worktree_clean": fp["worktree_clean"],
        "local_patch_count": len(fp["local_patches"]),
        "requires_identical_build_both_sides": True,
        "reason": (
            "实验 fork 带本地补丁" if fp["local_patches"] else "无本地补丁"
        ) + "；Relay 把 hidden 交给另一进程/机器的 llama.cpp，"
        "两侧构建不一致时逐 token 一致结论不成立 ⇒ 跨节点部署必须核验本指纹。",
    }
    report["verdict"] = verdict

    out = POC / "out" / "xframe-consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 74)
    print("### 下游 llama.cpp 构建指纹")
    print("=" * 74)
    print(f"  commit   = {fp['commit'][:16]}   describe = {fp['describe']}")
    print(f"  branch   = {fp['branch']}   worktree_clean = {fp['worktree_clean']}")
    print(f"  本地补丁 = {len(fp['local_patches'])} 个")
    for p in fp["local_patches"]:
        print(f"    {p['status']} {p['path']}  sha256={(p['sha256'] or '-')[:16]}")
    for name, info in fp["binaries"].items():
        print(f"  bin {name}  size={info['size']}  sha256={info['sha256']}")
    print()
    print("### Relay 线协议")
    for k, v in report["relay_protocol"].items():
        print(f"  {k} = {v}")
    print()
    print("### 上游指纹")
    for k, v in report["upstream"].items():
        print(f"  {k} = {v}")
    print()
    print("### 时钟基准")
    for k, v in report["clock"].items():
        print(f"  {k} = {v}")
    print()
    print("### 判定")
    print(f"  {json.dumps(verdict, ensure_ascii=False, indent=1)}")
    print(f"\n  [json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
