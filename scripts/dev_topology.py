"""主节点开发拓扑编排（微服务架构改造计划 §1.6，开发期脚本）。

并行共存（§1.4）：只编排、**不修改现有 Python 后端任何代码**。
拉起两个进程：
  1. inference-svc（`python src/inference_svc_main.py`，:8010）
  2. api_server（现有启动方式 :8000，阶段 1 仍承载全部 110 端点与调度器；
     其 scheduler 连 InferenceClient 属阶段 2 切换动作，本脚本不触碰）

健康检查互等（inference-svc /v1/health、api_server /api/health）；
任一未就绪则整体退出；Ctrl+C 一起终止。

用法:
    python scripts/dev_topology.py
    QLH_INFERENCE_PORT=18010 QLH_API_PORT=18000 python scripts/dev_topology.py
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Windows GBK 控制台无法编码部分符号（▶🎉等）：强制 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent


def _wait_health(url: str, name: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"✅ {name} 就绪: {url}", flush=True)
                    return True
        except Exception:
            time.sleep(0.5)
    print(f"❌ {name} 健康检查超时: {url}", flush=True)
    return False


def main() -> int:
    infer_port = os.environ.get("QLH_INFERENCE_PORT", "8010")
    api_port = os.environ.get("QLH_API_PORT", "8000")

    procs: list = []
    try:
        # ---- inference-svc ----
        env_i = dict(os.environ)
        env_i.setdefault("QLH_INFERENCE_PORT", infer_port)
        env_i.setdefault("QLH_NODE_ROLE", "master")
        p1 = subprocess.Popen(
            [sys.executable, "src/inference_svc_main.py"],
            cwd=REPO, env=env_i,
        )
        procs.append(p1)
        print(f"▶ inference-svc 启动中 (: {infer_port})", flush=True)

        # ---- api_server（现有后端，内含 scheduler）----
        env_a = dict(os.environ)
        env_a.setdefault("QLH_API_PORT", api_port)
        p2 = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api_server:app",
             "--host", "0.0.0.0", "--port", api_port],
            cwd=REPO, env=env_a,
        )
        procs.append(p2)
        print(f"▶ api_server 启动中 (: {api_port})", flush=True)

        ok1 = _wait_health(f"http://127.0.0.1:{infer_port}/v1/health", "inference-svc")
        ok2 = _wait_health(f"http://127.0.0.1:{api_port}/api/health", "api_server")
        if not (ok1 and ok2):
            print("❌ 拓扑未全部就绪，退出", flush=True)
            return 1

        print("🎉 主节点开发拓扑就绪（并行共存，未改动现有后端）", flush=True)
        print(f"   inference-svc: http://127.0.0.1:{infer_port}/v1/*")
        print(f"   api_server   : http://127.0.0.1:{api_port}/api/* （阶段 1 唯一对外入口）")
        print("按 Ctrl+C 停止全部进程", flush=True)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("已停止全部进程", flush=True)


if __name__ == "__main__":
    sys.exit(main())
