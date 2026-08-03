"""scheduler-svc 进程入口（微服务架构改造计划 §1.7）。

§1.4 规则 2（新增入口 import 复用）：本入口 **import 现有 scheduler.py
运行，scheduler.py 源码零改动**。阶段 0 就绪的 `Scheduler(host=...)`
注入参数在此使用：

  - 默认：`host=InferenceClient()` —— scheduler 经 HTTP :8010
    调用 inference-svc（数据面独立进程）
  - `QLH_MONOLITH=1`：`host=进程内 model_host` —— 一键回退单进程
    （随时可回，回退后行为与现有 api_server 内嵌 scheduler 等价）

角色/运行模式沿用现有环境变量（QLH_NODE_ROLE / QLH_NODE_ID /
QLH_SERVER_PORT / QLH_NODE_TYPE 等），config.py 读取逻辑不变。
"""
import logging
import os
import threading
import time


def build_scheduler():
    """按 QLH_MONOLITH 选择推理宿主并构造 Scheduler。"""
    from scheduler import Scheduler

    if os.environ.get("QLH_MONOLITH", "0") == "1":
        from model_host import get_model_host

        host = get_model_host()
        logging.getLogger("scheduler_svc_main").info("回退模式 QLH_MONOLITH=1：进程内 model_host")
    else:
        from inference_client import InferenceClient

        host = InferenceClient()
        logging.getLogger("scheduler_svc_main").info(
            "微服务模式：host=InferenceClient → inference-svc"
        )
    return Scheduler(host=host)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("scheduler_svc_main")
    t_start = time.time()

    scheduler = build_scheduler()
    logger.info(f"Scheduler 构造完成 ({time.time() - t_start:.2f}s)，调用 start()")

    scheduler.start()
    logger.info(
        f"scheduler-svc 已启动: role={scheduler._effective_role()}, "
        f"TCP={os.environ.get('QLH_SERVER_PORT', '8888')} "
        f"(QLH_MONOLITH={os.environ.get('QLH_MONOLITH', '0')})"
    )

    # ---- §4.2 HTTP 壳（:8020，QLH_SCHEDULER_HTTP_PORT；1.7 顺延至阶段 2 起点）----
    http_thread = None
    http_port = int(os.environ.get("QLH_SCHEDULER_HTTP_PORT", "8020"))
    try:
        import uvicorn
        from scheduler_svc_http import build_scheduler_app

        http_app = build_scheduler_app(scheduler)
        cfg = uvicorn.Config(
            http_app, host="127.0.0.1", port=http_port, log_level="warning",
        )
        server = uvicorn.Server(cfg)
        http_thread = threading.Thread(target=server.run, daemon=True)
        http_thread.start()
        logger.info(f"scheduler-svc HTTP 壳已启动: http://127.0.0.1:{http_port}/cluster/*")
    except Exception as exc:
        logger.warning(f"scheduler-svc HTTP 壳启动失败（继续 TCP 模式）: {exc}")

    # start() 启动 TCP 服务端后返回；本进程保持存活直到收到退出信号
    stop_event = threading.Event()

    def _on_signal(*_args):
        logger.info("收到退出信号，停止 scheduler-svc")
        try:
            scheduler.stop()
        except Exception:
            pass
        stop_event.set()

    import signal

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    stop_event.wait()


if __name__ == "__main__":
    main()
