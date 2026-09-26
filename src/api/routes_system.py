"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None
_RESOLUTION_NAMES = ("Request", "SystemShutdownRequest")

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module, _RESOLUTION_NAMES)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['system_shutdown', 'database_health', 'storage_health']}

async def system_shutdown(req: SystemShutdownRequest, request: Request):
    """
    优雅退出后端服务（资源清理 → 调度器/数据库/TCP 全部关闭后退出进程）。

    安全防护：
      - 仅允许本机来源（127.0.0.1 / ::1）直接调用；
      - 远程调用必须携带 X-QLH-Shutdown-Token，且需在启动前设置
        环境变量 QLH_SHUTDOWN_TOKEN（未设置时远程调用一律拒绝）。
    """
    client_host = (request.client.host if request.client else "") or ""
    is_local = client_host in ("127.0.0.1", "::1", "localhost", "")
    if not is_local:
        if not _api_module._SHUTDOWN_TOKEN:
            raise _api_module.HTTPException(
                status_code=403,
                detail="远程关闭被拒绝：服务端未配置 QLH_SHUTDOWN_TOKEN。",
            )
        token = request.headers.get("X-QLH-Shutdown-Token", "")
        if token != _api_module._SHUTDOWN_TOKEN:
            raise _api_module.HTTPException(status_code=403, detail="关闭令牌无效。")
    reason = (req.reason or "").strip()
    _api_module.logger.warning(f"event=system_shutdown_requested source={client_host or 'unknown'} reason={reason or 'unspecified'}")
    _api_module.threading.Thread(target=_api_module._graceful_exit, daemon=True, name="graceful-exit").start()
    return {"ok": True, "message": "后端正在优雅退出…"}

async def database_health():
    """主节点 SQLite 健康状态及旧 PostgreSQL 退场状态。"""
    try:
        local = _api_module._local_store.local_store_health()
    except Exception as exc:
        local = {
            "status": "unavailable",
            "backend": "sqlite",
            "writable": False,
            "error": str(exc),
        }

    remote = {
        "status": "retired",
        "backend": "postgresql",
        "mode": "retired",
    }

    return {
        "status": local.get("status", "unavailable"),
        "backend": "sqlite",
        "local": local,
        "remote": remote,
        "effective_mode": "local_only",
    }

async def storage_health():
    """Expose the local-first storage contract used by the canonical console.

    The monolith does not own a projection worker or a remote export queue, so
    those fields are explicit zero/retired states instead of leaking internal
    implementation details to the UI.
    """
    try:
        local = _api_module._local_store.local_store_health()
    except Exception as exc:
        local = {
            "status": "unavailable",
            "backend": "sqlite",
            "writable": False,
            "error": str(exc),
        }

    return {
        "local": local,
        "remote": {
            "status": "retired",
            "backend": "postgresql",
            "mode": "retired",
        },
        "projection": {
            "pending_events": 0,
            "oldest_event_age_seconds": None,
        },
        "export": {
            "pending_items": 0,
            "oldest_item_age_seconds": None,
        },
        "effective_mode": "local_only" if local.get("writable", False) else "readonly_failure",
        "retirement": {
            "status": "retired",
            "prepared_at": None,
            "retired_at": None,
        },
    }


def register_routes() -> None:
    router.add_api_route('/api/system/shutdown', system_shutdown, methods=['POST'])
    router.add_api_route('/api/db/health', database_health, methods=['GET'])
    router.add_api_route('/api/storage/health', storage_health, methods=['GET'])
