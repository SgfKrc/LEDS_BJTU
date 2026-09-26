"""Routes extracted from api_server; shared state remains facade-owned."""

from __future__ import annotations

from types import ModuleType

from fastapi import APIRouter, Request
from api._routing import configure_route_module

router = APIRouter()
_api_module: ModuleType | None = None

def configure_api_module(module: ModuleType) -> None:
    configure_route_module(globals(), module)

def exported_handlers() -> dict[str, object]:
    return {name: globals()[name] for name in ['delete_all_log_files', 'delete_log_file', 'download_log_file', 'export_logs_zip', 'get_log_stats', 'get_node_recent_logs', 'get_nodes_log_aggregate', 'get_nodes_log_summary', 'get_recent_logs', 'list_log_files', 'read_log_file', 'report_client_error']}

async def list_log_files(request: Request):
    """列出 LOG_DIR 中所有 .log 文件，按修改时间降序。"""
    from config import LOG_DIR
    from datetime import datetime

    _api_module._require_log_api_access(request)

    files = []
    with _api_module._LOG_FILE_LOCK:
        if not _api_module.os.path.isdir(LOG_DIR):
            return {"files": []}

        for fname in _api_module.os.listdir(LOG_DIR):
            if not _api_module._is_log_filename(fname):
                continue
            fpath = _api_module.os.path.join(LOG_DIR, fname)
            try:
                st = _api_module.os.stat(fpath)
                files.append({
                    "name": fname,
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "_mtime": st.st_mtime,
                })
            except OSError:
                continue
    files.sort(key=lambda item: item["_mtime"], reverse=True)
    for item in files:
        item.pop("_mtime", None)
    return {"files": files}


async def get_recent_logs(
    request: Request,
    limit: int = 200,
    level: str = "",
    name: str = "",
    node_id: str = "",
    request_id: str = "",
):
    """读取内存环形缓冲中的最近日志，不读取日志文件。"""
    _api_module._require_log_api_access(request)
    limit = _api_module._normalize_log_limit(limit)
    entries, total_seen = _api_module._snapshot_recent_logs()
    filtered = _api_module._filter_recent_logs(entries, level, name, node_id, request_id)
    result = filtered[-limit:]
    return {
        "logs": result,
        "count": len(result),
        "matched": len(filtered),
        "limit": limit,
        "buffer_size": len(entries),
        "buffer_capacity": _api_module._LOG_BUFFER_MAXLEN,
        "total_seen": total_seen,
        "truncated": len(filtered) > limit,
        "filters": {
            "level": level or None,
            "name": name or None,
            "node_id": node_id or None,
            "request_id": request_id or None,
        },
    }


async def get_log_stats(request: Request):
    """返回日志文件与内存缓冲区统计信息。"""
    from config import LOG_DIR

    _api_module._require_log_api_access(request)
    entries, total_seen = _api_module._snapshot_recent_logs()
    level_counts = _api_module.Counter(item.get("level", "UNKNOWN") for item in entries)
    logger_counts = _api_module.Counter(item.get("name", "unknown") for item in entries)
    node_counts = _api_module.Counter(item.get("node_id", "unknown") for item in entries)

    files = []
    total_file_bytes = 0
    with _api_module._LOG_FILE_LOCK:
        if _api_module.os.path.isdir(LOG_DIR):
            for fname in _api_module.os.listdir(LOG_DIR):
                if not _api_module._is_log_filename(fname):
                    continue
                try:
                    st = _api_module.os.stat(_api_module.os.path.join(LOG_DIR, fname))
                    total_file_bytes += st.st_size
                    files.append({
                        "name": fname,
                        "size": st.st_size,
                        "modified": st.st_mtime,
                    })
                except OSError:
                    continue

    return {
        "log_dir": LOG_DIR,
        "files_count": len(files),
        "files_total_bytes": total_file_bytes,
        "buffer_size": len(entries),
        "buffer_capacity": _api_module._LOG_BUFFER_MAXLEN,
        "buffer_total_seen": total_seen,
        "buffer_dropped_estimate": max(0, total_seen - len(entries)),
        "levels": dict(level_counts),
        "loggers": dict(logger_counts.most_common(20)),
        "nodes": dict(node_counts),
        "node_id": _api_module._current_node_id_safe(),
        "device_ip": _api_module._current_device_ip_safe(),
    }


async def download_log_file(request: Request, name: str):
    """下载单个日志文件。"""
    from config import LOG_DIR

    _api_module._require_log_api_access(request)
    safe_name = _api_module._validate_log_filename(name)
    file_path = _api_module.os.path.join(LOG_DIR, safe_name)
    with _api_module._LOG_FILE_LOCK:
        if not _api_module.os.path.isfile(file_path):
            raise _api_module.HTTPException(404, "文件不存在")
    return _api_module.FileResponse(
        file_path,
        media_type="text/plain; charset=utf-8",
        filename=safe_name,
    )


async def delete_all_log_files(request: Request):
    """删除 LOG_DIR 中所有 .log 文件。"""
    from config import LOG_DIR

    requester = _api_module._require_log_api_access(request)

    deleted = []
    failed = []
    with _api_module._LOG_FILE_LOCK:
        if not _api_module.os.path.isdir(LOG_DIR):
            return {"status": "ok", "deleted": [], "failed": []}

        # 仅关闭文件 handler，保留终端+内存 handler（避免删除期间日志丢失）
        _api_module._close_logging_handlers(keep_memory=True)
        try:
            for fname in _api_module.os.listdir(LOG_DIR):
                if not _api_module._is_log_filename(fname):
                    continue
                try:
                    _api_module.os.remove(_api_module.os.path.join(LOG_DIR, fname))
                    deleted.append(fname)
                except OSError as e:
                    failed.append({"name": fname, "error": str(e)})
        finally:
            _api_module.setup_logging()

    status = "ok" if not failed else "partial"
    _api_module._log_admin_action(
        "delete_all",
        requester,
        "*",
        status,
        "; ".join(f"{item['name']}: {item['error']}" for item in failed),
    )
    return {
        "status": status,
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
    }


async def export_logs_zip(request: Request):
    """将所有 .log 文件打包为 ZIP 并下载。"""
    import zipfile
    import io
    from config import LOG_DIR
    from datetime import datetime

    _api_module._require_log_api_access(request)

    # 在锁内收集文件列表，在锁外构建 ZIP（避免大 I/O 时阻塞其他日志操作）
    with _api_module._LOG_FILE_LOCK:
        if not _api_module.os.path.isdir(LOG_DIR):
            raise _api_module.HTTPException(404, "日志目录不存在")
        log_files = sorted(
            f for f in _api_module.os.listdir(LOG_DIR)
            if _api_module._is_log_filename(f)
        )

    buf = io.BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in log_files:
            fpath = _api_module.os.path.join(LOG_DIR, fname)
            try:
                zf.write(fpath, fname)
                file_count += 1
            except OSError as e:
                _api_module.logger.warning("日志导出跳过 %s: %s", fname, e)

    if file_count == 0:
        raise _api_module.HTTPException(404, "没有可导出的日志文件")

    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    node_id = _api_module._current_node_id_safe() or "node"
    filename = f"qlh-logs-{node_id}-{timestamp}.zip"

    _api_module.logger.info(
        "event=log_export files_count=%d requester=%s node_id=%s",
        file_count, _api_module._get_request_client(request), node_id,
    )
    return _api_module.StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def report_client_error(report: _api_module.ClientErrorReport, request: Request):
    """接收前端错误报告并写入后端诊断日志。"""
    client_host = _api_module._get_request_client(request)
    request_id = str(_api_module._request_id_ctx.get("-") or "")
    _api_module.logger.error(
        "event=client_error source=%s message=%s url=%s line=%d col=%d "
        "client=%s ua=%s stack=%s extra=%s request_id=%s",
        _api_module._truncate_log_field(report.source, 80),
        _api_module._truncate_log_field(report.message, 500),
        _api_module._truncate_log_field(report.url, 300),
        report.line,
        report.col,
        client_host,
        _api_module._truncate_log_field(report.user_agent or "-", 200),
        _api_module._truncate_log_field(report.stack, 2000),
        _api_module._truncate_log_field(
            _api_module.json.dumps(report.extra or {}, ensure_ascii=False, default=str),
            500,
        ),
        request_id,
    )
    return {"status": "ok", "logged": True}


async def get_node_recent_logs(
    node_id: str,
    request: Request,
    limit: int = 100,
    level: str = "",
    name: str = "",
    timeout: float = 5.0,
):
    """
    从指定从节点拉取最近日志（主节点代理）。

    仅主节点可调用；向该节点发送 LOG_REQUEST TCP 消息并等待响应。
    """
    _api_module._require_log_api_access(request)

    role = _api_module._get_effective_role_safe()
    if role != "master":
        raise _api_module.HTTPException(403, "仅主节点可拉取从节点日志")

    # 如果是本节点（或查询的是自己），直接返回本地 recent logs
    local_node_id = _api_module.scheduler.get_effective_node_id()
    if node_id == local_node_id or node_id == "master":
        entries, _ = _api_module._snapshot_recent_logs()
        filtered = _api_module._filter_recent_logs(entries, level, name, node_id="", request_id="")
        result_slice = filtered[-limit:]
        return {
            "node_id": local_node_id,
            "source": "local",
            "logs": result_slice,
            "count": len(result_slice),
            "matched": len(filtered),
            "buffer_size": len(entries),
        }

    # 远程节点：通过 scheduler.request_node_logs 走 TCP
    result = _api_module.scheduler.request_node_logs(
        node_id=node_id,
        limit=limit,
        level=level,
        name=name,
        timeout=timeout,
    )
    if result is None:
        raise _api_module.HTTPException(
            504,
            f"无法从节点 {node_id} 获取日志：节点不在线或超时 ({timeout}s)",
        )

    result["source"] = "remote"
    return result


async def get_nodes_log_aggregate(
    request: Request,
    limit: int = 50,
    level: str = "",
    name: str = "",
):
    """
    P7（2026-08-16）：聚合本地与全部在线从节点的最近日志行。

    本地取内存环形缓冲，worker 经 TCP request_node_logs 拉取；每节点
    返回独立标注的日志行，不落凭据/密钥（行内容由既有脱敏链路保障）。
    仅主节点可调用；单节点失败返回 error 摘要不中断其余节点。
    """
    _api_module._require_log_api_access(request)
    role = _api_module._get_effective_role_safe()
    if role != "master":
        raise _api_module.HTTPException(403, "仅主节点可聚合集群日志")

    limit = _api_module._normalize_log_limit(limit)
    from scheduler import NodeRole, NodeState

    # 本地
    local_entries, _total = _api_module._snapshot_recent_logs()
    local_filtered = _api_module._filter_recent_logs(local_entries, level, name)
    local_logs = [entry["message"] for entry in local_filtered[-limit:]]

    # 在线 worker。每个 TCP 等待都在线程中执行，且受 semaphore 与整体
    # deadline 限制，避免节点数把 async 事件循环线性拖长。
    with _api_module.scheduler._nodes_lock:
        online_workers = sorted([
            nid
            for nid, info in _api_module.scheduler.nodes.items()
            if info.role != NodeRole.MASTER
            and info.state == NodeState.ONLINE
        ])

    import asyncio

    aggregate_deadline = float(_api_module.LOG_AGGREGATE_DEADLINE_SECONDS)
    max_parallel = min(_api_module.LOG_AGGREGATE_MAX_CONCURRENCY, max(1, len(online_workers)))
    semaphore = asyncio.Semaphore(max_parallel)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + aggregate_deadline

    async def fetch_worker_logs(nid: str) -> dict:
        async with semaphore:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return {"node_id": nid, "logs": [], "count": 0, "error": "deadline"}
            try:
                result = await asyncio.to_thread(
                    _api_module.scheduler.request_node_logs,
                    node_id=nid,
                    limit=limit,
                    level=level,
                    name=name,
                    timeout=min(3.0, remaining),
                )
                if result and result.get("logs"):
                    return {
                        "node_id": nid,
                        "logs": list(result["logs"]),
                        "count": int(result.get("count", 0)),
                    }
                return {
                    "node_id": nid,
                    "logs": [],
                    "count": 0,
                    "error": "timeout" if result is None else "no logs",
                }
            except Exception as exc:
                return {
                    "node_id": nid,
                    "logs": [],
                    "count": 0,
                    "error": str(exc)[:100],
                }

    workers_by_id = {}
    if online_workers:
        tasks = {nid: asyncio.create_task(fetch_worker_logs(nid)) for nid in online_workers}
        done, pending = await asyncio.wait(
            tasks.values(), timeout=max(0.0, deadline - loop.time()),
        )
        for task in done:
            try:
                result = task.result()
            except Exception as exc:
                result = {"node_id": "unknown", "logs": [], "count": 0, "error": str(exc)[:100]}
            workers_by_id[result.get("node_id", "unknown")] = result
        for task in pending:
            task.cancel()
    for nid in online_workers:
        workers_by_id.setdefault(
            nid, {"node_id": nid, "logs": [], "count": 0, "error": "deadline"},
        )
    workers = [workers_by_id[nid] for nid in online_workers]

    return {
        "local": {"node_id": _api_module.scheduler.get_effective_node_id(), "logs": local_logs},
        "workers": workers,
        "limit": limit,
        "filters": {"level": level or None, "name": name or None},
        "total_workers": len(online_workers),
    }


async def get_nodes_log_summary(request: Request):
    """
    返回所有在线从节点的日志概要（文件数、大小、buffer 状态）。

    仅主节点可调用。不拉取完整日志内容，仅返回每个节点的统计摘要。
    """
    _api_module._require_log_api_access(request)

    role = _api_module._get_effective_role_safe()
    if role != "master":
        raise _api_module.HTTPException(403, "仅主节点可查看集群日志概要")

    # 本地节点统计
    local_entries, _ = _api_module._snapshot_recent_logs()
    nodes_summary = {
        "local": {
            "node_id": _api_module.scheduler.get_effective_node_id(),
            "buffer_size": len(local_entries),
            "buffer_capacity": _api_module._LOG_BUFFER_MAXLEN,
        },
        "workers": [],
    }

    from scheduler import NodeRole, NodeState

    # 对所有在线从节点拉取统计（快速超时）
    with _api_module.scheduler._nodes_lock:
        online_workers = [
            (nid, info)
            for nid, info in _api_module.scheduler.nodes.items()
            if info.role != NodeRole.MASTER
            and info.state == NodeState.ONLINE
        ]

    for nid, _info in online_workers:
        try:
            result = _api_module.scheduler.request_node_logs(
                node_id=nid, limit=5, timeout=3.0,
            )
            if result:
                nodes_summary["workers"].append({
                    "node_id": nid,
                    "buffer_size": result.get("buffer_size", 0),
                    "sample_count": result.get("count", 0),
                })
            else:
                nodes_summary["workers"].append({
                    "node_id": nid,
                    "buffer_size": 0,
                    "error": "timeout",
                })
        except Exception as e:
            nodes_summary["workers"].append({
                "node_id": nid,
                "buffer_size": 0,
                "error": str(e)[:100],
            })

    nodes_summary["total_workers"] = len(online_workers)
    return nodes_summary


async def read_log_file(filename: str, request: Request):
    """读取指定日志文件的内容（最多返回末 1 MB）。"""
    from config import LOG_DIR

    _api_module._require_log_api_access(request)
    safe_name = _api_module._validate_log_filename(filename)
    max_bytes = 1024 * 1024  # 1 MB
    try:
        file_path = _api_module.os.path.join(LOG_DIR, safe_name)
        # 锁内仅做存在性检查和大小获取，锁外读取文件内容（避免阻塞其他日志操作）
        with _api_module._LOG_FILE_LOCK:
            if not _api_module.os.path.isfile(file_path):
                raise _api_module.HTTPException(404, "文件不存在")
            file_size = _api_module.os.path.getsize(file_path)

        with open(file_path, "rb") as f:
            # 重新获取实际文件大小（锁外可能已被轮转截断）
            f.seek(0, _api_module.os.SEEK_END)
            actual_size = f.tell()
            truncated = actual_size > max_bytes
            if truncated:
                f.seek(max(0, actual_size - max_bytes))
                f.readline()  # 跳过不完整首行
            else:
                f.seek(0)
            content = f.read().decode("utf-8", errors="replace")
        return {"name": safe_name, "content": content, "truncated": truncated}
    except _api_module.HTTPException:
        raise
    except Exception as e:
        raise _api_module.HTTPException(500, f"读取失败: {e}")


async def delete_log_file(filename: str, request: Request):
    """删除指定的 .log 文件。"""
    from config import LOG_DIR

    requester = _api_module._require_log_api_access(request)
    safe_name = _api_module._validate_log_filename(filename)
    error_msg = ""
    with _api_module._LOG_FILE_LOCK:
        file_path = _api_module.os.path.join(LOG_DIR, safe_name)
        if not _api_module.os.path.isfile(file_path):
            raise _api_module.HTTPException(404, "文件不存在")

        # 仅关闭文件 handler，保留终端+内存 handler（避免删除期间日志丢失）
        _api_module._close_logging_handlers(keep_memory=True)
        try:
            _api_module.os.remove(file_path)
        except Exception as e:
            error_msg = str(e)
        finally:
            _api_module.setup_logging()

    if error_msg:
        _api_module._log_admin_action("delete", requester, safe_name, "failed", error_msg)
        raise _api_module.HTTPException(500, f"删除失败: {error_msg}")

    _api_module._log_admin_action("delete", requester, safe_name, "ok")
    return {"status": "ok", "deleted": safe_name, "failed": []}


def register_routes() -> None:
    router.add_api_route("/api/logs", list_log_files, methods=["GET"])
    router.add_api_route("/api/logs/recent", get_recent_logs, methods=["GET"])
    router.add_api_route("/api/logs/stats", get_log_stats, methods=["GET"])
    router.add_api_route("/api/logs/download", download_log_file, methods=["GET"])
    router.add_api_route("/api/logs", delete_all_log_files, methods=["DELETE"])
    router.add_api_route("/api/logs/export", export_logs_zip, methods=["GET"])
    router.add_api_route("/api/logs/client-error", report_client_error, methods=["POST"])
    router.add_api_route("/api/logs/node/{node_id}/recent", get_node_recent_logs, methods=["GET"])
    router.add_api_route("/api/cluster/nodes/log-aggregate", get_nodes_log_aggregate, methods=["GET"])
    router.add_api_route("/api/logs/nodes-summary", get_nodes_log_summary, methods=["GET"])
    router.add_api_route("/api/logs/{filename:path}", read_log_file, methods=["GET"])
    router.add_api_route("/api/logs/{filename:path}", delete_log_file, methods=["DELETE"])
