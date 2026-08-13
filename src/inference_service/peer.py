"""从节点客户端（微服务架构改造计划 §1.5）。

复制自 scheduler.py 的 client 角色分支（_connect_to_master_locked /
_handle_layer_config_locked / _handle_layer_forward_locked /
_send_layer_config_ack / _send_layer_result / pipeline_done|abort），
**scheduler.py 源文件不动**；复用 tcp_comm.TCPClient 既有协议
（帧格式 / MessageType / HMAC 认证 / 心跳语义不变）。

宿主适配（scheduler 内部状态 → 实例属性）：
  _layer_execution_lock / _layer_config_lock / _kv_cache_lock → 本类 RLock
  _active_layer_config / _local_pipeline_steps / _active_pipeline_task_ids
    / _local_pipeline_cancelled / _kv_cache → 本类实例属性
  model_manager / model_host → EngineHost（本进程数据面宿主）
  get_effective_node_id() → self._node_id

未复制（开发期从节点不承担，归 scheduler-svc 控制面）：
  链式直连 chain_forward（结果经主节点中转回退，主节点侧兼容）、
  bootstrap 首次连接部署、备用主节点/角色转让、DB 主节点发现。
"""
import base64
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("inference_service.peer")


class PeerClient:
    """从节点客户端：连接主节点 + 层段加载 + 层前向执行闭环。"""

    def __init__(
        self,
        master_host: Optional[str] = None,
        master_port: Optional[int] = None,
        node_id: Optional[str] = None,
        device_info: Optional[dict] = None,
    ):
        # 层段执行宿主（EngineHost 构造轻量：model_host + config，不触发
        # model_module；首次 load 才加载）
        from inference_service.engine_host import EngineHost

        self._host = EngineHost()
        self._host.role = "client"

        import config as cfg

        self._master_host = master_host or os.environ.get(
            "QLH_CLIENT_MASTER_HOST", "127.0.0.1"
        )
        self._master_port = master_port or int(os.environ.get(
            "QLH_CLIENT_MASTER_PORT", getattr(cfg, "SERVER_PORT", 8888)
        ))
        if node_id:
            self._node_id = node_id
        else:
            configured_node_id = os.environ.get("QLH_NODE_ID", "") or ""
            if not configured_node_id or configured_node_id == "master":
                self._node_id = f"client_{socket.gethostname()}"
            else:
                self._node_id = configured_node_id
        self._device_info = dict(device_info or {})

        # ---- 层配置/流水线状态（scheduler 内部状态 → 实例属性）----
        self._layer_execution_lock = threading.RLock()
        self._layer_config_lock = threading.RLock()
        self._kv_cache_lock = threading.RLock()
        self._active_layer_config: Optional[dict] = None
        self._local_pipeline_steps: Dict[str, int] = {}
        self._active_pipeline_task_ids: set = set()
        self._local_pipeline_cancelled: set = set()
        self._kv_cache: Dict[str, Any] = {}
        self._pending_layer_config: Optional[tuple] = None
        self._running = False
        self._client: Optional[Any] = None  # tcp_comm.TCPClient
        self._reconnect_delay = 5.0

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def connect(self) -> dict:
        """连接主节点（TCPClient 注册 + 心跳由 tcp_comm 内部处理）。"""
        import config as cfg
        from tcp_comm import TCPClient

        advertise_port = getattr(cfg, "SERVER_PORT", 8888)
        client = TCPClient(
            server_host=self._master_host,
            server_port=self._master_port,
            client_id=self._node_id,
            role="client",
            node_type=os.environ.get("QLH_NODE_TYPE", "pc"),
            advertise_port=advertise_port,
            device_info=self._device_info,
        )
        self._client = client

        def _on_heartbeat() -> None:
            self._report_device_profile()

        def _on_disconnect() -> None:
            logger.warning("与主节点连接断开: %s:%s", self._master_host, self._master_port)
            with self._layer_config_lock:
                self._active_layer_config = None
                self._local_pipeline_steps.clear()

        client.on_heartbeat = _on_heartbeat
        client.on_disconnect = _on_disconnect

        ok = client.connect(on_message=self._on_message)
        if ok:
            self._running = True
            self._report_device_profile()
            logger.info(
                "✅ 从节点已连接主节点: %s:%s (node_id=%s)",
                self._master_host, self._master_port, self._node_id,
            )
            return {
                "status": "connected",
                "node_id": self._node_id,
                "master": f"{self._master_host}:{self._master_port}",
            }
        return {
            "status": "failed",
            "reason": f"连接主节点 {self._master_host}:{self._master_port} 失败",
        }

    def run_forever(self) -> None:
        """阻塞运行：连接失败/断开后自动重连（简单退避）。"""
        while True:
            if self._client is None or not self._running:
                result = self.connect()
                if result.get("status") != "connected":
                    logger.info(
                        "连接失败: %s，%.0fs 后重试",
                        result.get("reason", "unknown"), self._reconnect_delay,
                    )
            time.sleep(self._reconnect_delay)

    # ------------------------------------------------------------------
    # TCP 消息分发（复制 scheduler._on_tcp_message 的 client 相关分支）
    # ------------------------------------------------------------------
    def _on_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        data = msg.get("data", {})
        if msg_type == "layer_config":
            threading.Thread(
                target=self._handle_layer_config,
                args=(data,),
                name="peer-layer-config",
                daemon=True,
            ).start()
        elif msg_type == "layer_forward":
            threading.Thread(
                target=self._handle_layer_forward,
                args=(data,),
                name=f"peer-layer-forward-{data.get('task_id', 'unknown')}",
                daemon=True,
            ).start()
        elif msg_type == "pipeline_done":
            self._handle_pipeline_done(data)
        elif msg_type == "pipeline_abort":
            self._handle_pipeline_abort(data)
        elif msg_type in ("heartbeat_ack", "register_ack", "status_res"):
            pass  # tcp_comm 内部处理
        else:
            logger.debug("从节点忽略消息类型: %s", msg_type)

    # ------------------------------------------------------------------
    # 层配置（复制 scheduler._handle_layer_config_locked 核心语义）
    # ------------------------------------------------------------------
    def _handle_layer_config(self, data: dict) -> None:
        with self._layer_execution_lock:
            self._handle_layer_config_locked(data)

    def _handle_layer_config_locked(self, data: dict) -> None:
        node_id = self._node_id

        # 兼容两种格式：新版直接是 assignment；旧版 {node_id: assignment}
        if isinstance(data, dict) and data.get("release"):
            target_node_id = str(data.get("node_id", node_id))
            if target_node_id != node_id:
                logger.warning("忽略目标不匹配的分层释放: target=%s local=%s", target_node_id, node_id)
                return
            with self._layer_config_lock:
                self._active_layer_config = None
                self._local_pipeline_steps.clear()
            if data.get("abort"):
                try:
                    from model_sync import remove_pipeline_assignment_cache

                    model_id = str(data.get("model_id", "") or "")
                    aborted_config_id = str(data.get("aborted_config_id", "") or "")
                    if model_id and aborted_config_id:
                        remove_pipeline_assignment_cache(
                            model_id, aborted_config_id, node_id,
                        )
                except Exception:
                    logger.warning("清理已中止的 assignment 缓存失败", exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": str(data.get("config_id", "")),
                "status": "released",
            })
            logger.info("分层配置已释放: config_id=%s", data.get("config_id", ""))
            return

        if isinstance(data, dict) and "start_layer" in data and "end_layer" in data:
            cfg = dict(data)
        elif isinstance(data, dict) and node_id in data:
            cfg = dict(data[node_id] or {})
        else:
            error = f"分层配置中未找到本节点 {node_id} 的有效 assignment"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": data.get("config_id", "") if isinstance(data, dict) else "",
                "status": "error",
                "error": error,
            })
            return

        config_id = str(cfg.get("config_id", ""))
        target_node_id = str(cfg.get("node_id", node_id))
        start = cfg.get("start_layer", 0)
        end = cfg.get("end_layer", 24)
        has_embed = cfg.get("has_embedding", False)
        has_lm = cfg.get("has_lm_head", False)
        model_id = str(cfg.get("model_id", ""))
        expected_sha256 = str(cfg.get("model_sha256", ""))
        expected_model_type = str(cfg.get("model_type", "")).lower()
        try:
            start = int(start)
            end = int(end)
            total_layers = int(cfg.get("total_layers", 0) or 0)
        except (TypeError, ValueError) as exc:
            error = f"分层配置数字字段无效: {exc}"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id, "config_id": config_id,
                "status": "error", "error": error,
            })
            return

        logger.info(
            f"🔧 收到分层配置: 节点={node_id}, Layer {start}-{end}, "
            f"embed={has_embed}, lm_head={has_lm}, config_id={config_id or 'legacy'}"
        )

        try:
            if target_node_id != node_id:
                raise ValueError(f"层配置目标节点 {target_node_id} 与本节点 {node_id} 不一致")
            if expected_model_type not in {"qwen", "qwen2"}:
                raise ValueError(f"不支持的流水线模型架构: {expected_model_type or 'unknown'}")
            missing_contract = [
                name for name, value in (
                    ("config_id", config_id), ("model_id", model_id),
                    ("model_sha256", expected_sha256), ("total_layers", total_layers),
                )
                if not value
            ]
            if missing_contract:
                raise ValueError("分层配置执行契约不完整: " + ", ".join(missing_contract))

            with self._layer_config_lock:
                self._active_layer_config = None
                self._local_pipeline_steps.clear()
            self._host._host.model_loaded = False

            # 模型同步：确保 worker 模型文件就绪后加载层段
            from model_sync import ensure_model_available, resolve_worker_model_path

            local_model_path = resolve_worker_model_path(model_id)
            if not local_model_path:
                logger.info(f"模型 {model_id} 尚未同步，开始拉取...")
                ensure_model_available(model_id)
                local_model_path = resolve_worker_model_path(model_id)
            if not local_model_path:
                raise RuntimeError(f"模型同步后仍无本地路径: {model_id}")

            self._host._host.load_model(
                model_path=local_model_path,
                quant_type="int4",
                profile=None,
                engine="pytorch",
            )
            self._host._host.load_layer_range(
                start_layer=start, end_layer=end,
                has_embedding=has_embed, has_lm_head=has_lm,
            )

            with self._layer_config_lock:
                self._active_layer_config = dict(cfg)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": config_id,
                "status": "ready",
                "layer_range": f"{start}-{end}",
            })
            logger.info(
                f"✅ 层段加载完成: Layer {start}-{end}, config_id={config_id}"
            )
        except Exception as e:
            error = f"分层配置执行失败: {e}"
            logger.error(error, exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id, "config_id": config_id,
                "status": "error", "error": error,
            })

    def _send_layer_config_ack(self, payload: dict) -> bool:
        from tcp_comm import MessageType

        client = self._client
        if client is None:
            logger.warning("TCP 客户端未连接，无法发送层配置 ACK")
            return False
        try:
            client.send_data(payload, MessageType.LAYER_CONFIG_ACK)
            return True
        except Exception as e:
            logger.error(f"发送层配置 ACK 失败: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 层前向（复制 scheduler._handle_layer_forward_locked 核心语义）
    # ------------------------------------------------------------------
    def _handle_layer_forward(self, data: dict) -> None:
        with self._layer_execution_lock:
            self._handle_layer_forward_locked(data)

    def _handle_layer_forward_locked(self, data: dict) -> None:
        import torch
        from tcp_comm import deserialize_tensor_fast, serialize_tensor_fast

        task_id = str(data.get("task_id", "unknown") or "unknown")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = -1
        use_kv_cache = data.get("use_kv_cache", False)
        config_id = str(data.get("config_id", ""))
        model_sha256 = str(data.get("model_sha256", ""))
        model_type = str(data.get("model_type", "")).lower()

        logger.info(
            f"🔬 收到层前向指令: task={task_id}, step={step}, "
            f"kv_cache={'on' if use_kv_cache else 'off'}"
        )

        try:
            with self._layer_config_lock:
                if task_id in self._local_pipeline_cancelled:
                    logger.info("忽略已取消任务的迟到层前向: task=%s", task_id)
                    return
                active_config = dict(self._active_layer_config or {})
                last_step = self._local_pipeline_steps.get(task_id)
                task_active = task_id in self._active_pipeline_task_ids
            if not active_config:
                raise RuntimeError("本节点没有已确认的活动层配置")
            for field, actual in (
                ("config_id", config_id),
                ("model_sha256", model_sha256),
                ("model_type", model_type),
            ):
                if not actual or actual != str(active_config.get(field, "")):
                    raise RuntimeError(
                        f"流水线执行契约不一致: {field}={actual or '-'}, "
                        f"expected={active_config.get(field, '-')}"
                    )
            if step < 0:
                raise RuntimeError(f"无效流水线 step: {step}")
            if step == 0:
                if use_kv_cache:
                    raise RuntimeError("prefill step 0 不得声明使用既有 KV cache")
                if task_active or last_step is not None:
                    raise RuntimeError(f"重复 prefill: task={task_id}")
            else:
                if not use_kv_cache:
                    raise RuntimeError(f"decode step {step} 必须使用 KV cache")
                if not task_active or last_step != step - 1:
                    raise RuntimeError(
                        f"流水线 step 越序: task={task_id}, step={step}, "
                        f"last_step={last_step}"
                    )

            mgr = self._host._host
            if not mgr or not getattr(mgr, "is_loaded", False):
                raise RuntimeError("模型未加载")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(getattr(loaded_config, "model_type", "") or "").lower()
            if getattr(mgr, "_engine_type", "") != "pytorch":
                raise RuntimeError(f"worker 引擎已变化: {getattr(mgr, '_engine_type', '')}")
            if actual_model_type != model_type:
                raise RuntimeError(
                    f"worker 模型架构已变化: actual={actual_model_type}, expected={model_type}"
                )

            t_start = time.time()
            input_ids = None
            hidden_states = None
            if "input_ids" in data and data["input_ids"] is not None:
                input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
            if "hidden_states" in data and data["hidden_states"] is not None:
                hidden_states = deserialize_tensor_fast(data["hidden_states"])

            past_kv = None
            if use_kv_cache:
                with self._kv_cache_lock:
                    past_kv = self._kv_cache.get(task_id)
                if past_kv is None:
                    raise RuntimeError(
                        f"decode step {step} 缺少本地 KV cache: task={task_id}"
                    )

            result = mgr.forward_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=(
                    torch.tensor(data["attention_mask"], dtype=torch.long)
                    if data.get("attention_mask") is not None else None
                ),
                position_ids=(
                    torch.tensor(data["position_ids"], dtype=torch.long)
                    if data.get("position_ids") is not None else None
                ),
                past_key_values=past_kv,
                use_cache=True,
                apply_lm_head=bool(data.get("apply_lm_head", False)),
            )
            if task_id in self._local_pipeline_cancelled:
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                logger.info("丢弃已取消任务的迟到计算结果: task=%s", task_id)
                return
            elapsed_ms = (time.time() - t_start) * 1000

            if result.get("past_key_values"):
                with self._kv_cache_lock:
                    self._kv_cache[task_id] = result["past_key_values"]
            else:
                raise RuntimeError("分层前向未返回 KV cache")
            with self._layer_config_lock:
                self._local_pipeline_steps[task_id] = step
                self._active_pipeline_task_ids.add(task_id)

            response = {
                "task_id": task_id,
                "node_id": self._node_id,
                "step": step,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
                "metrics": {
                    "time_ms": round(elapsed_ms, 1),
                    "kv_cache": use_kv_cache,
                    "memory_allocated_gb": (
                        round(torch.cuda.memory_allocated() / (1024**3), 2)
                        if torch.cuda.is_available() else 0
                    ),
                },
            }
            if "hidden_states" in result:
                hs_cpu = result["hidden_states"].detach().cpu()
                response["hidden_states"] = serialize_tensor_fast(hs_cpu)
                response["hidden_shape"] = list(hs_cpu.shape)
            if "logits" in result:
                logits_cpu = result["logits"].detach().cpu()
                response["logits"] = serialize_tensor_fast(logits_cpu)
                response["logits_shape"] = list(logits_cpu.shape)

            self._send_layer_result(task_id, response)
            logger.info(
                f"✅ 层前向完成: task={task_id}, step={step}, "
                f"time={elapsed_ms:.0f}ms"
            )
        except Exception as e:
            error = str(e)
            logger.error(f"层前向失败: task={task_id}, step={step}: {error}")
            self._send_layer_result(task_id, {}, error=error)
            with self._layer_config_lock:
                self._local_pipeline_cancelled.add(task_id)

    def _send_layer_result(self, task_id: str, result_data: dict,
                           error: str = None) -> bool:
        from tcp_comm import MessageType

        if self._client is None or not getattr(self._client, "_running", False):
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            return False

        payload = dict(result_data)
        payload["task_id"] = task_id
        if error:
            payload["error"] = error
        safe_payload = {}
        for k, v in payload.items():
            if isinstance(v, bytes):
                safe_payload[k] = base64.b64encode(v).decode("ascii")
            else:
                safe_payload[k] = v
        try:
            self._client.send_data(safe_payload, MessageType.LAYER_RESULT)
            return True
        except Exception as e:
            logger.error(f"发送层前向结果失败: {e}")
            try:
                self._client.disconnect()
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # 流水线任务清理（复制 scheduler._on_tcp_message 的 pipeline_* 分支）
    # ------------------------------------------------------------------
    def _handle_pipeline_done(self, data: dict) -> None:
        task_id = data.get("task_id", "")
        if task_id:
            with self._layer_config_lock:
                self._local_pipeline_cancelled.discard(task_id)
                self._local_pipeline_steps.pop(task_id, None)
            with self._kv_cache_lock:
                self._kv_cache.pop(task_id, None)
            logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")

    def _handle_pipeline_abort(self, data: dict) -> None:
        task_id = data.get("task_id", "")
        if task_id:
            with self._layer_config_lock:
                self._local_pipeline_steps.pop(task_id, None)
            self._local_pipeline_cancelled.add(task_id)
            with self._kv_cache_lock:
                self._kv_cache.pop(task_id, None)
            logger.info(f"流水线任务取消: {task_id}")

    # ------------------------------------------------------------------
    # 设备画像上报（心跳时附带）
    # ------------------------------------------------------------------
    def _report_device_profile(self) -> None:
        try:
            from device_profiler import get_profile

            profiler = get_profile()
            profile_dict = profiler.to_dict()
            self._device_info = profile_dict
            if self._client is not None:
                self._client.device_info = profile_dict
        except Exception as e:
            logger.warning(f"设备画像上报失败: {e}")


def run_peer() -> None:
    """从节点入口（inference_svc_main client 角色调用，不 import fastapi）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "从节点启动: master=%s:%s node_id=%s",
        os.environ.get("QLH_CLIENT_MASTER_HOST", "127.0.0.1"),
        os.environ.get("QLH_CLIENT_MASTER_PORT", "8888"),
        os.environ.get("QLH_NODE_ID", "") or f"client_{socket.gethostname()}",
    )
    peer = PeerClient()
    peer.run_forever()
