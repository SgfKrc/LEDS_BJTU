# TP 孤岛接入指南（路线 A 阶段 1 PoC）

> 状态：PoC 已实现（island 引擎）
>
> 前置阅读：[张量并行外部辅助与混合拆分调研方案](张量并行外部辅助与混合拆分调研方案.md) §2.1
>
> 一句话：孤岛内部用成熟运行时跑张量并行，网关节点把整岛封装为 QLH 集群里的**一个逻辑高算力节点**，承担整请求推理，不参与 PyTorch 层拆分。

---

## 1. 架构

```text
QLH mesh (Tailscale / 局域网, 10~100Mbps, 5~50ms RTT)
   主节点 ◄──TCP 注册/心跳/整请求转发──► [网关节点 = QLH 客户端 + island 引擎]
                                              │
                                              │ POST /v1/chat/completions   (孤岛内网, ≥1GbE, RTT<1ms)
                                              ▼
                               OpenAI 兼容端点 (vLLM / SGLang / llama-server)
                                              │
                                 ┌────────────┼────────────┐
                                 ▼            ▼            ▼
                              GPU rank0    GPU rank1    GPU rankN
                                 ◄── NCCL / RPC 张量并行 ──►
```

- 集群视角：网关节点 = 一个 `engine="island"` 的整请求推理节点，行为与 `llama_cpp` 节点完全一致（不接收 `LAYER_FORWARD`，不加载本地层段）。
- 网关视角：island 引擎将 `chat` / `chat_stream` 整请求转发到孤岛端点，"加载模型" = `GET /v1/models` 健康检查 + 解析后端模型名。
- 孤岛内部的 TP collective 全部发生在孤岛内网，QLH 不感知、不依赖其协议。

## 2. 部署步骤

### 2.1 在孤岛内启动张量并行推理服务

**方案 A：vLLM（推荐，多卡单机或 Ray 多机）**

```bash
# 2 卡张量并行，暴露 OpenAI 兼容端点
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

**方案 B：llama.cpp rpc-server（GGUF 路线，聚合同一局域网的 GPU 机器）**

```bash
# 在孤岛内每台 GPU 机器上启动 rpc-server（仅限孤岛内网，严禁暴露公网）
rpc-server --host 0.0.0.0 --port 50052

# 在其中一台机器上启动 llama-server，聚合各 rpc-server 设备
llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --rpc host1:50052,host2:50052 \
    --host 0.0.0.0 --port 8000
```

> 上游持续警告 GGML RPC 为 fragile/insecure，只允许在孤岛内网使用；这不改变《混合规划》§13.3 "GGML RPC 不作为 QLH 主路线"的结论——RPC 被封装在孤岛内部。

### 2.2 在网关机配置 QLH 节点

网关机需能同时访问孤岛内网与 QLH 集群网络。环境变量（全部 `QLH_ISLAND_*`）：

| 变量 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `QLH_ISLAND_ENABLED` | 是 | 孤岛引擎总开关 | `1` |
| `QLH_ISLAND_BASE_URL` | 是 | OpenAI 兼容端点 | `http://10.0.0.2:8000` |
| `QLH_ISLAND_API_KEY` | 否 | Bearer 凭据（日志/状态自动脱敏） | `sk-xxxx` |
| `QLH_ISLAND_MODEL` | 否 | 后端模型名；留空自动取 `/v1/models` 首个 | `Qwen/Qwen2.5-7B-Instruct` |
| `QLH_ISLAND_TIMEOUT` | 否 | 请求超时秒数（默认 120） | `180` |
| `QLH_ISLAND_CONNECT_TIMEOUT` | 否 | 连接超时秒数（默认 5） | `5` |
| `QLH_ISLAND_GPU_COUNT` | 建议 | 孤岛 GPU 总数（聚合画像） | `2` |
| `QLH_ISLAND_VRAM_GB` | 建议 | 孤岛聚合显存 GB（聚合画像） | `48` |
| `QLH_ISLAND_TP_SIZE` | 建议 | 张量并行度（展示用） | `2` |
| `QLH_ISLAND_BACKEND` | 建议 | 后端标签（展示用） | `vllm-tp2` |

启动网关（示例：作为从节点接入既有集群）：

```bash
export QLH_CLUSTER_SECRET=<集群密钥>
export QLH_NODE_ROLE=client
export QLH_ISLAND_ENABLED=1
export QLH_ISLAND_BASE_URL=http://10.0.0.2:8000
export QLH_ISLAND_GPU_COUNT=2
export QLH_ISLAND_VRAM_GB=48
export QLH_ISLAND_TP_SIZE=2
export QLH_ISLAND_BACKEND=vllm-tp2
python src/api_server.py
```

聚合画像说明：网关无法探测孤岛内部硬件，`GPU_COUNT`/`VRAM_GB` 由部署者声明后进入注册 `device_info.island` 段；主节点按 `min(聚合显存,96)/24×50 + min(GPU数,8)×5 + 60` 上浮节点权重，使孤岛节点排序靠前。**请如实填写**——虚报会导致主节点过度偏向孤岛（对应调研方案中"账面聚合、实际瓶颈"的画像失真风险）。

## 3. 验证

```bash
# 1) 先验证孤岛端点本身可用（在网关机上）
curl http://10.0.0.2:8000/v1/models
curl http://10.0.0.2:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"你好"}]}'

# 2) 启动网关后检查引擎状态（engine 应为 island，端点已脱敏）
curl http://<网关IP>:8000/api/health
curl http://<网关IP>:8000/api/status | python -m json.tool
#   → "engine": "island", "island": {"enabled": true, "backend": "vllm-tp2", ...}

# 3) 通过网关发起对话（整请求由孤岛完成）
curl -X POST http://<网关IP>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "介绍一下张量并行"}'

# 4) 流式（SSE）
curl -N -X POST http://<网关IP>:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "介绍一下张量并行"}'

# 5) 连接主节点后，在主节点确认孤岛节点已注册且档位/权重上浮
curl http://<主节点IP>:8000/api/nodes | python -m json.tool
#   → 孤岛节点 device_info.island.enabled = true，score 显著高于普通节点
```

## 4. 限制与后续

- **不参与 PyTorch 层拆分**：孤岛节点承担整请求推理。主节点在 `compute_layer_assignment` 中按 `device_info.island.enabled` 过滤孤岛节点；网关侧收到分层配置也会拒绝并主动退出分层 worker 池（双保险）。层段形态（A-1）留待后续。
- **首 token 延迟依赖孤岛内网与孤岛负载**：跨 mesh 只付一次请求/响应传输，但孤岛端排队/prefill 不受 QLH 调度控制。
- **凭据脱敏**：`api_key` 不落日志；`base_url` 中的内嵌账号与查询串在 `/api/status`、注册画像、日志中一律剥除。
- **无服务端取消原语**：QLH 侧取消请求时断开 SSE 连接（best-effort），孤岛后端可能继续算完当前请求。
- **模型身份**：孤岛模型无本地 artifact，任务链 ModelIdentity 以"端点指纹 + 后端模型名"的 sha256 摘要替代文件摘要（`format="openai_api"`），统计中如实标注，不伪装成本地文件。
- **网关单点**：孤岛内任一故障 = 整个逻辑节点故障，走现有节点下线→重编排路径，故障域天然干净。
- **后续**：孤岛内部 benchmark 门控（`internal_benchmark_passed`）、A-2 Full Worker（`STAGE_OFFER/STAGE_RESULT`）形态、多孤岛并存与会话粘性。

## 5. 故障排查

| 现象 | 中文错误 | 可能原因 | 处置 |
|---|---|---|---|
| 加载/对话报"孤岛后端不可达" | `IslandUnreachableError` | 孤岛服务未启动 / 防火墙 / IP 写错 | 在网关机 `curl <BASE_URL>/v1/models` 复核连通性 |
| 报"孤岛后端超时" | `IslandTimeoutError` | 孤岛过载、模型冷启动、上下文过长 | 调大 `QLH_ISLAND_TIMEOUT`；检查孤岛 GPU 利用率 |
| 报"孤岛后端 HTTP 错误：… 401/403" | `IslandHTTPError` | 缺少或错误的 `QLH_ISLAND_API_KEY` | 核对孤岛端鉴权配置 |
| 报"孤岛后端 HTTP 错误：… 404" | `IslandHTTPError` | BASE_URL 多写/漏写了 `/v1` 前缀 | BASE_URL 只写到端口（引擎自动拼 `/v1/...`） |
| 报"孤岛流式响应中断" | `IslandStreamInterruptedError` | 孤岛端崩溃 / 反向代理缓冲截断 SSE | 查孤岛服务日志；nginx 需关闭缓冲 (`X-Accel-Buffering: no`) |
| 报"未返回任何模型" | `IslandEngineError` | 后端 `/v1/models` 为空 | 显式设置 `QLH_ISLAND_MODEL` |
| `/api/status` 无 `island` 段 | — | `QLH_ISLAND_ENABLED` 未设或非 `1/true` | 检查环境变量后重启网关 |
| 主节点仍给孤岛节点派层 | — | 旧版本主节点（无 island 过滤） | 升级主节点；网关侧仍会拒绝分层配置兜底 |
