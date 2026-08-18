# 集群接入稳定性与本地 RAG 实施计划

> 状态：规划中（仅文档与调研，2026-08-19）
>
> 更新日期：2026-08-19
>
> 本文把手动加入集群、分布式角色与可用性审计、双机代码同步、主节点本地 RAG 和竞态/时序测试收敛为一组可独立排期的开发票。本文不把真实双机、CUDA、外部 SMTP、公开证书或公网可达性验收提前算作完成。

## 1. 结论摘要

1. **手动加入集群值得做**：用户输入对方主节点地址后进入一次性入群流程，成功后本机强制降级为从节点。不能把长期 `cluster_secret`、TOTP 种子或私钥直接放进邮件/二维码。
2. **认证采用一次性入群授权**：目标节点本地生成公钥/密钥对；管理员在主节点用 Auth App/TOTP 批准；邮件只承担“是否批准”的通知和回复，不承担长期秘密传输。二维码或字符串只携带短期、单次、绑定目标公钥的签名授权票据。
3. **分布式稳定性需要单独的时序门**：当前连接注册、重连、层配置 ACK、租约释放和节点状态投影均应以 `generation/attempt/config_id/lease_id` 形成可审计的状态机，不能靠日志文本或固定 sleep 判断成功。
4. **SSH 同步应是补丁工具的可选传输层**：现有签名补丁帧、`pull/push` 和 7897 代理继续作为默认兜底；SSH 只负责受控的 `status/fetch/apply/verify`，不复制数据库/模型，不执行任意远程 shell，不自动重启服务。
5. **本地 RAG 有必要，但先做 SQLite FTS5**：适合本地文档、模型卡、日志摘要、任务记录和用户手册；不索引权重张量，不把用户资料上传开发组。向量检索作为可选增强，Ollama embedding 或原生 llama embedding sidecar 必须与文本生成运行时隔离。
6. **竞态测试应前置到开发门**：新增确定性事件屏障、可控时钟、连接生命周期矩阵和 SQLite 崩溃恢复测试；真实双机只承担最后的环境门，不再承担发现全部时序缺陷的职责。

## 2. 手动加入集群与认证

### 2.1 用户流程

| 步骤 | 主节点 | 待加入节点 |
|---|---|---|
| 1 | 管理员打开“生成入群邀请”，输入 Auth App 动态码 | 用户输入主节点地址、端口和本机显示名 |
| 2 | 生成短期邀请请求摘要、目标集群 ID 和过期时间 | 本地生成 Ed25519 密钥对并提交公钥、请求摘要 |
| 3 | 管理员确认节点信息并用 Auth App 批准 | 等待一次性授权票据 |
| 4 | 可选：主节点向已配置管理员邮箱发送批准请求 | 管理员回复严格的 `Y <request_digest>`；该请求已由 Auth App/TOTP 确认 |
| 5 | 管理员界面展示二维码和短字符串；如邮件批准，则自动回信同一二维码/字符串 | 扫码或输入字符串，校验签名、TTL、nonce 和目标公钥 |
| 6 | 主节点记录入群审计事件并签发 client-only grant | 节点切换为 `client`，登记主节点 endpoint，开始注册/心跳 |

邮件不是必需依赖。SMTP/IMAP 不可用时，管理员仍可在主节点 Web/TUI 完成批准和二维码下发。邮件回复必须匹配预先配置的管理员地址、请求摘要和一次性 nonce，过期或重复回复均拒绝。自动回信的二维码只能是已签名、短期、单次且绑定目标节点公钥的 grant；它不含长期秘密，即使邮件被转发也不能用于另一台设备或在过期后复用。

### 2.2 授权票据边界

票据至少包含：`schema_version`、`cluster_id` 摘要、`target_node_public_key`、`role=client`、允许能力集合、签发时间、过期时间、nonce、签发者 key id 和签名。二维码只编码票据或受保护的一次性兑换引用，不编码：

- 长期 `cluster_secret`、TOTP seed、恢复码、私钥；
- SMTP 密码、管理员邮箱密码、Tailscale auth key；
- 模型路径、权重正文、用户 prompt 或聊天正文。

兑换成功后立即消费 nonce。主节点只保存哈希化邀请摘要、票据状态和审计事件；目标节点保存自己的私钥，主节点不能代替目标节点生成私钥。

### 2.3 角色与失败语义

角色状态固定为：

`unconfigured -> provisional_master -> joining -> client -> online_client -> disconnected/rejoin`

- 已确认的主节点不能因为收到地址就自动降级；只有本机明确执行“加入现有集群”并通过本地确认才允许切换。
- “加入集群”无论对方当前显示为什么角色，目标节点一律申请 `client`，禁止客户端反向取得 master 权限。
- 地址不可达、票据过期、签名错误、能力不匹配、重复 nonce、主节点拒绝和心跳失败必须返回稳定 reason code，并保留 `request_id`、`generation`、`attempt_id`。
- 入群失败不得清除本地身份、模型资产或用户数据；只有用户明确执行“重置身份”才进入破坏性流程。

## 3. 分布式角色、稳定性和可用性审计

### 3.1 需要冻结的状态机

| 子系统 | 必须具备的栅栏 | 当前审计重点 |
|---|---|---|
| TCP/WSS 连接 | `connection_generation` + 当前 socket 所有权 | 旧 socket 关闭后迟到的心跳/ACK 不得写入新连接；重复注册需幂等 |
| 注册确认 | `registration_attempt` + ACK barrier | 收到确认日志不等于发送通道已可用；ACK 只能在 active state 后发送 |
| 层配置 | `config_id` + `reserve -> prepare -> ACK -> commit` | 断线、旧配置、部分加载和释放必须可重入；过期 ACK 拒绝 |
| 任务执行 | `task_id/step/epoch/lease_id` | 重试只能产生一个 winner；取消、租约到期和迟到结果必须可解释 |
| 节点投影 | 拓扑变更与任务记账分开 | 避免把普通任务 accounting 打成“节点变更: update master”而误导排障 |
| SQLite 控制面 | 事务边界 + WAL checkpoint 策略 | 入群、身份切换、节点租约和审计事件不能出现半提交状态 |

### 3.2 可用性策略

- `distributed_required`：任一必需从节点不可用即 fail-closed，不能静默回退到主节点单机。
- `distributed_preferred`：允许有原因、有时限的本地回退，UI 必须显示 `fallback_reason` 和实际参与节点。
- worker 租约采用 TTL + 心跳续租；清理动作必须幂等，不能因为重复清理把新一代 worker 释放掉。
- 连接重试使用指数退避和抖动；同一 `attempt_id` 禁止同时启动 Legacy TCP 与 WSS 两条数据面。
- 对连续失败的路径启用 circuit breaker，冷却期只做健康探测，不派发大对象或层权重。
- 观测至少记录 `transport/path_kind/generation/attempt_id/config_id/lease_id/fallback_reason`，默认脱敏，不记录 IP、token、权重和正文。

## 4. 双机仓库同步与 SSH 方案

### 4.1 推荐架构

现有双机补丁工具保持主路径：签名补丁帧 -> 传输 -> 校验 -> 应用 -> `HEAD`/工作区验证。新增可选 SSH 传输层，不改变补丁格式和验签规则：

```text
patch_dispatch
  -> transport=tcp_listener (现有默认)
  -> transport=ssh (可选)
       -> 固定远程 helper: status / fetch / apply / verify
```

SSH 约束：

- 仅 Ed25519 key authentication；主节点保存 host fingerprint/allowlist，禁止密码和首次连接自动信任。
- 远端只允许固定 helper 和固定参数，禁止拼接用户输入执行任意 shell。
- 应用前取得维护锁，检查工作区是否为受控检出；应用后验证签名 commit、文件清单和当前 `HEAD`。
- 不通过 SSH 复制 SQLite 数据库、模型权重、用户附件或密钥；只同步代码补丁和必要的 manifest。
- 默认不自动重启服务；重启由从节点本地确认或后续受控 node-management 票处理。
- GitHub/Gitee 直连不稳定时，代码源拉取仍可使用用户的 7897 代理；SSH 失败自动回到现有 `pull/push`，不绕过签名门。

### 4.2 分期

| 票 | 状态 | 内容 | 验收 |
|---|---|---|---|
| `SYNC-SSH-S0` | Planned | 固定 helper、key/fingerprint、锁与失败码契约 | 文档和协议 fixture |
| `SYNC-SSH-S1` | Planned | fake SSH server + 签名补丁帧 loopback | 断连/重放/脏工作区 fail-closed |
| `SYNC-SSH-S2` | Blocked by environment | 同 LAN/Tailscale 双机实测 | 一行提交到两端 `HEAD` 一致且不改用户数据 |
| `SYNC-SSH-S3` | Optional | GUI/TUI 传输选择与 7897 代理诊断 | SSH、TCP、pull/push 三种路径可解释切换 |

在 S2 真机不可用期间，继续使用现有 `pull/push` 工具，不因 SSH 阻塞其他开发票。

## 5. 主节点本地 SQLite RAG

### 5.1 是否值得做

值得做，优先服务四类本地内容：项目文档/计划、模型卡与资产说明、脱敏运行日志、用户自己的任务/知识笔记。RAG 不负责读取或重排模型权重，不应成为推理任务的隐式依赖；没有 embedding 服务时必须可退化到 FTS-only。

文档维护 Agent 已有独立的 `docagent-events.sqlite` 与 embedding 适配器。本项目 RAG 使用主节点用户数据库或独立 `qlh-rag.sqlite`，可以复用适配器和 schema 经验，但不能把两类数据混库或互相暴露。

### 5.2 SQLite 最低实现

建议使用 WAL，并将索引更新放在可恢复的事务/任务队列中。最小表：

```text
rag_sources(source_id, owner_scope, relative_ref, sha256, mime, title, status, created_at, updated_at)
rag_documents(document_id, source_id, revision, text_digest, language, access_scope, status)
rag_chunks(chunk_id, document_id, ordinal, text_digest, token_count, start_offset, end_offset, metadata_json)
rag_embeddings(chunk_id, provider, model_id, model_sha256, dimensions, dtype, vector_blob, created_at)
rag_jobs(job_id, kind, state, cursor, error_code, created_at, updated_at)
rag_query_events(event_id, query_digest, filters_json, result_ids_json, created_at)
```

- FTS5 使用 external-content 表时必须有触发器或事务性重建，避免正文与索引不一致；排序先用 `bm25`。
- 向量扩展可选用 SQLite `vec1`；固定维度、模型 ID 和模型摘要必须与向量绑定。扩展不可用时回退 FTS5，不默认在 Python 中对全库做无界暴力距离计算。
- `owner_scope/access_scope` 是硬过滤条件；查询结果必须返回 source/chunk/revision 引用，支持删除、重建和过期 revision 清理。
- 默认不索引密钥、恢复码、TOTP、邮件认证内容、私钥和原始附件；日志先脱敏再入库。提示词注入内容只能作为资料，不能改变系统权限或任务策略。

### 5.3 Embedding provider

Ollama 提供本地 `/api/embed` 适配器，可接受单条或批量输入，候选模型记录为 `embeddinggemma`、`qwen3-embedding`、`all-minilm`，具体准入以本机资源和模型许可为准。原生 llama 引擎只有在明确声明 embedding 能力、维度和模型身份时才可作为第二 provider；文本生成模型不能因为“使用 llama 引擎”就自动当作 embedding 模型。

Provider 运行在独立 worker/sidecar，具有超时、取消、模型摘要校验和 CPU-only fallback；embedding 不应改变主文本引擎的 Python/Transformers 版本锁。

### 5.4 分期与门槛

| 票 | 状态 | 内容 | 证据门 |
|---|---|---|---|
| `RAG-S0` | Planned | 数据边界、schema、删除/重建和隐私契约 | 迁移 fixture + threat review |
| `RAG-S1` | Planned | SQLite WAL + FTS5、文档/日志/模型卡索引 | 断电/重复导入/删除重建一致 |
| `RAG-S2` | Planned | Ollama `/api/embed` sidecar adapter | 无网络、provider 超时、维度不匹配均可恢复 |
| `RAG-S3` | Candidate | `vec1` 或等价向量后端、FTS+向量混排 | 30 条标注查询的 top-k/引用率基线 |
| `RAG-S4` | Candidate | Web/TUI 检索、引用、重建和容量治理 | 不越权、不泄露敏感字段、模型更换可重建 |

## 6. 竞态与时序测试计划

### 6.1 分期

| 票 | 状态 | 内容 |
|---|---|---|
| `T-RACE-0` | Planned | 盘点连接、注册、ACK、层配置、租约、SQLite、补丁同步状态转移 |
| `T-RACE-1` | Planned | 可控时钟、事件屏障、随机延迟注入；新测试禁止固定 sleep 作为完成判定 |
| `T-RACE-2` | Planned | 同时注册、旧 socket 迟到心跳、发送中断、重复连接、ACK barrier 矩阵 |
| `T-RACE-3` | Planned | 层 reserve/prepare/ACK/release、任务 epoch/winner/取消/lease expiry |
| `T-RACE-4` | Planned | SQLite WAL 提交中断、进程 kill、重启恢复、幂等重放 |
| `T-RACE-5` | Planned | 3 次以上随机化压力/soak，记录失败序列和最小复现种子 |
| `T-RACE-6` | Blocked by environment | 两台真实设备长连接、断网、升级和恢复联合验收 |

### 6.2 最小场景矩阵

- 两个注册请求同时到达：只保留一个 active socket，另一方收到稳定的 duplicate/replace reason code。
- 发送线程在 `close()` 期间写入：不得出现未分类 `WinError 10038`；旧 generation 的写入必须被拒绝。
- 注册确认先到、active state 后到，或 ACK 先于本地加载完成：必须等待状态屏障，不能仅凭日志顺序判定成功。
- 层配置中途断开：释放旧 reservation，迟到 ACK/forward 使用旧 `config_id` 时被拒绝。
- 同一任务重复结果、取消与 lease expiry 交叉：最多一个 winner，所有迟到结果可解释。
- SQLite 提交或补丁应用中进程终止：重启后只能进入已提交或可回滚状态，不能半应用、半入群。

### 6.3 通过标准

新时序测试不得依赖固定秒数睡眠；必须使用事件屏障、可控时钟或明确的条件等待。连续三轮随机化运行无未分类终态、无重复 winner、无 reservation 泄漏；每次失败保存 seed、状态转移和 generation/attempt/config/lease 证据。全量回归仍按现有分钟级门执行，不能用短 smoke 替代。

## 7. 执行顺序

1. `NW4.1` Transport v2 契约与故障矩阵（文档）→ `T-RACE-0/1`。
2. 手动入群授权契约与角色状态机（文档）→ `SYNC-SSH-S0`。
3. `RAG-S0`/`RAG-S1` 本地主节点 SQLite FTS5，不等待外部 GPU 或公网证书。
4. `NW3.1` 本地自签名 WSS loopback，仅用于协议测试，不宣称生产信任。
5. 外部 SMTP、公开域名证书、双 CUDA、IPv4 打洞和真实双机作为后置环境门。

## 8. 调研依据

- [Let's Encrypt Certificates for localhost](https://letsencrypt.org/docs/certificates-for-localhost/)
- [Let's Encrypt Challenge Types](https://letsencrypt.org/docs/challenge-types/)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [SQLite vec1](https://sqlite.org/vec1/doc/trunk/doc/vec1.md)
- [Ollama Embeddings API](https://docs.ollama.com/api/embed)
- [Ollama Embeddings capability](https://docs.ollama.com/capabilities/embeddings)
