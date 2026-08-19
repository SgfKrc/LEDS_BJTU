# 集群接入稳定性与本地 RAG 实施计划

> 状态：部分实施（NW4.1 本地契约门、T-RACE-0 至 T-RACE-5、`CLUSTER-JOIN-S0`、`SYNC-SSH-S0/S1` 本机开发门已完成；真实 SSH、RAG、T-RACE-6 和 NW3.1 尚未实施，2026-08-19）
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

### 2.4 `CLUSTER-JOIN-S0` 本机实现（2026-08-19）

已新增 `src/cluster_join.py`，冻结 `qlh.cluster.join-grant.v1` 本地契约：

- 目标节点用 Ed25519 生成本地 keypair；`build_join_request()` 规范化 IPv4、`[IPv6]:port` 和 MagicDNS authority，并绑定 cluster、node、capabilities、request TTL 和 request digest。
- `issue_join_grant()` 只有在控制面已返回 `auth_verified=true` 时签发；签名使用主节点管理员 Ed25519 key，grant 固定 `grant_type=client_only`、`role=client`，不接受 master/admin capability。
- `encode_join_grant()` 输出 `qlhjoin1.<payload>.<signature>`，同一字符串可作为手工输入和 QR payload；编码不携带 cluster secret、TOTP seed、恢复码、私钥、邮件凭据、模型路径或用户正文。
- `JoinGrantLedger` 使用用户主节点 SQLite `WAL + synchronous=FULL` 记录 nonce，`verify_and_consume_join_grant()` 在验证签名、目标公钥、请求摘要、authority、TTL 和角色后原子消费；重启、并发或重复兑换均返回稳定 `nonce_replayed`。
- `tests/test_cluster_join.py` 覆盖 IPv6、Auth App 未批准、签名/目标/TTL/角色篡改、QR 畸形输入、重启/并发 nonce 重放和敏感字段不泄露，`.venv-test` `6 passed`。

本票不接入 Web/TUI 页面、TCP 注册自动切换、Tailscale CLI、SMTP/IMAP 或真实双机；控制面负责 Auth App/TOTP 校验。`SYNC-SSH-S0/S1` 随后已完成，当前下一票为 `SYNC-SSH-S2` 环境部署与实测。

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

### 3.2 `T-RACE-0` 审计结论（2026-08-19）

| 子系统 | 已确认的实现边界 | 本轮结论 | 后续证明票 |
|---|---|---|---|
| TCP 客户端连接/注册 | `_connect_lock` 串行化 `connect()`；`_connection_generation`、`_registration_epoch` 和幂等断连回调区分会话；发送锁保证单 socket 上的字节帧不交叉。 | 发现并修复一个真实竞态：旧代际接收线程若延迟启动，会在重连后读取新的 `self.sock`，并把新连接关闭，表现为注册后立即断连或 `WinError 10038`。线程现改为在注册成功时绑定 socket 实例和 generation。 | `T-RACE-2`：并发注册、旧 socket 迟到、发送/关闭交错与 ACK barrier。 |
| 注册确认与节点投影 | 注册确认、节点列表推送和层配置推送共用 TCP 接收顺序；服务端维护注册 epoch，调度器在断线时清除该节点的层配置状态。 | 不可把“收到任意首帧”当成注册完成；注册确认必须先于业务推送，客户端也应只在已注册会话上处理业务帧。现有真机曾出现首帧为 `layer_config` 的历史症状，需以确定性 fixture 覆盖其排序和拒绝语义。 | `T-RACE-2`：REGISTER/ACK/业务帧乱序矩阵。 |
| 分层配置事务 | `config_id`、接收 sequence、generation、prepare/commit/ready/release phase 和节点断线中止均已存在；ACK 校验预期分配和当前事务阶段。 | T-RACE-3 已用屏障覆盖 reserve/prepare/commit/ready/release、迟到/错误 generation ACK、断线中止和释放重入；T-RACE-5 再以 144 个固定 seed 注入代际切换、乱序、重复、篡改和 deadline，未出现未分类终态。 | `SYNC-SSH-S2`；真实双机仍后置。 |
| 工作流任务 lease | `workflow_id`、`stage_id`、`attempt_id`、`lease_id`、`lease_epoch` 与 provider identity 已进入 worker 协议；结果/取消按 attempt 身份校验。 | T-RACE-3 已覆盖同 epoch 结果竞态单 winner、取消/租约到期后的迟到结果拒绝和 reservation 清理；T-RACE-5 每个 seed 追加同 epoch 并发提交，始终只有一个 winner 且 reservation 清零。 | `SYNC-SSH-S2`；真实双机仍后置。 |
| 主节点 SQLite | `local_store` 使用 WAL + `synchronous=FULL`，写路径持有进程内锁并以 `BEGIN IMMEDIATE`、commit/rollback 包围；健康检查执行 quick-check 和可回滚写探针。 | T-RACE-4 新增整轮对话与消息计数原子提交、operation_id 收据去重，并用子进程未提交/已提交退出验证 WAL 重开；T-RACE-5 三轮并发写/读 soak 覆盖每个 operation_id 多次重放，计数、WAL 和健康检查均一致。 | `SYNC-SSH-S2`；真实断电/双机仍后置。 |
| 补丁分发 | 既有签名、来源/工作区检查和受控 pull/push 是当前默认路径；SSH S0/S1 本机门已完成。 | T-RACE-4 新增签名 SHA 前置可达性校验、精确 SHA reset、工作区外原子 patch journal、同目标重放和新目标 fencing；S0/S1 已固定 SSH helper、key/fingerprint、维护锁、失败矩阵和 fake loopback。 | `SYNC-SSH-S2`；真实双机应用和升级恢复仍后置。 |

本票不接线 Transport v2，也不改动默认 Legacy TCP 路由。修复的 socket 生命周期回归与现有 TCP、Transport v2、网络路径专项在 `.venv-test` 通过 `146 passed`；这只证明本机确定性行为，不替代从节点长连接验收。

`T-RACE-2` 在同一确定性矩阵中补齐了 REGISTER ACK 后边沿：调度器主动确认和 TCPServer 默认确认均只触发一次 `on_registration_confirmed`；服务端只在 ACK 写入成功后推送节点列表/层配置。旧绑定 socket 写入会被 `ConnectionError` 拦截，关闭交错的底层 `OSError` 保留为异常原因链。TCP/Transport/网络路径专项当前为 `149 passed`；仍不替代真实从节点验收。

`T-RACE-3` 收口了层配置与任务 lease 的本机状态矩阵：版本化层配置 ACK 必须同时匹配 `config_id` 与 `generation`，worker ACK 会回显 generation；覆盖 prepare/commit/ready/release、断线中止、释放重入和旧 ACK 拒绝。任务图新增同 epoch 结果并发提交门，只有一个 winner 能提交，取消/租约到期后的迟到结果被记录并拒绝，活动 reservation 最终清零。`.venv-test` 调度/任务专项 `343 passed`；未使用从节点或真实网络，后续转入 T-RACE-4 持久化恢复矩阵。

`T-RACE-4` 收口本机持久化恢复：主节点 SQLite 每条连接显式使用 `WAL + synchronous=FULL`，完整对话轮次、消息计数和 operation receipt 在同一事务中提交，整轮删除也原子更新计数；子进程在未提交/已提交断点退出后，重开库分别回滚/保留正确状态，重复 operation_id 返回幂等成功而不重复写入。补丁监听器新增工作区外原子 journal，签名目标必须是 fetch 后远端分支的可达 commit，应用时精确 reset 到签名 SHA；半应用只能重放同一签名目标，新目标被拒绝，启动探针不做自动修改。SQLite、任务 journal、补丁、Scheduler 和 API 联合定向回归 `588 passed`。

`T-RACE-5` 收口本机随机压力/soak：新增 `tests/test_t_race5_soak.py`，三轮固定 seed（每轮 48 个，共 144 个）分别覆盖 FakeTransportLink 的显式投递/丢弃/重复/重排/篡改/deadline/代际切换、SQLite WAL 并发写读与 operation_id 幂等重放、任务图同 epoch 并发 winner。`.venv-test` 三轮 `3 passed`，xdist 用时 `43.26s`；失败时会保存 `qlh.t-race-5.failure.v1` 的 seed、操作序列和状态快照。该门仍不替代真实从节点、断网和升级恢复验收；随后 `CLUSTER-JOIN-S0`、`SYNC-SSH-S0/S1` 已完成，当前环境票为 `SYNC-SSH-S2`。

### 3.3 可用性策略

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
| `SYNC-SSH-S0` | Completed（本机契约门） | 固定 helper、Ed25519 key/fingerprint、维护锁与失败码契约 | `11 passed`，不建立 SSH 连接 |
| `SYNC-SSH-S1` | Completed（本机 fake loopback 门） | fake SSH helper/server、临时 Git 真实 fetch/reset、签名补丁帧、维护锁与断连/重放/脏工作区故障 | `9 passed`，不连接真实 SSH |
| `SYNC-SSH-S2` | In progress（客户端完成，远端授权待配置） | 同 LAN/Tailscale 双机实测 | 一行提交到两端 `HEAD` 一致且不改用户数据 |
| `SYNC-SSH-S3` | Optional | GUI/TUI 传输选择与 7897 代理诊断 | SSH、TCP、pull/push 三种路径可解释切换 |

### 4.3 `SYNC-SSH-S0` 实施结果

`tools/ssh_sync_contract.py` 冻结 `qlh.sync_ssh.profile.v1`、`qlh.sync_ssh.helper.v1` 和维护锁 schema。profile 同时要求 Ed25519 host public key 与其 OpenSSH `SHA256:` fingerprint 匹配，生成受管 `known_hosts` 条目；密码、代理命令、远程命令和自动接受新 host key 字段一律拒绝。客户端 argv 固定 `BatchMode=yes`、key-only、`StrictHostKeyChecking=yes`、受管 `UserKnownHostsFile` 和 `GlobalKnownHostsFile=os.devnull`，不经过 shell。

远端仅可执行 `qlh-patch-helper` 的 `status/fetch/apply/verify`；`fetch/verify` 只能携带精确 commit SHA，`apply` 只携带有大小上限的既有签名 patch frame 与 SHA-256 operation ID，并强制附加由远端持有的 managed-checkout maintenance lock。自动重启、任意命令、SQLite/模型/附件传输均被契约拒绝。fixture 同时冻结 path-free helper result 的错误码、`retryable` 与 `next_action`，其中仅 transport unavailable 可回退既有 `pull/push`。`.venv-test` `tests/test_ssh_sync_contract.py` 为 `11 passed`。

本票未读取私钥、未启动 SSH、未写仓库，也不改变 TCP listener 或 `pull/push` 回退。S1 再提供 fake SSH helper/server，验证签名复核、维护锁占用、断连、重放和脏工作区的 fail-closed 行为。当前 Tailnet 双机只读探测确认从节点 API 健康响应且 SSH 端口可达，旧 TCP patch listener 未启动；这只解除 S2 的基础连通性门，不代表 SSH 身份认证或 helper 已部署。S2 仍需创建专用受限账号、部署 helper、交换主机 Ed25519 公钥/fingerprint，并在 S1 完成后执行无用户数据改动的实测。

### 4.4 `SYNC-SSH-S1` 实施结果

新增 `tools/ssh_sync_helper.py`。`FakeSshServer` 仅为本机内存 JSON 通道，不创建 socket；它和固定 CLI 共享同一 helper。helper 只接受 S0 schema 的四个 action，真实执行仅限 `git fetch origin dev`、精确 SHA 可达性校验和 `git reset --hard <signed_sha>`；repo、state、验签公钥由从节点部署配置提供，且三者均不得位于受管检出内。签名/branch/commit/key-id 任一不符都在 fetch/reset 前拒绝，dirty workspace 拒绝且不写成功收据，维护锁独占。

S1 的 `HelperLedger` 在受管检出外原子持久化成功收据，并将 operation ID 绑定至同一目标 commit。若 reset 已完成但 ACK 因连接中断丢失，重放同一 operation ID 直接返回收据、不再 reset；请求前断连、应用中断、已有维护锁、篡改帧、不匹配 `origin`、错误复用 operation ID 和不受管目录均 fail-closed。`tests/test_ssh_sync_helper.py` 以临时 bare remote、seed/worker 两份真实 Git checkout 验证上述路径，专项 `.venv-test` `9 passed`；连同 S0 与既有补丁工具回归为 `48 passed`。没有连接当前从节点 SSH，也没有修改任何真实 checkout。

S2 的部署前置现已明确：从节点安装 helper，生成检出外配置（repo/state/verify-key），用专用受限账号以 forced command 暴露 helper，登记 host Ed25519 public key/fingerprint 和主节点专用 client key；验收顺序必须是 `status -> fetch(target)`（只证明可达）-> `apply(signed frame)` -> `verify(target)`，因为 fetch 不改变 HEAD。

### 4.5 `SYNC-SSH-S2` 当前实施与现场门

新增 `tools/ssh_sync_client.py`：它只能渲染 S0 固定 `qlh-patch-helper` argv，原子写入独立 `known_hosts`，禁用 password/keyboard-interactive/首次信任，且只接受一行、受 S0 schema 约束的 helper JSON。超时、无法启动 SSH、无有效 helper JSON 以及“非零退出却宣称成功”分别收敛为 `transport_unavailable` 或 `remote_helper_rejected`，不会把 SSH banner、路径或 stderr 透传。`tools/patch_dispatch.py --write-frame <external-json>` 同时可原子保存现有发布私钥生成的公开签名 frame，供受控 `apply` 使用，不会保存私钥。

2026-08-19 首次探测曾因账号未授权而失败；随后从节点已完成 SSH 免密授权。重新执行严格 key-only 的固定 `qlh-patch-helper status` 后，`surface@100.100.52.106` 已通过认证，但远端返回 Windows “`qlh-patch-helper` 不是内部或外部命令”，即 helper/forced-command 尚未部署。登记的 Ed25519 host fingerprint 为 `SHA256:TKKb3kNjDJomz5Ko+0X2/Hymt+PrygSUdPZGM6ou3/g`，端口与身份层均正常；不得把该部署缺口误判为网络故障，也不得改用密码、遍历账户或普通 shell。

从节点管理员需在**从节点本地控制台**完成一次性配置：建立独立受限 `qlh_sync` 账号，向其授权主节点 Ed25519 公钥并禁用 PTY、端口转发、agent/X11 转发；为该账号设置只调用 `tools/ssh_sync_helper.py --forced-command` 的 forced command wrapper。wrapper 固定设置 `QLH_SSH_SYNC_HELPER_CONFIG`，配置、ledger/state 和 release 验签公钥均置于 checkout 外；checkout 仅可访问受管 `dev` 工作树，不能同步 SQLite、模型、附件或密钥。专用账号必须获得该工作树的最小读写 ACL，不能直接复用会被锁死为 forced command 的日常 `surface` 远程账户。

完成配置后，主节点在 checkout 外保存 S0 profile/managed known-hosts，依次运行 `ssh_sync_client.py` 的 `status`、`fetch --commit-sha <sha>`、`apply --frame <signed-frame.json> --operation-id <sha256>` 和 `verify --commit-sha <sha>`。验收 frame 由 `patch_dispatch.py --write-frame` 在 commit/push 成功后生成，operation ID 可取该 frame 文件的 SHA-256；只选择不涉及用户数据路径的代码提交。任一失败保留现有 `pull/push` 为默认兜底，不自动 reset、重启或清理从节点。

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
| `T-RACE-0` | Completed（本机审计） | 盘点连接、注册、ACK、层配置、租约、SQLite、补丁同步状态转移；修复旧接收线程迟启动时误关闭新 socket 的 generation/socket 绑定竞态 |
| `T-RACE-1` | Completed（测试基础） | `DeterministicClock`、`EventBarrier` 和 fake link 的无 sleep 时序骨架；随机延迟注入与业务状态矩阵仍后续 |
| `T-RACE-2` | Completed（本机确定性矩阵） | 并发注册、REGISTER/ACK/业务帧排序、旧 socket 迟到、发送/关闭交错、重复连接和 ACK barrier；修复 ACK 后推送边沿缺失 |
| `T-RACE-3` | Completed（本机确定性矩阵） | 层 reserve/prepare/ACK/release、generation 围栏、任务 epoch/winner/取消/lease expiry；`.venv-test` 调度/任务专项 `343 passed` |
| `T-RACE-4` | Completed（本机故障矩阵） | SQLite WAL 提交中断/进程退出/重启恢复/operation_id 幂等重放、消息计数原子维护；签名补丁目标校验、半应用 journal、同目标重放和新目标 fencing；相关专项 `588 passed` |
| `T-RACE-5` | Completed（本机随机压力门） | 三轮固定 seed、每轮 48 个，共 144 个；Transport/SQLite/任务图随机故障与同 epoch winner；`.venv-test` `3 passed / 43.26s`，失败证据保存 seed、操作序列和状态快照 |
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

1. `NW4.1` Transport v2 契约与故障矩阵（本地契约门已完成）→ `NW4.1-F1` fake transport（已完成）→ `T-RACE-0` 状态盘点（已完成）→ `T-RACE-2` TCP 生命周期矩阵（已完成）→ `T-RACE-3` 层配置与任务 lease 状态矩阵（已完成）→ `T-RACE-4` SQLite/补丁应用恢复矩阵（已完成）→ `T-RACE-5` 随机化压力/soak（已完成）→ `CLUSTER-JOIN-S0` 手动入群授权契约。
2. 手动入群授权契约与角色状态机（`CLUSTER-JOIN-S0` 本地契约已完成）→ `SYNC-SSH-S0/S1`（已完成）→ `SYNC-SSH-S2` 环境部署与真实双机验收。
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
