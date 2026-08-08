# MODEL-FLEET M3 凭据、代理与 gated 许可安全门（2026-08-08）

> **状态**：历史报告，使命完成（2026-08-08 登记）
>
> **生命周期**：废弃（待手动删除）
>
> **替代入口**：M3 结论与验收矩阵已并入《一键模型部署与自治集群远期计划》§16 与《总体下一步计划》L4-MODEL-FLEET 条目；真实下载细节见同期 [M3 真实下载与续传报告](MODEL-FLEET-M3真实下载与续传报告-20260808.md)；确认无必要引用后可手动删除本文件

## 1. 结论

M3 无硬件安全门已在 Windows 控制面完成：

- Hugging Face token 由当前 Windows 用户 DPAPI 保护，磁盘只保存 ciphertext；SQLite、source、pull job、manifest、API 响应和日志只允许出现 `credential_ref`；
- resolve/download 通过同一个模型 HTTP 传输层临时注入 Bearer header；代理按 `QLH_HTTP_PROXY > 用户持久化配置 > 直连` 动态选择，不修改系统代理、`HTTP_PROXY`、`HTTPS_PROXY`、`HF_HOME` 或全局解释器；
- gated 元数据和许可证 ID 进入 preflight；缺少凭据返回 `credential_required`，缺少显式接受记录返回 `license_required`；executor 在下载前再次检查，不能绕过 dry-run；
- 许可接受时间写入本地 SQLite 审计元数据，manifest 只记录许可证 ID、`acceptance_required` 和 `accepted_at`；许可证未知时保持阻断；
- 凭据写入/删除、许可接受/撤销、用户代理写入/清除仅接受 loopback 请求。远端客户端必须等待 gateway 鉴权/本机转发，不得直接调用控制面变更接口。

本机 `verge-mihomo` 的 `127.0.0.1:7897` 已完成 CONNECT、公开 Hugging Face resolve 和小型公开工件 download/Range/受控中断恢复；详见 [M3 真实下载与续传报告](MODEL-FLEET-M3真实下载与续传报告-20260808.md)。这不代表真实模型权重完整 pull、物理代理掉线或真实 gated 账号已经验收。Linux Secret Service/macOS Keychain adapter 也尚未实现。

## 2. 接口与配置

| 接口/配置 | 语义 | 安全边界 |
|---|---|---|
| `PUT /models/credentials/{id}` | 保存 `secret`，生成 `os:qlh/{id}` | loopback only；响应不回显 secret |
| `GET /models/credentials/{id}` | 查询存在性、保护方式、更新时间 | 不解密、不返回 secret |
| `DELETE /models/credentials/{id}` | 删除本机凭据 | loopback only |
| `GET /models/network` | 返回当前生效代理和用户持久化配置 | 区分 `QLH_HTTP_PROXY` / `user` / `direct`；不返回代理认证信息 |
| `PUT /models/network/proxy` | 保存模型专用用户代理 origin | loopback only；写入 SQLite；立即生效，无需重启 |
| `DELETE /models/network/proxy` | 清除模型专用用户代理 | loopback only；环境变量覆盖不受影响 |
| `POST /models/licenses/acceptances` | `accepted=true` 后登记 repo/license | loopback only；`unknown` 许可证不可接受 |
| `GET /models/licenses/acceptances` | 返回接受审计记录 | 无 token/secret |
| `DELETE /models/licenses/acceptances` | 撤销接受记录 | loopback only |
| `QLH_HTTP_PROXY` | 模型 resolve/download 临时代理覆盖，优先级最高 | 只接受 `http://`/`https://`；禁止 URL 内嵌账号密码和 path/query |
| `QLH_CREDENTIAL_STORE_DIR` | 覆盖 DPAPI ciphertext 文件目录 | 仅改变存储位置，不改变当前用户保护范围 |

默认 Windows 凭据目录为 `%LOCALAPPDATA%\QLH\credentials`（缺失时回退 `%APPDATA%`）。文件名是 `credential_ref` 的 SHA256，内容包含版本、引用、保护器名称、ciphertext 和更新时间，不包含明文。

代理行为对齐 Git 的“命令环境覆盖用户配置”思路：自动化、临时会话可设置 `QLH_HTTP_PROXY`；普通用户可通过本机 API 保存或清除长期配置；两者都只影响模型 resolve/download，不修改系统或其他进程的网络设置。当前不允许代理 URL 内嵌账号密码，认证代理需后续引入独立凭据引用。

## 3. 下载门控

```text
source.credential_ref
  -> DPAPI 临时解密（仅内存）
  -> resolve Authorization header
  -> gated/license 元数据
  -> credential_required / license_required / ready
  -> executor 再检查
  -> download Authorization header
  -> digest/inspection
  -> manifest（无 token）
```

`rejected`、`quarantined` 和 `rolled_back` 终态统一向 SSE 发出 `failed` 事件，订阅方不会在 gated 拒绝后继续等待。

## 4. 验证

- `control/test/model-security.e2e-spec.ts`：7/7，通过 fake protector 与真实 Windows DPAPI 往返、ciphertext 检查、代理 dispatcher、环境/用户/直连优先级、Bearer header、gated preflight、executor 防绕过、loopback API 和零明文扫描；
- M3 网络/安全专项六套（http-client/resolver/downloader/security/source/pull）：34/34；
- control-svc 全量：25 suites、250/250；
- MODEL-FLEET schema：25/25；
- `npm audit --omit=dev --audit-level=high`：0 vulnerability。新增 `undici 8.10.0`；Fastify 路由器通过兼容 override 从 `find-my-way 9.6.0` 收口到 `9.7.0`。
- 真实代理 smoke：`verge-mihomo` 监听 `127.0.0.1:7897`；`curl` CONNECT 返回 HTTP 200；项目 `ModelHttpClient -> HfResolver` 通过用户持久化配置解析 `openai-community/gpt2@main`，得到 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`、26 个文件、`gated=false`、MIT，耗时 892 ms，未下载权重。

## 5. 剩余门

1. 公开 Hub resolve、本机代理 CONNECT 和小型公开工件 download/Range/受控中断恢复已完成；真实模型权重完整 pull、进程重启 partial 恢复和工件注册仍待独立磁盘预算门。
2. 真实 gated 账号必须由用户先在上游接受条款，再通过本机凭据 API登记 token；不得在仓库、命令行参数或日志中提供真实 token。
3. Linux/macOS 凭据 adapter、客户端代理/凭据/许可 UI、认证代理的独立凭据引用和 gateway 正式鉴权仍待后续阶段。
4. 下一票进入 DeepSeek 7B runtime sidecar；Qwen 1.8B legacy adapter 完成前保持 inspection only。
