# MODEL-FLEET M3 凭据、代理与 gated 许可安全门（2026-08-08）

## 1. 结论

M3 无硬件安全门已在 Windows 控制面完成：

- Hugging Face token 由当前 Windows 用户 DPAPI 保护，磁盘只保存 ciphertext；SQLite、source、pull job、manifest、API 响应和日志只允许出现 `credential_ref`；
- resolve/download 通过同一个模型 HTTP 传输层临时注入 Bearer header，仅显式读取 `QLH_HTTP_PROXY`，不修改系统代理、`HTTP_PROXY`、`HTTPS_PROXY`、`HF_HOME` 或全局解释器；
- gated 元数据和许可证 ID 进入 preflight；缺少凭据返回 `credential_required`，缺少显式接受记录返回 `license_required`；executor 在下载前再次检查，不能绕过 dry-run；
- 许可接受时间写入本地 SQLite 审计元数据，manifest 只记录许可证 ID、`acceptance_required` 和 `accepted_at`；许可证未知时保持阻断；
- 凭据写入/删除、许可接受/撤销仅接受 loopback 请求。远端客户端必须等待 gateway 鉴权/本机转发，不得直接调用控制面变更接口。

这些结果不代表真实 Hugging Face、真实代理服务器或真实 gated 账号已经验收；Linux Secret Service/macOS Keychain adapter 也尚未实现。

## 2. 接口与配置

| 接口/配置 | 语义 | 安全边界 |
|---|---|---|
| `PUT /models/credentials/{id}` | 保存 `secret`，生成 `os:qlh/{id}` | loopback only；响应不回显 secret |
| `GET /models/credentials/{id}` | 查询存在性、保护方式、更新时间 | 不解密、不返回 secret |
| `DELETE /models/credentials/{id}` | 删除本机凭据 | loopback only |
| `GET /models/network` | 返回代理是否配置及无凭据 endpoint | 不返回代理认证信息 |
| `POST /models/licenses/acceptances` | `accepted=true` 后登记 repo/license | loopback only；`unknown` 许可证不可接受 |
| `GET /models/licenses/acceptances` | 返回接受审计记录 | 无 token/secret |
| `DELETE /models/licenses/acceptances` | 撤销接受记录 | loopback only |
| `QLH_HTTP_PROXY` | 模型 resolve/download 专用 HTTP(S) proxy origin | 只接受 `http://`/`https://`；禁止 URL 内嵌账号密码和 path/query |
| `QLH_CREDENTIAL_STORE_DIR` | 覆盖 DPAPI ciphertext 文件目录 | 仅改变存储位置，不改变当前用户保护范围 |

默认 Windows 凭据目录为 `%LOCALAPPDATA%\QLH\credentials`（缺失时回退 `%APPDATA%`）。文件名是 `credential_ref` 的 SHA256，内容包含版本、引用、保护器名称、ciphertext 和更新时间，不包含明文。

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

- `control/test/model-security.e2e-spec.ts`：6/6，通过 fake protector 与真实 Windows DPAPI 往返、ciphertext 检查、代理 dispatcher、Bearer header、gated preflight、executor 防绕过、loopback API 和零明文扫描；
- M3 三套专项（security/source/pull）：16/16；
- control-svc 全量：22 suites、232/232；
- MODEL-FLEET schema：25/25；
- `npm audit --omit=dev --audit-level=high`：0 vulnerability。新增 `undici 8.10.0`；Fastify 路由器通过兼容 override 从 `find-my-way 9.6.0` 收口到 `9.7.0`。

## 5. 剩余门

1. 使用公开仓库完成真实 Hub resolve/download；使用实际 `QLH_HTTP_PROXY` 完成 CONNECT/断线/Range 复测。
2. 真实 gated 账号必须由用户先在上游接受条款，再通过本机凭据 API登记 token；不得在仓库、命令行参数或日志中提供真实 token。
3. Linux/macOS 凭据 adapter、客户端凭据/许可 UI 和 gateway 正式鉴权仍待后续阶段。
4. M2 runtime sidecar 仍优先验证 DeepSeek 7B；Qwen 1.8B legacy adapter 完成前保持 inspection only。
