# MODEL-FLEET M3 真实下载与续传报告（2026-08-08）

> **状态**：历史报告，使命完成（2026-08-08 登记）
>
> **生命周期**：废弃（待手动删除）
>
> **替代入口**：真实网络门结论已并入《一键模型部署与自治集群远期计划》§16 与《总体下一步计划》L4-MODEL-FLEET 条目；确认无必要引用后可手动删除本文件

## 1. 结论

M3 小型公开工件网络门已通过：

- 固定公开 fixture 为 `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e` 的 `merges.txt`，不下载模型权重；
- `HfResolver` 通过 `blobs=true` 获取实际文件大小和 LFS SHA-256，缺失/非法 size 时 fail-closed，不再把未知尺寸当成 0；
- `HfDownloader` 对续传强制要求 `206` 和匹配的 `Content-Range`，拒绝服务器忽略 Range、起点错位、总大小冲突、响应体截断和已有文件尺寸欺骗；
- 通过本机 `verge-mihomo` 的 `127.0.0.1:7897` 完成完整下载、16 KiB 后受控中断、同一 partial 文件 Range 续传和最终 SHA-256 对比；
- 环境变量覆盖与 SQLite 用户持久化代理各实跑一次，二者均通过，用户配置运行明确报告 `source=user`。

本报告证明公开小工件的真实 HTTP 传输与恢复机制，不等于真实模型权重完整 pull、进程崩溃恢复、物理代理掉线或 gated 仓库已经验收。

## 2. 固定实测结果

| 阶段 | 结果 |
|---|---|
| resolve | revision 固定为 `607a30d783dfa663caf39e06633721c8d4cfcd7e`；`merges.txt` 大小 `456318` bytes |
| 完整下载 | HTTP `200`；完成 `456318` bytes |
| 受控中断 | 首次请求 HTTP `200`；写入 `16384` bytes 后由 AbortController 中断并保留 partial |
| Range 恢复 | HTTP `206`；`Content-Range: bytes 16384-456317/456318` |
| 完整性 | 完整文件与恢复文件均为 `sha256:1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| 环境代理运行 | `QLH_HTTP_PROXY=http://127.0.0.1:7897`；`status=passed` |
| 用户代理运行 | 独立 SQLite 保存 `http://127.0.0.1:7897`；环境覆盖清除；`source=user`；`status=passed`；3991 ms |

用户代理运行的本地 JSON 证据位于忽略目录 `build/model-fleet/hf-network-smoke-RoN0TQ/report.json`。该路径是本机运行产物，不作为仓库可移植 fixture。

## 3. 可重复入口

构建后运行固定 smoke：

```powershell
cd control
npm run build
$env:QLH_HTTP_PROXY = "http://127.0.0.1:7897"
npm run model-fleet:network-smoke -- --output-root ..\build\model-fleet
```

复用控制面用户代理配置时，不设置 `QLH_HTTP_PROXY`，并让 CLI 读取同一 SQLite：

```powershell
npm run model-fleet:network-smoke -- --sqlite <control.sqlite3>
```

CLI 固定 revision、文件名和 2 MiB 预算；每次在输出根目录创建唯一目录，保留完整文件、恢复文件和原子写入的 `report.json`。失败同样生成报告并返回非零退出码。

## 4. 自动化门

- `control/test/hf-resolver.e2e-spec.ts`：3/3，覆盖 blob 查询、LFS 元数据和未知 size 拒绝；
- `control/test/hf-downloader.e2e-spec.ts`：6/6，覆盖合法 206、Range 被忽略、范围错位、截断、已有尺寸错位和合法空文件；
- M3 网络/安全六套（http-client/resolver/downloader/security/source/pull）：34/34；
- control-svc 全量：25 suites、250/250；
- 生产依赖审计：0 vulnerability。

## 5. 下一门

1. 按 M2 遗留进入 DeepSeek 7B runtime sidecar：隔离进程加载、超时/崩溃回收、成功后登记可运行能力；不直接复用 control-svc 进程加载权重。
2. 真实模型权重完整 pull 需另设磁盘预算和固定 LFS SHA-256，再验证进程重启后的 partial 恢复与工件注册。
3. 物理代理进程退出/重启、认证代理、真实 gated 账号和非 Windows credential adapter 仍属于后续环境门。
