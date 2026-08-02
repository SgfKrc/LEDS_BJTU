# QLH API 网关（gateway/）

QLH 微服务改造阶段 2 的对外 API 网关：TypeScript + **NestJS**（`@nestjs/platform-fastify` adapter）。

- 对外端口：`8000`（`QLH_API_PORT` 可覆盖），挂载 `/api` 前缀
- 职责：对外 REST 110 端点契约、SSE 流式代理、静态前端托管；聚合/代理 scheduler-svc、inference-svc、control-svc
- 专项文档：[docs/TUI适配实施计划.md](../docs/TUI适配实施计划.md)（TUI 相关适配与验收）· [docs/微服务架构改造计划.md](../docs/微服务架构改造计划.md)（总体计划）

## 常用命令

```bash
npm install          # 安装依赖（含 fastify overrides，见下）
npm test             # 全部 e2e 契约测试（--runInBand）
npm run test:tui     # 仅 TUI 契约测试
npm run build        # tsc 编译到 dist/
npm start            # 启动网关（需 QLH_API_PORT/QLH_SCHEDULER_URL 等环境变量，T2 起）
```

## 已知技术坑（2026-08-02 排障记录，勿删）

> 两个坑在 T1（网关骨架 + TUI 契约测试）阶段实测踩到并已解决。**任何改动 fastify 版本、beforeAll 初始化或 404 处理的 PR 都必须先跑 `npm run test:tui`。**

### 坑 1：`@nestjs/platform-fastify` 硬编码 `fastify@5.10.0`，导致版本嵌套

**现象**：`npm ls fastify` 显示 `fastify@5.11.0`（顶层）与
`@nestjs/platform-fastify/node_modules/fastify@5.10.0`（嵌套）并存。

**根因**：`@nestjs/platform-fastify@11.1.28` 的 `dependencies` 中 fastify 是**精确版本** `5.10.0`（非范围），
npm 无法 dedupe 到顶层版本，必然嵌套安装。platform-fastify 实际加载的是嵌套的 5.10.0。

**修复**：`package.json` 中的 npm `overrides` 强制统一：

```json
"overrides": { "fastify": "^5.11.0" }
```

**验证**：`npm ls fastify` 无嵌套；`require('@nestjs/platform-fastify/node_modules/fastify')` 解析到顶层。

> 注：5.10.0 本身在 404 处理上有回归（与坑 2 叠加后表现为请求挂起/崩溃），统一到 5.11.0（修复版）是双保险。

### 坑 2：supertest 首个请求在 fastify `ready()` 前到达 → 404 路径崩溃

**现象**：`npm test` 首个请求即崩：
`TypeError: Cannot read properties of undefined (reading 'length')`，
堆栈 `fastify/lib/hooks.js → preParsingHookRunner → runPreParsing → routeHandler`，
随后 jest 报 `Exceeded timeout` 且 `Jest did not exit`。

**根因（排障链）**：裸 fastify 404 正常 → NestJS 实例 + 真实 HTTP 404 正常 → NestJS 实例 +
supertest 崩溃 → 先 `instance.inject()` 一次再 supertest 就正常。
结论：**NestJS 的 `app.init()` 不等待 fastify 的 `ready()`**；fastify 的
fourOhFour 404 context 的 hooks 初始化发生在 `preReady` 钩子（`four-oh-four.js`
中 `avvio.once('preReady', ...)`），`ready()` 前 `context.preParsing` 是 `undefined`
（而非 null），`route.js` 的 `runPreParsing` 判断 `preParsing !== null` 为真后
对 undefined 数组取 `.length` 崩溃。supertest 对 `app.getHttpServer()` 会自行
`server.listen(0)`，首个请求可能在 ready 完成前到达。

**修复**：测试 `beforeAll` 中显式等待 ready（`test/tui-contract.e2e-spec.ts`）：

```ts
await app.init();
await (app.getHttpAdapter().getInstance() as any).ready();
```

**影响面**：所有后续 e2e 测试任务（T2-T6）的 beforeAll 都必须带这一行；生产环境
`app.listen()` 内部会等 ready，不受影响。

## 结构说明

```
gateway/
├── src/
│   ├── app.ts          # createApp()：AppModule + /api 前缀 + CatchAllController(404)
│   └── main.ts         # 启动入口（T2 起完整启用）
├── test/
│   └── tui-contract.e2e-spec.ts   # TUI 契约测试：38 端点用例 + 5 细节断言(43 用例)，T2-T6 逐个打开
├── package.json        # 含 fastify overrides（见坑 1）
└── jest.config.js      # testRegex 限定 test/ 目录
```

- 404 统一走 `CatchAllController`（`@All('*')`）抛 `NotFoundException`，由 T2 的异常过滤器输出
  JSON `detail`；依赖 find-my-way 匹配优先级（静态 > 参数 > 通配）保证真实路由不被抢。
