# Harness Workbench UI

独立的 React/Vite 工作台。默认调用同源 harness API；开发时可设置
`VITE_HARNESS_API_BASE=http://127.0.0.1:8090`。API 不可用时界面明确显示
`FIXTURE`，只提供离线演示回应，不把 fixture 当作真实模型能力。

```text
npm install
npm run build
npm run dev
```

工作台的 `MCP 控制面` 读取 `/v1/mcp/manifest`，按后端 registry 展示工具、输入 schema 和 configured 状态，并通过 `/v1/mcp/call` 执行调用；external MCP 当前只展示配置声明，不建立真实连接。

聊天页从 `/v1/models` 读取在线模型，运行时页从 `/v1/model-profiles` 读取候选画像；画像的 `candidate` 和能力状态只作准入提示，不代表已经通过真实权重推理门。

视觉回归 smoke 会使用显式提供的 Playwright，不把浏览器依赖写入 harness 运行时：

```text
npm run visual:smoke -- http://127.0.0.1:5181/
```

仓库内若已有 `frontend_cybergothic/node_modules/playwright` 会自动复用；独立环境可通过 `HARNESS_PLAYWRIGHT` 指定模块路径。脚本覆盖桌面/窄屏、深浅色切换、键盘焦点、减少动效和横向溢出。
