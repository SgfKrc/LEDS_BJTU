# Harness Workbench UI

独立的 React/Vite 工作台。默认调用同源 harness API；开发时可设置
`VITE_HARNESS_API_BASE=http://127.0.0.1:8090`。API 不可用时界面明确显示
`FIXTURE`，只提供离线演示回应，不把 fixture 当作真实模型能力。

```text
npm install
npm run build
npm run dev
```
