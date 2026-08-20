import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// 与既有 frontend 的 e2e 端口错开，两套界面可同时跑测试。
const BASE_URL = 'http://127.0.0.1:15184';

export default defineConfig({
  testDir: path.join(HERE, 'tests'),
  timeout: 45_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [['line']],
  outputDir: path.join(HERE, '..', 'build', 'cybergothic-e2e'),
  use: {
    baseURL: BASE_URL,
    // 沿用仓库既有约定：用系统安装的 Edge，不额外下载浏览器。
    browserName: 'chromium',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1440, height: 900 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 15184',
    cwd: HERE,
    url: BASE_URL,
    reuseExistingServer: false,
    timeout: 40_000,
  },
});
