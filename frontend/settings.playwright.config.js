import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE_URL = 'http://127.0.0.1:15174';

export default defineConfig({
  testDir: path.join(HERE, 'tests'),
  testMatch: ['settings-ui.spec.js', 'shell-ux.spec.js'],
  timeout: 45_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [['line']],
  outputDir: path.join(HERE, '..', 'build', 'settings-ui-e2e'),
  use: {
    baseURL: BASE_URL,
    browserName: 'chromium',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1280, height: 960 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 15174',
    cwd: HERE,
    url: BASE_URL,
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
