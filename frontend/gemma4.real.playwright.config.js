import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Real-model test reuses the developer's explicitly configured local services.
export default defineConfig({
  testDir: path.join(HERE, 'tests'),
  testMatch: 'gemma4-real.spec.js',
  timeout: 180_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['line']],
  use: {
    baseURL: 'http://127.0.0.1:5175',
    browserName: 'chromium',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
});
