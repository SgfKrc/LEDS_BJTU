import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: path.join(HERE, 'tests'),
  testMatch: 'model-fleet-ui.spec.js',
  timeout: 45_000,
  expect: { timeout: 10_000 },
  workers: 1,
  reporter: [['line']],
  outputDir: path.join(HERE, '..', 'build', 'model-fleet-ui-e2e'),
  use: {
    baseURL: 'http://127.0.0.1:15173',
    browserName: 'chromium',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1280, height: 960 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
});
