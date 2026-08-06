import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const PYTHON = process.platform === 'win32'
  ? path.join(ROOT, '.venv-packaging-cuda', 'Scripts', 'python.exe')
  : path.join(ROOT, '.venv-packaging-cuda', 'bin', 'python');

function quoted(value) {
  return `"${value}"`;
}

export default defineConfig({
  testDir: path.join(HERE, 'tests'),
  testMatch: 'diffusion-real.spec.js',
  timeout: 5 * 60 * 1000,
  expect: { timeout: 3 * 60 * 1000 },
  fullyParallel: false,
  workers: 1,
  outputDir: path.join(ROOT, 'build', 'sd15-browser-e2e', 'test-results'),
  reporter: [['line']],
  use: {
    baseURL: 'http://127.0.0.1:18080',
    browserName: 'chromium',
    channel: 'msedge',
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: `${quoted(PYTHON)} ${quoted(path.join(ROOT, 'scripts', 'dev_stubs.py'))} --scheduler-port 18220 --inference-port 18210`,
      url: 'http://127.0.0.1:18220/cluster/my-role',
      timeout: 30_000,
      reuseExistingServer: false,
    },
    {
      command: `${quoted(PYTHON)} ${quoted(path.join(ROOT, 'src', 'legacy_control.py'))} --port 18440`,
      url: 'http://127.0.0.1:18440/sessions',
      timeout: 30_000,
      reuseExistingServer: false,
    },
    {
      command: `${quoted(PYTHON)} ${quoted(path.join(ROOT, 'src', 'inference_svc_main.py'))}`,
      env: {
        ...process.env,
        QLH_NODE_ROLE: 'master',
        QLH_INFERENCE_HOST: '127.0.0.1',
        QLH_INFERENCE_PORT: '18110',
      },
      url: 'http://127.0.0.1:18110/v1/health',
      timeout: 30_000,
      reuseExistingServer: false,
    },
    {
      command: `node ${quoted(path.join(ROOT, 'gateway', 'dist', 'main.js'))}`,
      env: {
        ...process.env,
        QLH_API_PORT: '18080',
        QLH_SCHEDULER_URL: 'http://127.0.0.1:18220',
        QLH_INFERENCE_URL: 'http://127.0.0.1:18110',
        QLH_LEGACY_CONTROL_URL: 'http://127.0.0.1:18440',
        QLH_FRONTEND_DIST: path.join(ROOT, 'frontend', 'dist'),
      },
      url: 'http://127.0.0.1:18080/api/health',
      timeout: 30_000,
      reuseExistingServer: false,
    },
  ],
});
