import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const url = 'http://127.0.0.1:15173';
let server = null;
let serverOutput = '';

async function ready() {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(1_000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForServer() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (await ready()) return;
    if (server?.exitCode !== null) {
      throw new Error(`Vite exited before becoming ready\n${serverOutput}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Vite did not become ready within 30s\n${serverOutput}`);
}

async function stopServer() {
  if (!server || server.exitCode !== null) return;
  server.kill();
  await Promise.race([
    once(server, 'exit'),
    new Promise((resolve) => setTimeout(resolve, 3_000)),
  ]);
  if (server.exitCode !== null) return;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], {
      stdio: 'ignore', windowsHide: true,
    });
  } else {
    server.kill('SIGKILL');
  }
}

try {
  if (!(await ready())) {
    server = spawn(process.execPath, [
      'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '15173',
    ], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const collect = (chunk) => {
      serverOutput = `${serverOutput}${chunk.toString()}`.slice(-8_000);
    };
    server.stdout.on('data', collect);
    server.stderr.on('data', collect);
    await waitForServer();
  }

  const test = spawn(process.execPath, [
    'node_modules/@playwright/test/cli.js', 'test',
    '--config', 'model-fleet.playwright.config.js',
  ], { cwd: root, stdio: 'inherit', windowsHide: true });
  const [code] = await once(test, 'exit');
  process.exitCode = typeof code === 'number' ? code : 1;
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  await stopServer();
}
