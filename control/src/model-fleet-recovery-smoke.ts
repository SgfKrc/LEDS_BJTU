/** M3.3 real control-process restart and transient proxy outage smoke. */
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { createHash } from 'crypto';
import { once } from 'events';
import * as fs from 'fs';
import * as http from 'http';
import * as net from 'net';
import * as path from 'path';

interface CliOptions {
  outputRoot?: string;
  report?: string;
}

interface Fixture {
  repoId: string;
  revision: string;
  filename: string;
  content: Buffer;
  sha256: string;
}

interface ControlHandle {
  child: ChildProcessWithoutNullStreams;
  output: string[];
}

interface ProxyHarness {
  server: http.Server;
  armDrop: (afterBytes: number, offlineMs: number) => void;
  stats: () => {
    connect_count: number;
    request_count: number;
    rejected_connect_count: number;
    drop_count: number;
  };
}

function usage(): string {
  return [
    'Usage: node dist/model-fleet-recovery-smoke.js [options]',
    '',
    'Options:',
    '  --output-root <dir>  Root for unique smoke artifacts',
    '  --report <file>      JSON report path (default: inside smoke directory)',
  ].join('\n');
}

function parseArgs(argv: string[]): CliOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--help' || key === '-h') throw new Error('help');
    if (!['--output-root', '--report'].includes(key) || index + 1 >= argv.length) {
      throw new Error(`invalid argument: ${key}`);
    }
    values.set(key.slice(2), argv[index + 1]);
    index += 1;
  }
  return {
    outputRoot: values.get('output-root'),
    report: values.get('report'),
  };
}

function sha256(data: Buffer): string {
  return createHash('sha256').update(data).digest('hex');
}

function ggufFixture(repoId: string, revisionChar: string, fill: number): Fixture {
  const content = Buffer.alloc(4 * 1024 * 1024, fill);
  content.write('GGUF', 0, 'ascii');
  content.writeUInt32LE(3, 4);
  content.writeBigUInt64LE(0n, 8);
  content.writeBigUInt64LE(0n, 16);
  return {
    repoId,
    revision: revisionChar.repeat(40),
    filename: 'model.gguf',
    content,
    sha256: sha256(content),
  };
}

function writeJsonAtomic(file: string, value: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf-8');
  fs.renameSync(temporary, file);
}

function listen(server: http.Server | net.Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.removeListener('error', reject);
      const address = server.address();
      if (!address || typeof address === 'string') {
        reject(new Error('server did not expose a TCP port'));
        return;
      }
      resolve(address.port);
    });
  });
}

function closeServer(server: http.Server | net.Server): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

function createOrigin(fixtures: Fixture[]): {
  server: http.Server;
  stats: () => Record<string, { resolves: number; range_starts: number[] }>;
} {
  const byRepo = new Map(fixtures.map((fixture) => [fixture.repoId, fixture]));
  const counters = new Map(fixtures.map((fixture) => [fixture.repoId, {
    resolves: 0,
    range_starts: [] as number[],
  }]));
  const server = http.createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://127.0.0.1');
    if (url.pathname.startsWith('/api/models/')) {
      const repoId = decodeURIComponent(url.pathname.slice('/api/models/'.length));
      const fixture = byRepo.get(repoId);
      if (!fixture) {
        response.writeHead(404).end('not found');
        return;
      }
      counters.get(repoId)!.resolves += 1;
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        sha: fixture.revision,
        gated: false,
        cardData: { license: 'apache-2.0' },
        siblings: [{
          rfilename: fixture.filename,
          size: fixture.content.length,
          sha256: fixture.sha256,
        }],
      }));
      return;
    }

    const fixture = fixtures.find((candidate) => url.pathname === (
      `/${candidate.repoId}/resolve/${candidate.revision}/${candidate.filename}`
    ));
    if (!fixture) {
      response.writeHead(404).end('not found');
      return;
    }
    const range = request.headers.range;
    const match = range ? /^bytes=(\d+)-$/.exec(range) : null;
    if (range && !match) {
      response.writeHead(416).end('invalid range');
      return;
    }
    const start = match ? Number(match[1]) : 0;
    if (!Number.isSafeInteger(start) || start < 0 || start >= fixture.content.length) {
      response.writeHead(416).end('range outside fixture');
      return;
    }
    counters.get(fixture.repoId)!.range_starts.push(start);
    const headers: http.OutgoingHttpHeaders = {
      'content-length': fixture.content.length - start,
      'content-type': 'application/octet-stream',
    };
    if (start > 0) {
      headers['content-range'] = `bytes ${start}-${fixture.content.length - 1}/${fixture.content.length}`;
    }
    response.writeHead(start > 0 ? 206 : 200, headers);
    let offset = start;
    let timer: NodeJS.Timeout | null = null;
    const send = (): void => {
      if (response.destroyed || offset >= fixture.content.length) {
        if (!response.destroyed) response.end();
        return;
      }
      const end = Math.min(offset + 64 * 1024, fixture.content.length);
      response.write(fixture.content.subarray(offset, end));
      offset = end;
      timer = setTimeout(send, 10);
    };
    response.once('close', () => {
      if (timer) clearTimeout(timer);
    });
    send();
  });
  return {
    server,
    stats: () => Object.fromEntries([...counters.entries()].map(([repo, value]) => [
      repo,
      { resolves: value.resolves, range_starts: [...value.range_starts] },
    ])),
  };
}

function createProxy(): ProxyHarness {
  let connectCount = 0;
  let requestCount = 0;
  let rejectedConnectCount = 0;
  let dropCount = 0;
  let dropRemaining: number | null = null;
  let offlineDurationMs = 0;
  let offlineUntil = 0;
  const sockets = new Set<net.Socket>();
  const relay = (
    chunk: Buffer,
    write: (value: Buffer) => void,
    destroy: () => void,
  ): void => {
    if (dropRemaining === null) {
      write(chunk);
      return;
    }
    const allowed = Math.min(dropRemaining, chunk.length);
    if (allowed > 0) write(chunk.subarray(0, allowed));
    dropRemaining -= allowed;
    if (dropRemaining <= 0) {
      dropRemaining = null;
      dropCount += 1;
      offlineUntil = Date.now() + offlineDurationMs;
      destroy();
    }
  };
  const server = http.createServer((request, response) => {
    requestCount += 1;
    if (Date.now() < offlineUntil) {
      rejectedConnectCount += 1;
      response.writeHead(503, { connection: 'close' }).end('proxy temporarily unavailable');
      return;
    }
    let target: URL;
    try {
      target = new URL(request.url ?? '');
    } catch {
      response.writeHead(400).end('absolute proxy URL required');
      return;
    }
    if (target.protocol !== 'http:') {
      response.writeHead(400).end('CONNECT required for non-HTTP targets');
      return;
    }
    const headers: http.OutgoingHttpHeaders = {
      ...request.headers,
      host: target.host,
    };
    delete headers['proxy-connection'];
    delete headers['proxy-authorization'];
    const upstreamRequest = http.request(target, {
      method: request.method,
      headers,
    }, (upstreamResponse) => {
      response.writeHead(upstreamResponse.statusCode ?? 502, upstreamResponse.headers);
      upstreamResponse.on('data', (chunk: Buffer) => relay(
        chunk,
        (value) => response.write(value),
        () => {
          upstreamResponse.destroy();
          response.destroy();
        },
      ));
      upstreamResponse.on('end', () => {
        if (!response.destroyed) response.end();
      });
      upstreamResponse.on('error', () => response.destroy());
    });
    request.on('data', (chunk) => upstreamRequest.write(chunk));
    request.on('end', () => upstreamRequest.end());
    request.on('error', () => upstreamRequest.destroy());
    upstreamRequest.on('error', () => {
      if (!response.headersSent) response.writeHead(502);
      response.end('proxy upstream unavailable');
    });
  });
  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.once('close', () => sockets.delete(socket));
  });
  server.on('connect', (request, client, head) => {
    connectCount += 1;
    if (Date.now() < offlineUntil) {
      rejectedConnectCount += 1;
      client.end('HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n');
      return;
    }
    const target = request.url ?? '';
    const separator = target.lastIndexOf(':');
    const hostname = target.slice(0, separator);
    const port = Number(target.slice(separator + 1));
    if (!hostname || !Number.isInteger(port) || port <= 0) {
      client.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n');
      return;
    }
    const upstream = net.connect(port, hostname);
    sockets.add(upstream);
    upstream.once('close', () => sockets.delete(upstream));
    upstream.once('connect', () => {
      client.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      if (head.length > 0) upstream.write(head);
    });
    client.on('data', (chunk) => upstream.write(chunk));
    client.on('end', () => upstream.end());
    client.on('error', () => upstream.destroy());
    upstream.on('data', (chunk: Buffer) => {
      relay(
        chunk,
        (value) => client.write(value),
        () => {
          upstream.destroy();
          client.destroy();
        },
      );
    });
    upstream.on('end', () => client.end());
    upstream.on('error', () => client.destroy());
  });
  server.once('close', () => {
    for (const socket of sockets) socket.destroy();
  });
  return {
    server,
    armDrop: (afterBytes, offlineMs) => {
      dropRemaining = afterBytes;
      offlineDurationMs = offlineMs;
    },
    stats: () => ({
      connect_count: connectCount,
      request_count: requestCount,
      rejected_connect_count: rejectedConnectCount,
      drop_count: dropCount,
    }),
  };
}

async function reservePort(): Promise<number> {
  const server = net.createServer();
  const port = await listen(server);
  await closeServer(server);
  return port;
}

function startControl(
  port: number,
  sqlitePath: string,
  modelStore: string,
  proxyUrl: string,
  workDir: string,
): Promise<ControlHandle> {
  const main = path.join(__dirname, 'main.js');
  if (!fs.existsSync(main)) {
    throw new Error(`compiled control entry is missing: ${main}`);
  }
  const output: string[] = [];
  const child = spawn(process.execPath, [main], {
    cwd: path.resolve(__dirname, '..'),
    windowsHide: true,
    env: {
      ...process.env,
      QLH_CONTROL_PORT: String(port),
      QLH_SQLITE_PATH: sqlitePath,
      QLH_MODEL_STORE: modelStore,
      QLH_HTTP_PROXY: proxyUrl,
      QLH_CATALOG_SEED_PATH: path.join(workDir, 'missing-catalog.json'),
      QLH_LEGACY_REGISTRY_PATH: path.join(workDir, 'missing-registry.json'),
      QLH_RUNTIME_PYTHON: process.execPath,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdout.setEncoding('utf-8');
  child.stderr.setEncoding('utf-8');
  const record = (chunk: string): void => {
    output.push(chunk);
    while (output.join('').length > 64 * 1024) output.shift();
  };
  child.stdout.on('data', record);
  child.stderr.on('data', record);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`control start timed out:\n${output.join('').slice(-4000)}`));
    }, 20_000);
    const onData = (): void => {
      if (!output.join('').includes(`CONTROL_SVC_LISTENING:${port}`)) return;
      clearTimeout(timeout);
      child.removeListener('exit', onExit);
      child.stdout.removeListener('data', onData);
      resolve({ child, output });
    };
    const onExit = (code: number | null): void => {
      clearTimeout(timeout);
      reject(new Error(`control exited during start (${code}):\n${output.join('').slice(-4000)}`));
    };
    child.stdout.on('data', onData);
    child.once('exit', onExit);
  });
}

async function stopControl(handle: ControlHandle | null): Promise<void> {
  if (!handle || handle.child.exitCode !== null) return;
  const exited = once(handle.child, 'exit');
  handle.child.kill('SIGKILL');
  await Promise.race([
    exited,
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
}

async function requestJson(
  baseUrl: string,
  pathname: string,
  init: RequestInit = {},
): Promise<Record<string, any>> {
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      ...(init.body ? { 'content-type': 'application/json' } : {}),
      ...init.headers,
    },
    signal: AbortSignal.timeout(10_000),
  });
  const body = await response.json() as Record<string, any>;
  if (!response.ok) {
    throw new Error(`${init.method ?? 'GET'} ${pathname} failed (${response.status}): ${JSON.stringify(body)}`);
  }
  return body;
}

async function waitFor<T>(
  description: string,
  read: () => Promise<T | null>,
  timeoutMs = 20_000,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await read();
    if (result !== null) return result;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`timed out waiting for ${description}`);
}

function verifyManifest(modelStore: string, fixture: Fixture): Record<string, unknown> {
  const name = fixture.repoId.split('/').pop()!;
  const manifestPath = path.join(
    modelStore, 'manifests', 'hub', name, `${fixture.revision.slice(0, 12)}.json`,
  );
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as {
    artifact_id?: string;
    files?: Array<{ sha256?: string; size?: number }>;
  };
  if (manifest.files?.[0]?.sha256 !== fixture.sha256
      || manifest.files[0].size !== fixture.content.length) {
    throw new Error(`registered manifest does not match ${fixture.repoId}`);
  }
  return { path: manifestPath, artifact_id: manifest.artifact_id ?? null };
}

export async function run(argv: string[]): Promise<number> {
  let options: CliOptions;
  try {
    options = parseArgs(argv);
  } catch (error) {
    if (error instanceof Error && error.message !== 'help') {
      process.stderr.write(`${error.message}\n`);
    }
    process.stderr.write(`${usage()}\n`);
    return error instanceof Error && error.message === 'help' ? 0 : 2;
  }

  const outputRoot = path.resolve(
    options.outputRoot ?? path.join(__dirname, '../../build/model-fleet'),
  );
  fs.mkdirSync(outputRoot, { recursive: true });
  const workDir = fs.mkdtempSync(path.join(outputRoot, 'm3-3-recovery-'));
  const reportPath = path.resolve(options.report ?? path.join(workDir, 'report.json'));
  const sqlitePath = path.join(workDir, 'control.sqlite3');
  const modelStore = path.join(workDir, 'model-store');
  const restartFixture = ggufFixture('fixture/restart', 'a', 0x31);
  const proxyFixture = ggufFixture('fixture/proxy', 'b', 0x32);
  const origin = createOrigin([restartFixture, proxyFixture]);
  const proxy = createProxy();
  const startedAt = new Date().toISOString();
  const started = Date.now();
  let control: ControlHandle | null = null;

  try {
    const originPort = await listen(origin.server);
    const proxyPort = await listen(proxy.server);
    const controlPort = await reservePort();
    const originUrl = `http://127.0.0.1:${originPort}`;
    const proxyUrl = `http://127.0.0.1:${proxyPort}`;
    const baseUrl = `http://127.0.0.1:${controlPort}`;
    control = await startControl(
      controlPort, sqlitePath, modelStore, proxyUrl, workDir,
    );
    await requestJson(baseUrl, '/models/sources/local-m3-3', {
      method: 'PUT',
      body: JSON.stringify({
        name: 'M3.3 local fixture',
        provider: 'huggingface',
        endpoint: originUrl,
        priority: 1,
        enabled: true,
      }),
    });

    const restartCreated = await requestJson(baseUrl, '/models/pull', {
      method: 'POST',
      body: JSON.stringify({
        idempotency_key: 'm3-3-process-restart',
        source_id: 'local-m3-3',
        source: {
          provider: 'gguf_huggingface',
          repo_id: restartFixture.repoId,
          requested_revision: restartFixture.revision,
          allow_patterns: ['*.gguf'],
        },
      }),
    });
    const restartJobId = String(restartCreated.job_id);
    const restartPartial = path.join(
      modelStore, 'staging', restartJobId, restartFixture.filename,
    );
    await waitFor('restart staging partial', async () => {
      if (fs.existsSync(restartPartial)) {
        const size = fs.statSync(restartPartial).size;
        if (size >= 256 * 1024 && size < restartFixture.content.length) return size;
      }
      const body = await requestJson(baseUrl, `/models/pull/${restartJobId}`);
      if (['failed', 'rejected', 'quarantined'].includes(body.job?.state)) {
        throw new Error(`restart fixture job terminated: ${JSON.stringify(body.job)}`);
      }
      return null;
    });
    await stopControl(control);
    control = null;
    const partialBytes = fs.statSync(restartPartial).size;
    control = await startControl(
      controlPort, sqlitePath, modelStore, proxyUrl, workDir,
    );
    const restartedJob = await waitFor('restarted pull registration', async () => {
      const body = await requestJson(baseUrl, `/models/pull/${restartJobId}`);
      return body.job?.state === 'registered' ? body.job : null;
    }, 30_000);
    if ((restartedJob.progress?.restart_count ?? 0) < 1
        || restartedJob.progress?.resumed_bytes !== partialBytes) {
      throw new Error(`restart recovery metadata mismatch: ${JSON.stringify(restartedJob.progress)}`);
    }

    proxy.armDrop(256 * 1024, 600);
    const proxyCreated = await requestJson(baseUrl, '/models/pull', {
      method: 'POST',
      body: JSON.stringify({
        idempotency_key: 'm3-3-proxy-drop',
        source_id: 'local-m3-3',
        source: {
          provider: 'gguf_huggingface',
          repo_id: proxyFixture.repoId,
          requested_revision: proxyFixture.revision,
          allow_patterns: ['*.gguf'],
        },
      }),
    });
    const proxyJobId = String(proxyCreated.job_id);
    const proxyJob = await waitFor('proxy outage recovery', async () => {
      const body = await requestJson(baseUrl, `/models/pull/${proxyJobId}`);
      if (['failed', 'rejected', 'quarantined'].includes(body.job?.state)) {
        throw new Error(`proxy recovery job terminated: ${JSON.stringify(body.job)}`);
      }
      return body.job?.state === 'registered' ? body.job : null;
    }, 30_000);
    if ((proxyJob.progress?.transfer_retry_count ?? 0) < 1
        || (proxyJob.progress?.resumed_bytes ?? 0) <= 0
        || proxy.stats().drop_count !== 1) {
      throw new Error(`proxy recovery metadata mismatch: ${JSON.stringify({
        progress: proxyJob.progress,
        proxy: proxy.stats(),
      })}`);
    }

    const report: Record<string, unknown> = {
      schema_version: 1,
      status: 'passed',
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      elapsed_ms: Date.now() - started,
      endpoints: { origin: originUrl, proxy: proxyUrl, control: baseUrl },
      restart: {
        job_id: restartJobId,
        killed_partial_bytes: partialBytes,
        progress: restartedJob.progress,
        artifact: verifyManifest(modelStore, restartFixture),
      },
      proxy_outage: {
        job_id: proxyJobId,
        progress: proxyJob.progress,
        proxy: proxy.stats(),
        artifact: verifyManifest(modelStore, proxyFixture),
      },
      origin: origin.stats(),
      artifacts: { work_dir: workDir, sqlite: sqlitePath, model_store: modelStore },
    };
    writeJsonAtomic(reportPath, report);
    process.stdout.write(`${JSON.stringify({ ...report, report: reportPath })}\n`);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeJsonAtomic(reportPath, {
      schema_version: 1,
      status: 'failed',
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      elapsed_ms: Date.now() - started,
      error: message,
      proxy: proxy.stats(),
      origin: origin.stats(),
      control_output: control?.output.join('').slice(-8000) ?? null,
      artifacts: { work_dir: workDir, sqlite: sqlitePath, model_store: modelStore },
    });
    process.stderr.write(`M3.3 recovery smoke failed: ${message}\nreport: ${reportPath}\n`);
    return 1;
  } finally {
    await stopControl(control);
    await Promise.all([
      closeServer(proxy.server),
      closeServer(origin.server),
    ]);
  }
}

if (require.main === module) {
  run(process.argv.slice(2)).then((code) => { process.exitCode = code; });
}
