import {
  existsSync, mkdtempSync, readFileSync, readdirSync, rmSync,
} from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { ProxyAgent } from 'undici';
import { AppModule } from '../src/app';
import { ArtifactStore } from '../src/data/artifact-store';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ConfigDao } from '../src/data/config-dao';
import {
  CredentialProtector, ModelCredentialStore, WindowsDpapiProtector,
} from '../src/data/model-credential-store';
import { HfDownloader } from '../src/data/hf-downloader';
import { HfResolver } from '../src/data/hf-resolver';
import { ModelHttpClient } from '../src/data/model-http-client';
import { ModelInspector } from '../src/data/model-inspector';
import { ModelLicenseAcceptanceRepository } from '../src/data/model-license-acceptance';
import { ModelDiskBudget } from '../src/data/model-disk-budget';
import { PullJobExecutor } from '../src/data/pull-job-executor';
import { PullJobService } from '../src/data/pull-job.service';
import { PullPreflightService } from '../src/data/pull-preflight.service';
import { SqliteStore } from '../src/data/sqlite-store';

const protector: CredentialProtector = {
  name: 'test-protector',
  protect: async (secret) => `enc:${Buffer.from(secret).reverse().toString('base64')}`,
  unprotect: async (ciphertext) => Buffer.from(
    ciphertext.slice(4), 'base64',
  ).reverse().toString(),
};

function tempStore(prefix: string): { dir: string; store: SqliteStore } {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  const store = new SqliteStore(join(dir, 'control.sqlite3'));
  store.open();
  return { dir, store };
}

function credentials(dir: string): ModelCredentialStore {
  return new ModelCredentialStore({
    rootDir: join(dir, 'credentials'), protector,
  });
}

function minimalGguf(): Buffer {
  const version = Buffer.alloc(4);
  version.writeUInt32LE(3);
  return Buffer.concat([
    Buffer.from('GGUF'), version, Buffer.alloc(8), Buffer.alloc(8),
  ]);
}

async function waitForTerminal(jobs: PullJobService, jobId: string): Promise<string> {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    const state = jobs.get(jobId)?.state ?? 'missing';
    if ([
      'registered', 'failed', 'cancelled', 'rejected', 'quarantined', 'rolled_back',
    ].includes(state)) return state;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  return jobs.get(jobId)?.state ?? 'missing';
}

describe('MODEL-FLEET M3 credential and network security', () => {
  const itOnWindows = process.platform === 'win32' ? it : it.skip;

  it('persists only protected ciphertext and returns status without the secret', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-credential-'));
    const vault = credentials(dir);
    const secret = 'hf_fixture_secret_123';
    const saved = await vault.set('os:qlh/hf-main', secret);
    expect(saved.exists).toBe(true);
    expect(JSON.stringify(saved)).not.toContain(secret);
    expect(await vault.get('os:qlh/hf-main')).toBe(secret);
    const files = readdirSync(vault.root).map(
      (name) => readFileSync(join(vault.root, name), 'utf-8'),
    ).join('\n');
    expect(files).not.toContain(secret);
    expect(vault.status('os:qlh/hf-main').exists).toBe(true);
    const rotated = 'hf_fixture_secret_rotated_456';
    await vault.set('os:qlh/hf-main', rotated);
    expect(await vault.get('os:qlh/hf-main')).toBe(rotated);
    expect(vault.delete('os:qlh/hf-main')).toBe(true);
    expect(await vault.get('os:qlh/hf-main')).toBeNull();
    rmSync(dir, { recursive: true, force: true });
  });

  itOnWindows('round-trips through Windows DPAPI without plaintext on disk', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-dpapi-'));
    const vault = new ModelCredentialStore({
      rootDir: join(dir, 'credentials'),
      protector: new WindowsDpapiProtector(),
    });
    const secret = 'dpapi_fixture_secret_456';
    await vault.set('os:qlh/dpapi-smoke', secret);
    expect(await vault.get('os:qlh/dpapi-smoke')).toBe(secret);
    const raw = readdirSync(vault.root).map(
      (name) => readFileSync(join(vault.root, name), 'utf-8'),
    ).join('\n');
    expect(raw).not.toContain(secret);
    rmSync(dir, { recursive: true, force: true });
  });

  it('uses only explicit QLH_HTTP_PROXY and injects authorization per request', async () => {
    let captured: RequestInit & { dispatcher?: unknown } = {};
    const secret = 'hf_request_only_secret';
    const client = new ModelHttpClient({
      env: { QLH_HTTP_PROXY: 'http://proxy.example:8080' },
      fetchFn: async (_url, init) => {
        captured = init ?? {};
        return new Response('{}', { status: 200 });
      },
    });
    await client.fetch('https://huggingface.co/api/models/org/model', {}, {
      token: secret,
    });
    expect(new Headers(captured.headers).get('authorization')).toBe(`Bearer ${secret}`);
    expect(captured.dispatcher).toBeInstanceOf(ProxyAgent);
    expect(client.proxyStatus()).toEqual({
      configured: true,
      source: 'QLH_HTTP_PROXY',
      endpoint: 'http://proxy.example:8080',
    });
    expect(JSON.stringify(client.proxyStatus())).not.toContain(secret);
    expect(() => new ModelHttpClient({
      proxyUrl: 'http://user:password@proxy.example:8080',
    })).toThrow('must not contain embedded credentials');
    await client.onApplicationShutdown();
  });

  it('returns gated preflight states until credential and license gates pass', async () => {
    const { dir, store } = tempStore('qlh-mf-gated-preflight-');
    const vault = credentials(dir);
    const licenses = new ModelLicenseAcceptanceRepository(
      new ClusterSettingsRepository(store),
    );
    const secret = 'hf_gated_fixture_secret';
    await vault.set('os:qlh/gated', secret);
    const seenAuth: Array<string | null> = [];
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async (_url, init) => {
        seenAuth.push(new Headers(init?.headers).get('authorization'));
        return new Response(JSON.stringify({
          sha: 'a'.repeat(40), gated: 'manual', cardData: { license: 'llama3' },
          siblings: [{ rfilename: 'model.gguf', size: 24 }],
        }), { status: 200 });
      },
    });
    const preflight = new PullPreflightService(
      new HfResolver(http),
      { evaluate: () => ({
        total_bytes: 24, existing_bytes: 0, disk_required_bytes: 24,
        disk_available_bytes: 100, sufficient: true,
      }) } as unknown as ModelDiskBudget,
      new ArtifactStore(join(dir, 'artifacts')),
      vault,
      licenses,
    );
    const source = {
      schema_version: 1 as const, source_id: 'gated', name: 'Gated',
      provider: 'huggingface' as const, endpoint: 'https://huggingface.co',
      credential_ref: 'os:qlh/gated', priority: 1, enabled: true, builtin: false,
    };
    const pending = await preflight.resolve({ source, repoId: 'org/gated-model' });
    expect(pending.status).toBe('license_required');
    expect(pending.access).toMatchObject({
      gated: true, credential_available: true,
      acceptance_required: true, accepted_at: null,
    });
    const acceptance = licenses.accept('org/gated-model', 'llama3');
    const ready = await preflight.resolve({ source, repoId: 'org/gated-model' });
    expect(ready.status).toBe('ready');
    expect(ready.access.accepted_at).toBe(acceptance.accepted_at);
    expect(() => licenses.accept('org/no-license', 'unknown')).toThrow(
      'license_id is invalid',
    );
    const withoutCredential = await preflight.resolve({
      source: { ...source, credential_ref: null }, repoId: 'org/gated-model',
    });
    expect(withoutCredential.status).toBe('credential_required');
    expect(seenAuth).toContain(`Bearer ${secret}`);
    expect(readFileSync(join(dir, 'control.sqlite3')).includes(Buffer.from(secret))).toBe(false);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('rechecks gated acceptance in the executor before downloading', async () => {
    const { dir, store } = tempStore('qlh-mf-gated-executor-');
    const jobs = new PullJobService(store);
    const vault = credentials(dir);
    const licenses = new ModelLicenseAcceptanceRepository(
      new ClusterSettingsRepository(store),
    );
    const secret = 'hf_executor_fixture_secret';
    await vault.set('os:qlh/gated', secret);
    const gguf = minimalGguf();
    let downloads = 0;
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async (url, init) => {
        expect(new Headers(init?.headers).get('authorization')).toBe(`Bearer ${secret}`);
        if (String(url).includes('/api/models/')) {
          return new Response(JSON.stringify({
            sha: 'b'.repeat(40), gated: true, cardData: { license: 'llama3' },
            siblings: [{ rfilename: 'model.gguf', size: gguf.length }],
          }), { status: 200 });
        }
        downloads += 1;
        return new Response(new Uint8Array(gguf), {
          status: 200, headers: { 'content-length': String(gguf.length) },
        });
      },
    });
    const artifactStore = new ArtifactStore(join(dir, 'artifacts'));
    const executor = new PullJobExecutor(
      jobs, new HfResolver(http),
      new HfDownloader(http, { progressThrottleMs: 0 }),
      artifactStore, new ModelInspector(), vault, licenses,
    );
    const source = {
      provider: 'gguf_huggingface' as const,
      repo_id: 'org/gated-model', requested_revision: 'main',
      allow_patterns: ['*.gguf'], credential_ref: 'os:qlh/gated',
    };
    const rejected = jobs.create({ idempotencyKey: 'gated-rejected', source });
    const rejectedEvents: string[] = [];
    const unsubscribe = jobs.subscribe((event) => {
      if (event.job_id === rejected.job_id) rejectedEvents.push(event.event);
    });
    executor.start(rejected.job_id);
    expect(await waitForTerminal(jobs, rejected.job_id)).toBe('rejected');
    expect(jobs.get(rejected.job_id)?.error?.code).toBe('license_acceptance_required');
    expect(rejectedEvents).toContain('failed');
    unsubscribe();
    expect(downloads).toBe(0);

    const acceptance = licenses.accept('org/gated-model', 'llama3');
    const accepted = jobs.create({ idempotencyKey: 'gated-accepted', source });
    executor.start(accepted.job_id);
    expect(await waitForTerminal(jobs, accepted.job_id)).toBe('registered');
    expect(downloads).toBe(1);
    const manifest = artifactStore.readManifest('hub', 'gated-model', 'bbbbbbbbbbbb');
    expect(manifest?.license).toEqual({
      id: 'llama3', acceptance_required: true,
      accepted_at: acceptance.accepted_at,
    });
    expect(JSON.stringify(jobs.get(accepted.job_id))).not.toContain(secret);
    expect(JSON.stringify(manifest)).not.toContain(secret);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('exposes credential status and explicit license acceptance without echoing secrets', async () => {
    const { dir, store } = tempStore('qlh-mf-security-api-');
    const vault = credentials(dir);
    const http = new ModelHttpClient({
      proxyUrl: 'http://proxy.example:8080',
      fetchFn: async () => new Response('{}'),
    });
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(store)
      .overrideProvider(ModelCredentialStore).useValue(vault)
      .overrideProvider(ModelHttpClient).useValue(http)
      .overrideProvider(ConfigDao).useValue({
        enabled: false, dbEnabled: () => false,
        getConnectionInfo: () => ({}), ping: async () => ({ ok: true }),
      })
      .compile();
    const app: any = moduleRef.createNestApplication(new FastifyAdapter());
    await app.init();
    const secret = 'hf_api_fixture_secret';
    try {
      const remote = await app.inject({
        method: 'PUT', url: '/models/credentials/hf-main',
        remoteAddress: '192.0.2.10', payload: { secret },
      });
      expect(remote.statusCode).toBe(403);
      const saved = await app.inject({
        method: 'PUT', url: '/models/credentials/hf-main',
        payload: { secret },
      });
      expect(saved.statusCode).toBe(200);
      expect(saved.body).not.toContain(secret);
      expect(saved.json().credential.credential_ref).toBe('os:qlh/hf-main');
      const status = await app.inject({
        method: 'GET', url: '/models/credentials/hf-main',
      });
      expect(status.json().credential.exists).toBe(true);
      expect(status.body).not.toContain(secret);
      const refused = await app.inject({
        method: 'POST', url: '/models/licenses/acceptances',
        payload: { repo_id: 'org/gated-model', license_id: 'llama3', accepted: false },
      });
      expect(refused.statusCode).toBe(422);
      const accepted = await app.inject({
        method: 'POST', url: '/models/licenses/acceptances',
        payload: { repo_id: 'org/gated-model', license_id: 'llama3', accepted: true },
      });
      expect(accepted.statusCode).toBe(200);
      const network = await app.inject({ method: 'GET', url: '/models/network' });
      expect(network.json().proxy).toEqual({
        configured: true, source: 'QLH_HTTP_PROXY',
        endpoint: 'http://proxy.example:8080',
      });
      expect(network.body).not.toContain(secret);
      const rawCredentials = readdirSync(vault.root).map(
        (name) => readFileSync(join(vault.root, name), 'utf-8'),
      ).join('\n');
      expect(rawCredentials).not.toContain(secret);
      expect(readFileSync(join(dir, 'control.sqlite3')).includes(Buffer.from(secret))).toBe(false);
    } finally {
      await app.close();
      store.close();
      if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
    }
  });
});
