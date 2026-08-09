import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Test } from '@nestjs/testing';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import { ArtifactStore } from '../src/data/artifact-store';
import { ConfigDao } from '../src/data/config-dao';
import { HfResolver } from '../src/data/hf-resolver';
import { ModelHttpClient } from '../src/data/model-http-client';
import { ModelCredentialStore } from '../src/data/model-credential-store';
import { ModelLicenseAcceptanceRepository } from '../src/data/model-license-acceptance';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ModelDiskBudget } from '../src/data/model-disk-budget';
import { ModelSourceRepository } from '../src/data/model-source-repository';
import { PullPreflightService } from '../src/data/pull-preflight.service';
import { SqliteStore } from '../src/data/sqlite-store';
import { AppModule } from '../src/app';

function tempStore(): { dir: string; store: SqliteStore } {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-source-'));
  const store = new SqliteStore(join(dir, 'control.sqlite3'));
  store.open();
  return { dir, store };
}

function resolverFixture(): HfResolver {
  const content = Buffer.from('model-bytes');
  const digest = require('crypto').createHash('sha256').update(content).digest('hex');
  return new HfResolver(new ModelHttpClient({
    proxyUrl: null,
    fetchFn: async (url: RequestInfo | URL) => {
      const target = String(url);
      if (!target.includes('/api/models/')) return new Response('not found', { status: 404 });
      return new Response(JSON.stringify({
        sha: target.includes('mirror') ? 'b'.repeat(40) : 'a'.repeat(40),
        siblings: [{ rfilename: 'model.gguf', size: content.length, sha256: digest }],
      }), { status: 200 });
    },
  }), 'https://default.invalid');
}

function securityDeps(dir: string, store: SqliteStore): {
  credentials: ModelCredentialStore;
  licenses: ModelLicenseAcceptanceRepository;
} {
  const protector = {
    name: 'test-protector',
    protect: async (secret: string) => Buffer.from(secret).toString('base64'),
    unprotect: async (ciphertext: string) => Buffer.from(ciphertext, 'base64').toString(),
  };
  return {
    credentials: new ModelCredentialStore({
      rootDir: join(dir, 'credentials'), protector,
    }),
    licenses: new ModelLicenseAcceptanceRepository(
      new ClusterSettingsRepository(store),
    ),
  };
}

describe('MODEL-FLEET sources and dry-run (MF-N4)', () => {
  it('allows loopback HTTP sources but rejects remote plaintext HTTP', () => {
    const { dir, store } = tempStore();
    const sources = new ModelSourceRepository(new ClusterSettingsRepository(store));
    expect(sources.upsert({
      source_id: 'local-fixture',
      name: 'Local fixture',
      provider: 'huggingface',
      endpoint: 'http://127.0.0.1:8080',
      credential_ref: null,
      priority: 1,
      enabled: true,
    }).endpoint).toBe('http://127.0.0.1:8080');
    expect(() => sources.upsert({
      source_id: 'lan-plaintext',
      name: 'LAN plaintext',
      provider: 'huggingface',
      endpoint: 'http://192.168.1.20:8080',
      credential_ref: null,
      priority: 2,
      enabled: true,
    })).toThrow('HTTP is loopback-only');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('keeps source overrides local, rejects plaintext credentials, and re-resolves per source', async () => {
    const { dir, store } = tempStore();
    const sources = new ModelSourceRepository(
      new (require('../src/data/cluster-settings-repository').ClusterSettingsRepository)(store),
    );
    const mirror = sources.upsert({
      source_id: 'team-mirror',
      name: 'Team mirror',
      provider: 'huggingface',
      endpoint: 'https://mirror.example',
      credential_ref: 'os:qlh/team-mirror',
      priority: 5,
      enabled: true,
    });
    expect(sources.preferred('huggingface')?.source_id).toBe('team-mirror');
    expect(JSON.stringify(sources.list())).not.toContain('plain-secret');

    const storeArtifacts = new ArtifactStore(join(dir, 'model-store'));
    const security = securityDeps(dir, store);
    await security.credentials.set('os:qlh/team-mirror', 'fixture-token');
    const diskBudget = {
      evaluate: () => ({
        total_bytes: 11,
        existing_bytes: 0,
        disk_required_bytes: 11,
        disk_available_bytes: 100,
        sufficient: true,
      }),
    };
    const preflight = new PullPreflightService(
      resolverFixture(), diskBudget as any, storeArtifacts,
      security.credentials, security.licenses,
    );
    const first = await preflight.resolve({
      source: mirror,
      repoId: 'org/model',
    });
    expect(first.status).toBe('ready');
    expect(first.source.endpoint).toBe('https://mirror.example');
    expect(first.resolved_revision).toBe('b'.repeat(40));
    expect(first.source.credential_ref).toBe('os:qlh/team-mirror');
    expect(storeArtifacts.listStaging('dry-run')).toEqual([]);

    sources.delete('team-mirror');
    expect(sources.get('team-mirror')).toBeNull();
    sources.reset();
    expect(sources.preferred('huggingface')?.source_id).toBe('hf-official');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('returns insufficient_storage without creating a pull job', async () => {
    const { dir, store } = tempStore();
    const sources = new ModelSourceRepository(
      new (require('../src/data/cluster-settings-repository').ClusterSettingsRepository)(store),
    );
    const budget = {
      evaluate: () => ({
        total_bytes: 11,
        existing_bytes: 0,
        disk_required_bytes: 11,
        disk_available_bytes: 10,
        sufficient: false,
      }),
    };
    const preflight = new PullPreflightService(
      resolverFixture(), budget as any, new ArtifactStore(join(dir, 'store')),
      securityDeps(dir, store).credentials, securityDeps(dir, store).licenses,
    );
    const result = await preflight.resolve({
      source: sources.preferred('huggingface')!,
      repoId: 'org/model',
    });
    expect(result.status).toBe('insufficient_storage');
    expect(result.disk_required_bytes).toBe(11);
    expect(store.prepare('SELECT COUNT(*) AS c FROM pull_jobs').get()).toEqual({ c: 0 });
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('MODEL-FLEET sources API (MF-N4)', () => {
  it('lists, switches, resets sources and resolves without storing a token', async () => {
    const { dir, store } = tempStore();
    const artifactStore = new ArtifactStore(join(dir, 'model-store'));
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(store)
      .overrideProvider(ArtifactStore).useValue(artifactStore)
      .overrideProvider(ConfigDao).useValue({
        enabled: false, dbEnabled: () => false,
        getConnectionInfo: () => ({}), ping: async () => ({ ok: true }),
      })
      .overrideProvider(HfResolver).useValue(resolverFixture())
      .overrideProvider(ModelDiskBudget).useValue({
        evaluate: () => ({
          total_bytes: 11, existing_bytes: 0, disk_required_bytes: 11,
          disk_available_bytes: 100, sufficient: true,
        }),
      })
      .compile();
    const app: any = moduleRef.createNestApplication(new FastifyAdapter());
    await app.init();
    try {
      const saved = await app.inject({
        method: 'PUT', url: '/models/sources/team',
        payload: {
          name: 'Team', provider: 'huggingface', endpoint: 'https://mirror.example',
          credential_ref: 'os:team', priority: 1, enabled: true,
          token: 'plain-secret',
        },
      });
      expect(saved.statusCode).toBe(422);
      const valid = await app.inject({
        method: 'PUT', url: '/models/sources/team',
        payload: {
          name: 'Team', provider: 'huggingface', endpoint: 'https://mirror.example',
          credential_ref: 'os:team', priority: 1, enabled: true,
        },
      });
      expect(valid.statusCode).toBe(200);
      const listed = await app.inject({ method: 'GET', url: '/models/sources' });
      expect(listed.json().sources[0].source_id).toBe('team');
      const resolved = await app.inject({
        method: 'POST', url: '/models/resolve',
        payload: { source_id: 'team', repo_id: 'org/model', requested_revision: 'main' },
      });
      expect(resolved.statusCode).toBe(200);
      expect(resolved.json().source.endpoint).toBe('https://mirror.example');
      expect(resolved.json().resolved_revision).toBe('b'.repeat(40));
      const badToken = await app.inject({
        method: 'POST', url: '/models/resolve',
        payload: { source_id: 'team', repo_id: 'org/model', token: 'plain-secret' },
      });
      expect(badToken.statusCode).toBe(422);
      const deleted = await app.inject({
        method: 'DELETE', url: '/models/sources/team',
      });
      expect(deleted.statusCode).toBe(200);
      const reset = await app.inject({ method: 'POST', url: '/models/sources/reset' });
      expect(reset.statusCode).toBe(200);
    } finally {
      await app.close();
      store.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
