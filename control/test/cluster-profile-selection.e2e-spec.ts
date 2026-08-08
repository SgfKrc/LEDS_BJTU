import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { Test } from '@nestjs/testing';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import { AppModule } from '../src/app';
import { ClusterDiscoveryService } from '../src/data/cluster-discovery.service';
import { ClusterProfileRepository } from '../src/data/cluster-profile-repository';
import { ClusterProfileSelectionService } from '../src/data/cluster-profile-selection';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ConfigDao } from '../src/data/config-dao';
import { SqliteStore } from '../src/data/sqlite-store';

describe('MODEL-FLEET profile selection and discovery (MF-N5)', () => {
  it('orders active profile before env and Tailscale candidates without duplicates', () => {
    const discovery = new ClusterDiscoveryService();
    const candidates = discovery.candidates({
      current: {
        profile_id: 'p1', cluster_id: 'c1', name: 'A',
        master_endpoint: 'http://100.64.0.1:8000', status: 'active',
        key_ref: 'os:c1', node_role: 'client', last_verified_at: null,
        created_at: new Date().toISOString(),
      },
      envHost: 'http://100.64.0.2:8000',
      tailscalePeers: [
        'http://100.64.0.3:8000',
        'http://100.64.0.1:8000',
      ],
    });
    expect(candidates.map((candidate) => candidate.endpoint)).toEqual([
      'http://100.64.0.1:8000',
      'http://100.64.0.2:8000',
      'http://100.64.0.3:8000',
    ]);
    expect(candidates.map((candidate) => candidate.priority)).toEqual([10, 20, 30]);
  });

  it('switches and clears the selected profile while keeping two clusters isolated', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-profile-select-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const profiles = new ClusterProfileRepository(store);
    const settings = new ClusterSettingsRepository(store);
    const selection = new ClusterProfileSelectionService(profiles, settings);
    const first = profiles.create({
      cluster_id: 'cluster-a', name: 'A', master_endpoint: 'http://100.64.0.1:8000',
    });
    const second = profiles.create({
      cluster_id: 'cluster-b', name: 'B', master_endpoint: 'http://100.64.0.2:8000',
    });
    expect(selection.activate(first.profile_id).cluster_id).toBe('cluster-a');
    expect(selection.activate(second.profile_id).cluster_id).toBe('cluster-b');
    expect(selection.current()?.cluster_id).toBe('cluster-b');
    selection.clearIf(first.profile_id);
    expect(selection.current()?.cluster_id).toBe('cluster-b');
    selection.clearIf(second.profile_id);
    expect(selection.current()).toBeNull();
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('exposes schema-shaped profile DTOs and discovery priority through the API', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-profile-api-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(store)
      .overrideProvider(ConfigDao).useValue({
        enabled: false, dbEnabled: () => false,
        getConnectionInfo: () => ({}), ping: async () => ({ ok: true }),
      })
      .compile();
    const app: any = moduleRef.createNestApplication(new FastifyAdapter());
    const priorMaster = process.env.QLH_MASTER_HOST;
    process.env.QLH_MASTER_HOST = 'http://100.64.0.99:8000';
    await app.init();
    try {
      const created = await app.inject({
        method: 'POST', url: '/cluster/profiles',
        payload: {
          cluster_id: 'cluster-a', name: 'A',
          master_endpoint: 'http://100.64.0.1:8000',
        },
      });
      expect(created.statusCode).toBe(201);
      expect(created.json().profile.master_endpoint).toEqual({
        scheme: 'http', host: '100.64.0.1', port: 8000,
      });
      const second = await app.inject({
        method: 'POST', url: '/cluster/profiles',
        payload: {
          cluster_id: 'cluster-b', name: 'B',
          master_endpoint: 'http://100.64.0.2:8000',
        },
      });
      const secondId = second.json().profile.profile_id;
      const activated = await app.inject({
        method: 'POST', url: `/cluster/profiles/${secondId}/activate`,
      });
      expect(activated.json().profile.cluster_id).toBe('cluster-b');
      const current = await app.inject({ method: 'GET', url: '/cluster/profiles/current' });
      expect(current.json().profile.cluster_id).toBe('cluster-b');
      const discovery = await app.inject({
        method: 'GET',
        url: '/cluster/discovery/candidates?tailscale_peers=http://100.64.0.3:8000,http://100.64.0.2:8000',
      });
      expect(discovery.json().candidates.map((candidate: { endpoint: string }) => candidate.endpoint))
        .toEqual([
          'http://100.64.0.2:8000',
          'http://100.64.0.99:8000',
          'http://100.64.0.3:8000',
        ]);
      const deleted = await app.inject({
        method: 'DELETE', url: `/cluster/profiles/${secondId}`,
      });
      expect(deleted.statusCode).toBe(200);
      expect((await app.inject({ method: 'GET', url: '/cluster/profiles/current' })).json().profile)
        .toBeNull();
    } finally {
      await app.close();
      if (priorMaster === undefined) delete process.env.QLH_MASTER_HOST;
      else process.env.QLH_MASTER_HOST = priorMaster;
      store.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
