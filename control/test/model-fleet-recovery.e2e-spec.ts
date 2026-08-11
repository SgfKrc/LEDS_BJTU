import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ClusterProfileRepository } from '../src/data/cluster-profile-repository';
import { ModelRegistryEntry, ModelRegistryRepository } from '../src/data/model-registry-repository';
import { OutboxService } from '../src/data/outbox.service';
import { PullJobService } from '../src/data/pull-job.service';
import { SqliteStore } from '../src/data/sqlite-store';

function model(modelId: string): ModelRegistryEntry {
  return {
    model_id: modelId,
    name: modelId,
    model_type: 'safetensors',
    model_path: `/models/${modelId}`,
    gguf_path: '',
    recommended_vram_gb: 8,
    max_context: 8192,
    huggingface_id: `local/${modelId}`,
    description: 'recovery fixture',
    quant_types: ['fp16'],
    sha256: modelId.padEnd(64, '0').slice(0, 64),
  };
}

describe('MODEL-FLEET local recovery (MF-N2)', () => {
  it('rolls business data and outbox back as one transaction', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-recovery-tx-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const registry = new ModelRegistryRepository(store);
    const outbox = new OutboxService(store);

    expect(() => store.transaction(() => {
      registry.upsert(model('rolled-back'));
      outbox.enqueue('model_registry', 'created', { model_id: 'rolled-back' });
      throw new Error('fault injection after local writes');
    })).toThrow('fault injection');

    expect(registry.get('rolled-back')).toBeNull();
    expect(outbox.pendingCount()).toBe(0);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('persists facts and legacy backlog across restart without remote projection', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-recovery-local-'));
    const sqlitePath = join(dir, 'control.sqlite3');
    let store = new SqliteStore(sqlitePath);
    store.open();

    const registry = new ModelRegistryRepository(store);
    const profiles = new ClusterProfileRepository(store);
    const pulls = new PullJobService(store);
    const outbox = new OutboxService(store);
    registry.upsert(model('model-before-restart'));
    profiles.create({
      cluster_id: 'cluster-a',
      name: 'Cluster A',
      master_endpoint: 'https://master-a.invalid',
    });
    const pull = pulls.create({
      idempotencyKey: 'recovery-pull',
      source: {
        provider: 'huggingface',
        repo_id: 'org/model',
        requested_revision: 'main',
      },
    });
    outbox.enqueue('model_registry', 'created', { model_id: 'model-before-restart' });
    store.close();

    store = new SqliteStore(sqlitePath);
    store.open();
    expect(new ModelRegistryRepository(store).get('model-before-restart')).not.toBeNull();
    expect(new ClusterProfileRepository(store).getByCluster('cluster-a')).not.toBeNull();
    expect(new PullJobService(store).get(pull.job_id)?.state).toBe('queued');
    expect(new PullJobService(store).listActive()).toHaveLength(1);

    let recoveredOutbox = new OutboxService(store);
    expect(recoveredOutbox.pendingCount()).toBe(1);

    new ModelRegistryRepository(store).upsert(model('model-after-restart'));
    recoveredOutbox.enqueue('model_registry', 'created', { model_id: 'model-after-restart' });
    store.close();

    store = new SqliteStore(sqlitePath);
    store.open();
    recoveredOutbox = new OutboxService(store);
    expect(recoveredOutbox.pendingCount()).toBe(2);
    expect(new ModelRegistryRepository(store).get('model-after-restart')).not.toBeNull();

    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('rejects stale explicit aggregate versions', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-recovery-version-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const outbox = new OutboxService(store);
    outbox.enqueue('deployments', 'prepared', { deployment_id: 'd1' }, 7);
    expect(() => outbox.enqueue(
      'deployments', 'prepared', { deployment_id: 'stale' }, 6,
    )).toThrow('aggregate_version');
    expect(outbox.pending().map((event) => event.aggregate_version)).toEqual([7]);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});
