/**
 * M1 任务 2 测试：三域 repository CRUD 与旧源迁移执行器（幂等/去重）。
 */
import { mkdtempSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { SqliteStore } from '../src/data/sqlite-store';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ModelRegistryRepository } from '../src/data/model-registry-repository';
import { ClusterEndpointsRepository } from '../src/data/cluster-endpoints-repository';
import { LegacyMigration } from '../src/data/legacy-migration';

function tempStore(): SqliteStore {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-m1r-'));
  return new SqliteStore(join(dir, 'control.sqlite3'));
}

describe('ClusterSettingsRepository', () => {
  it('upserts idempotently and lists sorted', () => {
    const store = tempStore();
    store.open();
    const repo = new ClusterSettingsRepository(store);
    repo.set('max_nodes', '3');
    repo.set('max_nodes', '5'); // 幂等覆盖
    repo.set('distributed_inference', 'true');
    expect(repo.get('max_nodes')?.value).toBe('5');
    expect(repo.list().map((s) => s.key)).toEqual([
      'distributed_inference', 'max_nodes',
    ]);
    store.close();
  });
});

describe('ModelRegistryRepository', () => {
  it('upserts by model_id and dedupes by digest', () => {
    const store = tempStore();
    store.open();
    const repo = new ModelRegistryRepository(store);
    const digestA = 'a'.repeat(64);
    const digestB = 'b'.repeat(64);
    repo.upsert({
      model_id: 'm1', name: 'M1', model_type: 'safetensors',
      model_path: '/x', gguf_path: '', recommended_vram_gb: 8,
      max_context: 4096, huggingface_id: '', description: '',
      quant_types: ['fp16'], sha256: digestA,
    });
    repo.upsert({
      model_id: 'm2', name: 'M2', model_type: 'gguf',
      model_path: '/y', gguf_path: '', recommended_vram_gb: 8,
      max_context: 4096, huggingface_id: '', description: '',
      quant_types: ['Q4_K_M'], sha256: digestB,
    });
    // 同摘要的新 model_id 应被标记为重复
    expect(repo.existsDigest(digestA, 'm3')).toBe(true);
    expect(repo.existsDigest(digestA, 'm1')).toBe(false); // 排除自身
    // 覆盖更新
    repo.upsert({
      model_id: 'm1', name: 'M1v2', model_type: 'safetensors',
      model_path: '/x2', gguf_path: '', recommended_vram_gb: 8,
      max_context: 4096, huggingface_id: '', description: '',
      quant_types: ['fp16'], sha256: digestA,
    });
    expect(repo.get('m1')?.name).toBe('M1v2');
    expect(repo.list().length).toBe(2);
    expect(repo.delete('m2')).toBe(true);
    expect(repo.delete('m2')).toBe(false);
    store.close();
  });
});

describe('ClusterEndpointsRepository', () => {
  it('upserts endpoint and preserves cluster_id', () => {
    const store = tempStore();
    store.open();
    const repo = new ClusterEndpointsRepository(store);
    repo.upsert({
      endpoint_id: 'ep1', cluster_id: 'cluster_a', name: '家里',
      scheme: 'http', host: '100.64.0.1', port: 8888, status: 'active',
    });
    repo.upsert({
      endpoint_id: 'ep1', cluster_id: 'cluster_a', name: '家里（新地址）',
      scheme: 'http', host: '100.64.0.2', port: 8888, status: 'active',
    });
    expect(repo.get('ep1')?.host).toBe('100.64.0.2');
    expect(repo.get('ep1')?.cluster_id).toBe('cluster_a');
    expect(repo.list().length).toBe(1);
    store.close();
  });
});

describe('LegacyMigration', () => {
  function tempDir(): string {
    return mkdtempSync(join(tmpdir(), 'qlh-mig-'));
  }

  it('imports catalog seed and registry with digest dedup, idempotently', async () => {
    const dir = tempDir();
    try {
      const seedPath = join(dir, 'catalog-seed.json');
      writeFileSync(seedPath, JSON.stringify([
        { model_id: 'qwen-1_8b', name: 'Qwen-1.8B-Chat', format: 'both',
          local_path: '/models/qwen-1_8b-chat', gguf_path: '/g.gguf',
          context_length: 4096, source: { repo_id: 'Qwen/Qwen-1.8B-Chat' },
          quantizations: ['fp16', 'int4'], origin: 'bundled' },
        { model_id: 'qwen2.5-7b', name: 'Qwen2.5-7B', format: 'safetensors',
          local_path: '/models/qwen2.5-7b', context_length: 32768,
          source: { repo_id: 'Qwen/Qwen2.5-7B-Instruct' },
          quantizations: ['int4'], origin: 'external' },
      ]), 'utf-8');

      const digest = 'd'.repeat(64);
      const registryPath = join(dir, 'model_registry.json');
      writeFileSync(registryPath, JSON.stringify([
        { model_id: 'user-model', name: 'U', model_path: '/u', sha256: digest },
        { model_id: 'user-model-copy', name: 'U2', model_path: '/u2', sha256: digest },
        { model_id: 'no-digest', name: 'X', model_path: '/x' }, // 无摘要 → NULL 入库
      ]), 'utf-8');

      const store = tempStore();
      store.open();
      const migration = new LegacyMigration(
        store,
        new ClusterSettingsRepository(store),
        new ModelRegistryRepository(store),
      );

      const first = await migration.run({
        catalogSeedPath: seedPath,
        registryJsonPath: registryPath,
      });
      expect(first.catalog_imported).toBe(2);
      expect(first.registry_imported).toBe(2);      // user-model + no-digest（NULL 摘要）
      expect(first.registry_skipped_digest).toBe(1); // user-model-copy 同摘要
      expect(first.settings_skipped).toBe(1);        // 无 loader

      // 幂等：重复执行不产生重复行
      const second = await migration.run({
        catalogSeedPath: seedPath,
        registryJsonPath: registryPath,
      });
      const catalogCount = (store.prepare(
        'SELECT COUNT(*) AS c FROM catalog_models',
      ).get() as { c: number }).c;
      const registryCount = (store.prepare(
        'SELECT COUNT(*) AS c FROM model_registry',
      ).get() as { c: number }).c;
      expect(catalogCount).toBe(2);
      expect(registryCount).toBe(2);
      expect(second.registry_skipped_digest).toBe(1);
      store.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('imports cluster settings via loader and skips on loader failure', async () => {
    const store = tempStore();
    store.open();
    const migration = new LegacyMigration(
      store,
      new ClusterSettingsRepository(store),
      new ModelRegistryRepository(store),
    );
    const result = await migration.run({
      catalogSeedPath: '/nonexistent-seed.json',
      clusterSettingsLoader: async () => [
        { key: 'max_nodes', value: '3' },
        { key: 'distributed_inference', value: 'true' },
      ],
    });
    expect(result.settings_imported).toBe(2);
    expect(result.settings_skipped).toBe(0);
    expect(new ClusterSettingsRepository(store).get('max_nodes')?.value).toBe('3');

    const failed = await migration.run({
      catalogSeedPath: '/nonexistent-seed.json',
      clusterSettingsLoader: async () => {
        throw new Error('pg down');
      },
    });
    expect(failed.settings_skipped).toBe(1);
    store.close();
  });

  it('real catalog seed exports validate against migration-map field names', () => {
    // 真实导出文件（scripts/export_model_catalog.py 生成）字段对齐映射清单
    const { existsSync } = require('fs') as typeof import('fs');
    const { join: joinPath } = require('path') as typeof import('path');
    const seedPath = joinPath(__dirname, '..', '..', 'build', 'model-fleet', 'catalog-seed.json');
    if (!existsSync(seedPath)) return; // 未导出时跳过（导出脚本单独验证）
    const seed = JSON.parse(
      (require('fs') as typeof import('fs')).readFileSync(seedPath, 'utf-8'),
    ) as Array<Record<string, unknown>>;
    expect(seed.length).toBeGreaterThanOrEqual(1);
    for (const row of seed) {
      expect(row).toHaveProperty('model_id');
      expect(row).toHaveProperty('name');
      expect(row).toHaveProperty('format');
      expect(row).toHaveProperty('local_path');
      expect(row).toHaveProperty('context_length');
      expect(row).toHaveProperty('origin');
    }
  });
});
