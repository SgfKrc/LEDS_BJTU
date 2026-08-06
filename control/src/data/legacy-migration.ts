/**
 * M1 旧源一次性迁移执行器（对齐 schemas/migration-map.json）。
 *
 * 语义：
 *  - 只读源文件（Python 导出 seed / control-svc 旧 model_registry.json）；
 *  - 幂等可重复执行：主键冲突跳过，同 sha256 不重复注册；
 *  - cluster_config（PostgreSQL）可选：pg 不可用时跳过并记录 skipped，
 *    不阻塞本地迁移（本地事实源先行）。
 */
import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import { SqliteStore } from './sqlite-store';
import { ClusterSettingsRepository } from './cluster-settings-repository';
import { ModelRegistryRepository } from './model-registry-repository';

export interface LegacyMigrationSources {
  /** scripts/export_model_catalog.py 的输出（catalog_models seed）。 */
  catalogSeedPath: string;
  /** control-svc 旧 model_registry.json（可为空路径=跳过）。 */
  registryJsonPath?: string;
  /** PostgreSQL cluster_config 读取器（pg 不可用时返回 []）。 */
  clusterSettingsLoader?: () => Promise<Array<{ key: string; value: string }>>;
}

export interface LegacyMigrationResult {
  catalog_imported: number;
  registry_imported: number;
  registry_skipped_digest: number;
  settings_imported: number;
  settings_skipped: number;
}

@Injectable()
export class LegacyMigration {
  constructor(
    private readonly store: SqliteStore,
    private readonly settings: ClusterSettingsRepository,
    private readonly registry: ModelRegistryRepository,
  ) {}

  /** 幂等迁移：重复执行不产生重复 artifact。 */
  async run(sources: LegacyMigrationSources): Promise<LegacyMigrationResult> {
    const result: LegacyMigrationResult = {
      catalog_imported: 0,
      registry_imported: 0,
      registry_skipped_digest: 0,
      settings_imported: 0,
      settings_skipped: 0,
    };

    // 1) curated catalog seed（Python 导出）
    if (fs.existsSync(sources.catalogSeedPath)) {
      const seed = JSON.parse(
        fs.readFileSync(sources.catalogSeedPath, 'utf-8'),
      ) as Array<Record<string, unknown>>;
      for (const row of seed) {
        const modelId = String(row.model_id ?? '');
        if (!modelId) continue;
        this.store.prepare(
          `INSERT INTO catalog_models (model_id, payload, created_at)
           VALUES (?, ?, ?)
           ON CONFLICT(model_id) DO NOTHING`,
        ).run(modelId, JSON.stringify(row), new Date().toISOString());
        result.catalog_imported += 1;
      }
    }

    // 2) 用户注册表（旧 JSON；有摘要去重，无摘要以 NULL 入库待 M2 补摘要）
    if (sources.registryJsonPath && fs.existsSync(sources.registryJsonPath)) {
      const rows = JSON.parse(
        fs.readFileSync(sources.registryJsonPath, 'utf-8'),
      ) as Array<Record<string, unknown>>;
      for (const row of rows) {
        const modelId = String(row.model_id ?? '');
        const digest = String(row.sha256 ?? row.file_sha256 ?? '');
        if (!modelId) continue;
        if (digest && this.registry.existsDigest(digest, modelId)) {
          result.registry_skipped_digest += 1;
          continue; // 同摘要只注册一次
        }
        this.registry.upsert({
          model_id: modelId,
          name: String(row.name ?? modelId),
          model_type: String(row.model_type ?? 'safetensors'),
          model_path: String(row.model_path ?? ''),
          gguf_path: String(row.gguf_path ?? ''),
          recommended_vram_gb: Number(row.recommended_vram_gb) || 8.0,
          max_context: Number(row.max_context) || 4096,
          huggingface_id: String(row.huggingface_id ?? ''),
          description: String(row.description ?? ''),
          quant_types: Array.isArray(row.quant_types)
            ? row.quant_types.map(String)
            : [],
          sha256: digest || null,
        });
        result.registry_imported += 1;
      }
    }

    // 3) cluster settings（pg 可选；不可用时跳过，本地事实源先行）
    if (sources.clusterSettingsLoader) {
      try {
        const entries = await sources.clusterSettingsLoader();
        for (const entry of entries) {
          this.settings.set(entry.key, entry.value);
          result.settings_imported += 1;
        }
      } catch {
        result.settings_skipped = 1;
      }
    } else {
      result.settings_skipped = 1;
    }

    return result;
  }
}
