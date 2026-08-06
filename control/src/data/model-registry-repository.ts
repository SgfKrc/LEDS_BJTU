/**
 * M1 model registry repository — 用户模型注册表（SQLite 事实源）。
 *
 * payload 列保存完整注册条目 JSON（对齐旧 ModelRegistryStore.RegisteredModel
 * 结构，向前兼容）；sha256 列是内容摘要去重键（旧数据可为 NULL，去重仅对
 * 非空摘要生效；正式摘要由 M2 inspector 补齐）。
 */
import { Injectable } from '@nestjs/common';
import { SqliteStore } from './sqlite-store';

/** 注册条目（兼容旧 ModelRegistryStore.RegisteredModel 全部字段）。 */
export interface ModelRegistryEntry {
  model_id: string;
  name: string;
  model_type: string;
  model_path: string;
  gguf_path: string;
  recommended_vram_gb: number;
  max_context: number;
  huggingface_id: string;
  description: string;
  quant_types: string[];
  /** 内容摘要（可空：旧数据/未校验工件为 NULL，部署前由 inspector 补齐）。 */
  sha256?: string | null;
}

export interface ModelRegistryRow {
  model_id: string;
  name: string;
  model_path: string;
  gguf_path: string | null;
  quantization: string | null;
  sha256: string | null;
  payload: string;
  created_at: string;
}

@Injectable()
export class ModelRegistryRepository {
  constructor(private readonly store: SqliteStore) {}

  get(modelId: string): ModelRegistryEntry | null {
    const row = this.store.prepare(
      'SELECT payload FROM model_registry WHERE model_id = ?',
    ).get(modelId) as { payload: string } | undefined;
    if (!row) return null;
    return JSON.parse(row.payload) as ModelRegistryEntry;
  }

  list(): ModelRegistryEntry[] {
    const rows = this.store.prepare(
      'SELECT payload FROM model_registry ORDER BY model_id',
    ).all() as unknown as Array<{ payload: string }>;
    return rows.map((row) => JSON.parse(row.payload) as ModelRegistryEntry);
  }

  /** 幂等 upsert；payload 保存完整条目。 */
  upsert(entry: ModelRegistryEntry): void {
    const now = new Date().toISOString();
    const digest = entry.sha256 ?? null;
    this.store.prepare(
      `INSERT INTO model_registry
         (model_id, name, model_path, gguf_path, quantization, sha256, payload, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(model_id) DO UPDATE SET
         name = excluded.name,
         model_path = excluded.model_path,
         gguf_path = excluded.gguf_path,
         quantization = excluded.quantization,
         sha256 = excluded.sha256,
         payload = excluded.payload`,
    ).run(
      entry.model_id,
      entry.name,
      entry.model_path,
      entry.gguf_path || null,
      entry.quant_types?.[0] ?? null,
      digest,
      JSON.stringify(entry),
      now,
    );
  }

  delete(modelId: string): boolean {
    const result = this.store.prepare(
      'DELETE FROM model_registry WHERE model_id = ?',
    ).run(modelId);
    return Number(result.changes) > 0;
  }

  /** 摘要去重：仅对非空摘要生效（NULL 摘要不参与去重）。 */
  existsDigest(sha256: string, excludeModelId?: string): boolean {
    const row = this.store.prepare(
      'SELECT COUNT(*) AS c FROM model_registry '
      + 'WHERE sha256 = ? AND model_id != ?',
    ).get(sha256, excludeModelId ?? '') as { c: number };
    return Number(row.c) > 0;
  }
}
