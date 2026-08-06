/**
 * M1 model registry repository — 用户模型注册表（SQLite 事实源）。
 * 幂等 upsert：model_id 主键；同一 sha256 只注册一次（迁移去重规则）。
 */
import { Injectable } from '@nestjs/common';
import { SqliteStore } from './sqlite-store';

export interface RegisteredModelRow {
  model_id: string;
  name: string;
  model_path: string;
  gguf_path: string | null;
  quantization: string | null;
  sha256: string;
  created_at: string;
}

export interface RegisteredModelInput {
  model_id: string;
  name: string;
  model_path: string;
  gguf_path?: string | null;
  quantization?: string | null;
  sha256: string;
}

@Injectable()
export class ModelRegistryRepository {
  constructor(private readonly store: SqliteStore) {}

  get(modelId: string): RegisteredModelRow | null {
    const row = this.store.prepare(
      `SELECT model_id, name, model_path, gguf_path, quantization, sha256, created_at
       FROM model_registry WHERE model_id = ?`,
    ).get(modelId) as RegisteredModelRow | undefined;
    return row ?? null;
  }

  list(): RegisteredModelRow[] {
    return this.store.prepare(
      `SELECT model_id, name, model_path, gguf_path, quantization, sha256, created_at
       FROM model_registry ORDER BY model_id`,
    ).all() as unknown as RegisteredModelRow[];
  }

  /** 幂等 upsert；同 sha256 已存在时返回既有行（不产生重复 artifact）。 */
  upsert(entry: RegisteredModelInput): RegisteredModelRow {
    const now = new Date().toISOString();
    this.store.prepare(
      `INSERT INTO model_registry
         (model_id, name, model_path, gguf_path, quantization, sha256, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(model_id) DO UPDATE SET
         name = excluded.name,
         model_path = excluded.model_path,
         gguf_path = excluded.gguf_path,
         quantization = excluded.quantization,
         sha256 = excluded.sha256`,
    ).run(
      entry.model_id, entry.name, entry.model_path,
      entry.gguf_path ?? null, entry.quantization ?? null,
      entry.sha256, now,
    );
    return {
      model_id: entry.model_id,
      name: entry.name,
      model_path: entry.model_path,
      gguf_path: entry.gguf_path ?? null,
      quantization: entry.quantization ?? null,
      sha256: entry.sha256,
      created_at: now,
    };
  }

  /** 摘要去重：sha256 是否已存在于其他 model_id。 */
  existsDigest(sha256: string, excludeModelId?: string): boolean {
    const row = this.store.prepare(
      'SELECT COUNT(*) AS c FROM model_registry WHERE sha256 = ? AND model_id != ?',
    ).get(sha256, excludeModelId ?? '') as { c: number };
    return Number(row.c) > 0;
  }
}
