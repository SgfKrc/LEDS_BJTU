/**
 * M1 cluster settings repository — SQLite 事实源（唯一写者）。
 * 幂等 upsert（key 主键）；读路径供 controller/迁移使用。
 */
import { Injectable } from '@nestjs/common';
import { SqliteStore } from './sqlite-store';

export interface ClusterSetting {
  key: string;
  value: string;
  updated_at: string;
}

@Injectable()
export class ClusterSettingsRepository {
  constructor(private readonly store: SqliteStore) {}

  /** 单事务执行（业务行 + outbox 同事务）。 */
  transaction<T>(fn: () => T): T {
    return this.store.transaction(fn);
  }

  get(key: string): ClusterSetting | null {
    const row = this.store.prepare(
      'SELECT key, value, updated_at FROM cluster_settings WHERE key = ?',
    ).get(key) as ClusterSetting | undefined;
    return row ?? null;
  }

  list(): ClusterSetting[] {
    return this.store.prepare(
      'SELECT key, value, updated_at FROM cluster_settings ORDER BY key',
    ).all() as unknown as ClusterSetting[];
  }

  /** 幂等 upsert（迁移与运行共用）。 */
  set(key: string, value: string): ClusterSetting {
    const now = new Date().toISOString();
    this.store.prepare(
      `INSERT INTO cluster_settings (key, value, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at`,
    ).run(key, value, now);
    return { key, value, updated_at: now };
  }
}
