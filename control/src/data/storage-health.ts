/** M1.3 存储健康：主节点 SQLite 是唯一生产事实源。 */
import { Injectable } from '@nestjs/common';
import { SqliteStore, LocalStorageHealth } from './sqlite-store';

export type EffectiveMode = 'local_only' | 'readonly_failure';

export interface RemoteStorageHealth {
  status: 'retired' | 'disabled';
  backend: 'postgresql';
  mode: 'retired' | 'legacy_cleanup_pending';
}

export interface ProjectionHealth {
  pending_events: number;
  oldest_event_age_seconds: number | null;
}

export interface LegacyExportHealth {
  pending_items: number;
  oldest_item_age_seconds: number | null;
}

export interface StorageHealth {
  local: LocalStorageHealth;
  remote: RemoteStorageHealth;
  projection: ProjectionHealth;
  export: LegacyExportHealth;
  effective_mode: EffectiveMode;
  retirement: {
    status: 'not_prepared' | 'prepared' | 'retired';
    prepared_at: string | null;
    retired_at: string | null;
  };
}

@Injectable()
export class StorageHealthService {
  constructor(private readonly store: SqliteStore) {}

  async snapshot(): Promise<StorageHealth> {
    const local = this.localHealth();
    const retirement = this.retirementHealth();
    const remote = this.remoteHealth(retirement.status);
    const projection = this.projectionHealth();
    return {
      local,
      remote,
      projection,
      export: {
        pending_items: projection.pending_events,
        oldest_item_age_seconds: projection.oldest_event_age_seconds,
      },
      effective_mode: local.writable ? 'local_only' : 'readonly_failure',
      retirement,
    };
  }

  localHealth(): LocalStorageHealth {
    return this.store.health();
  }

  remoteHealth(status: 'not_prepared' | 'prepared' | 'retired'): RemoteStorageHealth {
    return status === 'retired'
      ? { status: 'retired', backend: 'postgresql', mode: 'retired' }
      : { status: 'disabled', backend: 'postgresql', mode: 'legacy_cleanup_pending' };
  }

  projectionHealth(): ProjectionHealth {
    const exists = this.store.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='outbox'",
    ).get() as { name: string } | undefined;
    if (!exists) return { pending_events: 0, oldest_event_age_seconds: null };
    const row = this.store.prepare(
      `SELECT COUNT(*) AS pending, MIN(created_at) AS oldest
       FROM outbox WHERE projected_at IS NULL`,
    ).get() as { pending: number; oldest: string | null };
    const pending = Number(row.pending);
    const oldest = row.oldest
      ? Math.max(0, Math.floor((Date.now() - new Date(row.oldest).getTime()) / 1000))
      : null;
    return { pending_events: pending, oldest_event_age_seconds: oldest };
  }

  retirementHealth(): StorageHealth['retirement'] {
    const exists = this.store.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='storage_retirement'",
    ).get() as { name: string } | undefined;
    if (!exists) return { status: 'not_prepared', prepared_at: null, retired_at: null };
    const row = this.store.prepare(
      `SELECT status, prepared_at, retired_at FROM storage_retirement
       WHERE retirement_id = 1`,
    ).get() as {
      status: 'prepared' | 'retired';
      prepared_at: string;
      retired_at: string | null;
    } | undefined;
    return row ?? { status: 'not_prepared', prepared_at: null, retired_at: null };
  }
}
