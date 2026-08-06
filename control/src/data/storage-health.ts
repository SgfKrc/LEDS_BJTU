/**
 * M1 存储健康 — 本地/远端双健康结构（一键模型部署计划 §12.3）。
 *
 * effective_mode 语义：
 *  - local_primary：本地可写、远端在线且 outbox 无积压；
 *  - local_primary_pending：本地可写、远端在线但 outbox 有积压（或远端刚恢复）；
 *  - readonly_failure：本地 SQLite 不可写——控制面只读故障，禁止假成功。
 *  - local_only：本地可写、远端未配置/不可达（远端为可选投影）。
 */
import { Injectable } from '@nestjs/common';
import { SqliteStore, LocalStorageHealth } from './sqlite-store';
import { ConfigDao } from './config-dao';
import { OutboxService } from './outbox.service';

export type EffectiveMode =
  | 'local_primary'
  | 'local_primary_pending'
  | 'local_only'
  | 'readonly_failure';

export interface RemoteStorageHealth {
  status: 'ok' | 'unavailable' | 'not_configured';
  backend: 'postgresql';
  host?: string;
  port?: number;
  db?: string;
  error?: string;
}

export interface ProjectionHealth {
  pending_events: number;
  oldest_event_age_seconds: number | null;
}

export interface StorageHealth {
  local: LocalStorageHealth;
  remote: RemoteStorageHealth;
  projection: ProjectionHealth;
  effective_mode: EffectiveMode;
}

@Injectable()
export class StorageHealthService {
  constructor(
    private readonly store: SqliteStore,
    private readonly configDao: ConfigDao,
    private readonly outbox: OutboxService,
  ) {}

  async snapshot(): Promise<StorageHealth> {
    const local = this.localHealth();
    const remote = await this.remoteHealth();
    const projection = this.projectionHealth();
    const effectiveMode = this.computeMode(local, remote, projection);
    return {
      local,
      remote,
      projection,
      effective_mode: effectiveMode,
    };
  }

  localHealth(): LocalStorageHealth {
    return this.store.health();
  }

  async remoteHealth(): Promise<RemoteStorageHealth> {
    if (!this.configDao.dbEnabled()) {
      return { status: 'not_configured', backend: 'postgresql' };
    }
    const { ok, error } = await this.configDao.ping();
    if (!ok) {
      return {
        status: 'unavailable',
        backend: 'postgresql',
        error: error || 'postgresql 连接失败',
      };
    }
    const info = this.configDao.getConnectionInfo();
    return { status: 'ok', backend: 'postgresql', ...info };
  }

  projectionHealth(): ProjectionHealth {
    const pending = this.outbox.pendingCount();
    let oldest: number | null = null;
    if (pending > 0) {
      oldest = this.outbox.oldestPendingAgeSeconds();
    }
    return { pending_events: pending, oldest_event_age_seconds: oldest };
  }

  computeMode(
    local: LocalStorageHealth,
    remote: RemoteStorageHealth,
    projection: ProjectionHealth,
  ): EffectiveMode {
    if (!local.writable) return 'readonly_failure';
    if (remote.status === 'ok') {
      return projection.pending_events > 0
        ? 'local_primary_pending'
        : 'local_primary';
    }
    return 'local_only';
  }
}
