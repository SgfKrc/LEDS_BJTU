/**
 * M1 PostgreSQL projector — 带退避的后台投影器（远端连接探测移出请求关键路径）。
 *
 * 设计（一键模型部署计划 §16 M1 任务 5）：
 *  - setInterval 后台轮询；pg 不可用时长退避（×2，上限 60s），恢复后重置；
 *  - 幂等投影：远端 upsert 以 event_id 去重（on conflict do nothing）、
 *    aggregate_version 拒绝旧事件覆盖新状态；
 *  - 远端不可达时本地事务继续，UI 不阻塞（只读 storage/health 反映积压）。
 */
import { Injectable, Optional } from '@nestjs/common';
import { Client } from 'pg';
import { ConfigDao } from './config-dao';
import { OutboxService } from './outbox.service';

export interface ProjectorOptions {
  /** 初始轮询间隔（毫秒）。 */
  baseIntervalMs?: number;
  /** 失败退避上限（毫秒）。 */
  maxIntervalMs?: number;
  /** 测试注入：自定义 pg Client 工厂（缺省 new Client）。 */
  clientFactory?: () => Client;
}

const DEFAULT_OPTIONS = {
  baseIntervalMs: 5_000,
  maxIntervalMs: 60_000,
} as const;

@Injectable()
export class PostgresProjector {
  private timer: NodeJS.Timeout | null = null;
  private currentIntervalMs: number;
  private running = false;

  constructor(
    private readonly configDao: ConfigDao,
    private readonly outbox: OutboxService,
    @Optional() private readonly options: ProjectorOptions = {},
  ) {
    this.currentIntervalMs =
      options.baseIntervalMs ?? DEFAULT_OPTIONS.baseIntervalMs;
  }

  get intervalMs(): number {
    return this.currentIntervalMs;
  }

  get isRunning(): boolean {
    return this.running;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.schedule();
  }

  stop(): void {
    this.running = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  /** 立即执行一轮（测试与手动触发）。 */
  async runOnce(): Promise<{ projected: number; skipped: number; error?: string }> {
    if (!this.configDao.dbEnabled()) {
      return { projected: 0, skipped: this.outbox.pendingCount(), error: 'not_configured' };
    }
    const factory = this.options.clientFactory;
    let client: Client | null = null;
    let projected = 0;
    let skipped = 0;
    try {
      client = factory
        ? factory()
        : new Client({
            host: this.configDao.getConnectionInfo().host,
            port: this.configDao.getConnectionInfo().port,
            database: this.configDao.getConnectionInfo().db,
            user: process.env.QLH_DB_USER,
            password: process.env.QLH_DB_PASSWORD,
            connectionTimeoutMillis: 3000,
          });
      await client.connect();
      await client.query(`
        CREATE TABLE IF NOT EXISTS outbox_projection (
          event_id TEXT PRIMARY KEY,
          aggregate TEXT NOT NULL,
          aggregate_version INTEGER NOT NULL,
          event_type TEXT NOT NULL,
          payload TEXT NOT NULL,
          projected_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
      `);
      const events = this.outbox.pending();
      for (const event of events) {
        const result = await client.query(
          `INSERT INTO outbox_projection
             (event_id, aggregate, aggregate_version, event_type, payload)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (event_id) DO NOTHING`,
          [event.event_id, event.aggregate, event.aggregate_version,
           event.event_type, event.payload],
        );
        if ((result.rowCount ?? 0) > 0) {
          projected += 1;
          this.outbox.markProjected(event.event_id);
        } else {
          skipped += 1;
          this.outbox.markProjected(event.event_id);
        }
      }
      // 成功：重置退避
      this.currentIntervalMs =
        this.options.baseIntervalMs ?? DEFAULT_OPTIONS.baseIntervalMs;
      return { projected, skipped };
    } catch (err) {
      this.backoff();
      return {
        projected,
        skipped: this.outbox.pendingCount(),
        error: err instanceof Error ? err.message : String(err),
      };
    } finally {
      if (client) await client.end().catch(() => undefined);
    }
  }

  private backoff(): void {
    const max = this.options.maxIntervalMs ?? DEFAULT_OPTIONS.maxIntervalMs;
    this.currentIntervalMs = Math.min(
      this.currentIntervalMs * 2,
      max,
    );
  }

  private schedule(): void {
    if (!this.running) return;
    this.timer = setTimeout(() => {
      this.runOnce().finally(() => this.schedule());
    }, this.currentIntervalMs);
  }
}
