/**
 * M1 outbox — 本地事务库的事件出站队列（单向投影到远端 PostgreSQL）。
 *
 * 语义（一键模型部署计划 §10/§11）：
 *  - control-svc 本地 SQLite 是事实源；远端 PostgreSQL 是可选投影/查询库；
 *  - 事件以 event_id 全局唯一，投影幂等（远端 on conflict do nothing +
 *    aggregate_version 拒绝旧事件覆盖新状态）；
 *  - 本地写失败 = 控制面只读故障，禁止假成功。
 */
import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SqliteStore } from './sqlite-store';

export interface OutboxEvent {
  event_id: string;
  aggregate: string;
  aggregate_version: number;
  event_type: string;
  payload: string;
  created_at: string;
  projected_at: string | null;
}

@Injectable()
export class OutboxService {
  constructor(private readonly store: SqliteStore) {}

  /** 入队（aggregate 内版本单调递增）。 */
  enqueue(
    aggregate: string,
    eventType: string,
    payload: unknown,
    aggregateVersion?: number,
  ): OutboxEvent {
    const version =
      aggregateVersion ?? this.nextVersion(aggregate);
    const event: OutboxEvent = {
      event_id: randomUUID(),
      aggregate,
      aggregate_version: version,
      event_type: eventType,
      payload: JSON.stringify(payload),
      created_at: new Date().toISOString(),
      projected_at: null,
    };
    this.store.prepare(
      `INSERT INTO outbox
         (event_id, aggregate, aggregate_version, event_type, payload, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(
      event.event_id,
      event.aggregate,
      event.aggregate_version,
      event.event_type,
      event.payload,
      event.created_at,
    );
    return event;
  }

  nextVersion(aggregate: string): number {
    const row = this.store.prepare(
      'SELECT COALESCE(MAX(aggregate_version), 0) + 1 AS v '
      + 'FROM outbox WHERE aggregate = ?',
    ).get(aggregate) as { v: number };
    return Number(row.v);
  }

  /** 未投影事件（按 created_at 升序，批量上限）。 */
  pending(limit = 200): OutboxEvent[] {
    return this.store.prepare(
      'SELECT * FROM outbox WHERE projected_at IS NULL '
      + 'ORDER BY created_at ASC LIMIT ?',
    ).all(limit) as unknown as OutboxEvent[];
  }

  pendingCount(): number {
    const row = this.store.prepare(
      'SELECT COUNT(*) AS c FROM outbox WHERE projected_at IS NULL',
    ).get() as { c: number };
    return Number(row.c);
  }

  oldestPendingAgeSeconds(): number | null {
    const row = this.store.prepare(
      'SELECT MIN(created_at) AS t FROM outbox WHERE projected_at IS NULL',
    ).get() as { t: string | null };
    if (!row.t) return null;
    const ageMs = Date.now() - new Date(row.t).getTime();
    return Math.max(0, Math.floor(ageMs / 1000));
  }

  /** 标记已投影（幂等）。 */
  markProjected(eventId: string): void {
    this.store.prepare(
      'UPDATE outbox SET projected_at = ? WHERE event_id = ?',
    ).run(new Date().toISOString(), eventId);
  }
}
