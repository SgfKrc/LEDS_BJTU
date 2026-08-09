/**
 * 审查工单持久化 — 默认使用主节点 SQLite，对齐 src/review.py 数据结构
 * (微服务架构改造计划 阶段 3.2 review 域)
 *
 * 显式传入文件时保留 JSON 兼容模式；默认运行路径不依赖远端 PostgreSQL。
 *
 * 时间戳均为 epoch 秒（float，对齐 Python time.time()）。
 * 并发：Node 单线程 + 原子写（tmp + rename），对齐 Python 端
 * cast_vote 的"3 次重试防并发读改写"效果（TS 天然串行）。
 */
import { Inject, Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { SqliteStore } from './sqlite-store';

export type TicketStatus = 'pending' | 'approved' | 'rejected' | 'expired';

export interface Vote {
  voter_node_id: string;
  value: number; // -1 / 0 / +1
  timestamp: number;
  comment: string;
}

export interface ReviewTicket {
  ticket_id: string;
  status: TicketStatus;
  created_at: number;
  created_by: string;
  target_node_id: string;
  transfer_reason: string;
  votes: Vote[];
  score: number;
  expires_at: number;
  resolved_at: number | null;
  notification_sent: boolean;
}

export function nowEpoch(): number {
  return Date.now() / 1000;
}

/** 对齐 ReviewTicket.to_dict() 的 11 字段输出 */
export function ticketToDict(t: ReviewTicket): Record<string, unknown> {
  return {
    ticket_id: t.ticket_id,
    status: t.status,
    created_at: t.created_at,
    created_by: t.created_by,
    target_node_id: t.target_node_id,
    transfer_reason: t.transfer_reason,
    votes: t.votes.map((v) => ({
      voter_node_id: v.voter_node_id,
      value: v.value,
      timestamp: v.timestamp,
      comment: v.comment,
    })),
    score: t.score,
    expires_at: t.expires_at,
    resolved_at: t.resolved_at,
    notification_sent: t.notification_sent,
  };
}

export function resolveReviewFile(env: NodeJS.ProcessEnv = process.env): string {
  return env.QLH_REVIEW_STORE?.trim() || path.join(process.cwd(), 'review_tickets.json');
}

@Injectable()
export class ReviewStore {
  private readonly file: string;
  private readonly sqlite: SqliteStore | null;
  private legacyImportChecked = false;

  constructor(@Optional() @Inject(SqliteStore) storeOrFile?: SqliteStore | string) {
    if (typeof storeOrFile === 'string') {
      this.sqlite = null;
      this.file = storeOrFile;
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
    } else {
      this.sqlite = storeOrFile ?? new SqliteStore();
      this.file = '';
    }
  }

  private useSqlite(): SqliteStore | null {
    if (!this.sqlite) return null;
    this.sqlite.open();
    this.importLegacyJsonOnce();
    return this.sqlite;
  }

  private importLegacyJsonOnce(): void {
    if (!this.sqlite || this.legacyImportChecked) return;
    this.legacyImportChecked = true;
    const marker = '__legacy_json_review_v1__';
    const marked = this.sqlite.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get(marker) as { value: string } | undefined;
    if (marked) return;
    const legacyFile = resolveReviewFile();
    try {
      const raw = fs.readFileSync(legacyFile, 'utf-8');
      const parsed = JSON.parse(raw);
      const tickets = Array.isArray(parsed) ? parsed.map((d) => this.normalize(d)) : [];
      this.sqlite.transaction(() => {
        for (const ticket of tickets) {
          if (!ticket.ticket_id) continue;
          this.sqlite!.prepare(
            `INSERT INTO review_tickets (ticket_id, status, created_at, payload)
             VALUES (?, ?, ?, ?) ON CONFLICT(ticket_id) DO NOTHING`,
          ).run(ticket.ticket_id, ticket.status, ticket.created_at, JSON.stringify(ticket));
        }
        this.sqlite!.prepare(
          `INSERT INTO cluster_settings (key, value, updated_at)
           VALUES (?, '1', ?) ON CONFLICT(key) DO NOTHING`,
        ).run(marker, new Date().toISOString());
      });
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT') {
        console.warn(`[control-svc] review JSON 兼容导入失败，保留 SQLite 空域: ${String(err)}`);
      }
      this.sqlite.prepare(
        `INSERT INTO cluster_settings (key, value, updated_at)
         VALUES (?, '1', ?) ON CONFLICT(key) DO NOTHING`,
      ).run(marker, new Date().toISOString());
    }
  }

  private sqliteTickets(sqlite: SqliteStore): ReviewTicket[] {
    const rows = sqlite.prepare(
      'SELECT payload FROM review_tickets ORDER BY created_at DESC',
    ).all() as Array<{ payload: string }>;
    return rows.map((row) => this.normalize(JSON.parse(row.payload)));
  }

  loadAll(): ReviewTicket[] {
    const sqlite = this.useSqlite();
    if (sqlite) return this.sqliteTickets(sqlite);
    try {
      const raw = fs.readFileSync(this.file, 'utf-8');
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((d) => d && typeof d === 'object')
        .map((d) => this.normalize(d));
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code !== 'ENOENT') {
        // 对齐 local_store._read_json 的损坏重建语义
        console.warn(`[control-svc] review 存储损坏，重建: ${this.file}`);
      }
      return [];
    }
  }

  saveAll(tickets: ReviewTicket[]): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.transaction(() => {
        sqlite.prepare('DELETE FROM review_tickets').run();
        for (const ticket of tickets) {
          sqlite.prepare(
            `INSERT INTO review_tickets (ticket_id, status, created_at, payload)
             VALUES (?, ?, ?, ?)`,
          ).run(ticket.ticket_id, ticket.status, ticket.created_at, JSON.stringify(ticket));
        }
      });
      return;
    }
    const tmp = `${this.file}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify(tickets, null, 2), 'utf-8');
      fs.renameSync(tmp, this.file);
    } catch (err) {
      console.warn(`[control-svc] 写入 review 存储失败: ${this.file}: ${String(err)}`);
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
    }
  }

  get(ticketId: string): ReviewTicket | null {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const row = sqlite.prepare(
        'SELECT payload FROM review_tickets WHERE ticket_id = ?',
      ).get(ticketId) as { payload: string } | undefined;
      return row ? this.normalize(JSON.parse(row.payload)) : null;
    }
    return this.loadAll().find((t) => t.ticket_id === ticketId) ?? null;
  }

  upsert(ticket: ReviewTicket): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.prepare(
        `INSERT INTO review_tickets (ticket_id, status, created_at, payload)
         VALUES (?, ?, ?, ?) ON CONFLICT(ticket_id) DO UPDATE SET
           status = excluded.status, created_at = excluded.created_at,
           payload = excluded.payload`,
      ).run(ticket.ticket_id, ticket.status, ticket.created_at, JSON.stringify(ticket));
      return;
    }
    const tickets = this.loadAll();
    const idx = tickets.findIndex((t) => t.ticket_id === ticket.ticket_id);
    if (idx >= 0) tickets[idx] = ticket;
    else tickets.push(ticket);
    this.saveAll(tickets);
  }

  delete(ticketId: string): boolean {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const result = sqlite.prepare('DELETE FROM review_tickets WHERE ticket_id = ?').run(ticketId);
      return Number(result.changes) > 0;
    }
    const tickets = this.loadAll();
    const kept = tickets.filter((t) => t.ticket_id !== ticketId);
    if (kept.length === tickets.length) return false;
    this.saveAll(kept);
    return true;
  }

  /** 对齐 from_dict 的容错：非法 status 回退 pending、votes 兼容 JSON 字符串 */
  private normalize(d: Record<string, unknown>): ReviewTicket {
    const status = String(d.status ?? 'pending');
    const valid: TicketStatus[] = ['pending', 'approved', 'rejected', 'expired'];
    let rawVotes = d.votes;
    if (typeof rawVotes === 'string') {
      try {
        rawVotes = JSON.parse(rawVotes);
      } catch {
        rawVotes = [];
      }
    }
    const votes: Vote[] = Array.isArray(rawVotes)
      ? rawVotes
          .filter((v) => v && typeof v === 'object')
          .map((v) => ({
            voter_node_id: String(v.voter_node_id ?? ''),
            value: Number(v.value) || 0,
            timestamp: Number(v.timestamp) || 0,
            comment: String(v.comment ?? ''),
          }))
      : [];
    return {
      ticket_id: String(d.ticket_id ?? ''),
      status: valid.includes(status as TicketStatus)
        ? (status as TicketStatus)
        : 'pending',
      created_at: Number(d.created_at) || 0,
      created_by: String(d.created_by ?? ''),
      target_node_id: String(d.target_node_id ?? ''),
      transfer_reason: String(d.transfer_reason ?? ''),
      votes,
      score: Number(d.score) || 0,
      expires_at: Number(d.expires_at) || 0,
      resolved_at: d.resolved_at === null || d.resolved_at === undefined
        ? null
        : Number(d.resolved_at),
      notification_sent: Boolean(d.notification_sent),
    };
  }
}
