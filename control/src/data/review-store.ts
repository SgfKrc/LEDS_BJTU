/**
 * 审查工单持久化 — 对齐 src/review.py ReviewTicket 数据结构
 * (微服务架构改造计划 阶段 3.2 review 域)
 *
 * 存储：JSON 文件（对齐计划 §3.2 "DB 不可用回落本地 JSON 存储" 的降级语义；
 * Python 侧 PostgreSQL 分支未迁移——并行共存期间票数据仍在旧后端，
 * control-svc 从零独立积累；清理阶段切换时再处理数据迁移）。
 *
 * 时间戳均为 epoch 秒（float，对齐 Python time.time()）。
 * 并发：Node 单线程 + 原子写（tmp + rename），对齐 Python 端
 * cast_vote 的"3 次重试防并发读改写"效果（TS 天然串行）。
 */
import { Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

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

  constructor(@Optional() file?: string) {
    this.file = file ?? resolveReviewFile();
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
  }

  loadAll(): ReviewTicket[] {
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
    return this.loadAll().find((t) => t.ticket_id === ticketId) ?? null;
  }

  upsert(ticket: ReviewTicket): void {
    const tickets = this.loadAll();
    const idx = tickets.findIndex((t) => t.ticket_id === ticket.ticket_id);
    if (idx >= 0) tickets[idx] = ticket;
    else tickets.push(ticket);
    this.saveAll(tickets);
  }

  delete(ticketId: string): boolean {
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
