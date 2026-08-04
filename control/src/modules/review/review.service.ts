/**
 * 审查工单状态机 — 对齐 src/review.py ReviewManager（:137-527）
 *
 * 阈值：score >= +2 → approved；score <= -2 → rejected（:143-144）。
 * 超时：默认 48h；pending 且 now > expires_at → expired。
 *
 * 邮件通知降级：SMTP 未迁移（NoopReviewMailer 恒 false，不阻断状态机），
 * 仅 notification_sent 置位语义保留——真实 SMTP 接线留待通知域迁移。
 * 投票资格（can_node_vote GPU 检查）依赖 Python device_profiler，未迁移：
 * 降级为放行（can-vote 端点恒 true），记录于计划文档。
 */
import { Injectable, Optional } from '@nestjs/common';
import { randomBytes } from 'crypto';
import { ReviewStore, ReviewTicket, TicketStatus, nowEpoch, ticketToDict } from '../../data/review-store';

export interface ReviewMailer {
  /** 创建通知（对齐 send_review_created_alert）；返回是否成功 */
  sendReviewCreated(ticket: ReviewTicket): Promise<boolean>;
  /** 结果通知（对齐 send_review_resolved_alert）；返回是否成功 */
  sendReviewResolved(ticket: ReviewTicket): Promise<boolean>;
  /** 测试邮件（对齐 send_test_email）；返回是否成功 */
  sendTestEmail(): Promise<boolean>;
}

/** SMTP 未迁移时的默认实现：全部失败但不抛异常（对齐 Python 发送失败仅 warning） */
export class NoopReviewMailer implements ReviewMailer {
  async sendReviewCreated(): Promise<boolean> {
    return false;
  }

  async sendReviewResolved(): Promise<boolean> {
    return false;
  }

  async sendTestEmail(): Promise<boolean> {
    return false;
  }
}

export const APPROVE_THRESHOLD = 2;
export const REJECT_THRESHOLD = -2;
export const DEFAULT_TIMEOUT_HOURS = 48;

@Injectable()
export class ReviewService {
  private readonly store: ReviewStore;
  private readonly mailer: ReviewMailer;

  constructor(store: ReviewStore, @Optional() mailer?: ReviewMailer) {
    this.store = store;
    this.mailer = mailer ?? new NoopReviewMailer();
  }

  // ---- 创建工单（对齐 create_ticket :149-207） ----

  async createTicket(
    createdBy: string,
    targetNodeId: string,
    reason = '',
    timeoutHours = DEFAULT_TIMEOUT_HOURS,
  ): Promise<ReviewTicket> {
    const timeout = timeoutHours || DEFAULT_TIMEOUT_HOURS;
    const now = nowEpoch();
    const ticket: ReviewTicket = {
      ticket_id: `review_${randomHex12()}`,
      created_at: now,
      created_by: createdBy,
      target_node_id: targetNodeId,
      transfer_reason: reason,
      status: 'pending',
      votes: [],
      score: 0,
      expires_at: now + timeout * 3600,
      resolved_at: null,
      notification_sent: false,
    };
    this.store.upsert(ticket);
    try {
      const ok = await this.mailer.sendReviewCreated(ticket);
      if (ok) {
        ticket.notification_sent = true;
        this.store.upsert(ticket);
      }
    } catch (err) {
      console.warn(`[control-svc] 审查邮件通知发送失败: ${String(err)}`);
    }
    return ticket;
  }

  // ---- 投票（对齐 cast_vote :211-311） ----

  /**
   * 返回更新后的工单；None 表示工单不存在/已关闭/自投（端点映射 404）。
   * 值校验失败抛 Error（端点映射 400）。
   */
  async castVote(
    ticketId: string,
    voterNodeId: string,
    voteValue: number,
    comment = '',
  ): Promise<ReviewTicket | null> {
    if (![-1, 0, 1].includes(voteValue)) {
      throw new Error(`无效的投票值: ${voteValue}，需为 -1, 0 或 +1`);
    }
    const ticket = this.store.get(ticketId);
    if (!ticket) return null;
    if (ticket.status !== 'pending') return null;
    // 创建者和目标节点不可投票（对齐 :250-255）
    if (voterNodeId === ticket.created_by) return null;
    if (voterNodeId === ticket.target_node_id) return null;

    // 更新已有投票或追加（对齐 :257-263）
    const existing = ticket.votes.find((v) => v.voter_node_id === voterNodeId);
    if (existing) {
      existing.value = voteValue;
      existing.timestamp = nowEpoch();
      if (comment) existing.comment = comment;
    } else {
      ticket.votes.push({
        voter_node_id: voterNodeId,
        value: voteValue,
        timestamp: nowEpoch(),
        comment,
      });
    }
    ticket.score = ticket.votes.reduce((s, v) => s + v.value, 0);

    // 阈值判定（对齐 :264-286）
    let resolved = false;
    if (ticket.score >= APPROVE_THRESHOLD) {
      ticket.status = 'approved';
      ticket.resolved_at = nowEpoch();
      resolved = true;
    } else if (ticket.score <= REJECT_THRESHOLD) {
      ticket.status = 'rejected';
      ticket.resolved_at = nowEpoch();
      resolved = true;
    }
    this.store.upsert(ticket);

    if (resolved) {
      try {
        const ok = await this.mailer.sendReviewResolved(ticket);
        if (ok) {
          ticket.notification_sent = true;
          this.store.upsert(ticket);
        }
      } catch (err) {
        console.warn(`[control-svc] 审查结果邮件发送失败: ${String(err)}`);
      }
    }
    return ticket;
  }

  // ---- 查询 / 删除 / 过期（对齐 :315-432） ----

  getTicket(ticketId: string): ReviewTicket | null {
    return this.store.get(ticketId);
  }

  listTickets(status?: string): ReviewTicket[] {
    const tickets = this.store.loadAll();
    const filtered = status ? tickets.filter((t) => t.status === status) : tickets;
    // created_at DESC（对齐 list_tickets）
    return filtered.sort((a, b) => b.created_at - a.created_at);
  }

  deleteTicket(ticketId: string): boolean {
    return this.store.delete(ticketId);
  }

  deleteResolved(): number {
    const resolved = this.store.loadAll().filter((t) => t.status !== 'pending');
    if (resolved.length === 0) return 0;
    const ids = new Set(resolved.map((t) => t.ticket_id));
    this.store.saveAll(this.store.loadAll().filter((t) => !ids.has(t.ticket_id)));
    return resolved.length;
  }

  /** 过期检查：pending 且超时 → expired + 邮件；返回过期 id 列表（对齐 resolve_expired） */
  async resolveExpired(): Promise<string[]> {
    const now = nowEpoch();
    const tickets = this.store.loadAll();
    const expired: string[] = [];
    for (const t of tickets) {
      if (t.status === 'pending' && now > t.expires_at) {
        t.status = 'expired' as TicketStatus;
        t.resolved_at = now;
        expired.push(t.ticket_id);
        this.store.upsert(t);
        try {
          const ok = await this.mailer.sendReviewResolved(t);
          if (ok) {
            t.notification_sent = true;
            this.store.upsert(t);
          }
        } catch (err) {
          console.warn(`[control-svc] 过期邮件通知发送失败: ${String(err)}`);
        }
      }
    }
    return expired;
  }

  /** 供 email-test 端点调用（对齐 email_notifier.send_test_email） */
  async sendTestEmail(): Promise<boolean> {
    return this.mailer.sendTestEmail();
  }

  /** 供控制器输出 to_dict（对齐端点返回 ticket.to_dict()） */
  toDict(t: ReviewTicket): Record<string, unknown> {
    return ticketToDict(t);
  }
}

function randomHex12(): string {
  // 对齐 uuid.uuid4().hex[:12]
  return Array.from(randomBytes(6))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
