/**
 * 审查工单控制器 — 阶段 3.2 review 域（语义对齐 api_server.py:5990-6180）
 *
 * 端点（10 个）：
 *   POST   /cluster/review/create        → ticket.to_dict()（11 字段）
 *   POST   /cluster/review/vote          → 状态机后 to_dict()；404 不存在/已关闭/自投
 *   GET    /cluster/review/tickets       → {tickets, count}，?status= 过滤
 *   GET    /cluster/review/tickets/:id   → to_dict()；404
 *   GET    /cluster/review/can-vote      → {node_id, can_vote, reason}（降级放行）
 *   POST   /cluster/review/expire-check  → {expired, count}
 *   DELETE /cluster/review/tickets/:id   → {status:'deleted', ticket_id}；404
 *   DELETE /cluster/review/tickets       → {status:'deleted', count}
 *   POST   /cluster/review/mail-poll     → 降级（IMAP 未迁移）
 *   POST   /cluster/email-test           → mailer 结果（Noop 恒失败 → 500）
 *
 * 降级说明（已记录计划文档）：master 角色检查、spare-master 检查、
 * can_node_vote GPU 资格、IMAP 邮件投票均依赖 scheduler/device_profiler/
 * SMTP-IMAP 外部设施，control-svc 独立进程未迁移——角色与资格放行、
 * mail-poll 返回 skipped、邮件为 Noop（notification_sent 语义保留）。
 */
import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Post,
  Query,
} from '@nestjs/common';
import { ReviewService } from './review.service';
import { nodeIdOf } from '../../data/log-file-store';

interface CreateReviewRequest {
  target_node_id?: string;
  reason?: string;
  timeout_hours?: number;
}

interface CastVoteRequest {
  ticket_id?: string;
  vote?: number;
  comment?: string;
}

@Controller()
export class ReviewController {
  constructor(private readonly review: ReviewService) {}

  @Post('cluster/review/create')
  @HttpCode(200)
  async createTicket(@Body() body: CreateReviewRequest): Promise<Record<string, unknown>> {
    const target = body?.target_node_id;
    if (typeof target !== 'string' || !target.trim()) {
      throw new HttpException('target_node_id 必填', 400);
    }
    const timeout = Number(body?.timeout_hours) || undefined;
    const ticket = await this.review.createTicket(
      nodeIdOf(),
      target.trim(),
      typeof body?.reason === 'string' ? body.reason : '',
      timeout,
    );
    return this.review.toDict(ticket);
  }

  @Post('cluster/review/vote')
  @HttpCode(200)
  async castVote(@Body() body: CastVoteRequest): Promise<Record<string, unknown>> {
    const ticketId = body?.ticket_id;
    if (typeof ticketId !== 'string' || !ticketId.trim()) {
      throw new HttpException('ticket_id 必填', 400);
    }
    const vote = Number(body?.vote);
    if (![-1, 0, 1].includes(vote)) {
      throw new HttpException('投票值必须为 -1、0 或 +1', 400);
    }
    let ticket;
    try {
      ticket = await this.review.castVote(
        ticketId.trim(),
        nodeIdOf(),
        vote,
        typeof body?.comment === 'string' ? body.comment : '',
      );
    } catch (err) {
      throw new HttpException(`投票失败: ${String(err)}`, 500);
    }
    if (!ticket) {
      throw new HttpException(`工单 '${ticketId}' 不存在或已关闭`, 404);
    }
    return this.review.toDict(ticket);
  }

  @Get('cluster/review/tickets')
  listTickets(@Query('status') status?: string): Record<string, unknown> {
    const tickets = this.review.listTickets(status && status.trim() ? status.trim() : undefined);
    return {
      tickets: tickets.map((t) => this.review.toDict(t)),
      count: tickets.length,
    };
  }

  @Get('cluster/review/tickets/:ticketId')
  getTicket(@Param('ticketId') ticketId: string): Record<string, unknown> {
    const ticket = this.review.getTicket(ticketId);
    if (!ticket) {
      throw new HttpException(`工单 '${ticketId}' 不存在`, 404);
    }
    return this.review.toDict(ticket);
  }

  @Get('cluster/review/can-vote')
  canVote(): Record<string, unknown> {
    return {
      node_id: nodeIdOf(),
      // 降级：GPU 资格判定依赖 Python device_profiler，未迁移——控制面默认放行
      can_vote: true,
      reason: 'control-svc: 设备画像未迁移，默认放行',
    };
  }

  @Post('cluster/review/expire-check')
  @HttpCode(200)
  async expireCheck(): Promise<Record<string, unknown>> {
    const expired = await this.review.resolveExpired();
    return { expired, count: expired.length };
  }

  @Delete('cluster/review/tickets/:ticketId')
  @HttpCode(200)
  deleteTicket(@Param('ticketId') ticketId: string): Record<string, unknown> {
    if (!this.review.deleteTicket(ticketId)) {
      throw new HttpException(`工单 ${ticketId} 不存在或删除失败`, 404);
    }
    return { status: 'deleted', ticket_id: ticketId };
  }

  @Delete('cluster/review/tickets')
  @HttpCode(200)
  deleteResolved(): Record<string, unknown> {
    const count = this.review.deleteResolved();
    return { status: 'deleted', count };
  }

  @Post('cluster/review/mail-poll')
  @HttpCode(200)
  mailPoll(): Record<string, unknown> {
    // 降级：IMAP 轮询未迁移（Python 返回 poll_mail_once 的 stats）
    return { status: 'ok', polled: 0, skipped: 'imap-not-migrated' };
  }

  @Post('cluster/email-test')
  @HttpCode(200)
  async emailTest(): Promise<Record<string, unknown>> {
    try {
      const ok = await this.review.sendTestEmail();
      if (ok) {
        return { status: 'ok', message: '测试邮件已发送，请检查目标邮箱' };
      }
      throw new HttpException('邮件发送失败，请检查后端日志了解详情', 500);
    } catch (err) {
      if (err instanceof HttpException) throw err;
      throw new HttpException(`邮件发送异常: ${String(err)}`, 500);
    }
  }
}
