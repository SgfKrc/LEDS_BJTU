/**
 * M3 pull job 控制器（对齐 schemas/pull-job 与计划 §12.1）：
 *   POST   /models/pull                 创建幂等下载任务（后台执行）
 *   GET    /models/pull/{jobId}         查询状态与进度
 *   GET    /models/pull/{jobId}/events  SSE 下载/校验/注册事件流
 *   DELETE /models/pull/{jobId}         取消（保留可恢复 partial）
 *
 * 执行在后台线程（不阻塞请求）；SSE 事件经 PullJobService 订阅推送。
 */
import {
  Body, Controller, Delete, Get, HttpCode, HttpException, Param, Post, Sse,
} from '@nestjs/common';
import { Observable, Subject } from 'rxjs';
import { PullJobService, PullJob, PullJobEvent } from '../../data/pull-job.service';
import { PullJobExecutor } from '../../data/pull-job-executor';

class CreatePullRequest {
  idempotency_key?: string;
  source?: {
    provider?: string;
    repo_id?: string;
    requested_revision?: string;
    allow_patterns?: string[];
  };
  cancel_policy?: 'keep_partial' | 'cleanup';
}

@Controller('models/pull')
export class PullJobController {
  constructor(
    private readonly jobs: PullJobService,
    private readonly executor: PullJobExecutor,
  ) {}

  @Post()
  @HttpCode(202)
  create(@Body() body: CreatePullRequest): Record<string, unknown> {
    const source = body?.source;
    const repoId = source?.repo_id?.trim() ?? '';
    const provider = source?.provider ?? 'huggingface';
    if (!repoId) {
      throw new HttpException('source.repo_id 必填', 422);
    }
    if (!['huggingface', 'gguf_huggingface'].includes(provider)) {
      throw new HttpException('provider 仅支持 huggingface/gguf_huggingface', 400);
    }
    const job = this.jobs.create({
      idempotencyKey: body?.idempotency_key ?? `pull:${repoId}:${source?.requested_revision ?? 'main'}`,
      source: {
        provider: provider as 'huggingface' | 'gguf_huggingface',
        repo_id: repoId,
        requested_revision: source?.requested_revision?.trim() || 'main',
        allow_patterns: source?.allow_patterns ?? null,
      },
      cancelPolicy: body?.cancel_policy ?? 'keep_partial',
    });
    // 幂等：已存在且未执行 → 不重复启动
    if (job.state === 'queued' && !this.executor.isRunning(job.job_id)) {
      this.executor.start(job.job_id);
    }
    return { status: 'created', job_id: job.job_id, state: job.state };
  }

  @Get(':jobId')
  get(@Param('jobId') jobId: string): Record<string, unknown> {
    const job = this.jobs.get(jobId);
    if (!job) throw new HttpException(`job 不存在: ${jobId}`, 404);
    return { job };
  }

  @Delete(':jobId')
  @HttpCode(200)
  cancel(@Param('jobId') jobId: string): Record<string, unknown> {
    const job = this.jobs.get(jobId);
    if (!job) throw new HttpException(`job 不存在: ${jobId}`, 404);
    this.executor.cancelActive(jobId);
    const cancelled = this.jobs.cancel(jobId);
    return { status: 'cancelled', job_id: jobId, state: cancelled.state };
  }

  @Sse(':jobId/events')
  events(@Param('jobId') jobId: string): Observable<MessageEvent> {
    const job = this.jobs.get(jobId);
    if (!job) throw new HttpException(`job 不存在: ${jobId}`, 404);
    const subject = new Subject<MessageEvent>();
    const emit = (event: PullJobEvent): void => {
      if (event.job_id !== jobId) return;
      subject.next({ data: JSON.stringify(event) } as MessageEvent);
      if (event.event === 'registered'
          || event.event === 'failed'
          || event.event === 'cancelled') {
        subject.complete();
      }
    };
    const unsubscribe = this.jobs.subscribe(emit);
    // 初始快照：当前 job 状态
    const current = this.jobs.get(jobId);
    if (current) {
      subject.next({
        data: JSON.stringify({ event: 'started', job_id: jobId, payload: current, at: current.updated_at }),
      } as MessageEvent);
    }
    return new Observable<MessageEvent>((subscriber) => {
      const subscription = subject.subscribe(subscriber);
      return () => {
        subscription.unsubscribe();
        unsubscribe();
      };
    });
  }
}
