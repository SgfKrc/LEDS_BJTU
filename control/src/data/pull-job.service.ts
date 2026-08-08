/**
 * M3 pull job 服务 — 下载任务状态机（对齐 schemas/pull-job.schema.json）。
 *
 * 状态机：queued → resolving → downloading → verifying → adapting → registered
 * 终止态：failed / cancelled / rejected / quarantined / rolled_back
 *
 * 持久化：SQLite pull_jobs 表（M1 建），payload 保存完整 job JSON；
 * 服务重启后从最后持久状态恢复（不猜测进程内状态）。
 * 进度事件经订阅者推送给 SSE（controller 订阅）。
 */
import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SqliteStore } from './sqlite-store';
import { ArtifactRuntimeRecord } from './artifact-runtime-repository';

export type PullJobState =
  | 'queued' | 'resolving' | 'downloading' | 'verifying' | 'adapting'
  | 'registered' | 'failed' | 'cancelled' | 'rejected' | 'quarantined'
  | 'rolled_back';

export interface PullJob {
  schema_version: number;
  job_id: string;
  idempotency_key: string;
  state: PullJobState;
  source: {
    provider: 'huggingface' | 'gguf_huggingface' | 'local_directory' | 'modelscope';
    repo_id?: string;
    requested_revision: string;
    resolved_revision?: string | null;
    allow_patterns?: string[] | null;
    source_id?: string;
    endpoint?: string;
    credential_ref?: string | null;
  };
  cancel_policy: 'keep_partial' | 'cleanup';
  progress: {
    total_bytes: number;
    downloaded_bytes: number;
    files_total: number;
    files_done: number;
    current_file: string | null;
  };
  artifact_id?: string | null;
  runtime_check?: ArtifactRuntimeRecord | null;
  error?: { code: string; message: string } | null;
  created_at: string;
  updated_at: string;
}

export interface PullJobEvent {
  event: 'started' | 'progress' | 'resolved' | 'registered' | 'failed' | 'cancelled';
  job_id: string;
  payload: Partial<PullJob>;
  at: string;
}

const TERMINAL_STATES = new Set<PullJobState>([
  'registered', 'failed', 'cancelled', 'rejected', 'quarantined', 'rolled_back',
]);

const ALLOWED_TRANSITIONS: Record<PullJobState, ReadonlySet<PullJobState>> = {
  queued: new Set(['resolving', 'cancelled', 'failed', 'rejected']),
  resolving: new Set(['downloading', 'cancelled', 'failed', 'rejected']),
  downloading: new Set(['verifying', 'cancelled', 'failed', 'quarantined']),
  verifying: new Set(['adapting', 'registered', 'failed', 'quarantined']),
  adapting: new Set(['registered', 'failed', 'quarantined']),
  registered: new Set(),
  failed: new Set(),
  cancelled: new Set(),
  rejected: new Set(),
  quarantined: new Set(),
  rolled_back: new Set(),
};

@Injectable()
export class PullJobService {
  private readonly subscribers = new Set<(event: PullJobEvent) => void>();

  constructor(private readonly store: SqliteStore) {}

  // ---- 订阅（SSE）----

  subscribe(listener: (event: PullJobEvent) => void): () => void {
    this.subscribers.add(listener);
    return () => this.subscribers.delete(listener);
  }

  private emit(event: PullJobEvent): void {
    for (const listener of this.subscribers) {
      try {
        listener(event);
      } catch {
        // 订阅者异常不阻断 job 推进
      }
    }
  }

  // ---- CRUD ----

  /** 创建 job（幂等键去重：同键返回既有 job）。 */
  create(input: {
    idempotencyKey: string;
    source: PullJob['source'];
    cancelPolicy?: 'keep_partial' | 'cleanup';
    totalBytes?: number;
    filesTotal?: number;
  }): PullJob {
    const existing = this.findByKey(input.idempotencyKey);
    if (existing) return existing;
    const now = new Date().toISOString();
    const job: PullJob = {
      schema_version: 1,
      job_id: `pull_${randomUUID().slice(0, 12)}`,
      idempotency_key: input.idempotencyKey,
      state: 'queued',
      source: input.source,
      cancel_policy: input.cancelPolicy ?? 'keep_partial',
      progress: {
        total_bytes: input.totalBytes ?? 0,
        downloaded_bytes: 0,
        files_total: input.filesTotal ?? 0,
        files_done: 0,
        current_file: null,
      },
      artifact_id: null,
      runtime_check: null,
      error: null,
      created_at: now,
      updated_at: now,
    };
    this.persist(job);
    this.emit({ event: 'started', job_id: job.job_id, payload: job, at: now });
    return job;
  }

  get(jobId: string): PullJob | null {
    const row = this.store.prepare(
      'SELECT payload FROM pull_jobs WHERE job_id = ?',
    ).get(jobId) as { payload: string } | undefined;
    return row ? (JSON.parse(row.payload) as PullJob) : null;
  }

  findByKey(idempotencyKey: string): PullJob | null {
    const row = this.store.prepare(
      'SELECT payload FROM pull_jobs WHERE idempotency_key = ?',
    ).get(idempotencyKey) as { payload: string } | undefined;
    return row ? (JSON.parse(row.payload) as PullJob) : null;
  }

  list(): PullJob[] {
    const rows = this.store.prepare(
      'SELECT payload FROM pull_jobs ORDER BY created_at DESC LIMIT 100',
    ).all() as unknown as Array<{ payload: string }>;
    return rows.map((r) => JSON.parse(r.payload) as PullJob);
  }

  /** 重启恢复：未终止的 job（供启动时续跑）。 */
  listActive(): PullJob[] {
    return this.list().filter((j) => !TERMINAL_STATES.has(j.state));
  }

  // ---- 状态迁移 ----

  transition(jobId: string, state: PullJobState, patch: Partial<PullJob> = {}): PullJob {
    const job = this.get(jobId);
    if (!job) throw new Error(`job 不存在: ${jobId}`);
    if (TERMINAL_STATES.has(job.state)) {
      // 终止态只允许 registered/failed/cancelled 的复查更新（幂等）
      if (job.state === state) return job;
      throw new Error(`job 已终止（${job.state}），不能迁移到 ${state}`);
    }
    if (!ALLOWED_TRANSITIONS[job.state].has(state)) {
      throw new Error(`非法 job 状态迁移: ${job.state} -> ${state}`);
    }
    const next: PullJob = { ...job, ...patch, state, updated_at: new Date().toISOString() };
    this.persist(next);
    const eventMap: Record<string, PullJobEvent['event']> = {
      resolving: 'resolved',
      registered: 'registered',
      failed: 'failed',
      rejected: 'failed',
      quarantined: 'failed',
      rolled_back: 'failed',
      cancelled: 'cancelled',
    };
    const event = eventMap[state] ?? 'progress';
    this.emit({ event, job_id: jobId, payload: next, at: next.updated_at });
    return next;
  }

  updateProgress(
    jobId: string,
    progress: Partial<PullJob['progress']>,
  ): PullJob {
    const job = this.get(jobId);
    if (!job || TERMINAL_STATES.has(job.state)) return job as PullJob;
    const next: PullJob = {
      ...job,
      progress: { ...job.progress, ...progress },
      updated_at: new Date().toISOString(),
    };
    this.persist(next);
    this.emit({ event: 'progress', job_id: jobId, payload: next, at: next.updated_at });
    return next;
  }

  /** 取消（幂等）。 */
  cancel(jobId: string, policy?: 'keep_partial' | 'cleanup'): PullJob {
    const job = this.get(jobId);
    if (!job) throw new Error(`job 不存在: ${jobId}`);
    if (TERMINAL_STATES.has(job.state)) return job;
    return this.transition(jobId, 'cancelled', {
      cancel_policy: policy ?? job.cancel_policy,
    });
  }

  /** Restart recovery always re-resolves the source before reusing partial data. */
  requeue(jobId: string): PullJob {
    const job = this.get(jobId);
    if (!job) throw new Error(`job 不存在: ${jobId}`);
    if (job.state === 'queued') return job;
    if (TERMINAL_STATES.has(job.state)) {
      throw new Error(`job 已终止（${job.state}），不能恢复`);
    }
    const next: PullJob = {
      ...job,
      state: 'queued',
      source: { ...job.source, resolved_revision: null },
      progress: { ...job.progress, current_file: null },
      error: null,
      updated_at: new Date().toISOString(),
    };
    this.persist(next);
    return next;
  }

  private persist(job: PullJob): void {
    this.store.prepare(
      `INSERT INTO pull_jobs (job_id, idempotency_key, state, payload, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(job_id) DO UPDATE SET
         state = excluded.state,
         payload = excluded.payload,
         updated_at = excluded.updated_at`,
    ).run(
      job.job_id, job.idempotency_key, job.state,
      JSON.stringify(job), job.created_at, job.updated_at,
    );
  }
}
