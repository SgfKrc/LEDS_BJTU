/**
 * M3 pull job 执行器 — resolve → 下载（staging）→ 校验 → 工件库提交 → registered。
 *
 * 状态机推进走 PullJobService（SQLite 持久化，重启可恢复）；
 * 取消经 AbortController 传递；校验失败 → quarantine + failed。
 */
import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import {
  PullJobService, PullJob,
} from './pull-job.service';
import { HfResolver, ResolveResult } from './hf-resolver';
import { HfDownloader } from './hf-downloader';
import { ArtifactStore, sha256Hex } from './artifact-store';
import { ModelInspector } from './model-inspector';
import { ModelDiskBudget } from './model-disk-budget';
import { ModelCredentialStore } from './model-credential-store';
import { ModelLicenseAcceptanceRepository } from './model-license-acceptance';

@Injectable()
export class PullJobExecutor {
  private readonly active = new Map<string, AbortController>();

  constructor(
    private readonly jobs: PullJobService,
    private readonly resolver: HfResolver,
    private readonly downloader: HfDownloader,
    private readonly store: ArtifactStore,
    private readonly inspector: ModelInspector,
    private readonly credentials: ModelCredentialStore,
    private readonly licenses: ModelLicenseAcceptanceRepository,
    private readonly diskBudget: ModelDiskBudget = new ModelDiskBudget(),
  ) {}

  isRunning(jobId: string): boolean {
    return this.active.has(jobId);
  }

  cancelActive(jobId: string): void {
    this.active.get(jobId)?.abort();
  }

  /** 启动执行（异步，不阻塞请求）。 */
  start(jobId: string): void {
    if (this.active.has(jobId)) return;
    const persisted = this.jobs.get(jobId);
    if (!persisted || [
      'registered', 'failed', 'cancelled', 'rejected', 'quarantined', 'rolled_back',
    ].includes(persisted.state)) return;
    if (persisted.state !== 'queued') this.jobs.requeue(jobId);
    const controller = new AbortController();
    this.active.set(jobId, controller);
    this.execute(jobId, controller.signal)
      .catch((err) => {
        try {
          const message = err instanceof Error ? err.message : String(err);
          this.jobs.transition(jobId, 'failed', {
            error: { code: 'executor_error', message },
          });
        } catch {
          // 已终止则忽略
        }
      })
      .finally(() => this.active.delete(jobId));
  }

  resumeActive(): number {
    const active = this.jobs.listActive();
    for (const job of active) this.start(job.job_id);
    return active.length;
  }

  private async execute(jobId: string, signal: AbortSignal): Promise<void> {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error(`job 不存在: ${jobId}`);
    const repoId = job.source.repo_id;
    if (!repoId) throw new Error(`job 缺少 repo_id: ${jobId}`);

    // ---- resolving ----
    this.jobs.transition(jobId, 'resolving');
    const token = await this.credentials.get(job.source.credential_ref);
    let resolved: ResolveResult;
    try {
      resolved = await this.resolver.resolve(
        repoId, job.source.requested_revision, job.source.allow_patterns,
        job.source.endpoint, { token },
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.startsWith('HF credential ')) {
        this.jobs.transition(jobId, 'rejected', {
          error: { code: 'credential_unavailable', message },
        });
        return;
      }
      throw error;
    }
    const licenseId = resolved.license ?? 'unknown';
    const acceptance = resolved.gated && resolved.license
      ? this.licenses.get(repoId, licenseId)
      : null;
    if (resolved.gated && !token) {
      this.jobs.transition(jobId, 'rejected', {
        error: {
          code: 'credential_required',
          message: `gated repository requires credential_ref: ${repoId}`,
        },
      });
      return;
    }
    if (resolved.gated && !acceptance) {
      this.jobs.transition(jobId, 'rejected', {
        error: {
          code: 'license_acceptance_required',
          message: `license acceptance required: ${repoId} (${licenseId})`,
        },
      });
      return;
    }
    const disk = this.diskBudget.evaluate(resolved.files, this.store);
    if (!disk.sufficient) {
      this.jobs.transition(jobId, 'rejected', {
        error: {
          code: 'insufficient_storage',
          message: `required=${disk.disk_required_bytes}, available=${disk.disk_available_bytes}`,
        },
      });
      return;
    }
    this.jobs.transition(jobId, 'downloading', {
      source: { ...job.source, resolved_revision: resolved.resolvedRevision },
      progress: {
        total_bytes: resolved.files.reduce((s, f) => s + f.size, 0),
        downloaded_bytes: 0,
        files_total: resolved.files.length,
        files_done: 0,
        current_file: null,
      },
    });

    // ---- downloading（staging 只写）----
    const staged: Array<{ rel: string; size: number; sha256?: string | null }> = [];
    let done = 0;
    let bytes = 0;
    for (const file of resolved.files) {
      if (signal.aborted) {
        this.jobs.transition(jobId, 'cancelled');
        return;
      }
      const rel = file.rfilename;
      const dest = path.join(this.store.stagingDir(jobId), rel);
      await this.downloader.downloadFile(
        repoId, resolved.resolvedRevision, file.rfilename, dest, {
          signal,
          expectedSize: file.size,
          apiBase: job.source.endpoint,
          token,
          onProgress: (p) => {
            this.jobs.updateProgress(jobId, {
              downloaded_bytes: bytes + p.bytesDownloaded,
              current_file: rel,
            });
          },
        },
      );
      bytes += file.size;
      done += 1;
      staged.push({ rel, size: file.size, sha256: file.sha256 ?? null });
      this.jobs.updateProgress(jobId, {
        downloaded_bytes: bytes,
        files_done: done,
        current_file: null,
      });
    }
    if (signal.aborted) {
      this.jobs.transition(jobId, 'cancelled');
      return;
    }

    // ---- verifying（sha256 校验；HF 提供 sha256 则核对）----
    this.jobs.transition(jobId, 'verifying');
    const verified: Array<{ rel: string; size: number; sha256: string }> = [];
    for (const entry of staged) {
      const filePath = path.join(this.store.stagingDir(jobId), entry.rel);
      const actual = sha256Hex(fs.readFileSync(filePath));
      if (entry.sha256 && entry.sha256 !== actual) {
        this.jobs.transition(jobId, 'quarantined', {
          error: {
            code: 'sha256_mismatch',
            message: `${entry.rel}: 期望 ${entry.sha256}，实际 ${actual}`,
          },
        });
        this.store.quarantine(jobId, 'sha256_mismatch');
        return;
      }
      verified.push({ rel: entry.rel, size: entry.size, sha256: actual });
    }

    // ---- 静态检查（staging 副本）----
    const ggufFiles = verified.filter((f) => f.rel.endsWith('.gguf'));
    let inspection = null;
    if (ggufFiles.length > 0) {
      inspection = this.inspector.inspectGguf(
        path.join(this.store.stagingDir(jobId), ggufFiles[0].rel),
      );
      if (!inspection.ok) {
        this.jobs.transition(jobId, 'quarantined', {
          error: {
            code: 'inspection_failed',
            message: inspection.errors.join('; '),
          },
        });
        this.store.quarantine(jobId, 'inspection_failed');
        return;
      }
    }

    // ---- 提交工件库（原子 rename + 去重）----
    const committed: Array<{ path: string; size: number; sha256: string }> = [];
    for (const entry of verified) {
      const blob = this.store.commitBlob(jobId, entry.rel);
      committed.push({ path: entry.rel, size: blob.size, sha256: blob.digest });
    }
    const sorted = [...committed].sort((a, b) => a.path.localeCompare(b.path));
    const aggregate = sha256Hex(Buffer.from(sorted.map((f) => f.sha256).join('')));
    const artifactId = `sha256:${aggregate}`;
    const manifest: Record<string, unknown> = {
      schema_version: 1,
      namespace: 'hub',
      name: repoId.split('/').pop() ?? repoId,
      tag: resolved.resolvedRevision.slice(0, 12),
      artifact_id: artifactId,
      source: {
        provider: 'huggingface',
        repo_id: repoId,
        requested_revision: job.source.requested_revision,
        resolved_revision: resolved.resolvedRevision,
        source_id: job.source.source_id ?? null,
        endpoint: job.source.endpoint ?? null,
        credential_ref: job.source.credential_ref ?? null,
      },
      format: inspection ? 'gguf' : 'safetensors',
      engine: inspection ? 'llama_cpp' : 'pytorch_transformers',
      family: inspection?.family ?? null,
      quantization: inspection?.quantization ?? null,
      context_length: inspection?.context_length ?? null,
      files: sorted,
      capabilities: inspection?.capabilities ?? {
        full_worker: false, pytorch_layer_pipeline: false,
        llama_cpp: false, task_stage: false,
      },
      requirements: {
        runtime_profile: inspection ? 'llm-cpu-v1' : 'llm-cuda-v1',
      },
      license: {
        id: licenseId,
        acceptance_required: resolved.gated,
        accepted_at: acceptance?.accepted_at ?? null,
      },
      trust_policy: { trust_remote_code: false },
    };
    const manifestPath = this.store.writeManifest(manifest);
    this.store.cleanupStaging(jobId);

    // ---- registered ----
    this.jobs.transition(jobId, 'registered', {
      artifact_id: artifactId,
      error: null,
    });
    void manifestPath;
  }
}
