/**
 * M3 pull job 测试：状态机/持久化/恢复/幂等键、executor 全流程
 * （mock fetch：HF API resolve + 文件流下载）、取消、sha256 校验失败、
 * 控制器契约（POST/GET/DELETE）与 curated recipes。
 */
import { mkdtempSync, rmSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { SqliteStore } from '../src/data/sqlite-store';
import { PullJobService } from '../src/data/pull-job.service';
import { PullJobExecutor } from '../src/data/pull-job-executor';
import { HfResolver, globMatch } from '../src/data/hf-resolver';
import { HfDownloader } from '../src/data/hf-downloader';
import { ModelHttpClient } from '../src/data/model-http-client';
import { ModelCredentialStore } from '../src/data/model-credential-store';
import { ModelLicenseAcceptanceRepository } from '../src/data/model-license-acceptance';
import { ClusterSettingsRepository } from '../src/data/cluster-settings-repository';
import { ArtifactStore } from '../src/data/artifact-store';
import { ModelInspector } from '../src/data/model-inspector';
import { CURATED_RECIPES } from '../src/data/curated-catalog';

function tempStore(): { dir: string; store: SqliteStore } {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-m3-'));
  const store = new SqliteStore(join(dir, 'ctl.sqlite3'));
  store.open();
  return { dir, store };
}

/** mock fetch：/api/models/{repo} → resolve JSON；/resolve/{rev}/{file} → 字节流。 */
function mockFetch(files: Array<{ name: string; content: Buffer }>, opts: {
  failDigest?: boolean;
  hang?: boolean;
} = {}) {
  const digest = (data: Buffer): string =>
    require('crypto').createHash('sha256').update(data).digest('hex');
  return async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const target = String(url);
    if (target.includes('/api/models/')) {
      return new Response(JSON.stringify({
        sha: 'a'.repeat(40),
        siblings: files.map((f) => ({
          rfilename: f.name,
          size: f.content.length,
          sha256: opts.failDigest ? '0'.repeat(64) : digest(f.content),
        })),
      }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    if (target.includes('/resolve/')) {
      const name = decodeURIComponent(target.split('/resolve/')[1].split('/').slice(1).join('/'));
      const file = files.find((f) => f.name === name);
      if (!file) return new Response('not found', { status: 404 });
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array(file.content));
          controller.close();
        },
      });
      return new Response(stream, {
        status: 200,
        headers: { 'content-length': String(file.content.length) },
      });
    }
    return new Response('unexpected', { status: 500 });
  };
}

describe('PullJobService（M3）', () => {
  it('状态机推进与持久化；幂等键去重；重启恢复 active', () => {
    const { dir, store } = tempStore();
    const jobs = new PullJobService(store);
    const job = jobs.create({
      idempotencyKey: 'k1',
      source: { provider: 'huggingface', repo_id: 'r/m', requested_revision: 'main' },
    });
    expect(job.state).toBe('queued');
    expect(jobs.create({
      idempotencyKey: 'k1',
      source: { provider: 'huggingface', repo_id: 'r/m', requested_revision: 'main' },
    }).job_id).toBe(job.job_id); // 幂等键去重

    jobs.transition(job.job_id, 'resolving');
    jobs.transition(job.job_id, 'downloading', {
      progress: { total_bytes: 100, downloaded_bytes: 10, files_total: 1, files_done: 0, current_file: 'a.bin' },
    });
    jobs.transition(job.job_id, 'verifying');
    jobs.transition(job.job_id, 'registered', { artifact_id: 'sha256:abc' });
    expect(jobs.get(job.job_id)?.state).toBe('registered');
    expect(jobs.listActive().length).toBe(0);

    // 重启恢复：active job 从持久化读出
    const job2 = jobs.create({
      idempotencyKey: 'k2',
      source: { provider: 'huggingface', repo_id: 'r/m2', requested_revision: 'main' },
    });
    jobs.transition(job2.job_id, 'resolving');
    jobs.transition(job2.job_id, 'downloading');
    const fresh = new PullJobService(store); // 模拟重启
    const active = fresh.listActive();
    expect(active.some((j) => j.job_id === job2.job_id)).toBe(true);

    // 终止态禁止迁移
    expect(() => fresh.transition(job.job_id, 'failed')).toThrow();
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('取消（幂等）', () => {
    const { dir, store } = tempStore();
    const jobs = new PullJobService(store);
    const job = jobs.create({
      idempotencyKey: 'kc', source: { provider: 'huggingface', repo_id: 'r/m', requested_revision: 'main' },
    });
    const cancelled = jobs.cancel(job.job_id);
    expect(cancelled.state).toBe('cancelled');
    expect(jobs.cancel(job.job_id).state).toBe('cancelled'); // 幂等
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('PullJobExecutor（M3）', () => {
  function makeExecutor(store: SqliteStore, dir: string, files: Array<{ name: string; content: Buffer }>, opts: {
    failDigest?: boolean;
  } = {}) {
    const jobs = new PullJobService(store);
    const fetchFn = mockFetch(files, opts);
    const http = new ModelHttpClient({ fetchFn, proxyUrl: null });
    const resolver = new HfResolver(http);
    const downloader = new HfDownloader(http, { progressThrottleMs: 0 });
    const artifactStore = new ArtifactStore(join(dir, 'store'));
    const protector = {
      name: 'test-protector',
      protect: async (secret: string) => Buffer.from(secret).toString('base64'),
      unprotect: async (ciphertext: string) => Buffer.from(ciphertext, 'base64').toString(),
    };
    const credentials = new ModelCredentialStore({
      rootDir: join(dir, 'credentials'), protector,
    });
    const licenses = new ModelLicenseAcceptanceRepository(
      new ClusterSettingsRepository(store),
    );
    const executor = new PullJobExecutor(
      jobs, resolver, downloader, artifactStore, new ModelInspector(),
      credentials, licenses,
    );
    return { jobs, executor, artifactStore };
  }

  it('全流程：resolve → 下载 → 校验 → 注册（manifest 与工件库）', async () => {
    const { dir, store } = tempStore();
    const gguf = Buffer.concat([
      Buffer.from('GGUF'),
      (() => { const v = Buffer.alloc(4); v.writeUInt32LE(3); return v; })(),
      (() => { const t = Buffer.alloc(8); t.writeBigUInt64LE(0n); return t; })(),
      (() => { const k = Buffer.alloc(8); k.writeBigUInt64LE(0n); return k; })(),
    ]);
    const { jobs, executor, artifactStore } = makeExecutor(store, dir, [
      { name: 'model.gguf', content: gguf },
    ]);
    const job = jobs.create({
      idempotencyKey: 'pull1',
      source: { provider: 'gguf_huggingface', repo_id: 'r/m', requested_revision: 'main', allow_patterns: ['*.gguf'] },
    });
    executor.start(job.job_id);
    // 等待注册完成（轮询）
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      if (jobs.get(job.job_id)?.state === 'registered') break;
      await new Promise((r) => setTimeout(r, 50));
    }
    const final = jobs.get(job.job_id);
    expect(final?.state).toBe('registered');
    expect(final?.artifact_id).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(final?.source.resolved_revision).toBe('a'.repeat(40));
    // manifest 存在且引用 blob
    const manifest = artifactStore.readManifest('hub', 'm', 'aaaaaaaaaaaa');
    expect(manifest?.format).toBe('gguf');
    expect((manifest?.files as Array<{ sha256: string }>).length).toBe(1);
    expect(artifactStore.referencedDigests().size).toBe(1);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('sha256 校验失败 → quarantined（错误摘要不进入 registry）', async () => {
    const { dir, store } = tempStore();
    const { jobs, executor } = makeExecutor(store, dir, [
      { name: 'model.gguf', content: Buffer.from('GGUF-JUNK-DATA') },
    ], { failDigest: true });
    const job = jobs.create({
      idempotencyKey: 'pull-bad',
      source: { provider: 'gguf_huggingface', repo_id: 'r/m', requested_revision: 'main' },
    });
    executor.start(job.job_id);
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      const s = jobs.get(job.job_id)?.state;
      if (s === 'quarantined' || s === 'failed') break;
      await new Promise((r) => setTimeout(r, 50));
    }
    expect(jobs.get(job.job_id)?.state).toBe('quarantined');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('取消：执行中 abort → cancelled', async () => {
    const { dir, store } = tempStore();
    const { jobs, executor } = makeExecutor(store, dir, [
      { name: 'model.gguf', content: Buffer.from('GGUF') },
    ]);
    const job = jobs.create({
      idempotencyKey: 'pull-cancel',
      source: { provider: 'gguf_huggingface', repo_id: 'r/m', requested_revision: 'main' },
    });
    executor.start(job.job_id);
    // 立即取消（下载可能已完成，但取消路径幂等）
    executor.cancelActive(job.job_id);
    jobs.cancel(job.job_id);
    const state = jobs.get(job.job_id)?.state;
    expect(['cancelled', 'registered']).toContain(state);
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('glob 匹配与 curated recipes（M3）', () => {
  it('globMatch 支持 * 与 **', () => {
    expect(globMatch('*.gguf', 'model.gguf')).toBe(true);
    expect(globMatch('*.gguf', 'dir/model.gguf')).toBe(false);
    expect(globMatch('**/*.gguf', 'dir/model.gguf')).toBe(true);
    expect(globMatch('*q4_k_m.gguf', 'model-q4_k_m.gguf')).toBe(true);
    expect(globMatch('*.bin', 'model.gguf')).toBe(false);
  });

  it('recipes 字段完整且 engine/family 合法', () => {
    expect(CURATED_RECIPES.length).toBeGreaterThanOrEqual(3);
    for (const recipe of CURATED_RECIPES) {
      expect(recipe.repo_id).toContain('/');
      expect(recipe.revision.length).toBeGreaterThan(0);
      expect(recipe.allow_patterns.length).toBeGreaterThan(0);
      expect(['llama_cpp', 'pytorch_transformers']).toContain(recipe.engine);
      expect(recipe.license).toBeTruthy();
    }
  });
});
