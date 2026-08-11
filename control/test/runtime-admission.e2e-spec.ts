import { mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { FastifyAdapter } from '@nestjs/platform-fastify';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app';
import { JsonDetailFilter } from '../src/common/json-detail.filter';
import { RequestIdInterceptor } from '../src/common/request-id';
import { ArtifactStore } from '../src/data/artifact-store';
import { ModelRuntimeSidecar, RuntimeSidecarResult } from '../src/data/model-runtime-sidecar';
import { SqliteStore } from '../src/data/sqlite-store';

function ggufString(value: string): Buffer {
  const data = Buffer.from(value, 'utf-8');
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(data.length));
  return Buffer.concat([length, data]);
}

function qwenGguf(): Buffer {
  const type = Buffer.alloc(4);
  type.writeUInt32LE(8); // STRING
  const version = Buffer.alloc(4);
  version.writeUInt32LE(3);
  const tensorCount = Buffer.alloc(8);
  const metadataCount = Buffer.alloc(8);
  metadataCount.writeBigUInt64LE(1n);
  return Buffer.concat([
    Buffer.from('GGUF'), version, tensorCount, metadataCount,
    ggufString('general.architecture'), type, ggufString('qwen2'),
  ]);
}

describe('MODEL-FLEET runtime admission API', () => {
  let app: NestFastifyApplication;
  let root: string;
  let sqlite: SqliteStore;
  let artifacts: ArtifactStore;

  beforeEach(async () => {
    root = mkdtempSync(join(tmpdir(), 'qlh-runtime-admission-api-'));
    sqlite = new SqliteStore(join(root, 'control.sqlite3'));
    sqlite.open();
    artifacts = new ArtifactStore(join(root, 'artifacts'));
    process.env.QLH_NODE_ID = 'local-api';

    const fakeSidecar = {
      trialLoad: jest.fn(async (manifest: Record<string, unknown>): Promise<RuntimeSidecarResult> => {
        const requirements = manifest.requirements as Record<string, unknown>;
        return {
          schema_version: 1,
          request_id: 'fixture-request',
          artifact_id: String(manifest.artifact_id),
          status: 'ready',
          engine: String(manifest.engine),
          runtime_profile: String(requirements.runtime_profile),
          checked_at: new Date().toISOString(),
          loader_version: 'fixture-loader/1',
          load_ms: 2,
          details: { n_gpu_layers: 0, stderr_tail: 'local diagnostic' },
          error: null,
        };
      }),
    };
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(sqlite)
      .overrideProvider(ArtifactStore).useValue(artifacts)
      .overrideProvider(ModelRuntimeSidecar).useValue(fakeSidecar)
      .compile();
    app = moduleRef.createNestApplication(new FastifyAdapter());
    app.useGlobalFilters(new JsonDetailFilter());
    app.useGlobalInterceptors(new RequestIdInterceptor());
    await app.init();
    await app.getHttpAdapter().getInstance().ready();
  });

  afterEach(async () => {
    await app.close();
    sqlite.close();
    delete process.env.QLH_NODE_ID;
    rmSync(root, { recursive: true, force: true });
  });

  function createManifest(name = 'retry-model'): Record<string, unknown> {
    artifacts.stageWrite('fixture', 'model.gguf', qwenGguf());
    const blob = artifacts.commitBlob('fixture', 'model.gguf');
    const manifest = {
      schema_version: 1,
      namespace: 'user', name, tag: 'latest',
      artifact_id: `sha256:${'a'.repeat(64)}`,
      source: { provider: 'local_directory', requested_revision: 'local', resolved_revision: 'local' },
      format: 'gguf', engine: 'llama_cpp', family: 'qwen2',
      files: [{ path: 'model.gguf', size: blob.size, sha256: blob.digest }],
      capabilities: { full_worker: false, pytorch_layer_pipeline: false, llama_cpp: true, task_stage: false },
      requirements: { runtime_profile: 'llm-cpu-v1' },
      license: { id: 'unknown', acceptance_required: false },
      trust_policy: { trust_remote_code: false },
    };
    artifacts.writeManifest(manifest);
    return manifest;
  }

  it('retries, queries, and invalidates a local runtime check', async () => {
    const manifest = createManifest();
    const retried = await app.inject({
      method: 'POST', url: '/models/runtime-checks/retry',
      payload: { namespace: 'user', name: 'retry-model', tag: 'latest', node_id: 'local-api' },
    });
    expect(retried.statusCode).toBe(200);
    const runtimeCheck = retried.json().runtime_check;
    expect(runtimeCheck.status).toBe('ready');
    expect(runtimeCheck.runtime_fingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(retried.json().runnable).toBe(true);

    const listed = await app.inject({
      method: 'GET',
      url: `/models/runtime-checks?artifact_id=${encodeURIComponent(String(manifest.artifact_id))}`,
    });
    expect(listed.statusCode).toBe(200);
    expect(listed.json().runtime_checks).toHaveLength(1);
    expect(listed.json().runtime_checks[0].details.stderr_tail).toBeUndefined();

    const invalidated = await app.inject({
      method: 'DELETE',
      url: `/models/runtime-checks?artifact_id=${encodeURIComponent(String(manifest.artifact_id))}&reason=loader-upgraded`,
    });
    expect(invalidated.statusCode).toBe(200);
    expect(invalidated.json().count).toBe(1);
    expect(invalidated.json().runtime_checks[0].status).toBe('stale');
    expect(invalidated.json().runtime_checks[0].invalidation_reason).toBe('loader-upgraded');
  });

  it('summarizes artifact aliases, storage, and the latest local admission state', async () => {
    const manifest = createManifest('inventory-model');
    await app.inject({
      method: 'POST', url: '/models/runtime-checks/retry',
      payload: { namespace: 'user', name: 'inventory-model', tag: 'latest' },
    });

    const response = await app.inject({ method: 'GET', url: '/models/artifacts' });
    expect(response.statusCode).toBe(200);
    expect(response.json().node_id).toBe('local-api');
    expect(response.json().summary).toMatchObject({ total: 1, ready: 1, unchecked: 0 });
    expect(response.json().artifacts[0]).toMatchObject({
      artifact_id: manifest.artifact_id,
      reference: { namespace: 'user', name: 'inventory-model', tag: 'latest' },
      format: 'gguf',
      engine: 'llama_cpp',
      runnable: true,
      storage: { file_count: 1 },
      runtime_check: { status: 'ready', node_id: 'local-api' },
    });
    expect(response.json().artifacts[0].storage.total_bytes).toBeGreaterThan(0);
    expect(response.json().artifacts[0].runtime_check.details.stderr_tail).toBeUndefined();
  });

  it('imports a local model and runs admission before returning', async () => {
    const modelPath = join(root, 'local-qwen.gguf');
    writeFileSync(modelPath, qwenGguf());
    const response = await app.inject({
      method: 'POST', url: '/models/imports',
      payload: {
        source_path: modelPath,
        namespace: 'user', name: 'local-qwen', tag: 'checked', node_id: 'local-api',
      },
    });
    expect(response.statusCode).toBe(201);
    expect(response.json().status).toBe('imported');
    expect(response.json().runnable).toBe(true);
    expect(response.json().report.runtime_check.status).toBe('ready');
    expect(artifacts.readManifest('user', 'local-qwen', 'checked')).not.toBeNull();
  });

  it('rejects remote node impersonation and unscoped invalidation', async () => {
    createManifest();
    const retry = await app.inject({
      method: 'POST', url: '/models/runtime-checks/retry',
      payload: { namespace: 'user', name: 'retry-model', tag: 'latest', node_id: 'remote-node' },
    });
    expect(retry.statusCode).toBe(422);
    const invalidation = await app.inject({ method: 'DELETE', url: '/models/runtime-checks' });
    expect(invalidation.statusCode).toBe(422);
  });
});
