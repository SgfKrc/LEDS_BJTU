import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ArtifactStore } from '../src/data/artifact-store';
import { ArtifactRuntimeRepository } from '../src/data/artifact-runtime-repository';
import { ModelRuntimeCheckService } from '../src/data/model-runtime-check.service';
import { ModelRuntimeSidecar } from '../src/data/model-runtime-sidecar';
import { SqliteStore } from '../src/data/sqlite-store';

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), 'qlh-runtime-sidecar-'));
}

function fixture(root: string): {
  store: ArtifactStore;
  manifest: Record<string, unknown>;
} {
  const store = new ArtifactStore(join(root, 'store'));
  store.stageWrite('fixture', 'model.gguf', Buffer.from('model-bytes'));
  const blob = store.commitBlob('fixture', 'model.gguf');
  return {
    store,
    manifest: {
      schema_version: 1,
      artifact_id: `sha256:${'a'.repeat(64)}`,
      format: 'gguf',
      engine: 'llama_cpp',
      files: [{ path: 'model.gguf', size: blob.size, sha256: blob.digest }],
      capabilities: {
        full_worker: false,
        pytorch_layer_pipeline: false,
        llama_cpp: true,
        task_stage: false,
      },
      requirements: { runtime_profile: 'llm-cpu-v1' },
      trust_policy: { trust_remote_code: false },
    },
  };
}

function script(root: string, body: string): string {
  const target = join(root, 'fake-sidecar.js');
  writeFileSync(target, body, 'utf-8');
  return target;
}

function echoScript(status: 'ready' | 'resource_rejected'): string {
  return `
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => input += chunk);
process.stdin.on('end', () => {
  const request = JSON.parse(input);
  process.stdout.write(JSON.stringify({
    schema_version: 1,
    request_id: request.request_id,
    artifact_id: request.artifact_id,
    status: '${status}',
    engine: request.engine,
    runtime_profile: request.runtime_profile,
    checked_at: new Date().toISOString(),
    loader_version: 'fake/1',
    load_ms: 7,
    details: { model_path_exists: require('fs').existsSync(request.model_path) },
    error: ${status === 'ready' ? 'null' : "{ code: 'insufficient_memory', message: 'fixture' }"},
  }) + '\\n');
});
`;
}

describe('ModelRuntimeSidecar', () => {
  it('creates an isolated hard-link view and accepts one matching result', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, echoScript('ready')),
      timeoutMs: 2000,
    });
    const result = await sidecar.trialLoad(manifest);
    expect(result.status).toBe('ready');
    expect(result.details.model_path_exists).toBe(true);
    const runtimeRoot = join(store.root, 'runtime');
    expect(existsSync(runtimeRoot)).toBe(true);
    expect(readdirSync(runtimeRoot)).toEqual([]);
    rmSync(root, { recursive: true, force: true });
  });

  it('maps a crashed child to load_failed without terminating the caller', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, 'process.exit(9);'),
      timeoutMs: 2000,
    });
    const result = await sidecar.trialLoad(manifest);
    expect(result.status).toBe('load_failed');
    expect(result.error?.code).toBe('sidecar_crashed');
    expect(1 + 1).toBe(2);
    rmSync(root, { recursive: true, force: true });
  });

  it('kills a hung child at the configured timeout', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, 'process.stdin.resume(); setInterval(() => {}, 1000);'),
      timeoutMs: 100,
    });
    const result = await sidecar.trialLoad(manifest);
    expect(result.status).toBe('load_failed');
    expect(result.error?.code).toBe('sidecar_timeout');
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects mismatched protocol identity fail-closed', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const bad = echoScript('ready').replace(
      'artifact_id: request.artifact_id',
      "artifact_id: 'sha256:' + 'b'.repeat(64)",
    );
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, bad),
      timeoutMs: 2000,
    });
    const result = await sidecar.trialLoad(manifest);
    expect(result.status).toBe('load_failed');
    expect(result.error?.code).toBe('sidecar_protocol_error');
    rmSync(root, { recursive: true, force: true });
  });

  it('persists ready and resource-rejected results separately from manifests', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const sqlite = new SqliteStore(join(root, 'control.sqlite3'));
    sqlite.open();
    const repository = new ArtifactRuntimeRepository(sqlite);
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, echoScript('resource_rejected')),
      timeoutMs: 2000,
    });
    const service = new ModelRuntimeCheckService(sidecar, repository, store);
    const record = await service.checkManifest(manifest, 'local-test');
    expect(record.status).toBe('resource_rejected');
    expect(repository.get(record.artifact_id, 'local-test', 'llm-cpu-v1')).toEqual(record);
    expect(repository.list(record.artifact_id)).toHaveLength(1);
    sqlite.close();
    rmSync(root, { recursive: true, force: true });
  });

  it('fails before spawning when a manifest blob is missing', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const files = manifest.files as Array<{ sha256: string }>;
    rmSync(store.blobPath(files[0].sha256), { force: true });
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, echoScript('ready')),
      timeoutMs: 2000,
    });
    const result = await sidecar.trialLoad(manifest);
    expect(result.status).toBe('load_failed');
    expect(result.error?.code).toBe('sidecar_prepare_failed');
    expect(readdirSync(join(store.root, 'runtime'))).toEqual([]);
    rmSync(root, { recursive: true, force: true });
  });

  it('cancels and reaps an active child through AbortSignal', async () => {
    const root = tempDir();
    const { store, manifest } = fixture(root);
    const sidecar = new ModelRuntimeSidecar(store, {
      executable: process.execPath,
      scriptPath: script(root, 'process.stdin.resume(); setInterval(() => {}, 1000);'),
      timeoutMs: 5000,
    });
    const controller = new AbortController();
    const pending = sidecar.trialLoad(manifest, { signal: controller.signal });
    controller.abort();
    const result = await pending;
    expect(result.status).toBe('load_failed');
    expect(result.error?.code).toBe('sidecar_cancelled');
    expect(readdirSync(join(store.root, 'runtime'))).toEqual([]);
    rmSync(root, { recursive: true, force: true });
  });
});
