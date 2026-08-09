import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createHash } from 'crypto';
import { ArtifactManifestRepository } from '../src/data/artifact-manifest-repository';
import { ArtifactStore } from '../src/data/artifact-store';
import { ReviewStore } from '../src/data/review-store';
import { SessionStore } from '../src/data/session-store';
import { SqliteStore } from '../src/data/sqlite-store';
import {
  WorkflowJournalStore,
  WorkflowSnapshot,
} from '../src/data/workflow-journal-store';

function snapshot(workflowId: string, state: WorkflowSnapshot['state'] = 'running'): WorkflowSnapshot {
  return {
    workflow_id: workflowId,
    request_id: `req-${workflowId}`,
    session_id: 'session-1',
    model_identity: { model: 'test' },
    template: 'dual_candidate',
    state,
    last_sequence: 1,
    final_stage_id: 'stage-1',
    created_at: Date.now() / 1000,
    started_at: null,
    result_ready_at: null,
    finished_at: null,
    duration_seconds: 0,
    error: '',
    stage_count: 0,
    completed_stage_count: 0,
    failed_stage_count: 0,
    skipped_stage_count: 0,
    partial_result: false,
    cancelled_stage_count: 0,
    attempt_count: 0,
    retry_count: 0,
    same_provider_retry_count: 0,
    result_rejection_count: 0,
    cancel_requested: false,
    stages: [],
  };
}

describe('M1.2 SQLite core stores', () => {
  let tmpDir: string;
  let sqlitePath: string;
  let openedStores: SqliteStore[];

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-sqlite-core-'));
    sqlitePath = path.join(tmpDir, 'control.sqlite3');
    openedStores = [];
  });

  afterEach(() => {
    delete process.env.QLH_CHAT_HISTORY_DIR;
    delete process.env.QLH_WORKFLOW_JOURNAL_FILE;
    delete process.env.QLH_REVIEW_STORE;
    for (const store of openedStores) store.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function openStore(): SqliteStore {
    const store = new SqliteStore(sqlitePath);
    openedStores.push(store);
    return store;
  }

  it('会话、消息跨 SqliteStore 实例持久化并保留 metrics', () => {
    const store = openStore();
    const sessions = new SessionStore(store);
    const session = sessions.createSession('session-1', 'SQLite 会话');
    sessions.saveMessage(session.id, 'user', 'hello', { tokens: 3 });
    sessions.saveMessage(session.id, 'assistant', 'world');

    const reopened = new SessionStore(openStore());
    expect(reopened.getSession(session.id)?.title).toBe('SQLite 会话');
    expect(reopened.loadMessages(session.id, 0)).toEqual([
      { role: 'user', content: 'hello', created_at: expect.any(String), metrics: { tokens: 3 } },
      { role: 'assistant', content: 'world', created_at: expect.any(String) },
    ]);
    expect(reopened.stats()).toMatchObject({ backend: 'sqlite', session_count: 1, message_count: 2 });
  });

  it('首次打开 SQLite 时导入旧 JSON，并保留旧文件', () => {
    const legacyDir = path.join(tmpDir, 'chat_history');
    fs.mkdirSync(legacyDir, { recursive: true });
    fs.writeFileSync(path.join(legacyDir, '_sessions.json'), JSON.stringify([
      { id: 'legacy-1', title: '旧会话', created_at: '2026-01-01T00:00:00', updated_at: '2026-01-01T00:00:01', message_count: 1 },
    ]));
    fs.writeFileSync(path.join(legacyDir, 'legacy-1.json'), JSON.stringify([
      { role: 'user', content: '旧消息', created_at: '2026-01-01T00:00:02' },
    ]));
    process.env.QLH_CHAT_HISTORY_DIR = legacyDir;

    const sessions = new SessionStore(openStore());
    expect(sessions.getSession('legacy-1')?.title).toBe('旧会话');
    expect(sessions.loadMessages('legacy-1', 0)[0].content).toBe('旧消息');
    expect(fs.existsSync(path.join(legacyDir, '_sessions.json'))).toBe(true);
  });

  it('workflow journal 和 review ticket 默认写入 SQLite', () => {
    const store = openStore();
    const journal = new WorkflowJournalStore(store);
    journal.upsertSnapshot(snapshot('wf_sqlite01', 'completed'));
    expect(journal.status()).toMatchObject({ backend: 'sqlite', record_count: 1 });
    expect(journal.getSnapshot('wf_sqlite01')?.state).toBe('completed');

    const reviews = new ReviewStore(store);
    reviews.upsert({
      ticket_id: 'review_sqlite01',
      status: 'pending',
      created_at: 100,
      created_by: 'master',
      target_node_id: 'worker-1',
      transfer_reason: '',
      votes: [],
      score: 0,
      expires_at: 200,
      resolved_at: null,
      notification_sent: false,
    });
    expect(reviews.get('review_sqlite01')?.status).toBe('pending');
    expect(reviews.loadAll()).toHaveLength(1);
  });

  it('artifact manifest 校验后写入 SQLite v4 索引并可重建引用', () => {
    const store = openStore();
    const repository = new ArtifactManifestRepository(store);
    const artifacts = new ArtifactStore(path.join(tmpDir, 'model-store'), repository);
    artifacts.stageWrite('artifact-job', 'model.gguf', Buffer.from('GGUF-fixture'));
    const blob = artifacts.commitBlob('artifact-job', 'model.gguf');
    const aggregate = createHash('sha256').update(blob.digest).digest('hex');
    const manifest = {
      namespace: 'user',
      name: 'fixture',
      tag: 'latest',
      artifact_id: `sha256:${aggregate}`,
      files: [{ path: 'model.gguf', size: blob.size, sha256: blob.digest }],
    };

    artifacts.writeManifest(manifest);
    expect(repository.get('user/fixture:latest')).toMatchObject({
      artifact_id: manifest.artifact_id,
      file_count: 1,
      total_bytes: blob.size,
      blob_digests: [blob.digest],
    });
    store.prepare('DELETE FROM artifact_manifests').run();
    expect(artifacts.reindexManifests()).toMatchObject({ registered: 1, failed: 0 });
    fs.rmSync(artifacts.manifestPath('user', 'fixture', 'latest'));
    expect(artifacts.restoreIndexedManifests()).toMatchObject({ restored: 1, failed: 0 });
    expect(artifacts.readManifest('user', 'fixture', 'latest')).toMatchObject({
      artifact_id: manifest.artifact_id,
    });
    expect(repository.checkReferences((digest) => artifacts.blobExists(digest))).toMatchObject({
      ok: true,
      checked_manifests: 1,
      checked_blobs: 1,
    });
  });

  it('artifact manifest 缺少 blob 时 fail-closed 且不写文件/索引', () => {
    const store = openStore();
    const repository = new ArtifactManifestRepository(store);
    const artifacts = new ArtifactStore(path.join(tmpDir, 'model-store'), repository);
    const digest = 'a'.repeat(64);
    const aggregate = createHash('sha256').update(digest).digest('hex');
    expect(() => artifacts.writeManifest({
      namespace: 'user',
      name: 'missing',
      tag: 'latest',
      artifact_id: `sha256:${aggregate}`,
      files: [{ path: 'missing.bin', size: 1, sha256: digest }],
    })).toThrow('blob is missing');
    expect(repository.get('user/missing:latest')).toBeNull();
    expect(fs.existsSync(artifacts.manifestPath('user', 'missing', 'latest'))).toBe(false);
  });

  it('加密备份 dry-run 校验 artifact 引用，缺失 manifest 时拒绝导出', async () => {
    const store = openStore();
    const repository = new ArtifactManifestRepository(store);
    const artifacts = new ArtifactStore(path.join(tmpDir, 'model-store'), repository);
    artifacts.stageWrite('backup-job', 'model.gguf', Buffer.from('GGUF-backup'));
    const blob = artifacts.commitBlob('backup-job', 'model.gguf');
    const aggregate = createHash('sha256').update(blob.digest).digest('hex');
    const artifactId = `sha256:${aggregate}`;
    artifacts.writeManifest({
      namespace: 'user', name: 'backup', tag: 'latest', artifact_id: artifactId,
      files: [{ path: 'model.gguf', size: blob.size, sha256: blob.digest }],
    });
    store.prepare(
      `INSERT INTO deployments
         (deployment_id, artifact_id, node_id, status, epoch, payload, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run('dep_valid', artifactId, 'local', 'prepared', 0, '{}', new Date().toISOString());
    const backupPath = path.join(tmpDir, 'with-manifest.qlhbackup');
    const info = await store.exportEncryptedBackup(
      backupPath,
      'backup passphrase 123',
      (digest) => artifacts.blobExists(digest),
    );
    expect(info.reference_integrity).toMatchObject({
      ok: true,
      manifest_index: true,
      checked_manifests: 1,
      checked_references: 1,
      checked_blobs: 1,
      external_blobs_checked: true,
    });

    store.prepare(
      `INSERT INTO deployments
         (deployment_id, artifact_id, node_id, status, epoch, payload, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      'dep_missing',
      `sha256:${'f'.repeat(64)}`,
      'local',
      'prepared',
      0,
      '{}',
      new Date().toISOString(),
    );
    await expect(store.exportEncryptedBackup(
      path.join(tmpDir, 'invalid.qlhbackup'),
      'backup passphrase 123',
      (digest) => artifacts.blobExists(digest),
    )).rejects.toThrow('引用完整性检查失败');
  });
});
