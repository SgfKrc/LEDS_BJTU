import { createHash } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ArtifactManifestRepository } from '../src/data/artifact-manifest-repository';
import { ArtifactStore } from '../src/data/artifact-store';
import { SqliteStore } from '../src/data/sqlite-store';
import { StorageMigrationPackage } from '../src/data/storage-migration-package';

interface Fixture {
  store: SqliteStore;
  artifacts: ArtifactStore;
  repository: ArtifactManifestRepository;
  blobDigest: string;
  artifactId: string;
}

describe('M1.2 local asset migration package', () => {
  let tmpDir: string;
  let stores: SqliteStore[];

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-storage-package-'));
    stores = [];
  });

  afterEach(() => {
    jest.restoreAllMocks();
    for (const store of stores) store.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function trackedStore(filePath: string): SqliteStore {
    const store = new SqliteStore(filePath);
    stores.push(store);
    return store;
  }

  function createFixture(name: string): Fixture {
    const root = path.join(tmpDir, name);
    const store = trackedStore(path.join(root, 'control.sqlite3'));
    store.open();
    const repository = new ArtifactManifestRepository(store);
    const artifacts = new ArtifactStore(path.join(root, 'model-store'), repository);
    const content = Buffer.from(`GGUF-${name}-fixture`);
    artifacts.stageWrite('package-job', 'model.gguf', content);
    const blob = artifacts.commitBlob('package-job', 'model.gguf');
    const artifactId = `sha256:${createHash('sha256').update(blob.digest).digest('hex')}`;
    artifacts.writeManifest({
      namespace: 'user',
      name,
      tag: 'latest',
      artifact_id: artifactId,
      files: [{ path: 'model.gguf', size: blob.size, sha256: blob.digest }],
    });
    store.prepare(
      `INSERT INTO cluster_settings (key, value, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
    ).run('fixture_owner', name, new Date().toISOString());
    store.prepare(
      `INSERT INTO deployments
         (deployment_id, artifact_id, node_id, status, epoch, payload, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      `deployment-${name}`,
      artifactId,
      'local',
      'prepared',
      0,
      '{}',
      new Date().toISOString(),
    );
    return { store, artifacts, repository, blobDigest: blob.digest, artifactId };
  }

  it('流式导出、校验并恢复到全新 SQLite 和模型目录', async () => {
    const source = createFixture('source');
    const packagePath = path.join(tmpDir, 'source.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    const exporter = new StorageMigrationPackage(source.store, source.artifacts);
    const exported = await exporter.exportPackage(packagePath, passphrase);

    expect(exported).toMatchObject({
      version: 1,
      schema_version: 7,
      manifest_count: 1,
      blob_count: 1,
      reference_integrity: { ok: true, checked_manifests: 1, checked_blobs: 1 },
    });
    expect(exported.package_bytes).toBe(fs.statSync(packagePath).size);
    expect(exporter.verifyPackage(packagePath, passphrase)).toMatchObject({
      manifest_count: 1,
      blob_count: 1,
      reference_integrity: { ok: true },
    });

    const targetStore = trackedStore(path.join(tmpDir, 'restored', 'control.sqlite3'));
    const targetArtifacts = new ArtifactStore(path.join(tmpDir, 'restored', 'model-store'));
    const restored = new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase);

    expect(restored).toMatchObject({
      previous_sqlite_path: null,
      previous_model_store: null,
      manifest_count: 1,
      blob_count: 1,
    });
    expect(targetArtifacts.blobExists(source.blobDigest)).toBe(true);
    expect(targetArtifacts.readManifest('user', 'source', 'latest')).toMatchObject({
      artifact_id: source.artifactId,
    });
    const restoredRepository = new ArtifactManifestRepository(targetStore);
    expect(restoredRepository.get('user/source:latest')?.manifest_path).toBe(
      targetArtifacts.manifestPath('user', 'source', 'latest'),
    );
    expect(targetStore.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('fixture_owner')).toEqual({ value: 'source' });
  });

  it('篡改后的迁移包 fail-closed 且不修改现有资产', async () => {
    const source = createFixture('tamper-source');
    const packagePath = path.join(tmpDir, 'tampered.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    await new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(packagePath, passphrase);
    const packageBytes = fs.readFileSync(packagePath);
    const headerLength = packageBytes.readUInt32BE(8);
    const ciphertextOffset = 12 + headerLength;
    packageBytes[ciphertextOffset + 1] ^= 0xff;
    fs.writeFileSync(packagePath, packageBytes);

    const targetRoot = path.join(tmpDir, 'tamper-target');
    const targetStore = trackedStore(path.join(targetRoot, 'control.sqlite3'));
    targetStore.open();
    targetStore.prepare(
      'INSERT INTO cluster_settings (key, value, updated_at) VALUES (?, ?, ?)',
    ).run('fixture_owner', 'target', new Date().toISOString());
    const targetArtifacts = new ArtifactStore(path.join(targetRoot, 'model-store'));
    fs.mkdirSync(targetArtifacts.root, { recursive: true });
    fs.writeFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'keep');

    expect(() => new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase)).toThrow(/摘要校验失败|内容已损坏/);
    expect(fs.readFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'utf8')).toBe('keep');
    expect(targetStore.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('fixture_owner')).toEqual({ value: 'target' });
  });

  it('来源缺少被引用 blob 时拒绝生成迁移包', async () => {
    const source = createFixture('missing-blob');
    fs.rmSync(source.artifacts.blobPath(source.blobDigest));
    await expect(new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(
        path.join(tmpDir, 'missing.qlhmigrate'),
        'migration passphrase 123',
      )).rejects.toThrow('引用完整性检查失败');
  });

  it('磁盘空间门在解包前拒绝且保留目标目录', async () => {
    const source = createFixture('disk-source');
    const packagePath = path.join(tmpDir, 'disk.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    await new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(packagePath, passphrase);

    const targetRoot = path.join(tmpDir, 'disk-target');
    const targetStore = trackedStore(path.join(targetRoot, 'control.sqlite3'));
    const targetArtifacts = new ArtifactStore(path.join(targetRoot, 'model-store'));
    fs.mkdirSync(targetArtifacts.root, { recursive: true });
    fs.writeFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'keep');
    const migration = new StorageMigrationPackage(targetStore, targetArtifacts, {
      availableBytes: () => 0,
    });

    expect(() => migration.restorePackage(packagePath, passphrase)).toThrow('磁盘空间不足');
    expect(fs.readFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'utf8')).toBe('keep');
    expect(fs.existsSync(targetStore.filePath)).toBe(false);
  });

  it('SQLite 恢复阶段失败时回滚已切换的模型目录', async () => {
    const source = createFixture('rollback-source');
    const packagePath = path.join(tmpDir, 'rollback.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    await new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(packagePath, passphrase);

    const targetRoot = path.join(tmpDir, 'rollback-target');
    const targetStore = trackedStore(path.join(targetRoot, 'control.sqlite3'));
    targetStore.open();
    targetStore.prepare(
      'INSERT INTO cluster_settings (key, value, updated_at) VALUES (?, ?, ?)',
    ).run('fixture_owner', 'target', new Date().toISOString());
    const targetArtifacts = new ArtifactStore(path.join(targetRoot, 'model-store'));
    fs.mkdirSync(targetArtifacts.root, { recursive: true });
    fs.writeFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'keep');
    jest.spyOn(targetStore, 'restoreEncryptedBackup').mockImplementation(() => {
      throw new Error('injected sqlite restore failure');
    });

    expect(() => new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase)).toThrow('injected sqlite restore failure');
    expect(fs.readFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'utf8')).toBe('keep');
    expect(targetArtifacts.blobExists(source.blobDigest)).toBe(false);
    expect(targetStore.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('fixture_owner')).toEqual({ value: 'target' });
    expect(fs.readdirSync(targetRoot).some((name) => name.includes('.pre-restore-'))).toBe(false);
  });

  it('SQLite 已切换后 manifest reindex 失败时回滚两类资产', async () => {
    const source = createFixture('post-restore-source');
    const packagePath = path.join(tmpDir, 'post-restore.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    await new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(packagePath, passphrase);

    const targetRoot = path.join(tmpDir, 'post-restore-target');
    const targetStore = trackedStore(path.join(targetRoot, 'control.sqlite3'));
    targetStore.open();
    targetStore.prepare(
      'INSERT INTO cluster_settings (key, value, updated_at) VALUES (?, ?, ?)',
    ).run('fixture_owner', 'target', new Date().toISOString());
    const targetArtifacts = new ArtifactStore(path.join(targetRoot, 'model-store'));
    fs.mkdirSync(targetArtifacts.root, { recursive: true });
    fs.writeFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'keep');
    jest.spyOn(ArtifactStore.prototype, 'reindexManifests').mockReturnValueOnce({
      registered: 0,
      failed: 1,
      errors: ['injected reindex failure'],
    });

    expect(() => new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase)).toThrow('恢复后工件索引检查失败');
    expect(fs.readFileSync(path.join(targetArtifacts.root, 'sentinel.txt'), 'utf8')).toBe('keep');
    expect(targetArtifacts.blobExists(source.blobDigest)).toBe(false);
    expect(targetStore.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get('fixture_owner')).toEqual({ value: 'target' });
    expect(fs.readdirSync(targetRoot).some((name) => name.includes('.pre-restore-'))).toBe(false);
  });

  it('不可创建的模型目标父路径在恢复前被拒绝', async () => {
    const source = createFixture('blocked-source');
    const packagePath = path.join(tmpDir, 'blocked.qlhmigrate');
    const passphrase = 'migration passphrase 123';
    await new StorageMigrationPackage(source.store, source.artifacts)
      .exportPackage(packagePath, passphrase);

    const blocker = path.join(tmpDir, 'not-a-directory');
    fs.writeFileSync(blocker, 'blocked');
    const targetStore = trackedStore(path.join(tmpDir, 'blocked-target.sqlite3'));
    const targetArtifacts = new ArtifactStore(path.join(blocker, 'model-store'));
    expect(() => new StorageMigrationPackage(targetStore, targetArtifacts)
      .restorePackage(packagePath, passphrase)).toThrow();
    expect(fs.readFileSync(blocker, 'utf8')).toBe('blocked');
    expect(fs.existsSync(targetStore.filePath)).toBe(false);
  });
});
