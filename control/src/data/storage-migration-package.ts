import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
  scryptSync,
} from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  ArtifactManifestRecord,
  ArtifactManifestRepository,
} from './artifact-manifest-repository';
import { ArtifactStore, sha256FileHex } from './artifact-store';
import { ModelDiskBudget } from './model-disk-budget';
import {
  BackupReferenceIntegrity,
  EncryptedBackupInfo,
  RestoreResult,
  SqliteStore,
} from './sqlite-store';

const PACKAGE_MAGIC = Buffer.from('QLHMIG01', 'ascii');
const PACKAGE_VERSION = 1;
const PACKAGE_TAG_BYTES = 16;
const PACKAGE_KDF_N = 16_384;
const PACKAGE_KDF_R = 8;
const PACKAGE_KDF_P = 1;
const MAX_PACKAGE_HEADER_BYTES = 16 * 1024 * 1024;
const DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024;
const RESTORE_MARGIN_BYTES = 16 * 1024 * 1024;
const SQLITE_ENTRY_NAME = 'sqlite/control.qlhbackup';

interface StoragePackageEntry {
  kind: 'sqlite_backup' | 'blob';
  name: string;
  size: number;
  sha256: string;
  digest?: string;
}

interface StoragePackageHeader {
  version: number;
  cipher: 'aes-256-gcm';
  kdf: 'scrypt';
  kdf_n: number;
  kdf_r: number;
  kdf_p: number;
  salt: string;
  iv: string;
  created_at: string;
  schema_version: number;
  manifest_count: number;
  blob_count: number;
  plaintext_bytes: number;
  entries: StoragePackageEntry[];
}

interface PackageSourceEntry extends StoragePackageEntry {
  source_path: string;
}

interface ParsedPackage {
  header: StoragePackageHeader;
  header_bytes: Buffer;
  ciphertext_offset: number;
  auth_tag_offset: number;
  package_bytes: number;
}

export interface StorageMigrationPackageInfo {
  version: number;
  created_at: string;
  schema_version: number;
  manifest_count: number;
  blob_count: number;
  plaintext_bytes: number;
  ciphertext_bytes: number;
  package_bytes: number;
  package_path: string;
  reference_integrity: BackupReferenceIntegrity;
}

export interface StorageMigrationRestoreResult extends StorageMigrationPackageInfo {
  restored_sqlite_path: string;
  previous_sqlite_path: string | null;
  restored_model_store: string;
  previous_model_store: string | null;
}

export interface StorageMigrationPackageOptions {
  availableBytes?: (target: string) => number;
  chunkBytes?: number;
}

export class StorageMigrationPackage {
  private readonly availableBytes: (target: string) => number;
  private readonly chunkBytes: number;

  constructor(
    private readonly store: SqliteStore,
    private readonly artifacts: ArtifactStore,
    options: StorageMigrationPackageOptions = {},
  ) {
    const diskBudget = new ModelDiskBudget();
    this.availableBytes = options.availableBytes
      ?? ((target) => diskBudget.availableBytes(target));
    this.chunkBytes = options.chunkBytes ?? DEFAULT_CHUNK_BYTES;
    if (!Number.isSafeInteger(this.chunkBytes) || this.chunkBytes <= 0) {
      throw new Error('迁移包 chunkBytes 必须是正整数');
    }
  }

  async exportPackage(targetPath: string, passphrase: string): Promise<StorageMigrationPackageInfo> {
    this.requirePassphrase(passphrase);
    this.store.open();
    const target = path.resolve(targetPath);
    const targetDir = path.dirname(target);
    fs.mkdirSync(targetDir, { recursive: true });
    const buildRoot = fs.mkdtempSync(path.join(targetDir, '.qlh-package-build-'));
    const nestedBackup = path.join(buildRoot, 'control.qlhbackup');
    const tempPackage = path.join(buildRoot, 'package.tmp');
    try {
      const sqliteInfo = await this.store.exportEncryptedBackup(
        nestedBackup,
        passphrase,
        (digest) => this.artifacts.blobExists(digest),
      );
      const snapshotStore = new SqliteStore(path.join(buildRoot, 'snapshot.sqlite3'));
      let records: ArtifactManifestRecord[];
      try {
        snapshotStore.restoreEncryptedBackup(
          nestedBackup,
          passphrase,
          (digest) => this.artifacts.blobExists(digest),
        );
        const snapshotRepository = new ArtifactManifestRepository(snapshotStore);
        const references = snapshotRepository.checkReferences(
          (digest) => this.artifacts.blobExists(digest),
        );
        if (!references.ok) {
          throw new Error(`迁移包引用完整性检查失败: ${references.errors.join('; ')}`);
        }
        records = snapshotRepository.list();
      } finally {
        snapshotStore.close();
      }
      const expectedBlobSizes = this.expectedBlobSizes(records);
      const digests = [...new Set(records.flatMap((record) => record.blob_digests))].sort();
      const sources: PackageSourceEntry[] = [this.sourceEntry(
        'sqlite_backup',
        SQLITE_ENTRY_NAME,
        nestedBackup,
      )];
      for (const digest of digests) {
        const source = this.artifacts.blobPath(digest);
        const entry = this.sourceEntry(
          'blob',
          `model-store/blobs/sha256/${digest.slice(0, 2)}/${digest}`,
          source,
          digest,
        );
        if (entry.sha256 !== digest) {
          throw new Error(`模型 blob 摘要不匹配: ${digest}`);
        }
        if (entry.size !== expectedBlobSizes.get(digest)) {
          throw new Error(`模型 blob 大小与 manifest 不匹配: ${digest}`);
        }
        sources.push(entry);
      }
      const plaintextBytes = this.sumSizes(sources);
      const header: StoragePackageHeader = {
        version: PACKAGE_VERSION,
        cipher: 'aes-256-gcm',
        kdf: 'scrypt',
        kdf_n: PACKAGE_KDF_N,
        kdf_r: PACKAGE_KDF_R,
        kdf_p: PACKAGE_KDF_P,
        salt: randomBytes(16).toString('base64'),
        iv: randomBytes(12).toString('base64'),
        created_at: new Date().toISOString(),
        schema_version: sqliteInfo.schema_version,
        manifest_count: records.length,
        blob_count: digests.length,
        plaintext_bytes: plaintextBytes,
        entries: sources.map(({ source_path: _sourcePath, ...entry }) => entry),
      };
      const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
      if (headerBytes.length > MAX_PACKAGE_HEADER_BYTES) {
        throw new Error('迁移包头超过大小限制');
      }
      const estimatedBytes = PACKAGE_MAGIC.length + 4 + headerBytes.length
        + plaintextBytes + PACKAGE_TAG_BYTES;
      if (this.availableBytes(targetDir) < estimatedBytes) {
        throw new Error(`迁移包目标磁盘空间不足，需要至少 ${estimatedBytes} bytes`);
      }
      this.writeEncryptedPackage(tempPackage, header, headerBytes, sources, passphrase);
      this.replaceFile(tempPackage, target);
      return this.toInfo(target, header, sqliteInfo.reference_integrity, estimatedBytes);
    } finally {
      fs.rmSync(buildRoot, { recursive: true, force: true });
    }
  }

  verifyPackage(packagePath: string, passphrase: string): StorageMigrationPackageInfo {
    this.requirePassphrase(passphrase);
    const absolutePackage = path.resolve(packagePath);
    const parsed = this.readPackageHeader(absolutePackage);
    const extractRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'qlh-package-verify-'));
    try {
      this.extractPackage(absolutePackage, parsed, extractRoot, passphrase);
      const sqliteInfo = this.verifyExtractedPackage(extractRoot, parsed.header, passphrase);
      return this.toInfo(
        absolutePackage,
        parsed.header,
        sqliteInfo.reference_integrity,
        parsed.package_bytes,
      );
    } finally {
      fs.rmSync(extractRoot, { recursive: true, force: true });
    }
  }

  restorePackage(packagePath: string, passphrase: string): StorageMigrationRestoreResult {
    this.requirePassphrase(passphrase);
    const absolutePackage = path.resolve(packagePath);
    const parsed = this.readPackageHeader(absolutePackage);
    const modelRoot = this.assertManagedModelRoot(this.artifacts.root);
    const modelParent = path.dirname(modelRoot);
    const sqliteParent = path.dirname(this.store.filePath);
    this.ensureWritableDirectory(modelParent);
    this.ensureWritableDirectory(sqliteParent);
    const sqliteEntry = parsed.header.entries.find((entry) => entry.kind === 'sqlite_backup');
    if (!sqliteEntry) throw new Error('迁移包缺少 SQLite 备份');
    const modelRequired = parsed.header.plaintext_bytes
      + (sqliteEntry.size * 2) + RESTORE_MARGIN_BYTES;
    if (this.availableBytes(modelParent) < modelRequired) {
      throw new Error(`模型目录磁盘空间不足，需要至少 ${modelRequired} bytes`);
    }
    const sqliteRequired = (sqliteEntry.size * 2) + RESTORE_MARGIN_BYTES;
    if (path.parse(modelParent).root !== path.parse(sqliteParent).root
      && this.availableBytes(sqliteParent) < sqliteRequired) {
      throw new Error(`SQLite 目录磁盘空间不足，需要至少 ${sqliteRequired} bytes`);
    }

    const extractRoot = fs.mkdtempSync(path.join(modelParent, '.qlh-package-restore-'));
    let previousModelStore: string | null = null;
    let installedModelStore = false;
    let sqliteRestore: RestoreResult | null = null;
    try {
      this.extractPackage(absolutePackage, parsed, extractRoot, passphrase);
      const verified = this.verifyExtractedPackage(extractRoot, parsed.header, passphrase);
      const stagedModelStore = path.join(extractRoot, 'model-store');
      if (!fs.existsSync(stagedModelStore)) fs.mkdirSync(stagedModelStore, { recursive: true });
      if (fs.existsSync(modelRoot)) {
        if (!fs.statSync(modelRoot).isDirectory()) {
          throw new Error(`模型存储目标不是目录: ${modelRoot}`);
        }
        previousModelStore = `${modelRoot}.pre-restore-${Date.now()}-${randomBytes(4).toString('hex')}`;
        fs.renameSync(modelRoot, previousModelStore);
      }
      fs.renameSync(stagedModelStore, modelRoot);
      installedModelStore = true;

      const nestedBackup = path.join(extractRoot, ...SQLITE_ENTRY_NAME.split('/'));
      const installedArtifacts = new ArtifactStore(modelRoot);
      sqliteRestore = this.store.restoreEncryptedBackup(
        nestedBackup,
        passphrase,
        (digest) => installedArtifacts.blobExists(digest),
      );
      const installedRepository = new ArtifactManifestRepository(this.store);
      const indexedArtifacts = new ArtifactStore(modelRoot, installedRepository);
      const reindex = indexedArtifacts.reindexManifests();
      const finalReferences = installedRepository.checkReferences(
        (digest) => indexedArtifacts.blobExists(digest),
      );
      if (reindex.failed > 0 || !finalReferences.ok) {
        throw new Error(
          `恢复后工件索引检查失败: ${[...reindex.errors, ...finalReferences.errors].join('; ')}`,
        );
      }
      return {
        ...this.toInfo(
          absolutePackage,
          parsed.header,
          verified.reference_integrity,
          parsed.package_bytes,
        ),
        restored_sqlite_path: sqliteRestore.restored_path,
        previous_sqlite_path: sqliteRestore.previous_path,
        restored_model_store: modelRoot,
        previous_model_store: previousModelStore,
      };
    } catch (err) {
      const rollbackErrors: string[] = [];
      if (sqliteRestore) {
        try {
          this.rollbackSqliteRestore(sqliteRestore);
        } catch (rollbackErr) {
          rollbackErrors.push(
            `SQLite rollback: ${rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)}`,
          );
        }
      }
      try {
        if (installedModelStore && fs.existsSync(modelRoot)) {
          fs.rmSync(modelRoot, { recursive: true, force: true });
        }
        if (previousModelStore && fs.existsSync(previousModelStore)) {
          fs.renameSync(previousModelStore, modelRoot);
        }
      } catch (rollbackErr) {
        rollbackErrors.push(
          `model rollback: ${rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)}`,
        );
      }
      if (rollbackErrors.length > 0) {
        const message = err instanceof Error ? err.message : String(err);
        throw new Error(`${message}; rollback failed: ${rollbackErrors.join('; ')}`);
      }
      throw err;
    } finally {
      fs.rmSync(extractRoot, { recursive: true, force: true });
    }
  }

  private sourceEntry(
    kind: StoragePackageEntry['kind'],
    name: string,
    sourcePath: string,
    digest?: string,
  ): PackageSourceEntry {
    const source = path.resolve(sourcePath);
    const stat = fs.lstatSync(source);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      throw new Error(`迁移包来源不是普通文件: ${source}`);
    }
    return {
      kind,
      name,
      size: stat.size,
      sha256: sha256FileHex(source),
      ...(digest ? { digest } : {}),
      source_path: source,
    };
  }

  private writeEncryptedPackage(
    targetPath: string,
    header: StoragePackageHeader,
    headerBytes: Buffer,
    sources: PackageSourceEntry[],
    passphrase: string,
  ): void {
    const fd = fs.openSync(targetPath, 'wx', 0o600);
    const prefix = Buffer.allocUnsafe(PACKAGE_MAGIC.length + 4);
    PACKAGE_MAGIC.copy(prefix, 0);
    prefix.writeUInt32BE(headerBytes.length, PACKAGE_MAGIC.length);
    const cipher = createCipheriv(
      'aes-256-gcm',
      this.deriveKey(passphrase, Buffer.from(header.salt, 'base64')),
      Buffer.from(header.iv, 'base64'),
    );
    cipher.setAAD(headerBytes);
    try {
      this.writeAll(fd, prefix);
      this.writeAll(fd, headerBytes);
      const buffer = Buffer.allocUnsafe(this.chunkBytes);
      for (const source of sources) {
        const sourceFd = fs.openSync(source.source_path, 'r');
        const hash = createHash('sha256');
        try {
          let position = 0;
          while (position < source.size) {
            const length = Math.min(buffer.length, source.size - position);
            const bytesRead = fs.readSync(sourceFd, buffer, 0, length, position);
            if (bytesRead <= 0) throw new Error(`迁移包来源读取提前结束: ${source.name}`);
            hash.update(buffer.subarray(0, bytesRead));
            this.writeAll(fd, cipher.update(buffer.subarray(0, bytesRead)));
            position += bytesRead;
          }
        } finally {
          fs.closeSync(sourceFd);
        }
        if (hash.digest('hex') !== source.sha256) {
          throw new Error(`迁移包来源在导出期间发生变化: ${source.name}`);
        }
      }
      this.writeAll(fd, cipher.final());
      this.writeAll(fd, cipher.getAuthTag());
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
  }

  private readPackageHeader(packagePath: string): ParsedPackage {
    const stat = fs.statSync(packagePath);
    if (!stat.isFile()) throw new Error('迁移包不是普通文件');
    const fd = fs.openSync(packagePath, 'r');
    try {
      const prefix = this.readExact(fd, PACKAGE_MAGIC.length + 4, 0);
      if (!prefix.subarray(0, PACKAGE_MAGIC.length).equals(PACKAGE_MAGIC)) {
        throw new Error('不是有效的 QLH 本地资产迁移包');
      }
      const headerLength = prefix.readUInt32BE(PACKAGE_MAGIC.length);
      if (headerLength <= 0 || headerLength > MAX_PACKAGE_HEADER_BYTES) {
        throw new Error('迁移包头大小无效');
      }
      const headerBytes = this.readExact(fd, headerLength, prefix.length);
      let header: StoragePackageHeader;
      try {
        header = JSON.parse(headerBytes.toString('utf8')) as StoragePackageHeader;
      } catch {
        throw new Error('迁移包头无法解析');
      }
      this.validateHeader(header);
      const ciphertextOffset = prefix.length + headerLength;
      const authTagOffset = ciphertextOffset + header.plaintext_bytes;
      const expectedSize = authTagOffset + PACKAGE_TAG_BYTES;
      if (stat.size !== expectedSize) {
        throw new Error(`迁移包长度不匹配: 期望 ${expectedSize}，实际 ${stat.size}`);
      }
      return {
        header,
        header_bytes: headerBytes,
        ciphertext_offset: ciphertextOffset,
        auth_tag_offset: authTagOffset,
        package_bytes: stat.size,
      };
    } finally {
      fs.closeSync(fd);
    }
  }

  private validateHeader(header: StoragePackageHeader): void {
    if (header.version !== PACKAGE_VERSION || header.cipher !== 'aes-256-gcm'
      || header.kdf !== 'scrypt' || header.kdf_n !== PACKAGE_KDF_N
      || header.kdf_r !== PACKAGE_KDF_R || header.kdf_p !== PACKAGE_KDF_P) {
      throw new Error('不支持的迁移包格式');
    }
    if (!Number.isSafeInteger(header.schema_version) || header.schema_version < 0
      || !Number.isSafeInteger(header.manifest_count) || header.manifest_count < 0
      || !Number.isSafeInteger(header.blob_count) || header.blob_count < 0
      || !Array.isArray(header.entries)) {
      throw new Error('迁移包元数据无效');
    }
    const names = new Set<string>();
    let sqliteEntries = 0;
    let blobEntries = 0;
    for (const entry of header.entries) {
      if (!entry || !['sqlite_backup', 'blob'].includes(entry.kind)
        || !Number.isSafeInteger(entry.size) || entry.size < 0
        || !/^[0-9a-f]{64}$/.test(entry.sha256)
        || !this.isSafeEntryName(entry.name) || names.has(entry.name)) {
        throw new Error('迁移包条目元数据无效');
      }
      names.add(entry.name);
      if (entry.kind === 'sqlite_backup') {
        sqliteEntries += 1;
        if (entry.name !== SQLITE_ENTRY_NAME || entry.digest !== undefined) {
          throw new Error('迁移包 SQLite 条目无效');
        }
      } else {
        blobEntries += 1;
        const digest = String(entry.digest ?? '').toLowerCase();
        const expectedName = `model-store/blobs/sha256/${digest.slice(0, 2)}/${digest}`;
        if (!/^[0-9a-f]{64}$/.test(digest) || entry.sha256 !== digest
          || entry.name !== expectedName) {
          throw new Error('迁移包 blob 条目无效');
        }
      }
    }
    if (sqliteEntries !== 1 || blobEntries !== header.blob_count) {
      throw new Error('迁移包条目数量与元数据不一致');
    }
    const total = this.sumSizes(header.entries);
    if (total !== header.plaintext_bytes) {
      throw new Error('迁移包明文大小与条目不一致');
    }
    try {
      if (Buffer.from(header.salt, 'base64').length !== 16
        || Buffer.from(header.iv, 'base64').length !== 12) {
        throw new Error();
      }
    } catch {
      throw new Error('迁移包加密参数无效');
    }
  }

  private extractPackage(
    packagePath: string,
    parsed: ParsedPackage,
    extractRoot: string,
    passphrase: string,
  ): void {
    const fd = fs.openSync(packagePath, 'r');
    try {
      const authTag = this.readExact(fd, PACKAGE_TAG_BYTES, parsed.auth_tag_offset);
      const decipher = createDecipheriv(
        'aes-256-gcm',
        this.deriveKey(passphrase, Buffer.from(parsed.header.salt, 'base64')),
        Buffer.from(parsed.header.iv, 'base64'),
      );
      decipher.setAAD(parsed.header_bytes);
      decipher.setAuthTag(authTag);
      const buffer = Buffer.allocUnsafe(this.chunkBytes);
      let ciphertextPosition = parsed.ciphertext_offset;
      for (const entry of parsed.header.entries) {
        const target = this.safeEntryPath(extractRoot, entry.name);
        fs.mkdirSync(path.dirname(target), { recursive: true });
        const targetFd = fs.openSync(target, 'wx', 0o600);
        const hash = createHash('sha256');
        try {
          let remaining = entry.size;
          while (remaining > 0) {
            const length = Math.min(buffer.length, remaining);
            const bytesRead = fs.readSync(fd, buffer, 0, length, ciphertextPosition);
            if (bytesRead <= 0) throw new Error(`迁移包条目读取提前结束: ${entry.name}`);
            const plaintext = decipher.update(buffer.subarray(0, bytesRead));
            this.writeAll(targetFd, plaintext);
            hash.update(plaintext);
            ciphertextPosition += bytesRead;
            remaining -= bytesRead;
          }
          fs.fsyncSync(targetFd);
        } finally {
          fs.closeSync(targetFd);
        }
        if (hash.digest('hex') !== entry.sha256) {
          throw new Error(`迁移包条目摘要校验失败: ${entry.name}`);
        }
      }
      try {
        const final = decipher.final();
        if (final.length > 0) throw new Error('迁移包尾部包含未声明内容');
      } catch {
        throw new Error('迁移包口令错误或内容已损坏');
      }
    } finally {
      fs.closeSync(fd);
    }
  }

  private verifyExtractedPackage(
    extractRoot: string,
    header: StoragePackageHeader,
    passphrase: string,
  ): EncryptedBackupInfo {
    const nestedBackup = path.join(extractRoot, ...SQLITE_ENTRY_NAME.split('/'));
    const stagedModelStore = path.join(extractRoot, 'model-store');
    fs.mkdirSync(stagedModelStore, { recursive: true });
    const stagedArtifacts = new ArtifactStore(stagedModelStore);
    const verifyDir = path.join(extractRoot, '.verify');
    const verifyStore = new SqliteStore(path.join(verifyDir, 'control.sqlite3'));
    try {
      const sqliteRestore = verifyStore.restoreEncryptedBackup(
        nestedBackup,
        passphrase,
        (digest) => stagedArtifacts.blobExists(digest),
      );
      if (sqliteRestore.schema_version !== header.schema_version) {
        throw new Error('迁移包 schema_version 与 SQLite 备份不一致');
      }
      const repository = new ArtifactManifestRepository(verifyStore);
      const records = repository.list();
      if (records.length !== header.manifest_count) {
        throw new Error('迁移包 manifest 数量与 SQLite 索引不一致');
      }
      const references = repository.checkReferences(
        (digest) => stagedArtifacts.blobExists(digest),
      );
      if (!references.ok) {
        throw new Error(`迁移包工件引用检查失败: ${references.errors.join('; ')}`);
      }
      const expectedBlobSizes = this.expectedBlobSizes(records);
      for (const [digest, expectedSize] of expectedBlobSizes) {
        const actualSize = fs.statSync(stagedArtifacts.blobPath(digest)).size;
        if (actualSize !== expectedSize) {
          throw new Error(`迁移包 blob 大小与 manifest 不匹配: ${digest}`);
        }
      }
      const indexedArtifacts = new ArtifactStore(stagedModelStore, repository);
      const restored = indexedArtifacts.restoreIndexedManifests();
      if (restored.failed > 0 || restored.restored + restored.unchanged !== records.length) {
        throw new Error(`迁移包 manifest 重建失败: ${restored.errors.join('; ')}`);
      }
      return sqliteRestore;
    } finally {
      verifyStore.close();
      fs.rmSync(verifyDir, { recursive: true, force: true });
    }
  }

  private rollbackSqliteRestore(restored: RestoreResult): void {
    this.store.close();
    this.removeSqliteFiles(restored.restored_path);
    if (restored.previous_path && fs.existsSync(restored.previous_path)) {
      this.moveSqliteFiles(restored.previous_path, restored.restored_path);
      this.store.open();
    }
  }

  private toInfo(
    packagePath: string,
    header: StoragePackageHeader,
    referenceIntegrity: BackupReferenceIntegrity,
    packageBytes: number,
  ): StorageMigrationPackageInfo {
    return {
      version: header.version,
      created_at: header.created_at,
      schema_version: header.schema_version,
      manifest_count: header.manifest_count,
      blob_count: header.blob_count,
      plaintext_bytes: header.plaintext_bytes,
      ciphertext_bytes: header.plaintext_bytes,
      package_bytes: packageBytes,
      package_path: path.resolve(packagePath),
      reference_integrity: referenceIntegrity,
    };
  }

  private deriveKey(passphrase: string, salt: Buffer): Buffer {
    return scryptSync(passphrase, salt, 32, {
      N: PACKAGE_KDF_N,
      r: PACKAGE_KDF_R,
      p: PACKAGE_KDF_P,
      maxmem: 64 * 1024 * 1024,
    });
  }

  private requirePassphrase(passphrase: string): void {
    if (typeof passphrase !== 'string' || passphrase.length < 12) {
      throw new Error('迁移包口令至少需要 12 个字符');
    }
  }

  private sumSizes(entries: Array<{ size: number }>): number {
    let total = 0;
    for (const entry of entries) {
      total += entry.size;
      if (!Number.isSafeInteger(total)) throw new Error('迁移包总大小超过安全整数范围');
    }
    return total;
  }

  private expectedBlobSizes(records: ArtifactManifestRecord[]): Map<string, number> {
    const sizes = new Map<string, number>();
    for (const record of records) {
      const files = Array.isArray(record.manifest.files)
        ? record.manifest.files as Array<{ sha256?: unknown; size?: unknown }>
        : [];
      for (const file of files) {
        const digest = String(file.sha256 ?? '').toLowerCase();
        const size = Number(file.size);
        if (!/^[0-9a-f]{64}$/.test(digest) || !Number.isSafeInteger(size) || size < 0) {
          throw new Error(`manifest blob 元数据无效: ${record.manifest_key}`);
        }
        const previous = sizes.get(digest);
        if (previous !== undefined && previous !== size) {
          throw new Error(`同一 blob 存在冲突大小: ${digest}`);
        }
        sizes.set(digest, size);
      }
    }
    return sizes;
  }

  private isSafeEntryName(name: string): boolean {
    return Boolean(name)
      && !name.includes('\\')
      && !name.includes('\0')
      && !name.startsWith('/')
      && !name.split('/').some((part) => !part || part === '.' || part === '..');
  }

  private safeEntryPath(root: string, name: string): string {
    if (!this.isSafeEntryName(name)) throw new Error(`迁移包条目路径不安全: ${name}`);
    const resolvedRoot = path.resolve(root);
    const target = path.resolve(resolvedRoot, ...name.split('/'));
    if (!target.startsWith(`${resolvedRoot}${path.sep}`)) {
      throw new Error(`迁移包条目逃出 staging: ${name}`);
    }
    return target;
  }

  private assertManagedModelRoot(root: string): string {
    const resolved = path.resolve(root);
    if (resolved === path.parse(resolved).root) {
      throw new Error('模型存储不能指向文件系统根目录');
    }
    return resolved;
  }

  private ensureWritableDirectory(directory: string): void {
    fs.mkdirSync(directory, { recursive: true });
    const probe = path.join(directory, `.qlh-write-probe-${process.pid}-${randomBytes(4).toString('hex')}`);
    try {
      fs.writeFileSync(probe, 'ok', { flag: 'wx', mode: 0o600 });
    } finally {
      fs.rmSync(probe, { force: true });
    }
  }

  private replaceFile(source: string, target: string): void {
    const previous = fs.existsSync(target)
      ? `${target}.pre-write-${Date.now()}-${randomBytes(4).toString('hex')}`
      : null;
    try {
      if (previous) fs.renameSync(target, previous);
      fs.renameSync(source, target);
      if (previous) fs.rmSync(previous, { force: true });
    } catch (err) {
      if (fs.existsSync(target) && source !== target) fs.rmSync(target, { force: true });
      if (previous && fs.existsSync(previous)) fs.renameSync(previous, target);
      throw err;
    }
  }

  private readExact(fd: number, length: number, position: number): Buffer {
    const result = Buffer.allocUnsafe(length);
    let offset = 0;
    while (offset < length) {
      const bytesRead = fs.readSync(fd, result, offset, length - offset, position + offset);
      if (bytesRead <= 0) throw new Error('迁移包内容不完整');
      offset += bytesRead;
    }
    return result;
  }

  private writeAll(fd: number, data: Buffer): void {
    let offset = 0;
    while (offset < data.length) {
      offset += fs.writeSync(fd, data, offset, data.length - offset);
    }
  }

  private removeSqliteFiles(filePath: string): void {
    for (const suffix of ['', '-wal', '-shm']) {
      fs.rmSync(`${filePath}${suffix}`, { force: true });
    }
  }

  private moveSqliteFiles(source: string, target: string): void {
    for (const suffix of ['', '-wal', '-shm']) {
      const from = `${source}${suffix}`;
      if (fs.existsSync(from)) fs.renameSync(from, `${target}${suffix}`);
    }
  }
}
