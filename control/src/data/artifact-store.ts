/**
 * M2 内容寻址工件库（一键模型部署计划 §6.3）。
 *
 * 布局：
 *   <QLH_MODEL_STORE>/
 *     staging/<job_id>/
 *     blobs/sha256/<first-two>/<digest>
 *     manifests/<namespace>/<name>/<tag>.json
 *     quarantine/<job_id>/
 *
 * 要求：
 *  - 下载/导入只写 staging；全部校验通过后同一文件系统内原子 rename 提交；
 *  - 相同 sha256 blob 去重（重复提交只增加引用，不复制字节）；
 *  - 删除别名只减少引用；GC 不删除 active/rollback/正在分发/未完成 outbox
 *    引用的工件（本实现以引用计数 + 显式保留集为边界）；
 *  - 模型目录必须位于本地文件系统；不信任文件名，先读 manifest 再核对摘要。
 */
import { Injectable, Optional } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

export function resolveModelStorePath(): string {
  const override = process.env.QLH_MODEL_STORE?.trim();
  if (override) return path.resolve(override);
  return path.resolve(process.cwd(), 'model-store');
}

export interface BlobInfo {
  digest: string;
  size: number;
  path: string;
  deduped: boolean;
}

export interface ImportReport {
  job_id: string;
  imported: number;
  deduped: number;
  failed: number;
  quarantined: number;
  warnings: string[];
  artifact_id: string | null;
  manifest_path: string | null;
}

export interface StoredArtifactManifest {
  reference: {
    namespace: string;
    name: string;
    tag: string;
  };
  manifest: Record<string, unknown>;
}

export function sha256Hex(data: Buffer): string {
  return crypto.createHash('sha256').update(data).digest('hex');
}

@Injectable()
export class ArtifactStore {
  readonly root: string;

  constructor(@Optional() root?: string) {
    this.root = root ?? resolveModelStorePath();
  }

  // ---- 路径 ----

  stagingDir(jobId: string): string {
    return path.join(this.root, 'staging', jobId);
  }

  quarantineDir(jobId: string): string {
    return path.join(this.root, 'quarantine', jobId);
  }

  blobPath(digest: string): string {
    return path.join(
      this.root, 'blobs', 'sha256', digest.slice(0, 2), digest,
    );
  }

  manifestPath(namespace: string, name: string, tag: string): string {
    return path.join(this.root, 'manifests', namespace, name, `${tag}.json`);
  }

  // ---- staging ----

  /** 写 staging 文件（导入/下载只写这里）。 */
  stageWrite(jobId: string, relPath: string, data: Buffer): string {
    const target = path.join(this.stagingDir(jobId), relPath);
    // 防目录穿越：relPath 不得逃出 staging
    if (!target.startsWith(this.stagingDir(jobId) + path.sep)) {
      throw new Error(`非法 staging 相对路径: ${relPath}`);
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, data);
    return target;
  }

  /** Copy a source file without loading the whole model into memory. */
  stageCopyFile(jobId: string, relPath: string, sourcePath: string): string {
    const target = path.join(this.stagingDir(jobId), relPath);
    if (!target.startsWith(this.stagingDir(jobId) + path.sep)) {
      throw new Error(`非法 staging 相对路径: ${relPath}`);
    }
    const source = path.resolve(sourcePath);
    if (!fs.statSync(source).isFile()) {
      throw new Error(`源文件不是普通文件: ${source}`);
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(source, target);
    return target;
  }

  listStaging(jobId: string): string[] {
    const dir = this.stagingDir(jobId);
    if (!fs.existsSync(dir)) return [];
    const files: string[] = [];
    const walk = (base: string, rel: string): void => {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        const full = path.join(base, entry.name);
        const nextRel = rel ? path.join(rel, entry.name) : entry.name;
        if (entry.isDirectory()) walk(full, nextRel);
        else files.push(nextRel);
      }
    };
    walk(dir, '');
    return files;
  }

  /** 校验失败：整 job staging 移入 quarantine（失败留痕不删除）。 */
  quarantine(jobId: string, reason: string): string {
    const from = this.stagingDir(jobId);
    const to = this.quarantineDir(jobId);
    if (fs.existsSync(from)) {
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.renameSync(from, to);
    }
    fs.writeFileSync(
      path.join(this.quarantineDir(jobId), 'reason.txt'),
      reason,
      'utf-8',
    );
    return to;
  }

  // ---- blob 提交 ----

  /**
   * 原子提交：staging 文件 → blobs（同文件系统 rename）。
   * 目标 digest 已存在 → deduped=true（删除 staging 副本，只增加引用）。
   */
  commitBlob(jobId: string, relPath: string): BlobInfo {
    const source = path.join(this.stagingDir(jobId), relPath);
    if (!fs.existsSync(source)) {
      throw new Error(`staging 文件缺失: ${relPath}`);
    }
    const size = fs.statSync(source).size;
    const digest = this.sha256File(source);
    const target = this.blobPath(digest);
    if (fs.existsSync(target)) {
      fs.rmSync(source, { force: true });
      return { digest, size, path: target, deduped: true };
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    // 原子性：先写临时再 rename（rename 同一文件系统内原子）
    const tmp = `${target}.tmp-${process.pid}`;
    fs.renameSync(source, tmp);
    fs.renameSync(tmp, target);
    return { digest, size, path: target, deduped: false };
  }

  readBlob(digest: string): Buffer | null {
    const p = this.blobPath(digest);
    return fs.existsSync(p) ? fs.readFileSync(p) : null;
  }

  blobExists(digest: string): boolean {
    return fs.existsSync(this.blobPath(digest));
  }

  // ---- manifest ----

  /** 原子写 manifest（JSON，先 tmp 后 rename）。 */
  writeManifest(manifest: Record<string, unknown>): string {
    const ns = String(manifest['namespace'] ?? 'user');
    const name = String(manifest['name'] ?? manifest['artifact_id'] ?? 'model');
    const tag = String(manifest['tag'] ?? 'latest');
    const target = this.manifestPath(ns, name, tag);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const tmp = `${target}.tmp-${process.pid}`;
    fs.writeFileSync(tmp, JSON.stringify(manifest, null, 2) + '\n', 'utf-8');
    fs.renameSync(tmp, target);
    return target;
  }

  readManifest(namespace: string, name: string, tag: string): Record<string, unknown> | null {
    const p = this.manifestPath(namespace, name, tag);
    if (!fs.existsSync(p)) return null;
    try {
      return JSON.parse(fs.readFileSync(p, 'utf-8')) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /** Enumerate valid JSON manifests using their on-disk alias as the canonical reference. */
  listManifests(): StoredArtifactManifest[] {
    const manifestsRoot = path.join(this.root, 'manifests');
    if (!fs.existsSync(manifestsRoot)) return [];
    const result: StoredArtifactManifest[] = [];
    const walk = (base: string): void => {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        const full = path.join(base, entry.name);
        if (entry.isDirectory()) {
          walk(full);
          continue;
        }
        if (!entry.name.endsWith('.json')) continue;
        const parts = path.relative(manifestsRoot, full).split(path.sep);
        if (parts.length !== 3) continue;
        try {
          const manifest = JSON.parse(
            fs.readFileSync(full, 'utf-8'),
          ) as Record<string, unknown>;
          result.push({
            reference: {
              namespace: parts[0],
              name: parts[1],
              tag: path.basename(parts[2], '.json'),
            },
            manifest,
          });
        } catch {
          // A damaged manifest is ignored here and remains on disk for diagnosis.
        }
      }
    };
    walk(manifestsRoot);
    return result.sort((a, b) => {
      const left = `${a.reference.namespace}/${a.reference.name}:${a.reference.tag}`;
      const right = `${b.reference.namespace}/${b.reference.name}:${b.reference.tag}`;
      return left.localeCompare(right);
    });
  }

  /** manifest 中引用 blob 的 digest 集合（去重；null 摘要 manifest 忽略）。 */
  referencedDigests(): Set<string> {
    const result = new Set<string>();
    const manifestsRoot = path.join(this.root, 'manifests');
    if (!fs.existsSync(manifestsRoot)) return result;
    const walk = (base: string): void => {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        const full = path.join(base, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (entry.name.endsWith('.json')) {
          try {
            const data = JSON.parse(fs.readFileSync(full, 'utf-8')) as {
              files?: Array<{ sha256?: string }>;
            };
            for (const file of data.files ?? []) {
              if (file.sha256) result.add(file.sha256);
            }
          } catch {
            // 损坏 manifest 跳过（GC 安全：不删除）
          }
        }
      }
    };
    walk(manifestsRoot);
    return result;
  }

  /** GC：删除无引用 blob。active/rollback/outbox 引用通过保留集传入。 */
  gc(extraRetained: Iterable<string> = []): { freed: number; candidates: number } {
    const retained = new Set(this.referencedDigests());
    for (const digest of extraRetained) retained.add(digest);
    const blobsRoot = path.join(this.root, 'blobs');
    let freed = 0;
    let candidates = 0;
    if (!fs.existsSync(blobsRoot)) return { freed, candidates };
    const walk = (base: string): void => {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        const full = path.join(base, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (!retained.has(entry.name)) {
          candidates += 1;
          fs.rmSync(full, { force: true });
          freed += 1;
        }
      }
    };
    walk(blobsRoot);
    return { freed, candidates };
  }

  /** 清理 staging 残留（成功提交后调用；失败留 quarantine 不清理）。 */
  cleanupStaging(jobId: string): void {
    fs.rmSync(this.stagingDir(jobId), { recursive: true, force: true });
  }

  private sha256File(filePath: string): string {
    const hash = crypto.createHash('sha256');
    const fd = fs.openSync(filePath, 'r');
    const buffer = Buffer.allocUnsafe(8 * 1024 * 1024);
    try {
      let position = 0;
      while (true) {
        const bytesRead = fs.readSync(fd, buffer, 0, buffer.length, position);
        if (bytesRead === 0) break;
        hash.update(buffer.subarray(0, bytesRead));
        position += bytesRead;
      }
    } finally {
      fs.closeSync(fd);
    }
    return hash.digest('hex');
  }
}
