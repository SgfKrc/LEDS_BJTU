import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import * as path from 'path';
import { SqliteStore } from './sqlite-store';

export interface ArtifactManifestIntegrity {
  ok: boolean;
  artifact_id: string;
  manifest_key: string;
  file_count: number;
  total_bytes: number;
  blob_digests: string[];
  errors: string[];
}

export interface ArtifactManifestRecord extends ArtifactManifestIntegrity {
  namespace: string;
  name: string;
  tag: string;
  manifest_path: string;
  manifest_sha256: string;
  created_at: string;
  updated_at: string;
  manifest: Record<string, unknown>;
}

export interface ArtifactReferenceCheck {
  ok: boolean;
  checked_manifests: number;
  checked_blobs: number;
  errors: string[];
}

interface ManifestFile {
  path?: unknown;
  size?: unknown;
  sha256?: unknown;
}

@Injectable()
export class ArtifactManifestRepository {
  constructor(private readonly store: SqliteStore) {}

  validate(
    manifest: Record<string, unknown>,
    blobExists?: (digest: string) => boolean,
  ): ArtifactManifestIntegrity {
    const errors: string[] = [];
    const namespace = String(manifest.namespace ?? '').trim();
    const name = String(manifest.name ?? '').trim();
    const tag = String(manifest.tag ?? '').trim();
    const artifactId = String(manifest.artifact_id ?? '').toLowerCase();
    const manifestKey = `${namespace}/${name}:${tag}`;
    if (!namespace || !name || !tag) errors.push('manifest reference is incomplete');
    for (const [field, value] of [['namespace', namespace], ['name', name], ['tag', tag]]) {
      if (value === '.' || value === '..' || value.includes('/') || value.includes('\\')
        || value.includes('\0') || value.length > 256) {
        errors.push(`manifest ${field} is unsafe`);
      }
    }
    if (!/^sha256:[0-9a-f]{64}$/.test(artifactId)) {
      errors.push('artifact_id must be a sha256 digest');
    }
    const files = Array.isArray(manifest.files) ? manifest.files as ManifestFile[] : [];
    if (files.length === 0) errors.push('manifest files are required');
    let totalBytes = 0;
    const digestEntries: Array<{ path: string; digest: string }> = [];
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      const relPath = String(file.path ?? '');
      const digest = String(file.sha256 ?? '').toLowerCase();
      const size = Number(file.size);
      const normalized = path.normalize(relPath);
      if (!relPath || path.isAbsolute(relPath) || normalized === '..'
        || normalized.startsWith(`..${path.sep}`)) {
        errors.push(`files[${index}].path is invalid`);
      }
      if (!/^[0-9a-f]{64}$/.test(digest)) {
        errors.push(`files[${index}].sha256 is invalid`);
      } else {
        digestEntries.push({ path: relPath, digest });
        if (blobExists && !blobExists(digest)) {
          errors.push(`files[${index}] blob is missing: ${digest}`);
        }
      }
      if (!Number.isSafeInteger(size) || size < 0) {
        errors.push(`files[${index}].size is invalid`);
      } else {
        totalBytes += size;
      }
    }
    const sortedDigests = digestEntries
      .sort((left, right) => left.path.localeCompare(right.path))
      .map((entry) => entry.digest);
    if (/^sha256:[0-9a-f]{64}$/.test(artifactId) && sortedDigests.length === files.length) {
      const aggregate = crypto.createHash('sha256')
        .update(sortedDigests.join(''))
        .digest('hex');
      if (artifactId !== `sha256:${aggregate}`) {
        errors.push('artifact_id does not match manifest file digests');
      }
    }
    return {
      ok: errors.length === 0,
      artifact_id: artifactId,
      manifest_key: manifestKey,
      file_count: files.length,
      total_bytes: totalBytes,
      blob_digests: sortedDigests,
      errors,
    };
  }

  register(
    manifest: Record<string, unknown>,
    manifestPath: string,
    integrity = this.validate(manifest),
  ): ArtifactManifestRecord {
    if (!integrity.ok) {
      throw new Error(`artifact manifest integrity failed: ${integrity.errors.join('; ')}`);
    }
    const namespace = String(manifest.namespace);
    const name = String(manifest.name);
    const tag = String(manifest.tag);
    const payload = JSON.stringify(manifest);
    const digest = crypto.createHash('sha256').update(payload).digest('hex');
    const now = new Date().toISOString();
    this.store.prepare(
      `INSERT INTO artifact_manifests
         (manifest_key, artifact_id, namespace, name, tag, manifest_path,
          manifest_sha256, file_count, total_bytes, blob_digests, payload,
          created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(manifest_key) DO UPDATE SET
         artifact_id = excluded.artifact_id,
         manifest_path = excluded.manifest_path,
         manifest_sha256 = excluded.manifest_sha256,
         file_count = excluded.file_count,
         total_bytes = excluded.total_bytes,
         blob_digests = excluded.blob_digests,
         payload = excluded.payload,
         updated_at = excluded.updated_at`,
    ).run(
      integrity.manifest_key,
      integrity.artifact_id,
      namespace,
      name,
      tag,
      path.resolve(manifestPath),
      digest,
      integrity.file_count,
      integrity.total_bytes,
      JSON.stringify(integrity.blob_digests),
      payload,
      now,
      now,
    );
    return this.get(integrity.manifest_key) as ArtifactManifestRecord;
  }

  get(manifestKey: string): ArtifactManifestRecord | null {
    const row = this.store.prepare(
      `SELECT manifest_key, artifact_id, namespace, name, tag, manifest_path,
              manifest_sha256, file_count, total_bytes, blob_digests, payload,
              created_at, updated_at
       FROM artifact_manifests WHERE manifest_key = ?`,
    ).get(manifestKey) as Record<string, unknown> | undefined;
    return row ? this.toRecord(row) : null;
  }

  list(): ArtifactManifestRecord[] {
    const rows = this.store.prepare(
      `SELECT manifest_key, artifact_id, namespace, name, tag, manifest_path,
              manifest_sha256, file_count, total_bytes, blob_digests, payload,
              created_at, updated_at
       FROM artifact_manifests ORDER BY manifest_key`,
    ).all() as Array<Record<string, unknown>>;
    return rows.map((row) => this.toRecord(row));
  }

  checkReferences(blobExists?: (digest: string) => boolean): ArtifactReferenceCheck {
    const records = this.list();
    const errors: string[] = [];
    let checkedBlobs = 0;
    for (const record of records) {
      const integrity = this.validate(record.manifest, blobExists);
      checkedBlobs += integrity.blob_digests.length;
      for (const error of integrity.errors) errors.push(`${record.manifest_key}: ${error}`);
      if (record.artifact_id !== integrity.artifact_id) {
        errors.push(`${record.manifest_key}: indexed artifact_id differs from payload`);
      }
    }
    const referenced = this.store.prepare(
      `SELECT DISTINCT artifact_id FROM deployments
       UNION SELECT DISTINCT artifact_id FROM artifact_runtime_checks`,
    ).all() as Array<{ artifact_id: string }>;
    const known = new Set(records.map((record) => record.artifact_id));
    for (const row of referenced) {
      if (row.artifact_id && !known.has(row.artifact_id)) {
        errors.push(`missing manifest for referenced artifact: ${row.artifact_id}`);
      }
    }
    return {
      ok: errors.length === 0,
      checked_manifests: records.length,
      checked_blobs: checkedBlobs,
      errors,
    };
  }

  private toRecord(row: Record<string, unknown>): ArtifactManifestRecord {
    const manifest = JSON.parse(String(row.payload)) as Record<string, unknown>;
    const blobDigests = JSON.parse(String(row.blob_digests)) as string[];
    return {
      ok: true,
      errors: [],
      manifest_key: String(row.manifest_key),
      artifact_id: String(row.artifact_id),
      namespace: String(row.namespace),
      name: String(row.name),
      tag: String(row.tag),
      manifest_path: String(row.manifest_path),
      manifest_sha256: String(row.manifest_sha256),
      file_count: Number(row.file_count),
      total_bytes: Number(row.total_bytes),
      blob_digests: blobDigests,
      created_at: String(row.created_at),
      updated_at: String(row.updated_at),
      manifest,
    };
  }
}
