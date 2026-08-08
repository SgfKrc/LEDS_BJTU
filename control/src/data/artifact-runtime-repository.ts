import { Injectable } from '@nestjs/common';
import { SqliteStore } from './sqlite-store';

export type RuntimeLoadStatus = 'ready' | 'load_failed' | 'resource_rejected';
export type ArtifactRuntimeStatus = RuntimeLoadStatus | 'stale';

export interface ArtifactRuntimeRecord {
  schema_version: 1;
  artifact_id: string;
  node_id: string;
  runtime_profile: string;
  status: ArtifactRuntimeStatus;
  checked_at: string;
  engine: string;
  loader_version: string | null;
  runtime_fingerprint: string;
  load_ms: number | null;
  details: Record<string, unknown>;
  error: { code: string; message: string } | null;
  invalidated_at?: string | null;
  invalidation_reason?: string | null;
}

export interface ArtifactRuntimeFilters {
  artifactId?: string;
  nodeId?: string;
  runtimeProfile?: string;
  status?: ArtifactRuntimeStatus;
}

@Injectable()
export class ArtifactRuntimeRepository {
  constructor(private readonly store: SqliteStore) {}

  upsert(record: ArtifactRuntimeRecord): ArtifactRuntimeRecord {
    this.store.prepare(
      `INSERT INTO artifact_runtime_checks
         (artifact_id, node_id, runtime_profile, status, payload, checked_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(artifact_id, node_id, runtime_profile) DO UPDATE SET
         status = excluded.status,
         payload = excluded.payload,
         checked_at = excluded.checked_at`,
    ).run(
      record.artifact_id,
      record.node_id,
      record.runtime_profile,
      record.status,
      JSON.stringify(record),
      record.checked_at,
    );
    return record;
  }

  get(
    artifactId: string,
    nodeId: string,
    runtimeProfile: string,
  ): ArtifactRuntimeRecord | null {
    const row = this.store.prepare(
      `SELECT payload FROM artifact_runtime_checks
       WHERE artifact_id = ? AND node_id = ? AND runtime_profile = ?`,
    ).get(artifactId, nodeId, runtimeProfile) as { payload: string } | undefined;
    return row ? JSON.parse(row.payload) as ArtifactRuntimeRecord : null;
  }

  list(input: string | ArtifactRuntimeFilters = {}): ArtifactRuntimeRecord[] {
    const filters = typeof input === 'string' ? { artifactId: input } : input;
    const clauses: string[] = [];
    const values: Array<string> = [];
    if (filters.artifactId) {
      clauses.push('artifact_id = ?');
      values.push(filters.artifactId);
    }
    if (filters.nodeId) {
      clauses.push('node_id = ?');
      values.push(filters.nodeId);
    }
    if (filters.runtimeProfile) {
      clauses.push('runtime_profile = ?');
      values.push(filters.runtimeProfile);
    }
    if (filters.status) {
      clauses.push('status = ?');
      values.push(filters.status);
    }
    const where = clauses.length > 0 ? ` WHERE ${clauses.join(' AND ')}` : '';
    const rows = this.store.prepare(
      `SELECT payload FROM artifact_runtime_checks${where} ORDER BY checked_at DESC`,
    ).all(...values) as Array<{ payload: string }>;
    return rows.map((row) => JSON.parse(row.payload) as ArtifactRuntimeRecord);
  }

  invalidate(
    filters: ArtifactRuntimeFilters,
    reason: string,
  ): ArtifactRuntimeRecord[] {
    const records = this.list(filters);
    const now = new Date().toISOString();
    return records.map((record) => this.upsert({
      ...record,
      status: 'stale',
      invalidated_at: now,
      invalidation_reason: reason,
    }));
  }
}
