/**
 * M4 cluster profile repository — 多集群档案（SQLite 事实源）。
 *
 * cluster_id UNIQUE：endpoint/档案变更不创建重复集群；多集群并存互不串用。
 * key_ref 只存引用（真实密钥在 bootstrap/first-connect 时下发，不落库明文）。
 */
import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SqliteStore } from './sqlite-store';

export interface ClusterProfileRow {
  profile_id: string;
  cluster_id: string;
  name: string;
  master_endpoint: string;
  status: 'active' | 'pending_verification' | 'unreachable';
  key_ref: string;
  node_role: 'master' | 'client' | 'unknown';
  last_verified_at: string | null;
  created_at: string;
}

export interface ClusterProfileInput {
  cluster_id: string;
  name: string;
  master_endpoint: string;
  status?: ClusterProfileRow['status'];
  key_ref?: string;
  node_role?: ClusterProfileRow['node_role'];
}

@Injectable()
export class ClusterProfileRepository {
  constructor(private readonly store: SqliteStore) {}

  create(input: ClusterProfileInput): ClusterProfileRow {
    const now = new Date().toISOString();
    const row: ClusterProfileRow = {
      profile_id: `prof_${randomUUID().slice(0, 12)}`,
      cluster_id: input.cluster_id,
      name: input.name,
      master_endpoint: input.master_endpoint,
      status: input.status ?? 'pending_verification',
      key_ref: input.key_ref ?? `qlh:${input.cluster_id}:profile`,
      node_role: input.node_role ?? 'unknown',
      last_verified_at: null,
      created_at: now,
    };
    this.store.prepare(
      `INSERT INTO cluster_profiles
         (profile_id, cluster_id, name, master_endpoint, status, key_ref, node_role, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(cluster_id) DO UPDATE SET
         name = excluded.name,
         master_endpoint = excluded.master_endpoint,
         status = excluded.status,
         key_ref = excluded.key_ref,
         node_role = excluded.node_role`,
    ).run(
      row.profile_id, row.cluster_id, row.name, row.master_endpoint,
      row.status, row.key_ref, row.node_role, row.created_at,
    );
    return this.getByCluster(row.cluster_id) ?? row;
  }

  get(profileId: string): ClusterProfileRow | null {
    const row = this.store.prepare(
      `SELECT profile_id, cluster_id, name, master_endpoint, status, key_ref,
              node_role, last_verified_at, created_at
       FROM cluster_profiles WHERE profile_id = ?`,
    ).get(profileId) as ClusterProfileRow | undefined;
    return row ?? null;
  }

  getByCluster(clusterId: string): ClusterProfileRow | null {
    const row = this.store.prepare(
      `SELECT profile_id, cluster_id, name, master_endpoint, status, key_ref,
              node_role, last_verified_at, created_at
       FROM cluster_profiles WHERE cluster_id = ?`,
    ).get(clusterId) as ClusterProfileRow | undefined;
    return row ?? null;
  }

  list(): ClusterProfileRow[] {
    return this.store.prepare(
      `SELECT profile_id, cluster_id, name, master_endpoint, status, key_ref,
              node_role, last_verified_at, created_at
       FROM cluster_profiles ORDER BY created_at`,
    ).all() as unknown as ClusterProfileRow[];
  }

  /** 标记验证结果（verify 端点调用后更新）。 */
  markVerified(profileId: string, status: ClusterProfileRow['status']): ClusterProfileRow | null {
    const row = this.get(profileId);
    if (!row) return null;
    const now = new Date().toISOString();
    this.store.prepare(
      `UPDATE cluster_profiles SET status = ?, last_verified_at = ?
       WHERE profile_id = ?`,
    ).run(status, now, profileId);
    return { ...row, status, last_verified_at: now };
  }

  delete(profileId: string): boolean {
    const result = this.store.prepare(
      'DELETE FROM cluster_profiles WHERE profile_id = ?',
    ).run(profileId);
    return Number(result.changes) > 0;
  }
}
