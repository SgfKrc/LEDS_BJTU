/**
 * M1 cluster endpoints repository — 主节点端点（SQLite 事实源）。
 * cluster_id 保留去重（endpoint 变更不创建重复集群）；幂等 upsert。
 */
import { Injectable } from '@nestjs/common';
import { SqliteStore } from './sqlite-store';

export interface ClusterEndpointRow {
  endpoint_id: string;
  cluster_id: string;
  name: string;
  scheme: string;
  host: string;
  port: number;
  status: string;
  last_verified_at: string | null;
  created_at: string;
}

export interface ClusterEndpointInput {
  endpoint_id: string;
  cluster_id: string;
  name: string;
  scheme: string;
  host: string;
  port: number;
  status: string;
  last_verified_at?: string | null;
}

@Injectable()
export class ClusterEndpointsRepository {
  constructor(private readonly store: SqliteStore) {}

  get(endpointId: string): ClusterEndpointRow | null {
    const row = this.store.prepare(
      `SELECT endpoint_id, cluster_id, name, scheme, host, port, status,
              last_verified_at, created_at
       FROM cluster_endpoints WHERE endpoint_id = ?`,
    ).get(endpointId) as ClusterEndpointRow | undefined;
    return row ?? null;
  }

  list(): ClusterEndpointRow[] {
    return this.store.prepare(
      `SELECT endpoint_id, cluster_id, name, scheme, host, port, status,
              last_verified_at, created_at
       FROM cluster_endpoints ORDER BY created_at`,
    ).all() as unknown as ClusterEndpointRow[];
  }

  /**
   * 幂等 upsert；新库对 cluster_id 建 UNIQUE，旧 v2 文件可能没有该索引，
   * 因此先查再更新/插入，避免恢复旧主节点时 `ON CONFLICT` 失配。
   */
  upsert(entry: ClusterEndpointInput): ClusterEndpointRow {
    const now = new Date().toISOString();
    const existing = this.store.prepare(
      'SELECT endpoint_id FROM cluster_endpoints WHERE cluster_id = ?',
    ).get(entry.cluster_id) as { endpoint_id: string } | undefined;
    if (existing) {
      this.store.prepare(
        `UPDATE cluster_endpoints SET
           endpoint_id = ?, name = ?, scheme = ?, host = ?, port = ?,
           status = ?, last_verified_at = ?
         WHERE cluster_id = ?`,
      ).run(
        entry.endpoint_id, entry.name, entry.scheme, entry.host, entry.port,
        entry.status, entry.last_verified_at ?? null, entry.cluster_id,
      );
    } else {
      this.store.prepare(
        `INSERT INTO cluster_endpoints
           (endpoint_id, cluster_id, name, scheme, host, port, status,
            last_verified_at, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        entry.endpoint_id, entry.cluster_id, entry.name,
        entry.scheme, entry.host, entry.port, entry.status,
        entry.last_verified_at ?? null, now,
      );
    }
    return {
      endpoint_id: entry.endpoint_id,
      cluster_id: entry.cluster_id,
      name: entry.name,
      scheme: entry.scheme,
      host: entry.host,
      port: entry.port,
      status: entry.status,
      last_verified_at: entry.last_verified_at ?? null,
      created_at: now,
    };
  }
}
