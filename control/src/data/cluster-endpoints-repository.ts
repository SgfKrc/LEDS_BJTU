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

  /** 幂等 upsert；cluster_id UNIQUE——同 cluster_id 更新 endpoint（不重复建群）。 */
  upsert(entry: ClusterEndpointInput): ClusterEndpointRow {
    const now = new Date().toISOString();
    this.store.prepare(
      `INSERT INTO cluster_endpoints
         (endpoint_id, cluster_id, name, scheme, host, port, status,
          last_verified_at, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(cluster_id) DO UPDATE SET
         endpoint_id = excluded.endpoint_id,
         name = excluded.name, scheme = excluded.scheme,
         host = excluded.host, port = excluded.port,
         status = excluded.status, last_verified_at = excluded.last_verified_at`,
    ).run(
      entry.endpoint_id, entry.cluster_id, entry.name,
      entry.scheme, entry.host, entry.port, entry.status,
      entry.last_verified_at ?? null, now,
    );
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
