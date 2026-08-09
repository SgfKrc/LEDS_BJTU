/**
 * 任务图 journal 存储 — 默认使用主节点 SQLite，显式传入文件时保留 JSON 兼容模式
 * (微服务架构改造计划 阶段 3.2 任务图控制面域)
 *
 * JSON 兼容模式：每条 workflow 一条最新快照（对齐 workflow_snapshots 表的
 * 单行-per-workflow 语义）；内部维护 _updated_at（对齐表 updated_at 列，
 * cleanup 按它排序/过期，序列化时剔除）。
 * SQLite 模式与执行段 journal 使用同一主节点数据库；旧 JSON 首次启动只读导入。
 * 并发：Node 单线程 + 原子写（tmp + rename）。
 */
import { Inject, Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import { SqliteStore } from './sqlite-store';

export type WorkflowState =
  | 'created'
  | 'running'
  | 'result_ready'
  | 'completed'
  | 'failed'
  | 'cancelled';

export const TERMINAL_WORKFLOW_STATES: WorkflowState[] = ['completed', 'failed', 'cancelled'];

export const WORKFLOW_ID_PATTERN = /^wf_[A-Za-z0-9_-]{8,96}$/;

export interface AttemptSnapshot {
  attempt_id: string;
  provider: string;
  provider_kind: string;
  provider_node_id: string;
  reservation_id: string;
  lease_id: string;
  lease_epoch: number;
  lease_expires_at: number;
  state: string;
  started_at: number;
  finished_at: number | null;
  error: string;
  result_metadata: Record<string, unknown>;
  result_sha256: string;
}

export interface StageSnapshot {
  stage_id: string;
  stage_type: string;
  depends_on: string[];
  provider: string;
  requested_provider: string;
  fallback_providers: string[];
  pure: boolean;
  accept_timeout_seconds: number;
  lease_timeout_seconds: number;
  minimum_successful_dependencies: number;
  max_same_provider_retries: number;
  retry_safe: boolean;
  lease_epoch: number;
  winner_attempt_id: string;
  retry_count: number;
  same_provider_retry_count: number;
  last_retry_error_code: string;
  result_rejection_count: number;
  last_result_rejection_reason: string;
  last_result_rejected_at: number | null;
  state: string;
  started_at: number | null;
  finished_at: number | null;
  duration_seconds: number;
  error: string;
  attempts: AttemptSnapshot[];
  output_available: boolean;
  output_sha256: string;
  output_size_bytes: number;
}

/** 对齐 WorkflowRecord.snapshot() 的完整字段（task_graph.py:294-329） */
export interface WorkflowSnapshot {
  workflow_id: string;
  request_id: string;
  session_id: string;
  model_identity: unknown;
  template: string;
  state: WorkflowState;
  last_sequence: number;
  final_stage_id: string;
  created_at: number;
  started_at: number | null;
  result_ready_at: number | null;
  finished_at: number | null;
  duration_seconds: number;
  error: string;
  stage_count: number;
  completed_stage_count: number;
  failed_stage_count: number;
  skipped_stage_count: number;
  partial_result: boolean;
  cancelled_stage_count: number;
  attempt_count: number;
  retry_count: number;
  same_provider_retry_count: number;
  result_rejection_count: number;
  cancel_requested: boolean;
  /** 恢复标记（对齐 _decorate_persisted_snapshot / observability 消费） */
  recovered_after_restart?: boolean;
  recovery_reason?: string;
  error_code?: string;
  stages: StageSnapshot[];
}

/** 内部记录：快照 + 对齐 SQLite updated_at 的维护时间戳 */
interface JournalRecord {
  _updated_at: number;
  snapshot: WorkflowSnapshot;
}

export function resolveJournalFile(env: NodeJS.ProcessEnv = process.env): string {
  return (
    env.QLH_WORKFLOW_JOURNAL_FILE?.trim() ||
    path.join(process.cwd(), 'workflow_journal.json')
  );
}

@Injectable()
export class WorkflowJournalStore {
  private readonly file: string;
  private readonly sqlite: SqliteStore | null;
  private legacyImportChecked = false;

  constructor(@Optional() @Inject(SqliteStore) storeOrFile?: SqliteStore | string) {
    if (typeof storeOrFile === 'string') {
      this.sqlite = null;
      this.file = storeOrFile;
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
    } else {
      this.sqlite = storeOrFile ?? new SqliteStore();
      this.file = '';
    }
  }

  private useSqlite(): SqliteStore | null {
    if (!this.sqlite) return null;
    this.sqlite.open();
    this.importLegacyJsonOnce();
    return this.sqlite;
  }

  private importLegacyJsonOnce(): void {
    if (!this.sqlite || this.legacyImportChecked) return;
    this.legacyImportChecked = true;
    const marker = '__legacy_json_workflows_v1__';
    const marked = this.sqlite.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get(marker) as { value: string } | undefined;
    if (marked) return;
    const legacyFile = resolveJournalFile();
    try {
      const raw = fs.readFileSync(legacyFile, 'utf-8');
      const records = JSON.parse(raw) as JournalRecord[];
      this.sqlite.transaction(() => {
        for (const record of Array.isArray(records) ? records : []) {
          const workflowId = String(record?.snapshot?.workflow_id ?? '');
          if (!workflowId) continue;
          this.sqlite!.prepare(
            `INSERT INTO workflow_journal (workflow_id, updated_at, payload)
             VALUES (?, ?, ?) ON CONFLICT(workflow_id) DO NOTHING`,
          ).run(workflowId, Number(record._updated_at) || 0, JSON.stringify(record.snapshot));
        }
        this.sqlite!.prepare(
          `INSERT INTO cluster_settings (key, value, updated_at)
           VALUES (?, '1', ?) ON CONFLICT(key) DO NOTHING`,
        ).run(marker, new Date().toISOString());
      });
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT') {
        console.warn(`[control-svc] workflow JSON 兼容导入失败，保留 SQLite 空域: ${String(err)}`);
      }
      this.sqlite.prepare(
        `INSERT INTO cluster_settings (key, value, updated_at)
         VALUES (?, '1', ?) ON CONFLICT(key) DO NOTHING`,
      ).run(marker, new Date().toISOString());
    }
  }

  private sqliteRecords(sqlite: SqliteStore): JournalRecord[] {
    const rows = sqlite.prepare(
      'SELECT workflow_id, updated_at, payload FROM workflow_journal',
    ).all() as Array<{ workflow_id: string; updated_at: number; payload: string }>;
    return rows.map((row) => ({
      _updated_at: Number(row.updated_at) || 0,
      snapshot: JSON.parse(row.payload) as WorkflowSnapshot,
    }));
  }

  private load(): JournalRecord[] {
    try {
      const raw = fs.readFileSync(this.file, 'utf-8');
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((d) => d && typeof d === 'object' && d.snapshot?.workflow_id)
        .map((d) => ({
          _updated_at: Number(d._updated_at) || 0,
          snapshot: d.snapshot as WorkflowSnapshot,
        }));
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code !== 'ENOENT') {
        console.warn(`[control-svc] workflow journal 损坏，重建: ${this.file}`);
      }
      return [];
    }
  }

  private save(records: JournalRecord[]): void {
    const tmp = `${this.file}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify(records, null, 2), 'utf-8');
      fs.renameSync(tmp, this.file);
    } catch (err) {
      console.warn(`[control-svc] 写入 workflow journal 失败: ${this.file}: ${String(err)}`);
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
    }
  }

  getSnapshot(workflowId: string): WorkflowSnapshot | null {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const row = sqlite.prepare(
        'SELECT payload FROM workflow_journal WHERE workflow_id = ?',
      ).get(workflowId) as { payload: string } | undefined;
      return row ? JSON.parse(row.payload) as WorkflowSnapshot : null;
    }
    const rec = this.load().find((r) => r.snapshot.workflow_id === workflowId);
    return rec ? rec.snapshot : null;
  }

  /** 对齐 TaskGraphCoordinator.list：created_at DESC + session 过滤 + limit */
  listSnapshots(limit: number, sessionId = ''): WorkflowSnapshot[] {
    const sqlite = this.useSqlite();
    if (sqlite) {
      let workflows = this.sqliteRecords(sqlite).map((r) => r.snapshot);
      if (sessionId) workflows = workflows.filter((w) => w.session_id === sessionId);
      workflows.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
      return workflows.slice(0, Math.max(1, Math.min(Math.floor(limit) || 20, 100000)));
    }
    const safeLimit = Math.max(1, Math.min(Math.floor(limit) || 20, 100000));
    let workflows = this.load().map((r) => r.snapshot);
    if (sessionId) {
      workflows = workflows.filter((w) => w.session_id === sessionId);
    }
    workflows.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    return workflows.slice(0, safeLimit);
  }

  /** 对齐 TaskJournal.append_event 的持久化（最新快照覆盖旧快照） */
  upsertSnapshot(snapshot: WorkflowSnapshot): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.prepare(
        `INSERT INTO workflow_journal (workflow_id, updated_at, payload)
         VALUES (?, ?, ?) ON CONFLICT(workflow_id) DO UPDATE SET
           updated_at = excluded.updated_at, payload = excluded.payload`,
      ).run(snapshot.workflow_id, Date.now() / 1000, JSON.stringify(snapshot));
      return;
    }
    const records = this.load();
    const idx = records.findIndex((r) => r.snapshot.workflow_id === snapshot.workflow_id);
    const record: JournalRecord = { _updated_at: Date.now() / 1000, snapshot };
    if (idx >= 0) records[idx] = record;
    else records.push(record);
    this.save(records);
  }

  /** 对齐 cleanup_terminal：仅删终态；deleted_events 无事件表，等同 workflow 数 */
  cleanupTerminal(maxAgeDays: number, maxRecords: number): {
    deleted_workflows: number;
    deleted_events: number;
    deleted_by_age: number;
    deleted_by_limit: number;
    remaining_terminal: number;
  } {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const safeAge = Math.max(0, maxAgeDays) * 86400;
      const safeRecords = Math.max(0, maxRecords);
      const now = Date.now() / 1000;
      const records = this.sqliteRecords(sqlite);
      if (safeAge <= 0 && safeRecords <= 0) {
        return {
          deleted_workflows: 0,
          deleted_events: 0,
          deleted_by_age: 0,
          deleted_by_limit: 0,
          remaining_terminal: records.filter((r) => TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state)).length,
        };
      }
      const terminal = records
        .filter((r) => TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state))
        .sort((a, b) => b._updated_at - a._updated_at || b.snapshot.workflow_id.localeCompare(a.snapshot.workflow_id));
      const byAge = new Set<string>();
      if (safeAge > 0) {
        const cutoff = now - safeAge;
        for (const r of terminal) if (r._updated_at < cutoff) byAge.add(r.snapshot.workflow_id);
      }
      const byLimit = new Set<string>();
      if (safeRecords > 0 && terminal.length > safeRecords) {
        for (const r of terminal.slice(safeRecords)) byLimit.add(r.snapshot.workflow_id);
      }
      const deleteIds = new Set([...byAge, ...byLimit]);
      if (deleteIds.size > 0) {
        sqlite.transaction(() => {
          for (const id of deleteIds) sqlite.prepare('DELETE FROM workflow_journal WHERE workflow_id = ?').run(id);
        });
      }
      const remaining = this.sqliteRecords(sqlite).filter((r) => TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state)).length;
      return {
        deleted_workflows: deleteIds.size,
        deleted_events: deleteIds.size,
        deleted_by_age: byAge.size,
        deleted_by_limit: byLimit.size,
        remaining_terminal: remaining,
      };
    }
    const safeAge = Math.max(0, maxAgeDays) * 86400;
    const safeRecords = Math.max(0, maxRecords);
    const now = Date.now() / 1000;
    const records = this.load();
    if (safeAge <= 0 && safeRecords <= 0) {
      return {
        deleted_workflows: 0,
        deleted_events: 0,
        deleted_by_age: 0,
        deleted_by_limit: 0,
        remaining_terminal: records.filter((r) =>
          TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state),
        ).length,
      };
    }
    // 终态按 _updated_at DESC（对齐 ORDER BY updated_at DESC, workflow_id DESC）
    const terminal = records
      .filter((r) => TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state))
      .sort((a, b) => b._updated_at - a._updated_at || b.snapshot.workflow_id.localeCompare(a.snapshot.workflow_id));
    const byAge = new Set<string>();
    if (safeAge > 0) {
      const cutoff = now - safeAge;
      for (const r of terminal) {
        if (r._updated_at < cutoff) byAge.add(r.snapshot.workflow_id);
      }
    }
    const byLimit = new Set<string>();
    if (safeRecords > 0 && terminal.length > safeRecords) {
      for (const r of terminal.slice(safeRecords)) {
        byLimit.add(r.snapshot.workflow_id);
      }
    }
    const deleteIds = new Set([...byAge, ...byLimit]);
    if (deleteIds.size > 0) {
      this.save(records.filter((r) => !deleteIds.has(r.snapshot.workflow_id)));
    }
    const remaining = this.load().filter((r) =>
      TERMINAL_WORKFLOW_STATES.includes(r.snapshot.state),
    ).length;
    return {
      deleted_workflows: deleteIds.size,
      deleted_events: deleteIds.size,
      deleted_by_age: byAge.size,
      deleted_by_limit: byLimit.size,
      remaining_terminal: remaining,
    };
  }

  /** 对齐 journal_status/health 的公开形状（path/journal_mode 属 SQLite，不输出） */
  status(): Record<string, unknown> {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const row = sqlite.prepare('SELECT COUNT(*) AS count FROM workflow_journal').get() as { count: number };
      return {
        enabled: true,
        available: true,
        backend: 'sqlite',
        schema_version: sqlite.schemaVersion,
        latency_ms: 0,
        error: '',
        record_count: Number(row.count ?? 0),
        last_recovery: {},
        last_cleanup: {},
      };
    }
    const records = this.load();
    return {
      enabled: true,
      available: true,
      backend: 'json-file',
      schema_version: 1,
      latency_ms: 0,
      error: '',
      record_count: records.length,
      last_recovery: {},
      last_cleanup: {},
    };
  }
}
