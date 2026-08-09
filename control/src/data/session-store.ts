/**
 * 会话/对话存储 — 默认使用主节点 SQLite，显式传入目录时保留 JSON 兼容模式
 * (微服务架构改造计划 阶段 3.2 首迁域)
 *
 * 文件布局（与 local_store.py:7-10 一致）：
 *   <dir>/_sessions.json     # [{id, title, created_at, updated_at, message_count}, ...]
 *   <dir>/{session_id}.json  # [{role, content, created_at, metrics?}, ...]
 *
 * JSON 兼容目录解析：QLH_CHAT_HISTORY_DIR 环境变量优先；否则 <cwd>/chat_history
 * （与 config.LOG_DIR 的 dirname + chat_history 一致，控制面进程从项目根启动）。
 *
 * 硬验收：旧 Python 生成的 JSON 数据必须可读（格式逐字节兼容：
 * ensure_ascii=False + indent=2 + default=str，时间 "%Y-%m-%dT%H:%M:%S"）。
 * 并发：Node 单线程 + 原子写（tmp + rename），对齐 Python 的 threading.Lock 语义。
 */
import { Inject, Injectable, Optional } from '@nestjs/common';
import { randomUUID } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { SqliteStore } from './sqlite-store';

export interface SessionMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

/**
 * 会话 id 白名单：uuid4、'default' 及字母数字 id（对齐日志域的 isLogFilename 模式）。
 * 防御 path traversal——sessionId 来自 query/path 参数，可被远程控制；
 * Python local_store.py 同样存在此缺陷，但 TS 侧新代码必须加固。
 */
export function isValidSessionId(id: string): boolean {
  return /^[A-Za-z0-9_-]{1,64}$/.test(id);
}

export interface ChatMessage {
  role: string;
  content: string;
  created_at?: string;
  metrics?: Record<string, unknown>;
}

export function resolveChatHistoryDir(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.QLH_CHAT_HISTORY_DIR?.trim();
  return override || path.join(process.cwd(), 'chat_history');
}

function nowStr(): string {
  // 对齐 Python time.strftime("%Y-%m-%dT%H:%M:%S")（本地时间）
  const d = new Date();
  const p = (n: number): string => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` +
    `T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

@Injectable()
export class SessionStore {
  /** 进程内活跃会话（对齐 api_server 模块级 active_session_id；重启后为 null） */
  activeSessionId: string | null = null;

  private readonly dir: string;
  private readonly sqlite: SqliteStore | null;
  private legacyImportChecked = false;

  constructor(@Optional() @Inject(SqliteStore) storeOrDir?: SqliteStore | string) {
    if (typeof storeOrDir === 'string') {
      this.sqlite = null;
      this.dir = storeOrDir;
      fs.mkdirSync(this.dir, { recursive: true });
    } else {
      this.sqlite = storeOrDir ?? new SqliteStore();
      this.dir = '';
    }
  }

  private useSqlite(): SqliteStore | null {
    if (!this.sqlite) return null;
    this.sqlite.open();
    this.importLegacyJsonOnce();
    return this.sqlite;
  }

  /** 首次切换到 SQLite 时导入旧 JSON，原文件保留用于人工回滚。 */
  private importLegacyJsonOnce(): void {
    if (!this.sqlite || this.legacyImportChecked) return;
    this.legacyImportChecked = true;
    const marker = '__legacy_json_sessions_v1__';
    const marked = this.sqlite.prepare(
      'SELECT value FROM cluster_settings WHERE key = ?',
    ).get(marker) as { value: string } | undefined;
    if (marked) return;
    const legacyDir = resolveChatHistoryDir();
    const sessionsFile = path.join(legacyDir, '_sessions.json');
    try {
      const raw = fs.readFileSync(sessionsFile, 'utf-8');
      const sessions = JSON.parse(raw) as SessionMeta[];
      this.sqlite.transaction(() => {
        for (const session of Array.isArray(sessions) ? sessions : []) {
          if (!session || !isValidSessionId(String(session.id ?? ''))) continue;
          this.sqlite!.prepare(
            `INSERT INTO sessions (session_id, title, created_at, updated_at, message_count)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(session_id) DO NOTHING`,
          ).run(
            session.id,
            String(session.title ?? '新对话'),
            String(session.created_at ?? nowStr()),
            String(session.updated_at ?? session.created_at ?? nowStr()),
            Math.max(0, Number(session.message_count) || 0),
          );
          const messageFile = path.join(legacyDir, `${session.id}.json`);
          if (!fs.existsSync(messageFile)) continue;
          const messages = JSON.parse(fs.readFileSync(messageFile, 'utf-8')) as ChatMessage[];
          for (const message of Array.isArray(messages) ? messages : []) {
            if (!message || typeof message !== 'object') continue;
            this.sqlite!.prepare(
              `INSERT INTO session_messages (session_id, role, content, created_at, metrics)
               VALUES (?, ?, ?, ?, ?)`,
            ).run(
              session.id,
              String(message.role ?? ''),
              String(message.content ?? ''),
              message.created_at ? String(message.created_at) : null,
              message.metrics ? JSON.stringify(message.metrics) : null,
            );
          }
        }
        this.sqlite!.prepare(
          `INSERT INTO cluster_settings (key, value, updated_at)
           VALUES (?, '1', ?)
           ON CONFLICT(key) DO NOTHING`,
        ).run(marker, new Date().toISOString());
      });
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT') {
        console.warn(`[control-svc] 会话 JSON 兼容导入失败，保留 SQLite 空域: ${String(err)}`);
      }
      this.sqlite.prepare(
        `INSERT INTO cluster_settings (key, value, updated_at)
         VALUES (?, '1', ?)
         ON CONFLICT(key) DO NOTHING`,
      ).run(marker, new Date().toISOString());
    }
  }

  // ---------- 私有：文件 IO ----------

  private sessionsPath(): string {
    return path.join(this.dir, '_sessions.json');
  }

  private messagesPath(sessionId: string): string {
    if (!isValidSessionId(sessionId)) {
      throw new Error(`非法会话 id: ${sessionId}`);
    }
    return path.join(this.dir, `${sessionId}.json`);
  }

  private readJson<T>(file: string, def: T): T {
    try {
      const raw = fs.readFileSync(file, 'utf-8');
      return JSON.parse(raw) as T;
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code !== 'ENOENT') {
        // 损坏文件：对齐 local_store.py _read_json 的"重建"语义
        console.warn(`[control-svc] JSON 损坏，重建: ${file}`);
      }
      return def;
    }
  }

  private writeJson(file: string, data: unknown): void {
    // 对齐 local_store.py _write_json：ensure_ascii=False + indent=2 + default=str
    const tmp = `${file}.tmp`;
    try {
      fs.writeFileSync(tmp, JSON.stringify(data, null, 2), 'utf-8');
      fs.renameSync(tmp, file);
    } catch (err) {
      console.warn(`[control-svc] 写入本地储存失败: ${file}: ${String(err)}`);
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
    }
  }

  private loadSessions(): SessionMeta[] {
    return this.readJson<SessionMeta[]>(this.sessionsPath(), []);
  }

  private saveSessions(sessions: SessionMeta[]): void {
    this.writeJson(this.sessionsPath(), sessions);
  }

  private readMessages(sessionId: string): ChatMessage[] {
    return this.readJson<ChatMessage[]>(this.messagesPath(sessionId), []);
  }

  // ---------- 会话元数据（对齐 local_store.py 101-199） ----------

  createSession(sessionId: string, title: string): SessionMeta {
    const sqlite = this.useSqlite();
    const now = nowStr();
    const session: SessionMeta = {
      id: sessionId,
      title,
      created_at: now,
      updated_at: now,
      message_count: 0,
    };
    if (sqlite) {
      sqlite.prepare(
        `INSERT INTO sessions (session_id, title, created_at, updated_at, message_count)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(session_id) DO UPDATE SET title=excluded.title,
           updated_at=excluded.updated_at`,
      ).run(session.id, session.title, session.created_at, session.updated_at, session.message_count);
      return session;
    }
    const sessions = this.loadSessions();
    sessions.unshift(session);
    this.saveSessions(sessions);
    return session;
  }

  listSessions(limit: number, offset: number): { sessions: SessionMeta[]; total: number } {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const sessions = sqlite.prepare(
        `SELECT session_id AS id, title, created_at, updated_at, message_count
         FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?`,
      ).all(Math.max(0, Math.floor(limit)), Math.max(0, Math.floor(offset))) as unknown as SessionMeta[];
      const totalRow = sqlite.prepare('SELECT COUNT(*) AS total FROM sessions').get() as { total: number };
      return { sessions, total: Number(totalRow.total ?? 0) };
    }
    const sessions = this.loadSessions();
    // 对齐 local_store.py get_all_local_sessions：updated_at DESC
    sessions.sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));
    return { sessions: sessions.slice(offset, offset + limit), total: sessions.length };
  }

  getSession(sessionId: string): SessionMeta | null {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const row = sqlite.prepare(
        `SELECT session_id AS id, title, created_at, updated_at, message_count
         FROM sessions WHERE session_id = ?`,
      ).get(sessionId) as SessionMeta | undefined;
      return row ?? null;
    }
    for (const s of this.loadSessions()) {
      if (s.id === sessionId) return s;
    }
    return null;
  }

  renameSession(sessionId: string, title: string): SessionMeta | null {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const updatedAt = nowStr();
      const result = sqlite.prepare(
        'UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?',
      ).run(title, updatedAt, sessionId);
      if (Number(result.changes) === 0) return null;
      return this.getSession(sessionId);
    }
    const sessions = this.loadSessions();
    for (const s of sessions) {
      if (s.id === sessionId) {
        s.title = title;
        s.updated_at = nowStr();
        this.saveSessions(sessions);
        return s;
      }
    }
    return null;
  }

  /** 删除会话及其消息文件。返回删除的会话数（0 或 1）。 */
  deleteSession(sessionId: string): number {
    const sqlite = this.useSqlite();
    if (sqlite) {
      return sqlite.transaction(() => {
        sqlite.prepare('DELETE FROM session_messages WHERE session_id = ?').run(sessionId);
        const result = sqlite.prepare('DELETE FROM sessions WHERE session_id = ?').run(sessionId);
        return Number(result.changes);
      });
    }
    const sessions = this.loadSessions();
    const kept = sessions.filter((s) => s.id !== sessionId);
    let deleted = 0;
    if (kept.length < sessions.length) {
      deleted = 1;
      this.saveSessions(kept);
    }
    try {
      fs.rmSync(this.messagesPath(sessionId), { force: true });
    } catch (err) {
      console.warn(`[control-svc] 删除本地消息文件失败: ${String(err)}`);
    }
    return deleted;
  }

  incrementSessionMessageCount(sessionId: string): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.prepare(
        `UPDATE sessions SET message_count = message_count + 1,
         updated_at = ? WHERE session_id = ?`,
      ).run(nowStr(), sessionId);
      return;
    }
    const sessions = this.loadSessions();
    for (const s of sessions) {
      if (s.id === sessionId) {
        s.message_count = (s.message_count || 0) + 1;
        s.updated_at = nowStr();
        break;
      }
    }
    this.saveSessions(sessions);
  }

  decrementSessionMessageCount(sessionId: string, count = 2): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.prepare(
        `UPDATE sessions SET message_count = MAX(0, message_count - ?),
         updated_at = ? WHERE session_id = ?`,
      ).run(Math.max(0, Math.floor(count)), nowStr(), sessionId);
      return;
    }
    const sessions = this.loadSessions();
    for (const s of sessions) {
      if (s.id === sessionId) {
        s.message_count = Math.max(0, (s.message_count || 0) - count);
        s.updated_at = nowStr();
        break;
      }
    }
    this.saveSessions(sessions);
  }

  // ---------- 对话消息（对齐 local_store.py 206-269） ----------

  saveMessage(sessionId: string, role: string, content: string, metrics?: Record<string, unknown>): void {
    const sqlite = this.useSqlite();
    if (sqlite) {
      sqlite.prepare(
        `INSERT INTO session_messages (session_id, role, content, created_at, metrics)
         VALUES (?, ?, ?, ?, ?)`,
      ).run(
        sessionId,
        role,
        content,
        nowStr(),
        metrics && Object.keys(metrics).length > 0 ? JSON.stringify(metrics) : null,
      );
      return;
    }
    const messages = this.readMessages(sessionId);
    const msg: ChatMessage = { role, content, created_at: nowStr() };
    if (metrics && Object.keys(metrics).length > 0) {
      msg.metrics = metrics;
    }
    messages.push(msg);
    this.writeJson(this.messagesPath(sessionId), messages);
  }

  /** 读取消息（末尾 limit 条；limit<=0 或未提供取全部），对齐 load_local_conversation */
  loadMessages(sessionId: string, limit = 200): ChatMessage[] {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const rows = (limit > 0
        ? sqlite.prepare(
          `SELECT role, content, created_at, metrics FROM session_messages
           WHERE session_id = ? ORDER BY message_id DESC LIMIT ?`,
        ).all(sessionId, Math.floor(limit))
        : sqlite.prepare(
          `SELECT role, content, created_at, metrics FROM session_messages
           WHERE session_id = ? ORDER BY message_id ASC`,
        ).all(sessionId)) as Array<{ role: string; content: string; created_at: string | null; metrics: string | null }>;
      const ordered = limit > 0 ? rows.reverse() : rows;
      return ordered.map((row) => ({
        role: row.role,
        content: row.content,
        ...(row.created_at ? { created_at: row.created_at } : {}),
        ...(row.metrics ? { metrics: JSON.parse(row.metrics) as Record<string, unknown> } : {}),
      }));
    }
    const messages = this.readMessages(sessionId);
    if (limit > 0 && messages.length > limit) {
      return messages.slice(-limit);
    }
    return messages;
  }

  messageCountOf(sessionId: string): number {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const row = sqlite.prepare(
        'SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?',
      ).get(sessionId) as { count: number };
      return Number(row.count ?? 0);
    }
    return this.readMessages(sessionId).length;
  }

  /** 清空消息，返回删除的消息数（对齐 clear_local_conversation） */
  clearMessages(sessionId: string): number {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const result = sqlite.prepare('DELETE FROM session_messages WHERE session_id = ?').run(sessionId);
      return Number(result.changes);
    }
    const messages = this.readMessages(sessionId);
    this.writeJson(this.messagesPath(sessionId), []);
    return messages.length;
  }

  /** 删除指定轮次（user+assistant 两条）。返回删除数（0 或 2），对齐 delete_local_message_range */
  deleteMessageRange(sessionId: string, turnIndex: number): number {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const rows = sqlite.prepare(
        `SELECT message_id FROM session_messages WHERE session_id = ?
         ORDER BY message_id ASC LIMIT 2 OFFSET ?`,
      ).all(sessionId, Math.max(0, Math.floor(turnIndex) * 2)) as Array<{ message_id: number }>;
      if (rows.length < 2) return 0;
      const result = sqlite.prepare(
        'DELETE FROM session_messages WHERE message_id IN (?, ?)',
      ).run(rows[0].message_id, rows[1].message_id);
      return Number(result.changes);
    }
    const messages = this.readMessages(sessionId);
    const idx = turnIndex * 2;
    if (idx + 1 >= messages.length) return 0;
    messages.splice(idx, 2);
    this.writeJson(this.messagesPath(sessionId), messages);
    return 2;
  }

  /** 本地存储统计（对齐 local_store_stats） */
  stats(): Record<string, unknown> {
    const sqlite = this.useSqlite();
    if (sqlite) {
      const sessions = sqlite.prepare('SELECT COUNT(*) AS count FROM sessions').get() as { count: number };
      const messages = sqlite.prepare('SELECT COUNT(*) AS count FROM session_messages').get() as { count: number };
      return {
        store_dir: sqlite.filePath,
        session_count: Number(sessions.count ?? 0),
        message_count: Number(messages.count ?? 0),
        file_count: 1,
        backend: 'sqlite',
      };
    }
    try {
      const files = fs.readdirSync(this.dir);
      const msgFiles = files.filter((f) => f.endsWith('.json') && f !== '_sessions.json');
      let totalMessages = 0;
      for (const f of msgFiles) {
        totalMessages += this.readJson<unknown[]>(path.join(this.dir, f), []).length;
      }
      return {
        store_dir: this.dir,
        session_count: this.loadSessions().length,
        message_count: totalMessages,
        file_count: msgFiles.length,
      };
    } catch (err) {
      return { store_dir: this.dir, error: String(err) };
    }
  }

  /** 生成新会话 id（对齐 uuid.uuid4() 字符串形态） */
  newSessionId(): string {
    return randomUUID();
  }
}
