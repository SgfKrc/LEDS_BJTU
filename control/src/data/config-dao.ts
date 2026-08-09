/**
 * 控制面配置 DAO（cluster_config 表，语义对齐 src/db.py get_config/set_config）
 *
 * 存储边界（M1.2）：
 *  - 主节点 SQLite 是默认事实源；PostgreSQL 只有显式开启兼容出口时才可用。
 *  - 连接失败后每次请求重新探测（不缓存失败状态），兼容旧投影路径。
 *
 * 环境变量：QLH_DB_HOST/PORT/NAME/USER/PASSWORD/ENABLED/SSLMODE。
 * `QLH_DB_ENABLED` 默认为关闭；设为 1/true 只表示用户主动启用旧远端兼容出口。
 */
import { Injectable, Optional } from '@nestjs/common';
import type { Client as PgClient } from 'pg';

export interface DbConfig {
  host: string;
  port: number;
  name: string;
  user: string;
  password: string;
  enabled: boolean;
  sslmode: string;
}

export function loadDbConfig(env: NodeJS.ProcessEnv = process.env): DbConfig {
  const rawEnabled = env.QLH_DB_ENABLED;
  // M1.2：未设置时必须 local_only，避免干净安装隐式连接 PostgreSQL。
  const enabled = rawEnabled !== undefined
    && ['1', 'true', 'yes'].includes(rawEnabled.trim().toLowerCase());
  return {
    host: env.QLH_DB_HOST || 'localhost',
    port: Number(env.QLH_DB_PORT || 5432),
    name: env.QLH_DB_NAME || 'qlh_edge_inference',
    user: env.QLH_DB_USER || 'postgres',
    password: env.QLH_DB_PASSWORD || '',
    enabled,
    sslmode: (env.QLH_DB_SSLMODE || 'prefer').trim() || 'prefer',
  };
}

@Injectable()
export class ConfigDao {
  private cfg: DbConfig;

  constructor(@Optional() cfg?: DbConfig) {
    this.cfg = cfg ?? loadDbConfig();
  }

  private async connect(): Promise<PgClient> {
    const { Client } = await import('pg');
    const client = new Client({
      host: this.cfg.host,
      port: this.cfg.port,
      database: this.cfg.name,
      user: this.cfg.user,
      password: this.cfg.password,
      ssl: this.cfg.sslmode === 'disable' ? false : { rejectUnauthorized: false },
      connectionTimeoutMillis: 3000,
    });
    await client.connect();
    return client;
  }

  private isDbUsable(): boolean {
    return this.cfg.enabled;
  }

  /** 供控制器判断降级分支（对齐 api_server model_host._db_available 语义） */
  dbEnabled(): boolean {
    return this.cfg.enabled;
  }

  /** 主节点是否处于默认 local_only 模式。 */
  isLocalOnly(): boolean {
    return !this.cfg.enabled;
  }

  /** 连接信息（对齐 db_health 的 host/port/db 输出） */
  getConnectionInfo(): { host: string; port: number; db: string } {
    return { host: this.cfg.host, port: this.cfg.port, db: this.cfg.name };
  }

  /** 探测连接（对齐 db.py db_health 的 SELECT 1）；失败返回错误消息 */
  async ping(): Promise<{ ok: boolean; error?: string }> {
    let client: PgClient | null = null;
    try {
      client = await this.connect();
      await client.query('SELECT 1');
      return { ok: true };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    } finally {
      if (client) await client.end().catch(() => undefined);
    }
  }

  /** 读配置项（对齐 db.py get_config：无记录返回默认值） */
  async getConfig(key: string, def = ''): Promise<string> {
    if (!this.isDbUsable()) return def;
    let client: PgClient | null = null;
    try {
      client = await this.connect();
      const r = await client.query('SELECT value FROM cluster_config WHERE key = $1', [
        key,
      ]);
      return r.rows.length ? String(r.rows[0].value) : def;
    } catch {
      return def;
    } finally {
      if (client) await client.end().catch(() => undefined);
    }
  }

  /** 写配置项（对齐 db.py set_config：upsert + updated_at） */
  async setConfig(key: string, value: string): Promise<boolean> {
    if (!this.isDbUsable()) return false;
    let client: PgClient | null = null;
    try {
      client = await this.connect();
      await client.query(
        `INSERT INTO cluster_config (key, value)
         VALUES ($1, $2)
         ON CONFLICT (key) DO UPDATE SET
           value = EXCLUDED.value,
           updated_at = NOW()`,
        [key, value],
      );
      return true;
    } catch {
      return false;
    } finally {
      if (client) await client.end().catch(() => undefined);
    }
  }

  /** 读用户设置 JSON（对齐 db.py get_user_settings） */
  async getUserSettings(): Promise<Record<string, unknown>> {
    const val = await this.getConfig('user_settings', '');
    if (!val) return {};
    try {
      const parsed = JSON.parse(val);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }

  /** 写用户设置 + 同步专用键（对齐 api_server update_user_settings 语义） */
  async setUserSettings(settings: Record<string, unknown>): Promise<boolean> {
    const ok = await this.setConfig(
      'user_settings',
      JSON.stringify(settings),
    );
    if (!ok) return false;
    if (typeof settings['saveHistory'] === 'boolean') {
      await this.setConfig('save_history', settings['saveHistory'] ? 'true' : 'false');
    }
    if (typeof settings['distributedInference'] === 'boolean') {
      await this.setConfig(
        'distributed_inference_enabled',
        settings['distributedInference'] ? 'true' : 'false',
      );
    }
    return true;
  }
}
