/**
 * settings 域控制器（阶段 3.2 首迁域，语义对齐 api_server.py:6111-6160；
 * M1：pg 可用路径下 SQLite 本地事实源接管读写，pg 作投影）
 *
 *   GET /user/settings → {settings, source: none|database|error}
 *   PUT /user/settings → {status: ok|skipped, synced_fields} / 500
 *
 * 契约保持：DB 禁用 → source:'none' / status:'skipped'（与阶段 3.2 一致）；
 * DB 可用 → 读写走 SQLite（本地事实源）+ outbox 事件，pg 投影失败不阻断。
 */
import { Body, Controller, Get, HttpCode, HttpException, Put } from '@nestjs/common';
import { ConfigDao } from '../../data/config-dao';
import { ClusterSettingsRepository } from '../../data/cluster-settings-repository';
import { OutboxService } from '../../data/outbox.service';

export class UserSettingsRequest {
  settings: Record<string, unknown> = {};
}

const SETTINGS_KEYS = ['user_settings', 'save_history', 'distributed_inference_enabled'];

@Controller('user')
export class SettingsController {
  constructor(
    private readonly dao: ConfigDao,
    private readonly localSettings: ClusterSettingsRepository,
    private readonly outbox: OutboxService,
  ) {}

  @Get('settings')
  async getSettings(): Promise<unknown> {
    if (!this.dao.dbEnabled()) {
      return { settings: {}, source: 'none' };
    }
    // M1：本地 SQLite 事实源优先
    const local = this.localSettings.get('user_settings');
    if (local) {
      try {
        const parsed = JSON.parse(local.value);
        if (parsed && typeof parsed === 'object') {
          return { settings: parsed, source: 'database' };
        }
      } catch {
        // 本地值损坏：回退 pg
      }
    }
    try {
      const settings = await this.dao.getUserSettings();
      return { settings, source: 'database' };
    } catch (err) {
      this.logWarn(err, '读取用户设置失败');
      return { settings: {}, source: 'error' };
    }
  }

  @Put('settings')
  @HttpCode(200)
  async updateSettings(@Body() body: UserSettingsRequest): Promise<unknown> {
    const settings =
      body && typeof body.settings === 'object' && body.settings !== null
        ? (body.settings as Record<string, unknown>)
        : {};
    if (!this.dao.dbEnabled()) {
      return { status: 'skipped', reason: '数据库不可用' };
    }
    try {
      // M1：本地事实源写入 + outbox 事件（同事务：域行与事件原子提交）
      this.localSettings.transaction(() => {
        this.localSettings.set('user_settings', JSON.stringify(settings));
        if (typeof settings['saveHistory'] === 'boolean') {
          this.localSettings.set(
            'save_history', settings['saveHistory'] ? 'true' : 'false',
          );
        }
        if (typeof settings['distributedInference'] === 'boolean') {
          this.localSettings.set(
            'distributed_inference_enabled',
            settings['distributedInference'] ? 'true' : 'false',
          );
        }
        this.outbox.enqueue('cluster_settings', 'updated', {
          keys: SETTINGS_KEYS,
          payload: settings,
        });
      });
      // 投影 pg（失败不阻断：远端为可选投影，但必须留痕）
      try {
        const projected = await this.dao.setUserSettings(settings);
        if (!projected) {
          this.logWarn('pg 投影失败（setUserSettings 返回 false）', '本地已写入');
        }
      } catch (projectErr) {
        this.logWarn(projectErr, 'pg 投影失败（本地已写入，不阻断）');
      }
      return { status: 'ok', synced_fields: Object.keys(settings) };
    } catch (err) {
      this.logWarn(err, '存储用户设置失败');
      throw new HttpException(`存储失败: ${err instanceof Error ? err.message : String(err)}`, 500);
    }
  }

  private logWarn(err: unknown, msg: string): void {
    // eslint-disable-next-line no-console
    console.warn(`[control-svc] ${msg}: ${err instanceof Error ? err.message : String(err)}`);
  }
}
