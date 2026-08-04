/**
 * settings 域控制器（阶段 3.2 首迁域，语义对齐 api_server.py:6111-6160）
 *
 *   GET /user/settings → {settings, source: none|database|error}
 *   PUT /user/settings → {status: ok|skipped, synced_fields} / 500
 */
import { Body, Controller, Get, HttpCode, HttpException, Put } from '@nestjs/common';
import { ConfigDao } from '../../data/config-dao';

export class UserSettingsRequest {
  settings: Record<string, unknown> = {};
}

@Controller('user')
export class SettingsController {
  constructor(private readonly dao: ConfigDao) {}

  @Get('settings')
  async getSettings(): Promise<unknown> {
    if (!this.dao.dbEnabled()) {
      return { settings: {}, source: 'none' };
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
      const ok = await this.dao.setUserSettings(settings);
      if (!ok) {
        return { status: 'skipped', reason: '数据库不可用' };
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
