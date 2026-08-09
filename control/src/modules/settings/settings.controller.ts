/** settings 域控制器：主节点 SQLite 是唯一事实源。 */
import { Body, Controller, Get, HttpCode, HttpException, Put } from '@nestjs/common';
import { ClusterSettingsRepository } from '../../data/cluster-settings-repository';

export class UserSettingsRequest {
  settings: Record<string, unknown> = {};
}

@Controller('user')
export class SettingsController {
  constructor(private readonly localSettings: ClusterSettingsRepository) {}

  @Get('settings')
  async getSettings(): Promise<unknown> {
    const local = this.localSettings.get('user_settings');
    if (local) {
      try {
        const parsed = JSON.parse(local.value);
        if (parsed && typeof parsed === 'object') {
          return { settings: parsed, source: 'local' };
        }
      } catch {
        return { settings: {}, source: 'local' };
      }
    }
    return { settings: {}, source: 'local' };
  }

  @Put('settings')
  @HttpCode(200)
  async updateSettings(@Body() body: UserSettingsRequest): Promise<unknown> {
    const settings =
      body && typeof body.settings === 'object' && body.settings !== null
        ? (body.settings as Record<string, unknown>)
        : {};
    try {
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
      });
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
