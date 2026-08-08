import { Injectable } from '@nestjs/common';
import { ClusterSettingsRepository } from './cluster-settings-repository';
import { normalizeModelProxyUrl } from './model-http-client';

export interface UserModelProxyConfig {
  schema_version: 1;
  url: string;
  updated_at: string;
}

const SETTINGS_KEY = 'model_http_proxy_v1';

@Injectable()
export class ModelProxyConfigRepository {
  constructor(private readonly settings: ClusterSettingsRepository) {}

  get(): UserModelProxyConfig | null {
    const row = this.settings.get(SETTINGS_KEY);
    if (!row) return null;
    try {
      const parsed = JSON.parse(row.value) as UserModelProxyConfig;
      const url = normalizeModelProxyUrl(parsed.url);
      if (parsed.schema_version !== 1 || !url || !parsed.updated_at) {
        throw new Error('invalid proxy record');
      }
      return { ...parsed, url };
    } catch {
      throw new Error('saved model proxy configuration is invalid');
    }
  }

  set(rawUrl: string): UserModelProxyConfig {
    const url = normalizeModelProxyUrl(rawUrl);
    if (!url) throw new Error('proxy url is required');
    const config: UserModelProxyConfig = {
      schema_version: 1,
      url,
      updated_at: new Date().toISOString(),
    };
    this.settings.set(SETTINGS_KEY, JSON.stringify(config));
    return config;
  }

  clear(): boolean {
    return this.settings.delete(SETTINGS_KEY);
  }
}
