import { Injectable } from '@nestjs/common';
import { ClusterSettingsRepository } from './cluster-settings-repository';

export type ModelSourceProvider = 'huggingface' | 'modelscope';

export interface ModelSource {
  schema_version: 1;
  source_id: string;
  name: string;
  provider: ModelSourceProvider;
  endpoint: string;
  credential_ref: string | null;
  priority: number;
  enabled: boolean;
  builtin: boolean;
}

const SETTINGS_KEY = 'model_source_overrides_v1';

export const DEFAULT_MODEL_SOURCES: ReadonlyArray<ModelSource> = [
  {
    schema_version: 1,
    source_id: 'hf-official',
    name: 'Hugging Face',
    provider: 'huggingface',
    endpoint: 'https://huggingface.co',
    credential_ref: null,
    priority: 10,
    enabled: true,
    builtin: true,
  },
  {
    schema_version: 1,
    source_id: 'hf-mirror',
    name: 'Hugging Face mirror',
    provider: 'huggingface',
    endpoint: 'https://hf-mirror.com',
    credential_ref: null,
    priority: 20,
    enabled: false,
    builtin: true,
  },
  {
    schema_version: 1,
    source_id: 'modelscope-official',
    name: 'ModelScope',
    provider: 'modelscope',
    endpoint: 'https://modelscope.cn',
    credential_ref: null,
    priority: 30,
    enabled: false,
    builtin: true,
  },
];

export type ModelSourceInput = Omit<ModelSource, 'schema_version' | 'builtin'>;

@Injectable()
export class ModelSourceRepository {
  constructor(private readonly settings: ClusterSettingsRepository) {}

  list(): ModelSource[] {
    const merged = new Map(
      DEFAULT_MODEL_SOURCES.map((source) => [source.source_id, { ...source }]),
    );
    for (const override of this.readOverrides()) {
      const builtin = DEFAULT_MODEL_SOURCES.some(
        (source) => source.source_id === override.source_id,
      );
      merged.set(override.source_id, { ...override, builtin });
    }
    return [...merged.values()].sort((a, b) => (
      a.priority - b.priority || (a.source_id < b.source_id ? -1 : 1)
    ));
  }

  get(sourceId: string): ModelSource | null {
    return this.list().find((source) => source.source_id === sourceId) ?? null;
  }

  preferred(provider: ModelSourceProvider = 'huggingface'): ModelSource | null {
    return this.list().find(
      (source) => source.enabled && source.provider === provider,
    ) ?? null;
  }

  upsert(input: ModelSourceInput): ModelSource {
    const source = this.normalize(input);
    const overrides = this.readOverrides().filter(
      (entry) => entry.source_id !== source.source_id,
    );
    overrides.push(source);
    this.writeOverrides(overrides);
    return source;
  }

  delete(sourceId: string): boolean {
    const current = this.get(sourceId);
    if (!current) return false;
    const overrides = this.readOverrides().filter(
      (entry) => entry.source_id !== sourceId,
    );
    if (current.builtin) overrides.push({ ...current, enabled: false });
    this.writeOverrides(overrides);
    return true;
  }

  reset(): ModelSource[] {
    this.settings.delete(SETTINGS_KEY);
    return this.list();
  }

  private normalize(input: ModelSourceInput): ModelSource {
    const sourceId = input.source_id.trim();
    const name = input.name.trim();
    const endpoint = input.endpoint.trim().replace(/\/+$/, '');
    if (!sourceId || !/^[a-z0-9][a-z0-9._-]*$/i.test(sourceId)) {
      throw new Error('source_id is invalid');
    }
    if (!name) throw new Error('source name is required');
    let url: URL;
    try {
      url = new URL(endpoint);
    } catch {
      throw new Error('source endpoint is invalid');
    }
    if (url.protocol !== 'https:') {
      throw new Error('source endpoint must use HTTPS');
    }
    if (!Number.isInteger(input.priority) || input.priority < 0) {
      throw new Error('source priority must be a non-negative integer');
    }
    return {
      schema_version: 1,
      source_id: sourceId,
      name,
      provider: input.provider,
      endpoint,
      credential_ref: input.credential_ref?.trim() || null,
      priority: input.priority,
      enabled: input.enabled,
      builtin: DEFAULT_MODEL_SOURCES.some(
        (source) => source.source_id === sourceId,
      ),
    };
  }

  private readOverrides(): ModelSource[] {
    const row = this.settings.get(SETTINGS_KEY);
    if (!row) return [];
    try {
      const value = JSON.parse(row.value) as unknown;
      return Array.isArray(value) ? value as ModelSource[] : [];
    } catch {
      return [];
    }
  }

  private writeOverrides(sources: ModelSource[]): void {
    this.settings.set(SETTINGS_KEY, JSON.stringify(sources));
  }
}
