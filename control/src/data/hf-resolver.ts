import { Injectable, Optional } from '@nestjs/common';
import { ModelHttpClient } from './model-http-client';

export interface ResolvedFile {
  rfilename: string;
  size: number;
  sha256?: string | null;
}

export interface ResolveResult {
  repoId: string;
  requestedRevision: string;
  resolvedRevision: string;
  files: ResolvedFile[];
  gated: boolean;
  license: string | null;
}

export function globMatch(pattern: string, name: string): boolean {
  const regex = pattern
    .split('/')
    .map((part) => {
      if (part === '**') return '.*';
      return part
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*/g, '[^/]*');
    })
    .join('/');
  return new RegExp(`^${regex}$`).test(name);
}

@Injectable()
export class HfResolver {
  private readonly http: ModelHttpClient;
  private readonly apiBase: string;

  constructor(
    @Optional() http?: ModelHttpClient,
    @Optional() apiBase?: string,
  ) {
    this.http = http ?? new ModelHttpClient();
    this.apiBase = apiBase ?? 'https://huggingface.co';
  }

  async resolve(
    repoId: string,
    requestedRevision = 'main',
    allowPatterns?: string[] | null,
    apiBaseOverride?: string,
    auth: { token?: string | null } = {},
  ): Promise<ResolveResult> {
    const apiBase = (apiBaseOverride ?? this.apiBase).replace(/\/+$/, '');
    const url = `${apiBase}/api/models/${repoId}?revision=${encodeURIComponent(requestedRevision)}`;
    const response = await this.http.fetch(url, {
      headers: { accept: 'application/json' },
      signal: AbortSignal.timeout(30_000),
    }, auth);
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new Error(
          auth.token
            ? `HF credential rejected (${response.status})`
            : `HF credential required (${response.status})`,
        );
      }
      const text = await response.text().catch(() => '');
      throw new Error(
        `HF resolve failed (${response.status}): ${text.slice(0, 200) || repoId}`,
      );
    }
    const data = (await response.json()) as {
      sha?: string;
      gated?: boolean | string;
      cardData?: { license?: string | null };
      siblings?: Array<{
        rfilename: string;
        size?: number;
        sha256?: string | null;
      }>;
    };
    if (!data.sha) {
      throw new Error(`HF resolve response is missing sha: ${repoId}@${requestedRevision}`);
    }
    let files: ResolvedFile[] = (data.siblings ?? []).map((file) => ({
      rfilename: file.rfilename,
      size: Number(file.size ?? 0),
      sha256: file.sha256 ?? null,
    }));
    if (allowPatterns && allowPatterns.length > 0) {
      files = files.filter((file) => allowPatterns.some(
        (pattern) => globMatch(pattern, file.rfilename),
      ));
    }
    return {
      repoId,
      requestedRevision,
      resolvedRevision: data.sha,
      files,
      gated: data.gated === true || (
        typeof data.gated === 'string'
        && data.gated.length > 0
        && data.gated.toLowerCase() !== 'false'
      ),
      license: data.cardData?.license?.trim().toLowerCase() || null,
    };
  }
}
