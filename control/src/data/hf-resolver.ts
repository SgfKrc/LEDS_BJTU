/**
 * M3 Hugging Face resolve — 解析仓库 revision 与文件清单（Node 内置 fetch）。
 *
 * 不引入 huggingface_hub 依赖（隔离意图：不触碰推理环境解释器与 HF_HOME）。
 * resolve 只读 HF API，返回完整 commit revision 与文件列表（含 LFS 大小）。
 */
import { Injectable, Optional } from '@nestjs/common';

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
}

export function globMatch(pattern: string, name: string): boolean {
  // 简单 glob：* 不跨 /，** 跨 /；支持前缀/后缀通配
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

export interface ResolverOptions {
  /** 测试注入：自定义 fetch（缺省全局 fetch）。 */
  fetchFn?: typeof fetch;
  apiBase?: string;
}

@Injectable()
export class HfResolver {
  private readonly fetchFn: typeof fetch;
  private readonly apiBase: string;

  constructor(@Optional() options: ResolverOptions = {}) {
    this.fetchFn = options.fetchFn ?? globalThis.fetch;
    this.apiBase = options.apiBase ?? 'https://huggingface.co';
  }

  /** 解析仓库（不下载）。revision 支持 branch/tag/commit；返回完整 sha。 */
  async resolve(
    repoId: string,
    requestedRevision = 'main',
    allowPatterns?: string[] | null,
  ): Promise<ResolveResult> {
    const url = `${this.apiBase}/api/models/${repoId}?revision=${encodeURIComponent(requestedRevision)}`;
    const response = await this.fetchFn(url, {
      headers: { accept: 'application/json' },
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(
        `HF resolve 失败 (${response.status}): ${text.slice(0, 200) || repoId}`,
      );
    }
    const data = (await response.json()) as {
      sha?: string;
      siblings?: Array<{ rfilename: string; size?: number; sha256?: string | null }>;
    };
    if (!data.sha) {
      throw new Error(`HF resolve 响应缺少 sha（revision 不存在?）: ${repoId}@${requestedRevision}`);
    }
    let files: ResolvedFile[] = (data.siblings ?? []).map((s) => ({
      rfilename: s.rfilename,
      size: Number(s.size ?? 0),
      sha256: s.sha256 ?? null,
    }));
    if (allowPatterns && allowPatterns.length > 0) {
      files = files.filter((f) =>
        allowPatterns.some((p) => globMatch(p, f.rfilename)));
    }
    return {
      repoId,
      requestedRevision,
      resolvedRevision: data.sha,
      files,
    };
  }
}
