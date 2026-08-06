/**
 * M3 Hugging Face 下载器 — fetch 流式下载 + Range 断点续传 + 取消。
 *
 * 下载只写 staging（artifact-store）；进度回调节流；续传用
 * `Range: bytes=<已下载>-` 并校验已存在字节。
 */
import { Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import { HfResolver } from './hf-resolver';

export interface DownloadProgress {
  bytesDownloaded: number;
  totalBytes: number;
}

export interface DownloaderOptions {
  fetchFn?: typeof fetch;
  apiBase?: string;
  /** 进度回调节流间隔（毫秒）。 */
  progressThrottleMs?: number;
}

@Injectable()
export class HfDownloader {
  private readonly fetchFn: typeof fetch;
  private readonly apiBase: string;
  private readonly throttleMs: number;

  constructor(@Optional() options: DownloaderOptions = {}) {
    this.fetchFn = options.fetchFn ?? globalThis.fetch;
    this.apiBase = options.apiBase ?? 'https://huggingface.co';
    this.throttleMs = options.progressThrottleMs ?? 250;
  }

  fileUrl(repoId: string, revision: string, filename: string): string {
    return `${this.apiBase}/${repoId}/resolve/${encodeURIComponent(revision)}/${filename}`;
  }

  /**
   * 下载单文件到 destPath。
   * startBytes > 0 → Range 续传（已存在字节必须与 startBytes 一致，否则重建）。
   */
  async downloadFile(
    repoId: string,
    revision: string,
    filename: string,
    destPath: string,
    opts: {
      onProgress?: (p: DownloadProgress) => void;
      signal?: AbortSignal;
      startBytes?: number;
      expectedSize?: number;
    } = {},
  ): Promise<void> {
    fs.mkdirSync(require('path').dirname(destPath), { recursive: true });
    const startBytes = opts.startBytes ?? (fs.existsSync(destPath) ? fs.statSync(destPath).size : 0);

    // Range 续传：若已有字节与期望大小一致则跳过（已完整）
    if (opts.expectedSize !== undefined && startBytes >= opts.expectedSize) {
      return;
    }

    const headers: Record<string, string> = {};
    if (startBytes > 0) headers['range'] = `bytes=${startBytes}-`;
    const response = await this.fetchFn(this.fileUrl(repoId, revision, filename), {
      headers,
      signal: opts.signal,
    });
    if (response.status !== 200 && response.status !== 206) {
      throw new Error(
        `HF 下载失败 (${response.status}): ${filename}（${await response.text().catch(() => '')}）`,
      );
    }
    const totalBytes = startBytes + Number(
      response.headers.get('content-length') ?? 0,
    );
    const body = response.body;
    if (!body) {
      throw new Error(`HF 下载无响应体: ${filename}`);
    }

    // append 模式续传（保证与 startBytes 对齐）
    if (startBytes > 0) {
      const actual = fs.existsSync(destPath) ? fs.statSync(destPath).size : 0;
      if (actual !== startBytes) {
        throw new Error(`续传错位: 期望 ${startBytes}，实际 ${actual}`);
      }
    }
    const fd = fs.openSync(destPath, startBytes > 0 ? 'a' : 'w');
    const reader = body.getReader();
    let downloaded = startBytes;
    let lastEmit = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        fs.writeSync(fd, value);
        downloaded += value.length;
        const now = Date.now();
        if (opts.onProgress && now - lastEmit >= this.throttleMs) {
          lastEmit = now;
          opts.onProgress({ bytesDownloaded: downloaded, totalBytes });
        }
      }
      if (opts.onProgress) {
        opts.onProgress({ bytesDownloaded: downloaded, totalBytes });
      }
      if (opts.expectedSize !== undefined && downloaded !== opts.expectedSize) {
        throw new Error(
          `下载大小不匹配: ${filename} 期望 ${opts.expectedSize}，实际 ${downloaded}`,
        );
      }
    } finally {
      fs.closeSync(fd);
    }
  }
}

export { HfResolver };
