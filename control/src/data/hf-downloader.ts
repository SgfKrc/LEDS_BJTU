/**
 * M3 Hugging Face 下载器 — fetch 流式下载 + Range 断点续传 + 取消。
 *
 * 下载只写 staging（artifact-store）；进度回调节流；续传用
 * `Range: bytes=<已下载>-` 并校验已存在字节。
 */
import { Injectable, Optional } from '@nestjs/common';
import * as fs from 'fs';
import { HfResolver } from './hf-resolver';
import { ModelHttpClient } from './model-http-client';

export interface DownloadProgress {
  bytesDownloaded: number;
  totalBytes: number;
}

export interface DownloadResponseInfo {
  status: number;
  requestedStartBytes: number;
  contentRange: string | null;
  totalBytes: number;
}

export interface DownloaderOptions {
  apiBase?: string;
  /** 进度回调节流间隔（毫秒）。 */
  progressThrottleMs?: number;
}

@Injectable()
export class HfDownloader {
  private readonly http: ModelHttpClient;
  private readonly apiBase: string;
  private readonly throttleMs: number;

  constructor(
    @Optional() http?: ModelHttpClient,
    @Optional() options: DownloaderOptions = {},
  ) {
    this.http = http ?? new ModelHttpClient();
    this.apiBase = options.apiBase ?? 'https://huggingface.co';
    this.throttleMs = options.progressThrottleMs ?? 250;
  }

  fileUrl(
    repoId: string,
    revision: string,
    filename: string,
    apiBaseOverride?: string,
  ): string {
    const apiBase = (apiBaseOverride ?? this.apiBase).replace(/\/+$/, '');
    return `${apiBase}/${repoId}/resolve/${encodeURIComponent(revision)}/${filename}`;
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
      onResponse?: (response: DownloadResponseInfo) => void;
      signal?: AbortSignal;
      startBytes?: number;
      expectedSize?: number;
      apiBase?: string;
      token?: string | null;
    } = {},
  ): Promise<void> {
    fs.mkdirSync(require('path').dirname(destPath), { recursive: true });
    const destinationExists = fs.existsSync(destPath);
    const actualBytes = destinationExists ? fs.statSync(destPath).size : 0;
    const startBytes = opts.startBytes ?? actualBytes;
    if (!Number.isSafeInteger(startBytes) || startBytes < 0) {
      throw new Error(`续传起点无效: ${startBytes}`);
    }
    if (actualBytes !== startBytes) {
      throw new Error(`续传错位: 期望 ${startBytes}，实际 ${actualBytes}`);
    }
    if (opts.expectedSize !== undefined && (
      !Number.isSafeInteger(opts.expectedSize) || opts.expectedSize < 0
    )) {
      throw new Error(`预期大小无效: ${opts.expectedSize}`);
    }

    if (opts.expectedSize !== undefined) {
      if (startBytes === opts.expectedSize) {
        if (!destinationExists) fs.writeFileSync(destPath, Buffer.alloc(0));
        return;
      }
      if (startBytes > opts.expectedSize) {
        throw new Error(
          `已有文件超过预期大小: ${filename} 期望 ${opts.expectedSize}，实际 ${startBytes}`,
        );
      }
    }

    const headers: Record<string, string> = {};
    if (startBytes > 0) headers['range'] = `bytes=${startBytes}-`;
    const response = await this.http.fetch(this.fileUrl(
      repoId, revision, filename, opts.apiBase,
    ), {
      headers,
      signal: opts.signal,
    }, { token: opts.token });
    if (response.status !== 200 && response.status !== 206) {
      throw new Error(
        `HF 下载失败 (${response.status}): ${filename}（${await response.text().catch(() => '')}）`,
      );
    }
    if (startBytes > 0 && response.status !== 206) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`续传请求未返回 206: ${filename}（实际 ${response.status}）`);
    }

    const contentLengthRaw = response.headers.get('content-length');
    const contentLength = contentLengthRaw === null ? null : Number(contentLengthRaw);
    if (contentLength !== null && (
      !Number.isSafeInteger(contentLength) || contentLength < 0
    )) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`响应 Content-Length 无效: ${filename}`);
    }

    const contentRange = response.headers.get('content-range');
    let rangeEnd: number | null = null;
    let totalBytes = contentLength === null
      ? (opts.expectedSize ?? startBytes)
      : startBytes + contentLength;
    if (response.status === 206) {
      const match = /^bytes (\d+)-(\d+)\/(\d+|\*)$/i.exec(contentRange ?? '');
      if (!match) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(`续传响应缺少有效 Content-Range: ${filename}`);
      }
      const rangeStart = Number(match[1]);
      rangeEnd = Number(match[2]);
      const completeSize = match[3] === '*' ? null : Number(match[3]);
      if (rangeStart !== startBytes || rangeEnd < rangeStart) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(
          `续传响应范围错位: ${filename} 期望从 ${startBytes} 开始，实际 ${contentRange}`,
        );
      }
      const rangeLength = rangeEnd - rangeStart + 1;
      if (contentLength !== null && contentLength !== rangeLength) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(`续传响应长度与 Content-Range 不一致: ${filename}`);
      }
      if (completeSize !== null) {
        if (completeSize <= rangeEnd) {
          await response.body?.cancel().catch(() => undefined);
          throw new Error(`续传响应总大小无效: ${filename}`);
        }
        if (opts.expectedSize !== undefined && completeSize !== opts.expectedSize) {
          await response.body?.cancel().catch(() => undefined);
          throw new Error(
            `续传响应总大小不匹配: ${filename} 期望 ${opts.expectedSize}，实际 ${completeSize}`,
          );
        }
        totalBytes = completeSize;
      } else {
        totalBytes = rangeEnd + 1;
      }
    }

    opts.onResponse?.({
      status: response.status,
      requestedStartBytes: startBytes,
      contentRange,
      totalBytes,
    });
    const body = response.body;
    if (!body) {
      throw new Error(`HF 下载无响应体: ${filename}`);
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
      if (rangeEnd !== null && downloaded !== rangeEnd + 1) {
        throw new Error(
          `续传响应体长度不匹配: ${filename} 期望结束于 ${rangeEnd + 1}，实际 ${downloaded}`,
        );
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
