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
  attempt: number;
  status: number;
  requestedStartBytes: number;
  contentRange: string | null;
  totalBytes: number;
}

export interface DownloadRetryInfo {
  attempt: number;
  nextAttempt: number;
  retryCount: number;
  resumedBytes: number;
  errorCode: string;
}

export interface DownloaderOptions {
  apiBase?: string;
  /** 进度回调节流间隔（毫秒）。 */
  progressThrottleMs?: number;
  /** Includes the initial request. Only transport failures are retried. */
  maxAttempts?: number;
  retryBaseDelayMs?: number;
  requestTimeoutMs?: number;
}

export class HfDownloadTransportError extends Error {
  constructor(
    readonly code: string,
    message: string,
    options: { cause?: unknown } = {},
  ) {
    super(message);
    this.name = 'HfDownloadTransportError';
    if (options.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

@Injectable()
export class HfDownloader {
  private readonly http: ModelHttpClient;
  private readonly apiBase: string;
  private readonly throttleMs: number;
  private readonly maxAttempts: number;
  private readonly retryBaseDelayMs: number;
  private readonly requestTimeoutMs: number;

  constructor(
    @Optional() http?: ModelHttpClient,
    @Optional() options: DownloaderOptions = {},
  ) {
    this.http = http ?? new ModelHttpClient();
    this.apiBase = options.apiBase ?? 'https://huggingface.co';
    this.throttleMs = options.progressThrottleMs ?? 250;
    this.maxAttempts = options.maxAttempts ?? 4;
    this.retryBaseDelayMs = options.retryBaseDelayMs ?? 250;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 30_000;
    if (!Number.isInteger(this.maxAttempts) || this.maxAttempts < 1) {
      throw new Error(`下载尝试次数无效: ${this.maxAttempts}`);
    }
    if (!Number.isFinite(this.retryBaseDelayMs) || this.retryBaseDelayMs < 0) {
      throw new Error(`下载重试间隔无效: ${this.retryBaseDelayMs}`);
    }
    if (!Number.isFinite(this.requestTimeoutMs) || this.requestTimeoutMs <= 0) {
      throw new Error(`下载请求超时无效: ${this.requestTimeoutMs}`);
    }
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
      onRetry?: (retry: DownloadRetryInfo) => void;
      signal?: AbortSignal;
      startBytes?: number;
      expectedSize?: number;
      apiBase?: string;
      token?: string | null;
    } = {},
  ): Promise<void> {
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      try {
        await this.downloadAttempt(
          repoId, revision, filename, destPath, attempt,
          attempt === 1 ? opts : { ...opts, startBytes: undefined },
        );
        return;
      } catch (error) {
        if (!(error instanceof HfDownloadTransportError)
            || opts.signal?.aborted
            || attempt >= this.maxAttempts) {
          throw error;
        }
        const resumedBytes = fs.existsSync(destPath) ? fs.statSync(destPath).size : 0;
        opts.onRetry?.({
          attempt,
          nextAttempt: attempt + 1,
          retryCount: attempt,
          resumedBytes,
          errorCode: error.code,
        });
        await this.waitBeforeRetry(attempt, opts.signal);
      }
    }
  }

  private async downloadAttempt(
    repoId: string,
    revision: string,
    filename: string,
    destPath: string,
    attempt: number,
    opts: {
      onProgress?: (p: DownloadProgress) => void;
      onResponse?: (response: DownloadResponseInfo) => void;
      onRetry?: (retry: DownloadRetryInfo) => void;
      signal?: AbortSignal;
      startBytes?: number;
      expectedSize?: number;
      apiBase?: string;
      token?: string | null;
    },
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
    const timeoutController = new AbortController();
    const timeout = setTimeout(() => {
      timeoutController.abort(new Error('download response timeout'));
    }, this.requestTimeoutMs);
    const requestSignal = opts.signal
      ? AbortSignal.any([opts.signal, timeoutController.signal])
      : timeoutController.signal;
    let response: Response;
    try {
      response = await this.http.fetch(this.fileUrl(
        repoId, revision, filename, opts.apiBase,
      ), {
        headers,
        signal: requestSignal,
      }, { token: opts.token });
    } catch (error) {
      if (opts.signal?.aborted) throw error;
      throw new HfDownloadTransportError(
        timeoutController.signal.aborted
          ? 'download_request_timeout' : 'download_transport_unavailable',
        `HF 下载传输失败: ${filename}`,
        { cause: error },
      );
    } finally {
      // Bound connection/response headers only; large response bodies may validly take longer.
      clearTimeout(timeout);
    }
    if (response.status !== 200 && response.status !== 206) {
      const detail = await response.text().catch(() => '');
      const message = `HF 下载失败 (${response.status}): ${filename}（${detail}）`;
      if ([408, 425, 429, 500, 502, 503, 504].includes(response.status)) {
        throw new HfDownloadTransportError('download_transient_http_status', message);
      }
      throw new Error(message);
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
      attempt,
      status: response.status,
      requestedStartBytes: startBytes,
      contentRange,
      totalBytes,
    });
    const body = response.body;
    if (!body) {
      throw new HfDownloadTransportError(
        'download_response_body_missing', `HF 下载无响应体: ${filename}`,
      );
    }

    const fd = fs.openSync(destPath, startBytes > 0 ? 'a' : 'w');
    const reader = body.getReader();
    let downloaded = startBytes;
    let lastEmit = 0;
    try {
      for (;;) {
        let read: ReadableStreamReadResult<Uint8Array>;
        try {
          read = await reader.read();
        } catch (error) {
          if (opts.signal?.aborted) throw error;
          throw new HfDownloadTransportError(
            'download_stream_interrupted', `HF 下载流中断: ${filename}`,
            { cause: error },
          );
        }
        const { done, value } = read;
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
        if (downloaded < rangeEnd + 1) {
          throw new HfDownloadTransportError(
            'download_response_truncated',
            `续传响应体提前结束: ${filename} 期望结束于 ${rangeEnd + 1}，实际 ${downloaded}`,
          );
        }
        throw new Error(
          `续传响应体长度不匹配: ${filename} 期望结束于 ${rangeEnd + 1}，实际 ${downloaded}`,
        );
      }
      if (opts.expectedSize !== undefined && downloaded !== opts.expectedSize) {
        if (downloaded < opts.expectedSize) {
          throw new HfDownloadTransportError(
            'download_response_truncated',
            `下载响应体提前结束: ${filename} 期望 ${opts.expectedSize}，实际 ${downloaded}`,
          );
        }
        throw new Error(
          `下载大小不匹配: ${filename} 期望 ${opts.expectedSize}，实际 ${downloaded}`,
        );
      }
    } finally {
      fs.closeSync(fd);
    }
  }

  private async waitBeforeRetry(attempt: number, signal?: AbortSignal): Promise<void> {
    const delayMs = this.retryBaseDelayMs * (2 ** (attempt - 1));
    if (delayMs === 0) return;
    await new Promise<void>((resolve, reject) => {
      const onAbort = (): void => {
        clearTimeout(timer);
        reject(signal?.reason instanceof Error
          ? signal.reason : new Error('下载已取消'));
      };
      const timer = setTimeout(() => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      }, delayMs);
      if (signal?.aborted) {
        onAbort();
        return;
      }
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }
}

export { HfResolver };
