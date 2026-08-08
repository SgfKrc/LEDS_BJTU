import {
  existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import {
  DownloadResponseInfo, HfDownloader,
} from '../src/data/hf-downloader';
import { ModelHttpClient } from '../src/data/model-http-client';

function tempFile(): { dir: string; file: string } {
  const dir = mkdtempSync(join(tmpdir(), 'qlh-hf-downloader-'));
  return { dir, file: join(dir, 'artifact.bin') };
}

describe('MODEL-FLEET M3 strict Range downloader', () => {
  it('appends a validated 206 response and exposes response metadata', async () => {
    const { dir, file } = tempFile();
    writeFileSync(file, 'abc');
    let info: DownloadResponseInfo | null = null;
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response('def', {
        status: 206,
        headers: {
          'content-length': '3',
          'content-range': 'bytes 3-5/6',
        },
      }),
    });
    const downloader = new HfDownloader(http, { progressThrottleMs: 0 });
    await downloader.downloadFile('org/model', 'revision', 'artifact.bin', file, {
      expectedSize: 6,
      onResponse: (response) => { info = response; },
    });
    expect(readFileSync(file, 'utf-8')).toBe('abcdef');
    expect(info).toEqual({
      status: 206,
      requestedStartBytes: 3,
      contentRange: 'bytes 3-5/6',
      totalBytes: 6,
    });
    rmSync(dir, { recursive: true, force: true });
  });

  it('rejects a server that ignores Range instead of appending a full 200 body', async () => {
    const { dir, file } = tempFile();
    writeFileSync(file, 'abc');
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response('abcdef', { status: 200 }),
    });
    const downloader = new HfDownloader(http);
    await expect(downloader.downloadFile(
      'org/model', 'revision', 'artifact.bin', file, { expectedSize: 6 },
    )).rejects.toThrow('续传请求未返回 206');
    expect(readFileSync(file, 'utf-8')).toBe('abc');
    rmSync(dir, { recursive: true, force: true });
  });

  it('rejects a mismatched Content-Range before writing', async () => {
    const { dir, file } = tempFile();
    writeFileSync(file, 'abc');
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response('def', {
        status: 206,
        headers: { 'content-range': 'bytes 2-4/6' },
      }),
    });
    const downloader = new HfDownloader(http);
    await expect(downloader.downloadFile(
      'org/model', 'revision', 'artifact.bin', file, { expectedSize: 6 },
    )).rejects.toThrow('续传响应范围错位');
    expect(readFileSync(file, 'utf-8')).toBe('abc');
    rmSync(dir, { recursive: true, force: true });
  });

  it('detects a truncated 206 body', async () => {
    const { dir, file } = tempFile();
    writeFileSync(file, 'abc');
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => new Response('de', {
        status: 206,
        headers: { 'content-range': 'bytes 3-5/6' },
      }),
    });
    const downloader = new HfDownloader(http);
    await expect(downloader.downloadFile(
      'org/model', 'revision', 'artifact.bin', file, { expectedSize: 6 },
    )).rejects.toThrow('续传响应体长度不匹配');
    expect(readFileSync(file, 'utf-8')).toBe('abcde');
    rmSync(dir, { recursive: true, force: true });
  });

  it('checks the actual file size before treating a download as complete', async () => {
    const { dir, file } = tempFile();
    writeFileSync(file, 'abc');
    let requests = 0;
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => {
        requests += 1;
        return new Response('unused');
      },
    });
    const downloader = new HfDownloader(http);
    await expect(downloader.downloadFile(
      'org/model', 'revision', 'artifact.bin', file,
      { startBytes: 6, expectedSize: 6 },
    )).rejects.toThrow('续传错位');
    expect(requests).toBe(0);
    rmSync(dir, { recursive: true, force: true });
  });

  it('materializes a legitimate zero-byte destination without a request', async () => {
    const { dir, file } = tempFile();
    let requests = 0;
    const http = new ModelHttpClient({
      proxyUrl: null,
      fetchFn: async () => {
        requests += 1;
        return new Response('unused');
      },
    });
    await new HfDownloader(http).downloadFile(
      'org/model', 'revision', 'empty.txt', file, { expectedSize: 0 },
    );
    expect(existsSync(file)).toBe(true);
    expect(readFileSync(file).length).toBe(0);
    expect(requests).toBe(0);
    rmSync(dir, { recursive: true, force: true });
  });
});
