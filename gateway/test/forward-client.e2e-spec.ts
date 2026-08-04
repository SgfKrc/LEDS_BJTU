/**
 * ForwardClient 转发契约单测（2026-08-05 contract_diff 复测修复回归）
 *
 * 覆盖：
 *  - 2xx JSON → 原样返回解析结果
 *  - 2xx 非 JSON（octet-stream，logs/export zip / models/download 文件流）
 *    → 返回 Buffer（此前被包装成 {detail: <文本>} 破坏文件内容）
 *  - 非 2xx JSON → HttpException(detail, status)
 *  - 上游不可达 → 502
 */
import { HttpException } from '@nestjs/common';
import { ForwardClient } from '../src/clients/forward-client';

class TestClient extends ForwardClient {
  constructor(baseUrl: string) {
    super(baseUrl, 5000);
  }
  request(
    method: string,
    path: string,
    body?: unknown,
    extraHeaders?: Record<string, string>,
  ): Promise<unknown> {
    return super.request(method, path, body, extraHeaders);
  }
}

function jsonResponse(status: number, data: unknown, contentType = 'application/json'): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': contentType },
  });
}

describe('ForwardClient（二进制透传与错误语义）', () => {
  const realFetch = global.fetch;

  afterEach(() => {
    global.fetch = realFetch;
  });

  it('2xx JSON → 解析结果原样返回', async () => {
    global.fetch = jest.fn().mockResolvedValue(
      jsonResponse(200, { status: 'ok', list: [1, 2] }),
    ) as unknown as typeof fetch;
    const client = new TestClient('http://127.0.0.1:9');
    const r = await client.request('GET', '/x');
    expect(r).toEqual({ status: 'ok', list: [1, 2] });
  });

  it('2xx octet-stream → Buffer（zip/文件流不被 JSON 包装）', async () => {
    const zip = Buffer.from('PK\x03\x04fakezip-bytes');
    global.fetch = jest.fn().mockResolvedValue(
      new Response(zip, {
        status: 200,
        headers: { 'content-type': 'application/octet-stream' },
      }),
    ) as unknown as typeof fetch;
    const client = new TestClient('http://127.0.0.1:9');
    const r = await client.request('GET', '/logs/export');
    expect(Buffer.isBuffer(r)).toBe(true);
    expect((r as Buffer).subarray(0, 2).toString()).toBe('PK');
  });

  it('非 2xx JSON → HttpException(detail, status)', async () => {
    global.fetch = jest.fn().mockResolvedValue(
      jsonResponse(404, { detail: '模型未找到' }),
    ) as unknown as typeof fetch;
    const client = new TestClient('http://127.0.0.1:9');
    // NestJS HttpException(string) 的 getResponse() 为字符串本身（
    // 网关层由 JsonDetailFilter 包装为 {detail, request_id}）
    await expect(client.request('GET', '/nope')).rejects.toMatchObject({
      status: 404,
      response: '模型未找到',
    });
  });

  it('上游不可达 → 502', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('ECONNREFUSED')) as unknown as typeof fetch;
    const client = new TestClient('http://127.0.0.1:9');
    await expect(client.request('GET', '/x')).rejects.toMatchObject({
      status: 502,
    });
  });

  it('非 2xx 非 JSON → HttpException(文本详情, status)', async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response('plain error', {
        status: 503,
        headers: { 'content-type': 'text/plain' },
      }),
    ) as unknown as typeof fetch;
    const client = new TestClient('http://127.0.0.1:9');
    await expect(client.request('GET', '/x')).rejects.toMatchObject({
      status: 503,
      response: 'plain error',
    });
  });
});
