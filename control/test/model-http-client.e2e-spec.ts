import {
  ModelHttpClient, normalizeModelProxyUrl,
} from '../src/data/model-http-client';
import { ProxyAgent } from 'undici';

describe('MODEL-FLEET M3 model-http-client', () => {
  describe('normalizeModelProxyUrl', () => {
    it('accepts http/https origins and normalizes to origin', () => {
      expect(normalizeModelProxyUrl('http://proxy.example:8080'))
        .toBe('http://proxy.example:8080');
      expect(normalizeModelProxyUrl(' https://proxy.example/ '))
        .toBe('https://proxy.example');
    });

    it('returns null for blank or missing values', () => {
      expect(normalizeModelProxyUrl('')).toBeNull();
      expect(normalizeModelProxyUrl('   ')).toBeNull();
      expect(normalizeModelProxyUrl(null)).toBeNull();
      expect(normalizeModelProxyUrl(undefined)).toBeNull();
    });

    it('rejects non-URL, non-http(s), embedded credentials and path/query', () => {
      expect(() => normalizeModelProxyUrl('not a url'))
        .toThrow('model proxy is not a valid URL');
      expect(() => normalizeModelProxyUrl('ftp://proxy.example'))
        .toThrow('must use http:// or https://');
      expect(() => normalizeModelProxyUrl('http://user:pass@proxy.example:8080'))
        .toThrow('must not contain embedded credentials');
      expect(() => normalizeModelProxyUrl('http://proxy.example/path'))
        .toThrow('must be an origin without path/query/fragment');
      expect(() => normalizeModelProxyUrl('http://proxy.example?x=1'))
        .toThrow('must be an origin without path/query/fragment');
      expect(() => normalizeModelProxyUrl('http://proxy.example#frag'))
        .toThrow('must be an origin without path/query/fragment');
    });
  });

  it('reads QLH_HTTP_PROXY from the injected environment and reports status', () => {
    const client = new ModelHttpClient({
      env: { QLH_HTTP_PROXY: 'http://127.0.0.1:7897' },
      fetchFn: async () => new Response('{}'),
    });
    expect(client.proxyStatus()).toEqual({
      configured: true,
      source: 'QLH_HTTP_PROXY',
      endpoint: 'http://127.0.0.1:7897',
    });
    expect(JSON.stringify(client.proxyStatus())).not.toContain('secret');
  });

  it('stays unconfigured when QLH_HTTP_PROXY is absent', () => {
    const client = new ModelHttpClient({
      env: {},
      fetchFn: async () => new Response('{}'),
    });
    expect(client.proxyStatus()).toEqual({
      configured: false,
      source: 'direct',
      endpoint: null,
    });
  });

  it('injects Bearer token without leaking it into URL, headers echo or status', async () => {
    const secret = 'hf_http_client_secret';
    let captured: RequestInit & { dispatcher?: unknown } = {};
    const client = new ModelHttpClient({
      env: { QLH_HTTP_PROXY: 'http://proxy.example:8080' },
      fetchFn: async (_url, init) => {
        captured = init ?? {};
        return new Response('{}', { status: 200 });
      },
    });
    const response = await client.fetch(
      'https://huggingface.co/api/models/org/model',
      { headers: { accept: 'application/json' } },
      { token: secret },
    );
    expect(response.status).toBe(200);
    expect(new Headers(captured.headers).get('authorization')).toBe(`Bearer ${secret}`);
    expect(new Headers(captured.headers).get('accept')).toBe('application/json');
    expect(JSON.stringify(captured)).not.toContain(secret);
    expect(captured.dispatcher).toBeInstanceOf(ProxyAgent);
  });

  it('does not attach a proxy dispatcher when unconfigured', async () => {
    let captured: RequestInit & { dispatcher?: unknown } = {};
    const client = new ModelHttpClient({
      env: {},
      fetchFn: async (_url, init) => {
        captured = init ?? {};
        return new Response('{}');
      },
    });
    await client.fetch('https://huggingface.co/', {}, { token: 't' });
    expect(captured.dispatcher).toBeUndefined();
  });

  it('shuts down and closes the shared ProxyAgent', async () => {
    const client = new ModelHttpClient({
      env: { QLH_HTTP_PROXY: 'http://proxy.example:8080' },
      fetchFn: async () => new Response('{}'),
    });
    await client.fetch('https://huggingface.co/');
    await client.onApplicationShutdown();
    // 再次 fetch 会重建代理（懒加载），不再复用已关闭实例
    await client.fetch('https://huggingface.co/');
    await client.onApplicationShutdown();
  });
});
