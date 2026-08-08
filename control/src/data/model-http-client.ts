import { Injectable, OnApplicationShutdown, Optional } from '@nestjs/common';
import {
  Dispatcher, ProxyAgent, fetch as undiciFetch,
} from 'undici';

type ModelRequestInit = RequestInit & { dispatcher?: Dispatcher };
export type ModelFetchFn = (
  input: RequestInfo | URL,
  init?: ModelRequestInit,
) => Promise<Response>;

export interface ModelHttpClientOptions {
  fetchFn?: ModelFetchFn;
  env?: NodeJS.ProcessEnv;
  proxyUrl?: string | null;
}

export interface ModelProxyStatus {
  configured: boolean;
  source: 'QLH_HTTP_PROXY';
  endpoint: string | null;
}

export function normalizeModelProxyUrl(raw: string | null | undefined): string | null {
  const value = raw?.trim();
  if (!value) return null;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error('QLH_HTTP_PROXY is not a valid URL');
  }
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('QLH_HTTP_PROXY must use http:// or https://');
  }
  if (url.username || url.password) {
    throw new Error('QLH_HTTP_PROXY must not contain embedded credentials');
  }
  if ((url.pathname && url.pathname !== '/') || url.search || url.hash) {
    throw new Error('QLH_HTTP_PROXY must be an origin without path/query/fragment');
  }
  return url.origin;
}

@Injectable()
export class ModelHttpClient implements OnApplicationShutdown {
  private readonly fetchFn: ModelFetchFn;
  private readonly proxyUrl: string | null;
  private proxyAgent: ProxyAgent | null = null;

  constructor(@Optional() options: ModelHttpClientOptions = {}) {
    this.fetchFn = options.fetchFn
      ?? (undiciFetch as unknown as ModelFetchFn);
    const configured = options.proxyUrl !== undefined
      ? options.proxyUrl
      : (options.env ?? process.env).QLH_HTTP_PROXY;
    this.proxyUrl = normalizeModelProxyUrl(configured);
  }

  proxyStatus(): ModelProxyStatus {
    return {
      configured: this.proxyUrl !== null,
      source: 'QLH_HTTP_PROXY',
      endpoint: this.proxyUrl,
    };
  }

  async fetch(
    input: RequestInfo | URL,
    init: RequestInit = {},
    auth: { token?: string | null } = {},
  ): Promise<Response> {
    const headers = new Headers(init.headers);
    if (auth.token) headers.set('authorization', `Bearer ${auth.token}`);
    const request: ModelRequestInit = { ...init, headers };
    if (this.proxyUrl) {
      this.proxyAgent ??= new ProxyAgent(this.proxyUrl);
      request.dispatcher = this.proxyAgent;
    }
    return this.fetchFn(input, request);
  }

  async onApplicationShutdown(): Promise<void> {
    if (this.proxyAgent) await this.proxyAgent.close();
    this.proxyAgent = null;
  }
}
