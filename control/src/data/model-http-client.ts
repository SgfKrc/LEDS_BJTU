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
  proxyProvider?: () => { url: string; source: 'user' } | null;
}

export interface ModelProxyStatus {
  configured: boolean;
  source: 'QLH_HTTP_PROXY' | 'user' | 'direct';
  endpoint: string | null;
}

export function normalizeModelProxyUrl(raw: string | null | undefined): string | null {
  const value = raw?.trim();
  if (!value) return null;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error('model proxy is not a valid URL');
  }
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('model proxy must use http:// or https://');
  }
  if (url.username || url.password) {
    throw new Error('model proxy must not contain embedded credentials');
  }
  if ((url.pathname && url.pathname !== '/') || url.search || url.hash) {
    throw new Error('model proxy must be an origin without path/query/fragment');
  }
  return url.origin;
}

@Injectable()
export class ModelHttpClient implements OnApplicationShutdown {
  private readonly fetchFn: ModelFetchFn;
  private readonly env: NodeJS.ProcessEnv;
  private readonly explicitProxyUrl: string | null | undefined;
  private readonly proxyProvider?: ModelHttpClientOptions['proxyProvider'];
  private readonly proxyAgents = new Map<string, ProxyAgent>();

  constructor(@Optional() options: ModelHttpClientOptions = {}) {
    this.fetchFn = options.fetchFn
      ?? (undiciFetch as unknown as ModelFetchFn);
    this.env = options.env ?? process.env;
    this.explicitProxyUrl = options.proxyUrl === undefined
      ? undefined
      : normalizeModelProxyUrl(options.proxyUrl);
    this.proxyProvider = options.proxyProvider;
  }

  proxyStatus(): ModelProxyStatus {
    const selected = this.selectProxy();
    return {
      configured: selected !== null,
      source: selected?.source ?? 'direct',
      endpoint: selected?.url ?? null,
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
    const selected = this.selectProxy();
    if (selected) {
      let agent = this.proxyAgents.get(selected.url);
      if (!agent) {
        agent = new ProxyAgent(selected.url);
        this.proxyAgents.set(selected.url, agent);
      }
      request.dispatcher = agent;
    }
    return this.fetchFn(input, request);
  }

  async onApplicationShutdown(): Promise<void> {
    await Promise.all([...this.proxyAgents.values()].map((agent) => agent.close()));
    this.proxyAgents.clear();
  }

  private selectProxy(): { url: string; source: 'QLH_HTTP_PROXY' | 'user' } | null {
    if (this.explicitProxyUrl !== undefined) {
      return this.explicitProxyUrl
        ? { url: this.explicitProxyUrl, source: 'QLH_HTTP_PROXY' }
        : null;
    }
    const environmentUrl = normalizeModelProxyUrl(this.env.QLH_HTTP_PROXY);
    if (environmentUrl) {
      return { url: environmentUrl, source: 'QLH_HTTP_PROXY' };
    }
    const user = this.proxyProvider?.() ?? null;
    if (!user) return null;
    const url = normalizeModelProxyUrl(user.url);
    return url ? { url, source: 'user' } : null;
  }
}
