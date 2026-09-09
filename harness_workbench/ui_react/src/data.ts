export type ConnectionState = 'online' | 'offline' | 'fixture' | 'checking';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  meta?: string;
}

export interface SessionSummary {
  session_id: string;
  owner_scope: string;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface SessionDetail {
  session: SessionSummary;
  messages: Array<{
    message_id: string;
    session_id: string;
    ordinal: number;
    role: 'user' | 'assistant' | 'system' | 'tool';
    content: string;
    metadata: Record<string, unknown>;
  }>;
  assets: Array<Record<string, unknown>>;
}

export interface RagHit {
  source_id: string;
  chunk_id: string;
  title: string;
  ordinal: number;
  text: string;
  score: number;
}

export interface RagContext {
  text: string;
  citations: Array<{ source_id: string; chunk_id: string; title: string; ordinal: number }>;
  included_count: number;
  omitted_count: number;
  truncated: boolean;
}

export interface ImageCapabilities {
  backend: string;
  model_ids: string[];
  supports_txt2img: boolean;
  supports_edit: boolean;
  supports_distributed: boolean;
  runtime_available: boolean;
  evidence: Record<string, unknown>;
}

export interface ImageResult {
  asset_id?: string;
  url?: string;
  b64_json?: string;
  metadata?: Record<string, unknown>;
}

export interface HarnessModel {
  id: string;
  object?: string;
  owned_by?: string;
  created?: number;
}

export interface ModelProfileSummary {
  model_id: string;
  revision: string;
  backend: string;
  roles: string[];
  aliases?: string[];
  status: string;
  production_eligible: boolean;
  context: Record<string, unknown>;
  generation: Record<string, unknown>;
  adaptation: Record<string, unknown>;
  capabilities: Record<string, { status: string; evidence: string[] }>;
  evidence: Record<string, unknown>;
}

export class HarnessApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly retryable: boolean;

  constructor(message: string, options: { status: number; code?: string | null; retryable?: boolean }) {
    super(message);
    this.name = 'HarnessApiError';
    this.status = options.status;
    this.code = options.code || null;
    this.retryable = options.retryable === true;
  }
}

export interface MCPToolSchema {
  type: 'object';
  properties?: Record<string, { type: string; description?: string; enum?: Array<string | number | boolean>; minimum?: number; maximum?: number; maxLength?: number }>;
  required?: string[];
  additionalProperties?: boolean;
}

export interface MCPTool {
  name: string;
  description: string;
  inputSchema: MCPToolSchema;
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean; openWorldHint?: boolean };
  _meta?: { qlh?: { source?: string; capability?: string; configured?: boolean; server_id?: string; remote_name?: string; transport?: string } };
}

export interface MCPManifest {
  schema: string;
  server: { protocolVersion?: string; serverInfo?: { name?: string; version?: string }; capabilities?: Record<string, unknown> };
  tools: MCPTool[];
  external_mcp: { configurations: Array<Record<string, unknown>>; configuration_only: boolean; real_connections: boolean };
  transports: { jsonrpc: { method: string; path: string }; call: { method: string; path: string }; stdio: { available: boolean; command: string }; sse: { available: boolean; mode: string; path: string } };
}

export interface MCPCallResponse {
  jsonrpc: string;
  id: string | number | null;
  result?: { content?: Array<{ type: string; text: string }>; isError?: boolean; structuredContent?: Record<string, unknown> };
  error?: { code: number; message: string };
}

const apiBase = (import.meta.env.VITE_HARNESS_API_BASE ?? '').replace(/\/$/, '');

export function apiUrl(path: string): string {
  return `${apiBase}${path}`;
}

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  const payload = (await response.json().catch(() => ({}))) as T & {
    error?: { message?: string; code?: string; retryable?: boolean };
  };
  if (!response.ok) {
    throw new HarnessApiError(payload.error?.message || `HTTP ${response.status}`, {
      status: response.status,
      code: payload.error?.code || null,
      retryable: payload.error?.retryable === true,
    });
  }
  return payload;
}

export async function checkHealth(): Promise<{ status: string; backend?: string }> {
  return requestJson('/healthz');
}

export async function listModels(): Promise<{ object?: string; data: HarnessModel[] }> {
  return requestJson('/v1/models');
}

export async function listModelProfiles(): Promise<{ schema: string; profiles: ModelProfileSummary[] }> {
  return requestJson('/v1/model-profiles');
}

export async function completeChat(model: string, messages: ChatMessage[]): Promise<{ content: string }> {
  const response = await requestJson<{ choices?: Array<{ message?: { content?: string } }> }>('/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model,
      messages: messages.map(({ role, content }) => ({ role, content })),
      max_tokens: 512,
      temperature: 0.7,
    }),
  });
  return { content: response.choices?.[0]?.message?.content || '' };
}

export async function listSessions(): Promise<{ sessions: SessionSummary[] }> {
  return requestJson('/v1/sessions?owner_scope=local&limit=50');
}

export async function createSession(title = 'New session'): Promise<SessionSummary> {
  return requestJson('/v1/sessions', {
    method: 'POST',
    body: JSON.stringify({ owner_scope: 'local', title }),
  });
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return requestJson(`/v1/sessions/${encodeURIComponent(sessionId)}?owner_scope=local`);
}

export async function appendSessionMessage(sessionId: string, message: Pick<ChatMessage, 'role' | 'content'>, metadata: Record<string, unknown> = {}): Promise<void> {
  await requestJson(`/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: 'POST',
    body: JSON.stringify({ owner_scope: 'local', role: message.role, content: message.content, metadata }),
  });
}

export async function streamChat(
  model: string,
  messages: ChatMessage[],
  signal: AbortSignal,
  onDelta: (delta: string) => void,
): Promise<void> {
  const response = await fetch(apiUrl('/v1/chat/completions'), {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      model,
      messages: messages.map(({ role, content }) => ({ role, content })),
      stream: true,
      max_tokens: 512,
      temperature: 0.7,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { error?: { message?: string; code?: string; retryable?: boolean } };
    throw new HarnessApiError(payload.error?.message || `HTTP ${response.status}`, {
      status: response.status,
      code: payload.error?.code || null,
      retryable: payload.error?.retryable === true,
    });
  }
  if (!response.body) throw new Error('stream response body is unavailable');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;
  const consume = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('\n');
    if (!data || data === '[DONE]') {
      if (data === '[DONE]') finished = true;
      return;
    }
    const payload = JSON.parse(data) as { error?: { message?: string; code?: string; retryable?: boolean }; choices?: Array<{ delta?: { content?: string } }> };
    if (payload.error?.message) {
      throw new HarnessApiError(payload.error.message, {
        status: 200,
        code: payload.error.code || null,
        retryable: payload.error.retryable === true,
      });
    }
    const delta = payload.choices?.[0]?.delta?.content;
    if (delta) onDelta(delta);
  };

  while (!finished) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      consume(block);
      if (finished) break;
    }
    if (done) {
      if (buffer.trim()) consume(buffer);
      break;
    }
  }
}

export async function getRagHealth(): Promise<Record<string, unknown>> {
  return requestJson('/v1/rag/health');
}

export async function searchRag(query: string): Promise<{ hits: RagHit[]; context: RagContext }> {
  return requestJson('/v1/rag/search', {
    method: 'POST',
    body: JSON.stringify({ owner_scope: 'local', query, limit: 6, max_chars: 5000 }),
  });
}

export async function getImageCapabilities(): Promise<ImageCapabilities> {
  return requestJson('/v1/images/capabilities');
}

export async function generateImage(payload: {
  prompt: string;
  model?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  seed?: number;
}): Promise<{ data: ImageResult[] }> {
  return requestJson('/v1/images/generations', {
    method: 'POST',
    body: JSON.stringify({ ...payload, user: 'local', response_format: 'url' }),
  });
}

export async function attachSessionAsset(sessionId: string, assetId: string, metadata: Record<string, unknown> = {}): Promise<void> {
  await requestJson(`/v1/sessions/${encodeURIComponent(sessionId)}/assets`, {
    method: 'POST',
    body: JSON.stringify({ owner_scope: 'local', asset_id: assetId, kind: 'image', metadata }),
  });
}

export async function getMcpManifest(): Promise<MCPManifest> {
  return requestJson('/v1/mcp/manifest');
}

export async function callMcpTool(name: string, argumentsValue: Record<string, unknown>): Promise<MCPCallResponse> {
  return requestJson('/v1/mcp/call', {
    method: 'POST',
    body: JSON.stringify({ name, arguments: argumentsValue }),
  });
}
