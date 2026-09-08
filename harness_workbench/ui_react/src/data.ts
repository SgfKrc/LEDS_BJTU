export type ConnectionState = 'online' | 'offline' | 'fixture' | 'checking';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  meta?: string;
}

export interface RagHit {
  source_id: string;
  chunk_id: string;
  title: string;
  ordinal: number;
  text: string;
  score: number;
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
  const payload = (await response.json().catch(() => ({}))) as T & { error?: { message?: string } };
  if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
  return payload;
}

export async function checkHealth(): Promise<{ status: string; backend?: string }> {
  return requestJson('/healthz');
}

export async function completeChat(model: string, messages: ChatMessage[]): Promise<{ content: string }> {
  return requestJson('/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model,
      messages: messages.map(({ role, content }) => ({ role, content })),
      max_tokens: 512,
      temperature: 0.7,
    }),
  });
}

export async function searchRag(query: string): Promise<{ hits: RagHit[]; context: { omitted_count: number } }> {
  return requestJson('/v1/rag/search', {
    method: 'POST',
    body: JSON.stringify({ query, limit: 6, max_chars: 5000 }),
  });
}
