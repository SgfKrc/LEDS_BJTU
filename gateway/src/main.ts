/**
 * QLH API 网关 — 启动入口（T2 起完整启用，T1 仅为占位）。
 */
import 'reflect-metadata';
import { createApp } from './app';

async function bootstrap(): Promise<void> {
  const app = await createApp();
  const port = Number(process.env.QLH_API_PORT || 8000);
  const host = process.env.QLH_API_HOST?.trim() || '::';
  await app.listen({ port, host, ipv6Only: false });
}

bootstrap().catch((err) => {
  console.error('gateway bootstrap failed:', err);
  process.exit(1);
});
