/**
 * QLH control-svc 启动入口（:8030，QLH_CONTROL_PORT）。
 */
import 'reflect-metadata';
import { createApp } from './app';

async function bootstrap(): Promise<void> {
  const app = await createApp();
  const port = Number(process.env.QLH_CONTROL_PORT || 8030);
  // control-svc is an internal dependency; the gateway is the authenticated
  // public boundary. Remote exposure must be an explicit deployment choice.
  const host = process.env.QLH_CONTROL_HOST?.trim() || '127.0.0.1';
  await app.listen({ port, host, ipv6Only: false });
  // eslint-disable-next-line no-console
  console.log(`CONTROL_SVC_LISTENING:${port}`);
}

bootstrap().catch((err) => {
  console.error('control-svc bootstrap failed:', err);
  process.exit(1);
});
