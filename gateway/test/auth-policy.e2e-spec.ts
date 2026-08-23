import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { ExecutionContext } from '@nestjs/common';
import { ForbiddenException } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { ControlClient } from '../src/clients/control.client';
import { AuthPolicyGuard } from '../src/modules/auth/auth-policy.guard';

function context(request: FastifyRequest): ExecutionContext {
  return {
    switchToHttp: () => ({ getRequest: () => request }),
  } as unknown as ExecutionContext;
}

async function readRequestBody(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf-8');
}

describe('MF-AUTH-N1A gateway authorization policy HTTP integration', () => {
  const previousAuthRequired = process.env.QLH_AUTH_REQUIRED;
  let server: Server;
  let controlUrl: string;
  let received: { method?: string; url?: string; authorization?: string; body?: string };

  beforeAll(async () => {
    server = createServer(async (request, response) => {
      received = {
        method: request.method,
        url: request.url,
        authorization: request.headers.authorization,
        body: await readRequestBody(request),
      };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ user: { user_id: 'member-1', role: 'member' } }));
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    controlUrl = `http://127.0.0.1:${port}`;
  });

  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    if (previousAuthRequired === undefined) delete process.env.QLH_AUTH_REQUIRED;
    else process.env.QLH_AUTH_REQUIRED = previousAuthRequired;
  });

  it('validates a member session through the real ControlClient HTTP request', async () => {
    process.env.QLH_AUTH_REQUIRED = '1';
    received = {};
    const request = {
      method: 'POST',
      url: '/api/chat',
      headers: { authorization: 'Bearer integration-token' },
    } as FastifyRequest & { qlhAuthSession?: Record<string, unknown> };
    const guard = new AuthPolicyGuard(new ControlClient(controlUrl, 1000));

    await expect(guard.canActivate(context(request))).resolves.toBe(true);
    expect(received).toEqual({
      method: 'GET',
      url: '/auth/session',
      authorization: 'Bearer integration-token',
      body: '',
    });
    expect(request.qlhAuthSession).toEqual({
      user: { user_id: 'member-1', role: 'member' },
    });
  });

  it('rejects a member session from the AND-CTRL-05 management surface', async () => {
    process.env.QLH_AUTH_REQUIRED = '1';
    const guard = new AuthPolicyGuard(new ControlClient(controlUrl, 1000));
    for (const url of ['/api/auth/manage/summary', '/api/auth/manage/audit']) {
      const request = {
        method: 'GET',
        url,
        headers: { authorization: 'Bearer integration-token' },
      } as FastifyRequest & { qlhAuthSession?: Record<string, unknown> };
      await expect(guard.canActivate(context(request)))
        .rejects.toBeInstanceOf(ForbiddenException);
    }
    const revokeRequest = {
      method: 'POST',
      url: '/api/auth/tailscale/bindings/b-1/revoke',
      headers: { authorization: 'Bearer integration-token' },
    } as FastifyRequest & { qlhAuthSession?: Record<string, unknown> };
    await expect(guard.canActivate(context(revokeRequest)))
      .rejects.toBeInstanceOf(ForbiddenException);
  });
});
