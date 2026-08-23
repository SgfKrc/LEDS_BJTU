import { ExecutionContext, ForbiddenException, UnauthorizedException } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import {
  accessLevelFor,
  authPolicyEnabled,
  AuthPolicyGuard,
} from '../src/modules/auth/auth-policy.guard';

function context(method: string, url: string, authorization?: string): ExecutionContext {
  const request = {
    method,
    url,
    headers: authorization ? { authorization } : {},
  } as FastifyRequest;
  return {
    switchToHttp: () => ({ getRequest: () => request }),
  } as unknown as ExecutionContext;
}

describe('MF-AUTH-N1A gateway authorization policy', () => {
  const previousAuthRequired = process.env.QLH_AUTH_REQUIRED;
  const request = jest.fn();
  const guard = new AuthPolicyGuard({ request } as any);

  beforeEach(() => {
    process.env.QLH_AUTH_REQUIRED = '1';
    request.mockReset();
  });

  afterAll(() => {
    if (previousAuthRequired === undefined) delete process.env.QLH_AUTH_REQUIRED;
    else process.env.QLH_AUTH_REQUIRED = previousAuthRequired;
  });

  it('keeps login and node trust routes outside local-user sessions', () => {
    expect(accessLevelFor('POST', '/api/auth/login')).toBe('public');
    expect(accessLevelFor('POST', '/api/cluster/nodes/register')).toBe('machine');
    expect(accessLevelFor('GET', '/api/models/files/model.gguf')).toBe('machine');
  });

  it('classifies member work and manager mutations separately', () => {
    expect(accessLevelFor('POST', '/api/chat')).toBe('authenticated');
    expect(accessLevelFor('POST', '/api/diffusion/generate')).toBe('authenticated');
    expect(accessLevelFor('GET', '/api/auth/tailscale/local-status')).toBe('authenticated');
    expect(accessLevelFor('GET', '/api/users')).toBe('manager');
    expect(accessLevelFor('POST', '/api/models/load')).toBe('manager');
    expect(accessLevelFor('POST', '/api/models/local-assets/qwen3-4b/preflight')).toBe('authenticated');
    expect(accessLevelFor('PATCH', '/api/cluster/settings')).toBe('manager');
  });

  it('classifies AND-CTRL-05 management surface as manager-only', () => {
    // 管理摘要/审计/确认签发只对 owner/admin 开放
    expect(accessLevelFor('GET', '/api/auth/manage/summary')).toBe('manager');
    expect(accessLevelFor('GET', '/api/auth/manage/audit')).toBe('manager');
    expect(accessLevelFor('POST', '/api/auth/manage/confirm')).toBe('manager');
    // 成员自操作的 Tailnet prepare/confirm 仍为 authenticated；跨用户 revoke 提级 manager
    expect(accessLevelFor('POST', '/api/auth/tailscale/bindings')).toBe('authenticated');
    expect(accessLevelFor('POST', '/api/auth/tailscale/bindings/b-1/confirm')).toBe('authenticated');
    expect(accessLevelFor('POST', '/api/auth/tailscale/bindings/b-1/revoke')).toBe('manager');
    expect(accessLevelFor('DELETE', '/api/auth/users/u-1')).toBe('manager');
  });

  it('supports explicit rollout switches while defaulting off only in tests', () => {
    expect(authPolicyEnabled({ NODE_ENV: 'production' })).toBe(true);
    expect(authPolicyEnabled({ NODE_ENV: 'test' })).toBe(false);
    expect(authPolicyEnabled({ NODE_ENV: 'test', QLH_AUTH_REQUIRED: 'on' })).toBe(true);
    expect(authPolicyEnabled({ NODE_ENV: 'production', QLH_AUTH_REQUIRED: 'off' })).toBe(false);
  });

  it('rejects protected routes without a Bearer token', async () => {
    await expect(guard.canActivate(context('GET', '/api/status')))
      .rejects.toBeInstanceOf(UnauthorizedException);
    expect(request).not.toHaveBeenCalled();
  });

  it('allows member work but rejects manager routes for members', async () => {
    request.mockResolvedValue({ user: { user_id: 'member-1', role: 'member' } });
    await expect(guard.canActivate(context('POST', '/api/chat', 'Bearer member-token')))
      .resolves.toBe(true);
    await expect(guard.canActivate(context('GET', '/api/users', 'Bearer member-token')))
      .rejects.toBeInstanceOf(ForbiddenException);
  });

  it('allows owner and admin sessions through manager routes', async () => {
    request.mockResolvedValueOnce({ user: { user_id: 'owner-1', role: 'owner' } });
    await expect(guard.canActivate(context('GET', '/api/users', 'Bearer owner-token')))
      .resolves.toBe(true);
    request.mockResolvedValueOnce({ user: { user_id: 'admin-1', role: 'admin' } });
    await expect(guard.canActivate(context('POST', '/api/models/load', 'Bearer admin-token')))
      .resolves.toBe(true);
    expect(request).toHaveBeenCalledWith(
      'GET', '/auth/session', undefined, expect.objectContaining({ authorization: expect.any(String) }),
    );
  });
});
