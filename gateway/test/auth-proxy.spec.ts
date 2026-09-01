import type { FastifyRequest } from 'fastify';
import { ControlController } from '../src/modules/control/control.controller';

describe('MF-AUTH-N1 gateway auth proxy', () => {
  const previousControlUrl = process.env.QLH_CONTROL_URL;
  const previousAuthRequired = process.env.QLH_AUTH_REQUIRED;
  const request = jest.fn();
  const control = { request } as any;
  const legacy = { request: jest.fn() } as any;
  const controller = new ControlController(legacy, control);

  beforeEach(() => {
    process.env.QLH_CONTROL_URL = 'http://127.0.0.1:8030';
    process.env.QLH_AUTH_REQUIRED = '1';
    request.mockReset().mockResolvedValue({ status: 'ok' });
    legacy.request.mockReset().mockResolvedValue({ status: 'legacy' });
  });

  afterAll(() => {
    if (previousControlUrl === undefined) delete process.env.QLH_CONTROL_URL;
    else process.env.QLH_CONTROL_URL = previousControlUrl;
    if (previousAuthRequired === undefined) delete process.env.QLH_AUTH_REQUIRED;
    else process.env.QLH_AUTH_REQUIRED = previousAuthRequired;
  });

  it('forwards auth session Bearer only to control-svc', async () => {
    await controller.authSub({
      method: 'GET',
      url: '/api/auth/session',
      headers: { authorization: 'Bearer local-session-token' },
    } as FastifyRequest);
    expect(request).toHaveBeenCalledWith(
      'GET', '/auth/session', undefined,
      { authorization: 'Bearer local-session-token' },
    );
    expect(legacy.request).not.toHaveBeenCalled();
  });

  it('forwards bootstrap body and users routes without inventing credentials', async () => {
    await controller.authSub({
      method: 'POST',
      url: '/api/auth/bootstrap',
      headers: {},
      body: { username: 'owner' },
    } as FastifyRequest);
    await controller.usersRoot({
      method: 'GET',
      url: '/api/users',
      headers: { authorization: 'Bearer manager-token' },
    } as FastifyRequest);
    expect(request).toHaveBeenNthCalledWith(
      1, 'POST', '/auth/bootstrap', { username: 'owner' }, {},
    );
    expect(request).toHaveBeenNthCalledWith(
      2, 'GET', '/users', undefined, { authorization: 'Bearer manager-token' },
    );
  });

  it('advertises auth and keeps auth on control-svc without the migration flag', async () => {
    delete process.env.QLH_CONTROL_URL;
    request.mockResolvedValueOnce({
      required: true,
      enforced: true,
      available: true,
      mode: 'local_totp',
      policy_version: 'n1a-v1',
      service: 'control-svc',
      bootstrap_available: true,
    });
    await expect(controller.authCapability()).resolves.toEqual(expect.objectContaining({
      required: true,
      enforced: true,
      available: true,
      mode: 'local_totp',
      policy_version: 'n1a-v1',
      service: 'control-svc',
      bootstrap_available: true,
    }));
    expect(request).toHaveBeenCalledWith('GET', '/auth/capability');
    await controller.authSub({
      method: 'POST',
      url: '/api/auth/login',
      headers: {},
      body: { username: 'owner', code: '123456' },
    } as FastifyRequest);
    expect(request).toHaveBeenCalledWith(
      'POST', '/auth/login', { username: 'owner', code: '123456' }, {},
    );
    expect(legacy.request).not.toHaveBeenCalled();
  });
});
