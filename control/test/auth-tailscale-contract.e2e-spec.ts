import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app';
import { CredentialProtector, ModelCredentialStore } from '../src/data/model-credential-store';
import { SqliteStore } from '../src/data/sqlite-store';
import { totp } from '../src/data/auth-service';
import { TailscaleLocalStatusService } from '../src/data/tailscale-local-status';

class TestProtector implements CredentialProtector {
  readonly name = 'test-protector';

  async protect(secret: string): Promise<string> {
    return Buffer.from(secret, 'utf8').toString('base64');
  }

  async unprotect(ciphertext: string): Promise<string> {
    return Buffer.from(ciphertext, 'base64').toString('utf8');
  }
}

describe('MF-AUTH-N2 Tailscale binding contract', () => {
  let app: NestFastifyApplication;
  let store: SqliteStore;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-auth-n2-'));
    store = new SqliteStore(path.join(tmpDir, 'control.sqlite3'));
    store.open();
    const vault = new ModelCredentialStore({
      rootDir: path.join(tmpDir, 'credentials'),
      protector: new TestProtector(),
    });
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(store)
      .overrideProvider(ModelCredentialStore).useValue(vault)
      .overrideProvider(TailscaleLocalStatusService).useValue({
        inspect: async () => ({
          available: true,
          state: 'ready',
          reason_code: null,
          source: 'tailscale_status_json',
          observed_at: '2026-08-10T10:00:00.000Z',
          requires_confirmation: true,
          candidate: {
            tailnet_id: 'tailnet-local.ts.net',
            tailnet_id_source: 'magic_dns_suffix',
            tailnet_display_name: 'Local tailnet',
            tailscale_user_id: '12345',
            node_id: 'node-local',
            hostname: 'local-node',
            dns_name: 'local-node.tailnet-local.ts.net.',
            addresses: ['100.64.0.2', 'fd7a:115c:a1e0::2'],
          },
        }),
      })
      .compile();
    app = moduleRef.createNestApplication(
      new (require('@nestjs/platform-fastify').FastifyAdapter)(),
    ) as NestFastifyApplication;
    await app.init();
    await app.getHttpAdapter().getInstance().ready();
  });

  afterEach(async () => {
    await app.close();
    store.close();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  async function bootstrapOwner(): Promise<{
    token: string;
    userId: string;
    secret: string;
  }> {
    const bootstrap = await app.inject({
      method: 'POST',
      url: '/auth/bootstrap',
      payload: { username: 'owner' },
    });
    expect(bootstrap.statusCode).toBe(201);
    const provisioning = bootstrap.json().provisioning as {
      user_id: string;
      authenticator_id: string;
      secret: string;
    };
    const code = totp(provisioning.secret, Date.now(), 'SHA1', 6, 30, 0);
    const verified = await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: {
        user_id: provisioning.user_id,
        authenticator_id: provisioning.authenticator_id,
        code,
      },
    });
    expect(verified.statusCode).toBe(200);
    const login = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', code },
    });
    expect(login.statusCode).toBe(200);
    return {
      token: login.json().access_token as string,
      userId: provisioning.user_id,
      secret: provisioning.secret,
    };
  }

  it('binds, switches tailnets atomically, and exposes no credential reference', async () => {
    const owner = await bootstrapOwner();
    const headers = { authorization: `Bearer ${owner.token}` };
    const first = await app.inject({
      method: 'POST',
      url: '/auth/tailscale/bindings',
      headers,
      payload: { authorization_method: 'local_status' },
    });
    expect(first.statusCode).toBe(201);
    const firstBinding = first.json().binding as { binding_id: string };
    expect(first.json().binding).not.toHaveProperty('credential_ref');

    const confirmed = await app.inject({
      method: 'POST',
      url: `/auth/tailscale/bindings/${firstBinding.binding_id}/confirm`,
      headers,
      payload: {
        tailnet_id: 'tailnet-old',
        tailscale_user_id: 'ts-user-owner',
        node_id: 'node-owner',
      },
    });
    expect(confirmed.statusCode).toBe(200);
    expect(confirmed.json().binding).toMatchObject({ state: 'active', tailnet_id: 'tailnet-old' });

    const pending = await app.inject({
      method: 'POST',
      url: '/auth/tailscale/bindings',
      headers,
      payload: { authorization_method: 'tailscale_cli' },
    });
    expect(pending.statusCode).toBe(201);
    const pendingId = pending.json().binding.binding_id as string;
    const switched = await app.inject({
      method: 'POST',
      url: `/auth/tailscale/bindings/${pendingId}/confirm`,
      headers,
      payload: {
        tailnet_id: 'tailnet-new',
        tailscale_user_id: 'ts-user-owner-new',
        node_id: 'node-owner-new',
      },
    });
    expect(switched.statusCode).toBe(200);
    const bindings = await app.inject({ method: 'GET', url: '/auth/tailscale/bindings', headers });
    expect(bindings.statusCode).toBe(200);
    expect(bindings.json().bindings).toEqual(expect.arrayContaining([
      expect.objectContaining({ binding_id: firstBinding.binding_id, state: 'revoked' }),
      expect.objectContaining({ binding_id: pendingId, state: 'active', tailnet_id: 'tailnet-new' }),
    ]));
    expect(store.prepare(
      "SELECT COUNT(*) AS count FROM auth_audit_events WHERE event_type LIKE 'tailscale_binding_%' AND actor_user_id = ?",
    ).get(owner.userId)).toEqual({ count: 4 });
  });

  it('requires a local session before returning the safe status candidate', async () => {
    const unauthenticated = await app.inject({ method: 'GET', url: '/auth/tailscale/local-status' });
    expect(unauthenticated.statusCode).toBe(401);

    const owner = await bootstrapOwner();
    const response = await app.inject({
      method: 'GET',
      url: '/auth/tailscale/local-status',
      headers: { authorization: `Bearer ${owner.token}` },
    });
    expect(response.statusCode).toBe(200);
    expect(response.json().local_status).toMatchObject({
      available: true,
      requires_confirmation: true,
      candidate: {
        tailnet_id: 'tailnet-local.ts.net',
        tailscale_user_id: '12345',
        node_id: 'node-local',
      },
    });
    expect(response.json().local_status).not.toHaveProperty('Peer');
    expect(response.json().local_status.candidate).not.toHaveProperty('login_name');
    expect(response.json().local_status.candidate).not.toHaveProperty('public_key');
  });

  it('rejects cross-user identity reuse and preserves both states', async () => {
    const owner = await bootstrapOwner();
    const headers = { authorization: `Bearer ${owner.token}` };
    const created = await app.inject({
      method: 'POST',
      url: '/users',
      headers,
      payload: { username: 'member' },
    });
    const memberId = created.json().user.user_id as string;
    const ownerBinding = await app.inject({
      method: 'POST',
      url: '/auth/tailscale/bindings',
      headers,
      payload: {},
    });
    const ownerBindingId = ownerBinding.json().binding.binding_id as string;
    await app.inject({
      method: 'POST',
      url: `/auth/tailscale/bindings/${ownerBindingId}/confirm`,
      headers,
      payload: { tailnet_id: 'tailnet-shared', tailscale_user_id: 'ts-same' },
    });

    const memberBinding = await app.inject({
      method: 'POST',
      url: `/auth/users/${memberId}/tailscale`,
      headers,
      payload: { authorization_method: 'local_status' },
    });
    expect(memberBinding.statusCode).toBe(201);
    const memberBindingId = memberBinding.json().binding.binding_id as string;
    const conflict = await app.inject({
      method: 'POST',
      url: `/auth/tailscale/bindings/${memberBindingId}/confirm`,
      headers,
      payload: { tailnet_id: 'tailnet-shared', tailscale_user_id: 'ts-same' },
    });
    expect(conflict.statusCode).toBe(409);
    expect((await app.inject({ method: 'GET', url: '/auth/tailscale/bindings', headers })).json().bindings)
      .toEqual(expect.arrayContaining([expect.objectContaining({ binding_id: ownerBindingId, state: 'active' })]));
    expect((await app.inject({ method: 'GET', url: `/auth/users/${memberId}/tailscale`, headers })).json().bindings)
      .toEqual(expect.arrayContaining([expect.objectContaining({ binding_id: memberBindingId, state: 'pending' })]));
  });

  it('enforces ownership, rejects OAuth Apps, and makes revoke idempotent', async () => {
    const owner = await bootstrapOwner();
    const ownerHeaders = { authorization: `Bearer ${owner.token}` };
    const created = await app.inject({ method: 'POST', url: '/users', headers: ownerHeaders, payload: { username: 'member' } });
    const memberId = created.json().user.user_id as string;
    const oauth = await app.inject({
      method: 'POST',
      url: '/auth/tailscale/bindings',
      headers: ownerHeaders,
      payload: { authorization_method: 'oauth_app' },
    });
    expect(oauth.statusCode).toBe(501);

    const memberProvisioning = await app.inject({
      method: 'POST',
      url: `/auth/users/${memberId}/totp`,
      headers: ownerHeaders,
    });
    const provision = memberProvisioning.json().provisioning as { user_id: string; authenticator_id: string; secret: string };
    const code = totp(provision.secret, Date.now(), 'SHA1', 6, 30, 0);
    await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: { user_id: provision.user_id, authenticator_id: provision.authenticator_id, code },
    });
    const memberLogin = await app.inject({ method: 'POST', url: '/auth/login', payload: { username: 'member', code } });
    expect(memberLogin.statusCode).toBe(200);
    const memberHeaders = { authorization: `Bearer ${memberLogin.json().access_token as string}` };

    const forbidden = await app.inject({ method: 'GET', url: `/auth/users/${owner.userId}/tailscale`, headers: memberHeaders });
    expect(forbidden.statusCode).toBe(403);
    const binding = await app.inject({ method: 'POST', url: '/auth/tailscale/bindings', headers: memberHeaders, payload: {} });
    expect(binding.statusCode).toBe(201);
    const bindingId = binding.json().binding.binding_id as string;
    const revoke = await app.inject({ method: 'POST', url: `/auth/tailscale/bindings/${bindingId}/revoke`, headers: memberHeaders });
    expect(revoke.statusCode).toBe(200);
    expect(revoke.json().binding.state).toBe('revoked');
    const repeat = await app.inject({ method: 'POST', url: `/auth/tailscale/bindings/${bindingId}/revoke`, headers: memberHeaders });
    expect(repeat.statusCode).toBe(200);
    expect(repeat.json().binding.state).toBe('revoked');
  });
});
