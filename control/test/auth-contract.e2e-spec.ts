import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app';
import { CredentialProtector, ModelCredentialStore } from '../src/data/model-credential-store';
import { SqliteStore } from '../src/data/sqlite-store';
import { totp } from '../src/data/auth-service';

class TestProtector implements CredentialProtector {
  readonly name = 'test-protector';

  async protect(secret: string): Promise<string> {
    return Buffer.from(secret, 'utf8').toString('base64');
  }

  async unprotect(ciphertext: string): Promise<string> {
    return Buffer.from(ciphertext, 'base64').toString('utf8');
  }
}

describe('MF-AUTH-N1 local Auth App and user management contract', () => {
  let app: NestFastifyApplication;
  let store: SqliteStore;
  let vault: ModelCredentialStore;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-auth-n1-'));
    store = new SqliteStore(path.join(tmpDir, 'control.sqlite3'));
    store.open();
    vault = new ModelCredentialStore({
      rootDir: path.join(tmpDir, 'credentials'),
      protector: new TestProtector(),
    });
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] })
      .overrideProvider(SqliteStore).useValue(store)
      .overrideProvider(ModelCredentialStore).useValue(vault)
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

  async function bootstrapAndLogin(): Promise<{
    user: Record<string, unknown>;
    provisioning: Record<string, string>;
    recoveryCodes: string[];
    token: string;
  }> {
    const bootstrap = await app.inject({
      method: 'POST',
      url: '/auth/bootstrap',
      payload: { username: 'owner', display_name: 'Main owner' },
    });
    expect(bootstrap.statusCode).toBe(201);
    const provisioning = bootstrap.json().provisioning as Record<string, string>;
    expect(provisioning.secret).toMatch(/^[A-Z2-7]{32}$/);
    expect(provisioning.otpauth_uri).toBe(provisioning.qr_payload);
    expect(provisioning.otpauth_uri).toContain('otpauth://totp/');
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
    const recoveryCodes = verified.json().recovery_codes as string[];
    expect(recoveryCodes).toHaveLength(10);
    const login = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', code },
    });
    expect(login.statusCode).toBe(200);
    return {
      user: verified.json().user,
      provisioning,
      recoveryCodes,
      token: login.json().access_token,
    };
  }

  it('bootstraps owner with string and QR provisioning, then logs in locally', async () => {
    const result = await bootstrapAndLogin();
    const session = await app.inject({
      method: 'GET',
      url: '/auth/session',
      headers: { authorization: `Bearer ${result.token}` },
    });
    expect(session.statusCode).toBe(200);
    expect(session.json().user).toMatchObject({ username: 'owner', role: 'owner', status: 'active' });
    const users = await app.inject({
      method: 'GET',
      url: '/users',
      headers: { authorization: `Bearer ${result.token}` },
    });
    expect(users.statusCode).toBe(200);
    expect(users.json().users[0]).not.toHaveProperty('secret');
    expect(users.json().users[0]).not.toHaveProperty('otpauth_uri');
    const databaseBytes = fs.readFileSync(store.filePath).toString('utf8');
    expect(databaseBytes).not.toContain(result.provisioning.secret);
    expect(databaseBytes).not.toContain(result.provisioning.otpauth_uri);
  });

  it('creates a member, provisions its Auth App separately, and rotates recovery codes', async () => {
    const result = await bootstrapAndLogin();
    const created = await app.inject({
      method: 'POST',
      url: '/users',
      headers: { authorization: `Bearer ${result.token}` },
      payload: { username: 'member-one', display_name: 'Member One', role: 'member' },
    });
    expect(created.statusCode).toBe(201);
    const member = created.json().user as { user_id: string };
    expect(created.json().user).not.toHaveProperty('secret');
    const provisioning = await app.inject({
      method: 'POST',
      url: `/auth/users/${member.user_id}/totp`,
      headers: { authorization: `Bearer ${result.token}` },
    });
    expect(provisioning.statusCode).toBe(201);
    const memberProvisioning = provisioning.json().provisioning as Record<string, string>;
    const memberCode = totp(memberProvisioning.secret, Date.now(), 'SHA1', 6, 30, 0);
    const memberVerified = await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: {
        user_id: member.user_id,
        authenticator_id: memberProvisioning.authenticator_id,
        code: memberCode,
      },
    });
    expect(memberVerified.statusCode).toBe(200);
    const rotated = await app.inject({
      method: 'POST',
      url: '/auth/recovery-codes/rotate',
      headers: { authorization: `Bearer ${result.token}` },
      payload: { code: totp(result.provisioning.secret, Date.now(), 'SHA1', 6, 30, 0) },
    });
    expect(rotated.statusCode).toBe(200);
    expect(rotated.json().recovery_codes).toHaveLength(10);
    expect(rotated.json().recovery_codes).not.toEqual(result.recoveryCodes);
  });

  it('persists login lockout without allowing a recovery-code bypass', async () => {
    const result = await bootstrapAndLogin();
    for (let index = 0; index < 5; index += 1) {
      const failed = await app.inject({
        method: 'POST',
        url: '/auth/login',
        payload: { username: 'owner', code: '000000' },
      });
      expect(failed.statusCode).toBe(401);
    }
    const locked = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', code: '000000' },
    });
    expect(locked.statusCode).toBe(429);

    const recoveryLogin = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', recovery_code: result.recoveryCodes[0] },
    });
    expect(recoveryLogin.statusCode).toBe(429);

    const sessions = store.prepare('SELECT COUNT(*) AS count FROM auth_sessions').get() as { count: number };
    expect(sessions.count).toBe(1);
  });

  it('consumes a recovery code once', async () => {
    const result = await bootstrapAndLogin();
    await app.inject({
      method: 'POST',
      url: '/auth/logout',
      headers: { authorization: `Bearer ${result.token}` },
    });
    const first = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', recovery_code: result.recoveryCodes[0] },
    });
    expect(first.statusCode).toBe(200);
    const second = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'owner', recovery_code: result.recoveryCodes[0] },
    });
    expect(second.statusCode).toBe(401);
  });

  it('revokes local sessions on logout', async () => {
    const result = await bootstrapAndLogin();
    const logout = await app.inject({
      method: 'POST',
      url: '/auth/logout',
      headers: { authorization: `Bearer ${result.token}` },
    });
    expect(logout.statusCode).toBe(200);
    const session = await app.inject({
      method: 'GET',
      url: '/auth/session',
      headers: { authorization: `Bearer ${result.token}` },
    });
    expect(session.statusCode).toBe(401);
    expect(store.prepare(
      "SELECT COUNT(*) AS count FROM auth_audit_events WHERE event_type = 'logout' AND outcome = 'success'",
    ).get()).toEqual({ count: 1 });
  });

  it('revokes a user and its active sessions together', async () => {
    const result = await bootstrapAndLogin();
    const created = await app.inject({
      method: 'POST',
      url: '/users',
      headers: { authorization: `Bearer ${result.token}` },
      payload: { username: 'member-session' },
    });
    const user = created.json().user as { user_id: string; aggregate_version: number };
    const provisioning = await app.inject({
      method: 'POST',
      url: `/auth/users/${user.user_id}/totp`,
      headers: { authorization: `Bearer ${result.token}` },
    });
    const memberProvisioning = provisioning.json().provisioning as Record<string, string>;
    const code = totp(memberProvisioning.secret, Date.now(), 'SHA1', 6, 30, 0);
    await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: {
        user_id: user.user_id,
        authenticator_id: memberProvisioning.authenticator_id,
        code,
      },
    });
    const memberLogin = await app.inject({
      method: 'POST',
      url: '/auth/login',
      payload: { username: 'member-session', code },
    });
    expect(memberLogin.statusCode).toBe(200);
    const revoked = await app.inject({
      method: 'DELETE',
      url: `/users/${user.user_id}`,
      headers: { authorization: `Bearer ${result.token}` },
      payload: { expected_version: user.aggregate_version },
    });
    expect(revoked.statusCode).toBe(200);
    const session = await app.inject({
      method: 'GET',
      url: '/auth/session',
      headers: { authorization: `Bearer ${memberLogin.json().access_token}` },
    });
    expect(session.statusCode).toBe(401);
  });
});
