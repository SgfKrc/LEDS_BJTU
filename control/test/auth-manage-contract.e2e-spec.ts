/**
 * AND-CTRL-05 前置契约 e2e：管理摘要投影（②）、审计只读（③）、二次确认（④）。
 * 复用 MF-AUTH-N1 e2e 基建（内存 SqliteStore + TestProtector + app.inject）。
 */

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

describe('AND-CTRL-05 management contract (② matrix / ③ audit / ④ confirm)', () => {
  let app: NestFastifyApplication;
  let store: SqliteStore;
  let vault: ModelCredentialStore;
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'control-and-ctrl-05-'));
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

  /** bootstrap owner → login → 返回 Bearer token 与 owner 用户信息。 */
  async function ownerSession(): Promise<{ token: string; userId: string }> {
    const bootstrap = await app.inject({ method: 'POST', url: '/auth/bootstrap', payload: { username: 'owner' } });
    expect(bootstrap.statusCode).toBe(201);
    const provisioning = bootstrap.json().provisioning as Record<string, string>;
    const code = totp(String(provisioning.secret), Date.now(), 'SHA1', 6, 30, 0);
    const verify = await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: { user_id: provisioning.user_id, authenticator_id: provisioning.authenticator_id, code },
    });
    expect(verify.statusCode).toBe(200);
    const login = await app.inject({ method: 'POST', url: '/auth/login', payload: { username: 'owner', code } });
    expect(login.statusCode).toBe(200);
    const body = login.json() as { access_token: string; user: { user_id: string } };
    return { token: body.access_token, userId: body.user.user_id };
  }

  /** 创建 member 并登录，返回其 Bearer。 */
  async function memberSession(ownerAuth: { token: string }): Promise<string> {
    const created = await app.inject({
      method: 'POST',
      url: '/users',
      headers: { authorization: `Bearer ${ownerAuth.token}` },
      payload: { username: 'member' },
    });
    expect(created.statusCode).toBe(201);
    const member = created.json().user as { user_id: string };
    const provisioning = await app.inject({
      method: 'POST',
      url: `/auth/users/${member.user_id}/totp`,
      headers: { authorization: `Bearer ${ownerAuth.token}` },
    });
    const provision = provisioning.json().provisioning as Record<string, string>;
    const code = totp(String(provision.secret), Date.now(), 'SHA1', 6, 30, 0);
    await app.inject({
      method: 'POST',
      url: '/auth/totp/verify',
      payload: { user_id: provision.user_id, authenticator_id: provision.authenticator_id, code },
    });
    const login = await app.inject({ method: 'POST', url: '/auth/login', payload: { username: 'member', code } });
    expect(login.statusCode).toBe(200);
    return (login.json() as { access_token: string }).access_token;
  }

  it('② projects manager matrix summary with full counts', async () => {
    const owner = await ownerSession();
    await memberSession(owner);
    const summary = await app.inject({
      method: 'GET',
      url: '/auth/manage/summary',
      headers: { authorization: `Bearer ${owner.token}` },
    });
    expect(summary.statusCode).toBe(200);
    const body = summary.json() as {
      role: string;
      actions: Record<string, { allowed: boolean; confirm_required: boolean; audited: boolean }>;
      counts: Record<string, number>;
      review_admin_auth_pending: boolean;
    };
    expect(body.role).toBe('owner');
    expect(body.counts.users).toBeGreaterThanOrEqual(2); // owner + member
    expect(Number.isInteger(body.counts.bindings)).toBe(true);
    expect(body.actions.user_manage.allowed).toBe(true);
    expect(body.actions.user_manage.confirm_required).toBe(true);
    expect(body.actions.tailnet_bind.confirm_required).toBe(true);
    expect(body.actions.review_admin.confirm_required).toBe(false);
    expect(body.review_admin_auth_pending).toBe(true);
  });

  it('③ audit endpoint is manager-only and bounded', async () => {
    const owner = await ownerSession();
    const member = await memberSession(owner);

    // member 403
    const forbidden = await app.inject({
      method: 'GET',
      url: '/auth/manage/audit',
      headers: { authorization: `Bearer ${member}` },
    });
    expect(forbidden.statusCode).toBe(403);

    // owner 200（空库也有 bootstrap/login 事件）
    const ok = await app.inject({
      method: 'GET',
      url: '/auth/manage/audit?limit=5',
      headers: { authorization: `Bearer ${owner.token}` },
    });
    expect(ok.statusCode).toBe(200);
    const events = (ok.json() as { events: Array<{ event_type: string; outcome: string }> }).events;
    expect(events.length).toBeGreaterThanOrEqual(1);
    // 脱敏契约：事件不含 details token 字段
    expect(events.every((e) => !Object.keys(e).includes('details'))).toBe(true);
  });

  it('④ confirm issues one-shot token and revoke workflow succeeds', async () => {
    const owner = await ownerSession();
    const created = await app.inject({
      method: 'POST',
      url: '/users',
      headers: { authorization: `Bearer ${owner.token}` },
      payload: { username: 'victim' },
    });
    const victim = created.json().user as { user_id: string; aggregate_version: number };

    // 无 token 直接 revoke → 403
    const noToken = await app.inject({
      method: 'DELETE',
      url: `/users/${victim.user_id}`,
      headers: { authorization: `Bearer ${owner.token}` },
      payload: { expected_version: victim.aggregate_version },
    });
    expect(noToken.statusCode).toBe(403);

    // 错误 token → 403
    const badToken = await app.inject({
      method: 'DELETE',
      url: `/users/${victim.user_id}`,
      headers: { authorization: `Bearer ${owner.token}`, 'x-qlh-confirm-token': 'garbage' },
      payload: { expected_version: victim.aggregate_version },
    });
    expect(badToken.statusCode).toBe(403);

    // 先确认再撤销 → 200
    const confirm = await app.inject({
      method: 'POST',
      url: '/auth/manage/confirm',
      headers: { authorization: `Bearer ${owner.token}` },
      payload: { action: 'user_manage', target_id: victim.user_id },
    });
    expect(confirm.statusCode).toBe(200);
    const confirmToken = (confirm.json() as { confirm_token: string }).confirm_token;
    expect(confirmToken.length).toBeGreaterThanOrEqual(32);

    const revoked = await app.inject({
      method: 'DELETE',
      url: `/users/${victim.user_id}`,
      headers: { authorization: `Bearer ${owner.token}`, 'x-qlh-confirm-token': confirmToken },
      payload: { expected_version: victim.aggregate_version },
    });
    expect(revoked.statusCode).toBe(200);
    expect(revoked.json().user.status).toBe('revoked');

    // 审计事件已落库（user_revoked）
    const audit = await app.inject({
      method: 'GET',
      url: '/auth/manage/audit?event_type=user_revoked',
      headers: { authorization: `Bearer ${owner.token}` },
    });
    expect(audit.statusCode).toBe(200);
    expect(audit.json().events.length).toBeGreaterThanOrEqual(1);
    expect(audit.json().events[0].subject_id).toBe(victim.user_id);
  });

  it('④ confirm validation: unknown action 422, missing target 422, member 403', async () => {
    const owner = await ownerSession();
    const member = await memberSession(owner);

    const unknown = await app.inject({
      method: 'POST',
      url: '/auth/manage/confirm',
      headers: { authorization: `Bearer ${owner.token}` },
      payload: { action: 'no_such_action', target_id: 'u-1' },
    });
    expect(unknown.statusCode).toBe(422);

    const missingTarget = await app.inject({
      method: 'POST',
      url: '/auth/manage/confirm',
      headers: { authorization: `Bearer ${owner.token}` },
      payload: { action: 'user_manage' },
    });
    expect(missingTarget.statusCode).toBe(422);

    const memberTry = await app.inject({
      method: 'POST',
      url: '/auth/manage/confirm',
      headers: { authorization: `Bearer ${member}` },
      payload: { action: 'user_manage', target_id: 'u-1' },
    });
    expect(memberTry.statusCode).toBe(403);
  });
});