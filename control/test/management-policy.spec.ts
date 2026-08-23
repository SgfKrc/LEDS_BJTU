/**
 * AND-CTRL-05 前置契约单测：管理授权矩阵（②）+ 一次性确认令牌（④）。
 * 矩阵语义：owner>admin>member；user_manage/tailnet_bind 需 admin 且二次确认；
 * review_admin 在 control 为「降级未迁移」，不签发确认 token。
 */

import { ConfirmTokenStore, MANAGEMENT_POLICY, roleAllows } from '../src/modules/auth/management-policy';

describe('management policy matrix (票② 授权矩阵)', () => {
  it('exposes one rule per management action', () => {
    expect(Object.keys(MANAGEMENT_POLICY).sort())
      .toEqual(['review_admin', 'tailnet_bind', 'user_manage']);
  });

  it('requires admin for user_manage and tailnet_bind, and does not confirm review_admin', () => {
    expect(MANAGEMENT_POLICY.user_manage.minRole).toBe('admin');
    expect(MANAGEMENT_POLICY.user_manage.confirmRequired).toBe(true);
    expect(MANAGEMENT_POLICY.user_manage.audited).toBe(true);

    expect(MANAGEMENT_POLICY.tailnet_bind.minRole).toBe('admin');
    expect(MANAGEMENT_POLICY.tailnet_bind.confirmRequired).toBe(true);

    // review 审批域：control 未迁移角色校验时明确不签发确认、不入审计（不假装闭环）
    expect(MANAGEMENT_POLICY.review_admin.minRole).toBe('admin');
    expect(MANAGEMENT_POLICY.review_admin.confirmRequired).toBe(false);
    expect(MANAGEMENT_POLICY.review_admin.audited).toBe(false);
  });

  it('roleAllows: owner/admin allow management actions, member does not', () => {
    expect(roleAllows('user_manage', 'owner')).toBe(true);
    expect(roleAllows('user_manage', 'admin')).toBe(true);
    expect(roleAllows('user_manage', 'member')).toBe(false);
    expect(roleAllows('tailnet_bind', 'admin')).toBe(true);
    expect(roleAllows('review_admin', 'admin')).toBe(true);
    expect(roleAllows('review_admin', 'member')).toBe(false);
  });

  it('roleAllows fails closed for unknown action/role', () => {
    expect(roleAllows('user_manage', 'root')).toBe(false);
    expect(roleAllows('user_manage', null)).toBe(false);
    expect(roleAllows('user_manage', undefined)).toBe(false);
    expect((roleAllows as (action: string, role: string | null | undefined) => boolean)('no_such_action', 'owner')).toBe(false);
  });
});

describe('ConfirmTokenStore (票④ 二次确认令牌)', () => {
  let now = 1_000_000;
  const clock = () => now;
  let store: ConfirmTokenStore;

  beforeEach(() => {
    now = 1_000_000;
    store = new ConfirmTokenStore({ ttlMs: 120_000, nowMs: clock });
  });

  it('issues a single-use token bound to action+target+role+actor', () => {
    const record = store.issue('user_manage', 'user-1', 'admin', 'actor-1');
    expect(record.token.length).toBeGreaterThanOrEqual(32);
    expect(record.expiresAtMs).toBe(now + 120_000);

    // 正确消费一次成功
    const consumed = store.consume('user_manage', 'user-1', 'admin', 'actor-1', record.token);
    expect(consumed).not.toBeNull();
    // 重放被拒（一次性）
    expect(store.consume('user_manage', 'user-1', 'admin', 'actor-1', record.token)).toBeNull();
  });

  it('rejects mismatched action/target/role/actor', () => {
    // 每个不匹配维度独立签发 token：否则前一个断言已消费令牌，
    // 后三个断言会退化为「已消费返回 null」的假绿。
    const mismatches: Array<[string, string, string, string]> = [
      ['tailnet_bind', 'user-1', 'admin', 'actor-1'], // action 不匹配
      ['user_manage', 'user-2', 'admin', 'actor-1'], // target 不匹配
      ['user_manage', 'user-1', 'owner', 'actor-1'], // role 不匹配
      ['user_manage', 'user-1', 'admin', 'actor-2'], // actor 不匹配
    ];
    for (const [action, targetId, role, actorUserId] of mismatches) {
      const record = store.issue('user_manage', 'user-1', 'admin', 'actor-1');
      expect(
        store.consume(action as 'user_manage' | 'tailnet_bind', targetId, role, actorUserId, record.token),
      ).toBeNull();
    }
  });

  it('rejects expired tokens (still single-use: consumed on attempt)', () => {
    const record = store.issue('user_manage', 'user-1', 'admin', 'actor-1');
    now += 120_001;
    expect(store.consume('user_manage', 'user-1', 'admin', 'actor-1', record.token)).toBeNull();
    // 过期尝试也消费了令牌
    now -= 120_000;
    expect(store.consume('user_manage', 'user-1', 'admin', 'actor-1', record.token)).toBeNull();
  });

  it('rejects empty/garbage tokens', () => {
    expect(store.consume('user_manage', 'user-1', 'admin', 'actor-1', '')).toBeNull();
    expect(store.consume('user_manage', 'user-1', 'admin', 'actor-1', 'garbage')).toBeNull();
  });
});