/**
 * 管理操作授权矩阵与一次性确认令牌（AND-CTRL-05 前置契约：② 授权矩阵收口、④ 二次确认）。
 *
 * - MANAGEMENT_POLICY：操作类别 → 最小角色 / 是否需要二次确认 / 是否入审计。
 * - roleAllows()：owner/admin/member 有序判定（admin 可管非 owner，owner 全权；
 *   service 层 requireManager/requireBindingAccess 细粒度仍作为第二道防线保留）。
 * - ConfirmTokenStore：一次性、短 TTL、绑定 action+target+角色 的确认令牌。
 *   review_admin 审批域在 control 仍为"降级未迁移"（无会话绑定），本矩阵返回
 *   audited=false / confirm 不签发，Android 侧只能只读——不与后端假装闭环。
 */

export const MANAGER_ROLES = ['owner', 'admin'] as const;
export type ManagerRole = (typeof MANAGER_ROLES)[number];
export type LocalUserRole = ManagerRole | 'member';

export type ManagementAction = 'user_manage' | 'tailnet_bind' | 'review_admin';

export interface ManagementPolicyRule {
  /** 执行该操作所需的最低角色（owner > admin > member）。 */
  minRole: LocalUserRole;
  /** 写操作是否需要一次性确认令牌（X-QLH-Confirm-Token）。 */
  confirmRequired: boolean;
  /** 写操作是否强制落审计（actor 可绑定时才为 true）。 */
  audited: boolean;
  /** 展示用描述。 */
  description: string;
}

export const MANAGEMENT_POLICY: Record<ManagementAction, ManagementPolicyRule> = {
  user_manage: {
    minRole: 'admin',
    confirmRequired: true,
    audited: true,
    description: '成员创建/编辑/撤销、角色授予（仅 owner 可授 admin/owner，service 层强制）',
  },
  tailnet_bind: {
    minRole: 'admin',
    confirmRequired: true,
    audited: true,
    description: 'Tailnet 绑定撤销由管理员执行；成员自绑定的 prepare/confirm 不属本矩阵',
  },
  review_admin: {
    minRole: 'admin',
    confirmRequired: false,
    audited: false,
    description: '入群审批（control review 域 master/角色判定未迁移，仅只读，不签发确认令牌）',
  },
};

const ROLE_ORDER: Record<LocalUserRole, number> = { owner: 3, admin: 2, member: 1 };

export function roleAllows(action: ManagementAction, role: LocalUserRole | string | null | undefined): boolean {
  const rule = MANAGEMENT_POLICY[action];
  if (!rule) return false;
  if (!ROLE_ORDER[role as LocalUserRole]) return false;
  return ROLE_ORDER[role as LocalUserRole] >= ROLE_ORDER[rule.minRole];
}

export interface ConfirmTokenRecord {
  token: string;
  action: ManagementAction;
  targetId: string;
  role: string;
  actorUserId: string;
  expiresAtMs: number;
}

export interface ConfirmTokenStoreOptions {
  /** 令牌有效期毫秒（默认 120s）。 */
  ttlMs?: number;
  /** 当前时间戳（测试注入）。 */
  nowMs?: () => number;
}

/**
 * 内存一次性确认令牌存储。每次 issue 生成 32 字节随机令牌；
 * consume 校验「存在 + 未过期 + action/target/角色匹配」后单次消费删除。
 */
export class ConfirmTokenStore {
  private readonly store = new Map<string, ConfirmTokenRecord>();
  private readonly ttlMs: number;
  private readonly nowMs: () => number;

  constructor(options: ConfirmTokenStoreOptions = {}) {
    this.ttlMs = options.ttlMs ?? 120_000;
    this.nowMs = options.nowMs ?? (() => Date.now());
  }

  issue(action: ManagementAction, targetId: string, role: string, actorUserId: string): ConfirmTokenRecord {
    const token = randomToken();
    const record: ConfirmTokenRecord = {
      token,
      action,
      targetId,
      role,
      actorUserId,
      expiresAtMs: this.nowMs() + this.ttlMs,
    };
    this.store.set(token, record);
    return record;
  }

  /**
   * 消费确认令牌。返回 null 表示拒绝（不存在/过期/不匹配/已用），
   * 消费成功后记录会被移除（一次性）。
   */
  consume(action: ManagementAction, targetId: string, role: string, actorUserId: string, token: string): ConfirmTokenRecord | null {
    const raw = String(token || '').trim();
    if (!raw) return null;
    const record = this.store.get(raw);
    if (!record) return null;
    this.store.delete(raw); // 无论校验是否通过都一次性消费，防重放
    if (record.action !== action) return null;
    if (record.targetId !== targetId) return null;
    if (record.role !== role) return null;
    if (record.actorUserId !== actorUserId) return null;
    if (this.nowMs() > record.expiresAtMs) return null;
    return record;
  }
}

function randomToken(): string {
  // crypto.randomUUID 可用则用 32+ 位随机串；否则 Math.random 兜底（仅测试路径）。
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { randomBytes } = require('crypto') as typeof import('crypto');
    return randomBytes(24).toString('hex');
  } catch {
    return Array.from({ length: 48 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
  }
}

/** 进程内共享的确认令牌单例（controller 与 service 共用）。 */
export const CONTROL_CONFIRM_TOKENS = new ConfirmTokenStore();