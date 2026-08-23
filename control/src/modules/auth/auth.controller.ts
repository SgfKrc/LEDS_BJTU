import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Patch,
  Post,
  Query,
  Req,
} from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import {
  AuthService,
  AuthServiceError,
  AuthenticatedSession,
} from '../../data/auth-service';
import { LocalUserRole } from '../../data/auth-asset-repository';
import { TailscaleLocalStatusService } from '../../data/tailscale-local-status';
import {
  CONTROL_CONFIRM_TOKENS,
  MANAGEMENT_POLICY,
  ManagementAction,
  roleAllows,
} from './management-policy';

interface BootstrapRequest {
  username?: string;
  display_name?: string;
}

interface TotpVerifyRequest {
  user_id?: string;
  authenticator_id?: string;
  code?: string;
}

interface LoginRequest {
  username?: string;
  code?: string;
  recovery_code?: string;
}

interface UserRequest {
  username?: string;
  display_name?: string;
  role?: LocalUserRole;
  expected_version?: number;
  status?: 'active' | 'suspended' | 'revoked';
}

interface RecoveryRotateRequest {
  code?: string;
}

interface TailscalePrepareRequest {
  user_id?: string;
  authorization_method?: 'tailscale_cli' | 'local_status' | 'oauth_app';
  credential_ref?: string | null;
}

interface TailscaleConfirmRequest {
  tailnet_id?: string;
  tailscale_user_id?: string;
  node_id?: string | null;
}

function bearerToken(req: FastifyRequest): string | null {
  const raw = req.headers.authorization;
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (!value) return null;
  const match = /^Bearer\s+(.+)$/i.exec(value.trim());
  return match ? match[1].trim() : null;
}

// 管理操作二次确认令牌：与 controller 同生命周期（进程内单例）。
// 真实部署中由主节点进程持有；control 契约面与 Android 契约测试共用同一语义。
const CONFIRM_TOKENS = CONTROL_CONFIRM_TOKENS;

function confirmTokenHeader(req: FastifyRequest): string | null {
  const raw = req.headers['x-qlh-confirm-token'];
  const value = Array.isArray(raw) ? raw[0] : raw;
  return value ? String(value).trim() : null;
}

/** 二次确认：危险管理写操作必须先 POST /auth/manage/confirm 换取一次性令牌。 */
function requireConfirm(session: AuthenticatedSession, action: ManagementAction, targetId: string, req: FastifyRequest): void {
  const token = confirmTokenHeader(req);
  if (!token) {
    throw new AuthServiceError(403, `需要二次确认：请先 POST /auth/manage/confirm 获取确认令牌（action=${action}）`);
  }
  const consumed = CONFIRM_TOKENS.consume(action, targetId, session.user.role, session.user.user_id, token);
  if (!consumed) {
    throw new AuthServiceError(403, '确认令牌无效/过期或与目标不匹配，请重新发起确认');
  }
}

@Controller('auth')
export class AuthController {
  constructor(
    private readonly auth: AuthService,
    private readonly tailscaleStatus: TailscaleLocalStatusService,
  ) {}

  @Post('bootstrap')
  @HttpCode(201)
  async bootstrap(@Body() body: BootstrapRequest): Promise<Record<string, unknown>> {
    return this.run(async () => {
      if (!body?.username) throw new AuthServiceError(422, 'username 必填');
      const provisioning = await this.auth.bootstrapOwner({
        username: body.username,
        display_name: body.display_name,
      });
      return { status: 'pending', provisioning };
    });
  }

  @Post('totp/verify')
  @HttpCode(200)
  async verifyTotp(@Body() body: TotpVerifyRequest): Promise<Record<string, unknown>> {
    return this.run(async () => {
      if (!body?.user_id || !body.authenticator_id || !body.code) {
        throw new AuthServiceError(422, 'user_id、authenticator_id 和 code 必填');
      }
      const result = await this.auth.verifyProvisioning({
        user_id: body.user_id,
        authenticator_id: body.authenticator_id,
        code: body.code,
      });
      return { status: 'active', user: result.user, recovery_codes: result.recovery_codes };
    });
  }

  @Post('login')
  @HttpCode(200)
  async login(@Body() body: LoginRequest): Promise<Record<string, unknown>> {
    return this.run(async () => {
      if (!body?.username || (!body.code && !body.recovery_code)) {
        throw new AuthServiceError(422, 'username 以及 code 或 recovery_code 必填');
      }
      return { ...await this.auth.login({
        username: body.username,
        code: body.code,
        recovery_code: body.recovery_code,
      }) };
    });
  }

  @Get('session')
  session(@Req() req: FastifyRequest): Record<string, unknown> {
    const current = this.currentSession(req);
    return { session_id: current.session_id, expires_at: current.expires_at, user: current.user };
  }

  @Get('sessions')
  listSessions(
    @Req() req: FastifyRequest,
    @Query('user_id') userId?: string,
  ): Record<string, unknown> {
    return this.runSync(() => {
      const current = this.currentSession(req);
      return { sessions: this.auth.listAuthSessions(current, userId || current.user.user_id) };
    });
  }

  @Delete('sessions/:sessionId')
  @HttpCode(200)
  revokeSession(
    @Req() req: FastifyRequest,
    @Param('sessionId') sessionId: string,
  ): Record<string, unknown> {
    return this.runSync(() => ({
      status: 'revoked',
      session: this.auth.revokeAuthSession(this.currentSession(req), sessionId),
    }));
  }

  @Post('logout')
  @HttpCode(200)
  logout(@Req() req: FastifyRequest): Record<string, string> {
    this.auth.logout(bearerToken(req));
    return { status: 'logged_out' };
  }

  @Get('manage/summary')
  manageSummary(@Req() req: FastifyRequest): Record<string, unknown> {
    return this.runSync(() => {
      const session = this.currentSession(req);
      // 管理摘要即管理功能：契约面与 gateway 语义一致，member 直接 403
      //（低权限投影由前端按 role 自理，不向 member 暴露管理入口）。
      if (!roleAllows('user_manage', session.user.role)) {
        throw new AuthServiceError(403, '需要 owner 或 admin 权限');
      }
      const role = session.user.role;
      const actions: Record<string, unknown> = {};
      for (const [action, rule] of Object.entries(MANAGEMENT_POLICY) as Array<[ManagementAction, typeof MANAGEMENT_POLICY[ManagementAction]]>) {
        actions[action] = {
          allowed: roleAllows(action, role),
          confirm_required: rule.confirmRequired,
          audited: rule.audited,
          description: rule.description,
        };
      }
      return {
        role,
        policy_version: 'and-ctrl-05-v1',
        audit_available: true,
        confirm_ttl_seconds: 120,
        actions,
        counts: {
          users: this.auth.listUsers(session).length,
          bindings: this.auth.countTailscaleBindings(session),
        },
        // review 审批域在 control 仍为「master/角色判定降级未迁移」：只读可用，不签发确认。
        review_admin_auth_pending: true,
      };
    });
  }

  @Get('manage/audit')
  manageAudit(
    @Req() req: FastifyRequest,
    @Query('limit') limit?: string,
    @Query('event_type') eventType?: string,
  ): Record<string, unknown> {
    return this.runSync(() => {
      const session = this.currentSession(req);
      // 审计查询仅对 manager 开放（owner/admin）。
      if (!roleAllows('user_manage', session.user.role)) {
        throw new AuthServiceError(403, '需要 owner 或 admin 权限');
      }
      return { events: this.auth.listAuditEvents(Number(limit) || 50, eventType) };
    });
  }

  @Post('manage/confirm')
  @HttpCode(200)
  manageConfirm(@Req() req: FastifyRequest, @Body() body: Record<string, unknown>): Record<string, unknown> {
    return this.runSync(() => {
      const session = this.currentSession(req);
      const action = String(body?.action || '');
      if (!Object.prototype.hasOwnProperty.call(MANAGEMENT_POLICY, action)) {
        throw new AuthServiceError(422, `未知管理操作: ${action}`);
      }
      const typed = action as ManagementAction;
      const rule = MANAGEMENT_POLICY[typed];
      if (!roleAllows(typed, session.user.role)) {
        throw new AuthServiceError(403, '当前角色不允许该管理操作');
      }
      if (!rule.confirmRequired) {
        throw new AuthServiceError(409, '该操作无需二次确认（或审批域尚未开放确认契约）');
      }
      const targetId = body?.target_id ? String(body.target_id) : '';
      if (!targetId) {
        throw new AuthServiceError(422, '管理操作确认必须携带 target_id');
      }
      const record = CONFIRM_TOKENS.issue(typed, targetId, session.user.role, session.user.user_id);
      return {
        confirm_token: record.token,
        expires_at: new Date(record.expiresAtMs).toISOString(),
        action: record.action,
        target_id: record.targetId,
      };
    });
  }

  @Post('recovery-codes/rotate')
  @HttpCode(200)
  async rotateRecoveryCodes(@Req() req: FastifyRequest, @Body() body: RecoveryRotateRequest): Promise<Record<string, unknown>> {
    const current = this.currentSession(req);
    return this.run(async () => {
      if (!body?.code) throw new AuthServiceError(422, 'code 必填');
      return { status: 'rotated', recovery_codes: await this.auth.rotateRecoveryCodes(current, body.code) };
    });
  }

  @Get('tailscale/bindings')
  listTailscaleBindings(@Req() req: FastifyRequest): Record<string, unknown> {
    return this.runSync(() => ({ bindings: this.auth.listTailscaleBindings(this.currentSession(req)) }));
  }

  @Get('tailscale/local-status')
  async localTailscaleStatus(@Req() req: FastifyRequest): Promise<Record<string, unknown>> {
    this.currentSession(req);
    return { local_status: await this.tailscaleStatus.inspect() };
  }

  @Post('tailscale/bindings')
  @HttpCode(201)
  prepareTailscaleBinding(@Req() req: FastifyRequest, @Body() body: TailscalePrepareRequest): Record<string, unknown> {
    return this.runSync(() => ({
      status: 'pending',
      binding: this.auth.prepareTailscaleBinding(this.currentSession(req), {
        user_id: body?.user_id,
        authorization_method: body?.authorization_method,
        credential_ref: body?.credential_ref,
      }),
    }));
  }

  @Post('tailscale/bindings/:bindingId/confirm')
  @HttpCode(200)
  confirmTailscaleBinding(@Req() req: FastifyRequest, @Param('bindingId') bindingId: string, @Body() body: TailscaleConfirmRequest): Record<string, unknown> {
    return this.runSync(() => {
      if (!body?.tailnet_id || !body.tailscale_user_id) {
        throw new AuthServiceError(422, 'tailnet_id 和 tailscale_user_id 必填');
      }
      return {
        status: 'active',
        binding: this.auth.confirmTailscaleBinding(this.currentSession(req), bindingId, {
          tailnet_id: body.tailnet_id,
          tailscale_user_id: body.tailscale_user_id,
          node_id: body.node_id,
        }),
      };
    });
  }

  @Post('tailscale/bindings/:bindingId/revoke')
  @HttpCode(200)
  revokeTailscaleBinding(@Req() req: FastifyRequest, @Param('bindingId') bindingId: string): Record<string, unknown> {
    return this.runSync(() => {
      const session = this.currentSession(req);
      // 确认令牌由 service 按「自撤销 or 跨用户管理」判定是否必需（成员可自撤销免确认）。
      return {
        status: 'revoked',
        binding: this.auth.revokeTailscaleBinding(session, bindingId, confirmTokenHeader(req)),
      };
    });
  }

  @Get('users/:userId/tailscale')
  listUserTailscaleBindings(@Req() req: FastifyRequest, @Param('userId') userId: string): Record<string, unknown> {
    return this.runSync(() => ({ bindings: this.auth.listTailscaleBindings(this.currentSession(req), userId) }));
  }

  @Post('users/:userId/tailscale')
  @HttpCode(201)
  prepareUserTailscaleBinding(@Req() req: FastifyRequest, @Param('userId') userId: string, @Body() body: TailscalePrepareRequest): Record<string, unknown> {
    return this.runSync(() => ({
      status: 'pending',
      binding: this.auth.prepareTailscaleBinding(this.currentSession(req), {
        user_id: userId,
        authorization_method: body?.authorization_method,
        credential_ref: body?.credential_ref,
      }),
    }));
  }

  @Post('users/:userId/totp')
  @HttpCode(201)
  async createUserTotp(@Req() req: FastifyRequest, @Param('userId') userId: string): Promise<Record<string, unknown>> {
    const current = this.currentSession(req);
    return this.run(async () => {
      if (current.user.role !== 'owner' && current.user.role !== 'admin') {
        throw new AuthServiceError(403, '需要 owner 或 admin 权限');
      }
      return {
        status: 'pending',
        provisioning: await this.auth.createProvisioningForUser(current, userId),
      };
    });
  }

  private currentSession(req: FastifyRequest): AuthenticatedSession {
    try {
      return this.auth.authenticateToken(bearerToken(req));
    } catch (error) {
      throw this.httpError(error);
    }
  }

  private async run<T>(fn: () => Promise<T>): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      throw this.httpError(error);
    }
  }

  private runSync<T>(fn: () => T): T {
    try {
      return fn();
    } catch (error) {
      throw this.httpError(error);
    }
  }

  private httpError(error: unknown): HttpException {
    if (error instanceof AuthServiceError) return new HttpException(error.message, error.status);
    const message = error instanceof Error ? error.message : String(error);
    if (/UNIQUE constraint|constraint failed|already exists/i.test(message)) {
      return new HttpException('本地用户已存在或状态冲突', 409);
    }
    if (/credential|OS credential/i.test(message)) {
      return new HttpException('OS credential store 不可用', 503);
    }
    return new HttpException('认证操作失败', 500);
  }
}

@Controller('users')
export class UsersController {
  constructor(private readonly auth: AuthService) {}

  @Get()
  list(@Req() req: FastifyRequest): Record<string, unknown> {
    return this.runSync(() => ({ users: this.auth.listUsers(this.currentSession(req)) }));
  }

  @Post()
  @HttpCode(201)
  create(@Req() req: FastifyRequest, @Body() body: UserRequest): Record<string, unknown> {
    return this.runSync(() => {
      if (!body?.username) throw new AuthServiceError(422, 'username 必填');
      const user = this.auth.createUser(this.currentSession(req), {
        username: body.username,
        display_name: body.display_name,
        role: body.role,
      });
      return { status: 'created', user };
    });
  }

  @Patch(':userId')
  @HttpCode(200)
  update(@Req() req: FastifyRequest, @Param('userId') userId: string, @Body() body: UserRequest): Record<string, unknown> {
    return this.runSync(() => {
      if (!Number.isInteger(body?.expected_version) || Number(body.expected_version) < 1) {
        throw new AuthServiceError(422, 'expected_version 必须为正整数');
      }
      const user = this.auth.updateUser(this.currentSession(req), userId, Number(body.expected_version), {
        display_name: body.display_name,
        role: body.role,
        status: body.status,
      });
      return { status: 'updated', user };
    });
  }

  @Delete(':userId')
  @HttpCode(200)
  revoke(@Req() req: FastifyRequest, @Param('userId') userId: string, @Body() body: UserRequest): Record<string, unknown> {
    return this.runSync(() => {
      if (!Number.isInteger(body?.expected_version) || Number(body.expected_version) < 1) {
        throw new AuthServiceError(422, 'expected_version 必须为正整数');
      }
      const session = this.currentSession(req);
      requireConfirm(session, 'user_manage', userId, req);
      const user = this.auth.updateUser(session, userId, Number(body.expected_version), {
        status: 'revoked',
      });
      return { status: 'revoked', user };
    });
  }

  private currentSession(req: FastifyRequest): AuthenticatedSession {
    try {
      return this.auth.authenticateToken(bearerToken(req));
    } catch (error) {
      throw this.httpError(error);
    }
  }

  private runSync<T>(fn: () => T): T {
    try {
      return fn();
    } catch (error) {
      throw this.httpError(error);
    }
  }

  private httpError(error: unknown): HttpException {
    if (error instanceof AuthServiceError) return new HttpException(error.message, error.status);
    const message = error instanceof Error ? error.message : String(error);
    if (/UNIQUE constraint|constraint failed|already exists/i.test(message)) {
      return new HttpException('本地用户已存在或状态冲突', 409);
    }
    return new HttpException('用户管理操作失败', 500);
  }
}
