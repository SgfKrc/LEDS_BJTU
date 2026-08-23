import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { ControlClient } from '../../clients/control.client';

export type AuthAccessLevel = 'public' | 'machine' | 'authenticated' | 'manager';

const PUBLIC_ROUTES = new Set([
  'GET /api/health',
  'GET /api/auth/capability',
  'POST /api/auth/bootstrap',
  'POST /api/auth/totp/verify',
  'POST /api/auth/login',
]);

// These endpoints have their own node or tailnet trust contract. A local user
// session must not replace that machine-to-machine identity.
const MACHINE_ROUTES = new Set([
  'POST /api/cluster/nodes/register',
  'POST /api/cluster/android/register',
  'POST /api/cluster/android/heartbeat',
  'GET /api/cluster/master-health',
  'GET /api/cluster/discover',
  'GET /api/models/downloadable',
]);

const MANAGER_PREFIXES = [
  '/api/users',
  '/api/logs',
  '/api/db/health',
  '/api/models/artifacts',
  '/api/models/imports',
  '/api/models/runtime-checks',
  '/api/models/pull',
  '/api/models/network',
  '/api/models/sources',
  '/api/models/resolve',
  '/api/models/credentials',
  '/api/models/licenses',
  '/api/models/registry',
  // AND-CTRL-05 前置契约：管理摘要/审计/确认签发只对 owner/admin 开放。
  '/api/auth/manage',
];

const MEMBER_MUTATION_PREFIXES = [
  '/api/chat',
  '/api/sessions',
  '/api/conversations',
  '/api/workflows',
  '/api/diffusion/generate',
  '/api/diffusion/edit',
  '/api/diffusion/blobs',
  '/api/diffusion/jobs',
  '/api/experimental/speculative',
  '/api/user/settings',
  '/api/auth',
];

function isReadOnlyLocalAssetPreflight(method: string, path: string): boolean {
  return method === 'POST'
    && /^\/api\/models\/local-assets\/[^/]+\/preflight$/.test(path);
}

function normalizedPath(url: string): string {
  const path = String(url || '').split('?', 1)[0] || '/';
  return path.length > 1 ? path.replace(/\/+$/, '') : path;
}

function startsWithRoute(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`);
}

export function authPolicyEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const explicit = String(env.QLH_AUTH_REQUIRED ?? '').trim().toLowerCase();
  if (['0', 'false', 'off'].includes(explicit)) return false;
  if (['1', 'true', 'on'].includes(explicit)) return true;
  return env.NODE_ENV !== 'test';
}

export function accessLevelFor(method: string, url: string): AuthAccessLevel {
  const verb = String(method || 'GET').toUpperCase();
  const path = normalizedPath(url);
  const route = `${verb} ${path}`;
  if (PUBLIC_ROUTES.has(route)) return 'public';
  if (MACHINE_ROUTES.has(route) || startsWithRoute(path, '/api/models/files')) return 'machine';
  if (MANAGER_PREFIXES.some((prefix) => startsWithRoute(path, prefix))) return 'manager';
  if (isReadOnlyLocalAssetPreflight(verb, path)) return 'authenticated';
  if (startsWithRoute(path, '/api/auth/users')) return 'manager';
  // Tailnet 绑定撤销是管理写操作（AND-CTRL-05 前置③）：prepare/confirm 仍是
  // 成员自操作（authenticated），仅 revoke 提级为 manager。
  if (/^\/api\/auth\/tailscale\/bindings\/[^/]+\/revoke$/.test(path)) return 'manager';
  if (verb !== 'GET' && (
    startsWithRoute(path, '/api/cluster')
    || startsWithRoute(path, '/api/device')
    || startsWithRoute(path, '/api/settings')
  )) return 'manager';
  if (verb !== 'GET'
      && (startsWithRoute(path, '/api/models') || startsWithRoute(path, '/api/diffusion'))
      && !MEMBER_MUTATION_PREFIXES.some((prefix) => startsWithRoute(path, prefix))) {
    return 'manager';
  }
  return 'authenticated';
}

function authorizationHeader(req: FastifyRequest): string {
  const raw = req.headers.authorization;
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (!value || !/^Bearer\s+\S+$/i.test(value.trim())) {
    throw new UnauthorizedException('需要本地主节点登录');
  }
  return value.trim();
}

@Injectable()
export class AuthPolicyGuard implements CanActivate {
  constructor(private readonly control: ControlClient) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    if (!authPolicyEnabled()) return true;
    const req = context.switchToHttp().getRequest<FastifyRequest & {
      qlhAuthSession?: Record<string, unknown>;
    }>();
    const access = accessLevelFor(req.method, req.url);
    if (access === 'public' || access === 'machine') return true;

    const authorization = authorizationHeader(req);
    const session = await this.control.request(
      'GET',
      '/auth/session',
      undefined,
      { authorization },
    ) as { user?: { role?: string } };
    const role = session?.user?.role;
    if (!['owner', 'admin', 'member'].includes(String(role))) {
      throw new UnauthorizedException('本地主节点会话无效');
    }
    if (access === 'manager' && role !== 'owner' && role !== 'admin') {
      throw new ForbiddenException('需要 owner 或 admin 权限');
    }
    req.qlhAuthSession = session as Record<string, unknown>;
    return true;
  }
}
