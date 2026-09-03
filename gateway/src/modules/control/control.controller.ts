/**
 * 控制面域控制器（阶段 2 过渡：legacy-control 承载；阶段 3 迁 control-svc）
 *
 * 覆盖域：sessions / conversations / settings / review / workflows / bootstrap /
 *         models registry / downloadable / gguf / files / download。
 * 1:1 透传：对外 /api/<域>/* → 内部 /<域>/*（去 /api 前缀）。
 *
 * 渐进切换（并行共存）：默认全部走 legacy-control（QLH_LEGACY_CONTROL_URL，
 * 基线不变）；显式设置 QLH_CONTROL_URL 后，已迁移域（阶段 3.2 完成：
 * sessions/conversations/settings/review/workflows/bootstrap/models
 * registry/gguf/download/presets/db-health）改走 control-svc，未迁移域
 * （models downloadable/files——集群模型分发，tailnet 鉴权，随集群面处理）
 * 仍走 legacy。
 * 注：fastify adapter 下同一方法叠加多个 @All 会覆盖，故每域拆根/子两个方法。
 */
import { All, Controller, Get, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { ControlClient } from '../../clients/control.client';
import { ForwardClient } from '../../clients/forward-client';
import { LegacyControlClient } from '../../clients/legacy.client';

@Controller()
export class ControlController {
  constructor(
    private readonly legacy: LegacyControlClient,
    private readonly control: ControlClient,
  ) {}

  /** migrated=true 的域：设置 QLH_CONTROL_URL 后走 control-svc，否则 legacy */
  private pick(migrated: boolean): ForwardClient {
    return migrated && process.env.QLH_CONTROL_URL
      ? this.control
      : this.legacy;
  }

  private forward(
    req: FastifyRequest,
    migrated: boolean,
    includeAuthorization = false,
    forceControl = false,
  ): Promise<unknown> {
    const full = req.url; // 含 query
    const subPath = full.slice('/api'.length);
    const body = (req as { body?: unknown }).body;
    const rawAuthorization = req.headers.authorization;
    const authorization = Array.isArray(rawAuthorization)
      ? rawAuthorization[0]
      : rawAuthorization;
    const headers: Record<string, string> = {};
    if (includeAuthorization && authorization) headers.authorization = authorization;
    const client = forceControl ? this.control : this.pick(migrated);
    return client.request(req.method, subPath, body, headers);
  }

  // ---- 已迁移域（阶段 3.2 完成） ----

  @All('sessions') sessionsRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('sessions/*') sessionsSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @All('conversations') conversationsRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('conversations/*') conversationsSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @All('settings') settingsRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('settings/*') settingsSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('user/settings') userSettings(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @All('review') reviewRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('review/*') reviewSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @All('workflows') workflowsRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('workflows/*') workflowsSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @All('bootstrap') bootstrapRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('bootstrap/*') bootstrapSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  @Get('auth/capability')
  async authCapability(): Promise<unknown> {
    // Capability must come from the user-owned control service so gateway
    // clients see both bootstrap availability and a real unavailable state.
    return this.control.request('GET', '/auth/capability');
  }

  // Authentication is a control-svc-only domain. It must not silently fall
  // back to the unauthenticated legacy monolith when the migration flag is absent.
  @All('auth') authRoot(@Req() r: FastifyRequest) { return this.forward(r, true, true, true); }
  @All('auth/*') authSub(@Req() r: FastifyRequest) { return this.forward(r, true, true, true); }
  @All('users') usersRoot(@Req() r: FastifyRequest) { return this.forward(r, true, true, true); }
  @All('users/*') usersSub(@Req() r: FastifyRequest) { return this.forward(r, true, true, true); }

  @All('models/registry') registryRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/registry/*') registrySub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/gguf') gguf(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/download/*') downloadSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  // MODEL-FLEET local control plane. These paths are intentionally exact so
  // inference-owned /api/models/current|load|switch routes remain untouched.
  @All('models/artifacts') modelArtifacts(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/imports') modelImports(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/runtime-checks') runtimeChecks(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/runtime-checks/*') runtimeChecksSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/pull') modelPull(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/pull/*') modelPullSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/network') modelNetwork(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/network/*') modelNetworkSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/sources') modelSources(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/sources/*') modelSourcesSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/resolve') modelResolve(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/credentials') modelCredentials(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/credentials/*') modelCredentialsSub(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/licenses') modelLicenses(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('models/licenses/*') modelLicensesSub(@Req() r: FastifyRequest) { return this.forward(r, true); }

  // ---- 已迁移域补（presets / db-health，阶段 3.2 末两域） ----

  @All('presets') presetsRoot(@Req() r: FastifyRequest) { return this.forward(r, true); }
  @All('db/health') dbHealth(@Req() r: FastifyRequest) { return this.forward(r, true); }

  // ---- 未迁移域（仍走 legacy-control；集群模型分发，tailnet 鉴权，随集群面处理） ----

  @All('models/downloadable') downloadable(@Req() r: FastifyRequest) { return this.forward(r, false); }
  @All('models/files/*') filesSub(@Req() r: FastifyRequest) { return this.forward(r, false); }
}
