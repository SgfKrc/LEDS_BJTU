/**
 * 控制面域控制器（阶段 2 过渡：legacy-control 承载，阶段 3 迁 control-svc）
 *
 * 覆盖域：sessions / conversations / settings / review / workflows / bootstrap /
 *         models registry / downloadable / gguf / files / download。
 * 1:1 透传：对外 /api/<域>/* → legacy-control /<域>/*（去 /api 前缀）。
 * 注：fastify adapter 下同一方法叠加多个 @All 会覆盖，故每域拆根/子两个方法。
 */
import { All, Controller, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { LegacyControlClient } from '../../clients/legacy.client';

@Controller()
export class ControlController {
  constructor(private readonly legacy: LegacyControlClient) {}

  private forward(req: FastifyRequest): Promise<unknown> {
    const full = req.url; // 含 query
    const subPath = full.slice('/api'.length);
    const body = (req as { body?: unknown }).body;
    return this.legacy.request(req.method, subPath, body);
  }

  @All('sessions') sessionsRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('sessions/*') sessionsSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('conversations') conversationsRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('conversations/*') conversationsSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('settings') settingsRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('settings/*') settingsSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('review') reviewRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('review/*') reviewSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('workflows') workflowsRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('workflows/*') workflowsSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('bootstrap') bootstrapRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('bootstrap/*') bootstrapSub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('presets') presetsRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('user/settings') userSettings(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('db/health') dbHealth(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('models/registry') registryRoot(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('models/registry/*') registrySub(@Req() r: FastifyRequest) { return this.forward(r); }

  @All('models/downloadable') downloadable(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('models/gguf') gguf(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('models/files/*') filesSub(@Req() r: FastifyRequest) { return this.forward(r); }
  @All('models/download/*') downloadSub(@Req() r: FastifyRequest) { return this.forward(r); }
}
