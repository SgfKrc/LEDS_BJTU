/**
 * status 域控制器（TUI 契约用例 2、40、42）
 *
 * GET /api/status 聚合（对齐 src/api_server.py:2013-2156 的对外字段语义）：
 *   - scheduler-svc  GET /cluster/status   → run_mode / node_role / node_id / max_nodes
 *   - inference-svc  GET /v1/status        → model_loaded / current_quant / model_name /
 *                                            active_model_id / engine / gpu / kv_cache
 *   - scheduler-svc  GET /device/profile   → device 摘要（tier/tier_label/score/gpus/
 *                                            recommendations/warnings）
 * 单源失败容错：对应字段回落默认值，不整体 500（网关先行语义）。
 */
import { Controller, Get } from '@nestjs/common';
import { InferenceClient } from '../../clients/inference.client';
import { SchedulerClient } from '../../clients/scheduler.client';

interface DeviceProfile {
  tier?: unknown;
  tier_label?: string;
  score_total?: unknown;
  gpus?: unknown[];
  recommendations?: unknown[];
  warnings?: unknown[];
}

@Controller('status')
export class StatusController {
  constructor(
    private readonly scheduler: SchedulerClient,
    private readonly inference: InferenceClient,
  ) {}

  @Get()
  async status(): Promise<Record<string, unknown>> {
    const [sched, inf, dev] = await Promise.allSettled([
      this.scheduler.request('GET', '/cluster/status'),
      this.inference.request('GET', '/v1/status'),
      this.scheduler.request('GET', '/device/profile'),
    ]);

    const schedStatus = sched.status === 'fulfilled' ? (sched.value as Record<string, unknown>) : {};
    const infStatus = inf.status === 'fulfilled' ? (inf.value as Record<string, unknown>) : {};
    const profile = dev.status === 'fulfilled' ? (dev.value as DeviceProfile) : undefined;

    const device = profile
      ? {
          tier: profile.tier ?? null,
          tier_label: profile.tier_label ?? '',
          score: profile.score_total ?? null,
          gpus: profile.gpus ?? [],
          recommendations: profile.recommendations ?? [],
          warnings: profile.warnings ?? [],
        }
      : null;

    return {
      model_loaded: infStatus.model_loaded ?? false,
      current_quant: infStatus.current_quant ?? null,
      model_name: infStatus.model_name ?? '',
      active_model_id: infStatus.active_model_id ?? null,
      engine: infStatus.engine ?? '',
      run_mode: schedStatus.run_mode ?? 'single',
      node_role: schedStatus.node_role ?? 'master',
      node_id: schedStatus.node_id ?? null,
      max_nodes: schedStatus.max_nodes ?? 0,
      gpu: infStatus.gpu ?? {},
      kv_cache: infStatus.kv_cache ?? {},
      device,
    };
  }
}
