/**
 * cluster 域控制器（TUI 契约用例 4-32、39）
 *
 * 1:1 透传代理：对外 /api/cluster/* → scheduler-svc /cluster/*（去 /api 前缀）。
 * 例外：/api/cluster/review/* 属控制面（审查票状态机，review.py），阶段 2 由
 * legacy-control 承载（对齐主计划 §2.2 "review → 阶段 3 前暂代理 FastAPI 遗留"），
 * 阶段 3 迁 control-svc。
 * 用 @All('cluster/*') 通配覆盖全部子路径与 HTTP 方法：
 *   - 静态路由优先级高于通配符（find-my-way），已注册的具体路由不会被抢
 *   - scheduler-svc 未来新增端点无需改动网关
 * 保留 PUT/DELETE 方法语义（对外契约，见 docs/TUI适配实施计划.md §3.2）。
 */
import { All, Controller, NotFoundException, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { ControlClient } from '../../clients/control.client';
import { LegacyControlClient } from '../../clients/legacy.client';
import { SchedulerClient } from '../../clients/scheduler.client';

const CLUSTER_PREFIX = '/api/cluster';

@Controller()
export class ClusterController {
  constructor(
    private readonly scheduler: SchedulerClient,
    private readonly legacy: LegacyControlClient,
    private readonly control: ControlClient,
  ) {}

  // /api/cluster/review 与 /api/cluster/review/* → control-svc（设置 QLH_CONTROL_URL 时）
  // 否则 legacy-control /cluster/review/*（并行共存基线）
  // （find-my-way 具体度优先于下方 @All('cluster/*') 通配）
  @All('cluster/review')
  reviewRoot(@Req() req: FastifyRequest): Promise<unknown> {
    return this.forwardReview(req);
  }

  @All('cluster/review/*')
  reviewSub(@Req() req: FastifyRequest): Promise<unknown> {
    return this.forwardReview(req);
  }

  private forwardReview(req: FastifyRequest): Promise<unknown> {
    const full = req.url;
    const subPath = full.slice('/api'.length); // /cluster/review/...
    const body = (req as { body?: unknown }).body;
    // 阶段 3.2 review 域已迁 control-svc（路径 /cluster/review/* 一致）
    const target =
      process.env.QLH_CONTROL_URL
        ? this.control
        : this.legacy;
    return target.request(req.method, subPath, body);
  }

  @All('cluster/*')
  async proxy(@Req() req: FastifyRequest): Promise<unknown> {
    const full = req.url; // 含 query，如 /api/cluster/nodes?x=1
    if (!full.startsWith(CLUSTER_PREFIX + '/')) {
      throw new NotFoundException(`Route ${req.method}:${full} not found`);
    }
    // 内部端点路径 = 对外路径去掉 /api 前缀（保留 /cluster 段）
    const subPath = full.slice('/api'.length); // /cluster/nodes?x=1
    const body = (req as { body?: unknown }).body;
    return this.scheduler.request(req.method, subPath, body);
  }
}
