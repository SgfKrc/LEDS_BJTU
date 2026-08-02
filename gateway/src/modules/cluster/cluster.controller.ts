/**
 * cluster 域控制器（TUI 契约用例 4-32、39）
 *
 * 1:1 透传代理：对外 /api/cluster/* → scheduler-svc /cluster/*（去 /api 前缀）。
 * 用 @All('cluster/*') 通配覆盖全部子路径与 HTTP 方法：
 *   - 静态路由优先级高于通配符（find-my-way），已注册的具体路由不会被抢
 *   - scheduler-svc 未来新增端点无需改动网关
 * 保留 PUT/DELETE 方法语义（对外契约，见 docs/TUI适配实施计划.md §3.2）。
 */
import { All, Controller, NotFoundException, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { SchedulerClient } from '../../clients/scheduler.client';

const CLUSTER_PREFIX = '/api/cluster';

@Controller()
export class ClusterController {
  constructor(private readonly scheduler: SchedulerClient) {}

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
