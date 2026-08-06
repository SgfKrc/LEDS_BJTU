/**
 * device 域控制器（TUI 契约用例 33-35）
 *
 * 与 cluster 同模式的 1:1 透传代理：对外 /api/device/* → scheduler-svc /device/*。
 * 画像采集库 device_profiler 保留在 Python（从节点为纯 Python 进程，TS 无法代采），
 * scheduler-svc 承载采集并按 docs/TUI适配实施计划.md §2.2 #33-35 字段提供。
 */
import { All, Controller, NotFoundException, Req } from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { SchedulerClient } from '../../clients/scheduler.client';

const DEVICE_PREFIX = '/api/device';

@Controller()
export class DeviceController {
  constructor(private readonly scheduler: SchedulerClient) {}

  @All('device/*')
  async proxy(@Req() req: FastifyRequest): Promise<unknown> {
    const full = req.url; // 含 query
    if (!full.startsWith(DEVICE_PREFIX + '/')) {
      throw new NotFoundException(`Route ${req.method}:${full} not found`);
    }
    const subPath = full.slice('/api'.length); // /device/profile
    const body = (req as { body?: unknown }).body;
    return this.scheduler.request(req.method, subPath, body);
  }
}
