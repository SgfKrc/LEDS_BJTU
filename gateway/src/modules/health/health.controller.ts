/**
 * 健康检查端点（TUI 契约用例 1/41：GET /api/health → {"status":"ok"}）
 *
 * 语义：网关自身进程存活即可返回 ok，不依赖 scheduler/inference 就绪
 * （网关先行拓扑，见 docs/微服务架构改造计划.md §1.8/§3.1 细节④）。
 * 模型/调度就绪状态由 /api/status 的 model_loaded 等字段承载。
 */
import { Controller, Get } from '@nestjs/common';

@Controller('health')
export class HealthController {
  @Get()
  health(): { status: string } {
    return { status: 'ok' };
  }
}
