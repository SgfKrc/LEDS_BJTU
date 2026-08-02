/**
 * QLH API 网关 — 应用工厂
 *
 * T2 完成：request-id 拦截器 + JSON detail 异常过滤器 + /api/health。
 * T3 起挂载各域控制器（cluster/queue/layers/device 代理，见
 * docs/TUI适配实施计划.md §5 与 docs/微服务架构改造计划.md §2.2）。
 */
import { All, Controller, Module, NotFoundException, Req } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import {
  FastifyAdapter,
  NestFastifyApplication,
} from '@nestjs/platform-fastify';
import { JsonDetailFilter } from './common/json-detail.filter';
import { RequestIdInterceptor } from './common/request-id';
import { InferenceClient } from './clients/inference.client';
import { LegacyControlClient } from './clients/legacy.client';
import { SchedulerClient } from './clients/scheduler.client';
import { ClusterController } from './modules/cluster/cluster.controller';
import { ChatController } from './modules/chat/chat.controller';
import { ControlController } from './modules/control/control.controller';
import { DeviceController } from './modules/device/device.controller';
import { ExperimentalController } from './modules/experimental/experimental.controller';
import { HealthController } from './modules/health/health.controller';
import { LogsController } from './modules/logs/logs.controller';
import { ModelsController } from './modules/models/models.controller';
import { StatusController } from './modules/status/status.controller';

@Controller()
export class CatchAllController {
  // find-my-way 匹配优先级：静态路由 > 参数路由 > 通配符，
  // 因此已注册的 /api/health 等真实路由不会被本控制器抢匹配。
  @All('*')
  notFound(@Req() req: any): never {
    throw new NotFoundException(`Route ${req.method}:${req.url} not found`);
  }
}

@Module({
  controllers: [
    HealthController,
    ClusterController,
    DeviceController,
    StatusController,
    ModelsController,
    LogsController,
    ChatController,
    ControlController,
    ExperimentalController,
    CatchAllController,
  ],
  providers: [SchedulerClient, InferenceClient, LegacyControlClient],
})
export class AppModule {}

export async function createApp(): Promise<NestFastifyApplication> {
  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule,
    new FastifyAdapter(),
  );
  // TUI 客户端硬编码 base_url + "/api" + path（tui_admin.py:218），前缀不可配置。
  app.setGlobalPrefix('api');
  app.useGlobalInterceptors(new RequestIdInterceptor());
  app.useGlobalFilters(new JsonDetailFilter());
  return app;
}
