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
import type { FastifyReply, FastifyRequest } from 'fastify';
import fastifyStatic from '@fastify/static';
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
import { resolveFrontendDist, StaticService } from './modules/static/static.service';
import { StatusController } from './modules/status/status.controller';

@Controller()
export class CatchAllController {
  // find-my-way 匹配优先级：静态路由 > 参数路由 > 通配符，
  // 因此已注册的 /api/health 等真实路由不会被本控制器抢匹配。
  // 注意：global prefix 'api' 使本控制器只覆盖 /api/*；根路径与 /assets/*
  // 等由 createApp 中的 setNotFoundHandler（静态托管）处理。
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
  providers: [SchedulerClient, InferenceClient, LegacyControlClient, StaticService],
})
export class AppModule {}

export { resolveFrontendDist } from './modules/static/static.service';

export async function createApp(): Promise<NestFastifyApplication> {
  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule,
    new FastifyAdapter(),
  );
  // @fastify/static：仅提供 reply.sendFile 能力（serve:false 不注册通配路由），
  // 路由由 setNotFoundHandler 控制（见下）。
  await app.register(fastifyStatic, {
    root: resolveFrontendDist(),
    wildcard: false,
    serve: false,
  });
  // TUI 客户端硬编码 base_url + "/api" + path（tui_admin.py:218），前缀不可配置。
  app.setGlobalPrefix('api');
  app.useGlobalInterceptors(new RequestIdInterceptor());
  app.useGlobalFilters(new JsonDetailFilter());

  // 阶段 2.3 静态托管：根路径 /、/assets/*、前端路由 SPA 回退。
  // 注册 Fastify 原生通配路由 /*（早于 Nest 路由注册）：find-my-way 按
  // 具体度匹配（精确 > 参数 > 通配），/api/health 等 Nest 路由优先；
  // /api/* 由 CatchAllController 覆盖保持 JSON 404；本路由处理其余全部
  // 方法/路径——GET 先尝试静态文件（SPA 回退），其余返回 JSON 404。
  const fastify = app.getHttpAdapter().getInstance();
  const staticService = app.get(StaticService);
  fastify.all('/*', async (req: FastifyRequest, reply: FastifyReply) => {
    if (req.method === 'GET' && (await staticService.tryServe(req, reply))) {
      return;
    }
    staticService.sendNotFound(req, reply);
  });
  return app;
}
