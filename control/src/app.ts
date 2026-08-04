/**
 * QLH control-svc — 控制面服务（微服务架构改造计划 阶段 3）
 *
 * 对外契约：与 api_server 控制面端点同路径（网关透传：对外 /api/* 去掉
 * /api 前缀转发至此，对齐 legacy_control 的透传模式）。
 *
 * 3.1 骨架：/health + settings 域（首迁域，cluster_config 表语义对齐 db.py）
 * 3.2 逐域迁移：sessions/conversations/logs/review/models/workflows/bootstrap
 */
import { All, Controller, Module, NotFoundException, Req } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { JsonDetailFilter } from './common/json-detail.filter';
import { RequestIdInterceptor } from './common/request-id';
import { ConfigDao } from './data/config-dao';
import { LogBuffer } from './data/log-buffer';
import { LogFileStore } from './data/log-file-store';
import { ModelRegistryStore } from './data/model-registry-store';
import { ReviewStore } from './data/review-store';
import { SessionStore } from './data/session-store';
import { WorkflowJournalStore } from './data/workflow-journal-store';
import { HealthController } from './modules/health/health.controller';
import { BootstrapController } from './modules/bootstrap/bootstrap.controller';
import { ClientErrorController } from './modules/logs/client-error.controller';
import { LogsController } from './modules/logs/logs.controller';
import { ModelsController } from './modules/models/models.controller';
import { ReviewController } from './modules/review/review.controller';
import { ReviewService } from './modules/review/review.service';
import { SessionsController } from './modules/sessions/sessions.controller';
import { SettingsController } from './modules/settings/settings.controller';
import { WorkflowsController } from './modules/workflows/workflows.controller';

@Controller()
export class CatchAllController {
  @All('*')
  notFound(@Req() req: any): never {
    throw new NotFoundException(`Route ${req.method}:${req.url} not found`);
  }
}

@Module({
  controllers: [
    HealthController,
    SettingsController,
    SessionsController,
    LogsController,
    ClientErrorController,
    ReviewController,
    ModelsController,
    WorkflowsController,
    BootstrapController,
    CatchAllController,
  ],
  providers: [
    ConfigDao,
    SessionStore,
    LogBuffer,
    LogFileStore,
    ReviewStore,
    ReviewService,
    ModelRegistryStore,
    WorkflowJournalStore,
  ],
})
export class AppModule {}

export async function createApp(): Promise<NestFastifyApplication> {
  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule,
    new FastifyAdapter(),
  );
  // 注意：不注册 useBodyParser 自定义 JSON parser（gateway 为对齐 FastAPI
  // 空 body 语义而加；control-svc 端点均带 body，空 body 由 Fastify 默认
  // 400 + JsonDetailFilter 输出 detail 结构，契约一致）。实测 useBodyParser
  // 在本环境使 GET 请求的 preParsing 阶段 TypeError（hooks.js functions
  // undefined，框架内部问题，见 gateway 同代码对比）。
  // 不用 setGlobalPrefix：gateway 用 'api' 前缀；control-svc 直接接收
  // 内部路径（/user/settings 等），空字符串前缀会触发 fastify 路由
  // context 构建异常（kRouteContext.preParsing undefined → preParsing
  // runner TypeError）。
  app.useGlobalInterceptors(new RequestIdInterceptor());
  app.useGlobalFilters(new JsonDetailFilter());

  // 显式 init：NestFactory.create() 只构建不注册路由（app.listen() 内部
  // 会 init，但 Test/直连场景不会）——不 init 则所有路由 404（fastify
  // 原生 404，非 JsonDetailFilter 的 detail 结构）。
  await app.init();

  // 关键：触发 fastify ready（context 的 preParsing 等 hooks 在 avvio
  // preReady 事件中构建）。Nest app.init() 不触发 fastify ready——
  // 生产 app.listen() 会，但测试/直连 server.listen() 不会 → 请求时
  // kRouteContext.preParsing undefined → preParsingHookRunner TypeError。
  // fastify.ready() 幂等，重复调用安全。
  const fastify = app.getHttpAdapter().getInstance();
  await fastify.ready();

  return app;
}
