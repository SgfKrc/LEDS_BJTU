/**
 * QLH API 网关 — 应用工厂
 *
 * T1 阶段：最小可测骨架。
 *  - AppModule + /api 全局前缀
 *  - CatchAllController：未匹配路由返回 404（绕过 fastify 5.x 的
 *    fourOhFour 内部路径——supertest 在该路径下会崩溃
 *    "Cannot read properties of undefined (reading 'length')"，
 *    真实 HTTP 正常但测试组合不可用；catch-all 让所有请求走正常
 *    route handler，同时是 T2 统一 JSON detail 404 的落点）。
 * T2 起挂载 request-id 中间件、JSON detail 异常过滤器与各域控制器
 * （见 docs/TUI适配实施计划.md §5 与 docs/微服务架构改造计划.md §2.2）。
 */
import { All, Controller, Module, NotFoundException, Req } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import {
  FastifyAdapter,
  NestFastifyApplication,
} from '@nestjs/platform-fastify';

@Controller()
export class CatchAllController {
  // find-my-way 匹配优先级：静态路由 > 参数路由 > 通配符，
  // 因此 T2 起注册的 /api/health 等真实路由不会被本控制器抢匹配。
  @All('*')
  notFound(@Req() req: any): never {
    throw new NotFoundException(`Route ${req.method}:${req.url} not found`);
  }
}

@Module({ controllers: [CatchAllController] })
export class AppModule {}

export async function createApp(): Promise<NestFastifyApplication> {
  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule,
    new FastifyAdapter(),
  );
  // TUI 客户端硬编码 base_url + "/api" + path（tui_admin.py:218），前缀不可配置。
  app.setGlobalPrefix('api');
  return app;
}
