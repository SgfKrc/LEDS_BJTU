/**
 * QLH control-svc — 控制面服务（微服务架构改造计划 阶段 3）
 *
 * 对外契约：与 api_server 控制面端点同路径（网关透传：对外 /api/* 去掉
 * /api 前缀转发至此，对齐 legacy_control 的透传模式）。
 *
 * 3.1 骨架：/health + settings 域（首迁域，cluster_config 表语义对齐 db.py）
 * 3.2 逐域迁移：sessions/conversations/logs/review/models/workflows/bootstrap
 */
import { All, Controller, Module, NotFoundException, OnApplicationBootstrap, Req } from '@nestjs/common';
import * as path from 'path';
import { NestFactory } from '@nestjs/core';
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { JsonDetailFilter } from './common/json-detail.filter';
import { RequestIdInterceptor } from './common/request-id';
import { ConfigDao } from './data/config-dao';
import { LogBuffer } from './data/log-buffer';
import { LogFileStore } from './data/log-file-store';
import { ReviewStore } from './data/review-store';
import { SessionStore } from './data/session-store';
import { WorkflowJournalStore } from './data/workflow-journal-store';
import { SqliteStore } from './data/sqlite-store';
import { OutboxService } from './data/outbox.service';
import { StorageHealthService } from './data/storage-health';
import { PostgresProjector } from './data/postgres-projector';
import { ClusterSettingsRepository } from './data/cluster-settings-repository';
import { ModelRegistryRepository } from './data/model-registry-repository';
import { ClusterEndpointsRepository } from './data/cluster-endpoints-repository';
import { LegacyMigration } from './data/legacy-migration';
import { ArtifactStore } from './data/artifact-store';
import { ModelInspector } from './data/model-inspector';
import { ModelImportService } from './data/model-import-service';
import { ModelBatchImporter } from './data/model-batch-import';
import { PullJobService } from './data/pull-job.service';
import { PullJobExecutor } from './data/pull-job-executor';
import { HfResolver } from './data/hf-resolver';
import { HfDownloader } from './data/hf-downloader';
import { ModelDiskBudget } from './data/model-disk-budget';
import { ModelSourceRepository } from './data/model-source-repository';
import { PullPreflightService } from './data/pull-preflight.service';
import { DeploymentSimulator } from './data/deployment-simulator';
import { ClusterProfileRepository } from './data/cluster-profile-repository';
import { ClusterProfileSelectionService } from './data/cluster-profile-selection';
import { ClusterDiscoveryService } from './data/cluster-discovery.service';
import { HealthController } from './modules/health/health.controller';
import { DbController } from './modules/db/db.controller';
import { StorageHealthController } from './modules/db/storage-health.controller';
import { BootstrapController } from './modules/bootstrap/bootstrap.controller';
import { ClientErrorController } from './modules/logs/client-error.controller';
import { LogsController } from './modules/logs/logs.controller';
import { ModelsController } from './modules/models/models.controller';
import { PullJobController } from './modules/models/pull-job.controller';
import { ModelSourcesController } from './modules/models/model-sources.controller';
import { DeploymentSimulationController } from './modules/models/deployment-simulation.controller';
import { ClusterProfilesController } from './modules/cluster/cluster-profiles.controller';
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
    DbController,
    SettingsController,
    SessionsController,
    LogsController,
    ClientErrorController,
    ReviewController,
    ModelsController,
    PullJobController,
    ModelSourcesController,
    DeploymentSimulationController,
    ClusterProfilesController,
    WorkflowsController,
    BootstrapController,
    StorageHealthController,
    CatchAllController,
  ],
  providers: [
    ConfigDao,
    SessionStore,
    LogBuffer,
    LogFileStore,
    ReviewStore,
    ReviewService,
    WorkflowJournalStore,
    // M1：本地 SQLite 事实源（唯一写者，惰性打开）+ outbox + 投影
    {
      provide: SqliteStore,
      useFactory: () => new SqliteStore(),
    },
    OutboxService,
    StorageHealthService,
    PostgresProjector,
    // M1 任务 2：三域 repository + 旧源迁移执行器
    ClusterSettingsRepository,
    ModelRegistryRepository,
    ClusterEndpointsRepository,
    LegacyMigration,
    // M2：内容寻址工件库 + 静态 inspector + 本地导入
    ArtifactStore,
    ModelInspector,
    ModelImportService,
    ModelBatchImporter,
    // M3：pull job + HF resolve/下载
    PullJobService,
    HfResolver,
    HfDownloader,
    ModelDiskBudget,
    ModelSourceRepository,
    PullPreflightService,
    PullJobExecutor,
    DeploymentSimulator,
    // M4：多集群档案
    ClusterProfileRepository,
    ClusterProfileSelectionService,
    ClusterDiscoveryService,
  ],
})
export class AppModule implements OnApplicationBootstrap {
  constructor(
    private readonly migration: LegacyMigration,
    private readonly pullExecutor: PullJobExecutor,
  ) {}

  /** 启动时自动执行旧源一次性迁移（幂等；源不存在/为空则无操作）。 */
  async onApplicationBootstrap(): Promise<void> {
    const cwd = process.cwd();
    await this.migration.run({
      // catalog seed 由 scripts/export_model_catalog.py 生成（build/ 为忽略产物）
      catalogSeedPath: process.env.QLH_CATALOG_SEED_PATH
        || path.join(cwd, '..', 'build', 'model-fleet', 'catalog-seed.json'),
      // 旧 control-svc JSON 注册表（ModelRegistryStore 默认路径）
      registryJsonPath: process.env.QLH_LEGACY_REGISTRY_PATH
        || path.join(cwd, 'model_registry.json'),
    });
    this.pullExecutor.resumeActive();
  }
}

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
