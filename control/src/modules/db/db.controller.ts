/**
 * 数据库健康控制器 — 阶段 3.2 db/health 域（语义对齐 api_server.py:6669-6700）
 *
 * 端点（1 个）：
 *   GET /db/health → 三态：
 *     - {status:'ok', host, port, db}                    连接成功（SELECT 1）
 *     - {status:'unavailable', reason:'not_configured',
 *        message:'数据库未配置，正在使用本地文件存储',
 *        retry_in_seconds:0}                             QLH_DB_ENABLED 禁用
 *     - {status:'unavailable', reason:'connection_failed',
 *        message:<驱动错误>, retry_in_seconds:0}         配置启用但连接失败
 *
 * 降级说明（已记录计划文档）：api_server 的 driver_missing 分支（psycopg2
 * 未安装）在 TS 侧不存在——pg 为编译期依赖，恒可导入，故省略；retry_in_seconds
 * 恒 0（api_server 的退避重试来自 scheduler 连接池状态，control-svc 每次请求
 * 重新探测，对齐 ConfigDao「连接失败后不缓存失败状态」语义）。
 */
import { Controller, Get } from '@nestjs/common';
import { ConfigDao } from '../../data/config-dao';

@Controller()
export class DbController {
  constructor(private readonly dao: ConfigDao) {}

  @Get('db/health')
  async dbHealth(): Promise<Record<string, unknown>> {
    if (!this.dao.dbEnabled()) {
      return {
        status: 'unavailable',
        reason: 'not_configured',
        message: '数据库未配置，正在使用本地文件存储',
        retry_in_seconds: 0,
      };
    }
    const { ok, error } = await this.dao.ping();
    if (!ok) {
      return {
        status: 'unavailable',
        reason: 'connection_failed',
        message: error || '数据库连接失败',
        retry_in_seconds: 0,
      };
    }
    return { status: 'ok', ...this.dao.getConnectionInfo() };
  }
}
