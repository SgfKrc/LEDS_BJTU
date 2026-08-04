/**
 * 日志管理控制器 — 阶段 3.2 日志域（语义对齐 api_server.py:6883-7585）
 *
 * 端点（11 个路由方法 / 10 个唯一路径）：
 *   GET    /logs                    → {files:[{name,size,modified}]} mtime 降序
 *   GET    /logs/recent             → 内存缓冲最近日志（limit clamp 1-1000）
 *   GET    /logs/stats              → 文件 + 缓冲统计
 *   GET    /logs/download?name=     → 文件流下载（attachment）
 *   DELETE /logs                    → 清空全部 .log
 *   GET    /logs/export             → ZIP 打包（无文件 404）
 *   POST   /logs/client-error       → 前端错误上报（无鉴权）
 *   GET    /logs/node/{id}/recent   → 本机直读；远程降级（TCP 聚合未迁移）
 *   GET    /logs/nodes-summary      → {local, workers:[], total_workers:0}（聚合降级）
 *   GET    /logs/{filename:path}    → 末 1MB 内容
 *   DELETE /logs/{filename:path}    → 删除单个文件
 *
 * 除 client-error 外全部挂 LogAccessGuard（本机 IP 放行 / token 校验）。
 * 降级说明：node/{id}/recent 的远程 TCP 拉取与 nodes-summary 的在线节点
 * 聚合依赖 scheduler 集群状态，control-svc（独立进程）未迁移——远程节点
 * 返回空 remote 结果、workers 恒空数组；清理阶段接线 scheduler-svc 后补齐。
 * 角色检查（仅 master）未迁移：control-svc 假定部署于主节点控制面。
 */
import {
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Query,
  Res,
  UseGuards,
} from '@nestjs/common';
import type { FastifyReply } from 'fastify';
import { LogAccessGuard } from '../../common/log-token.guard';
import { LogBuffer } from '../../data/log-buffer';
import {
  deviceIpOf,
  isLogFilename,
  LogFileStore,
  nodeIdOf,
} from '../../data/log-file-store';

function normalizeLimit(raw: string | undefined, def: number, min: number, max: number): number {
  const n = raw === undefined || raw === '' ? def : Number(raw);
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(Math.floor(n), max));
}

@Controller()
@UseGuards(LogAccessGuard)
export class LogsController {
  constructor(
    private readonly buffer: LogBuffer,
    private readonly store: LogFileStore,
  ) {}

  // ---------- 列表 / recent / stats ----------

  @Get('logs')
  listFiles(): Record<string, unknown> {
    return this.store.listFiles();
  }

  @Get('logs/recent')
  recentLogs(
    @Query('limit') limitRaw?: string,
    @Query('level') level = '',
    @Query('name') name = '',
    @Query('node_id') nodeId = '',
    @Query('request_id') requestId = '',
  ): Record<string, unknown> {
    const limit = normalizeLimit(limitRaw, 200, 1, 1000);
    return this.buffer.query(limit, level, name, nodeId, requestId);
  }

  @Get('logs/stats')
  stats(): Record<string, unknown> {
    const fileStats = this.store.fileStats();
    const buf = this.buffer.stats();
    return {
      log_dir: this.store.logDir,
      files_count: fileStats.files.length,
      files_total_bytes: fileStats.totalBytes,
      buffer_size: buf.buffer_size,
      buffer_capacity: buf.buffer_capacity,
      buffer_total_seen: buf.buffer_total_seen,
      buffer_dropped_estimate: buf.buffer_dropped_estimate,
      levels: buf.levels,
      loggers: buf.loggers,
      nodes: buf.nodes,
      node_id: nodeIdOf(),
      device_ip: deviceIpOf(),
    };
  }

  // ---------- 下载 / 清空 / 导出 ----------

  @Get('logs/download')
  download(@Query('name') name: string, @Res() reply: FastifyReply): void {
    const safeName = this.validateName(name);
    let info;
    try {
      info = this.store.createDownloadStream(safeName);
    } catch (err) {
      this.throwNotFoundOr500(err, 'download');
    }
    reply
      .header('Content-Type', 'text/plain; charset=utf-8')
      .header('Content-Disposition', `attachment; filename="${safeName}"`)
      .send(info.stream);
  }

  @Delete('logs')
  @HttpCode(200)
  deleteAll(): Record<string, unknown> {
    const { deleted, failed } = this.store.deleteAll();
    return {
      status: failed.length ? 'partial' : 'ok',
      deleted,
      failed,
      deleted_count: deleted.length,
      failed_count: failed.length,
    };
  }

  @Get('logs/export')
  async exportZip(@Res() reply: FastifyReply): Promise<void> {
    const zip = await this.store.exportZip();
    if (!zip) {
      throw new HttpException('没有可导出的日志文件', 404);
    }
    const timestamp = tsStamp();
    const filename = `qlh-logs-${nodeIdOf() || 'node'}-${timestamp}.zip`;
    reply
      .header('Content-Type', 'application/zip')
      .header('Content-Disposition', `attachment; filename="${filename}"`)
      .send(zip);
  }

  // ---------- 多节点（降级：本地直读，远程聚合未迁移） ----------

  @Get('logs/node/:nodeId/recent')
  nodeRecent(
    @Param('nodeId') nodeId: string,
    @Query('limit') limitRaw?: string,
    @Query('level') level = '',
    @Query('name') name = '',
  ): Record<string, unknown> {
    const limit = normalizeLimit(limitRaw, 100, 1, 1000);
    const localNodeId = nodeIdOf();
    if (nodeId === localNodeId || nodeId === 'master') {
      const { entries, totalSeen } = this.buffer.snapshot();
      const filtered = this.buffer.query(limit, level, name, '', '');
      return {
        node_id: localNodeId,
        source: 'local',
        logs: filtered.logs,
        count: filtered.count,
        matched: filtered.matched,
        buffer_size: entries.length,
        total_seen: totalSeen,
      };
    }
    // 远程节点：TCP 拉取未迁移（并行共存期间网关仍走 legacy_control 桩）
    return {
      node_id: nodeId,
      source: 'remote-unavailable',
      logs: [],
      count: 0,
      matched: 0,
      buffer_size: 0,
    };
  }

  @Get('logs/nodes-summary')
  nodesSummary(): Record<string, unknown> {
    const buf = this.buffer.stats();
    return {
      local: {
        node_id: nodeIdOf(),
        buffer_size: buf.buffer_size,
        buffer_capacity: buf.buffer_capacity,
      },
      workers: [],
      total_workers: 0,
    };
  }

  // ---------- 通配（必须放最后，避免抢占特定路由；Fastify 静态优先兜底） ----------

  @Get('logs/*')
  readFile(@Param('*') filename: string): Record<string, unknown> {
    const safeName = this.validateName(filename);
    try {
      return this.store.readFileContent(safeName);
    } catch (err) {
      this.throwNotFoundOr500(err, 'read');
      return {};
    }
  }

  @Delete('logs/*')
  @HttpCode(200)
  deleteFile(@Param('*') filename: string): Record<string, unknown> {
    const safeName = this.validateName(filename);
    try {
      this.store.deleteFile(safeName);
    } catch (err) {
      this.throwNotFoundOr500(err, 'delete');
      return {};
    }
    return { status: 'ok', deleted: safeName, failed: [] };
  }

  // ---------- 工具 ----------

  private validateName(name: string): string {
    if (typeof name !== 'string' || !isLogFilename(name)) {
      throw new HttpException('无效的日志文件名', 400);
    }
    return name;
  }

  private throwNotFoundOr500(err: unknown, action: string): never {
    if (err instanceof Error && err.message === '文件不存在') {
      throw new HttpException('文件不存在', 404);
    }
    throw new HttpException(`读取失败: ${String(err)}`, 500);
  }
}

function tsStamp(): string {
  const d = new Date();
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
