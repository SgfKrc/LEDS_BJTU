/**
 * bootstrap 首连引导控制器 — 阶段 3.2 bootstrap 域（语义对齐 api_server.py:5365-5468）
 *
 * 端点（2 个）：
 *   POST /bootstrap/first-connect → 首连部署响应（cluster/node/android 结构）
 *   GET  /bootstrap/info          → 发现协议（tailnet 探测契约）
 *
 * 降级说明（已记录计划文档）：真实注册（scheduler.manual_register_node）与
 * cluster_secret 管理（ensure_local_cluster_secret）在 Python 侧——本实现
 * 以环境变量承载（QLH_CLUSTER_ID / QLH_CLUSTER_SECRET），不写调度器注册表；
 * master 角色恒 true（control-svc 假定主节点控制面）；master_tcp_host 不做
 * tailnet 探测（直接取请求 Host）；QLH_MASTER_TCP_PORT 默认 8888（对齐
 * config.SERVER_PORT）。清理阶段接 scheduler-svc 后补齐真实注册与密钥下发。
 */
import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpException,
  Post,
  Req,
} from '@nestjs/common';
import type { FastifyRequest } from 'fastify';
import { nodeIdOf } from '../../data/log-file-store';
import {
  isTrustedBootstrapSource,
  normalizeNodeId,
  normalizeNodeType,
  resolveTrustedCidrs,
} from '../../common/bootstrap-trust';
import { ClusterEndpointsRepository } from '../../data/cluster-endpoints-repository';
import { buildEndpointUrl, canonicalHost } from '../../common/network-address';

interface FirstConnectBootstrapRequest {
  node_id?: string;
  node_type?: string;
  hostname?: string;
  platform?: string;
  app_variant?: string;
  app_version?: string;
  capabilities?: Record<string, unknown>;
}

function envDisabled(env: NodeJS.ProcessEnv = process.env, key: string): boolean {
  const raw = env[key];
  return raw !== undefined && ['0', 'false', 'no'].includes(raw.trim().toLowerCase());
}

function envInt(env: NodeJS.ProcessEnv, key: string, def: number): number {
  const raw = env[key]?.trim();
  if (!raw) return def;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : def;
}

@Controller()
export class BootstrapController {
  constructor(
    private readonly endpoints: ClusterEndpointsRepository,
  ) {}

  @Post('bootstrap/first-connect')
  @HttpCode(200)
  firstConnect(
    @Body() body: FirstConnectBootstrapRequest | undefined,
    @Req() req: FastifyRequest,
  ): Record<string, unknown> {
    // 开关（对齐 api_server.py:5373）
    if (envDisabled(process.env, 'QLH_BOOTSTRAP_ENABLED')) {
      throw new HttpException('bootstrap disabled', 403);
    }
    const peerHost = req.ip || '';
    // 信任校验（对齐 :5379-5382；QLH_BOOTSTRAP_REQUIRE_TAILSCALE 默认 true）
    if (!envDisabled(process.env, 'QLH_BOOTSTRAP_REQUIRE_TAILSCALE')) {
      if (!isTrustedBootstrapSource(peerHost, resolveTrustedCidrs())) {
        throw new HttpException('source network is not trusted', 403);
      }
    }

    const nodeType = normalizeNodeType(body?.node_type);
    const nodeId = normalizeNodeId(body?.node_id, nodeType);
    if (nodeId === 'master') {
      throw new HttpException('reserved node_id', 400);
    }

    // 主节点地址（对齐 :5400-5407；hostname 取请求 Host，端口取本地监听端口）
    const apiHost = canonicalHost(req.hostname || peerHost);
    const apiPort = req.socket.localPort || envInt(process.env, 'QLH_MASTER_API_PORT', 8000);
    const tcpPort = envInt(process.env, 'QLH_MASTER_TCP_PORT', 8888);
    const clusterSecret =
      process.env.QLH_CLUSTER_SECRET?.trim() || 'local-bootstrap-not-configured';

    const pipelineWorker = nodeType === 'pc';
    const hostname = body?.hostname || nodeId;
    const address = peerHost;

    // M1：登记主节点 endpoint 到本地 SQLite 事实源（cluster_id 保留不重复）
    const clusterId = process.env.QLH_CLUSTER_ID?.trim() || 'qlh-default';
    try {
      this.endpoints.upsert({
        endpoint_id: `ep_${clusterId}`,
        cluster_id: clusterId,
        name: 'bootstrap-registered',
        scheme: 'http',
        host: apiHost,
        port: apiPort,
        status: 'active',
        last_verified_at: new Date().toISOString(),
      });
    } catch (err) {
      // 登记失败不阻断首连响应（本地事实源尽力写入）
      // eslint-disable-next-line no-console
      console.warn(`[control-svc] endpoint 登记失败: ${err instanceof Error ? err.message : String(err)}`);
    }

    return {
      status: 'ok',
      cluster: {
        cluster_id: clusterId,
        master_api_host: apiHost,
        master_api_port: apiPort,
        master_tcp_host: apiHost,
        master_tcp_port: tcpPort,
        cluster_secret: clusterSecret,
      },
      node: {
        node_id: nodeId,
        role: 'client',
        node_type: nodeType,
        pipeline_worker: pipelineWorker,
      },
      android: {
        presence_interval_seconds: 45,
        pipeline_worker: false,
        model_manifest_url: buildEndpointUrl(
          'http', apiHost, apiPort, '/api/models/downloadable',
        ),
      },
    };
  }

  @Get('bootstrap/info')
  bootstrapInfo(@Req() req: FastifyRequest): Record<string, unknown> {
    // 无条件信任校验（对齐 api_server.py:5459）
    const peerHost = req.ip || '';
    if (!isTrustedBootstrapSource(peerHost, resolveTrustedCidrs())) {
      throw new HttpException('source network is not trusted', 403);
    }
    return {
      status: 'ok',
      is_master: true, // 降级：control-svc 假定主节点控制面
      node_id: nodeIdOf(),
      master_api_port: envInt(process.env, 'QLH_MASTER_API_PORT', 8000),
      master_tcp_port: envInt(process.env, 'QLH_MASTER_TCP_PORT', 8888),
    };
  }
}
