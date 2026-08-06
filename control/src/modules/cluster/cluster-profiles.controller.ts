/**
 * M4 多集群档案控制器（计划 §12.2/§4.1）：
 *   POST /cluster/profiles/verify   无副作用探测主节点身份与版本（/bootstrap/info）
 *   POST /cluster/profiles          保存档案（cluster_id 防重复建群）
 *   GET  /cluster/profiles          列表
 *   DELETE /cluster/profiles/:id    删除
 *   GET  /cluster/endpoints         查看本机 advertised endpoint
 *   POST /cluster/endpoints/verify  自检 endpoint 可达性
 *
 * verify 不写任何状态（无副作用）；保存/更新经 repository（幂等）。
 */
import {
  Body, Controller, Delete, Get, HttpCode, HttpException, Param, Post,
} from '@nestjs/common';
import { ClusterProfileRepository, ClusterProfileRow } from '../../data/cluster-profile-repository';
import { ClusterEndpointsRepository } from '../../data/cluster-endpoints-repository';

class VerifyProfileRequest {
  name?: string;
  master_endpoint?: string;
}

class CreateProfileRequest {
  cluster_id?: string;
  name?: string;
  master_endpoint?: string;
  node_role?: 'master' | 'client' | 'unknown';
}

class UpdateEndpointRequest {
  scheme?: string;
  host?: string;
  port?: number;
}

interface BootstrapInfo {
  cluster_id?: string;
  version?: string;
  master?: { node_id?: string };
  status?: string;
}

@Controller('cluster')
export class ClusterProfilesController {
  constructor(
    private readonly profiles: ClusterProfileRepository,
    private readonly endpoints: ClusterEndpointsRepository,
  ) {}

  /** 无副作用探测：只读主节点 /bootstrap/info，不写任何状态。 */
  @Post('profiles/verify')
  @HttpCode(200)
  async verify(@Body() body: VerifyProfileRequest): Promise<Record<string, unknown>> {
    const endpoint = String(body?.master_endpoint ?? '').trim();
    if (!endpoint) {
      throw new HttpException('master_endpoint 必填（Tailscale IP 或 MagicDNS）', 422);
    }
    const normalized = endpoint.startsWith('http') ? endpoint : `http://${endpoint}`;
    const url = `${normalized.replace(/\/+$/, '')}/bootstrap/info`;
    let info: BootstrapInfo;
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      info = (await response.json()) as BootstrapInfo;
    } catch (err) {
      return {
        status: 'unreachable',
        master_endpoint: normalized,
        error: err instanceof Error ? err.message : String(err),
      };
    }
    return {
      status: 'ok',
      master_endpoint: normalized,
      cluster_id: info.cluster_id ?? null,
      version: info.version ?? null,
      master_node_id: info.master?.node_id ?? null,
    };
  }

  @Post('profiles')
  @HttpCode(201)
  create(@Body() body: CreateProfileRequest): Record<string, unknown> {
    const clusterId = body?.cluster_id?.trim() ?? '';
    const name = body?.name?.trim() ?? '';
    const endpoint = body?.master_endpoint?.trim() ?? '';
    if (!clusterId) throw new HttpException('cluster_id 必填', 422);
    if (!name) throw new HttpException('name 必填', 422);
    if (!endpoint) throw new HttpException('master_endpoint 必填', 422);
    const row = this.profiles.create({
      cluster_id: clusterId,
      name,
      master_endpoint: endpoint,
      node_role: body?.node_role,
    });
    return { status: 'created', profile: row };
  }

  @Get('profiles')
  list(): Record<string, unknown> {
    return { profiles: this.profiles.list() };
  }

  @Delete('profiles/:profileId')
  @HttpCode(200)
  delete(@Param('profileId') profileId: string): Record<string, unknown> {
    if (!this.profiles.delete(profileId)) {
      throw new HttpException(`档案不存在: ${profileId}`, 404);
    }
    return { status: 'deleted', profile_id: profileId };
  }

  @Get('endpoints')
  endpointsList(): Record<string, unknown> {
    return { endpoints: this.endpoints.list() };
  }

  /** 更新本机 advertised endpoint 并自检。 */
  @Post('endpoints/verify')
  @HttpCode(200)
  async verifyEndpoint(@Body() body: UpdateEndpointRequest): Promise<Record<string, unknown>> {
    const scheme = body?.scheme ?? 'http';
    const host = body?.host ?? '127.0.0.1';
    const port = Number(body?.port) || 8000;
    const url = `${scheme}://${host}:${port}/health`;
    let reachable = false;
    let detail = '';
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(5000) });
      reachable = response.ok;
      if (!reachable) detail = `HTTP ${response.status}`;
    } catch (err) {
      detail = err instanceof Error ? err.message : String(err);
    }
    return { endpoint: `${scheme}://${host}:${port}`, reachable, detail };
  }
}

export { ClusterProfileRow };
