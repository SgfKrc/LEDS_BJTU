import {
  Body, Controller, Delete, Get, HttpCode, HttpException, Post, Query, Req,
} from '@nestjs/common';
import { FastifyRequest } from 'fastify';
import * as path from 'path';
import {
  ArtifactRuntimeFilters, ArtifactRuntimeRecord, ArtifactRuntimeRepository,
  ArtifactRuntimeStatus,
} from '../../data/artifact-runtime-repository';
import { ModelImportAdmissionService } from '../../data/model-import-admission.service';
import { ModelRuntimeCheckService } from '../../data/model-runtime-check.service';

class RetryRuntimeCheckRequest {
  namespace?: string;
  name?: string;
  tag?: string;
  node_id?: string;
}

class ImportModelRequest {
  source_path?: string;
  namespace?: string;
  name?: string;
  tag?: string;
  node_id?: string;
}

@Controller('models')
export class RuntimeAdmissionController {
  constructor(
    private readonly repository: ArtifactRuntimeRepository,
    private readonly runtimeChecks: ModelRuntimeCheckService,
    private readonly imports: ModelImportAdmissionService,
  ) {}

  @Get('runtime-checks')
  list(
    @Query('artifact_id') artifactId?: string,
    @Query('node_id') nodeId?: string,
    @Query('runtime_profile') runtimeProfile?: string,
    @Query('status') status?: string,
  ): Record<string, unknown> {
    try {
      const filters = this.filters(artifactId, nodeId, runtimeProfile, status);
      return {
        runtime_checks: this.repository.list(filters)
          .map((record) => this.publicRecord(record)),
      };
    } catch (error) {
      throw this.unprocessable(error);
    }
  }

  @Post('runtime-checks/retry')
  @HttpCode(200)
  async retry(
    @Body() body: RetryRuntimeCheckRequest,
    @Req() request: FastifyRequest,
  ): Promise<Record<string, unknown>> {
    this.assertLocal(request);
    try {
      const namespace = body?.namespace ?? '';
      const name = body?.name ?? '';
      const tag = body?.tag ?? 'latest';
      const nodeId = this.localNodeId(body?.node_id);
      const runtimeCheck = await this.runtimeChecks.checkManifestReference(
        namespace, name, tag, nodeId,
      );
      return { runtime_check: runtimeCheck, runnable: runtimeCheck.status === 'ready' };
    } catch (error) {
      throw this.unprocessable(error);
    }
  }

  @Delete('runtime-checks')
  @HttpCode(200)
  invalidate(
    @Req() request: FastifyRequest,
    @Query('artifact_id') artifactId?: string,
    @Query('node_id') nodeId?: string,
    @Query('runtime_profile') runtimeProfile?: string,
    @Query('reason') reason?: string,
  ): Record<string, unknown> {
    this.assertLocal(request);
    try {
      if (!artifactId && !nodeId) {
        throw new Error('artifact_id or node_id is required');
      }
      const filters = this.filters(artifactId, nodeId, runtimeProfile);
      const invalidated = this.repository.invalidate(
        filters, reason?.trim() || 'runtime_context_changed',
      );
      return { status: 'invalidated', count: invalidated.length, runtime_checks: invalidated };
    } catch (error) {
      throw this.unprocessable(error);
    }
  }

  @Post('imports')
  @HttpCode(201)
  async importLocal(
    @Body() body: ImportModelRequest,
    @Req() request: FastifyRequest,
  ): Promise<Record<string, unknown>> {
    this.assertLocal(request);
    try {
      if (!body?.source_path?.trim()) throw new Error('source_path is required');
      const report = await this.imports.importLocal(
        path.resolve(body.source_path),
        { namespace: body.namespace, name: body.name, tag: body.tag },
        this.localNodeId(body.node_id),
      );
      return {
        status: report.failed > 0 ? 'failed' : 'imported',
        runnable: report.runtime_check?.status === 'ready',
        report,
      };
    } catch (error) {
      throw this.unprocessable(error);
    }
  }

  private filters(
    artifactId?: string,
    nodeId?: string,
    runtimeProfile?: string,
    status?: string,
  ): ArtifactRuntimeFilters {
    if (artifactId && !/^sha256:[0-9a-f]{64}$/.test(artifactId)) {
      throw new Error('artifact_id is invalid');
    }
    if (nodeId && !/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(nodeId)) {
      throw new Error('node_id is invalid');
    }
    if (runtimeProfile && !/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(runtimeProfile)) {
      throw new Error('runtime_profile is invalid');
    }
    if (status && !['ready', 'load_failed', 'resource_rejected', 'stale'].includes(status)) {
      throw new Error('status is invalid');
    }
    return {
      artifactId, nodeId, runtimeProfile,
      status: status as ArtifactRuntimeStatus | undefined,
    };
  }

  private localNodeId(requested?: string): string {
    const local = process.env.QLH_NODE_ID?.trim() || 'local';
    if (requested && requested !== local) {
      throw new Error(`node_id must match the local node: ${local}`);
    }
    return local;
  }

  private publicRecord(record: ArtifactRuntimeRecord): ArtifactRuntimeRecord {
    const details = { ...(record.details ?? {}) };
    delete details.stderr_tail;
    return { ...record, details };
  }

  private assertLocal(request: FastifyRequest): void {
    const address = request.ip;
    if (address !== '127.0.0.1' && address !== '::1'
        && address !== '::ffff:127.0.0.1') {
      throw new HttpException('runtime admission changes are loopback-only', 403);
    }
  }

  private unprocessable(error: unknown): HttpException {
    return new HttpException(error instanceof Error ? error.message : String(error), 422);
  }
}
