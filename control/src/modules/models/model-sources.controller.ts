import {
  Body, Controller, Delete, Get, HttpCode, HttpException, Param, Post, Put,
} from '@nestjs/common';
import {
  ModelSourceInput, ModelSourceProvider, ModelSourceRepository,
} from '../../data/model-source-repository';
import { PullPreflightService } from '../../data/pull-preflight.service';
import { PullPreflightResult } from '../../data/pull-preflight.service';

class SourceRequest {
  name?: string;
  provider?: ModelSourceProvider;
  endpoint?: string;
  credential_ref?: string | null;
  priority?: number;
  enabled?: boolean;
  token?: string;
  access_token?: string;
}

class ResolveRequest {
  source_id?: string;
  repo_id?: string;
  requested_revision?: string;
  allow_patterns?: string[];
  token?: string;
  access_token?: string;
}

@Controller('models')
export class ModelSourcesController {
  constructor(
    private readonly sources: ModelSourceRepository,
    private readonly preflight: PullPreflightService,
  ) {}

  @Get('sources')
  list(): Record<string, unknown> {
    return { sources: this.sources.list() };
  }

  @Put('sources/:sourceId')
  upsert(
    @Param('sourceId') sourceId: string,
    @Body() body: SourceRequest,
  ): Record<string, unknown> {
    this.rejectPlaintextToken(body);
    if (!body?.name || !body?.provider || !body?.endpoint) {
      throw new HttpException('name/provider/endpoint 必填', 422);
    }
    if (!['huggingface', 'modelscope'].includes(body.provider)) {
      throw new HttpException('provider 不受支持', 422);
    }
    try {
      const input: ModelSourceInput = {
        source_id: sourceId,
        name: body.name,
        provider: body.provider,
        endpoint: body.endpoint,
        credential_ref: body.credential_ref ?? null,
        priority: body.priority ?? 100,
        enabled: body.enabled ?? true,
      };
      return { status: 'saved', source: this.sources.upsert(input) };
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  @Delete('sources/:sourceId')
  @HttpCode(200)
  delete(@Param('sourceId') sourceId: string): Record<string, unknown> {
    if (!this.sources.delete(sourceId)) {
      throw new HttpException(`source 不存在: ${sourceId}`, 404);
    }
    return { status: 'deleted', source_id: sourceId };
  }

  @Post('sources/reset')
  @HttpCode(200)
  reset(): Record<string, unknown> {
    return { status: 'reset', sources: this.sources.reset() };
  }

  @Post('resolve')
  @HttpCode(200)
  async resolve(@Body() body: ResolveRequest): Promise<PullPreflightResult> {
    this.rejectPlaintextToken(body);
    const repoId = body?.repo_id?.trim();
    if (!repoId) throw new HttpException('repo_id 必填', 422);
    const source = body?.source_id
      ? this.sources.get(body.source_id)
      : this.sources.preferred('huggingface');
    if (!source) throw new HttpException('model source 不存在', 422);
    try {
      return await this.preflight.resolve({
        source,
        repoId,
        requestedRevision: body.requested_revision?.trim() || 'main',
        allowPatterns: body.allow_patterns ?? null,
      });
    } catch (error) {
      throw new HttpException(
        error instanceof Error ? error.message : String(error), 422,
      );
    }
  }

  private rejectPlaintextToken(body: { token?: string; access_token?: string }): void {
    if (body?.token || body?.access_token) {
      throw new HttpException('token 明文禁止落库；请使用 credential_ref', 422);
    }
  }
}
