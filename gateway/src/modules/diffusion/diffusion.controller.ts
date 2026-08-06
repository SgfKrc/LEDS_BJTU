/** Stable Diffusion 1.5 data-plane proxy (/api/diffusion -> inference-svc). */
import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Post,
  Req,
  Res,
} from '@nestjs/common';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { InferenceClient } from '../../clients/inference.client';

const LOCAL_PATH_CLIENTS = new Set([
  '127.0.0.1',
  '::1',
  '::ffff:127.0.0.1',
]);
const LIFECYCLE_TIMEOUT_MS = 120_000;

@Controller('diffusion')
export class DiffusionController {
  constructor(private readonly inference: InferenceClient) {}

  private requireLocalPathClient(req: FastifyRequest): void {
    if (!LOCAL_PATH_CLIENTS.has(req.ip)) {
      throw new HttpException(
        'SD 模型路径检查和登记仅允许在主节点本机执行',
        403,
      );
    }
  }

  @Get('capabilities')
  capabilities(): Promise<unknown> {
    return this.inference.request('GET', '/v1/diffusion/capabilities');
  }

  @Post('artifacts/inspect')
  @HttpCode(200)
  inspect(@Req() req: FastifyRequest, @Body() body: unknown): Promise<unknown> {
    this.requireLocalPathClient(req);
    return this.inference.request(
      'POST',
      '/v1/diffusion/artifacts/inspect',
      body,
      {},
      LIFECYCLE_TIMEOUT_MS,
    );
  }

  @Post('artifacts/register')
  @HttpCode(200)
  register(@Req() req: FastifyRequest, @Body() body: unknown): Promise<unknown> {
    this.requireLocalPathClient(req);
    return this.inference.request(
      'POST',
      '/v1/diffusion/artifacts/register',
      body,
      {},
      LIFECYCLE_TIMEOUT_MS,
    );
  }

  @Get('artifacts')
  artifacts(): Promise<unknown> {
    return this.inference.request('GET', '/v1/diffusion/artifacts');
  }

  @Get('assets/catalog')
  assetCatalog(): Promise<unknown> {
    return this.inference.request('GET', '/v1/diffusion/assets/catalog');
  }

  @Get('assets/:id/status')
  assetStatus(@Param('id') id: string): Promise<unknown> {
    return this.inference.request(
      'GET',
      `/v1/diffusion/assets/${encodeURIComponent(id)}/status`,
    );
  }

  @Post('assets/:id/download')
  @HttpCode(202)
  downloadAsset(
    @Req() req: FastifyRequest,
    @Param('id') id: string,
    @Body() body: unknown,
  ): Promise<unknown> {
    this.requireLocalPathClient(req);
    return this.inference.request(
      'POST',
      `/v1/diffusion/assets/${encodeURIComponent(id)}/download`,
      body,
    );
  }

  @Post('assets/import')
  @HttpCode(200)
  importAsset(
    @Req() req: FastifyRequest,
    @Body() body: unknown,
  ): Promise<unknown> {
    this.requireLocalPathClient(req);
    return this.inference.request(
      'POST',
      '/v1/diffusion/assets/import',
      body,
      {},
      LIFECYCLE_TIMEOUT_MS,
    );
  }

  @Post('load')
  @HttpCode(200)
  load(@Body() body: unknown): Promise<unknown> {
    return this.inference.request(
      'POST',
      '/v1/diffusion/load',
      body,
      {},
      LIFECYCLE_TIMEOUT_MS,
    );
  }

  @Post('unload')
  @HttpCode(200)
  unload(): Promise<unknown> {
    return this.inference.request(
      'POST',
      '/v1/diffusion/unload',
      undefined,
      {},
      LIFECYCLE_TIMEOUT_MS,
    );
  }

  @Post('generate')
  @HttpCode(202)
  generate(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/diffusion/generate', body);
  }

  @Post('blobs')
  @HttpCode(201)
  async uploadBlob(@Req() req: FastifyRequest): Promise<unknown> {
    const upload = await req.file();
    if (!upload) {
      throw new HttpException('missing multipart file field', 400);
    }
    const fields = upload.fields as Record<string, { value?: unknown }>;
    const purpose = fields.purpose?.value;
    if (purpose !== 'input_image' && purpose !== 'mask') {
      throw new HttpException('purpose must be input_image or mask', 400);
    }
    return this.inference.requestMultipart(
      'POST',
      '/v1/diffusion/blobs',
      { purpose },
      {
        data: Buffer.from(await upload.toBuffer()),
        filename: upload.filename || 'upload',
        contentType: upload.mimetype,
      },
    );
  }

  @Post('edit')
  @HttpCode(202)
  edit(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/diffusion/edit', body);
  }

  @Get('jobs/:id')
  job(@Param('id') id: string): Promise<unknown> {
    return this.inference.request(
      'GET',
      `/v1/diffusion/jobs/${encodeURIComponent(id)}`,
    );
  }

  @Post('jobs/:id/cancel')
  @HttpCode(200)
  cancel(@Param('id') id: string): Promise<unknown> {
    return this.inference.request(
      'POST',
      `/v1/diffusion/jobs/${encodeURIComponent(id)}/cancel`,
    );
  }

  @Get('blobs/:id')
  async blob(
    @Param('id') id: string,
    @Res() reply: FastifyReply,
  ): Promise<void> {
    let upstream: Response;
    try {
      upstream = await this.inference.diffusionBlobRaw(id);
    } catch (err) {
      throw new HttpException(
        `inference-svc 不可达: ${err instanceof Error ? err.message : String(err)}`,
        502,
      );
    }
    if (!upstream.ok) {
      let detail = `inference upstream ${upstream.status}`;
      try {
        const payload = (await upstream.json()) as { detail?: unknown };
        if (payload?.detail !== undefined) detail = String(payload.detail);
      } catch {
        // Keep the stable upstream status fallback.
      }
      throw new HttpException(detail, upstream.status);
    }
    const content = Buffer.from(await upstream.arrayBuffer());
    reply.status(upstream.status);
    reply.header(
      'content-type',
      upstream.headers.get('content-type') || 'image/png',
    );
    for (const header of ['cache-control', 'etag', 'content-disposition']) {
      const value = upstream.headers.get(header);
      if (value) reply.header(header, value);
    }
    reply.send(content);
  }

  @Delete('blobs/:id')
  deleteBlob(@Param('id') id: string): Promise<unknown> {
    return this.inference.request(
      'DELETE',
      `/v1/diffusion/blobs/${encodeURIComponent(id)}`,
    );
  }
}
