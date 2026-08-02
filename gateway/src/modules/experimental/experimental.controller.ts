/**
 * experimental 域控制器（阶段 2）
 *
 * POST /api/experimental/speculative → inference-svc /v1/speculative/run
 * （投机解码实验端点；QLH_SPEC_ENABLED 门控语义由 inference-svc 承载，
 *  对齐 api_server 实验端点：默认关闭时 404）。
 */
import { Body, Controller, HttpCode, Post } from '@nestjs/common';
import { InferenceClient } from '../../clients/inference.client';

@Controller('experimental')
export class ExperimentalController {
  constructor(private readonly inference: InferenceClient) {}

  @Post('speculative')
  @HttpCode(200)
  speculative(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/speculative/run', body);
  }
}
