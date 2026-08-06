/**
 * models 域控制器（TUI 契约用例 3、40）
 *
 * GET /api/models/current：透传 inference-svc /v1/models/current
 * （对齐 src/api_server.py:2159-2178 的返回形状：loaded/model_id/quant_type/
 *  model_name/model_path/engine/total_params/device/gpu_allocated_gb/...）。
 */
import { Body, Controller, Get, HttpCode, Post } from '@nestjs/common';
import { InferenceClient } from '../../clients/inference.client';

@Controller('models')
export class ModelsController {
  constructor(private readonly inference: InferenceClient) {}

  @Get()
  list(): Promise<unknown> {
    // 对齐 api_server.py:5059-5070 list_models：全部模型配置 + active_model_id
    return this.inference.request('GET', '/v1/models');
  }

  @Get('current')
  current(): Promise<unknown> {
    return this.inference.request('GET', '/v1/models/current');
  }

  @Post('load')
  @HttpCode(200)
  load(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/models/load', body);
  }

  @Post('unload')
  @HttpCode(200)
  unload(): Promise<unknown> {
    return this.inference.request('POST', '/v1/models/unload', undefined, {}, 30_000);
  }

  @Post('switch')
  @HttpCode(200)
  switchModel(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/models/switch', body);
  }

  @Get('available')
  available(): Promise<unknown> {
    return this.inference.request('GET', '/v1/models/available');
  }
}
