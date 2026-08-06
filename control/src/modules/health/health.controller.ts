/**
 * 探活控制器（/health，编排健康检查用）。
 */
import { Controller, Get } from '@nestjs/common';

@Controller()
export class HealthController {
  @Get('health')
  health(): unknown {
    return { status: 'ok', service: 'control-svc', version: '0.1.0' };
  }
}
