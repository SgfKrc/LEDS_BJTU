/**
 * M1 存储健康控制器 — GET /storage/health（本地/远端双健康，§12.3）。
 * GET /db/health 保持阶段 3.2 三态契约不变（兼容映射：本端点提供新结构）。
 */
import { Controller, Get } from '@nestjs/common';
import { StorageHealthService, StorageHealth } from '../../data/storage-health';

@Controller()
export class StorageHealthController {
  constructor(private readonly service: StorageHealthService) {}

  @Get('storage/health')
  async storageHealth(): Promise<StorageHealth> {
    return this.service.snapshot();
  }
}
