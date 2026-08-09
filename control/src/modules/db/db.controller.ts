/** 兼容健康端点：M1.3 后报告主节点 SQLite，不再发起远端连接。 */
import { Controller, Get } from '@nestjs/common';
import { StorageHealthService } from '../../data/storage-health';

@Controller()
export class DbController {
  constructor(private readonly storage: StorageHealthService) {}

  @Get('db/health')
  async dbHealth(): Promise<Record<string, unknown>> {
    const snapshot = await this.storage.snapshot();
    const local = snapshot.local;
    if (!local.writable) {
      return {
        status: 'unavailable',
        reason: 'local_storage_unavailable',
        message: '主节点 SQLite 不可写',
        retry_in_seconds: 0,
        backend: 'sqlite',
        path: local.path,
        schema_version: local.schema_version,
      };
    }
    return {
      status: 'ok',
      backend: 'sqlite',
      mode: 'local_only',
      path: local.path,
      schema_version: local.schema_version,
      legacy_remote: snapshot.remote.status,
    };
  }
}
