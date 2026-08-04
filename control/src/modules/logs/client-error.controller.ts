/**
 * 前端错误上报 — 对齐 api_server.py report_client_error (7243-7265)
 *
 * 无鉴权（前端上报，不校验 X-QLH-Log-Token，也无本机限制）。
 * 字段截断语义对齐 _truncate_log_field：超限时截断 + "...[truncated]"。
 */
import { Body, Controller, HttpCode, Post, Req } from '@nestjs/common';
import { RequestWithRequestId } from '../../common/request-id';
import { LogBuffer } from '../../data/log-buffer';
import { deviceIpOf, nodeIdOf } from '../../data/log-file-store';

interface ClientErrorReport {
  message?: string;
  source?: string;
  stack?: string;
  url?: string;
  line?: number;
  col?: number;
  user_agent?: string;
  extra?: Record<string, unknown>;
}

function truncateField(value: unknown, limit: number): string {
  const text = value === null || value === undefined ? '' : String(value);
  return text.length <= limit ? text : `${text.slice(0, limit)}...[truncated]`;
}

@Controller()
export class ClientErrorController {
  constructor(private readonly buffer: LogBuffer) {}

  @Post('logs/client-error')
  @HttpCode(200)
  reportClientError(
    @Body() report: ClientErrorReport | undefined,
    @Req() req: RequestWithRequestId,
  ): Record<string, unknown> {
    const r: ClientErrorReport = report && typeof report === 'object' ? report : {};
    const clientHost = req.ip || 'unknown';
    // RequestIdInterceptor 已在控制器执行前写入 req.requestId（对齐 Python
    // api_server.py:7247 的 request_id=%s 输出）
    const requestId = req.requestId || '';
    const message =
      'event=client_error' +
      ` source=${truncateField(r.source, 80)}` +
      ` message=${truncateField(r.message, 500)}` +
      ` url=${truncateField(r.url, 300)}` +
      ` line=${Number(r.line) || 0}` +
      ` col=${Number(r.col) || 0}` +
      ` client=${clientHost}` +
      ` ua=${truncateField(r.user_agent || '-', 200)}` +
      ` stack=${truncateField(r.stack, 2000)}` +
      ` extra=${truncateField(JSON.stringify(r.extra ?? {}), 500)}` +
      ` request_id=${requestId}`;
    console.error(`[client-error] ${message}`);
    this.buffer.append({
      level: 'ERROR',
      levelno: 40,
      name: 'client_error',
      message,
      filename: '',
      lineno: 0,
      funcName: 'report_client_error',
      request_id: requestId,
      node_id: nodeIdOf(),
      device_ip: deviceIpOf(),
      thread: '',
    });
    return { status: 'ok', logged: true };
  }
}
