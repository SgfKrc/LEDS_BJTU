/**
 * chat 域控制器（阶段 2 数据面）
 *
 * 代理 inference-svc（docs/微服务架构改造计划.md §4.1）：
 *   POST /api/chat           → POST /v1/chat（完整对话）
 *   POST /api/chat/stream    → POST /v1/chat/stream（SSE 流式逐字节保真转发）
 *   POST /api/chat/clear     → POST /v1/chat/clear（清会话历史）
 *   POST /api/chat/generations/:id/cancel → POST /v1/chat/cancel
 *
 * 说明：/api/chat/upload（multipart）留待阶段 2.6 前处理（需 @fastify/multipart）。
 */
import {
  Body,
  Controller,
  HttpCode,
  HttpException,
  Param,
  Post,
  Req,
  Res,
} from '@nestjs/common';
import type { FastifyReply, FastifyRequest } from 'fastify';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { InferenceClient } from '../../clients/inference.client';

@Controller('chat')
export class ChatController {
  constructor(private readonly inference: InferenceClient) {}

  @Post()
  @HttpCode(200)
  chat(@Body() body: unknown): Promise<unknown> {
    return this.inference.request('POST', '/v1/chat', body);
  }

  @Post('clear')
  @HttpCode(200)
  clear(): Promise<unknown> {
    return this.inference.request('POST', '/v1/chat/clear');
  }

  @Post('generations/:id/cancel')
  @HttpCode(200)
  cancel(@Param('id') id: string): Promise<unknown> {
    return this.inference.request('POST', '/v1/chat/cancel', {
      generation_id: id,
    });
  }

  @Post('stream')
  async stream(
    @Req() req: FastifyRequest,
    @Res() reply: FastifyReply,
  ): Promise<void> {
    const body = (req as { body?: unknown }).body;
    const controller = new AbortController();
    // 客户端断开时中止上游 fetch（对齐 FastAPI StreamingResponse 取消语义）
    req.raw.on('close', () => controller.abort());

    let upstream: Response;
    try {
      upstream = await this.inference.chatStreamRaw(body, controller.signal);
    } catch (err) {
      throw new HttpException(
        `inference-svc 不可达: ${err instanceof Error ? err.message : String(err)}`,
        502,
      );
    }
    if (!upstream.ok || !upstream.body) {
      reply
        .status(upstream.status)
        .send({ detail: `inference upstream ${upstream.status}` });
      return;
    }

    const nodeStream = Readable.fromWeb(
      upstream.body as unknown as import('stream/web').ReadableStream,
    );
    reply.raw.writeHead(upstream.status, {
      'content-type':
        upstream.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    try {
      await pipeline(nodeStream, reply.raw, { end: true });
    } catch {
      // 客户端断开或上游中止：静默结束
    }
  }
}
