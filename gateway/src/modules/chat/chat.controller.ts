/**
 * chat 域控制器（阶段 2 数据面）
 *
 * 代理 inference-svc（docs/微服务架构改造计划.md §4.1）：
 *   POST /api/chat           → POST /v1/chat（完整对话）
 *   POST /api/chat/stream    → POST /v1/chat/stream（SSE 流式逐字节保真转发）
 *   POST /api/chat/clear     → POST /v1/chat/clear（清会话历史）
 *   POST /api/chat/generations/:id/cancel → POST /v1/chat/cancel
 *   POST /api/chat/upload    → 本地解析文本文件（无上游；阶段 2.6 实现，
 *     语义对齐 api_server.py:1729-1816：扩展名校验/5MB/编码探测/5000 行截断）
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

/** 支持的文件类型（复制自 api_server.py:1712-1725 ALLOWED_TEXT_EXTENSIONS） */
const ALLOWED_TEXT_EXTENSIONS = new Set([
  '.txt', '.md', '.csv', '.py', '.json', '.log',
  '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf',
  '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
  '.sh', '.bash', '.zsh', '.ps1',
  '.cpp', '.c', '.h', '.java', '.go', '.rs', '.rb',
  '.sql', '.r', '.m', '.swift', '.kt',
  '.toml', '.properties', '.env',
]);
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024; // 5 MB
const MAX_UPLOAD_LINES = 5000; // 超过截断（保留前 5000 行）

/** 语言映射（复制自 api_server.py:1780-1790 lang_map） */
const LANG_MAP: Record<string, string> = {
  '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
  '.jsx': 'jsx', '.tsx': 'tsx', '.html': 'html', '.css': 'css',
  '.json': 'json', '.md': 'markdown', '.csv': 'csv',
  '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
  '.sh': 'bash', '.bash': 'bash', '.ps1': 'powershell',
  '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.java': 'java',
  '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
  '.sql': 'sql', '.r': 'r', '.swift': 'swift', '.kt': 'kotlin',
  '.toml': 'toml', '.ini': 'ini', '.cfg': 'ini',
};

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

  /**
   * POST /api/chat/upload — 上传文本文件（multipart）。
   * 语义对齐 api_server.py:1729-1816：扩展名校验 → 5MB 限制 →
   * UTF-8/GBK/latin-1 解码 → 5000 行截断 → 语言检测 → 结构化返回。
   * 纯本地解析，不依赖上游服务。
   */
  @Post('upload')
  @HttpCode(200)
  async upload(@Req() req: FastifyRequest): Promise<unknown> {
    const data = await req.file();
    if (!data) {
      throw new HttpException('缺少文件字段 file', 400);
    }
    const filename = data.filename || 'untitled';
    const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED_TEXT_EXTENSIONS.has(ext)) {
      const supported = [...ALLOWED_TEXT_EXTENSIONS].sort().join(', ');
      throw new HttpException(
        `不支持的文件类型: ${ext}。支持的格式: ${supported}`,
        400,
      );
    }

    const raw = Buffer.from(await data.toBuffer());
    if (raw.length > MAX_UPLOAD_BYTES) {
      throw new HttpException(
        `文件过大 (${(raw.length / 1024 / 1024).toFixed(1)} MB)，限制 ${MAX_UPLOAD_BYTES / 1024 / 1024} MB`,
        413,
      );
    }

    // 解码（尝试 UTF-8 → GBK → latin-1；GBK 用 TextDecoder 的 gbk 标签）
    let content: string | null = null;
    for (const enc of ['utf-8', 'gbk', 'latin1'] as const) {
      try {
        content = new TextDecoder(enc, { fatal: true }).decode(raw);
        break;
      } catch {
        // 继续尝试下一编码
      }
    }
    if (content === null) {
      throw new HttpException(
        '无法解码文件内容，请确认文件编码为 UTF-8 或 GBK',
        400,
      );
    }

    const lines = content.split('\n');
    const totalLines = lines.length;
    let truncated = false;
    if (totalLines > MAX_UPLOAD_LINES) {
      content = lines.slice(0, MAX_UPLOAD_LINES).join('\n');
      truncated = true;
    }

    const charCount = content.length;
    const wordCount = content.split(/\s+/).filter(Boolean).length;
    const language = LANG_MAP[ext] ?? 'plaintext';

    return {
      filename,
      extension: ext,
      language,
      char_count: charCount,
      word_count: wordCount,
      line_count: truncated ? MAX_UPLOAD_LINES : totalLines,
      total_lines: totalLines,
      truncated,
      truncated_lines: truncated ? totalLines - MAX_UPLOAD_LINES : 0,
      size_bytes: raw.length,
      content,
    };
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
