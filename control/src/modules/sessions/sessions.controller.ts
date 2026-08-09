/**
 * 会话/对话管理控制器 — 阶段 3.2 首迁域（语义对齐 api_server.py:6245-6662）
 *
 * 迁移范围：主节点 SQLite 会话/消息表；旧 local_store JSON 首次启动只读导入，
 * 显式注入目录时保留兼容测试/回滚路径。远端数据库已退出生产运行时，
 * sync-status 保留 db_connected 字段并固定为 false 以兼容旧客户端。
 * source 恒为 "local_store"（TS 无 session_histories 内存态，
 * memory_fallback 分支折叠为本地文件空结果，行为等价：均返回空消息）。
 *
 * 对外路径无 /api 前缀（网关去掉 /api 后透传至此），与 settings 域一致。
 * 错误体走 JsonDetailFilter 的 {detail, request_id} 结构（对齐 FastAPI）。
 */
import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Post,
  Put,
  Query,
} from '@nestjs/common';
import { ChatMessage, isValidSessionId, SessionMeta, SessionStore } from '../../data/session-store';

interface CreateSessionRequest {
  title?: string;
  first_message?: string;
}

interface RenameSessionRequest {
  title?: string;
}

@Controller()
export class SessionsController {
  constructor(private readonly store: SessionStore) {}

  // ---------- 对话云同步状态（对齐 api_server.py:6245-6264） ----------

  @Get('conversations/sync-status')
  syncStatus(): Record<string, unknown> {
    return {
      save_history: true, // 主节点 SQLite 已承载会话历史
      db_connected: false,
      local_save_enabled: true, // localStorage 始终可用
      local_store_enabled: true, // 本地文件降级始终可用
      cloud_sync_enabled: false,
    };
  }

  // ---------- 对话历史（对齐 api_server.py:6271-6363） ----------

  @Get('conversations')
  getConversations(
    @Query('session_id') sessionId = 'default',
    @Query('limit') limitRaw?: string,
  ): Record<string, unknown> {
    this.validateSessionId(sessionId);
    const limit = this.parseLimit(limitRaw);
    const messages = this.store.loadMessages(sessionId, limit);
    return {
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      count: messages.length,
      source: 'local_store',
    };
  }

  @Delete('conversations')
  @HttpCode(200)
  clearConversations(@Query('session_id') sessionId = 'default'): Record<string, unknown> {
    this.validateSessionId(sessionId);
    // 对齐 api_server.py:6331-6334：default 时解析为活跃会话
    const resolved =
      sessionId === 'default' && this.store.activeSessionId
        ? this.store.activeSessionId
        : sessionId;
    const deletedCount = this.store.clearMessages(resolved);
    return {
      status: 'cleared',
      session_id: resolved,
      deleted_count: deletedCount,
    };
  }

  // ---------- 会话管理（对齐 api_server.py:6379-6570） ----------

  @Post('sessions')
  @HttpCode(200)
  createSession(@Body() body?: CreateSessionRequest): Record<string, unknown> {
    let title = '新对话';
    const firstMsg = body?.first_message;
    if (typeof firstMsg === 'string' && firstMsg.trim()) {
      const t = firstMsg.trim();
      title = t.length > 30 ? `${t.slice(0, 30)}...` : t;
    } else if (typeof body?.title === 'string' && body.title) {
      title = body.title;
    }
    const sessionId = this.store.newSessionId();
    this.store.createSession(sessionId, title);
    this.store.activeSessionId = sessionId;
    return {
      id: sessionId,
      title,
      message_count: 0,
      active: true,
    };
  }

  @Get('sessions')
  listSessions(
    @Query('limit') limitRaw?: string,
    @Query('offset') offsetRaw?: string,
  ): Record<string, unknown> {
    const limit = this.parseInt(limitRaw, 50);
    const offset = this.parseInt(offsetRaw, 0);
    const { sessions, total } = this.store.listSessions(limit, offset);
    return {
      sessions: sessions.map((s) => ({
        id: s.id,
        title: s.title,
        message_count: s.message_count,
        created_at: s.created_at,
        updated_at: s.updated_at,
      })),
      active_session_id: this.store.activeSessionId,
      total,
    };
  }

  @Get('sessions/:id')
  getSession(@Param('id') sessionId: string): Record<string, unknown> {
    this.validateSessionId(sessionId);
    const meta = this.store.getSession(sessionId);
    if (meta) {
      // 对齐 api_server.py:6498-6500：用实际消息数优先（本地文件真实条数）
      const messageCount = this.store.messageCountOf(sessionId) || meta.message_count || 0;
      return {
        id: meta.id,
        title: meta.title,
        message_count: messageCount,
        created_at: meta.created_at,
        updated_at: meta.updated_at,
        active: sessionId === this.store.activeSessionId,
      };
    }
    // 最终降级：未知会话返回空壳（对齐 api_server.py:6505-6511，无 404）
    return {
      id: sessionId,
      title: '新对话',
      message_count: 0,
      active: sessionId === this.store.activeSessionId,
    };
  }

  @Put('sessions/:id')
  renameSession(
    @Param('id') sessionId: string,
    @Body() body: RenameSessionRequest,
  ): SessionMeta {
    this.validateSessionId(sessionId);
    const title = body?.title;
    if (typeof title !== 'string' || title.length < 1 || title.length > 256) {
      // 对齐 pydantic RenameSessionRequest(min_length=1, max_length=256) 校验失败
      // （FastAPI 对 body 校验失败返回 422）
      throw new HttpException('title 必须为 1-256 字符的字符串', 422);
    }
    const updated = this.store.renameSession(sessionId, title);
    if (!updated) {
      throw new HttpException(`会话不存在: ${sessionId}`, 404);
    }
    return updated;
  }

  @Delete('sessions/:id')
  @HttpCode(200)
  deleteSession(@Param('id') sessionId: string): Record<string, unknown> {
    this.validateSessionId(sessionId);
    this.store.deleteSession(sessionId);
    if (this.store.activeSessionId === sessionId) {
      this.store.activeSessionId = null;
    }
    return { status: 'deleted', session_id: sessionId };
  }

  @Post('sessions/:id/activate')
  @HttpCode(200)
  activateSession(@Param('id') sessionId: string): Record<string, unknown> {
    this.validateSessionId(sessionId);
    // 对齐 api_server.py:6578-6589：切换活跃会话并返回消息历史
    this.store.activeSessionId = sessionId;
    const history = this.store.loadMessages(sessionId, 0);
    return {
      session_id: sessionId,
      messages: history.map((m) => ({ role: m.role, content: m.content })),
      count: history.length,
    };
  }

  @Delete('sessions/:id/turns/:turnIndex')
  @HttpCode(200)
  deleteTurn(
    @Param('id') sessionId: string,
    @Param('turnIndex') turnIndexRaw: string,
  ): Record<string, unknown> {
    this.validateSessionId(sessionId);
    const turnIndex = Number(turnIndexRaw);
    const messages = this.store.loadMessages(sessionId, 0);
    if (!messages.length) {
      throw new HttpException(`会话不存在或无消息: ${sessionId}`, 404);
    }
    const maxTurn = Math.floor(messages.length / 2) - 1;
    if (!Number.isInteger(turnIndex)) {
      // 对齐 FastAPI 路径参数 int 校验失败 422
      throw new HttpException(`无效的轮次索引: ${turnIndexRaw}`, 422);
    }
    if (turnIndex < 0 || turnIndex > maxTurn) {
      // 范围越界对齐 api_server.py:6626 的 400
      throw new HttpException(
        `无效的轮次索引: ${turnIndex}（有效范围: 0-${maxTurn}）`,
        400,
      );
    }
    const deletedCount = this.store.deleteMessageRange(sessionId, turnIndex);
    this.store.decrementSessionMessageCount(sessionId, 2);
    const remaining = this.store.loadMessages(sessionId, 0).length;
    return {
      status: 'deleted',
      session_id: sessionId,
      turn_index: turnIndex,
      deleted_count: deletedCount,
      remaining_turns: Math.floor(remaining / 2),
    };
  }

  // ---------- 工具 ----------

  private validateSessionId(sessionId: string): void {
    if (!isValidSessionId(sessionId)) {
      throw new HttpException(`无效的会话 id: ${sessionId}`, 400);
    }
  }

  private parseLimit(raw?: string): number {
    const limit = this.parseInt(raw, 200);
    return limit <= 0 ? 200 : limit;
  }

  private parseInt(raw: string | undefined, def: number): number {
    if (raw === undefined || raw === '') return def;
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : def;
  }
}

// 类型导出供测试复用
export type { ChatMessage };
