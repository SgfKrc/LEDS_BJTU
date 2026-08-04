/**
 * 内存日志环形缓冲 — 对齐 api_server.py MemoryLogHandler/_log_buffer
 * (微服务架构改造计划 阶段 3.2 日志域)
 *
 * 语义对齐：
 *  - 容量 5000（_LOG_BUFFER_MAXLEN），满时丢弃最旧条目
 *  - 条目字段：timestamp, level, levelno, name, message, filename,
 *    lineno, funcName, request_id, node_id, device_ip, thread, seq(+exc_text)
 *    —— 与 MemoryLogHandler.emit 的字段集一致（TUI 读 level/name/message/time|timestamp）
 *  - 过滤（_filter_recent_logs 语义）：level 名可解析时按 levelno >= 过滤，
 *    否则精确匹配 level 字符串；name 子串；node_id/request_id 精确
 *
 * 注意：control-svc 是独立进程，本缓冲记录的是 control-svc 自身的日志
 * （服务内部 append）；Python api_server 的内存缓冲由它自己的 handler 填充。
 * 并行共存期间两者各自独立；清理阶段切换后由 control-svc 统一承载。
 */
export interface LogEntry {
  timestamp: string;
  level: string;
  levelno: number;
  name: string;
  message: string;
  filename: string;
  lineno: number;
  funcName: string;
  request_id: string;
  node_id: string;
  device_ip: string;
  thread: string;
  seq: number;
  exc_text?: string;
}

export const LOG_BUFFER_CAPACITY = 5000;

// Python logging._nameToLevel 的常用映射
const LEVEL_NOS: Record<string, number> = {
  CRITICAL: 50,
  ERROR: 40,
  WARNING: 30,
  INFO: 20,
  DEBUG: 10,
  NOTSET: 0,
};

function nowStr(): string {
  // 对齐 Python 侧日志时间格式（本地时间）
  const d = new Date();
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export class LogBuffer {
  private entries: LogEntry[] = [];
  private totalSeen = 0;
  private seq = 0;

  constructor(private readonly capacity: number = LOG_BUFFER_CAPACITY) {}

  /** 追加一条日志（对齐 MemoryLogHandler.emit 的入队语义） */
  append(
    entry: Pick<LogEntry, 'level' | 'levelno' | 'message'> &
      Partial<Omit<LogEntry, 'level' | 'levelno' | 'message'>>,
  ): LogEntry {
    const full: LogEntry = {
      timestamp: entry.timestamp ?? nowStr(),
      level: entry.level,
      levelno: entry.levelno,
      name: entry.name ?? '',
      message: entry.message,
      filename: entry.filename ?? '',
      lineno: entry.lineno ?? 0,
      funcName: entry.funcName ?? '',
      request_id: entry.request_id ?? '',
      node_id: entry.node_id ?? '',
      device_ip: entry.device_ip ?? '',
      thread: entry.thread ?? '',
      seq: this.seq++,
      exc_text: entry.exc_text,
    };
    this.entries.push(full);
    this.totalSeen += 1;
    if (this.entries.length > this.capacity) {
      this.entries.shift();
    }
    return full;
  }

  /** 快照（对齐 _snapshot_recent_logs：返回副本 + total_seen） */
  snapshot(): { entries: LogEntry[]; totalSeen: number } {
    return { entries: this.entries.map((e) => ({ ...e })), totalSeen: this.totalSeen };
  }

  /** 过滤（对齐 _filter_recent_logs + get_recent_logs 的组装） */
  query(
    limit: number,
    level = '',
    name = '',
    nodeId = '',
    requestId = '',
  ): {
    logs: LogEntry[];
    count: number;
    matched: number;
    limit: number;
    buffer_size: number;
    buffer_capacity: number;
    total_seen: number;
    truncated: boolean;
    filters: { level: string | null; name: string | null; node_id: string | null; request_id: string | null };
  } {
    const { entries, totalSeen } = this.snapshot();
    let filtered = entries;
    const lv = level.trim().toUpperCase();
    if (lv) {
      const levelno = LEVEL_NOS[lv];
      if (typeof levelno === 'number') {
        filtered = filtered.filter((e) => e.levelno >= levelno);
      } else {
        filtered = filtered.filter((e) => e.level.toUpperCase() === lv);
      }
    }
    const nm = name.trim();
    if (nm) filtered = filtered.filter((e) => e.name.includes(nm));
    const nid = nodeId.trim();
    if (nid) filtered = filtered.filter((e) => e.node_id === nid);
    const rid = requestId.trim();
    if (rid) filtered = filtered.filter((e) => e.request_id === rid);

    const result = filtered.slice(-limit);
    return {
      logs: result,
      count: result.length,
      matched: filtered.length,
      limit,
      buffer_size: entries.length,
      buffer_capacity: this.capacity,
      total_seen: totalSeen,
      truncated: filtered.length > limit,
      filters: {
        level: level.trim() || null,
        name: name.trim() || null,
        node_id: nodeId.trim() || null,
        request_id: requestId.trim() || null,
      },
    };
  }

  /** 统计（对齐 get_log_stats 的缓冲部分 + Counter 转 dict） */
  stats(): {
    buffer_size: number;
    buffer_capacity: number;
    buffer_total_seen: number;
    buffer_dropped_estimate: number;
    levels: Record<string, number>;
    loggers: Record<string, number>;
    nodes: Record<string, number>;
  } {
    const { entries, totalSeen } = this.snapshot();
    const levels: Record<string, number> = {};
    const loggers: Record<string, number> = {};
    const nodes: Record<string, number> = {};
    for (const e of entries) {
      levels[e.level || 'UNKNOWN'] = (levels[e.level || 'UNKNOWN'] || 0) + 1;
      loggers[e.name || 'unknown'] = (loggers[e.name || 'unknown'] || 0) + 1;
      nodes[e.node_id || 'unknown'] = (nodes[e.node_id || 'unknown'] || 0) + 1;
    }
    // 对齐 Counter.most_common(20)：按计数降序取前 20
    const topLoggers: Record<string, number> = {};
    for (const [k, v] of Object.entries(loggers).sort((a, b) => b[1] - a[1]).slice(0, 20)) {
      topLoggers[k] = v;
    }
    return {
      buffer_size: entries.length,
      buffer_capacity: this.capacity,
      buffer_total_seen: totalSeen,
      buffer_dropped_estimate: Math.max(0, totalSeen - entries.length),
      levels,
      loggers: topLoggers,
      nodes,
    };
  }

  size(): number {
    return this.entries.length;
  }
}
