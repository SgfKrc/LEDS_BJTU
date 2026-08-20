/**
 * 后端状态字符串 → 语义色调映射。
 *
 * 集中在一处，避免每个组件各自判断 'online' / 'succeeded' / 'rejected'。
 */

import type { StatusTone } from '../data/types';

const OK_STATES = new Set(['online', 'ok', 'ready', 'succeeded', 'success', 'admitted', 'healthy', 'completed']);
const WARN_STATES = new Set(['degraded', 'pending', 'waiting', 'aged', 'rejected', 'paused', 'partial', 'stale']);
const DANGER_STATES = new Set(['offline', 'failed', 'error', 'cancelled', 'canceled', 'unavailable', 'timeout']);
const INFO_STATES = new Set(['running', 'active', 'connecting', 'dispatched', 'preparing', 'leased']);

export function toneForState(state: string | undefined | null): StatusTone {
  const key = String(state || '').trim().toLowerCase();
  if (!key) return 'idle';
  if (OK_STATES.has(key)) return 'ok';
  if (INFO_STATES.has(key)) return 'info';
  if (WARN_STATES.has(key)) return 'warn';
  if (DANGER_STATES.has(key)) return 'danger';
  return 'idle';
}

/** 英文状态 → 中文标签（正文不依赖英文术语，验收标准 §8）。 */
const STATE_LABELS: Record<string, string> = {
  online: '在线',
  offline: '离线',
  ok: '正常',
  ready: '就绪',
  succeeded: '已完成',
  success: '成功',
  failed: '失败',
  error: '错误',
  running: '运行中',
  active: '活动',
  pending: '等待中',
  waiting: '排队中',
  cancelled: '已取消',
  canceled: '已取消',
  paused: '已暂停',
  degraded: '降级',
  rejected: '未准入',
  admitted: '已准入',
  unavailable: '不可用',
  healthy: '健康',
  timeout: '超时',
  preparing: '准备中',
  dispatched: '已派发',
  leased: '已租约',
  stale: '数据陈旧',
  completed: '已完成',
  connecting: '连接中',
  master: '主节点',
  client: '从节点',
  distributed: '分布式',
  standalone: '单机',
  mlfq: '多级反馈队列',
  fifo: '先进先出',
};

export function labelForState(state: string | undefined | null): string {
  const raw = String(state || '').trim();
  if (!raw) return '未知';
  return STATE_LABELS[raw.toLowerCase()] || raw;
}

/** 日志级别 → 色调。 */
export function toneForLogLevel(level: string): StatusTone {
  switch (String(level || '').toUpperCase()) {
    case 'ERROR':
    case 'CRITICAL':
      return 'danger';
    case 'WARNING':
      return 'warn';
    case 'INFO':
      return 'info';
    default:
      return 'idle';
  }
}
