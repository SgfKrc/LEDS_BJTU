/**
 * 活动时间线 — 语义化列表 + aria-live 更新区（§5.5）。
 */

import type { StatusTone } from '../data/types';
import { EmptyState, SkeletonRows } from './EmptyState';
import { CommandButton } from './CommandButton';
import { StatusBadge } from './StatusBadge';
import type { LoadState } from '../data/types';

export interface TimelineItem {
  id: string;
  /** 左侧时间文本。 */
  time: string;
  /** 标题行。 */
  title: string;
  /** 详细描述；等宽字体显示原始日志内容。 */
  detail?: string;
  tone?: StatusTone;
  /** 状态标签文本，可选。 */
  badge?: string;
  /** 来源标记，例如 node_id。 */
  source?: string;
}

interface ActivityTimelineProps {
  items: TimelineItem[];
  state: LoadState;
  error?: string;
  onRetry?: () => void;
  emptyTitle?: string;
  /** 实时追加时用 aria-live 播报最新一条。 */
  live?: boolean;
  /** 提供时每条记录可点开详情（真实 button，键盘可达）。 */
  onSelect?: (id: string) => void;
}

export function ActivityTimeline({
  items,
  state,
  error = '',
  onRetry,
  emptyTitle = '暂无活动记录',
  live = false,
  onSelect,
}: ActivityTimelineProps) {
  if (state === 'error') {
    return (
      <EmptyState
        kind="error"
        title="活动记录加载失败"
        description="日志接口可能需要管理令牌，或后端未启动。"
        detail={error}
        {...(onRetry
          ? { action: <CommandButton variant="ghost" size="sm" onClick={onRetry}>重试</CommandButton> }
          : {})}
      />
    );
  }

  if (state === 'loading' && items.length === 0) {
    return <SkeletonRows rows={5} columns={2} />;
  }

  if (state !== 'loading' && items.length === 0) {
    return (
      <EmptyState
        kind="empty"
        title={emptyTitle}
        description="系统产生新事件后会出现在这里。"
        {...(onRetry
          ? { action: <CommandButton variant="ghost" size="sm" onClick={onRetry}>刷新</CommandButton> }
          : {})}
      />
    );
  }

  return (
    <ol
      className="timeline"
      {...(live ? { 'aria-live': 'polite' as const, 'aria-relevant': 'additions' as const } : {})}
    >
      {items.map((item) => {
        const body = (
          <>
            <div className="timeline__head">
              <time className="timeline__time mono-label">{item.time}</time>
              {item.badge ? (
                <StatusBadge label={item.badge} tone={item.tone ?? 'idle'} size="sm" />
              ) : null}
              {item.source ? <span className="timeline__source">{item.source}</span> : null}
            </div>
            <p className="timeline__title">{item.title}</p>
            {item.detail ? <p className="timeline__detail">{item.detail}</p> : null}
          </>
        );
        return (
          <li className={`timeline__item timeline__item--${item.tone ?? 'idle'}`} key={item.id}>
            <span className="timeline__marker" aria-hidden="true" />
            {onSelect ? (
              <button
                type="button"
                className="timeline__body timeline__body--action"
                onClick={() => onSelect(item.id)}
              >
                {body}
              </button>
            ) : (
              <div className="timeline__body">{body}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
