/**
 * 空 / 错误 / 加载状态占位 — 占位文案保持直接，不留空白区域（验收标准 §8）。
 *
 * 插画使用低成本内联线条 SVG（静态，无动画），装饰层 aria-hidden。
 */

import type { ReactNode } from 'react';

type Kind = 'empty' | 'error' | 'loading' | 'denied';

interface EmptyStateProps {
  kind?: Kind;
  title: string;
  description?: string;
  /** 操作区，例如「重试」按钮。 */
  action?: ReactNode;
  /** 错误详情，如 request_id，等宽小字显示。 */
  detail?: string;
  compact?: boolean;
}

function LineArt({ kind }: { kind: Kind }) {
  // 单色线条图形，随主题色变化；不使用图片资源避免首屏布局跳动。
  if (kind === 'error' || kind === 'denied') {
    return (
      <svg viewBox="0 0 72 72" className="emptystate__art" aria-hidden="true">
        <path d="M36 10 L64 60 H8 Z" fill="none" strokeWidth="2.5" />
        <path d="M36 28 V44" strokeWidth="2.5" strokeLinecap="square" />
        <path d="M36 50 V52.5" strokeWidth="3" strokeLinecap="square" />
      </svg>
    );
  }
  if (kind === 'loading') {
    return (
      <svg viewBox="0 0 72 72" className="emptystate__art" aria-hidden="true">
        <rect x="10" y="20" width="52" height="8" fill="none" strokeWidth="2.5" />
        <rect x="10" y="34" width="36" height="8" fill="none" strokeWidth="2.5" />
        <rect x="10" y="48" width="44" height="8" fill="none" strokeWidth="2.5" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 72 72" className="emptystate__art" aria-hidden="true">
      <rect x="12" y="16" width="48" height="40" fill="none" strokeWidth="2.5" />
      <path d="M12 28 H60" strokeWidth="2.5" />
      <path d="M26 42 H46" strokeWidth="2.5" strokeLinecap="square" />
    </svg>
  );
}

export function EmptyState({
  kind = 'empty',
  title,
  description,
  action,
  detail,
  compact = false,
}: EmptyStateProps) {
  return (
    <div
      className={`emptystate emptystate--${kind}${compact ? ' emptystate--compact' : ''}`}
      role={kind === 'error' ? 'alert' : 'status'}
    >
      <LineArt kind={kind} />
      <div className="emptystate__body">
        <p className="emptystate__title">{title}</p>
        {description ? <p className="emptystate__desc">{description}</p> : null}
        {detail ? <p className="emptystate__detail">{detail}</p> : null}
        {action ? <div className="emptystate__action">{action}</div> : null}
      </div>
    </div>
  );
}

/** 列表/表格加载中的骨架行。 */
export function SkeletonRows({ rows = 4, columns = 1 }: { rows?: number; columns?: number }) {
  return (
    <div className="skeleton" aria-hidden="true">
      {Array.from({ length: rows }).map((_, r) => (
        <div className="skeleton__row" key={r}>
          {Array.from({ length: columns }).map((__, c) => (
            <span className="skeleton__cell" key={c} />
          ))}
        </div>
      ))}
    </div>
  );
}
