/**
 * 状态标签 — 颜色 + 文本双通道，不依赖颜色单独传达状态（§5.5）。
 */

import type { StatusTone } from '../data/types';
import { labelForState, toneForState } from './statusTone';

interface StatusBadgeProps {
  /** 后端原始状态字符串；会自动映射色调与中文标签。 */
  state?: string | null;
  /** 覆盖自动推导的色调。 */
  tone?: StatusTone;
  /** 覆盖自动推导的文本。 */
  label?: string;
  /** 运行中状态显示脉冲点（减少动效时自动静止）。 */
  pulse?: boolean;
  size?: 'sm' | 'md';
}

export function StatusBadge({
  state,
  tone,
  label,
  pulse = false,
  size = 'md',
}: StatusBadgeProps) {
  const resolvedTone = tone ?? toneForState(state);
  const resolvedLabel = label ?? labelForState(state);

  return (
    <span className={`badge badge--${resolvedTone} badge--${size}`}>
      <span className={`badge__dot${pulse ? ' badge__dot--pulse' : ''}`} aria-hidden="true" />
      {resolvedLabel}
    </span>
  );
}
