/**
 * 可拖拽分屏 — 左控制台 / 右对话，比例由用户自己定。
 *
 * 交互要点：
 *   - 指针拖拽用 setPointerCapture，鼠标移出窗口也不会丢事件；
 *   - 分隔条是真 separator role，方向键可调（§5.5 键盘可达）；
 *   - 比例写进 localStorage，刷新后保持；
 *   - 窄屏（≤860px）由 CSS 改成上下堆叠，此时分隔条隐藏且不可聚焦。
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { GripVertical } from 'lucide-react';

const STORAGE_KEY = 'qlh_cg_split_ratio';
/** 与 tokens.css 的 --split-min 保持一致；JS 侧也要夹住，拖拽才不会越界。 */
const MIN = 22;
const MAX = 78;
const DEFAULT_RATIO = 42; // 接近 5:7，文档 §4.4 给的桌面比例

/** 常用比例预设，双击分隔条时循环切换。 */
const PRESETS = [42, 50, 58];

function clamp(value: number): number {
  return Math.min(MAX, Math.max(MIN, value));
}

function readStored(): number {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_RATIO;
    const parsed = Number.parseFloat(raw);
    return Number.isFinite(parsed) ? clamp(parsed) : DEFAULT_RATIO;
  } catch {
    return DEFAULT_RATIO;
  }
}

interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  /** 无障碍名称，读屏时说明这条分隔条在调什么。 */
  label?: string;
  leftLabel?: string;
  rightLabel?: string;
}

export function SplitPane({
  left,
  right,
  label = '调整控制台与对话的宽度比例',
  leftLabel = '控制台',
  rightLabel = '对话',
}: SplitPaneProps) {
  const [ratio, setRatio] = useState<number>(() => readStored());
  const [dragging, setDragging] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const readoutId = useId();

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(Math.round(ratio)));
    } catch {
      /* 隐私模式下写不进去，比例只在本次会话有效，不影响可用性 */
    }
  }, [ratio]);

  /** 由指针横坐标反算比例。 */
  const ratioFromClientX = useCallback((clientX: number): number => {
    const el = rootRef.current;
    if (!el) return DEFAULT_RATIO;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0) return DEFAULT_RATIO;
    return clamp(((clientX - rect.left) / rect.width) * 100);
  }, []);

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    // 只响应主键；右键菜单和多指手势不该改布局。
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  }, []);

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging) return;
      event.preventDefault();
      setRatio(ratioFromClientX(event.clientX));
    },
    [dragging, ratioFromClientX],
  );

  const endDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDragging(false);
  }, []);

  const onKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 8 : 2;
    switch (event.key) {
      case 'ArrowLeft':
        setRatio((r) => clamp(r - step));
        break;
      case 'ArrowRight':
        setRatio((r) => clamp(r + step));
        break;
      case 'Home':
        setRatio(MIN);
        break;
      case 'End':
        setRatio(MAX);
        break;
      case 'Enter':
      case ' ':
        // 在预设之间循环，键盘用户不必按十几次方向键。
        setRatio((r) => {
          const next = PRESETS.find((p) => p > r + 1);
          return next ?? PRESETS[0] ?? DEFAULT_RATIO;
        });
        break;
      default:
        return;
    }
    event.preventDefault();
  }, []);

  const rounded = Math.round(ratio);

  return (
    <div
      className="split"
      ref={rootRef}
      data-dragging={dragging ? 'true' : undefined}
      style={{ ['--split-ratio' as string]: `${ratio}%` }}
    >
      <section className="split__pane split__pane--left" aria-label={leftLabel}>
        {left}
      </section>

      <div
        className="split__handle"
        role="separator"
        tabIndex={0}
        aria-orientation="vertical"
        aria-label={label}
        aria-valuenow={rounded}
        aria-valuemin={MIN}
        aria-valuemax={MAX}
        aria-valuetext={`控制台 ${rounded}%，对话 ${100 - rounded}%`}
        aria-describedby={readoutId}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => setRatio(DEFAULT_RATIO)}
        onKeyDown={onKeyDown}
      >
        <span className="split__grip" aria-hidden="true">
          <GripVertical size={14} strokeWidth={2} />
        </span>
        <span className="split__rail" aria-hidden="true" />
        <span className="split__readout mono-label" id={readoutId}>
          {rounded} / {100 - rounded}
        </span>
      </div>

      <section className="split__pane split__pane--right" aria-label={rightLabel}>
        {right}
      </section>
    </div>
  );
}
