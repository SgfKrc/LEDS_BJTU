/**
 * 关键指标条 — 只显示 3-4 个指标，数字进入视口后递增一次（§5.3）。
 */

import { formatCompact, useCountUp, useInView } from '../motion/countUp';
import type { StatusTone } from '../data/types';

export interface Metric {
  /** 小号等宽标签。 */
  label: string;
  /** 数值；用于递增动画。 */
  value: number;
  /** 单位或后缀，例如 `%`、`GB`。 */
  suffix?: string;
  /** 小数位，默认 0。 */
  decimals?: number;
  /** 超过 1e4 时自动紧凑显示。 */
  compact?: boolean;
  /** 指标下方的一行说明。 */
  hint?: string;
  /** 色调：影响数字颜色，用于提示异常。 */
  tone?: StatusTone;
}

function MetricCell({ metric, active }: { metric: Metric; active: boolean }) {
  const decimals = metric.decimals ?? 0;
  const animated = useCountUp(metric.value, { decimals, active });
  const display = metric.compact
    ? formatCompact(animated)
    : animated.toFixed(decimals);

  return (
    <div className={`metric metric--${metric.tone ?? 'idle'}`}>
      <span className="metric__label mono-label">{metric.label}</span>
      <span className="metric__value num-display">
        {display}
        {metric.suffix ? <span className="metric__suffix">{metric.suffix}</span> : null}
      </span>
      {metric.hint ? <span className="metric__hint">{metric.hint}</span> : null}
    </div>
  );
}

interface MetricStripProps {
  metrics: Metric[];
  /** 可访问性标题；视觉上隐藏。 */
  caption?: string;
}

export function MetricStrip({ metrics, caption = '关键指标' }: MetricStripProps) {
  const [ref, inView] = useInView<HTMLDivElement>();
  // 超过 4 个会挤压首屏，按 §5.3 截断。
  const shown = metrics.slice(0, 4);

  return (
    <div className="metricstrip" ref={ref} data-reveal>
      <h2 className="sr-only">{caption}</h2>
      {shown.map((metric) => (
        <MetricCell key={metric.label} metric={metric} active={inView} />
      ))}
    </div>
  );
}
