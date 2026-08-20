/**
 * 数字递增 — 进入视口后播放一次（§5.4「数据刷新」一栏）。
 *
 * 不做无限循环：值变化时重新从旧值过渡到新值。
 * 减少动效时直接返回目标值。
 */

import { useEffect, useRef, useState } from 'react';
import { useReducedMotion } from './useReducedMotion';

interface CountUpOptions {
  /** 动画时长，默认 720ms。 */
  duration?: number;
  /** 保留小数位，默认 0。 */
  decimals?: number;
  /** false 时保持在起始值不播放（用于等待进入视口）。 */
  active?: boolean;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function useCountUp(target: number, options: CountUpOptions = {}): number {
  const { duration = 720, decimals = 0, active = true } = options;
  const reduced = useReducedMotion();
  const [value, setValue] = useState(() => (reduced || !active ? target : 0));
  const fromRef = useRef(0);
  const frameRef = useRef(0);

  useEffect(() => {
    if (!active) return;

    const safeTarget = Number.isFinite(target) ? target : 0;

    if (reduced) {
      setValue(safeTarget);
      fromRef.current = safeTarget;
      return;
    }

    const from = fromRef.current;
    if (from === safeTarget) {
      setValue(safeTarget);
      return;
    }

    const start = performance.now();
    const factor = Math.pow(10, decimals);

    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = easeOutCubic(progress);
      const next = from + (safeTarget - from) * eased;
      setValue(Math.round(next * factor) / factor);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = safeTarget;
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration, decimals, reduced, active]);

  return value;
}

/**
 * 元素是否已进入视口（只触发一次）。
 * 与 useCountUp 搭配，实现「进入视口后递增一次」。
 */
export function useInView<T extends HTMLElement>(): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.25 },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, inView];
}

/** 大数字的紧凑格式化（用于指标条，避免 8188 挤成一团）。 */
export function formatCompact(value: number, decimals = 1): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(decimals)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(decimals)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(decimals)}K`;
  return String(Math.round(value));
}

/** 字节 → 人类可读。 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** 秒级时间戳 → 相对时间中文描述。 */
export function formatRelative(epochSeconds: number): string {
  if (!Number.isFinite(epochSeconds) || epochSeconds <= 0) return '—';
  const deltaS = Date.now() / 1000 - epochSeconds;
  if (deltaS < 0) return '刚刚';
  if (deltaS < 60) return `${Math.round(deltaS)} 秒前`;
  if (deltaS < 3600) return `${Math.round(deltaS / 60)} 分钟前`;
  if (deltaS < 86400) return `${Math.round(deltaS / 3600)} 小时前`;
  return `${Math.round(deltaS / 86400)} 天前`;
}

/** 秒数 → 简短时长。 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}
