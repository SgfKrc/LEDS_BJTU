/**
 * 动效偏好 — 合并系统 `prefers-reduced-motion` 与用户在设置页的手动开关。
 *
 * 任何循环动画、视差、Canvas 都必须先查询这里（§5.4 强制要求）。
 */

import { useEffect, useState } from 'react';

const MOTION_PREF_KEY = 'qlh_cg_motion';

export type MotionPreference = 'system' | 'full' | 'reduced';

const listeners = new Set<() => void>();

function readStoredPreference(): MotionPreference {
  try {
    const raw = window.localStorage?.getItem(MOTION_PREF_KEY);
    if (raw === 'full' || raw === 'reduced') return raw;
    return 'system';
  } catch {
    return 'system';
  }
}

let currentPreference: MotionPreference =
  typeof window === 'undefined' ? 'system' : readStoredPreference();

export function getMotionPreference(): MotionPreference {
  return currentPreference;
}

export function setMotionPreference(next: MotionPreference): void {
  currentPreference = next;
  try {
    if (next === 'system') window.localStorage?.removeItem(MOTION_PREF_KEY);
    else window.localStorage?.setItem(MOTION_PREF_KEY, next);
  } catch {
    // 存储不可用时仅本次会话生效。
  }
  applyDocumentFlag();
  listeners.forEach((fn) => fn());
}

function systemPrefersReduced(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function resolveReduced(): boolean {
  if (currentPreference === 'reduced') return true;
  if (currentPreference === 'full') return false;
  return systemPrefersReduced();
}

/** 把最终结果写到 <html data-reduced-motion>，供纯 CSS 动画一并关闭。 */
function applyDocumentFlag(): void {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.reducedMotion = resolveReduced() ? 'true' : 'false';
}

if (typeof window !== 'undefined') {
  applyDocumentFlag();
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => {
      applyDocumentFlag();
      listeners.forEach((fn) => fn());
    };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else mq.addListener(onChange);
  }
}

/** 返回 true 表示应当关闭非必要动效。 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(resolveReduced);

  useEffect(() => {
    const sync = () => setReduced(resolveReduced());
    listeners.add(sync);
    sync();
    return () => {
      listeners.delete(sync);
    };
  }, []);

  return reduced;
}

/** 供设置页读取/切换三态偏好。 */
export function useMotionPreference(): [MotionPreference, (next: MotionPreference) => void] {
  const [pref, setPref] = useState<MotionPreference>(getMotionPreference);

  useEffect(() => {
    const sync = () => setPref(getMotionPreference());
    listeners.add(sync);
    return () => {
      listeners.delete(sync);
    };
  }, []);

  return [pref, setMotionPreference];
}
