/**
 * 操作反馈 — 写操作（暂停队列、取消任务）的统一提示。
 *
 * 用 aria-live 区域播报，不遮挡主内容；4 秒后自动消失。
 * 模块级订阅，任何页面都可以 pushToast，不需要包 Provider。
 */

import { useEffect, useState } from 'react';
import type { StatusTone } from '../data/types';

export interface ToastItem {
  id: number;
  message: string;
  tone: Extract<StatusTone, 'ok' | 'danger' | 'info' | 'warn'>;
}

type Listener = (items: ToastItem[]) => void;

let items: ToastItem[] = [];
let seq = 0;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((fn) => fn(items));
}

export function pushToast(message: string, tone: ToastItem['tone'] = 'info'): void {
  seq += 1;
  const item: ToastItem = { id: seq, message, tone };
  items = [...items, item];
  emit();
  window.setTimeout(() => {
    items = items.filter((t) => t.id !== item.id);
    emit();
  }, 4000);
}

export function ToastHost() {
  const [current, setCurrent] = useState<ToastItem[]>(items);

  useEffect(() => {
    listeners.add(setCurrent);
    return () => {
      listeners.delete(setCurrent);
    };
  }, []);

  return (
    <div className="toasthost" aria-live="polite" aria-atomic="false">
      {current.map((t) => (
        <div className={`toast toast--${t.tone}`} key={t.id} role="status">
          {t.message}
        </div>
      ))}
    </div>
  );
}
