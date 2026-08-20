/**
 * 全局刷新总线 — 顶栏刷新按钮触发当前挂载页面的重新加载。
 *
 * 用模块级订阅而不是 Context，避免 AppShell ↔ routes ↔ pages 形成循环导入。
 */

import { useEffect } from 'react';

type RefreshFn = () => void;

const subscribers = new Set<RefreshFn>();

/** 页面注册自己的刷新逻辑；卸载时自动注销。 */
export function useRegisterRefresh(fn: RefreshFn): void {
  useEffect(() => {
    subscribers.add(fn);
    return () => {
      subscribers.delete(fn);
    };
  }, [fn]);
}

/** 顶栏调用：广播给所有已挂载页面。 */
export function triggerRefresh(): void {
  subscribers.forEach((fn) => {
    try {
      fn();
    } catch {
      // 单个页面刷新失败不影响其他订阅者。
    }
  });
}
