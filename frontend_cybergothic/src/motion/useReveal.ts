/**
 * 滚动显现 — 用单个 IntersectionObserver 统一处理 `[data-reveal]`。
 *
 * 参考 cmys-fight 的 useReveal 思路（§3.1）：避免每个区块重复写监听逻辑。
 * 只播放一次，元素显现后立即 unobserve。
 */

import { useEffect } from 'react';
import { useReducedMotion } from './useReducedMotion';

/**
 * 在页面根组件挂载一次即可。
 * @param deps 依赖变化时重新扫描（例如列表数据到达后新增了节点）
 */
export function useReveal(deps: unknown[] = []): void {
  const reduced = useReducedMotion();

  useEffect(() => {
    const targets = Array.from(
      document.querySelectorAll<HTMLElement>('[data-reveal]:not([data-revealed])'),
    );
    if (targets.length === 0) return;

    // 减少动效时直接标记为已显现，跳过位移动画。
    if (reduced || typeof IntersectionObserver === 'undefined') {
      targets.forEach((el) => {
        el.dataset.revealed = 'true';
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target as HTMLElement;
          el.dataset.revealed = 'true';
          observer.unobserve(el);
        });
      },
      { rootMargin: '0px 0px -12% 0px', threshold: 0.12 },
    );

    targets.forEach((el) => observer.observe(el));

    // 兜底：内容不可见是比动画丢失严重得多的故障。若某个区块因为
    // 布局塌陷、被祖先隐藏等原因始终不触发回调，2s 后强制显现。
    const fallback = window.setTimeout(() => {
      targets.forEach((el) => {
        if (el.dataset.revealed !== 'true' && el.getBoundingClientRect().top < window.innerHeight) {
          el.dataset.revealed = 'true';
          observer.unobserve(el);
        }
      });
    }, 2000);

    return () => {
      window.clearTimeout(fallback);
      observer.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, ...deps]);
}
