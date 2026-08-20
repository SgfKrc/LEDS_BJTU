/**
 * 资源加载 hook — 统一 loading / ready / error / 轮询 / 手动刷新。
 *
 * 页面只声明「取什么数据」，不重复写 AbortController 和状态机。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { describeError } from './api';
import type { LoadState } from './types';

interface UseResourceOptions {
  /** 轮询间隔（毫秒）；0 或省略表示只加载一次。 */
  pollMs?: number;
  /** false 时不发起请求（例如非 master 角色时跳过队列接口）。 */
  enabled?: boolean;
  /**
   * 请求参数指纹。fetcher 存在 ref 里不参与依赖，所以参数变化（例如日志级别筛选）
   * 必须通过这里告知，否则要等下一次轮询才生效。
   */
  key?: string;
}

export interface ResourceResult<T> {
  data: T | null;
  state: LoadState;
  error: string;
  /** 最近一次成功加载的时间戳（毫秒）。 */
  updatedAt: number;
  /** 手动刷新；不会把界面打回 loading 骨架。 */
  refresh: () => void;
  /** 后台刷新进行中（用于顶栏的细微指示，不遮挡内容）。 */
  refreshing: boolean;
}

export function useResource<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  options: UseResourceOptions = {},
): ResourceResult<T> {
  const { pollMs = 0, enabled = true, key = '' } = options;

  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<LoadState>(enabled ? 'loading' : 'idle');
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState(0);
  const [refreshing, setRefreshing] = useState(false);

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const mountedRef = useRef(true);
  const [nonce, setNonce] = useState(0);

  // 参数变了就丢掉旧数据：旧数据属于另一次查询，继续展示会误导
  // （例如筛 ERROR 时还留着 INFO 行）。渲染期同步处理，避免多渲染一帧脏数据。
  const keyRef = useRef(key);
  if (keyRef.current !== key) {
    keyRef.current = key;
    if (data !== null) setData(null);
    if (enabled) setState('loading');
  }

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const run = useCallback(
    async (signal: AbortSignal, isBackground: boolean) => {
      if (isBackground) setRefreshing(true);
      try {
        const next = await fetcherRef.current(signal);
        if (signal.aborted || !mountedRef.current) return;
        setData(next);
        setState('ready');
        setError('');
        setUpdatedAt(Date.now());
      } catch (err) {
        if (signal.aborted || !mountedRef.current) return;
        const message = describeError(err);
        if (!message) return; // AbortError
        setError(message);
        setState('error');
      } finally {
        if (mountedRef.current && !signal.aborted) setRefreshing(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!enabled) {
      setState('idle');
      return;
    }

    const controller = new AbortController();
    // 已有数据时的重新加载视为后台刷新，避免闪回骨架屏。
    void run(controller.signal, data !== null);

    let timer = 0;
    if (pollMs > 0) {
      timer = window.setInterval(() => {
        // 页面不可见时暂停轮询，减少无意义请求（§6 性能约束）。
        if (document.visibilityState === 'hidden') return;
        void run(controller.signal, true);
      }, pollMs);
    }

    return () => {
      controller.abort();
      if (timer) window.clearInterval(timer);
    };
    // data 故意不入依赖：仅用于首次判断是否后台刷新。
    // key 入依赖：fetcher 本身不在依赖里，参数变化只能靠它触发重新请求。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pollMs, nonce, key, run]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  return { data, state, error, updatedAt, refresh, refreshing };
}
