/**
 * 领域数据 hooks — 页面唯一的数据入口。
 *
 * 每个 hook 在 fixture 模式下返回本地样例，否则打真实接口。
 * 页面代码不需要知道数据来自哪里（§4.3）。
 */

import * as api from './api';
import {
  capacityFixture,
  fixturesEnabled,
  logsFixture,
  nodesFixture,
  queueFixture,
  ragFixture,
  sessionsFixture,
  statusFixture,
  workflowsFixture,
} from './fixtures';
import { useResource, type ResourceResult } from './useResource';
import type {
  ClusterNodesResponse,
  MyRoleResponse,
  PipelineCapacityResponse,
  QueueResponse,
  RagHealthResponse,
  RecentLogsResponse,
  SessionsResponse,
  SystemStatusResponse,
  WorkflowsResponse,
} from './types';

/** Python logging 级别数值，用于 fixture 模式下复现后端的级别过滤。 */
const LEVEL_NUMBERS: Record<string, number> = {
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  ERROR: 40,
  CRITICAL: 50,
};

/** fixture 也走一次微小延迟，便于验证 loading 骨架不会闪烁。 */
function withFixtureDelay<T>(value: T, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(value), 220);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    });
  });
}

export function useSystemStatus(pollMs = 15_000): ResourceResult<SystemStatusResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<SystemStatusResponse>(
    (signal) =>
      useFixtures ? withFixtureDelay(statusFixture, signal) : api.fetchSystemStatus(signal),
    { pollMs },
  );
}

export function useClusterNodes(pollMs = 15_000): ResourceResult<ClusterNodesResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ClusterNodesResponse>(
    (signal) =>
      useFixtures ? withFixtureDelay(nodesFixture, signal) : api.fetchClusterNodes(signal),
    { pollMs },
  );
}

export function useQueue(pollMs = 5_000, enabled = true): ResourceResult<QueueResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<QueueResponse>(
    (signal) => (useFixtures ? withFixtureDelay(queueFixture, signal) : api.fetchQueue(signal)),
    { pollMs, enabled },
  );
}

export function useWorkflows(limit = 20, pollMs = 8_000): ResourceResult<WorkflowsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<WorkflowsResponse>(
    (signal) =>
      useFixtures ? withFixtureDelay(workflowsFixture, signal) : api.fetchWorkflows(limit, signal),
    { pollMs, key: String(limit) },
  );
}

export function usePipelineCapacity(pollMs = 30_000): ResourceResult<PipelineCapacityResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<PipelineCapacityResponse>(
    (signal) =>
      useFixtures ? withFixtureDelay(capacityFixture, signal) : api.fetchPipelineCapacity(signal),
    { pollMs },
  );
}

export function useRagHealth(pollMs = 60_000): ResourceResult<RagHealthResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<RagHealthResponse>(
    (signal) => (useFixtures ? withFixtureDelay(ragFixture, signal) : api.fetchRagHealth(signal)),
    { pollMs },
  );
}

export function useRecentLogs(
  params: { limit?: number; level?: string } = {},
  pollMs = 10_000,
): ResourceResult<RecentLogsResponse> {
  const useFixtures = fixturesEnabled();
  const { limit, level } = params;
  return useResource<RecentLogsResponse>(
    (signal) => {
      if (!useFixtures) return api.fetchRecentLogs({ ...(limit ? { limit } : {}), ...(level ? { level } : {}) }, signal);
      // 与后端一致：level 是「最低级别」而不是精确匹配。
      const threshold = level ? (LEVEL_NUMBERS[level.toUpperCase()] ?? 0) : 0;
      const logs = logsFixture.logs.filter((l) => (l.levelno ?? 0) >= threshold);
      return withFixtureDelay(
        { ...logsFixture, logs, count: logs.length, matched: logs.length },
        signal,
      );
    },
    { pollMs, key: `${limit ?? ''}:${level ?? ''}` },
  );
}

export function useSessions(limit = 20, pollMs = 30_000): ResourceResult<SessionsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<SessionsResponse>(
    (signal) =>
      useFixtures ? withFixtureDelay(sessionsFixture, signal) : api.fetchSessions(limit, signal),
    { pollMs, key: String(limit) },
  );
}

export function useMyRole(): ResourceResult<MyRoleResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<MyRoleResponse>(
    (signal) =>
      useFixtures
        ? withFixtureDelay(
            {
              node_role: 'master',
              node_id: 'master',
              is_master: true,
              is_client: false,
              max_nodes: 3,
              run_mode: 'distributed',
            },
            signal,
          )
        : api.fetchMyRole(signal),
    {},
  );
}
