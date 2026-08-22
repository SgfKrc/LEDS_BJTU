/**
 * 领域数据 hooks — 页面唯一的数据入口。
 *
 * 每个 hook 在 fixture 模式下返回本地样例，否则打真实接口。
 * 页面代码不需要知道数据来自哪里（§4.3）。
 */

import * as api from './api';
import {
  authCapabilityFixture,
  authSessionFixture,
  authSessionsFixture,
  capacityFixture,
  clusterConfigFixture,
  clusterInviteFixture,
  clusterStatusFixture,
  diffusionArtifactsFixture,
  diffusionAssetsFixture,
  diffusionCapabilitiesFixture,
  availableModelsFixture,
  currentModelFixture,
  deviceProfileFixture,
  fixturesEnabled,
  localModelAssetsFixture,
  localTailscaleStatusFixture,
  logFilesFixture,
  logStatsFixture,
  logsFixture,
  modelsFixture,
  managedUsersFixture,
  masterHealthFixture,
  nodesFixture,
  nodesLogAggregateFixture,
  nodesLogSummaryFixture,
  queueFixture,
  ragFixture,
  canVoteFixture,
  reviewTicketsFixture,
  sessionsFixture,
  speculativeCapabilityFixture,
  storageHealthFixture,
  statusFixture,
  tailscaleBindingsFixture,
  workflowsFixture,
} from './fixtures';
import { useResource, type ResourceResult } from './useResource';
import type {
  AuthCapabilityResponse,
  AuthSessionResponse,
  AuthSessionsResponse,
  ClusterNodesResponse,
  ClusterConfigResponse,
  ClusterInviteResponse,
  ClusterStatusResponse,
  DiffusionArtifactsResponse,
  DiffusionAssetsResponse,
  DiffusionCapabilitiesResponse,
  AvailableModelsResponse,
  CurrentModelResponse,
  DeviceProfileResponse,
  LocalModelAssetsResponse,
  LocalTailscaleStatusResponse,
  CanVoteResponse,
  LogFilesResponse,
  LogStatsResponse,
  NodeLogAggregateResponse,
  NodesLogSummaryResponse,
  ManagedUsersResponse,
  ModelsResponse,
  MasterHealthResponse,
  MyRoleResponse,
  PipelineCapacityResponse,
  QueueResponse,
  RagHealthResponse,
  RagSourcesResponse,
  RecentLogsResponse,
  ReviewTicketsResponse,
  SessionsResponse,
  SpeculativeCapabilityResponse,
  StorageHealthResponse,
  SystemStatusResponse,
  TailscaleBindingsResponse,
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

export function useClusterStatus(pollMs = 15_000): ResourceResult<ClusterStatusResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ClusterStatusResponse>(
    (signal) => useFixtures ? withFixtureDelay(clusterStatusFixture, signal) : api.fetchClusterStatus(signal),
    { pollMs },
  );
}

export function useClusterConfig(pollMs = 30_000): ResourceResult<ClusterConfigResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ClusterConfigResponse>(
    (signal) => useFixtures ? withFixtureDelay(clusterConfigFixture, signal) : api.fetchClusterConfig(signal),
    { pollMs },
  );
}

export function useClusterInvite(pollMs = 30_000, enabled = true): ResourceResult<ClusterInviteResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ClusterInviteResponse>(
    (signal) => useFixtures ? withFixtureDelay(clusterInviteFixture, signal) : api.fetchClusterInvite(signal),
    { pollMs, enabled },
  );
}

export function useMasterHealth(pollMs = 15_000, enabled = true): ResourceResult<MasterHealthResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<MasterHealthResponse>(
    (signal) => useFixtures ? withFixtureDelay(masterHealthFixture, signal) : api.fetchMasterHealth(signal),
    { pollMs, enabled },
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

export function useStorageHealth(pollMs = 30_000): ResourceResult<StorageHealthResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<StorageHealthResponse>(
    (signal) => useFixtures ? withFixtureDelay(storageHealthFixture, signal) : api.fetchStorageHealth(signal),
    { pollMs },
  );
}

export function useSpeculativeCapability(pollMs = 30_000): ResourceResult<SpeculativeCapabilityResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<SpeculativeCapabilityResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(speculativeCapabilityFixture, signal)
      : api.fetchSpeculativeCapability(signal),
    { pollMs },
  );
}

export function useRagSources(ownerScope = '', pollMs = 30_000): ResourceResult<RagSourcesResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<RagSourcesResponse>(
    (signal) => useFixtures
      ? withFixtureDelay({ storage: 'sqlite', sources: [
          { source_id: 'workspace-notes', display_name: 'Workspace notes', relative_ref: 'docs/notes', document_count: 8, chunk_count: 42 },
          { source_id: 'api-contracts', display_name: 'API contracts', relative_ref: 'docs/contracts', document_count: 5, chunk_count: 27 },
        ] }, signal)
      : api.fetchRagSources(ownerScope, signal),
    { pollMs, key: ownerScope },
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

export function useLogFiles(pollMs = 30_000): ResourceResult<LogFilesResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<LogFilesResponse>(
    (signal) => useFixtures ? withFixtureDelay(logFilesFixture, signal) : api.fetchLogFiles(signal),
    { pollMs },
  );
}

export function useLogStats(pollMs = 15_000): ResourceResult<LogStatsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<LogStatsResponse>(
    (signal) => useFixtures ? withFixtureDelay(logStatsFixture, signal) : api.fetchLogStats(signal),
    { pollMs },
  );
}

export function useNodesLogSummary(enabled = true, pollMs = 30_000): ResourceResult<NodesLogSummaryResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<NodesLogSummaryResponse>(
    (signal) => useFixtures ? withFixtureDelay(nodesLogSummaryFixture, signal) : api.fetchNodesLogSummary(signal),
    { enabled, pollMs },
  );
}

export function useNodesLogAggregate(
  enabled = true,
  pollMs = 20_000,
): ResourceResult<NodeLogAggregateResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<NodeLogAggregateResponse>(
    (signal) => useFixtures ? withFixtureDelay(nodesLogAggregateFixture, signal) : api.fetchNodesLogAggregate({ limit: 50 }, signal),
    { enabled, pollMs },
  );
}

export function useReviewTickets(pollMs = 20_000): ResourceResult<ReviewTicketsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ReviewTicketsResponse>(
    (signal) => useFixtures ? withFixtureDelay(reviewTicketsFixture, signal) : api.fetchReviewTickets('', signal),
    { pollMs },
  );
}

export function useCanVote(pollMs = 30_000): ResourceResult<CanVoteResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<CanVoteResponse>(
    (signal) => useFixtures ? withFixtureDelay(canVoteFixture, signal) : api.checkCanVote(signal),
    { pollMs },
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

export function useAuthCapability(pollMs = 60_000): ResourceResult<AuthCapabilityResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<AuthCapabilityResponse>(
    (signal) => useFixtures ? withFixtureDelay(authCapabilityFixture, signal) : api.fetchAuthCapability(signal),
    { pollMs },
  );
}

export function useAuthSession(enabled = true): ResourceResult<AuthSessionResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<AuthSessionResponse>(
    (signal) => useFixtures ? withFixtureDelay(authSessionFixture, signal) : api.fetchAuthSession(signal),
    { enabled },
  );
}

export function useAuthSessions(enabled = true, pollMs = 30_000): ResourceResult<AuthSessionsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<AuthSessionsResponse>(
    (signal) => useFixtures ? withFixtureDelay(authSessionsFixture, signal) : api.fetchAuthSessions('', signal),
    { enabled, pollMs },
  );
}

export function useManagedUsers(enabled = true, pollMs = 30_000): ResourceResult<ManagedUsersResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ManagedUsersResponse>(
    (signal) => useFixtures ? withFixtureDelay(managedUsersFixture, signal) : api.fetchManagedUsers(signal),
    { enabled, pollMs },
  );
}

export function useTailscaleBindings(enabled = true, pollMs = 30_000): ResourceResult<TailscaleBindingsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<TailscaleBindingsResponse>(
    (signal) => useFixtures ? withFixtureDelay(tailscaleBindingsFixture, signal) : api.fetchTailscaleBindings('', signal),
    { enabled, pollMs },
  );
}

export function useLocalTailscaleStatus(enabled = true): ResourceResult<LocalTailscaleStatusResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<LocalTailscaleStatusResponse>(
    (signal) => useFixtures ? withFixtureDelay(localTailscaleStatusFixture, signal) : api.fetchLocalTailscaleStatus(signal),
    { enabled },
  );
}

export function useDiffusionCapabilities(pollMs = 15_000): ResourceResult<DiffusionCapabilitiesResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<DiffusionCapabilitiesResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(diffusionCapabilitiesFixture, signal)
      : api.fetchDiffusionCapabilities(signal),
    { pollMs },
  );
}

export function useDiffusionArtifacts(pollMs = 30_000): ResourceResult<DiffusionArtifactsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<DiffusionArtifactsResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(diffusionArtifactsFixture, signal)
      : api.fetchDiffusionArtifacts(signal),
    { pollMs },
  );
}

export function useDiffusionAssets(pollMs = 30_000): ResourceResult<DiffusionAssetsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<DiffusionAssetsResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(diffusionAssetsFixture, signal)
      : api.fetchDiffusionAssets(signal),
    { pollMs },
  );
}

export function useModels(pollMs = 15_000): ResourceResult<ModelsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<ModelsResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(modelsFixture, signal)
      : api.fetchModels(signal),
    { pollMs },
  );
}

export function useAvailableModels(pollMs = 30_000): ResourceResult<AvailableModelsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<AvailableModelsResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(availableModelsFixture, signal)
      : api.fetchAvailableModels(signal),
    { pollMs },
  );
}

export function useCurrentModel(pollMs = 15_000): ResourceResult<CurrentModelResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<CurrentModelResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(currentModelFixture, signal)
      : api.fetchCurrentModel(signal),
    { pollMs },
  );
}

export function useLocalModelAssets(pollMs = 30_000): ResourceResult<LocalModelAssetsResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<LocalModelAssetsResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(localModelAssetsFixture, signal)
      : api.fetchLocalModelAssets(signal),
    { pollMs },
  );
}

export function useDeviceProfile(pollMs = 30_000): ResourceResult<DeviceProfileResponse> {
  const useFixtures = fixturesEnabled();
  return useResource<DeviceProfileResponse>(
    (signal) => useFixtures
      ? withFixtureDelay(deviceProfileFixture, signal)
      : api.fetchDeviceProfile(signal),
    { pollMs },
  );
}
