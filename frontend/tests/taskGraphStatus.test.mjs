import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeTaskGraphWorkflow } from '../src/taskGraphStatus.js';
import { fetchTaskGraphStatus } from '../src/api/client.js';

test('task graph capability query is scoped to the active session', async () => {
  const originalFetch = globalThis.fetch;
  const requested = [];
  globalThis.fetch = async (url) => {
    requested.push(url);
    return {
      ok: true,
      status: 200,
      headers: { get: () => null },
      text: async () => '{"workflows":[]}',
    };
  };
  try {
    await fetchTaskGraphStatus('session/a');
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(requested, [
    '/api/workflows?limit=1&session_id=session%2Fa',
  ]);
});

test('result_ready is distinct from completed', () => {
  const status = normalizeTaskGraphWorkflow({
    workflow_id: 'wf_resultready1',
    state: 'result_ready',
    observability: { state: 'result_ready', result_ready: true },
    journal: { available: true },
  });

  assert.equal(status.stateLabel, '结果待提交');
  assert.equal(status.resultReady, true);
  assert.equal(status.terminal, false);
  assert.equal(status.tone, 'warning');
});

test('restart recovery failure has an explicit reason', () => {
  const status = normalizeTaskGraphWorkflow({
    workflow_id: 'wf_recovered1',
    state: 'failed',
    recovered_after_restart: true,
    recovery_reason: 'coordinator_restarted_before_result_commit',
  });

  assert.equal(status.recovered, true);
  assert.equal(status.recoveryLabel, '重启前结果未提交，工作流已失败');
  assert.equal(status.tone, 'error');
});

test('retry and late result rejection remain visible after completion', () => {
  const status = normalizeTaskGraphWorkflow({
    workflow_id: 'wf_retrydone1',
    state: 'completed',
    observability: {
      state: 'completed',
      retry_count: 2,
      result_rejection_count: 1,
      last_result_rejection_reason: 'winner_already_committed',
      actual_providers: ['worker-a', 'worker-b'],
    },
  });

  assert.equal(status.retryCount, 2);
  assert.equal(status.rejectionCount, 1);
  assert.equal(status.rejectionLabel, '迟到结果已拒绝');
  assert.deepEqual(status.providers, ['worker-a', 'worker-b']);
  assert.equal(status.tone, 'warning');
});

test('legacy snapshots derive counters from stages', () => {
  const status = normalizeTaskGraphWorkflow({
    workflow_id: 'wf_legacy001',
    state: 'failed',
    stages: [{
      retry_count: 1,
      result_rejection_count: 1,
      last_result_rejection_reason: 'stale_lease_epoch',
      last_result_rejected_at: 12,
    }],
  });

  assert.equal(status.retryCount, 1);
  assert.equal(status.rejectionCount, 1);
  assert.equal(status.rejectionLabel, '旧租约结果已拒绝');
});
