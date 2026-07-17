import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeExecutionSettings } from '../src/settings.js';

test('task graph mode always uses full streaming semantics', () => {
  assert.deepEqual(
    normalizeExecutionSettings({
      executionMode: 'task_graph',
      streamingMode: 'fast',
      temperature: 0.7,
    }),
    {
      executionMode: 'task_graph',
      streamingMode: 'full',
      temperature: 0.7,
      taskGraphRemoteMode: 'local',
    },
  );
});

test('standard execution preserves the selected streaming mode', () => {
  assert.equal(
    normalizeExecutionSettings({
      executionMode: 'auto',
      streamingMode: 'fast',
    }).streamingMode,
    'fast',
  );
});

test('invalid task graph remote policy falls back to local', () => {
  assert.equal(
    normalizeExecutionSettings({ taskGraphRemoteMode: 'unexpected' })
      .taskGraphRemoteMode,
    'local',
  );
});
