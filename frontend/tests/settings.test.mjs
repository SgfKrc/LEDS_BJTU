import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_SETTINGS,
  createSettings,
  mergeSettingsSources,
  normalizeExecutionSettings,
} from '../src/settings.js';

test('default settings are created as a mutable copy', () => {
  const settings = createSettings();

  assert.deepEqual(settings, DEFAULT_SETTINGS);
  assert.notEqual(settings, DEFAULT_SETTINGS);
});

test('partial settings are completed without mutating the caller input', () => {
  const overrides = { temperature: 0.3 };

  const settings = createSettings(overrides);

  assert.equal(settings.temperature, 0.3);
  assert.equal(settings.maxNewTokens, 1024);
  assert.deepEqual(overrides, { temperature: 0.3 });
});

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

test('primary settings replace defaults when the browser has no local override', () => {
  const settings = mergeSettingsSources({
    maxNewTokens: 2048,
    temperature: 0.3,
  });

  assert.equal(settings.maxNewTokens, 2048);
  assert.equal(settings.temperature, 0.3);
  assert.equal(settings.topP, DEFAULT_SETTINGS.topP);
});

test('explicit browser keys override primary settings while missing keys still restore', () => {
  const settings = mergeSettingsSources(
    { maxNewTokens: 2048, temperature: 0.3, topP: 0.8 },
    { temperature: 0.9 },
  );

  assert.equal(settings.maxNewTokens, 2048);
  assert.equal(settings.temperature, 0.9);
  assert.equal(settings.topP, 0.8);
});

test('invalid source values are ignored and task graph invariants remain enforced', () => {
  assert.deepEqual(
    mergeSettingsSources(null, []),
    DEFAULT_SETTINGS,
  );
  assert.equal(
    mergeSettingsSources(
      { executionMode: 'task_graph', streamingMode: 'fast' },
    ).streamingMode,
    'full',
  );
});
