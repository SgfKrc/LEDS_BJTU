export const DEFAULT_SETTINGS = Object.freeze({
  saveHistory: true,
  maxNewTokens: 1024,
  temperature: 0.7,
  topP: 0.9,
  distributedInference: false,
  executionMode: 'auto',
  taskGraphRemoteMode: 'local',
  cloudSync: true,
  showThinking: false,
  streamingMode: 'full',
});

function asSettingsObject(settings) {
  return settings && typeof settings === 'object' && !Array.isArray(settings)
    ? settings
    : {};
}

export function normalizeExecutionSettings(settings) {
  const normalized = { ...asSettingsObject(settings) };
  if (!['local', 'auto'].includes(normalized.taskGraphRemoteMode)) {
    normalized.taskGraphRemoteMode = 'local';
  }
  if (normalized.executionMode === 'task_graph') {
    normalized.streamingMode = 'full';
  }
  return normalized;
}

export function createSettings(overrides = {}) {
  return normalizeExecutionSettings({
    ...DEFAULT_SETTINGS,
    ...asSettingsObject(overrides),
  });
}

export function mergeSettingsSources(primarySettings = {}, localOverrides = null) {
  return createSettings({
    ...asSettingsObject(primarySettings),
    ...asSettingsObject(localOverrides),
  });
}
