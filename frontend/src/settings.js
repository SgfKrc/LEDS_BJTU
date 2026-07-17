export function normalizeExecutionSettings(settings) {
  const normalized = { ...settings };
  if (!['local', 'auto'].includes(normalized.taskGraphRemoteMode)) {
    normalized.taskGraphRemoteMode = 'local';
  }
  if (normalized.executionMode === 'task_graph') {
    normalized.streamingMode = 'full';
  }
  return normalized;
}
