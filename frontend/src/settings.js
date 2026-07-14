export function normalizeExecutionSettings(settings) {
  const normalized = { ...settings };
  if (normalized.executionMode === 'task_graph') {
    normalized.streamingMode = 'full';
  }
  return normalized;
}
