const STATE_LABELS = {
  pending_registration: '等待注册',
  created: '已创建',
  running: '执行中',
  result_ready: '结果待提交',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const REJECTION_LABELS = {
  winner_already_committed: '迟到结果已拒绝',
  winner_digest_mismatch: '冲突结果已拒绝',
  stale_lease_epoch: '旧租约结果已拒绝',
  attempt_epoch_mismatch: '错误租约结果已拒绝',
  lease_expired: '过期结果已拒绝',
  attempt_not_owned_by_stage: '未知执行结果已拒绝',
  provider_identity_mismatch: '错误节点结果已拒绝',
  invalid_result_schema: '无效结果已拒绝',
};

const RECOVERY_LABELS = {
  coordinator_restarted_during_execution: '重启中断，工作流已失败',
  coordinator_restarted_before_result_commit: '重启前结果未提交，工作流已失败',
};

function sumStages(stages, key) {
  return stages.reduce((total, stage) => total + Number(stage?.[key] || 0), 0);
}

function safeCount(value) {
  const count = Number(value || 0);
  return Number.isFinite(count) && count > 0 ? Math.floor(count) : 0;
}

export function normalizeTaskGraphWorkflow(workflow) {
  if (!workflow || typeof workflow !== 'object') return null;
  const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
  const observability = workflow.observability || {};
  const state = String(observability.state || workflow.state || 'unknown');
  const retryCount = safeCount(
    observability.retry_count ?? workflow.retry_count ?? sumStages(stages, 'retry_count'),
  );
  const rejectionCount = safeCount(
    observability.result_rejection_count
      ?? workflow.result_rejection_count
      ?? sumStages(stages, 'result_rejection_count'),
  );
  const lastRejectedStage = [...stages]
    .filter((stage) => stage?.last_result_rejection_reason)
    .sort((left, right) => Number(left.last_result_rejected_at || 0) - Number(right.last_result_rejected_at || 0))
    .at(-1);
  const rejectionReason = String(
    observability.last_result_rejection_reason
      || lastRejectedStage?.last_result_rejection_reason
      || '',
  );
  const recovered = Boolean(
    observability.recovered_after_restart || workflow.recovered_after_restart,
  );
  const recoveryReason = String(
    observability.recovery_reason || workflow.recovery_reason || '',
  );
  const providers = Array.isArray(observability.actual_providers)
    ? observability.actual_providers.filter(Boolean)
    : [];
  const tone = recovered || state === 'failed'
    ? 'error'
    : (state === 'result_ready' || retryCount > 0 || rejectionCount > 0 ? 'warning' : (state === 'completed' ? 'success' : 'active'));

  return {
    workflowId: String(workflow.workflow_id || ''),
    state,
    stateLabel: STATE_LABELS[state] || state,
    tone,
    resultReady: state === 'result_ready',
    terminal: ['completed', 'failed', 'cancelled'].includes(state),
    recovered,
    recoveryReason,
    recoveryLabel: RECOVERY_LABELS[recoveryReason] || (recovered ? '重启恢复失败' : ''),
    retryCount,
    rejectionCount,
    rejectionReason,
    rejectionLabel: REJECTION_LABELS[rejectionReason] || (rejectionCount > 0 ? '结果已拒绝' : ''),
    providers,
    journalAvailable: workflow.journal?.available !== false,
  };
}
