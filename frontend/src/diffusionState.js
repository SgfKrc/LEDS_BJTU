export const DIFFUSION_TERMINAL_STATES = new Set([
  'completed',
  'failed',
  'cancelled',
]);

export const EDIT_STRENGTH_DEFAULT = 0.75;

export function canUseLocalDiffusion(role) {
  return Boolean(role?.is_master || role?.runtime_node_role === 'master');
}

export function presetToForm(preset) {
  if (!preset) {
    return {
      presetId: '',
      prompt: '',
      negativePrompt: '',
      seed: 19950101,
      width: 512,
      height: 512,
      steps: 28,
      guidanceScale: 7.5,
      strength: EDIT_STRENGTH_DEFAULT,
      scheduler: '',
    };
  }
  return {
    presetId: preset.preset_id || '',
    prompt: preset.prompt || '',
    negativePrompt: preset.negative_prompt || '',
    seed: Number(preset.seeds?.[0] ?? 19950101),
    width: Number(preset.width ?? 512),
    height: Number(preset.height ?? 512),
    steps: Number(preset.steps ?? 28),
    guidanceScale: Number(preset.guidance_scale ?? 7.5),
    strength: EDIT_STRENGTH_DEFAULT,
    scheduler: preset.scheduler || '',
  };
}

export function normalizeDiffusionJob(job) {
  const state = String(job?.state || 'unknown');
  const rawStep = Number(job?.progress?.step || 0);
  const rawTotal = Number(job?.progress?.total || 0);
  const step = Number.isFinite(rawStep) ? Math.max(0, rawStep) : 0;
  const total = Number.isFinite(rawTotal) ? Math.max(0, rawTotal) : 0;
  return {
    ...job,
    state,
    terminal: DIFFUSION_TERMINAL_STATES.has(state),
    progress: { step, total },
    progressPercent: total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0,
  };
}

export function loadedArtifactId(capabilities) {
  return capabilities?.loaded_artifact?.artifact_id || '';
}

export function buildEditRequest(form, sourceBlobId) {
  const strength = Number(form?.strength ?? EDIT_STRENGTH_DEFAULT);
  return {
    mode: 'img2img',
    preset_id: form.presetId || null,
    source_blob_id: String(sourceBlobId || '').trim(),
    prompt: String(form.prompt || '').trim(),
    negative_prompt: String(form.negativePrompt || '').trim(),
    seed: Number(form.seed),
    width: Number(form.width),
    height: Number(form.height),
    steps: Number(form.steps),
    guidance_scale: Number(form.guidanceScale),
    strength: Number.isFinite(strength)
      ? Math.min(1, Math.max(0.05, strength))
      : EDIT_STRENGTH_DEFAULT,
    scheduler: form.scheduler || null,
  };
}
