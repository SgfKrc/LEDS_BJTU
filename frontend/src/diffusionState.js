export const DIFFUSION_TERMINAL_STATES = new Set([
  'completed',
  'failed',
  'cancelled',
]);

export const EDIT_STRENGTH_DEFAULT = 0.75;
export const IP_ADAPTER_SCALE_DEFAULT = 0.6;
export const IMAGE_GUIDANCE_SCALE_DEFAULT = 1.0;

export function canUseLocalDiffusion(role) {
  return Boolean(role?.is_master || role?.runtime_node_role === 'master');
}

export function supportsDedicatedEditProfile(engineConfig, requestedProfile = 'balanced') {
  if (!engineConfig) return requestedProfile === 'balanced';
  return (
    engineConfig.quantization === 'none'
    && engineConfig.qkv_fusion !== true
    && engineConfig.model_cpu_offload === true
  );
}

export function profileIdFromEngineConfig(engineConfig) {
  if (!engineConfig) return '';
  if (engineConfig.quantization !== 'none') {
    return engineConfig.qkv_fusion === true ? 'unet_8bit_qkv' : 'unet_8bit';
  }
  if (engineConfig.qkv_fusion === true) return 'qkv_fp16';
  return engineConfig.model_cpu_offload === true ? 'balanced' : 'resident_fp16';
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
      ipAdapterScale: IP_ADAPTER_SCALE_DEFAULT,
      imageGuidanceScale: IMAGE_GUIDANCE_SCALE_DEFAULT,
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
    ipAdapterScale: IP_ADAPTER_SCALE_DEFAULT,
    imageGuidanceScale: IMAGE_GUIDANCE_SCALE_DEFAULT,
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

export function buildReferenceRequest(form, sourceBlobId, adapterId) {
  const scale = Number(form?.ipAdapterScale ?? IP_ADAPTER_SCALE_DEFAULT);
  return {
    mode: 'reference',
    preset_id: form.presetId || null,
    source_blob_id: String(sourceBlobId || '').trim(),
    edit_adapter_id: String(adapterId || '').trim(),
    ip_adapter_scale: Number.isFinite(scale)
      ? Math.min(2, Math.max(0, scale))
      : IP_ADAPTER_SCALE_DEFAULT,
    prompt: String(form.prompt || '').trim(),
    negative_prompt: String(form.negativePrompt || '').trim(),
    seed: Number(form.seed),
    width: Number(form.width),
    height: Number(form.height),
    steps: Number(form.steps),
    guidance_scale: Number(form.guidanceScale),
    scheduler: form.scheduler || null,
  };
}

export function buildInpaintRequest(form, sourceBlobId, maskBlobId, pipelineId) {
  const strength = Number(form?.strength ?? EDIT_STRENGTH_DEFAULT);
  return {
    mode: 'inpaint',
    preset_id: form.presetId || null,
    source_blob_id: String(sourceBlobId || '').trim(),
    mask_blob_id: String(maskBlobId || '').trim(),
    edit_adapter_id: String(pipelineId || '').trim(),
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

export function buildInstructionRequest(form, sourceBlobId, pipelineId) {
  const imageGuidanceScale = Number(
    form?.imageGuidanceScale ?? IMAGE_GUIDANCE_SCALE_DEFAULT,
  );
  const instruction = String(form?.prompt || '').trim();
  return {
    mode: 'instruction',
    preset_id: form.presetId || null,
    source_blob_id: String(sourceBlobId || '').trim(),
    edit_adapter_id: String(pipelineId || '').trim(),
    prompt: instruction,
    instruction,
    negative_prompt: String(form.negativePrompt || '').trim(),
    seed: Number(form.seed),
    width: Number(form.width),
    height: Number(form.height),
    steps: Number(form.steps),
    guidance_scale: Number(form.guidanceScale),
    image_guidance_scale: Number.isFinite(imageGuidanceScale)
      ? Math.min(4, Math.max(0, imageGuidanceScale))
      : IMAGE_GUIDANCE_SCALE_DEFAULT,
    scheduler: form.scheduler || null,
  };
}
