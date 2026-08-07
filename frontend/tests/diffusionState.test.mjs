import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildEditRequest,
  buildInpaintRequest,
  buildInstructionRequest,
  buildReferenceRequest,
  canUseLocalDiffusion,
  loadedArtifactId,
  normalizeDiffusionJob,
  presetToForm,
  profileIdFromEngineConfig,
  supportsDedicatedEditProfile,
} from '../src/diffusionState.js';
import {
  fetchDiffusionBlob,
  fetchDiffusionJob,
  fetchEmailConfig,
  updateEmailConfig,
  editDiffusionImage,
  generateDiffusionImage,
  registerDiffusionArtifact,
  uploadDiffusionBlob,
  unloadModel,
} from '../src/api/client.js';

test('diffusion job progress is bounded and terminal states are explicit', () => {
  const running = normalizeDiffusionJob({
    job_id: 'sdjob_running',
    state: 'running',
    progress: { step: 12, total: 10 },
  });
  const completed = normalizeDiffusionJob({
    job_id: 'sdjob_done',
    state: 'completed',
    progress: { step: 10, total: 10 },
  });

  assert.equal(running.terminal, false);
  assert.equal(running.progressPercent, 100);
  assert.equal(completed.terminal, true);
  assert.equal(completed.progressPercent, 100);
});

test('missing or malformed progress cannot produce NaN in the progress bar', () => {
  const snapshot = normalizeDiffusionJob({
    state: 'queued',
    progress: { step: 'invalid', total: Number.NaN },
  });

  assert.deepEqual(snapshot.progress, { step: 0, total: 0 });
  assert.equal(snapshot.progressPercent, 0);
  assert.equal(Number.isNaN(snapshot.progressPercent), false);
});

test('preset form keeps deterministic seed and numeric controls', () => {
  const form = presetToForm({
    preset_id: 'sd15_original_v1',
    prompt: 'lake',
    negative_prompt: 'blur',
    seeds: ['17'],
    width: '512',
    height: '512',
    steps: '28',
    guidance_scale: '7.5',
    scheduler: 'PNDMScheduler',
  });

  assert.equal(form.presetId, 'sd15_original_v1');
  assert.equal(form.seed, 17);
  assert.equal(form.width, 512);
  assert.equal(form.guidanceScale, 7.5);
  assert.equal(form.scheduler, 'PNDMScheduler');
});

test('only master nodes can use the local diffusion UI', () => {
  assert.equal(canUseLocalDiffusion({ is_master: true }), true);
  assert.equal(canUseLocalDiffusion({
    node_role: 'unknown',
    runtime_node_role: 'master',
    is_master: false,
    is_provisional: true,
  }), true);
  assert.equal(canUseLocalDiffusion({ is_master: false, is_client: true }), false);
  assert.equal(loadedArtifactId({ loaded_artifact: { artifact_id: 'sd-local' } }), 'sd-local');
  assert.equal(loadedArtifactId({ loaded_artifact: null }), '');
});

test('dedicated edit support follows the loaded engine instead of the pending selector', () => {
  assert.equal(supportsDedicatedEditProfile(null, 'balanced'), true);
  assert.equal(supportsDedicatedEditProfile(null, 'resident_fp16'), false);
  assert.equal(supportsDedicatedEditProfile({
    quantization: 'none',
    qkv_fusion: false,
    model_cpu_offload: false,
  }, 'balanced'), false);
  assert.equal(supportsDedicatedEditProfile({
    quantization: 'none',
    qkv_fusion: false,
    model_cpu_offload: true,
  }, 'resident_fp16'), true);
});

test('loaded engine configuration maps back to the visible profile', () => {
  assert.equal(profileIdFromEngineConfig(null), '');
  assert.equal(profileIdFromEngineConfig({
    quantization: 'none',
    qkv_fusion: false,
    model_cpu_offload: true,
  }), 'balanced');
  assert.equal(profileIdFromEngineConfig({
    quantization: 'none',
    qkv_fusion: false,
    model_cpu_offload: false,
  }), 'resident_fp16');
  assert.equal(profileIdFromEngineConfig({
    quantization: 'none',
    qkv_fusion: true,
    model_cpu_offload: true,
  }), 'qkv_fp16');
  assert.equal(profileIdFromEngineConfig({
    quantization: 'bitsandbytes_8bit_unet',
    qkv_fusion: false,
  }), 'unet_8bit');
  assert.equal(profileIdFromEngineConfig({
    quantization: 'bitsandbytes_8bit_unet',
    qkv_fusion: true,
  }), 'unet_8bit_qkv');
});

test('edit request carries the source blob and clamps strength into the valid range', () => {
  const form = {
    presetId: 'sd15_original_v1',
    prompt: '  make it sunset  ',
    negativePrompt: 'blur',
    seed: 17,
    width: 512,
    height: 512,
    steps: 28,
    guidanceScale: 7.5,
    strength: '1.8',
    scheduler: 'PNDMScheduler',
  };

  const request = buildEditRequest(form, 'img_source');
  assert.equal(request.mode, 'img2img');
  assert.equal(request.source_blob_id, 'img_source');
  assert.equal(request.prompt, 'make it sunset');
  assert.equal(request.strength, 1);

  const low = buildEditRequest({ ...form, strength: '0.01' }, 'img_source');
  assert.equal(low.strength, 0.05);

  const invalid = buildEditRequest({ ...form, strength: 'oops' }, 'img_source');
  assert.equal(invalid.strength, 0.75);

  const missing = buildEditRequest(form, '');
  assert.equal(missing.source_blob_id, '');
});

test('preset form carries a default strength usable by the edit flow', () => {
  const form = presetToForm(null);
  assert.equal(form.strength, 0.75);
  const fromPreset = presetToForm({ preset_id: 'p', prompt: 'x' });
  assert.equal(fromPreset.strength, 0.75);
});

test('reference request carries adapter identity and clamps its scale', () => {
  const form = presetToForm(null);
  form.prompt = 'same character in a new scene';
  form.ipAdapterScale = '2.5';

  const request = buildReferenceRequest(form, 'img_reference', 'ip-adapter');

  assert.equal(request.mode, 'reference');
  assert.equal(request.source_blob_id, 'img_reference');
  assert.equal(request.edit_adapter_id, 'ip-adapter');
  assert.equal(request.ip_adapter_scale, 2);
  assert.equal(request.strength, undefined);
});

test('inpaint request binds source, mask, and dedicated pipeline identities', () => {
  const form = presetToForm(null);
  form.prompt = 'replace the selected window';
  form.strength = '0.6';

  const request = buildInpaintRequest(
    form,
    'img_source',
    'img_mask',
    'sd15_inpaint_v1',
  );

  assert.equal(request.mode, 'inpaint');
  assert.equal(request.source_blob_id, 'img_source');
  assert.equal(request.mask_blob_id, 'img_mask');
  assert.equal(request.edit_adapter_id, 'sd15_inpaint_v1');
  assert.equal(request.strength, 0.6);
});

test('instruction request separates the edit command and image guidance', () => {
  const form = presetToForm(null);
  form.prompt = '  make it a snowy winter day  ';
  form.imageGuidanceScale = '5';

  const request = buildInstructionRequest(
    form,
    'img_source',
    'sd15_instruct_pix2pix_v1',
  );

  assert.equal(request.mode, 'instruction');
  assert.equal(request.source_blob_id, 'img_source');
  assert.equal(request.edit_adapter_id, 'sd15_instruct_pix2pix_v1');
  assert.equal(request.prompt, 'make it a snowy winter day');
  assert.equal(request.instruction, 'make it a snowy winter day');
  assert.equal(request.image_guidance_scale, 4);
  assert.equal(request.strength, undefined);
});

test('diffusion JSON API keeps encoded identifiers and request payloads', async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options = {}) => {
    requests.push({ url, options });
    return new Response('{"ok":true}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    await unloadModel();
    await registerDiffusionArtifact('C:/models/sd15', { name: 'SD local' });
    await generateDiffusionImage({ prompt: 'lake', seed: 17 });
    await fetchDiffusionJob('job/a');
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests[0].url, '/api/models/unload');
  assert.deepEqual(JSON.parse(requests[0].options.body), {});
  assert.deepEqual(JSON.parse(requests[1].options.body), {
    path: 'C:/models/sd15',
    compute_hash: false,
    name: 'SD local',
  });
  assert.deepEqual(JSON.parse(requests[2].options.body), { prompt: 'lake', seed: 17 });
  assert.equal(requests[3].url, '/api/diffusion/jobs/job%2Fa');
});

test('diffusion blob API preserves binary PNG responses', async () => {
  const originalFetch = globalThis.fetch;
  let requested;
  globalThis.fetch = async (url, options = {}) => {
    requested = { url, options };
    return new Response(new Uint8Array([137, 80, 78, 71]), {
      status: 200,
      headers: { 'content-type': 'image/png', etag: 'sha256-test' },
    });
  };
  try {
    const result = await fetchDiffusionBlob('image/a');
    assert.equal(result.blob.size, 4);
    assert.equal(result.contentType, 'image/png');
    assert.equal(result.etag, 'sha256-test');
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requested.url, '/api/diffusion/blobs/image%2Fa');
  assert.equal(requested.options.headers.Accept, 'image/png');
});

test('email config API reads without secrets and posts the recipient', async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options = {}) => {
    requests.push({ url, options });
    return new Response('{"ok":true}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    await fetchEmailConfig();
    await updateEmailConfig('ops@example.com');
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests[0].url, '/api/cluster/email-config');
  assert.equal(requests[1].url, '/api/cluster/email-config');
  assert.equal(requests[1].options.method, 'POST');
  assert.deepEqual(JSON.parse(requests[1].options.body), { recipient: 'ops@example.com' });
});


test('diffusion upload leaves multipart boundaries to fetch and edit stays JSON', async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options = {}) => {
    requests.push({ url, options });
    return new Response('{ok:true}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
  try {
    await uploadDiffusionBlob(new Blob(['png']), 'mask');
    await editDiffusionImage({
      mode: 'img2img',
      source_blob_id: 'img_source',
      prompt: 'sketch',
    });
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests[0].url, '/api/diffusion/blobs');
  assert.equal(requests[0].options.body instanceof FormData, true);
  assert.equal(requests[0].options.body.get('purpose'), 'mask');
  assert.equal('Content-Type' in requests[0].options.headers, false);
  assert.equal(requests[1].options.headers['Content-Type'], 'application/json');
  assert.equal(JSON.parse(requests[1].options.body).source_blob_id, 'img_source');
});
