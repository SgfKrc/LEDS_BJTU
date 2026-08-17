import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from '@playwright/test';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const MODEL_PATH = path.join(ROOT, 'models', 'sd15-original-v1');
const SOURCE_IMAGE = path.join(
  ROOT,
  'logs',
  'sd15',
  'sd15_original_v1_seed19950101.png',
);
const OUTPUT_DIR = path.join(ROOT, 'build', 'sd15-browser-e2e');

async function imageStatistics(locator) {
  return locator.evaluate(async image => {
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let minimum = 255;
    let maximum = 0;
    for (let index = 0; index < pixels.length; index += 64) {
      minimum = Math.min(minimum, pixels[index]);
      maximum = Math.max(maximum, pixels[index]);
    }
    return {
      width: image.naturalWidth,
      height: image.naturalHeight,
      sampledRange: maximum - minimum,
    };
  });
}

async function completedJob(request, jobId) {
  const response = await request.get(`/api/diffusion/jobs/${jobId}`);
  expect(response.ok()).toBeTruthy();
  const payload = await response.json();
  expect(payload.state).toBe('completed');
  return payload;
}

test('real Edge img2img upload and result continuation', async ({ browserName, page, request }) => {
  test.skip(browserName !== 'chromium', 'The real gate uses the installed Edge channel.');
  await fs.access(MODEL_PATH);
  await fs.access(SOURCE_IMAGE);
  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const cleanupBlobIds = [];
  const report = {
    schema_version: 1,
    status: 'running',
    model_path: MODEL_PATH,
    source_image: SOURCE_IMAGE,
    browser: null,
    timings: {},
    jobs: [],
  };

  try {
    report.browser = await page.context().browser().version();
    await page.goto('/?view=image');
    await expect(page.getByTestId('diffusion-workspace')).toBeVisible();
    await page.getByTestId('diffusion-workspace-assets').click();

    await page.getByTestId('diffusion-model-path').fill(MODEL_PATH);
    const registerStarted = performance.now();
    await page.getByTestId('diffusion-register').click();
    await expect(page.getByTestId('diffusion-artifact-select').locator('option')).not.toHaveCount(0);
    report.timings.register_seconds = (performance.now() - registerStarted) / 1000;

    await page.getByTestId('diffusion-profile').selectOption('balanced');
    const loadStarted = performance.now();
    await page.getByTestId('diffusion-load').click();
    await expect(page.getByTestId('diffusion-load')).toContainText('图像模型已就绪');
    await page.getByTestId('diffusion-workspace-generate').click();
    report.timings.load_seconds = (performance.now() - loadStarted) / 1000;

    await page.getByTestId('diffusion-mode-img2img').click();
    await page.getByTestId('diffusion-steps').fill('4');
    await page.getByTestId('diffusion-strength').fill('0.55');

    const uploadResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/blobs')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-source-input').setInputFiles(SOURCE_IMAGE);
    const uploadResponse = await uploadResponsePromise;
    expect(uploadResponse.status()).toBe(201);
    const uploaded = await uploadResponse.json();
    cleanupBlobIds.push(uploaded.blob_id);
    await expect(page.getByTestId('diffusion-source-preview')).toHaveAttribute(
      'data-blob-id',
      uploaded.blob_id,
    );

    const firstStarted = performance.now();
    const firstResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/edit')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-submit').click();
    const firstResponse = await firstResponsePromise;
    expect(firstResponse.status()).toBe(202);
    const firstQueued = await firstResponse.json();
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-id',
      firstQueued.job_id,
    );
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-state',
      'completed',
    );
    const firstJob = await completedJob(request, firstQueued.job_id);
    const firstBlobId = firstJob.blob.blob_id;
    cleanupBlobIds.push(firstBlobId);
    report.jobs.push({
      job_id: firstQueued.job_id,
      blob_id: firstBlobId,
      source_blob_id: uploaded.blob_id,
      elapsed_seconds: firstJob.metrics?.elapsed_seconds,
      browser_lifecycle_seconds: (performance.now() - firstStarted) / 1000,
      progress: firstJob.progress,
    });

    const resultImage = page.getByTestId('diffusion-result-image');
    await expect(resultImage).toBeVisible();
    const firstImage = await imageStatistics(resultImage);
    expect(firstImage).toMatchObject({ width: 512, height: 512 });
    expect(firstImage.sampledRange).toBeGreaterThan(10);

    await page.getByTestId('diffusion-continue-edit').click();
    await expect(page.getByTestId('diffusion-source-preview')).toHaveAttribute(
      'data-blob-id',
      firstBlobId,
    );
    await page.getByTestId('diffusion-seed').fill('19950104');

    const secondStarted = performance.now();
    const secondResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/edit')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-submit').click();
    const secondResponse = await secondResponsePromise;
    expect(secondResponse.status()).toBe(202);
    const secondQueued = await secondResponse.json();
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-id',
      secondQueued.job_id,
    );
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-state',
      'completed',
    );
    const secondJob = await completedJob(request, secondQueued.job_id);
    const secondBlobId = secondJob.blob.blob_id;
    cleanupBlobIds.push(secondBlobId);
    expect(secondBlobId).not.toBe(firstBlobId);
    report.jobs.push({
      job_id: secondQueued.job_id,
      blob_id: secondBlobId,
      source_blob_id: firstBlobId,
      elapsed_seconds: secondJob.metrics?.elapsed_seconds,
      browser_lifecycle_seconds: (performance.now() - secondStarted) / 1000,
      progress: secondJob.progress,
    });

    await expect(page.getByTestId('diffusion-result')).toHaveAttribute(
      'data-blob-id',
      secondBlobId,
    );
    const secondImage = await imageStatistics(resultImage);
    expect(secondImage).toMatchObject({ width: 512, height: 512 });
    expect(secondImage.sampledRange).toBeGreaterThan(10);
    report.image_checks = { first: firstImage, second: secondImage };

    const screenshotPath = path.join(OUTPUT_DIR, 'img2img-result-continuation.png');
    const resultScreenshotPath = path.join(OUTPUT_DIR, 'img2img-result-panel.png');
    await page.getByTestId('diffusion-result').screenshot({ path: resultScreenshotPath });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    report.screenshots = {
      page: screenshotPath,
      result: resultScreenshotPath,
    };

    await page.getByTestId('diffusion-unload').click();
    await expect(page.getByTestId('diffusion-unload')).toBeHidden();
    report.status = 'passed';
  } finally {
    for (const blobId of cleanupBlobIds.reverse()) {
      await request.delete(`/api/diffusion/blobs/${blobId}`).catch(() => {});
    }
    await request.post('/api/diffusion/unload').catch(() => {});
    await fs.writeFile(
      path.join(OUTPUT_DIR, 'browser-report.json'),
      `${JSON.stringify(report, null, 2)}\n`,
      'utf8',
    );
  }
});

test('real Edge inpaint canvas uploads a mask and completes an edit', async ({ browserName, page, request }) => {
  test.skip(browserName !== 'chromium', 'The real gate uses the installed Edge channel.');
  await fs.access(MODEL_PATH);
  await fs.access(SOURCE_IMAGE);
  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const cleanupBlobIds = [];
  const report = {
    schema_version: 1,
    status: 'running',
    model_path: MODEL_PATH,
    inpaint_artifact_id: 'sd15_inpaint_v1',
    source_image: SOURCE_IMAGE,
    browser: null,
    timings: {},
  };

  try {
    report.browser = await page.context().browser().version();
    await page.goto('/?view=image');
    await expect(page.getByTestId('diffusion-workspace')).toBeVisible();
    await page.getByTestId('diffusion-workspace-assets').click();
    await page.getByTestId('diffusion-model-path').fill(MODEL_PATH);
    await page.getByTestId('diffusion-register').click();
    await expect(page.getByTestId('diffusion-artifact-select').locator('option')).not.toHaveCount(0);
    await page.getByTestId('diffusion-profile').selectOption('balanced');
    await page.getByTestId('diffusion-load').click();
    await expect(page.getByTestId('diffusion-load')).toContainText('图像模型已就绪');
    await page.getByTestId('diffusion-workspace-generate').click();

    await page.getByTestId('diffusion-mode-inpaint').click();
    await page.getByTestId('diffusion-steps').fill('4');
    await page.getByTestId('diffusion-strength').fill('1');

    const sourceResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/blobs')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-source-input').setInputFiles(SOURCE_IMAGE);
    const sourceResponse = await sourceResponsePromise;
    expect(sourceResponse.status()).toBe(201);
    const sourceBlob = await sourceResponse.json();
    cleanupBlobIds.push(sourceBlob.blob_id);

    await expect(page.getByTestId('diffusion-inpaint-select')).toHaveValue('sd15_inpaint_v1');
    const canvas = page.getByTestId('diffusion-mask-canvas');
    await expect(canvas).toBeVisible();
    const bounds = await canvas.boundingBox();
    expect(bounds).not.toBeNull();
    const maskResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/blobs')
      && response.request().method() === 'POST'
    ));
    await page.mouse.move(bounds.x + bounds.width * 0.35, bounds.y + bounds.height * 0.35);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width * 0.65, bounds.y + bounds.height * 0.65, { steps: 8 });
    await page.mouse.up();
    const maskResponse = await maskResponsePromise;
    expect(maskResponse.status()).toBe(201);
    const maskBlob = await maskResponse.json();
    expect(maskBlob.purpose).toBe('mask');
    cleanupBlobIds.push(maskBlob.blob_id);
    await expect(page.getByTestId('diffusion-submit')).toBeEnabled();

    const editStarted = performance.now();
    const editResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/edit')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-submit').click();
    const editResponse = await editResponsePromise;
    expect(editResponse.status()).toBe(202);
    const queued = await editResponse.json();
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-state',
      'completed',
    );
    const completed = await completedJob(request, queued.job_id);
    cleanupBlobIds.push(completed.blob.blob_id);
    expect(completed.parameters).toMatchObject({
      mode: 'inpaint',
      source_blob_id: sourceBlob.blob_id,
      mask_blob_id: maskBlob.blob_id,
      edit_adapter_id: 'sd15_inpaint_v1',
    });
    expect(completed.metrics.engine).toBe('diffusers_sd15_inpaint');
    const resultImage = page.getByTestId('diffusion-result-image');
    const statistics = await imageStatistics(resultImage);
    expect(statistics).toMatchObject({ width: 512, height: 512 });
    expect(statistics.sampledRange).toBeGreaterThan(10);

    report.status = 'passed';
    report.timings.edit_seconds = (performance.now() - editStarted) / 1000;
    report.job = {
      job_id: queued.job_id,
      blob_id: completed.blob.blob_id,
      source_blob_id: sourceBlob.blob_id,
      mask_blob_id: maskBlob.blob_id,
      elapsed_seconds: completed.metrics.elapsed_seconds,
    };
    report.screenshot = path.join(OUTPUT_DIR, 'inpaint-result.png');
    await page.screenshot({ path: report.screenshot, fullPage: true });
  } finally {
    for (const blobId of cleanupBlobIds.reverse()) {
      await request.delete(`/api/diffusion/blobs/${blobId}`).catch(() => {});
    }
    await request.post('/api/diffusion/unload').catch(() => {});
    await fs.writeFile(
      path.join(OUTPUT_DIR, 'inpaint-browser-report.json'),
      `${JSON.stringify(report, null, 2)}\n`,
      'utf8',
    );
  }
});

test('real Edge instruction editing uses the dedicated pipeline', async ({ browserName, page, request }) => {
  test.skip(browserName !== 'chromium', 'The real gate uses the installed Edge channel.');
  await fs.access(MODEL_PATH);
  await fs.access(SOURCE_IMAGE);
  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const cleanupBlobIds = [];
  const report = {
    schema_version: 1,
    status: 'running',
    model_path: MODEL_PATH,
    instruction_artifact_id: 'sd15_instruct_pix2pix_v1',
    source_image: SOURCE_IMAGE,
    browser: null,
    timings: {},
  };

  try {
    report.browser = await page.context().browser().version();
    await page.goto('/?view=image');
    await expect(page.getByTestId('diffusion-workspace')).toBeVisible();
    await page.getByTestId('diffusion-workspace-assets').click();
    await page.getByTestId('diffusion-model-path').fill(MODEL_PATH);
    await page.getByTestId('diffusion-register').click();
    await expect(page.getByTestId('diffusion-artifact-select').locator('option')).not.toHaveCount(0);
    await page.getByTestId('diffusion-profile').selectOption('balanced');
    await page.getByTestId('diffusion-load').click();
    await expect(page.getByTestId('diffusion-load')).toContainText('图像模型已就绪');
    await page.getByTestId('diffusion-workspace-generate').click();

    await page.getByTestId('diffusion-mode-instruction').click();
    await expect(page.getByTestId('diffusion-instruction-select')).toHaveValue(
      'sd15_instruct_pix2pix_v1',
    );
    await page.getByTestId('diffusion-prompt').fill('make it a snowy winter day');
    await page.getByTestId('diffusion-steps').fill('4');
    await page.getByTestId('diffusion-image-guidance-scale').fill('1');

    const sourceResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/blobs')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-source-input').setInputFiles(SOURCE_IMAGE);
    const sourceResponse = await sourceResponsePromise;
    expect(sourceResponse.status()).toBe(201);
    const sourceBlob = await sourceResponse.json();
    cleanupBlobIds.push(sourceBlob.blob_id);
    await expect(page.getByTestId('diffusion-submit')).toBeEnabled();

    const editStarted = performance.now();
    const editResponsePromise = page.waitForResponse(response => (
      response.url().endsWith('/api/diffusion/edit')
      && response.request().method() === 'POST'
    ));
    await page.getByTestId('diffusion-submit').click();
    const editResponse = await editResponsePromise;
    expect(editResponse.status()).toBe(202);
    const requestPayload = editResponse.request().postDataJSON();
    expect(requestPayload).toMatchObject({
      mode: 'instruction',
      source_blob_id: sourceBlob.blob_id,
      edit_adapter_id: 'sd15_instruct_pix2pix_v1',
      instruction: 'make it a snowy winter day',
      image_guidance_scale: 1,
    });
    const queued = await editResponse.json();
    await expect(page.getByTestId('diffusion-job-status')).toHaveAttribute(
      'data-job-state',
      'completed',
    );
    const completed = await completedJob(request, queued.job_id);
    cleanupBlobIds.push(completed.blob.blob_id);
    expect(completed.parameters).toMatchObject({
      mode: 'instruction',
      source_blob_id: sourceBlob.blob_id,
      edit_adapter_id: 'sd15_instruct_pix2pix_v1',
      instruction: 'make it a snowy winter day',
      image_guidance_scale: 1,
    });
    expect(completed.metrics.engine).toBe('diffusers_sd15_instruct_pix2pix');
    expect(completed.metrics.instruction_pipeline_sha256).toBe(
      'a6626f7fedd356273f726b1707293266f11f6548a57730785ccbffe8efc872ab',
    );
    const statistics = await imageStatistics(page.getByTestId('diffusion-result-image'));
    expect(statistics).toMatchObject({ width: 512, height: 512 });
    expect(statistics.sampledRange).toBeGreaterThan(10);

    report.status = 'passed';
    report.timings.edit_seconds = (performance.now() - editStarted) / 1000;
    report.job = {
      job_id: queued.job_id,
      blob_id: completed.blob.blob_id,
      source_blob_id: sourceBlob.blob_id,
      elapsed_seconds: completed.metrics.elapsed_seconds,
      pipeline_sha256: completed.metrics.instruction_pipeline_sha256,
    };
    report.screenshot = path.join(OUTPUT_DIR, 'instruction-result.png');
    await page.screenshot({ path: report.screenshot, fullPage: true });
  } finally {
    for (const blobId of cleanupBlobIds.reverse()) {
      await request.delete(`/api/diffusion/blobs/${blobId}`).catch(() => {});
    }
    await request.post('/api/diffusion/unload').catch(() => {});
    await fs.writeFile(
      path.join(OUTPUT_DIR, 'instruction-browser-report.json'),
      `${JSON.stringify(report, null, 2)}\n`,
      'utf8',
    );
  }
});
