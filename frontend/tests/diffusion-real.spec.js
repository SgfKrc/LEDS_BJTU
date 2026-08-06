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

    await page.getByTestId('diffusion-model-path').fill(MODEL_PATH);
    const registerStarted = performance.now();
    await page.getByTestId('diffusion-register').click();
    await expect(page.getByTestId('diffusion-artifact-select').locator('option')).not.toHaveCount(0);
    report.timings.register_seconds = (performance.now() - registerStarted) / 1000;

    await page.getByTestId('diffusion-profile').selectOption('balanced');
    const loadStarted = performance.now();
    await page.getByTestId('diffusion-load').click();
    await expect(page.getByTestId('diffusion-load')).toContainText('图像模型已就绪');
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
