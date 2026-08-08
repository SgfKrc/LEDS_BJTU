import {
  existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ArtifactStore } from '../src/data/artifact-store';
import { ModelBatchImporter } from '../src/data/model-batch-import';
import { ModelImportService } from '../src/data/model-import-service';
import { ModelInspector } from '../src/data/model-inspector';
import { run } from '../src/model-fleet-import';

function minimalGguf(): Buffer {
  const version = Buffer.alloc(4);
  version.writeUInt32LE(3);
  const tensorCount = Buffer.alloc(8);
  const metadataCount = Buffer.alloc(8);
  return Buffer.concat([
    Buffer.from('GGUF', 'ascii'), version, tensorCount, metadataCount,
  ]);
}

function safetensorsModel(target: string): void {
  mkdirSync(target, { recursive: true });
  writeFileSync(join(target, 'config.json'), JSON.stringify({
    model_type: 'qwen2',
    architectures: ['Qwen2ForCausalLM'],
    max_position_embeddings: 8192,
  }));
  const header = Buffer.from(JSON.stringify({
    weight: { dtype: 'F16', shape: [1], data_offsets: [0, 2] },
  }));
  const headerLength = Buffer.alloc(8);
  headerLength.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(
    join(target, 'model.safetensors'),
    Buffer.concat([headerLength, header, Buffer.alloc(2)]),
  );
  writeFileSync(join(target, 'tokenizer.json'), '{}');
}

describe('MODEL-FLEET batch model import (MF-N3)', () => {
  it('imports supported entries, quarantines failures, and writes a report', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-batch-'));
    const source = join(dir, 'models');
    mkdirSync(source);
    safetensorsModel(join(source, 'qwen-safe'));
    writeFileSync(join(source, 'qwen.gguf'), minimalGguf());
    writeFileSync(join(source, 'broken.gguf'), Buffer.from('broken'));
    writeFileSync(join(source, 'README.txt'), 'ignored');

    const store = new ArtifactStore(join(dir, 'store'));
    const batch = new ModelBatchImporter(
      new ModelImportService(store, new ModelInspector()),
    );
    const first = batch.importDirectory(source, {
      namespace: 'migration',
      tag: 'baseline',
    });
    expect(first.schema_version).toBe(1);
    expect(first.totals).toMatchObject({
      candidates: 3,
      succeeded: 2,
      failed: 1,
      quarantined: 1,
      ignored_entries: 1,
    });
    expect(first.totals.imported_blobs).toBeGreaterThanOrEqual(4);
    expect(first.ignored).toEqual(['README.txt: unsupported entry']);
    expect(first.entries.map((entry) => entry.name)).toEqual([
      'broken', 'qwen-safe', 'qwen',
    ]);
    expect(first.entries.find((entry) => entry.name === 'broken')?.report.artifact_id)
      .toBeNull();

    const reportPath = batch.writeReport(first, join(dir, 'reports', 'batch.json'));
    expect(existsSync(reportPath)).toBe(true);
    expect(JSON.parse(readFileSync(reportPath, 'utf-8')).totals.succeeded).toBe(2);

    const second = batch.importDirectory(source, {
      namespace: 'migration',
      tag: 'baseline',
    });
    expect(second.totals.succeeded).toBe(2);
    expect(second.totals.deduped_blobs).toBeGreaterThanOrEqual(4);
    rmSync(dir, { recursive: true, force: true });
  });

  it('exposes an executable CLI with deterministic exit codes', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-batch-cli-'));
    const source = join(dir, 'single-model');
    safetensorsModel(source);
    const reportPath = join(dir, 'report.json');
    const output = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
    const error = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
    try {
      expect(run([
        '--source', source,
        '--store', join(dir, 'store'),
        '--report', reportPath,
      ])).toBe(0);
      expect(existsSync(reportPath)).toBe(true);
      expect(run([])).toBe(2);
    } finally {
      output.mockRestore();
      error.mockRestore();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('records catalog source status without turning missing paths into imports', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-catalog-'));
    const present = join(dir, 'present');
    safetensorsModel(present);
    const store = new ArtifactStore(join(dir, 'store'));
    const batch = new ModelBatchImporter(
      new ModelImportService(store, new ModelInspector()),
    );
    const report = batch.importCatalog(join(dir, 'catalog.json'), [
      {
        model_id: 'present-model', name: 'Present', format: 'safetensors',
        local_path: present,
      },
      {
        model_id: 'missing-model', name: 'Missing', format: 'both',
        local_path: join(dir, 'missing-dir'), gguf_path: join(dir, 'missing.gguf'),
      },
    ], { namespace: 'migration', tag: 'catalog-test' });
    expect(report.totals).toMatchObject({
      models: 2, expected_sources: 3, succeeded_models: 1,
      partial_models: 1, missing_sources: 2, failed_sources: 0,
    });
    expect(report.entries.map((entry) => entry.status)).toEqual([
      'succeeded', 'missing', 'missing',
    ]);
    const reportPath = batch.writeCatalogReport(report, join(dir, 'reports', 'catalog.json'));
    expect(existsSync(reportPath)).toBe(true);
    rmSync(dir, { recursive: true, force: true });
  });
});
