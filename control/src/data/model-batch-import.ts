/**
 * MF-N3 / M2.1: scan an existing model directory, import supported entries into
 * the content-addressed store, and emit one auditable aggregate report.
 */
import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { ImportReport } from './artifact-store';
import { ModelImportService } from './model-import-service';

export interface BatchImportEntry {
  source_path: string;
  name: string;
  report: ImportReport;
}

export interface BatchImportReport {
  schema_version: 1;
  source_root: string;
  started_at: string;
  completed_at: string;
  totals: {
    candidates: number;
    succeeded: number;
    failed: number;
    quarantined: number;
    imported_blobs: number;
    deduped_blobs: number;
    ignored_entries: number;
  };
  ignored: string[];
  entries: BatchImportEntry[];
}

export interface BatchImportOptions {
  namespace?: string;
  tag?: string;
}

export interface CatalogImportCandidate {
  model_id: string;
  name: string;
  format: 'safetensors' | 'gguf' | 'both';
  local_path?: string;
  gguf_path?: string;
}

export interface CatalogMigrationEntry {
  model_id: string;
  model_name: string;
  format: 'safetensors' | 'gguf';
  source_path: string;
  status: 'succeeded' | 'missing' | 'failed';
  reason?: string;
  report: ImportReport | null;
}

export interface CatalogMigrationReport {
  schema_version: 1;
  catalog_path: string;
  started_at: string;
  completed_at: string;
  totals: {
    models: number;
    expected_sources: number;
    succeeded_models: number;
    partial_models: number;
    missing_sources: number;
    failed_sources: number;
    quarantined_sources: number;
    imported_blobs: number;
    deduped_blobs: number;
  };
  entries: CatalogMigrationEntry[];
}

@Injectable()
export class ModelBatchImporter {
  constructor(private readonly importer: ModelImportService) {}

  importDirectory(
    sourceRoot: string,
    options: BatchImportOptions = {},
  ): BatchImportReport {
    const root = path.resolve(sourceRoot);
    const stat = fs.statSync(root);
    if (!stat.isDirectory()) {
      throw new Error(`批量导入源必须是目录: ${root}`);
    }
    const startedAt = new Date().toISOString();
    const { candidates, ignored } = this.discover(root);
    const entries: BatchImportEntry[] = [];
    for (const sourcePath of candidates) {
      const basename = path.basename(sourcePath);
      const name = basename.toLowerCase().endsWith('.gguf')
        ? basename.slice(0, -5)
        : basename;
      const jobId = `batch_${crypto.randomUUID().slice(0, 12)}`;
      const report = this.importer.importLocal(sourcePath, {
        jobId,
        namespace: options.namespace ?? 'migration',
        name,
        tag: options.tag ?? 'imported',
      });
      entries.push({ source_path: sourcePath, name, report });
    }

    return {
      schema_version: 1,
      source_root: root,
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      totals: {
        candidates: entries.length,
        succeeded: entries.filter((entry) => entry.report.failed === 0).length,
        failed: entries.filter((entry) => entry.report.failed > 0).length,
        quarantined: entries.reduce(
          (total, entry) => total + entry.report.quarantined, 0,
        ),
        imported_blobs: entries.reduce(
          (total, entry) => total + entry.report.imported, 0,
        ),
        deduped_blobs: entries.reduce(
          (total, entry) => total + entry.report.deduped, 0,
        ),
        ignored_entries: ignored.length,
      },
      ignored,
      entries,
    };
  }

  writeReport(report: BatchImportReport, targetPath: string): string {
    return this.writeJsonReport(report, targetPath);
  }

  writeCatalogReport(report: CatalogMigrationReport, targetPath: string): string {
    return this.writeJsonReport(report, targetPath);
  }

  importCatalog(
    catalogPath: string,
    candidates: CatalogImportCandidate[],
    options: BatchImportOptions = {},
  ): CatalogMigrationReport {
    const startedAt = new Date().toISOString();
    const entries: CatalogMigrationEntry[] = [];
    for (const candidate of candidates) {
      const sources: Array<{ format: 'safetensors' | 'gguf'; sourcePath: string }> = [];
      if (candidate.format === 'safetensors' || candidate.format === 'both') {
        sources.push({ format: 'safetensors', sourcePath: candidate.local_path ?? '' });
      }
      if (candidate.format === 'gguf' || candidate.format === 'both') {
        sources.push({ format: 'gguf', sourcePath: candidate.gguf_path ?? '' });
      }
      for (const source of sources) {
        const sourcePath = source.sourcePath ? path.resolve(source.sourcePath) : '';
        if (!sourcePath || !fs.existsSync(sourcePath)) {
          entries.push({
            model_id: candidate.model_id,
            model_name: candidate.name,
            format: source.format,
            source_path: sourcePath,
            status: 'missing',
            reason: sourcePath ? 'source path does not exist' : 'catalog path is empty',
            report: null,
          });
          continue;
        }
        const report = this.importer.importLocal(sourcePath, {
          jobId: `catalog_${crypto.randomUUID().slice(0, 12)}`,
          namespace: options.namespace ?? 'migration',
          name: `${candidate.model_id}-${source.format}`,
          tag: options.tag ?? 'catalog-imported',
        });
        entries.push({
          model_id: candidate.model_id,
          model_name: candidate.name,
          format: source.format,
          source_path: sourcePath,
          status: report.failed === 0 ? 'succeeded' : 'failed',
          reason: report.failed === 0 ? undefined : report.warnings.join('; '),
          report,
        });
      }
    }
    const modelIds = [...new Set(candidates.map((candidate) => candidate.model_id))];
    const completeModels = modelIds.filter((modelId) => {
      const modelEntries = entries.filter((entry) => entry.model_id === modelId);
      return modelEntries.length > 0 && modelEntries.every((entry) => entry.status === 'succeeded');
    });
    return {
      schema_version: 1,
      catalog_path: path.resolve(catalogPath),
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      totals: {
        models: modelIds.length,
        expected_sources: entries.length,
        succeeded_models: completeModels.length,
        partial_models: modelIds.length - completeModels.length,
        missing_sources: entries.filter((entry) => entry.status === 'missing').length,
        failed_sources: entries.filter((entry) => entry.status === 'failed').length,
        quarantined_sources: entries.reduce(
          (total, entry) => total + (entry.report?.quarantined ?? 0), 0,
        ),
        imported_blobs: entries.reduce(
          (total, entry) => total + (entry.report?.imported ?? 0), 0,
        ),
        deduped_blobs: entries.reduce(
          (total, entry) => total + (entry.report?.deduped ?? 0),
          0,
        ),
      },
      entries,
    };
  }

  private writeJsonReport(report: object, targetPath: string): string {
    const target = path.resolve(targetPath);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const temp = `${target}.tmp-${process.pid}`;
    fs.writeFileSync(temp, `${JSON.stringify(report, null, 2)}\n`, 'utf-8');
    fs.renameSync(temp, target);
    return target;
  }

  private discover(root: string): { candidates: string[]; ignored: string[] } {
    const rootEntries = fs.readdirSync(root, { withFileTypes: true });
    const rootIsModel = rootEntries.some(
      (entry) => entry.isFile()
        && (entry.name === 'config.json' || entry.name.endsWith('.safetensors')),
    );
    if (rootIsModel) return { candidates: [root], ignored: [] };

    const candidates: string[] = [];
    const ignored: string[] = [];
    for (const entry of rootEntries.sort((a, b) => (
      a.name < b.name ? -1 : a.name > b.name ? 1 : 0
    ))) {
      const full = path.join(root, entry.name);
      if (entry.isSymbolicLink()) {
        ignored.push(`${entry.name}: symbolic link`);
      } else if (entry.isDirectory() || (
        entry.isFile() && entry.name.toLowerCase().endsWith('.gguf')
      )) {
        candidates.push(full);
      } else {
        ignored.push(`${entry.name}: unsupported entry`);
      }
    }
    return { candidates, ignored };
  }
}
