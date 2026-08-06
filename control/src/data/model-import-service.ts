/**
 * M2 本地导入服务：目录 / 单 GGUF 文件 → staging → 静态检查 → 原子提交 →
 * 内容寻址 manifest。源文件只读，绝不修改用户原文件。
 *
 * 失败路径：检查失败 → 整 job 移入 quarantine（留痕不删除），错误摘要永不
 * 进入 registry；相同 blob 去重（重复导入只增加引用）。
 */
import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import {
  ArtifactStore, ImportReport, sha256Hex,
} from './artifact-store';
import { ModelInspector, InspectionResult } from './model-inspector';

export interface ImportOptions {
  jobId?: string;
  namespace?: string;
  name?: string;
  tag?: string;
}

const SKIP_FILES = new Set(['.git', '.sha256', '.DS_Store']);

@Injectable()
export class ModelImportService {
  constructor(
    private readonly store: ArtifactStore,
    private readonly inspector: ModelInspector,
  ) {}

  /** 本地导入（目录 = Safetensors 仓库；单文件 = GGUF）。 */
  importLocal(sourcePath: string, options: ImportOptions = {}): ImportReport {
    const jobId = options.jobId ?? `import_${crypto.randomUUID().slice(0, 8)}`;
    const report: ImportReport = {
      job_id: jobId,
      imported: 0,
      deduped: 0,
      failed: 0,
      quarantined: 0,
      warnings: [],
      artifact_id: null,
      manifest_path: null,
    };
    const stat = fs.statSync(sourcePath);
    if (!stat.isDirectory() && !sourcePath.endsWith('.gguf')) {
      report.failed = 1;
      report.warnings.push(`不支持的导入类型: ${sourcePath}`);
      return report;
    }

    try {
      // 1) 只读源 → staging 拷贝（隔离；不修改源）
      if (stat.isDirectory()) {
        this.copyDirToStaging(sourcePath, jobId, report);
      } else {
        this.store.stageWrite(
          jobId, path.basename(sourcePath), fs.readFileSync(sourcePath),
        );
      }

      // 2) 静态检查（对 staging 副本，隔离于源）
      const stagingRoot = this.store.stagingDir(jobId);
      const inspection = stat.isDirectory()
        ? this.inspector.inspectSafetensorsDir(stagingRoot)
        : this.inspector.inspectGguf(
            path.join(stagingRoot, path.basename(sourcePath)),
          );
      report.warnings.push(...inspection.warnings);
      if (!inspection.ok || inspection.errors.length > 0) {
        report.failed = 1;
        report.quarantined = 1;
        this.store.quarantine(
          jobId,
          `inspection failed: ${inspection.errors.join('; ')}`,
        );
        return report;
      }

      // 3) 原子提交（同 digest 去重）
      const files: Array<{ path: string; size: number; sha256: string }> = [];
      for (const rel of this.store.listStaging(jobId)) {
        const blob = this.store.commitBlob(jobId, rel);
        files.push({ path: rel, size: blob.size, sha256: blob.digest });
        if (blob.deduped) report.deduped += 1;
        else report.imported += 1;
      }
      if (files.length === 0) {
        report.failed = 1;
        report.warnings.push('staging 为空（没有可导入文件）');
        this.store.cleanupStaging(jobId);
        return report;
      }

      // 4) manifest（对齐 schemas/artifact-manifest）
      const sorted = [...files].sort((a, b) => a.path.localeCompare(b.path));
      const aggregate = sha256Hex(
        Buffer.from(sorted.map((f) => f.sha256).join('')),
      );
      const artifactId = `sha256:${aggregate}`;
      const format = inspection.format;
      const manifest: Record<string, unknown> = {
        schema_version: 1,
        namespace: options.namespace ?? 'user',
        name: options.name ?? path.basename(sourcePath),
        tag: options.tag ?? 'latest',
        artifact_id: artifactId,
        source: {
          provider: format === 'gguf' ? 'gguf_huggingface' : 'local_directory',
          requested_revision: 'local',
          resolved_revision: 'local',
        },
        format,
        engine: format === 'gguf' ? 'llama_cpp' : 'pytorch_transformers',
        family: inspection.family ?? 'unknown',
        quantization: inspection.quantization,
        context_length: inspection.context_length,
        files: sorted,
        capabilities: inspection.capabilities,
        requirements: {
          runtime_profile: format === 'gguf' ? 'llm-cpu-v1' : 'llm-cuda-v1',
        },
        license: { id: 'unknown', acceptance_required: false },
        trust_policy: { trust_remote_code: false },
      };
      report.artifact_id = artifactId;
      report.manifest_path = this.store.writeManifest(manifest);
      this.store.cleanupStaging(jobId);
      return report;
    } catch (err) {
      report.failed = 1;
      report.quarantined = 1;
      report.warnings.push(
        `导入异常: ${err instanceof Error ? err.message : String(err)}`,
      );
      try {
        this.store.quarantine(jobId, 'import exception');
      } catch {
        // quarantine 失败不掩盖原异常
      }
      return report;
    }
  }

  private copyDirToStaging(
    sourceDir: string,
    jobId: string,
    report: ImportReport,
  ): void {
    const walk = (base: string, rel: string): void => {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        const full = path.join(base, entry.name);
        const nextRel = rel ? path.join(rel, entry.name) : entry.name;
        if (entry.isDirectory()) {
          if (!SKIP_FILES.has(entry.name)) walk(full, nextRel);
        } else if (!SKIP_FILES.has(entry.name)) {
          this.store.stageWrite(jobId, nextRel, fs.readFileSync(full));
        }
      }
    };
    walk(sourceDir, '');
  }
}

export { ImportReport, InspectionResult };
