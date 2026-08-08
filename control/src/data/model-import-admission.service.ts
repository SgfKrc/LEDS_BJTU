import { Injectable } from '@nestjs/common';
import { ArtifactRuntimeRecord } from './artifact-runtime-repository';
import { ImportReport } from './artifact-store';
import { ImportOptions, ModelImportService } from './model-import-service';
import { ModelRuntimeCheckService, RuntimeCheckOptions } from './model-runtime-check.service';

export interface AdmittedImportReport extends ImportReport {
  runtime_check: ArtifactRuntimeRecord | null;
}

@Injectable()
export class ModelImportAdmissionService {
  constructor(
    private readonly imports: ModelImportService,
    private readonly runtimeChecks: ModelRuntimeCheckService,
  ) {}

  async importLocal(
    sourcePath: string,
    options: ImportOptions = {},
    nodeId = 'local',
    runtimeOptions: RuntimeCheckOptions = {},
  ): Promise<AdmittedImportReport> {
    const report = this.imports.importLocal(sourcePath, options);
    if (!report.manifest_path || report.failed > 0) {
      return { ...report, runtime_check: null };
    }
    const runtimeCheck = await this.runtimeChecks.checkManifestFile(
      report.manifest_path, nodeId, runtimeOptions,
    );
    if (runtimeCheck.status !== 'ready') {
      report.warnings.push(`runtime admission: ${runtimeCheck.status}`);
    }
    return { ...report, runtime_check: runtimeCheck };
  }
}
