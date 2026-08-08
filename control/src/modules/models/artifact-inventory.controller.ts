import { Controller, Get } from '@nestjs/common';
import {
  ArtifactRuntimeRecord, ArtifactRuntimeRepository,
} from '../../data/artifact-runtime-repository';
import { ArtifactStore } from '../../data/artifact-store';

interface ManifestFile {
  path?: unknown;
  size?: unknown;
  sha256?: unknown;
}

@Controller('models')
export class ArtifactInventoryController {
  constructor(
    private readonly artifacts: ArtifactStore,
    private readonly runtimeChecks: ArtifactRuntimeRepository,
  ) {}

  @Get('artifacts')
  list(): Record<string, unknown> {
    const nodeId = process.env.QLH_NODE_ID?.trim() || 'local';
    const checks = this.runtimeChecks.list({ nodeId });
    const entries = this.artifacts.listManifests().map(({ reference, manifest }) => {
      const artifactId = String(manifest.artifact_id ?? '');
      const requirements = this.record(manifest.requirements);
      const runtimeProfile = String(requirements.runtime_profile ?? '');
      const runtimeCheck = checks.find((check) => (
        check.artifact_id === artifactId
        && (!runtimeProfile || check.runtime_profile === runtimeProfile)
      )) ?? null;
      const files = Array.isArray(manifest.files)
        ? (manifest.files as ManifestFile[]).map((file) => ({
            path: String(file.path ?? ''),
            size: this.nonNegativeNumber(file.size),
            sha256: String(file.sha256 ?? ''),
          }))
        : [];
      return {
        artifact_id: artifactId,
        reference,
        format: String(manifest.format ?? 'unknown'),
        engine: String(manifest.engine ?? 'unknown'),
        family: String(manifest.family ?? 'unknown'),
        quantization: manifest.quantization ?? null,
        context_length: manifest.context_length ?? null,
        source: this.record(manifest.source),
        capabilities: this.record(manifest.capabilities),
        requirements,
        license: this.record(manifest.license),
        files,
        storage: {
          file_count: files.length,
          total_bytes: files.reduce((sum, file) => sum + file.size, 0),
        },
        runtime_check: runtimeCheck ? this.publicRecord(runtimeCheck) : null,
        runnable: runtimeCheck?.status === 'ready',
      };
    });
    return {
      node_id: nodeId,
      artifacts: entries,
      summary: {
        total: entries.length,
        ready: entries.filter((entry) => entry.runtime_check?.status === 'ready').length,
        stale: entries.filter((entry) => entry.runtime_check?.status === 'stale').length,
        attention: entries.filter((entry) => (
          entry.runtime_check && !['ready', 'stale'].includes(entry.runtime_check.status)
        )).length,
        unchecked: entries.filter((entry) => !entry.runtime_check).length,
        total_bytes: entries.reduce((sum, entry) => sum + entry.storage.total_bytes, 0),
      },
    };
  }

  private record(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? value as Record<string, unknown>
      : {};
  }

  private nonNegativeNumber(value: unknown): number {
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : 0;
  }

  private publicRecord(record: ArtifactRuntimeRecord): ArtifactRuntimeRecord {
    const details = { ...(record.details ?? {}) };
    delete details.stderr_tail;
    return { ...record, details };
  }
}
