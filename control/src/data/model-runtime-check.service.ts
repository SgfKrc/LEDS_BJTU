import { Injectable } from '@nestjs/common';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import { ArtifactStore } from './artifact-store';
import { ArtifactRuntimeRecord, ArtifactRuntimeRepository } from './artifact-runtime-repository';
import { ModelRuntimeSidecar, RuntimeSidecarResult } from './model-runtime-sidecar';

export interface RuntimeCheckOptions {
  signal?: AbortSignal;
}

@Injectable()
export class ModelRuntimeCheckService {
  constructor(
    private readonly sidecar: ModelRuntimeSidecar,
    private readonly repository: ArtifactRuntimeRepository,
    private readonly store: ArtifactStore,
  ) {}

  async checkManifest(
    manifest: Record<string, unknown>,
    nodeId = 'local',
    options: RuntimeCheckOptions = {},
  ): Promise<ArtifactRuntimeRecord> {
    const result = await this.sidecar.trialLoad(manifest, { signal: options.signal });
    const record = this.toRecord(result, nodeId);
    this.repository.upsert(record);
    return record;
  }

  async checkManifestFile(
    manifestPath: string,
    nodeId = 'local',
    options: RuntimeCheckOptions = {},
  ): Promise<ArtifactRuntimeRecord> {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    return this.checkManifest(manifest, nodeId, options);
  }

  async checkManifestReference(
    namespace: string,
    name: string,
    tag: string,
    nodeId = 'local',
    options: RuntimeCheckOptions = {},
  ): Promise<ArtifactRuntimeRecord> {
    for (const [label, value] of Object.entries({ namespace, name, tag })) {
      if (!/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(value)) {
        throw new Error(`${label} is invalid`);
      }
    }
    const manifest = this.store.readManifest(namespace, name, tag);
    if (!manifest) throw new Error(`manifest not found: ${namespace}/${name}:${tag}`);
    return this.checkManifest(manifest, nodeId, options);
  }

  private toRecord(result: RuntimeSidecarResult, nodeId: string): ArtifactRuntimeRecord {
    const normalizedNodeId = nodeId.trim();
    if (!/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(normalizedNodeId)) {
      throw new Error('node_id is invalid');
    }
    return {
      schema_version: 1,
      artifact_id: result.artifact_id,
      node_id: normalizedNodeId,
      runtime_profile: result.runtime_profile,
      status: result.status,
      checked_at: result.checked_at,
      engine: result.engine,
      loader_version: result.loader_version,
      runtime_fingerprint: this.runtimeFingerprint(result),
      load_ms: result.load_ms,
      details: result.details,
      error: result.error,
      invalidated_at: null,
      invalidation_reason: null,
    };
  }

  private runtimeFingerprint(result: RuntimeSidecarResult): string {
    const cpus = os.cpus();
    const stableContext = {
      schema_version: 1,
      platform: process.platform,
      arch: process.arch,
      os_release: os.release(),
      cpu_model: cpus[0]?.model ?? 'unknown',
      cpu_count: cpus.length,
      total_memory_bytes: os.totalmem(),
      total_vram_bytes: result.details.total_vram_bytes ?? null,
      engine: result.engine,
      runtime_profile: result.runtime_profile,
      loader_version: result.loader_version,
      n_gpu_layers: result.details.n_gpu_layers ?? null,
    };
    return `sha256:${crypto.createHash('sha256')
      .update(JSON.stringify(stableContext)).digest('hex')}`;
  }
}
