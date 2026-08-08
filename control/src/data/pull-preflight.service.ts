import { Injectable } from '@nestjs/common';
import { ArtifactStore } from './artifact-store';
import { HfResolver } from './hf-resolver';
import { ModelDiskBudget } from './model-disk-budget';
import { ModelSource } from './model-source-repository';

export interface PullPreflightResult {
  schema_version: 1;
  status: 'ready' | 'insufficient_storage';
  source: {
    source_id: string;
    provider: string;
    endpoint: string;
    credential_ref: string | null;
  };
  repo_id: string;
  requested_revision: string;
  resolved_revision: string;
  files: Array<{ path: string; size: number; sha256: string | null }>;
  total_bytes: number;
  existing_bytes: number;
  disk_required_bytes: number;
  disk_available_bytes: number;
}

@Injectable()
export class PullPreflightService {
  constructor(
    private readonly resolver: HfResolver,
    private readonly diskBudget: ModelDiskBudget,
    private readonly store: ArtifactStore,
  ) {}

  async resolve(input: {
    source: ModelSource;
    repoId: string;
    requestedRevision?: string;
    allowPatterns?: string[] | null;
  }): Promise<PullPreflightResult> {
    if (!input.source.enabled) throw new Error('model source is disabled');
    if (input.source.provider !== 'huggingface') {
      throw new Error(`source provider is not implemented: ${input.source.provider}`);
    }
    const resolved = await this.resolver.resolve(
      input.repoId,
      input.requestedRevision ?? 'main',
      input.allowPatterns,
      input.source.endpoint,
    );
    const disk = this.diskBudget.evaluate(resolved.files, this.store);
    return {
      schema_version: 1,
      status: disk.sufficient ? 'ready' : 'insufficient_storage',
      source: {
        source_id: input.source.source_id,
        provider: input.source.provider,
        endpoint: input.source.endpoint,
        credential_ref: input.source.credential_ref,
      },
      repo_id: resolved.repoId,
      requested_revision: resolved.requestedRevision,
      resolved_revision: resolved.resolvedRevision,
      files: resolved.files.map((file) => ({
        path: file.rfilename,
        size: file.size,
        sha256: file.sha256 ?? null,
      })),
      ...disk,
    };
  }
}
