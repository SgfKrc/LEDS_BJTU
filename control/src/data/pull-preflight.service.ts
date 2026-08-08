import { Injectable } from '@nestjs/common';
import { ArtifactStore } from './artifact-store';
import { HfResolver } from './hf-resolver';
import { ModelCredentialStore } from './model-credential-store';
import { ModelDiskBudget } from './model-disk-budget';
import { ModelLicenseAcceptanceRepository } from './model-license-acceptance';
import { ModelSource } from './model-source-repository';

export type PullPreflightStatus =
  | 'ready'
  | 'insufficient_storage'
  | 'credential_required'
  | 'license_required';

export interface PullPreflightResult {
  schema_version: 1;
  status: PullPreflightStatus;
  source: {
    source_id: string;
    provider: string;
    endpoint: string;
    credential_ref: string | null;
  };
  access: {
    gated: boolean;
    license_id: string;
    credential_required: boolean;
    credential_available: boolean;
    acceptance_required: boolean;
    accepted_at: string | null;
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
    private readonly credentials: ModelCredentialStore,
    private readonly licenses: ModelLicenseAcceptanceRepository,
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
    const token = await this.credentials.get(input.source.credential_ref);
    const resolved = await this.resolver.resolve(
      input.repoId,
      input.requestedRevision ?? 'main',
      input.allowPatterns,
      input.source.endpoint,
      { token },
    );
    const disk = this.diskBudget.evaluate(resolved.files, this.store);
    const licenseId = resolved.license ?? 'unknown';
    const acceptance = resolved.gated && resolved.license
      ? this.licenses.get(resolved.repoId, licenseId)
      : null;
    let status: PullPreflightStatus = disk.sufficient
      ? 'ready'
      : 'insufficient_storage';
    if (resolved.gated && !token) status = 'credential_required';
    else if (resolved.gated && !acceptance) status = 'license_required';
    return {
      schema_version: 1,
      status,
      source: {
        source_id: input.source.source_id,
        provider: input.source.provider,
        endpoint: input.source.endpoint,
        credential_ref: input.source.credential_ref,
      },
      access: {
        gated: resolved.gated,
        license_id: licenseId,
        credential_required: resolved.gated,
        credential_available: token !== null,
        acceptance_required: resolved.gated,
        accepted_at: acceptance?.accepted_at ?? null,
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
