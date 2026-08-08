import { Injectable } from '@nestjs/common';
import { ClusterSettingsRepository } from './cluster-settings-repository';

export interface ModelLicenseAcceptance {
  schema_version: 1;
  repo_id: string;
  license_id: string;
  accepted_at: string;
  accepted_by: 'local_user';
}

const SETTINGS_KEY = 'model_license_acceptances_v1';

function normalizeRepoId(value: string): string {
  const repoId = value.trim();
  if (!/^[^/\s]+\/[^/\s]+$/.test(repoId)) {
    throw new Error('repo_id is invalid');
  }
  return repoId;
}

function normalizeLicenseId(value: string): string {
  const licenseId = value.trim().toLowerCase();
  if (!licenseId || licenseId === 'unknown'
      || licenseId.length > 128 || /[\r\n]/.test(licenseId)) {
    throw new Error('license_id is invalid');
  }
  return licenseId;
}

@Injectable()
export class ModelLicenseAcceptanceRepository {
  constructor(private readonly settings: ClusterSettingsRepository) {}

  list(): ModelLicenseAcceptance[] {
    const row = this.settings.get(SETTINGS_KEY);
    if (!row) return [];
    try {
      const parsed = JSON.parse(row.value) as unknown;
      if (!Array.isArray(parsed)) return [];
      return (parsed as ModelLicenseAcceptance[]).sort((a, b) => (
        a.repo_id.localeCompare(b.repo_id) || a.license_id.localeCompare(b.license_id)
      ));
    } catch {
      return [];
    }
  }

  get(repoId: string, licenseId: string): ModelLicenseAcceptance | null {
    const repo = normalizeRepoId(repoId);
    const license = normalizeLicenseId(licenseId);
    return this.list().find(
      (entry) => entry.repo_id === repo && entry.license_id === license,
    ) ?? null;
  }

  accept(repoId: string, licenseId: string): ModelLicenseAcceptance {
    const repo = normalizeRepoId(repoId);
    const license = normalizeLicenseId(licenseId);
    const existing = this.get(repo, license);
    if (existing) return existing;
    const acceptance: ModelLicenseAcceptance = {
      schema_version: 1,
      repo_id: repo,
      license_id: license,
      accepted_at: new Date().toISOString(),
      accepted_by: 'local_user',
    };
    this.settings.set(SETTINGS_KEY, JSON.stringify([
      ...this.list(), acceptance,
    ]));
    return acceptance;
  }

  revoke(repoId: string, licenseId: string): boolean {
    const repo = normalizeRepoId(repoId);
    const license = normalizeLicenseId(licenseId);
    const current = this.list();
    const next = current.filter(
      (entry) => entry.repo_id !== repo || entry.license_id !== license,
    );
    if (next.length === current.length) return false;
    this.settings.set(SETTINGS_KEY, JSON.stringify(next));
    return true;
  }
}
