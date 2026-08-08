import { Injectable } from '@nestjs/common';
import { ClusterProfileRepository, ClusterProfileRow } from './cluster-profile-repository';
import { ClusterSettingsRepository } from './cluster-settings-repository';

const ACTIVE_PROFILE_KEY = 'active_cluster_profile_v1';

@Injectable()
export class ClusterProfileSelectionService {
  constructor(
    private readonly profiles: ClusterProfileRepository,
    private readonly settings: ClusterSettingsRepository,
  ) {}

  current(): ClusterProfileRow | null {
    const selected = this.settings.get(ACTIVE_PROFILE_KEY)?.value;
    return selected ? this.profiles.get(selected) : null;
  }

  activate(profileId: string): ClusterProfileRow {
    const profile = this.profiles.get(profileId);
    if (!profile) throw new Error(`档案不存在: ${profileId}`);
    this.settings.set(ACTIVE_PROFILE_KEY, profileId);
    return profile;
  }

  clearIf(profileId: string): void {
    if (this.settings.get(ACTIVE_PROFILE_KEY)?.value === profileId) {
      this.settings.delete(ACTIVE_PROFILE_KEY);
    }
  }
}
