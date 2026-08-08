import { Injectable } from '@nestjs/common';
import { ClusterProfileRow } from './cluster-profile-repository';

export interface DiscoveryCandidate {
  endpoint: string;
  source: 'active_profile' | 'env' | 'tailscale_peer';
  priority: number;
  cluster_id: string | null;
}

@Injectable()
export class ClusterDiscoveryService {
  candidates(input: {
    current: ClusterProfileRow | null;
    envHost?: string | null;
    tailscalePeers?: string[];
  }): DiscoveryCandidate[] {
    const result: DiscoveryCandidate[] = [];
    const seen = new Set<string>();
    const add = (
      endpoint: string | null | undefined,
      source: DiscoveryCandidate['source'],
      priority: number,
      clusterId: string | null,
    ): void => {
      const normalized = endpoint?.trim();
      if (!normalized || seen.has(normalized)) return;
      seen.add(normalized);
      result.push({ endpoint: normalized, source, priority, cluster_id: clusterId });
    };
    add(input.current?.master_endpoint, 'active_profile', 10,
      input.current?.cluster_id ?? null);
    add(input.envHost, 'env', 20, null);
    for (const peer of input.tailscalePeers ?? []) add(peer, 'tailscale_peer', 30, null);
    return result;
  }
}
