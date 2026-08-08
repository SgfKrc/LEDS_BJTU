import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SqliteStore } from './sqlite-store';
import {
  ArtifactRuntimeRecord, ArtifactRuntimeRepository,
} from './artifact-runtime-repository';

export type SimulationStatus =
  | 'planned' | 'preparing' | 'distributing' | 'ready' | 'active'
  | 'failed' | 'partial' | 'rolled_back';

export interface SimulationNode {
  node_id: string;
  artifact_ids: string[];
  capabilities: string[];
  runtime_fingerprint: string;
  available?: boolean;
}

export interface DeploymentRecord {
  schema_version: 1;
  deployment_id: string;
  artifact_id: string;
  node_id: string;
  status: SimulationStatus;
  epoch: number;
  digest_verified: boolean;
  runtime_profile: string;
  runtime_fingerprint: string | null;
  runtime_checked_at: string | null;
  prepared_at: string | null;
  activated_at: string | null;
  error: { code: string; message: string } | null;
}

export interface DeploymentSimulationPlan {
  schema_version: 1;
  plan_id: string;
  artifact_id: string;
  runtime_profile: string;
  required_capabilities: string[];
  target_nodes: string[];
  actual_nodes: string[];
  status: SimulationStatus;
  epoch: number;
  records: DeploymentRecord[];
  created_at: string;
  updated_at: string;
}

@Injectable()
export class DeploymentSimulator {
  constructor(
    private readonly store: SqliteStore,
    private readonly runtimeChecks: ArtifactRuntimeRepository,
  ) {}

  createPlan(input: {
    artifactId: string;
    nodes: SimulationNode[];
    runtimeProfile: string;
    requiredCapabilities?: string[];
  }): DeploymentSimulationPlan {
    if (!/^sha256:[0-9a-f]{64}$/.test(input.artifactId)) {
      throw new Error('artifact_id must be a sha256 digest');
    }
    if (input.nodes.length === 0) throw new Error('at least one node is required');
    if (!/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(input.runtimeProfile)) {
      throw new Error('runtime_profile is required and must be valid');
    }
    if (input.nodes.some((node) => !/^sha256:[0-9a-f]{64}$/.test(node.runtime_fingerprint))) {
      throw new Error('each node must provide a runtime_fingerprint');
    }
    const targetNodes = input.nodes.map((node) => node.node_id);
    if (new Set(targetNodes).size !== targetNodes.length) {
      throw new Error('node_id must be unique in a plan');
    }
    const now = new Date().toISOString();
    const plan: DeploymentSimulationPlan = {
      schema_version: 1,
      plan_id: `sim_${randomUUID().slice(0, 12)}`,
      artifact_id: input.artifactId,
      runtime_profile: input.runtimeProfile,
      required_capabilities: [...(input.requiredCapabilities ?? [])],
      target_nodes: targetNodes,
      actual_nodes: [],
      status: 'planned',
      epoch: 0,
      records: input.nodes.map((node) => ({
        schema_version: 1,
        deployment_id: `dep_${randomUUID().slice(0, 12)}`,
        artifact_id: input.artifactId,
        node_id: node.node_id,
        status: 'planned',
        epoch: 0,
        digest_verified: false,
        runtime_profile: input.runtimeProfile,
        runtime_fingerprint: null,
        runtime_checked_at: null,
        prepared_at: null,
        activated_at: null,
        error: null,
      })),
      created_at: now,
      updated_at: now,
    };
    this.save(plan, input.nodes);
    return plan;
  }

  get(planId: string): DeploymentSimulationPlan | null {
    const rows = this.store.prepare(
      'SELECT payload FROM deployments ORDER BY created_at, rowid',
    ).all() as unknown as Array<{ payload: string }>;
    const selected = rows
      .map((row) => JSON.parse(row.payload) as PersistedRecord)
      .filter((row) => row.plan_id === planId);
    if (selected.length === 0) return null;
    const first = selected[0];
    return {
      schema_version: 1,
      plan_id: first.plan_id,
      artifact_id: first.artifact_id,
      runtime_profile: first.runtime_profile,
      required_capabilities: first.required_capabilities,
      target_nodes: first.target_nodes,
      actual_nodes: first.actual_nodes,
      status: first.plan_status,
      epoch: first.plan_epoch,
      records: first.target_nodes
        .map((nodeId) => selected.find((row) => row.node_id === nodeId)?.record)
        .filter((record): record is DeploymentRecord => Boolean(record)),
      created_at: first.created_at,
      updated_at: first.updated_at,
    };
  }

  prepare(planId: string, nodes: SimulationNode[]): DeploymentSimulationPlan {
    const plan = this.require(planId);
    if (plan.status === 'active' || plan.status === 'rolled_back') {
      throw new Error(`plan cannot prepare from ${plan.status}`);
    }
    const byId = new Map(nodes.map((node) => [node.node_id, node]));
    const now = new Date().toISOString();
    const records = plan.records.map((record) => {
      const node = byId.get(record.node_id);
      if (!node || node.available === false) {
        return this.failed(record, 'node_unavailable', 'node is unavailable');
      }
      if (!node.artifact_ids.includes(plan.artifact_id)) {
        return this.failed(record, 'digest_mismatch', 'node does not have the requested artifact digest');
      }
      const missing = plan.required_capabilities.filter(
        (capability) => !node.capabilities.includes(capability),
      );
      if (missing.length > 0) {
        return this.failed(record, 'capability_mismatch', `missing capabilities: ${missing.join(',')}`);
      }
      const runtime = this.runtimeChecks.get(
        plan.artifact_id, record.node_id, plan.runtime_profile,
      );
      const runtimeFailure = this.runtimeFailure(record, node, runtime);
      if (runtimeFailure) return runtimeFailure;
      return {
        ...record,
        status: 'ready' as const,
        digest_verified: true,
        runtime_profile: plan.runtime_profile,
        runtime_fingerprint: runtime?.runtime_fingerprint ?? null,
        runtime_checked_at: runtime?.checked_at ?? null,
        prepared_at: now,
        error: null,
      };
    });
    const ready = records.filter((record) => record.status === 'ready').length;
    return this.saveAndReturn({
      ...plan,
      records,
      actual_nodes: [],
      status: ready === 0 ? 'failed' : ready === records.length ? 'ready' : 'partial',
      updated_at: now,
    }, nodes);
  }

  activate(planId: string): DeploymentSimulationPlan {
    const plan = this.require(planId);
    const prepared = plan.records.filter((record) => record.status === 'ready');
    if (prepared.length === 0) throw new Error('no prepared node can be activated');
    const now = new Date().toISOString();
    const epoch = plan.epoch + 1;
    const records = plan.records.map((record) => {
      if (record.status !== 'ready') return record;
      const runtime = this.runtimeChecks.get(
        plan.artifact_id, record.node_id, plan.runtime_profile,
      );
      if (!runtime || runtime.status !== 'ready'
          || runtime.runtime_fingerprint !== record.runtime_fingerprint) {
        return this.runtimeFailed(
          record,
          'runtime_admission_changed',
          'runtime admission changed after prepare; run prepare again',
          runtime,
        );
      }
      return { ...record, status: 'active' as const, epoch, activated_at: now };
    });
    const active = records.filter((record) => record.status === 'active');
    return this.saveAndReturn({
      ...plan,
      records,
      actual_nodes: active.map((record) => record.node_id),
      status: active.length === 0 ? 'failed'
        : active.length === records.length ? 'active' : 'partial',
      epoch,
      updated_at: now,
    }, []);
  }

  rollback(planId: string): DeploymentSimulationPlan {
    const plan = this.require(planId);
    if (!['active', 'partial'].includes(plan.status)) {
      throw new Error(`plan cannot rollback from ${plan.status}`);
    }
    const epoch = plan.epoch + 1;
    const records = plan.records.map((record) => record.status === 'active'
      ? { ...record, status: 'rolled_back' as const, epoch }
      : record);
    return this.saveAndReturn({
      ...plan,
      records,
      actual_nodes: [],
      status: 'rolled_back',
      epoch,
      updated_at: new Date().toISOString(),
    }, []);
  }

  private failed(record: DeploymentRecord, code: string, message: string): DeploymentRecord {
    return {
      ...record,
      status: 'failed',
      digest_verified: false,
      error: { code, message },
    };
  }

  private runtimeFailure(
    record: DeploymentRecord,
    node: SimulationNode,
    runtime: ArtifactRuntimeRecord | null,
  ): DeploymentRecord | null {
    if (!runtime) {
      return this.runtimeFailed(
        record, 'runtime_unchecked',
        'artifact has no runtime check for this node and profile', null,
      );
    }
    if (runtime.status !== 'ready') {
      return this.runtimeFailed(
        record, 'runtime_not_ready', `runtime status is ${runtime.status}`, runtime,
      );
    }
    if (!runtime.runtime_fingerprint
        || runtime.runtime_fingerprint !== node.runtime_fingerprint) {
      return this.runtimeFailed(
        record, 'runtime_context_changed',
        'node runtime fingerprint differs from the checked runtime', runtime,
      );
    }
    return null;
  }

  private runtimeFailed(
    record: DeploymentRecord,
    code: string,
    message: string,
    runtime: ArtifactRuntimeRecord | null,
  ): DeploymentRecord {
    return {
      ...record,
      status: 'failed',
      digest_verified: true,
      runtime_fingerprint: runtime?.runtime_fingerprint ?? null,
      runtime_checked_at: runtime?.checked_at ?? null,
      error: { code, message },
    };
  }

  private require(planId: string): DeploymentSimulationPlan {
    const plan = this.get(planId);
    if (!plan) throw new Error(`deployment simulation not found: ${planId}`);
    return plan;
  }

  private saveAndReturn(
    plan: DeploymentSimulationPlan,
    nodes: SimulationNode[],
  ): DeploymentSimulationPlan {
    this.save(plan, nodes);
    return plan;
  }

  private save(plan: DeploymentSimulationPlan, nodes: SimulationNode[]): void {
    const byId = new Map(nodes.map((node) => [node.node_id, node]));
    this.store.transaction(() => {
      for (const record of plan.records) {
        const persisted: PersistedRecord = {
          plan_id: plan.plan_id,
          plan_status: plan.status,
          plan_epoch: plan.epoch,
          required_capabilities: plan.required_capabilities,
          target_nodes: plan.target_nodes,
          actual_nodes: plan.actual_nodes,
          created_at: plan.created_at,
          updated_at: plan.updated_at,
          artifact_id: plan.artifact_id,
          runtime_profile: plan.runtime_profile,
          node_id: record.node_id,
          record,
          node: byId.get(record.node_id) ?? null,
        };
        this.store.prepare(
          `INSERT INTO deployments
             (deployment_id, artifact_id, node_id, status, epoch, payload, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(deployment_id) DO UPDATE SET
             artifact_id = excluded.artifact_id,
             status = excluded.status,
             epoch = excluded.epoch,
             payload = excluded.payload`,
        ).run(
          record.deployment_id, record.artifact_id, record.node_id,
          record.status, record.epoch, JSON.stringify(persisted), plan.created_at,
        );
      }
    });
  }
}

interface PersistedRecord {
  plan_id: string;
  plan_status: SimulationStatus;
  plan_epoch: number;
  required_capabilities: string[];
  target_nodes: string[];
  actual_nodes: string[];
  created_at: string;
  updated_at: string;
  artifact_id: string;
  runtime_profile: string;
  node_id: string;
  record: DeploymentRecord;
  node: SimulationNode | null;
}
