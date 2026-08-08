import { Injectable } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SqliteStore } from './sqlite-store';

export type SimulationStatus =
  | 'planned' | 'preparing' | 'distributing' | 'ready' | 'active'
  | 'failed' | 'partial' | 'rolled_back';

export interface SimulationNode {
  node_id: string;
  artifact_ids: string[];
  capabilities: string[];
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
  prepared_at: string | null;
  activated_at: string | null;
  error: { code: string; message: string } | null;
}

export interface DeploymentSimulationPlan {
  schema_version: 1;
  plan_id: string;
  artifact_id: string;
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
  constructor(private readonly store: SqliteStore) {}

  createPlan(input: {
    artifactId: string;
    nodes: SimulationNode[];
    requiredCapabilities?: string[];
  }): DeploymentSimulationPlan {
    if (!/^sha256:[0-9a-f]{64}$/.test(input.artifactId)) {
      throw new Error('artifact_id must be a sha256 digest');
    }
    if (input.nodes.length === 0) throw new Error('at least one node is required');
    const targetNodes = input.nodes.map((node) => node.node_id);
    if (new Set(targetNodes).size !== targetNodes.length) {
      throw new Error('node_id must be unique in a plan');
    }
    const now = new Date().toISOString();
    const plan: DeploymentSimulationPlan = {
      schema_version: 1,
      plan_id: `sim_${randomUUID().slice(0, 12)}`,
      artifact_id: input.artifactId,
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
      return {
        ...record,
        status: 'ready' as const,
        digest_verified: true,
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
    const ready = plan.records.filter((record) => record.status === 'ready');
    if (ready.length === 0) throw new Error('no prepared node can be activated');
    const now = new Date().toISOString();
    const epoch = plan.epoch + 1;
    const records = plan.records.map((record) => record.status === 'ready'
      ? { ...record, status: 'active' as const, epoch, activated_at: now }
      : record);
    return this.saveAndReturn({
      ...plan,
      records,
      actual_nodes: ready.map((record) => record.node_id),
      status: ready.length === records.length ? 'active' : 'partial',
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
  node_id: string;
  record: DeploymentRecord;
  node: SimulationNode | null;
}
