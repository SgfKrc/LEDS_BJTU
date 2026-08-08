import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { DeploymentSimulator, SimulationNode } from '../src/data/deployment-simulator';
import {
  ArtifactRuntimeRecord, ArtifactRuntimeRepository, ArtifactRuntimeStatus,
} from '../src/data/artifact-runtime-repository';
import { SqliteStore } from '../src/data/sqlite-store';

const DIGEST = `sha256:${'a'.repeat(64)}`;
const FINGERPRINT = `sha256:${'c'.repeat(64)}`;

function runtimeRecord(
  nodeId: string,
  status: ArtifactRuntimeStatus = 'ready',
  fingerprint = FINGERPRINT,
): ArtifactRuntimeRecord {
  return {
    schema_version: 1,
    artifact_id: DIGEST,
    node_id: nodeId,
    runtime_profile: 'llm-cpu-v1',
    status,
    checked_at: new Date().toISOString(),
    engine: 'llama_cpp',
    loader_version: 'fixture/1',
    runtime_fingerprint: fingerprint,
    load_ms: 1,
    details: {},
    error: status === 'ready' ? null : { code: 'fixture', message: status },
  };
}

function nodes(): SimulationNode[] {
  return [
    {
      node_id: 'pc-ready', artifact_ids: [DIGEST],
      capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT, available: true,
    },
    {
      node_id: 'pc-wrong-digest', artifact_ids: [`sha256:${'b'.repeat(64)}`],
      capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT, available: true,
    },
    {
      node_id: 'pc-offline', artifact_ids: [DIGEST],
      capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT, available: false,
    },
  ];
}

describe('MODEL-FLEET deployment simulator (M5.0)', () => {
  it('persists partial prepare, rejects wrong digest, activates an epoch, and rolls back after restart', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-deploy-'));
    const sqlitePath = join(dir, 'control.sqlite3');
    let store = new SqliteStore(sqlitePath);
    store.open();
    let runtime = new ArtifactRuntimeRepository(store);
    runtime.upsert(runtimeRecord('pc-ready'));
    const simulator = new DeploymentSimulator(store, runtime);
    const plan = simulator.createPlan({
      artifactId: DIGEST,
      runtimeProfile: 'llm-cpu-v1',
      nodes: nodes(),
      requiredCapabilities: ['llama_cpp'],
    });
    expect(plan.status).toBe('planned');
    expect(() => simulator.createPlan({
      artifactId: 'wrong', runtimeProfile: 'llm-cpu-v1', nodes: nodes(),
    })).toThrow('sha256');

    const prepared = simulator.prepare(plan.plan_id, nodes());
    expect(prepared.status).toBe('partial');
    expect(prepared.records.find((record) => record.node_id === 'pc-ready')?.status)
      .toBe('ready');
    expect(prepared.records.find((record) => record.node_id === 'pc-wrong-digest')?.error?.code)
      .toBe('digest_mismatch');
    expect(prepared.records.find((record) => record.node_id === 'pc-offline')?.error?.code)
      .toBe('node_unavailable');

    const active = simulator.activate(plan.plan_id);
    expect(active.status).toBe('partial');
    expect(active.actual_nodes).toEqual(['pc-ready']);
    expect(active.epoch).toBe(1);
    store.close();

    store = new SqliteStore(sqlitePath);
    store.open();
    runtime = new ArtifactRuntimeRepository(store);
    const restarted = new DeploymentSimulator(store, runtime);
    expect(restarted.get(plan.plan_id)?.actual_nodes).toEqual(['pc-ready']);
    const rolledBack = restarted.rollback(plan.plan_id);
    expect(rolledBack.status).toBe('rolled_back');
    expect(rolledBack.epoch).toBe(2);
    expect(rolledBack.actual_nodes).toEqual([]);
    expect(rolledBack.records.find((record) => record.node_id === 'pc-ready')?.status)
      .toBe('rolled_back');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('rejects unchecked, non-ready, and changed runtime contexts', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-runtime-filter-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const runtime = new ArtifactRuntimeRepository(store);
    runtime.upsert(runtimeRecord('ready'));
    runtime.upsert(runtimeRecord('stale', 'stale'));
    runtime.upsert(runtimeRecord('changed'));
    const simulator = new DeploymentSimulator(store, runtime);
    const testNodes: SimulationNode[] = [
      { node_id: 'ready', artifact_ids: [DIGEST], capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT },
      { node_id: 'stale', artifact_ids: [DIGEST], capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT },
      { node_id: 'changed', artifact_ids: [DIGEST], capabilities: ['llama_cpp'], runtime_fingerprint: `sha256:${'d'.repeat(64)}` },
      { node_id: 'unchecked', artifact_ids: [DIGEST], capabilities: ['llama_cpp'], runtime_fingerprint: FINGERPRINT },
    ];
    const plan = simulator.createPlan({
      artifactId: DIGEST,
      runtimeProfile: 'llm-cpu-v1',
      nodes: testNodes,
      requiredCapabilities: ['llama_cpp'],
    });
    const prepared = simulator.prepare(plan.plan_id, testNodes);
    expect(prepared.status).toBe('partial');
    expect(prepared.records.find((record) => record.node_id === 'ready')?.status).toBe('ready');
    expect(prepared.records.find((record) => record.node_id === 'stale')?.error?.code)
      .toBe('runtime_not_ready');
    expect(prepared.records.find((record) => record.node_id === 'changed')?.error?.code)
      .toBe('runtime_context_changed');
    expect(prepared.records.find((record) => record.node_id === 'unchecked')?.error?.code)
      .toBe('runtime_unchecked');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it('blocks activation when admission becomes stale after prepare', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-runtime-toctou-'));
    const store = new SqliteStore(join(dir, 'control.sqlite3'));
    store.open();
    const runtime = new ArtifactRuntimeRepository(store);
    runtime.upsert(runtimeRecord('pc-ready'));
    const simulator = new DeploymentSimulator(store, runtime);
    const onlyNode = [nodes()[0]];
    const plan = simulator.createPlan({
      artifactId: DIGEST,
      runtimeProfile: 'llm-cpu-v1',
      nodes: onlyNode,
      requiredCapabilities: ['llama_cpp'],
    });
    expect(simulator.prepare(plan.plan_id, onlyNode).status).toBe('ready');
    runtime.invalidate({ artifactId: DIGEST, nodeId: 'pc-ready' }, 'runtime upgraded');
    const activated = simulator.activate(plan.plan_id);
    expect(activated.status).toBe('failed');
    expect(activated.actual_nodes).toEqual([]);
    expect(activated.records[0].error?.code).toBe('runtime_admission_changed');
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });
});
