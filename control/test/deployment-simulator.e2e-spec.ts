import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { DeploymentSimulator, SimulationNode } from '../src/data/deployment-simulator';
import { SqliteStore } from '../src/data/sqlite-store';

const DIGEST = `sha256:${'a'.repeat(64)}`;

function nodes(): SimulationNode[] {
  return [
    {
      node_id: 'pc-ready', artifact_ids: [DIGEST],
      capabilities: ['llama_cpp'], available: true,
    },
    {
      node_id: 'pc-wrong-digest', artifact_ids: [`sha256:${'b'.repeat(64)}`],
      capabilities: ['llama_cpp'], available: true,
    },
    {
      node_id: 'pc-offline', artifact_ids: [DIGEST],
      capabilities: ['llama_cpp'], available: false,
    },
  ];
}

describe('MODEL-FLEET deployment simulator (M5.0)', () => {
  it('persists partial prepare, rejects wrong digest, activates an epoch, and rolls back after restart', () => {
    const dir = mkdtempSync(join(tmpdir(), 'qlh-mf-deploy-'));
    const sqlitePath = join(dir, 'control.sqlite3');
    let store = new SqliteStore(sqlitePath);
    store.open();
    const simulator = new DeploymentSimulator(store);
    const plan = simulator.createPlan({
      artifactId: DIGEST,
      nodes: nodes(),
      requiredCapabilities: ['llama_cpp'],
    });
    expect(plan.status).toBe('planned');
    expect(() => simulator.createPlan({ artifactId: 'wrong', nodes: nodes() })).toThrow('sha256');

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
    const restarted = new DeploymentSimulator(store);
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
});
