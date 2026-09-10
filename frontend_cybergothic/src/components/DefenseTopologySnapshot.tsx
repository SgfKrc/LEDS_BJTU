import { Layers3, Network, ShieldCheck } from 'lucide-react';
import topologySnapshot from '../data/defense-topology.json';
import { SectionHead } from './PageHeader';
import { StatusBadge } from './StatusBadge';

type SnapshotState = 'ready' | 'degraded' | 'offline';

interface SnapshotNode {
  alias: string;
  role: string;
  state: SnapshotState;
  heartbeat: string;
  summary: string;
}

interface LayerAssignment {
  node_alias: string;
  start_layer: number;
  end_layer_exclusive: number;
  layer_count: number;
  capabilities: string[];
}

interface DefenseTopologyData {
  snapshot_id: string;
  readiness: {
    state: SnapshotState;
    checks: Array<{ id: string; label: string; state: SnapshotState; evidence: string }>;
  };
  nodes: SnapshotNode[];
  layer_plan: {
    total_layers: number;
    strategy: string;
    assignments: LayerAssignment[];
  };
}

function toneFor(state: SnapshotState): 'ok' | 'warn' | 'idle' {
  if (state === 'ready') return 'ok';
  if (state === 'degraded') return 'warn';
  return 'idle';
}

/** Read-only, redacted defense fixture. It must never be presented as live telemetry. */
export function DefenseTopologySnapshot() {
  const snapshot = topologySnapshot as DefenseTopologyData;
  const totalLayers = snapshot.layer_plan.total_layers;
  const readyNodes = snapshot.nodes.filter((node) => node.state === 'ready').length;

  return (
    <section className="cluster-defense" data-testid="defense-topology-snapshot" aria-labelledby="defense-topology-title">
      <div className="cluster-defense__claim" role="note" aria-label="Snapshot claim boundary">
        <ShieldCheck size={18} aria-hidden="true" />
        <div>
          <strong>FIXTURE / REDACTED / NOT LIVE</strong>
          <span>Control-plane UI evidence only · no real model · no physical dual-host claim</span>
        </div>
      </div>

      <div className="cluster-defense__heading">
        <div>
          <span className="cluster-panel__eyebrow">DEF-P4 · {snapshot.snapshot_id}</span>
          <h2 id="defense-topology-title">Defense topology snapshot</h2>
          <p>Stable aliases expose readiness, node state, and an end-exclusive layer plan without hostnames, addresses, hardware, or credentials.</p>
        </div>
        <StatusBadge label={snapshot.readiness.state.toUpperCase()} tone="ok" pulse />
      </div>

      <div className="cluster-defense__summary" aria-label="Topology summary">
        <div><Network size={17} aria-hidden="true" /><span>ACTIVE NODES</span><strong>{readyNodes} / {snapshot.nodes.length}</strong></div>
        <div><Layers3 size={17} aria-hidden="true" /><span>LAYER COVERAGE</span><strong>0–{totalLayers - 1}</strong></div>
        <div><ShieldCheck size={17} aria-hidden="true" /><span>READINESS CHECKS</span><strong>{snapshot.readiness.checks.length} / {snapshot.readiness.checks.length}</strong></div>
      </div>

      <section className="cluster-defense__section" aria-labelledby="defense-readiness-title">
        <SectionHead title="Readiness snapshot" hint="fixed acceptance evidence" />
        <h3 id="defense-readiness-title" className="sr-only">Readiness snapshot checks</h3>
        <div className="cluster-defense__checks">
          {snapshot.readiness.checks.map((check) => (
            <div key={check.id} className="cluster-defense__check">
              <StatusBadge label={check.state.toUpperCase()} tone="ok" size="sm" />
              <div><strong>{check.label}</strong><span>{check.evidence}</span></div>
            </div>
          ))}
        </div>
      </section>

      <section className="cluster-defense__section" aria-label="Redacted node topology">
        <SectionHead title="Node states" hint="aliases only" />
        <div className="cluster-defense__topology">
          {snapshot.nodes.map((node, index) => (
            <div className="cluster-defense__node-wrap" key={node.alias}>
              {index > 0 ? <span className="cluster-defense__link" aria-hidden="true" /> : null}
              <article className="cluster-defense__node" data-state={node.state}>
                <span className="cluster-defense__node-dot" aria-hidden="true" />
                <div><strong>{node.alias}</strong><span>{node.role} · heartbeat {node.heartbeat}</span><small>{node.summary}</small></div>
                <StatusBadge label={node.state.toUpperCase()} tone={toneFor(node.state)} size="sm" />
              </article>
            </div>
          ))}
        </div>
      </section>

      <section className="cluster-defense__section" aria-label="Redacted layer allocation">
        <SectionHead title="Layer segments" hint={`${snapshot.layer_plan.strategy} · end-exclusive contract`} />
        <div className="cluster-defense__layer-bar" aria-hidden="true">
          {snapshot.layer_plan.assignments.map((assignment) => (
            <span
              key={assignment.node_alias}
              style={{ width: `${(assignment.layer_count / totalLayers) * 100}%` }}
              title={`${assignment.node_alias}: layers ${assignment.start_layer}-${assignment.end_layer_exclusive - 1}`}
            />
          ))}
        </div>
        <div className="cluster-defense__layers">
          {snapshot.layer_plan.assignments.map((assignment) => (
            <div key={assignment.node_alias} className="cluster-defense__layer-row">
              <div><strong>{assignment.node_alias}</strong><span>{assignment.capabilities.join(' + ')}</span></div>
              <div><strong>L{assignment.start_layer}–L{assignment.end_layer_exclusive - 1}</strong><span>{assignment.layer_count} layers</span></div>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

export const defenseTopologyCanvasNodes: Array<{ node_id: string; role: string; state: string; is_available: boolean }> = (
  (topologySnapshot as DefenseTopologyData).nodes
).map((node) => ({
  node_id: node.alias,
  role: node.role,
  state: node.state,
  is_available: node.state === 'ready',
}));
