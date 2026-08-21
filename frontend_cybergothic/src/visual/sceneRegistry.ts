import type { ComponentType } from 'react';
import { ArchiveClockCanvas } from './ArchiveClockCanvas';
import { BellTowerRainCanvas } from './BellTowerRainCanvas';
import { ClusterConstellationCanvas } from './ClusterConstellationCanvas';
import { ClockworkArchiveCanvas } from './ClockworkArchiveCanvas';
import { FoundryCanvas } from './FoundryCanvas';
import { GearworksQueueCanvas } from './GearworksQueueCanvas';
import { GothicWorksCanvas } from './GothicWorksCanvas';
import { IronGateCanvas } from './IronGateCanvas';
import { ModelOrbitCanvas } from './ModelOrbitCanvas';
import { ObservatoryNaveCanvas } from './ObservatoryNaveCanvas';
import { StainedGlassScriptoriumCanvas } from './StainedGlassScriptoriumCanvas';

export type SceneId = 'workbench' | 'overview' | 'tasks' | 'activity' | 'image' | 'models' | 'cluster' | 'account' | 'audit' | 'settings' | 'help';
export type SceneComponent = ComponentType<{ className?: string }>;

export interface SceneDefinition {
  id: SceneId;
  label: string;
  component: SceneComponent;
  tone: 'violet' | 'gold' | 'cyan' | 'rose' | 'green';
}

export const SCENE_REGISTRY: Record<SceneId, SceneDefinition> = {
  workbench: { id: 'workbench', label: 'Cathedral Works', component: GothicWorksCanvas, tone: 'violet' },
  overview: { id: 'overview', label: 'Observatory Nave', component: ObservatoryNaveCanvas, tone: 'cyan' },
  tasks: { id: 'tasks', label: 'Gearworks Queue', component: GearworksQueueCanvas, tone: 'gold' },
  activity: { id: 'activity', label: 'Bell Tower Rain', component: BellTowerRainCanvas, tone: 'violet' },
  image: { id: 'image', label: 'Alchemical Foundry', component: FoundryCanvas, tone: 'rose' },
  models: { id: 'models', label: 'Reliquary Engine', component: ModelOrbitCanvas, tone: 'green' },
  cluster: { id: 'cluster', label: 'Gargoyle Relay', component: ClusterConstellationCanvas, tone: 'cyan' },
  account: { id: 'account', label: 'Iron Gate', component: IronGateCanvas, tone: 'gold' },
  audit: { id: 'audit', label: 'Archive Clock', component: ArchiveClockCanvas, tone: 'gold' },
  settings: { id: 'settings', label: 'Clockwork Archive', component: ClockworkArchiveCanvas, tone: 'violet' },
  help: { id: 'help', label: 'Stained-glass Scriptorium', component: StainedGlassScriptoriumCanvas, tone: 'cyan' },
};
