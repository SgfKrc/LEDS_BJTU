import type { SceneId } from './sceneRegistry';
import { SCENE_REGISTRY } from './sceneRegistry';

interface PageBackdropProps {
  scene: SceneId;
  className?: string;
}

/** Single scene entry point for page-specific Canvas backdrops. */
export function PageBackdrop({ scene, className = '' }: PageBackdropProps) {
  const definition = SCENE_REGISTRY[scene];
  if (!definition) return null;
  const Scene = definition.component;
  return <Scene className={className} />;
}
