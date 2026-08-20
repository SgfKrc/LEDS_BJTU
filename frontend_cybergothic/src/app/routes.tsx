/**
 * 路由表 — 首期为 hash 单页应用，保留后续接入真实 router 的能力（§4.3）。
 *
 * 使用 hash 而不是 history API：后端以静态文件方式托管 dist 时无需额外 rewrite 规则。
 */

import { useEffect, useState } from 'react';
import {
  Activity,
  Boxes,
  CircleHelp,
  Columns2,
  Image,
  LayoutDashboard,
  ListTree,
  SlidersHorizontal,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ComponentType } from 'react';
import { WorkbenchPage } from '../pages/WorkbenchPage';
import { OverviewPage } from '../pages/OverviewPage';
import { TasksPage } from '../pages/TasksPage';
import { ActivityPage } from '../pages/ActivityPage';
import { SettingsPage } from '../pages/SettingsPage';
import { HelpPage } from '../pages/HelpPage';
import { ImageStudioPage } from '../pages/ImageStudioPage';
import { ModelsPage } from '../pages/ModelsPage';

export type RouteId = 'workbench' | 'overview' | 'tasks' | 'activity' | 'image' | 'models' | 'settings' | 'help';

export interface RouteDef {
  id: RouteId;
  /** 导航中文标签（主导航不使用内部代号，§4.1）。 */
  label: string;
  /** 小号英文副标签，仅作品牌气质。 */
  tag: string;
  icon: LucideIcon;
  /** 页面标题下方的一句说明。 */
  description: string;
  component: ComponentType;
  /**
   * true 表示这一页自己管滚动、要占满视口高度（分屏工作台）。
   * AppShell 据此去掉外层内容边距，避免出现两条滚动条。
   */
  fullBleed?: boolean;
}

export const ROUTES: RouteDef[] = [
  {
    id: 'workbench',
    label: '工作台',
    tag: 'WORKBENCH',
    icon: Columns2,
    description: '左侧集群实况，右侧对话；比例可拖动调整。',
    component: WorkbenchPage,
    fullBleed: true,
  },
  {
    id: 'overview',
    label: '概览',
    tag: 'OVERVIEW',
    icon: LayoutDashboard,
    description: '集群当前状态、关键指标与最近活动。',
    component: OverviewPage,
  },
  {
    id: 'tasks',
    label: '任务',
    tag: 'TASKS',
    icon: ListTree,
    description: '推理队列、工作流与执行提供者。',
    component: TasksPage,
  },
  {
    id: 'activity',
    label: '活动',
    tag: 'ACTIVITY',
    icon: Activity,
    description: '运行事件时间线与错误详情。',
    component: ActivityPage,
  },
  {
    id: 'image',
    label: '生图',
    tag: 'IMAGE STUDIO',
    icon: Image,
    description: 'Stable Diffusion 资产、生成任务与结果列表。',
    component: ImageStudioPage,
  },
  {
    id: 'models',
    label: 'Models',
    tag: 'MODEL LAB',
    icon: Boxes,
    description: 'Model runtime, local assets, and load controls.',
    component: ModelsPage,
  },
  {
    id: 'settings',
    label: '设置',
    tag: 'SETTINGS',
    icon: SlidersHorizontal,
    description: '连接、动效偏好与数据来源。',
    component: SettingsPage,
  },
  {
    id: 'help',
    label: '帮助',
    tag: 'HELP',
    icon: CircleHelp,
    description: '使用说明、接口契约与版本信息。',
    component: HelpPage,
  },
];

const DEFAULT_ROUTE: RouteId = 'workbench';

/**
 * 解析当前 hash。
 *
 * 返回 null 表示「这个 hash 不是路由」——例如跳过链接的 `#main` 锚点。
 * 此时必须保留当前页面，否则在任务页按「跳到主内容」会被弹回概览。
 */
function parseHash(): RouteId | null {
  const hash = window.location.hash;
  if (!hash || hash === '#') return DEFAULT_ROUTE;
  if (!hash.startsWith('#/')) return null;
  const raw = hash.slice(2).split('?')[0] ?? '';
  if (raw === '') return DEFAULT_ROUTE;
  const match = ROUTES.find((r) => r.id === raw);
  return match ? match.id : DEFAULT_ROUTE;
}

/** hash 上附带的查询串（如 `?fixtures=1`），跳转时需要带过去。 */
function hashQuery(): string {
  const q = window.location.hash.indexOf('?');
  return q === -1 ? '' : window.location.hash.slice(q);
}

/** 当前路由 + 跳转函数。 */
export function useRoute(): [RouteDef, (id: RouteId) => void] {
  const [id, setId] = useState<RouteId>(() =>
    typeof window === 'undefined' ? DEFAULT_ROUTE : (parseHash() ?? DEFAULT_ROUTE),
  );

  useEffect(() => {
    const onHashChange = () => {
      const next = parseHash();
      if (next !== null) setId(next);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (next: RouteId) => {
    if (next === id) return;
    window.location.hash = `#/${next}${hashQuery()}`;
  };

  const route = ROUTES.find((r) => r.id === id) ?? ROUTES[0]!;
  return [route, navigate];
}

/** 页内链接的 href；保留当前 hash 上的查询串（例如 fixtures 开关）。 */
export function routeHref(id: RouteId): string {
  const query = typeof window === 'undefined' ? '' : hashQuery();
  return `#/${id}${query}`;
}
