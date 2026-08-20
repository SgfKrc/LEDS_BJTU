/**
 * 顶栏 — 品牌、页面导航、全局刷新与动效开关（§5.3）。
 *
 * 高度 60px（§4.4）；当前路由有明确选中态；窄屏隐藏次要链接并交给 MobileNav。
 */

import { Menu, RefreshCw, Zap } from 'lucide-react';
import { ROUTES, routeHref, type RouteDef, type RouteId } from '../routes';
import { useMotionPreference } from '../../motion/useReducedMotion';

interface TopbarProps {
  current: RouteDef;
  onNavigate: (id: RouteId) => void;
  onOpenMenu: () => void;
  /** 全局刷新：由 AppShell 广播给各页面。 */
  onRefresh: () => void;
  refreshing: boolean;
  /** 后端连通性，用于顶栏右侧的连接指示。 */
  online: boolean | null;
}

export function Topbar({
  current,
  onNavigate,
  onOpenMenu,
  onRefresh,
  refreshing,
  online,
}: TopbarProps) {
  const [motionPref, setMotionPref] = useMotionPreference();
  const motionReduced = motionPref === 'reduced';

  const connectionLabel =
    online === null ? '检测中' : online ? '已连接' : '未连接';
  const connectionTone = online === null ? 'idle' : online ? 'ok' : 'danger';

  return (
    <header className="topbar">
      <div className="topbar__inner">
        <a className="brand" href={routeHref('overview')} aria-label="QLH 控制台 首页">
          <span className="brand__mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="22" height="22">
              <path d="M7 24 L16 6 L25 24 Z" fill="none" stroke="currentColor" strokeWidth="2.5" />
            </svg>
          </span>
          <span className="brand__text">
            <span className="brand__name">QLH</span>
            <span className="brand__sub">控制台</span>
          </span>
        </a>

        <nav className="topnav" aria-label="主导航">
          <ul className="topnav__list">
            {ROUTES.map((route) => {
              const isActive = route.id === current.id;
              return (
                <li key={route.id}>
                  <a
                    className="topnav__link"
                    href={routeHref(route.id)}
                    aria-current={isActive ? 'page' : undefined}
                    data-active={isActive ? 'true' : undefined}
                    onClick={(e) => {
                      // 保留原生 a 语义（可中键/新窗口打开），左键走内部导航。
                      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                      e.preventDefault();
                      onNavigate(route.id);
                    }}
                  >
                    {route.label}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="topbar__actions">
          <span className={`conn conn--${connectionTone}`} title={`后端 ${connectionLabel}`}>
            <span className="conn__dot" aria-hidden="true" />
            <span className="conn__text">{connectionLabel}</span>
          </span>

          <button
            type="button"
            className="iconbtn"
            onClick={() => setMotionPref(motionReduced ? 'full' : 'reduced')}
            aria-pressed={motionReduced}
            title={motionReduced ? '动效已减少，点击恢复' : '减少动效'}
          >
            <Zap size={16} strokeWidth={2.25} />
            <span className="sr-only">{motionReduced ? '恢复动效' : '减少动效'}</span>
          </button>

          <button
            type="button"
            className="iconbtn"
            onClick={onRefresh}
            data-spinning={refreshing ? 'true' : undefined}
            title="刷新当前页数据"
          >
            <RefreshCw size={16} strokeWidth={2.25} />
            <span className="sr-only">刷新</span>
          </button>

          <button
            type="button"
            className="iconbtn iconbtn--menu"
            onClick={onOpenMenu}
            aria-label="打开菜单"
          >
            <Menu size={18} strokeWidth={2.25} />
          </button>
        </div>
      </div>
    </header>
  );
}
