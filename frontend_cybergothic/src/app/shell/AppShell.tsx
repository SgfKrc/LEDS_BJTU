/**
 * 应用外壳 — 统一背景、顶栏、内容最大宽度和移动端安全区（§5.3）。
 */

import { useCallback, useEffect, useState } from 'react';
import { Topbar } from './Topbar';
import { MobileNav } from './MobileNav';
import { GrainOverlay } from '../../visual/GrainOverlay';
import { ToastHost } from '../../components/Toast';
import { triggerRefresh } from '../refreshBus';
import { useRoute } from '../routes';
import { fetchHealth } from '../../data/api';
import { fixturesEnabled } from '../../data/fixtures';
import { APP_VERSION } from '../version';

export function AppShell() {
  const [route, navigate] = useRoute();
  const [menuOpen, setMenuOpen] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const usingFixtures = fixturesEnabled();

  // 路由切换后回到顶部，并把焦点交给主内容区（键盘可达性）。
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
    document.title = `${route.label} — QLH 控制台`;
  }, [route.id, route.label]);

  // 后端连通性探测：60s 一次，页面隐藏时跳过。
  useEffect(() => {
    if (usingFixtures) {
      setOnline(true);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    const probe = async () => {
      if (document.visibilityState === 'hidden') return;
      try {
        await fetchHealth(controller.signal);
        if (!cancelled) setOnline(true);
      } catch {
        if (!cancelled) setOnline(false);
      }
    };

    void probe();
    const timer = window.setInterval(probe, 60_000);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, [usingFixtures]);

  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    triggerRefresh();
    // 刷新指示只是反馈，真实完成时间由各页面的 loading 态体现。
    window.setTimeout(() => setRefreshing(false), 600);
  }, []);

  const Page = route.component;

  return (
    <div className={`shell${route.fullBleed ? ' shell--bleed' : ''}`}>
      <GrainOverlay />
      <div className="shell__diagonals" aria-hidden="true" />

      <a className="skip-link" href="#main">
        跳到主内容
      </a>

      <Topbar
        current={route}
        onNavigate={navigate}
        onOpenMenu={() => setMenuOpen(true)}
        onRefresh={handleRefresh}
        refreshing={refreshing}
        online={online}
      />

      <MobileNav
        open={menuOpen}
        currentId={route.id}
        onClose={() => setMenuOpen(false)}
        onNavigate={navigate}
      />

      {usingFixtures ? (
        <div className="fixturebar" role="status">
          <span className="mono-label">演示数据模式</span>
          <span>当前显示本地 fixture，不代表真实集群状态。可在设置页关闭。</span>
        </div>
      ) : null}

      {online === false && !usingFixtures ? (
        <div className="fixturebar fixturebar--warn" role="status">
          <span className="mono-label">离线</span>
          <span>
            无法连接后端 <code>/api</code>。请确认 FastAPI 已启动，或在设置页启用演示数据。
          </span>
        </div>
      ) : null}

      <main
        className={`shell__main${route.fullBleed ? ' shell__main--bleed' : ''}`}
        id="main"
        tabIndex={-1}
      >
        {route.fullBleed ? (
          <Page />
        ) : (
          <div className="shell__content">
            <Page />
          </div>
        )}
      </main>

      {/* 分屏页自己占满剩余高度，页脚会挤掉可用空间，所以不渲染 */}
      {route.fullBleed ? null : (
        <footer className="shell__footer">
          <div className="shell__content shell__footer-inner">
            <span className="mono-label">QLH EDGE INFERENCE · CYBERGOTHIC</span>
            <span className="shell__footer-meta">
              版本 {APP_VERSION} · 数据来源 {usingFixtures ? '本地 fixture' : '/api'}
            </span>
          </div>
        </footer>
      )}

      <ToastHost />
    </div>
  );
}
