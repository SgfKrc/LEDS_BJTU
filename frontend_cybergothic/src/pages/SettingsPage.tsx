/** Settings workspace: navigation rail, a locally scrolling editor, and persistent runtime context. */

import { useCallback, useEffect, useState } from 'react';
import {
  Check,
  Cpu,
  Database,
  Eye,
  EyeOff,
  KeyRound,
  Network,
  Power,
  RefreshCw,
  Save,
  SlidersHorizontal,
} from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { CommandButton } from '../components/CommandButton';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/EmptyState';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import {
  useMotionPreference,
  type MotionPreference,
} from '../motion/useReducedMotion';
import { fixturesEnabled, setFixturesEnabled } from '../data/fixtures';
import { getAuthToken, getLogToken, setLogToken } from '../data/api';
import * as api from '../data/api';
import { useMyRole, useSystemStatus } from '../data/hooks';
import { APP_VERSION } from '../app/version';
import { SettingsWorkspace, type SettingsWorkspaceTab } from '../components/SettingsWorkspace';
import { PageBackdrop } from '../visual/PageBackdrop';

const MOTION_OPTIONS: Array<{ id: MotionPreference; label: string; hint: string }> = [
  { id: 'system', label: '跟随系统', hint: '使用操作系统的“减少动态效果”设置。' },
  { id: 'full', label: '完整动效', hint: '启用入场、揭示与背景 Canvas 动画。' },
  { id: 'reduced', label: '减少动效', hint: '只保留必要状态变化，停止位移与持续绘制。' },
];

const SETTINGS_SECTIONS = [
  { id: 'device', label: '设备配置', short: 'RUNTIME', icon: Cpu },
  { id: 'rag', label: 'Local RAG', short: 'INDEX', icon: Database },
  { id: 'connection', label: '连接状态', short: 'LINK', icon: Network },
  { id: 'preferences', label: '动效偏好', short: 'MOTION', icon: SlidersHorizontal },
  { id: 'fixtures', label: '演示数据', short: 'SOURCE', icon: Database },
  { id: 'token', label: '日志令牌', short: 'ACCESS', icon: KeyRound },
] as const;

type SettingsSection = (typeof SETTINGS_SECTIONS)[number]['id'];

export function SettingsPage() {
  const status = useSystemStatus(30_000);
  const role = useMyRole();
  const [section, setSection] = useState<SettingsSection>('device');
  const [motion, setMotion] = useMotionPreference();
  const [useFixtures, setUseFixtures] = useState(() => fixturesEnabled());
  const [logToken, setLogTokenInput] = useState(() => getLogToken());
  const [tokenVisible, setTokenVisible] = useState(false);
  const [savedToken, setSavedToken] = useState(() => getLogToken());
  const [shutdownBusy, setShutdownBusy] = useState(false);

  const refresh = useCallback(() => {
    status.refresh();
    role.refresh();
  }, [role, status]);
  useRegisterRefresh(refresh);

  const tokenDirty = logToken !== savedToken;
  const current = SETTINGS_SECTIONS.find((item) => item.id === section) ?? SETTINGS_SECTIONS[0];
  const apiBase = `${window.location.origin}/api`;

  useEffect(() => {
    setSavedToken(getLogToken());
  }, []);

  const saveToken = useCallback(() => {
    setLogToken(logToken.trim());
    const now = getLogToken();
    setSavedToken(now);
    setLogTokenInput(now);
    pushToast(now ? '日志令牌已保存。' : '日志令牌已清除。', 'ok');
  }, [logToken]);

  const toggleFixtures = useCallback((next: boolean) => {
    setUseFixtures(next);
    setFixturesEnabled(next);
    pushToast(
      next ? '已切换到演示数据，页面将重新加载。' : '已切回真实后端数据，页面将重新加载。',
      'info',
    );
    const url = new URL(window.location.href);
    url.searchParams.delete('fixtures');
    window.setTimeout(() => window.location.replace(url.toString()), 400);
  }, []);

  const shutdownBackend = useCallback(async () => {
    if (shutdownBusy || useFixtures || !window.confirm('Stop the QLH backend on this node?')) return;
    setShutdownBusy(true);
    try {
      await api.shutdownSystem('settings_operator');
      pushToast('Backend shutdown requested', 'warn');
    } catch (error) {
      pushToast(`Shutdown request failed: ${api.describeError(error)}`, 'danger');
      setShutdownBusy(false);
    }
  }, [shutdownBusy, useFixtures]);

  const selectWorkspace = useCallback((next: SettingsWorkspaceTab) => {
    setSection(next);
  }, []);

  return (
    <div className="settings-page" data-testid="settings-page">
      <PageBackdrop scene="settings" className="settings-page__bg" />
      <PageHeader
        tag="SETTINGS"
        title="设置"
        description="设备、索引、连接与本地偏好分置于独立工作区；每个区域只在自身面板内滚动。"
        actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={status.refreshing || role.refreshing} onClick={refresh}>刷新</CommandButton>}
      />

      <div className="settings-layout">
        <aside className="settings-rail" aria-label="设置导航">
          <section className="settings-panel settings-identity">
            <span className="mono-label">CONSOLE ARCHIVE</span>
            <strong>{status.data?.node_id || role.data?.node_id || 'checking runtime'}</strong>
            <span>{useFixtures ? 'fixture source enabled' : 'backend source enabled'}</span>
          </section>

          <nav className="settings-nav" aria-label="设置领域">
            {SETTINGS_SECTIONS.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  type="button"
                  data-active={section === item.id ? 'true' : undefined}
                  aria-pressed={section === item.id}
                  onClick={() => setSection(item.id)}
                >
                  <Icon size={15} aria-hidden="true" />
                  <span>{item.label}</span>
                  <em>{item.short}</em>
                </button>
              );
            })}
          </nav>

          <section className="settings-panel settings-rail-note">
            <span className="mono-label">PERSISTENCE</span>
            <p>浏览器本地保存动效、数据源和日志访问令牌。</p>
          </section>
        </aside>

        <main className="settings-main">
          <section className="settings-panel settings-current" data-section={section} aria-labelledby="settings-current-title">
            <div className="settings-current__head">
              <SectionHead
                id="settings-current-title"
                title={current.label}
                hint={section === 'device' ? '硬件检测、GPU 选择与推荐运行配置。' : section === 'rag' ? '检索本地知识索引并维护 FTS 数据。' : section === 'connection' ? '当前控制台与本机节点之间的连接上下文。' : section === 'preferences' ? '为全部页面设置动效强度。' : section === 'fixtures' ? '在演示数据与真实后端之间切换。' : '为受保护的日志与审计接口设置访问令牌。'}
              />
            </div>

            <div className="settings-current__scroll">
              {section === 'device' || section === 'rag' ? (
                <SettingsWorkspace tab={section} onTabChange={selectWorkspace} showNavigation={false} />
              ) : null}

              {section === 'connection' ? (
                <>
                  <dl className="kvgrid settings-kvgrid">
                    <div><dt>API 地址</dt><dd className="cell-mono">{apiBase}</dd></div>
                    <div><dt>运行模式</dt><dd><StatusBadge state={status.data?.run_mode} label={status.data?.run_mode === 'distributed' ? '分布式' : status.data?.run_mode} tone={status.data?.run_mode === 'distributed' ? 'ok' : 'info'} size="sm" /></dd></div>
                    <div><dt>本机角色</dt><dd><StatusBadge state={role.data?.node_role} size="sm" /></dd></div>
                    <div><dt>节点 ID</dt><dd className="cell-mono">{status.data?.node_id || role.data?.node_id || '—'}</dd></div>
                    <div><dt>登录令牌</dt><dd><StatusBadge label={getAuthToken() ? '已登录' : '未登录'} tone={getAuthToken() ? 'ok' : 'idle'} size="sm" /></dd></div>
                    <div><dt>控制台版本</dt><dd className="cell-mono">v{APP_VERSION}</dd></div>
                  </dl>
                  {status.state === 'error' ? <EmptyState kind="error" title="无法读取后端状态" detail={status.error} errorKind={status.errorKind} errorStatus={status.errorStatus} action={<CommandButton variant="ghost" size="sm" icon={RefreshCw} onClick={refresh}>重试</CommandButton>} /> : null}
                  {role.data?.is_master ? <div className="settings-danger-zone"><SectionHead title="Backend lifecycle" hint="Master-only operator action" /><CommandButton variant="danger" size="sm" icon={Power} busy={shutdownBusy} onClick={() => void shutdownBackend()}>Shutdown backend</CommandButton></div> : null}
                </>
              ) : null}

              {section === 'preferences' ? (
                <fieldset className="optionset" id="settings-preferences">
                  <legend className="sr-only">动效偏好</legend>
                  {MOTION_OPTIONS.map((opt) => (
                    <label className="option" key={opt.id} data-checked={motion === opt.id ? 'true' : undefined}>
                      <input type="radio" name="motion-preference" value={opt.id} checked={motion === opt.id} onChange={() => { setMotion(opt.id); pushToast(`动效偏好：${opt.label}`, 'ok'); }} />
                      <span className="option__mark" aria-hidden="true"><Check size={14} strokeWidth={3} aria-hidden="true" /></span>
                      <span className="option__text"><span className="option__label">{opt.label}</span><span className="option__hint">{opt.hint}</span></span>
                    </label>
                  ))}
                </fieldset>
              ) : null}

              {section === 'fixtures' ? (
                <div className="settings-source">
                  <div className="switchrow">
                    <label className="switch">
                      <input type="checkbox" checked={useFixtures} onChange={(event) => toggleFixtures(event.target.checked)} />
                      <span className="switch__track" aria-hidden="true"><span className="switch__thumb" /></span>
                      <span className="switch__label">使用演示数据</span>
                    </label>
                    <p className="switchrow__hint">启用后页面读取本地 fixture，写操作会被拦截。网址中也可使用 <code>?fixtures=1</code> 临时开启。</p>
                  </div>
                </div>
              ) : null}

              {section === 'token' ? (
                <form className="tokenform" onSubmit={(event) => { event.preventDefault(); saveToken(); }}>
                  <div className="field">
                    <label className="field__label" htmlFor="log-token">管理令牌</label>
                    <div className="field__row">
                      <input id="log-token" className="field__input cell-mono" type={tokenVisible ? 'text' : 'password'} value={logToken} autoComplete="off" spellCheck={false} placeholder="留空表示不发送该请求头" onChange={(event) => setLogTokenInput(event.target.value)} aria-describedby="log-token-hint" />
                      <CommandButton variant="ghost" size="sm" icon={tokenVisible ? EyeOff : Eye} ariaLabel={tokenVisible ? '隐藏令牌' : '显示令牌'} onClick={() => setTokenVisible((value) => !value)}>{tokenVisible ? '隐藏' : '显示'}</CommandButton>
                    </div>
                    <p className="field__hint" id="log-token-hint">令牌只保存在浏览器 localStorage，并和日志、审计页面共用同一键。</p>
                  </div>
                  <div className="tokenform__actions">
                    <CommandButton icon={Save} size="sm" type="submit">保存令牌</CommandButton>
                    {logToken ? <CommandButton variant="ghost" size="sm" onClick={() => { setLogTokenInput(''); setLogToken(''); setSavedToken(''); pushToast('日志令牌已清除。', 'ok'); }}>清除</CommandButton> : null}
                    {tokenDirty ? <span className="field__dirty" role="status">有未保存的修改</span> : null}
                  </div>
                </form>
              ) : null}
            </div>
          </section>
        </main>

        <aside className="settings-details" aria-label="当前设置状态">
          <section className="settings-panel settings-detail-panel">
            <SectionHead title="运行上下文" hint={current.short} />
            <dl className="kvlist">
              <div><dt>当前领域</dt><dd>{current.label}</dd></div>
              <div><dt>数据源</dt><dd><StatusBadge label={useFixtures ? 'FIXTURE' : 'BACKEND'} tone={useFixtures ? 'warn' : 'ok'} size="sm" /></dd></div>
              <div><dt>运行模式</dt><dd><StatusBadge state={status.data?.run_mode} size="sm" /></dd></div>
              <div><dt>节点角色</dt><dd><StatusBadge state={role.data?.node_role} size="sm" /></dd></div>
              <div><dt>日志令牌</dt><dd><StatusBadge label={getLogToken() ? 'STORED' : 'EMPTY'} tone={getLogToken() ? 'ok' : 'idle'} size="sm" /></dd></div>
            </dl>
            {tokenDirty ? <p className="settings-unsaved" role="status"><KeyRound size={14} aria-hidden="true" />日志令牌尚未保存</p> : <p className="settings-saved"><Check size={14} aria-hidden="true" />本地偏好已同步</p>}
          </section>
        </aside>
      </div>
    </div>
  );
}
