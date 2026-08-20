/**
 * 设置 — 连接信息、动效偏好、演示数据与日志令牌。
 *
 * 所有偏好都写入 localStorage，刷新后保持；动效偏好同时反映到 <html data-reduced-motion>。
 */

import { useCallback, useEffect, useState } from 'react';
import { Check, Eye, EyeOff, RotateCcw, Save } from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { CommandButton } from '../components/CommandButton';
import { StatusBadge } from '../components/StatusBadge';
import { EmptyState } from '../components/EmptyState';
import { pushToast } from '../components/Toast';
import { useReveal } from '../motion/useReveal';
import { useRegisterRefresh } from '../app/refreshBus';
import {
  useMotionPreference,
  type MotionPreference,
} from '../motion/useReducedMotion';
import { fixturesEnabled, setFixturesEnabled } from '../data/fixtures';
import { getAuthToken, getLogToken, setLogToken } from '../data/api';
import { useMyRole, useSystemStatus } from '../data/hooks';
import { APP_VERSION } from '../app/version';

const MOTION_OPTIONS: Array<{ id: MotionPreference; label: string; hint: string }> = [
  { id: 'system', label: '跟随系统', hint: '使用操作系统的「减少动态效果」设置。' },
  { id: 'full', label: '完整动效', hint: '入场、揭示与背景动画全部启用。' },
  { id: 'reduced', label: '减少动效', hint: '只保留必要的状态变化，去掉位移与脉冲。' },
];

export function SettingsPage() {
  const status = useSystemStatus(30_000);
  const role = useMyRole();

  const [motion, setMotion] = useMotionPreference();
  const [useFixtures, setUseFixtures] = useState(() => fixturesEnabled());
  const [logToken, setLogTokenInput] = useState(() => getLogToken());
  const [tokenVisible, setTokenVisible] = useState(false);
  const [savedToken, setSavedToken] = useState(() => getLogToken());

  const refresh = useCallback(() => {
    status.refresh();
    role.refresh();
  }, [status.refresh, role.refresh]);
  useRegisterRefresh(refresh);

  useReveal([status.data, role.data]);

  // 键入令牌后未保存时给出提示，避免以为已生效。
  const tokenDirty = logToken !== savedToken;

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
    // URL 上的 ?fixtures= 优先级高于本开关，切换时一并清掉，否则设置看起来「没生效」。
    const url = new URL(window.location.href);
    url.searchParams.delete('fixtures');
    // fixture 开关影响所有 hook 的取数分支，整页重载最省心也最可预期。
    window.setTimeout(() => window.location.replace(url.toString()), 400);
  }, []);

  const apiBase = `${window.location.origin}/api`;

  return (
    <>
      <PageHeader
        tag="SETTINGS"
        title="设置"
        description="控制台自身的偏好项。这些设置只保存在当前浏览器，不会写入后端配置。"
      />

      <section className="band" data-reveal>
        <SectionHead title="连接" hint="当前控制台连接的后端与本机在集群中的角色。" />
        <dl className="kvgrid">
          <div>
            <dt>API 地址</dt>
            <dd className="cell-mono">{apiBase}</dd>
          </div>
          <div>
            <dt>运行模式</dt>
            <dd>
              <StatusBadge
                state={status.data?.run_mode}
                label={status.data?.run_mode === 'distributed' ? '分布式' : status.data?.run_mode}
                tone={status.data?.run_mode === 'distributed' ? 'ok' : 'info'}
                size="sm"
              />
            </dd>
          </div>
          <div>
            <dt>本机角色</dt>
            <dd>
              <StatusBadge state={role.data?.node_role} size="sm" />
            </dd>
          </div>
          <div>
            <dt>节点 ID</dt>
            <dd className="cell-mono">{status.data?.node_id || role.data?.node_id || '—'}</dd>
          </div>
          <div>
            <dt>登录令牌</dt>
            <dd>
              <StatusBadge
                label={getAuthToken() ? '已登录' : '未登录'}
                tone={getAuthToken() ? 'ok' : 'idle'}
                size="sm"
              />
            </dd>
          </div>
          <div>
            <dt>控制台版本</dt>
            <dd className="cell-mono">v{APP_VERSION}</dd>
          </div>
        </dl>
        {status.state === 'error' ? (
          <EmptyState
            kind="error"
            title="无法读取后端状态"
            detail={status.error}
            compact
            action={
              <CommandButton variant="ghost" size="sm" icon={RotateCcw} onClick={refresh}>
                重试
              </CommandButton>
            }
          />
        ) : null}
      </section>

      <section className="band band--alt" data-reveal>
        <SectionHead
          title="动效"
          hint="减少动效会关闭入场位移、揭示动画与背景 Canvas 的持续绘制。"
        />
        <fieldset className="optionset">
          <legend className="sr-only">动效偏好</legend>
          {MOTION_OPTIONS.map((opt) => (
            <label className="option" key={opt.id} data-checked={motion === opt.id ? 'true' : undefined}>
              <input
                type="radio"
                name="motion-preference"
                value={opt.id}
                checked={motion === opt.id}
                onChange={() => {
                  setMotion(opt.id);
                  pushToast(`动效偏好：${opt.label}`, 'ok');
                }}
              />
              <span className="option__mark" aria-hidden="true">
                <Check size={14} strokeWidth={3} aria-hidden />
              </span>
              <span className="option__text">
                <span className="option__label">{opt.label}</span>
                <span className="option__hint">{opt.hint}</span>
              </span>
            </label>
          ))}
        </fieldset>
      </section>

      <section className="band" data-reveal>
        <SectionHead
          title="演示数据"
          hint="用内置样例替代真实接口，便于在没有后端时演示界面与各类状态。"
        />
        <div className="switchrow">
          <label className="switch">
            <input
              type="checkbox"
              checked={useFixtures}
              onChange={(e) => toggleFixtures(e.target.checked)}
            />
            <span className="switch__track" aria-hidden="true">
              <span className="switch__thumb" />
            </span>
            <span className="switch__label">使用演示数据</span>
          </label>
          <p className="switchrow__hint">
            开启后所有页面读取本地 fixture，写操作被拦截。也可以在网址后加
            <code> ?fixtures=1 </code>
            临时开启。
          </p>
        </div>
      </section>

      <section className="band band--alt" data-reveal>
        <SectionHead
          title="日志令牌"
          hint="后端若启用了日志接口保护，需要填入 X-QLH-Log-Token 才能读取活动页的日志。"
        />
        <form
          className="tokenform"
          onSubmit={(e) => {
            e.preventDefault();
            saveToken();
          }}
        >
          <div className="field">
            <label className="field__label" htmlFor="log-token">
              管理令牌
            </label>
            <div className="field__row">
              <input
                id="log-token"
                className="field__input cell-mono"
                type={tokenVisible ? 'text' : 'password'}
                value={logToken}
                autoComplete="off"
                spellCheck={false}
                placeholder="留空表示不发送该请求头"
                onChange={(e) => setLogTokenInput(e.target.value)}
                aria-describedby="log-token-hint"
              />
              <CommandButton
                variant="ghost"
                size="sm"
                icon={tokenVisible ? EyeOff : Eye}
                ariaLabel={tokenVisible ? '隐藏令牌' : '显示令牌'}
                onClick={() => setTokenVisible((v) => !v)}
              >
                {tokenVisible ? '隐藏' : '显示'}
              </CommandButton>
            </div>
            <p className="field__hint" id="log-token-hint">
              令牌保存在浏览器 localStorage，与既有前端共用同一个键。
            </p>
          </div>
          <div className="tokenform__actions">
            <CommandButton icon={Save} size="sm" type="submit">
              保存令牌
            </CommandButton>
            {logToken ? (
              <CommandButton
                variant="ghost"
                size="sm"
                onClick={() => {
                  setLogTokenInput('');
                  setLogToken('');
                  setSavedToken('');
                  pushToast('日志令牌已清除。', 'ok');
                }}
              >
                清除
              </CommandButton>
            ) : null}
            {tokenDirty ? (
              <span className="field__dirty" role="status">
                有未保存的修改
              </span>
            ) : null}
          </div>
        </form>
      </section>
    </>
  );
}
