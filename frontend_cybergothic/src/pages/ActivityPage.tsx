/** Activity workspace: live logs, saved sessions, and a persistent detail context. */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { KeyRound, RefreshCw } from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { ActivityTimeline, type TimelineItem } from '../components/ActivityTimeline';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { CommandButton } from '../components/CommandButton';
import { StatusBadge } from '../components/StatusBadge';
import { Drawer } from '../components/Drawer';
import { useRegisterRefresh } from '../app/refreshBus';
import { useReveal } from '../motion/useReveal';
import { toneForLogLevel } from '../components/statusTone';
import { useRecentLogs, useSessions } from '../data/hooks';
import { getLogToken } from '../data/api';
import { routeHref } from '../app/routes';
import type { LogEntry, SessionSummary } from '../data/types';
import { PageBackdrop } from '../visual/PageBackdrop';

const LEVELS = [
  { id: '', label: '全部' },
  { id: 'INFO', label: 'INFO+' },
  { id: 'WARNING', label: 'WARN+' },
  { id: 'ERROR', label: 'ERROR+' },
] as const;

type LevelId = (typeof LEVELS)[number]['id'];
const VIRTUAL_THRESHOLD = 100;
const VISIBLE_STEP = 60;

function LogDetail({ log }: { log: LogEntry }) {
  return (
    <>
      <dl className="kvlist">
        <div><dt>时间</dt><dd className="cell-mono">{log.timestamp}</dd></div>
        <div><dt>级别</dt><dd><StatusBadge label={log.level} tone={toneForLogLevel(log.level)} size="sm" /></dd></div>
        <div><dt>来源</dt><dd className="cell-mono">{log.name}</dd></div>
        <div><dt>节点</dt><dd className="cell-mono">{log.node_id || '—'}</dd></div>
        <div><dt>请求 ID</dt><dd className="cell-mono">{log.request_id || '—'}</dd></div>
      </dl>
      <h3 className="drawer__subtitle">消息</h3>
      <pre className="codeblock">{log.message}</pre>
    </>
  );
}

function SessionDetail({ session }: { session: SessionSummary }) {
  return (
    <dl className="kvlist">
      <div><dt>标题</dt><dd>{session.title || '未命名会话'}</dd></div>
      <div><dt>会话 ID</dt><dd className="cell-mono">{session.id}</dd></div>
      <div><dt>消息数</dt><dd className="num-display">{session.message_count}</dd></div>
      <div><dt>创建时间</dt><dd className="cell-mono">{session.created_at}</dd></div>
      <div><dt>更新时间</dt><dd className="cell-mono">{session.updated_at}</dd></div>
    </dl>
  );
}

export function ActivityPage() {
  const [level, setLevel] = useState<LevelId>('');
  const [visible, setVisible] = useState(VISIBLE_STEP);
  const [activeLog, setActiveLog] = useState<LogEntry | null>(null);
  const [activeSession, setActiveSession] = useState<SessionSummary | null>(null);
  const [compactDetails, setCompactDetails] = useState(false);

  const logs = useRecentLogs({ limit: 200, ...(level ? { level } : {}) }, 10_000);
  const sessions = useSessions(20, 30_000);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1199px)');
    const update = () => setCompactDetails(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);

  const refresh = useCallback(() => {
    logs.refresh();
    sessions.refresh();
  }, [logs.refresh, sessions.refresh]);
  useRegisterRefresh(refresh);
  useReveal([logs.data, sessions.data]);

  const entries = logs.data?.logs ?? [];
  const windowed = useMemo(() => entries.length > VIRTUAL_THRESHOLD ? entries.slice(0, visible) : entries, [entries, visible]);
  const items: TimelineItem[] = useMemo(
    () => windowed.map((entry) => ({
      id: `${entry.seq}`,
      time: entry.timestamp,
      title: entry.message,
      detail: entry.name,
      tone: toneForLogLevel(entry.level),
      badge: entry.level,
      ...(entry.node_id ? { source: entry.node_id } : {}),
    })),
    [windowed],
  );

  const sessionList = sessions.data?.sessions ?? [];
  const selectLevel = useCallback((id: LevelId) => {
    setLevel(id);
    setVisible(VISIBLE_STEP);
  }, []);
  const selectLog = useCallback((id: string) => {
    const found = entries.find((entry) => `${entry.seq}` === id) ?? null;
    setActiveLog(found);
    setActiveSession(null);
  }, [entries]);
  const selectSession = useCallback((session: SessionSummary) => {
    setActiveSession(session);
    setActiveLog(null);
  }, []);

  const logsDenied = logs.errorKind === 'forbidden' || logs.errorKind === 'unauthorized';
  const hasToken = Boolean(getLogToken());

  return (
    <div className="activity-page" data-testid="activity-page">
      <PageBackdrop scene="activity" className="activity-page__bg" />
      <PageHeader
        tag="ACTIVITY"
        title="活动"
        description="运行日志、会话索引与事件详情在同一工作区内独立滚动。"
        actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={logs.refreshing} onClick={refresh}>刷新</CommandButton>}
      />

      <div className="activity-layout">
        <aside className="activity-rail" aria-label="Activity filters and sessions">
          <section className="activity-panel activity-rail__identity">
            <span className="mono-label">LIVE INDEX</span>
            <strong>{logs.data ? `${logs.data.count} buffered` : 'checking buffer'}</strong>
            <span>{sessions.data ? `${sessions.data.total} saved sessions` : 'session index pending'}</span>
          </section>

          <nav className="activity-nav" aria-label="Activity sections">
            <a href="#activity-feed" data-active="true">运行日志 <em>{entries.length.toString().padStart(2, '0')}</em></a>
            <a href="#activity-sessions">会话 <em>{sessionList.length.toString().padStart(2, '0')}</em></a>
            <a href="#activity-detail">详情 <em>{activeLog || activeSession ? 'ON' : '—'}</em></a>
          </nav>

          <section className="activity-panel activity-filter" aria-labelledby="activity-filter-title">
            <SectionHead title="日志筛选" id="activity-filter-title" hint="后端按最低级别筛选" />
            <div className="chiprow" role="group" aria-label="按日志级别筛选">
              {LEVELS.map((item) => <button key={item.id || 'all'} type="button" className="chip" data-active={level === item.id ? 'true' : undefined} aria-pressed={level === item.id} onClick={() => selectLevel(item.id)}>{item.label}</button>)}
            </div>
            {!hasToken && !logsDenied ? <p className="inline-note">未配置日志令牌时，受保护后端会拒绝读取。</p> : null}
          </section>

          <section className="activity-panel activity-sessions" id="activity-sessions" aria-labelledby="activity-sessions-title">
            <SectionHead title="会话" id="activity-sessions-title" hint={sessions.data ? `共 ${sessions.data.total} 个` : '历史会话索引'} />
            {sessions.state === 'loading' && sessionList.length === 0 ? <SkeletonRows rows={4} columns={1} /> : sessions.state === 'error' ? <EmptyState kind="error" title="会话加载失败" description="会话索引暂时不可用。" detail={sessions.error} errorKind={sessions.errorKind} errorStatus={sessions.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={sessions.refresh}>重试</CommandButton>} compact /> : sessionList.length === 0 ? <EmptyState kind="empty" title="暂无会话" description="发起对话后会话会出现在这里。" compact /> : <div className="activity-session-list">{sessionList.map((session) => <button type="button" className="activity-session" data-active={activeSession?.id === session.id ? 'true' : undefined} key={session.id} onClick={() => selectSession(session)}><strong>{session.title || '未命名会话'}</strong><span>{session.message_count} messages</span><em>{session.updated_at}</em></button>)}</div>}
          </section>
        </aside>

        <main className="activity-main">
          <section className="activity-panel activity-feed" id="activity-feed" aria-labelledby="activity-feed-title">
            <SectionHead title="运行日志" id="activity-feed-title" hint={logs.data ? `缓冲区 ${logs.data.count} / ${logs.data.buffer_capacity} 条，匹配 ${logs.data.matched} 条` : '最近的后端日志'} />
            <div className="activity-feed__scroll">
              {logsDenied ? <EmptyState kind="denied" title="需要日志访问令牌" description="后端要求 X-QLH-Log-Token 才能读取日志缓冲区。" detail={logs.error} errorKind={logs.errorKind} errorStatus={logs.errorStatus} action={<CommandButton variant="ghost" size="sm" icon={KeyRound} href={routeHref('settings')}>前往设置</CommandButton>} /> : logs.state === 'error' ? <EmptyState kind="error" title="日志加载失败" description="请确认后端正在运行。" detail={logs.error} errorKind={logs.errorKind} errorStatus={logs.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={logs.refresh}>重试</CommandButton>} /> : logs.state === 'loading' && entries.length === 0 ? <SkeletonRows rows={6} columns={2} /> : entries.length === 0 ? <EmptyState kind="empty" title={level ? '该级别下暂无日志' : '日志缓冲区为空'} description="产生新的后端事件后会出现在这里。" /> : <><ActivityTimeline items={items} state={logs.state} errorKind={logs.errorKind} errorStatus={logs.errorStatus} live onSelect={selectLog} />{entries.length > windowed.length ? <div className="loadmore"><CommandButton variant="ghost" size="sm" onClick={() => setVisible((current) => current + VISIBLE_STEP)}>加载更多（剩余 {entries.length - windowed.length} 条）</CommandButton></div> : null}</>}
            </div>
          </section>
        </main>

        <aside className="activity-details" id="activity-detail" aria-label="Selected activity detail">
          <section className="activity-panel activity-detail-panel">
            <SectionHead title="详情" hint={activeLog ? `LOG #${activeLog.seq}` : activeSession ? 'SESSION' : '选择一条记录'} />
            {activeLog ? <LogDetail log={activeLog} /> : activeSession ? <SessionDetail session={activeSession} /> : <EmptyState kind="empty" title="未选择记录" description="点击日志或会话以固定详情。" compact />}
          </section>
        </aside>
      </div>

      <Drawer open={compactDetails && activeLog !== null} tag="LOG" title={activeLog ? `${activeLog.level} · #${activeLog.seq}` : ''} onClose={() => setActiveLog(null)}>
        {activeLog ? <LogDetail log={activeLog} /> : null}
      </Drawer>
    </div>
  );
}
