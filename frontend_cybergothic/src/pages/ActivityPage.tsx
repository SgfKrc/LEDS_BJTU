/**
 * 活动 — 运行日志时间线与会话记录。
 *
 * 日志按级别筛选，新条目通过 aria-live 播报（§5.5）。
 * 超过 100 条时启用窗口化渲染，避免长列表掉帧（§6）。
 */

import { useCallback, useMemo, useState } from 'react';
import { KeyRound, RefreshCw } from 'lucide-react';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { ActivityTimeline, type TimelineItem } from '../components/ActivityTimeline';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { CommandButton } from '../components/CommandButton';
import { StatusBadge } from '../components/StatusBadge';
import { Drawer } from '../components/Drawer';
import { TaskTable, type Column } from '../components/TaskTable';
import { useRegisterRefresh } from '../app/refreshBus';
import { useReveal } from '../motion/useReveal';
import { toneForLogLevel } from '../components/statusTone';
import { useRecentLogs, useSessions } from '../data/hooks';
import { getLogToken } from '../data/api';
import { routeHref } from '../app/routes';
import type { LogEntry, SessionSummary } from '../data/types';

/**
 * 后端 /api/logs/recent 的 level 参数是「最低级别」语义（levelno >= 阈值），
 * 所以这里直接传日志级别名即可。
 */
const LEVELS = [
  { id: '', label: '全部' },
  { id: 'INFO', label: 'INFO+' },
  { id: 'WARNING', label: 'WARN+' },
  { id: 'ERROR', label: 'ERROR' },
] as const;

type LevelId = (typeof LEVELS)[number]['id'];

/** 长列表窗口化：超过该行数后只渲染前 VISIBLE_STEP 条，按需增量展开。 */
const VIRTUAL_THRESHOLD = 100;
const VISIBLE_STEP = 60;

export function ActivityPage() {
  const [level, setLevel] = useState<LevelId>('');
  const [visible, setVisible] = useState(VISIBLE_STEP);
  const [activeLog, setActiveLog] = useState<LogEntry | null>(null);

  const logs = useRecentLogs({ limit: 200, ...(level ? { level } : {}) }, 10_000);
  const sessions = useSessions(20, 30_000);

  const refresh = useCallback(() => {
    logs.refresh();
    sessions.refresh();
  }, [logs.refresh, sessions.refresh]);
  useRegisterRefresh(refresh);

  useReveal([logs.data, sessions.data]);

  const hasToken = Boolean(getLogToken());
  const entries = logs.data?.logs ?? [];

  const windowed = useMemo(
    () => (entries.length > VIRTUAL_THRESHOLD ? entries.slice(0, visible) : entries),
    [entries, visible],
  );

  const items: TimelineItem[] = useMemo(
    () =>
      windowed.map((entry) => ({
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

  const sessionColumns: Column<SessionSummary>[] = useMemo(
    () => [
      {
        key: 'title',
        header: '标题',
        render: (row) => row.title || '未命名会话',
      },
      {
        key: 'id',
        header: '会话 ID',
        secondary: true,
        render: (row) => <span className="cell-mono">{row.id}</span>,
      },
      {
        key: 'messages',
        header: '消息',
        numeric: true,
        render: (row) => <span className="num-display">{row.message_count}</span>,
      },
      {
        key: 'updated',
        header: '更新时间',
        secondary: true,
        render: (row) => row.updated_at || '—',
      },
    ],
    [],
  );

  const selectLevel = useCallback((id: LevelId) => {
    setLevel(id);
    setVisible(VISIBLE_STEP);
  }, []);

  // 403 = 缺少或过期的日志令牌，属于配置问题而不是后端故障。
  const logsDenied = logs.state === 'error' && logs.error.startsWith('无权限');

  return (
    <>
      <PageHeader
        tag="ACTIVITY"
        title="活动"
        description="节点运行日志与会话记录，用于确认某次请求真正发生了什么。"
        actions={
          <CommandButton
            variant="ghost"
            size="sm"
            icon={RefreshCw}
            busy={logs.refreshing}
            onClick={refresh}
          >
            刷新
          </CommandButton>
        }
      />

      <section className="band" data-reveal>
        <SectionHead
          title="运行日志"
          hint={
            logs.data
              ? `缓冲区 ${logs.data.count} / ${logs.data.buffer_capacity} 条，匹配 ${logs.data.matched} 条。`
              : '按级别筛选最近的后端日志。'
          }
          actions={
            <div className="chiprow" role="group" aria-label="按日志级别筛选">
              {LEVELS.map((l) => (
                <button
                  key={l.id || 'all'}
                  type="button"
                  className="chip"
                  data-active={level === l.id ? 'true' : undefined}
                  aria-pressed={level === l.id}
                  onClick={() => selectLevel(l.id)}
                >
                  {l.label}
                </button>
              ))}
            </div>
          }
        />

        {logsDenied ? (
          <EmptyState
            kind="denied"
            title="需要日志访问令牌"
            description="后端要求 X-QLH-Log-Token 才能读取日志缓冲区。在设置页填入令牌后即可查看。"
            detail={logs.error}
            action={
              <CommandButton variant="ghost" size="sm" icon={KeyRound} href={routeHref('settings')}>
                前往设置
              </CommandButton>
            }
          />
        ) : logs.state === 'error' ? (
          <EmptyState
            kind="error"
            title="日志加载失败"
            description="请确认后端正在运行。"
            detail={logs.error}
            action={
              <CommandButton variant="ghost" size="sm" onClick={logs.refresh}>
                重试
              </CommandButton>
            }
          />
        ) : logs.state === 'loading' && entries.length === 0 ? (
          <SkeletonRows rows={6} columns={2} />
        ) : entries.length === 0 ? (
          <EmptyState
            kind="empty"
            title={level ? '该级别下没有日志' : '日志缓冲区为空'}
            description={
              level ? '试试放宽到「全部」级别。' : '后端刚启动或缓冲区已被清空。'
            }
          />
        ) : (
          <>
            <ActivityTimeline
              items={items}
              state={logs.state}
              live
              onSelect={(id) => {
                const found = entries.find((e) => `${e.seq}` === id);
                setActiveLog(found ?? null);
              }}
            />
            {entries.length > windowed.length ? (
              <div className="loadmore">
                <CommandButton
                  variant="ghost"
                  size="sm"
                  onClick={() => setVisible((v) => v + VISIBLE_STEP)}
                >
                  加载更多（剩余 {entries.length - windowed.length} 条）
                </CommandButton>
              </div>
            ) : null}
          </>
        )}

        {!hasToken && !logsDenied ? (
          <p className="inline-note">
            未配置日志令牌。若后端启用了令牌校验，日志将返回无权限。
          </p>
        ) : null}
      </section>

      <section className="band band--alt" data-reveal>
        <SectionHead
          title="会话"
          hint="已保存的对话会话；活动会话以主色标记。"
          actions={
            sessions.data ? (
              <StatusBadge
                label={`共 ${sessions.data.total} 个`}
                tone="info"
                size="sm"
              />
            ) : null
          }
        />
        <TaskTable<SessionSummary>
          caption="已保存的对话会话"
          columns={sessionColumns}
          rows={sessions.data?.sessions ?? []}
          rowKey={(row) => row.id}
          state={sessions.state}
          error={sessions.error}
          emptyTitle="还没有会话"
          emptyDescription="发起一次对话后，会话会出现在这里。"
          onRetry={sessions.refresh}
        />
        {sessions.data?.source ? (
          <p className="inline-note">数据来源：{sessions.data.source}</p>
        ) : null}
      </section>

      <Drawer
        open={activeLog !== null}
        tag="LOG"
        title={activeLog ? `${activeLog.level} · #${activeLog.seq}` : ''}
        onClose={() => setActiveLog(null)}
      >
        {activeLog ? (
          <>
            <dl className="kvlist">
              <div>
                <dt>时间</dt>
                <dd className="cell-mono">{activeLog.timestamp}</dd>
              </div>
              <div>
                <dt>级别</dt>
                <dd>
                  <StatusBadge label={activeLog.level} tone={toneForLogLevel(activeLog.level)} size="sm" />
                </dd>
              </div>
              <div>
                <dt>来源</dt>
                <dd className="cell-mono">{activeLog.name}</dd>
              </div>
              <div>
                <dt>节点</dt>
                <dd className="cell-mono">{activeLog.node_id || '—'}</dd>
              </div>
              <div>
                <dt>请求 ID</dt>
                <dd className="cell-mono">{activeLog.request_id || '—'}</dd>
              </div>
            </dl>
            <h3 className="drawer__subtitle">消息</h3>
            <pre className="codeblock">{activeLog.message}</pre>
          </>
        ) : null}
      </Drawer>
    </>
  );
}
