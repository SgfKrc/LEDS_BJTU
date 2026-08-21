import { useCallback, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Archive,
  BarChart3,
  Check,
  Download,
  FileSearch,
  Gauge,
  Gavel,
  HardDrive,
  KeyRound,
  RefreshCw,
  Trash2,
  UsersRound,
  Vote,
} from 'lucide-react';
import { CommandButton } from '../components/CommandButton';
import { EmptyState, SkeletonRows } from '../components/EmptyState';
import { PageHeader, SectionHead } from '../components/PageHeader';
import { StatusBadge } from '../components/StatusBadge';
import { pushToast } from '../components/Toast';
import { useRegisterRefresh } from '../app/refreshBus';
import { routeHref } from '../app/routes';
import * as api from '../data/api';
import { fixturesEnabled } from '../data/fixtures';
import {
  useCanVote,
  useLogFiles,
  useLogStats,
  useMyRole,
  useNodesLogAggregate,
  useNodesLogSummary,
  useReviewTickets,
} from '../data/hooks';
import type { LogEntry, LogFileContentResponse, LogFileRecord, NodeLogSummary, ReviewTicket } from '../data/types';
import { ArchiveClockCanvas } from '../visual/ArchiveClockCanvas';

type WorkspaceId = 'archive' | 'nodes' | 'review';

const WORKSPACES: Array<{ id: WorkspaceId; label: string; icon: typeof Archive; marker: string }> = [
  { id: 'archive', label: 'Archive', icon: Archive, marker: '01' },
  { id: 'nodes', label: 'Node relay', icon: UsersRound, marker: '02' },
  { id: 'review', label: 'Review', icon: Gavel, marker: '03' },
];

function formatBytes(value?: number): string {
  if (!value || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function dateLabel(value?: string | number | null): string {
  if (value === null || value === undefined || value === '') return 'unknown';
  const date = new Date(typeof value === 'number' ? value * 1000 : value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function toneForTicket(status?: string): 'ok' | 'warn' | 'danger' | 'info' | 'idle' {
  if (status === 'approved') return 'ok';
  if (status === 'rejected' || status === 'expired') return 'danger';
  if (status === 'pending') return 'warn';
  return 'idle';
}

function previewFixture(file: LogFileRecord): LogFileContentResponse {
  return {
    name: file.name,
    truncated: false,
    content: [
      `# ${file.name}`,
      '2026-08-21 09:29:45 INFO api_server event=http_request method=GET path=/api/logs/stats status=200 duration_ms=3',
      '2026-08-21 09:29:44 WARNING scheduler event=pipeline_capacity_rejected reason=insufficient_participants',
      '2026-08-21 09:28:56 ERROR task_graph event=stage_failed stage_id=cand_b code=PROVIDER_UNAVAILABLE',
    ].join('\n'),
  };
}

function sourceSummaries(summary: NodeLogSummary | undefined): Array<[string, number]> {
  return Object.entries(summary?.levels ?? {}).sort(([, a], [, b]) => b - a);
}

type RelayLog = LogEntry & {
  source: string;
  _sortKey: string;
};

function normalizeRelayLog(entry: unknown, source: string, index: number): RelayLog {
  const raw = entry && typeof entry === 'object' ? entry as Record<string, unknown> : {};
  const timestamp = String(raw.timestamp ?? raw.time ?? raw.created_at ?? '');
  const sequence = Number(raw.seq);
  return {
    name: String(raw.name ?? 'unknown'),
    timestamp,
    level: String(raw.level ?? 'INFO'),
    levelno: Number(raw.levelno ?? 0),
    message: String(raw.message ?? raw.msg ?? ''),
    seq: Number.isFinite(sequence) ? sequence : index,
    ...(raw.request_id != null ? { request_id: String(raw.request_id) } : {}),
    ...(raw.node_id != null ? { node_id: String(raw.node_id) } : {}),
    source,
    _sortKey: `${timestamp}-${String(Number.isFinite(sequence) ? sequence : index)}`,
  };
}

export function AuditPage() {
  const role = useMyRole();
  const canManageCluster = role.state === 'ready' && role.data?.is_master === true;
  const usingFixtures = fixturesEnabled();
  const files = useLogFiles();
  const stats = useLogStats();
  const nodeSummary = useNodesLogSummary(canManageCluster);
  const aggregate = useNodesLogAggregate(canManageCluster);
  const tickets = useReviewTickets();
  const canVote = useCanVote();

  const [workspace, setWorkspace] = useState<WorkspaceId>('archive');
  const [fileOverride, setFileOverride] = useState<LogFileRecord[] | null>(null);
  const [ticketOverride, setTicketOverride] = useState<ReviewTicket[] | null>(null);
  const [selectedFile, setSelectedFile] = useState<LogFileContentResponse | null>(null);
  const [busy, setBusy] = useState('');
  const [reviewForm, setReviewForm] = useState({ targetNodeId: '', reason: '', timeoutHours: '48' });

  const fileList = fileOverride ?? files.data?.files ?? [];
  const ticketList = ticketOverride ?? tickets.data?.tickets ?? [];
  const nodeRows = useMemo(
    () => [nodeSummary.data?.local, ...(nodeSummary.data?.workers ?? [])].filter(Boolean) as NodeLogSummary[],
    [nodeSummary.data],
  );
  const aggregateEntries = useMemo(
    () => [
      ...(aggregate.data?.local?.logs ?? []).map((entry, index) => normalizeRelayLog(entry, aggregate.data?.local?.node_id || 'local', index)),
      ...(aggregate.data?.workers ?? []).flatMap((worker) => (worker?.logs ?? []).map((entry, index) => normalizeRelayLog(entry, worker?.node_id || 'unknown', index))),
    ].sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || '') || b._sortKey.localeCompare(a._sortKey)).slice(0, 10),
    [aggregate.data],
  );
  const pendingTickets = ticketList.filter((ticket) => ticket.status === 'pending');
  const logDenied = files.errorKind === 'forbidden' || stats.errorKind === 'forbidden' || files.errorKind === 'unauthorized' || stats.errorKind === 'unauthorized';
  const missingLogToken = !usingFixtures && !api.getLogToken();

  const refresh = useCallback(() => {
    role.refresh();
    files.refresh();
    stats.refresh();
    nodeSummary.refresh();
    aggregate.refresh();
    tickets.refresh();
    canVote.refresh();
  }, [aggregate, canVote, files, nodeSummary, role, stats, tickets]);
  useRegisterRefresh(refresh);

  const openFile = async (file: LogFileRecord) => {
    setBusy(`open:${file.name}`);
    try {
      const content = usingFixtures ? previewFixture(file) : await api.fetchLogContent(file.name);
      setSelectedFile(content);
    } catch (error) {
      pushToast(`Log preview failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const removeFile = async (file: LogFileRecord) => {
    if (!window.confirm(`Delete log file ${file.name}?`)) return;
    setBusy(`delete:${file.name}`);
    try {
      if (usingFixtures) {
        setFileOverride(fileList.filter((item) => item.name !== file.name));
      } else {
        await api.deleteLogFile(file.name);
        await files.refresh();
        await stats.refresh();
      }
      if (selectedFile?.name === file.name) setSelectedFile(null);
      pushToast(usingFixtures ? 'Fixture log archived out of the list' : 'Log file deleted', 'ok');
    } catch (error) {
      pushToast(`Log deletion failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const saveBlob = (blob: Blob, filename: string) => {
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = href;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  const downloadFile = async (file: LogFileRecord) => {
    setBusy(`download:${file.name}`);
    try {
      if (usingFixtures) {
        saveBlob(new Blob([previewFixture(file).content], { type: 'text/plain' }), file.name);
      } else {
        saveBlob(await api.downloadLogFile(file.name), file.name);
      }
      pushToast(`Downloaded ${file.name}`, usingFixtures ? 'info' : 'ok');
    } catch (error) {
      pushToast(`Log download failed: ${api.describeError(error)}`, 'danger');
    } finally { setBusy(''); }
  };

  const exportArchive = async () => {
    setBusy('export');
    try {
      if (usingFixtures) saveBlob(new Blob(['Fixture archive'], { type: 'application/zip' }), 'qlh-logs-fixture.zip');
      else saveBlob(await api.exportLogs(), 'qlh-logs-export.zip');
      pushToast('Log archive exported', usingFixtures ? 'info' : 'ok');
    } catch (error) {
      pushToast(`Log export failed: ${api.describeError(error)}`, 'danger');
    } finally { setBusy(''); }
  };

  const createTicket = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!reviewForm.targetNodeId.trim()) {
      pushToast('Target node ID is required', 'danger');
      return;
    }
    const timeout = Number(reviewForm.timeoutHours);
    if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 720) {
      pushToast('Review timeout must be between 1 and 720 hours', 'danger');
      return;
    }
    setBusy('create-review');
    try {
      if (usingFixtures) {
        const next: ReviewTicket = {
          ticket_id: `review_fixture_${Date.now()}`,
          status: 'pending',
          created_at: Date.now() / 1000,
          created_by: role.data?.node_id || 'master',
          target_node_id: reviewForm.targetNodeId.trim(),
          transfer_reason: reviewForm.reason.trim(),
          score: 0,
          expires_at: Date.now() / 1000 + timeout * 3600,
          votes: [],
        };
        setTicketOverride([next, ...ticketList]);
      } else {
        await api.createReviewTicket(reviewForm.targetNodeId.trim(), reviewForm.reason.trim(), timeout);
        await tickets.refresh();
      }
      setReviewForm({ targetNodeId: '', reason: '', timeoutHours: '48' });
      pushToast('Review ticket created', 'ok');
    } catch (error) {
      pushToast(`Review creation failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const vote = async (ticket: ReviewTicket, value: -1 | 0 | 1) => {
    setBusy(`vote:${ticket.ticket_id}:${value}`);
    try {
      if (usingFixtures) {
        const votes = [...(ticket.votes ?? []), { voter_node_id: canVote.data?.node_id || 'master', value, timestamp: Date.now() / 1000, comment: '' }];
        setTicketOverride(ticketList.map((item) => item.ticket_id === ticket.ticket_id ? { ...item, votes, score: (item.score ?? 0) + value } : item));
      } else {
        await api.castReviewVote(ticket.ticket_id, value);
        await tickets.refresh();
      }
      pushToast(value > 0 ? 'Approval recorded' : value < 0 ? 'Block vote recorded' : 'Abstention recorded', 'ok');
    } catch (error) {
      pushToast(`Vote failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const expireTickets = async () => {
    setBusy('expire-review');
    try {
      if (usingFixtures) {
        const now = Date.now() / 1000;
        setTicketOverride(ticketList.map((ticket) => ticket.status === 'pending' && typeof ticket.expires_at === 'number' && ticket.expires_at < now ? { ...ticket, status: 'expired', resolved_at: now } : ticket));
      } else {
        await api.expireReviewCheck();
        await tickets.refresh();
      }
      pushToast('Review expiry check completed', 'ok');
    } catch (error) {
      pushToast(`Expiry check failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  const pollMail = async () => {
    setBusy('mail-poll');
    try {
      if (usingFixtures) pushToast('Fixture mail poll completed', 'info');
      else await api.pollReviewMail();
      await tickets.refresh();
      pushToast('Mail vote poll completed', usingFixtures ? 'info' : 'ok');
    } catch (error) {
      pushToast(`Mail poll failed: ${api.describeError(error)}`, 'danger');
    } finally { setBusy(''); }
  };

  const clearResolved = async () => {
    if (!window.confirm('Clear all resolved review tickets?')) return;
    setBusy('clear-resolved');
    try {
      if (usingFixtures) {
        setTicketOverride(ticketList.filter((ticket) => ticket.status === 'pending'));
      } else {
        await api.deleteResolvedReviewTickets();
        await tickets.refresh();
      }
      pushToast('Resolved review tickets cleared', 'ok');
    } catch (error) {
      pushToast(`Clearing reviews failed: ${api.describeError(error)}`, 'danger');
    } finally {
      setBusy('');
    }
  };

  return (
    <div className="audit-page" data-testid="audit-page">
      <ArchiveClockCanvas className="audit-page__bg" />
      <div className="audit-page__content">
        <PageHeader
          tag="AUDIT LEDGER"
          title="Audit Ledger"
          description="Archive log evidence, inspect node relays, and keep transfer reviews traceable."
          actions={<CommandButton variant="ghost" size="sm" icon={RefreshCw} busy={files.refreshing || tickets.refreshing} onClick={refresh}>Refresh</CommandButton>}
        />

        <div className="audit-layout">
          <aside className="audit-rail">
            <section className="audit-panel audit-identity">
              <div className="audit-panel__eyebrow">ACCESS ENVELOPE</div>
              <strong>{role.data?.node_id || 'detecting node'}</strong>
              <StatusBadge label={canManageCluster ? 'MASTER / AUDIT WRITE' : 'READ ONLY NODE'} tone={canManageCluster ? 'ok' : 'info'} />
              <p>{canManageCluster ? 'Node relay and review administration are available.' : 'Cluster-only operations remain locked on this node.'}</p>
            </section>

            <nav className="audit-nav" aria-label="Audit workspaces">
              {WORKSPACES.map(({ id, label, icon: Icon, marker }) => (
                <button key={id} type="button" data-active={workspace === id ? 'true' : undefined} onClick={() => setWorkspace(id)}>
                  <span><Icon size={15} aria-hidden="true" />{label}</span>
                  <em>{marker}</em>
                </button>
              ))}
            </nav>

            <section className="audit-panel audit-access-note">
              <KeyRound size={16} aria-hidden="true" />
              <div>
                <strong>Log access</strong>
                <p>{usingFixtures ? 'Fixture evidence is active.' : missingLogToken ? 'Configure a log token before reading archive evidence.' : 'Log token detected in this browser.'}</p>
              </div>
              {missingLogToken ? <CommandButton variant="ghost" size="sm" href={routeHref('settings')}>Settings</CommandButton> : null}
            </section>
          </aside>

          <main className="audit-main">
            {workspace === 'archive' ? (
              <section className="audit-panel audit-workspace" aria-label="Log archive">
                <SectionHead title="Log archive" hint="Files remain separate from the live Activity timeline." actions={<div className="audit-review-actions"><CommandButton variant="ghost" size="sm" icon={Download} busy={busy === 'export'} onClick={() => void exportArchive()}>Export ZIP</CommandButton><StatusBadge label={`${fileList.length} FILES`} tone="info" size="sm" /></div>} />
                {logDenied ? (
                  <EmptyState kind="denied" title="Log token required" description="Archive and statistics APIs require X-QLH-Log-Token." detail={files.error || stats.error} action={<CommandButton variant="ghost" size="sm" icon={KeyRound} href={routeHref('settings')}>Open settings</CommandButton>} />
                ) : files.state === 'error' || stats.state === 'error' ? (
                  <EmptyState kind="error" title="Archive evidence is unavailable" description="The live Activity page remains independent from this archive." detail={files.error || stats.error} errorKind={files.errorKind || stats.errorKind} errorStatus={files.errorStatus || stats.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={refresh}>Retry</CommandButton>} />
                ) : files.state === 'loading' && fileList.length === 0 ? <SkeletonRows rows={4} columns={3} /> : (
                  <>
                    <div className="audit-stat-grid">
                      <div><HardDrive size={17} aria-hidden="true" /><span>ARCHIVED</span><strong>{formatBytes(stats.data?.files_total_bytes)}</strong></div>
                      <div><Gauge size={17} aria-hidden="true" /><span>BUFFER</span><strong>{stats.data ? `${stats.data.buffer_size} / ${stats.data.buffer_capacity}` : '---'}</strong></div>
                      <div><BarChart3 size={17} aria-hidden="true" /><span>DROPPED EST.</span><strong>{stats.data?.buffer_dropped_estimate ?? '---'}</strong></div>
                    </div>
                    {missingLogToken ? <p className="audit-inline-note">No log token is currently stored. A protected backend will reject archive requests until it is configured.</p> : null}
                    <div className="audit-file-list">
                      {fileList.map((file) => (
                        <article className="audit-file-row" key={file.name}>
                          <FileSearch size={17} aria-hidden="true" />
                          <div><strong>{file.name}</strong><span>{formatBytes(file.size)} · {dateLabel(file.modified)}</span></div>
                          <div className="audit-file-row__actions">
                            <CommandButton variant="ghost" size="sm" busy={busy === `open:${file.name}`} onClick={() => void openFile(file)}>Preview</CommandButton>
                            <CommandButton variant="ghost" size="sm" icon={Download} busy={busy === `download:${file.name}`} onClick={() => void downloadFile(file)}>Download</CommandButton>
                            <CommandButton variant="danger" size="sm" icon={Trash2} busy={busy === `delete:${file.name}`} onClick={() => void removeFile(file)}>Delete</CommandButton>
                          </div>
                        </article>
                      ))}
                    </div>
                    {selectedFile ? <section className="audit-preview"><div><span className="mono-label">FILE PREVIEW</span><strong>{selectedFile.name}</strong>{selectedFile.truncated ? <StatusBadge label="TRUNCATED" tone="warn" size="sm" /> : null}</div><pre className="codeblock">{selectedFile.content}</pre></section> : null}
                  </>
                )}
              </section>
            ) : null}

            {workspace === 'nodes' ? (
              <section className="audit-panel audit-workspace" aria-label="Node log relay">
                <SectionHead title="Node relay" hint="Master-only aggregate pull; a failing worker never blocks local evidence." actions={<StatusBadge label={canManageCluster ? `${nodeRows.length} NODES` : 'MASTER ONLY'} tone={canManageCluster ? 'info' : 'idle'} size="sm" />} />
                {!canManageCluster ? <div className="audit-readonly-callout"><EmptyState kind="denied" title="Node relay is master-only" description="This client may read its own Activity timeline but cannot request or aggregate cluster logs." /></div> : nodeSummary.state === 'error' || aggregate.state === 'error' ? <EmptyState kind="error" title="Node relay is unavailable" description="Check cluster reachability and the log token, then retry." detail={nodeSummary.error || aggregate.error} errorKind={nodeSummary.errorKind || aggregate.errorKind} errorStatus={nodeSummary.errorStatus || aggregate.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={refresh}>Retry</CommandButton>} /> : nodeSummary.state === 'loading' && nodeRows.length === 0 ? <SkeletonRows rows={3} columns={3} /> : <>
                  <div className="audit-node-grid">
                    {nodeRows.map((node) => <article key={node.node_id}><div><span>{node.node_id}</span><StatusBadge label={node.error ? 'UNREACHABLE' : String(node.state || 'ONLINE').toUpperCase()} tone={node.error ? 'danger' : 'ok'} size="sm" /></div><strong>{node.error || `${node.buffer_size ?? 0} / ${node.buffer_capacity ?? 0} buffered`}</strong><p>{formatBytes(node.files_total_bytes)} in {node.files_count ?? 0} files · {node.buffer_dropped_estimate ?? 0} dropped</p><div className="audit-levels">{sourceSummaries(node).map(([level, count]) => <span key={level}>{level} <b>{count}</b></span>)}</div></article>)}
                  </div>
                  <section className="audit-relay-feed"><SectionHead title="Recent relay" hint={`${aggregateEntries.length} most recent entries`} />{aggregateEntries.length ? aggregateEntries.map((entry) => <article key={`${entry.source}-${entry.seq}-${entry.timestamp}`}><span>{entry.timestamp}</span><StatusBadge label={entry.level} tone={entry.level === 'ERROR' ? 'danger' : entry.level === 'WARNING' ? 'warn' : 'info'} size="sm" /><strong>{entry.message}</strong><em>{entry.source}</em></article>) : <EmptyState kind="empty" title="No relay entries" description="No remote node has returned a log entry yet." compact />}</section>
                </>}
              </section>
            ) : null}

            {workspace === 'review' ? (
              <section className="audit-panel audit-workspace" aria-label="Transfer review tickets">
                <SectionHead title="Transfer review" hint="A traceable gate before primary-node handover." actions={<div className="audit-review-actions">{canManageCluster ? <><CommandButton variant="ghost" size="sm" busy={busy === 'mail-poll'} onClick={() => void pollMail()}>Poll mail</CommandButton><CommandButton variant="ghost" size="sm" busy={busy === 'expire-review'} onClick={() => void expireTickets()}>Expiry check</CommandButton></> : null}{canManageCluster && ticketList.some((ticket) => ticket.status !== 'pending') ? <CommandButton variant="danger" size="sm" busy={busy === 'clear-resolved'} onClick={() => void clearResolved()}>Clear resolved</CommandButton> : null}</div>} />
                <div className="audit-review-status"><StatusBadge label={canVote.data?.can_vote ? 'VOTE ELIGIBLE' : 'NO VOTE'} tone={canVote.data?.can_vote ? 'ok' : 'idle'} size="sm" /><span>{canVote.data?.reason || 'Checking local review eligibility.'}</span><strong>{pendingTickets.length} pending</strong></div>
                {canManageCluster ? <form className="audit-review-form" onSubmit={createTicket}><label><span>TARGET NODE ID</span><input aria-label="TARGET NODE ID" value={reviewForm.targetNodeId} onChange={(event) => setReviewForm((current) => ({ ...current, targetNodeId: event.target.value }))} /></label><label><span>REASON</span><input aria-label="REASON" value={reviewForm.reason} onChange={(event) => setReviewForm((current) => ({ ...current, reason: event.target.value }))} /></label><label><span>TIMEOUT / HOURS</span><input aria-label="TIMEOUT / HOURS" type="number" min="1" max="720" value={reviewForm.timeoutHours} onChange={(event) => setReviewForm((current) => ({ ...current, timeoutHours: event.target.value }))} /></label><CommandButton type="submit" size="sm" icon={Check} busy={busy === 'create-review'}>Create review</CommandButton></form> : <p className="audit-inline-note">Only the master node may create, expire, or remove transfer review tickets.</p>}
                {tickets.state === 'error' ? <EmptyState kind="error" title="Review tickets are unavailable" description="Cluster review storage did not return a usable response." detail={tickets.error} errorKind={tickets.errorKind} errorStatus={tickets.errorStatus} action={<CommandButton variant="ghost" size="sm" onClick={tickets.refresh}>Retry</CommandButton>} /> : tickets.state === 'loading' && ticketList.length === 0 ? <SkeletonRows rows={3} columns={2} /> : <div className="audit-ticket-list">{ticketList.map((ticket) => <article key={ticket.ticket_id}><div className="audit-ticket__head"><div><span className="mono-label">{ticket.ticket_id}</span><strong>{ticket.target_node_id || 'Unknown target'}</strong></div><StatusBadge label={String(ticket.status || 'unknown').toUpperCase()} tone={toneForTicket(ticket.status)} size="sm" /></div><p>{ticket.transfer_reason || 'No transfer reason recorded.'}</p><dl><div><dt>SCORE</dt><dd>{ticket.score ?? 0}</dd></div><div><dt>VOTES</dt><dd>{ticket.votes?.length ?? 0}</dd></div><div><dt>EXPIRES</dt><dd>{dateLabel(ticket.expires_at)}</dd></div></dl>{ticket.status === 'pending' && canVote.data?.can_vote ? <div className="audit-votes"><CommandButton variant="ghost" size="sm" icon={Vote} busy={busy === `vote:${ticket.ticket_id}:1`} onClick={() => void vote(ticket, 1)}>Approve</CommandButton><CommandButton variant="ghost" size="sm" busy={busy === `vote:${ticket.ticket_id}:0`} onClick={() => void vote(ticket, 0)}>Abstain</CommandButton><CommandButton variant="danger" size="sm" busy={busy === `vote:${ticket.ticket_id}:-1`} onClick={() => void vote(ticket, -1)}>Block</CommandButton></div> : null}</article>)}</div>}
              </section>
            ) : null}
          </main>
        </div>
      </div>
    </div>
  );
}
