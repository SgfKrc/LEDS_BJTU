import { SqliteStore } from './data/sqlite-store';

type ProbeMode = 'seed' | 'verify' | 'assert-unavailable';

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  if (index < 0 || !process.argv[index + 1]) {
    throw new Error(`missing ${name}`);
  }
  return process.argv[index + 1];
}

function assertCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function seed(store: SqliteStore): Record<string, unknown> {
  store.open();
  const pythonSession = store.prepare(
    'SELECT title, message_count FROM sessions WHERE session_id = ?',
  ).get('python-clean') as { title: string; message_count: number } | undefined;
  assertCondition(pythonSession?.title === 'Python clean seed', 'python session missing');
  const pythonMessages = store.prepare(
    'SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?',
  ).get('python-clean') as { count: number };
  assertCondition(Number(pythonMessages.count) === 2, 'python messages missing');

  const now = new Date().toISOString();
  store.transaction(() => {
    store.prepare(
      `INSERT INTO sessions(session_id, title, created_at, updated_at, message_count)
       VALUES (?, ?, ?, ?, 1)
       ON CONFLICT(session_id) DO UPDATE SET
         title=excluded.title, updated_at=excluded.updated_at, message_count=1`,
    ).run('node-clean', 'Node clean seed', now, now);
    store.prepare('DELETE FROM session_messages WHERE session_id = ?').run('node-clean');
    store.prepare(
      `INSERT INTO session_messages(session_id, role, content, created_at, metrics)
       VALUES (?, ?, ?, ?, ?)`,
    ).run('node-clean', 'assistant', 'node persisted', now, JSON.stringify({ source: 'node' }));
    store.prepare(
      `INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
    ).run('environment_gate_node', 'ready', now);
  });
  return { schema_version: store.schemaVersion, python_messages: 2, node_seeded: true };
}

function verify(store: SqliteStore): Record<string, unknown> {
  store.open();
  const version = store.schemaVersion;
  const pythonMessages = store.prepare(
    'SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?',
  ).get('python-clean') as { count: number };
  const nodeMessage = store.prepare(
    'SELECT content FROM session_messages WHERE session_id = ? ORDER BY message_id LIMIT 1',
  ).get('node-clean') as { content: string } | undefined;
  const marker = store.prepare(
    'SELECT value FROM cluster_settings WHERE key = ?',
  ).get('environment_gate_node') as { value: string } | undefined;
  assertCondition(version === 5, `unexpected schema version ${version}`);
  assertCondition(Number(pythonMessages.count) === 2, 'python messages not retained');
  assertCondition(nodeMessage?.content === 'node persisted', 'node message missing');
  assertCondition(marker?.value === 'ready', 'node marker missing');
  const health = store.health();
  assertCondition(health.status === 'ok' && health.writable, 'store is not writable');
  return { schema_version: version, health, verified: true };
}

function assertUnavailable(store: SqliteStore): Record<string, unknown> {
  let health;
  try {
    store.open();
    health = store.health();
  } catch (error) {
    return {
      unavailable: true,
      rejected_during_open: true,
      error: error instanceof Error ? error.message : String(error),
    };
  }
  assertCondition(
    health.status === 'unavailable' && !health.writable,
    'SQLite unexpectedly remained writable',
  );
  return { unavailable: true, health };
}

function main(): void {
  const filePath = argument('--path');
  const mode = argument('--mode') as ProbeMode;
  assertCondition(
    mode === 'seed' || mode === 'verify' || mode === 'assert-unavailable',
    `unsupported mode ${mode}`,
  );
  const store = new SqliteStore(filePath);
  try {
    const result = mode === 'seed'
      ? seed(store)
      : mode === 'verify'
        ? verify(store)
        : assertUnavailable(store);
    process.stdout.write(`${JSON.stringify({ mode, path: filePath, ...result })}\n`);
  } finally {
    store.close();
  }
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
}
