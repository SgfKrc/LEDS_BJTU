/** One-time, user-owned retirement command for the legacy PostgreSQL path. */
import * as path from 'path';
import { ArtifactStore, resolveModelStorePath } from './data/artifact-store';
import { PostgresRetirementService } from './data/postgres-retirement';
import { resolveSqlitePath, SqliteStore } from './data/sqlite-store';

interface ParsedArgs {
  command: 'prepare' | 'retire' | 'verify';
  sqlite: string;
  modelStore: string;
  backup: string;
  manifest: string;
  envFile?: string;
  passphrase: string;
}

function parseArgs(argv: string[]): ParsedArgs {
  const command = argv[0] as ParsedArgs['command'];
  if (!['prepare', 'retire', 'verify'].includes(command)) {
    throw new Error(
      '用法: storage-retirement <prepare|retire|verify> '
      + '--backup path --manifest path [--sqlite path] [--model-store path] '
      + '[--env-file path]',
    );
  }
  const values = new Map<string, string>();
  for (let i = 1; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--') || i + 1 >= argv.length) {
      throw new Error(`无效参数: ${key}`);
    }
    values.set(key.slice(2), argv[++i]);
  }
  const requireValue = (name: string, fallback?: string): string => {
    const value = values.get(name) || fallback;
    if (!value) throw new Error(`缺少 --${name}`);
    return value;
  };
  return {
    command,
    sqlite: path.resolve(values.get('sqlite') || resolveSqlitePath()),
    modelStore: path.resolve(values.get('model-store') || resolveModelStorePath()),
    backup: path.resolve(requireValue('backup')),
    manifest: path.resolve(requireValue('manifest')),
    envFile: values.get('env-file') ? path.resolve(values.get('env-file')!) : undefined,
    passphrase: requireValue('passphrase', process.env.QLH_BACKUP_PASSPHRASE),
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const store = new SqliteStore(args.sqlite);
  const artifacts = new ArtifactStore(args.modelStore);
  const service = new PostgresRetirementService(store, artifacts);
  const options = {
    backupPath: args.backup,
    manifestPath: args.manifest,
    passphrase: args.passphrase,
    envFile: args.envFile,
  };
  try {
    const result = args.command === 'prepare'
      ? await service.prepare(options)
      : args.command === 'retire'
        ? service.retire(options)
        : service.verify(options);
    console.log(JSON.stringify(result));
  } finally {
    store.close();
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
