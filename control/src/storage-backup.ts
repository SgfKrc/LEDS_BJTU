/**
 * 用户主节点 SQLite 备份工具。
 *
 * 口令优先从 QLH_BACKUP_PASSPHRASE 读取，避免出现在 shell history 和进程列表；
 * --passphrase 仅用于受控本机脚本/测试。备份包始终由用户保存和恢复。
 */
import { SqliteStore, resolveSqlitePath } from './data/sqlite-store';
import { ArtifactStore, resolveModelStorePath } from './data/artifact-store';
import { StorageMigrationPackage } from './data/storage-migration-package';

interface ParsedArgs {
  command:
    | 'backup'
    | 'verify'
    | 'restore'
    | 'package'
    | 'verify-package'
    | 'restore-package';
  sqlite?: string;
  output?: string;
  input?: string;
  passphrase?: string;
  modelStore?: string;
}

function parseArgs(argv: string[]): ParsedArgs {
  const command = argv[0] as ParsedArgs['command'];
  if (![
    'backup',
    'verify',
    'restore',
    'package',
    'verify-package',
    'restore-package',
  ].includes(command)) {
    throw new Error(
      '用法: storage-backup <backup|verify|restore|package|verify-package|restore-package> '
      + '[--sqlite path] '
      + '[--model-store path] [--output path|--input path]',
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
  return {
    command,
    sqlite: values.get('sqlite'),
    output: values.get('output'),
    input: values.get('input'),
    passphrase: values.get('passphrase') || process.env.QLH_BACKUP_PASSPHRASE,
    modelStore: values.get('model-store'),
  };
}

function requireValue(value: string | undefined, name: string): string {
  if (!value) throw new Error(`缺少 --${name}`);
  return value;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const passphrase = requireValue(args.passphrase, 'passphrase 或 QLH_BACKUP_PASSPHRASE');
  const store = new SqliteStore(args.sqlite || resolveSqlitePath());
  const artifacts = new ArtifactStore(args.modelStore || resolveModelStorePath());
  const blobExists = (digest: string): boolean => artifacts.blobExists(digest);
  const migrationPackage = new StorageMigrationPackage(store, artifacts);
  try {
    if (args.command === 'backup') {
      store.open();
      const info = await store.exportEncryptedBackup(
        requireValue(args.output, 'output'),
        passphrase,
        blobExists,
      );
      console.log(JSON.stringify(info));
      return;
    }
    if (args.command === 'package') {
      const info = await migrationPackage.exportPackage(
        requireValue(args.output, 'output'),
        passphrase,
      );
      console.log(JSON.stringify(info));
      return;
    }
    if (args.command === 'verify-package') {
      const info = migrationPackage.verifyPackage(
        requireValue(args.input, 'input'),
        passphrase,
      );
      console.log(JSON.stringify(info));
      return;
    }
    if (args.command === 'restore-package') {
      const result = migrationPackage.restorePackage(
        requireValue(args.input, 'input'),
        passphrase,
      );
      console.log(JSON.stringify(result));
      return;
    }
    if (args.command === 'verify') {
      const info = store.verifyEncryptedBackup(
        requireValue(args.input, 'input'),
        passphrase,
        blobExists,
      );
      console.log(JSON.stringify(info));
      return;
    }
    const result = store.restoreEncryptedBackup(
      requireValue(args.input, 'input'),
      passphrase,
      blobExists,
    );
    console.log(JSON.stringify(result));
  } finally {
    store.close();
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
