import * as path from 'path';
import { ArtifactStore } from './data/artifact-store';
import { ArtifactRuntimeRepository } from './data/artifact-runtime-repository';
import { ModelRuntimeCheckService } from './data/model-runtime-check.service';
import { ModelRuntimeSidecar } from './data/model-runtime-sidecar';
import { SqliteStore } from './data/sqlite-store';

interface CliArgs {
  manifest: string;
  nodeId: string;
  sqlitePath?: string;
  modelStore?: string;
  python?: string;
  timeoutMs: number;
}

function parseArgs(argv: string[]): CliArgs {
  const values = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith('--') || i + 1 >= argv.length) {
      throw new Error(`invalid argument: ${key}`);
    }
    values.set(key, argv[++i]);
  }
  const manifest = values.get('--manifest');
  if (!manifest) throw new Error('--manifest is required');
  const timeoutMs = Number(values.get('--timeout-ms') ?? 240_000);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 30 * 60_000) {
    throw new Error('--timeout-ms must be between 1000 and 1800000');
  }
  return {
    manifest: path.resolve(manifest),
    nodeId: values.get('--node-id') ?? 'local',
    sqlitePath: values.get('--sqlite') ? path.resolve(values.get('--sqlite')!) : undefined,
    modelStore: values.get('--model-store') ? path.resolve(values.get('--model-store')!) : undefined,
    python: values.get('--python'),
    timeoutMs,
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const store = new SqliteStore(args.sqlitePath);
  store.open();
  try {
    const artifacts = new ArtifactStore(args.modelStore);
    const repository = new ArtifactRuntimeRepository(store);
    const sidecar = new ModelRuntimeSidecar(artifacts, {
      executable: args.python,
      timeoutMs: args.timeoutMs,
    });
    const service = new ModelRuntimeCheckService(sidecar, repository, artifacts);
    const result = await service.checkManifestFile(args.manifest, args.nodeId);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    process.exitCode = result.status === 'ready' ? 0 : 2;
  } finally {
    store.close();
  }
}

main().catch((error) => {
  process.stderr.write(`model runtime check failed: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
