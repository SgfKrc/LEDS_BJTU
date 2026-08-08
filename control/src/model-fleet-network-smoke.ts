/** Repeatable M3 public download/interrupt/resume smoke. */
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { ClusterSettingsRepository } from './data/cluster-settings-repository';
import {
  DownloadResponseInfo, HfDownloader,
} from './data/hf-downloader';
import { HfResolver } from './data/hf-resolver';
import { ModelHttpClient } from './data/model-http-client';
import { ModelProxyConfigRepository } from './data/model-proxy-config-repository';
import { resolveSqlitePath, SqliteStore } from './data/sqlite-store';

const FIXTURE = {
  repoId: 'openai-community/gpt2',
  revision: '607a30d783dfa663caf39e06633721c8d4cfcd7e',
  filename: 'merges.txt',
  maxBytes: 2 * 1024 * 1024,
};

interface CliOptions {
  outputRoot?: string;
  report?: string;
  sqlite?: string;
}

interface SmokePhase {
  status: number;
  requested_start_bytes: number;
  content_range: string | null;
  total_bytes: number;
}

function usage(): string {
  return [
    'Usage: node dist/model-fleet-network-smoke.js [options]',
    '',
    'Options:',
    '  --output-root <dir>  Root for unique smoke artifacts',
    '  --report <file>      JSON report path (default: inside smoke directory)',
    '  --sqlite <file>      Control SQLite used for saved user proxy config',
    '',
    'Proxy precedence: QLH_HTTP_PROXY > saved user config > direct.',
  ].join('\n');
}

function parseArgs(argv: string[]): CliOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--help' || key === '-h') throw new Error('help');
    if (!['--output-root', '--report', '--sqlite'].includes(key)
        || index + 1 >= argv.length) {
      throw new Error(`invalid argument: ${key}`);
    }
    values.set(key.slice(2), argv[index + 1]);
    index += 1;
  }
  return {
    outputRoot: values.get('output-root'),
    report: values.get('report'),
    sqlite: values.get('sqlite'),
  };
}

function sha256(file: string): string {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function phase(info: DownloadResponseInfo | null): SmokePhase {
  if (!info) throw new Error('download did not expose response metadata');
  return {
    status: info.status,
    requested_start_bytes: info.requestedStartBytes,
    content_range: info.contentRange,
    total_bytes: info.totalBytes,
  };
}

function writeJsonAtomic(file: string, value: Record<string, unknown>): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf-8');
  fs.renameSync(temporary, file);
}

export async function run(argv: string[]): Promise<number> {
  let options: CliOptions;
  try {
    options = parseArgs(argv);
  } catch (error) {
    if (error instanceof Error && error.message !== 'help') {
      process.stderr.write(`${error.message}\n`);
    }
    process.stderr.write(`${usage()}\n`);
    return error instanceof Error && error.message === 'help' ? 0 : 2;
  }

  const defaultRoot = path.resolve(__dirname, '../../build/model-fleet');
  const outputRoot = path.resolve(options.outputRoot ?? defaultRoot);
  fs.mkdirSync(outputRoot, { recursive: true });
  const workDir = fs.mkdtempSync(path.join(outputRoot, 'hf-network-smoke-'));
  const reportPath = path.resolve(options.report ?? path.join(workDir, 'report.json'));
  const fullPath = path.join(workDir, 'full-merges.txt');
  const resumedPath = path.join(workDir, 'resumed-merges.txt');
  const startedAt = new Date().toISOString();
  const started = Date.now();

  let store: SqliteStore | null = null;
  let savedProxy: ModelProxyConfigRepository | null = null;
  const http = new ModelHttpClient({
    proxyProvider: () => {
      const current = savedProxy?.get() ?? null;
      return current ? { url: current.url, source: 'user' } : null;
    },
  });
  const downloader = new HfDownloader(http, { progressThrottleMs: 0 });
  try {
    const sqlitePath = path.resolve(options.sqlite ?? resolveSqlitePath());
    if (fs.existsSync(sqlitePath)) {
      store = new SqliteStore(sqlitePath);
      store.open();
      savedProxy = new ModelProxyConfigRepository(
        new ClusterSettingsRepository(store),
      );
    } else if (options.sqlite) {
      throw new Error(`SQLite file does not exist: ${sqlitePath}`);
    }

    const resolved = await new HfResolver(http).resolve(
      FIXTURE.repoId, FIXTURE.revision, [FIXTURE.filename],
    );
    if (resolved.resolvedRevision !== FIXTURE.revision) {
      throw new Error(
        `resolved revision changed: ${resolved.resolvedRevision}`,
      );
    }
    const resolvedFile = resolved.files.find(
      (file) => file.rfilename === FIXTURE.filename,
    );
    if (!resolvedFile) {
      throw new Error(`fixture file is missing: ${FIXTURE.filename}`);
    }
    if (resolvedFile.size <= 0 || resolvedFile.size > FIXTURE.maxBytes) {
      throw new Error(
        `fixture size is outside the smoke budget: ${resolvedFile.size}`,
      );
    }

    let fullResponse: DownloadResponseInfo | null = null;
    await downloader.downloadFile(
      FIXTURE.repoId, FIXTURE.revision, FIXTURE.filename, fullPath,
      {
        expectedSize: resolvedFile.size,
        signal: AbortSignal.timeout(30_000),
        onResponse: (response) => { fullResponse = response; },
      },
    );
    const expectedSize = fs.statSync(fullPath).size;
    if (expectedSize !== resolvedFile.size) {
      throw new Error(
        `downloaded size differs from resolve metadata: ${expectedSize}/${resolvedFile.size}`,
      );
    }
    const expectedDigest = sha256(fullPath);

    const interruption = new AbortController();
    let interruptedResponse: DownloadResponseInfo | null = null;
    let interruptionObserved = false;
    try {
      await downloader.downloadFile(
        FIXTURE.repoId, FIXTURE.revision, FIXTURE.filename, resumedPath,
        {
          expectedSize,
          signal: AbortSignal.any([
            interruption.signal, AbortSignal.timeout(30_000),
          ]),
          onResponse: (response) => { interruptedResponse = response; },
          onProgress: (progress) => {
            if (progress.bytesDownloaded > 0
                && progress.bytesDownloaded < expectedSize
                && !interruption.signal.aborted) {
              interruption.abort();
            }
          },
        },
      );
    } catch {
      interruptionObserved = interruption.signal.aborted;
    }
    const partialBytes = fs.existsSync(resumedPath)
      ? fs.statSync(resumedPath).size : 0;
    if (!interruptionObserved || partialBytes <= 0 || partialBytes >= expectedSize) {
      throw new Error(
        `controlled interruption did not leave a partial file: ${partialBytes}/${expectedSize}`,
      );
    }

    let resumedResponse: DownloadResponseInfo | null = null;
    await downloader.downloadFile(
      FIXTURE.repoId, FIXTURE.revision, FIXTURE.filename, resumedPath,
      {
        expectedSize,
        signal: AbortSignal.timeout(30_000),
        onResponse: (response) => { resumedResponse = response; },
      },
    );
    const resumedDigest = sha256(resumedPath);
    if (resumedDigest !== expectedDigest) {
      throw new Error(
        `resumed digest mismatch: ${resumedDigest} != ${expectedDigest}`,
      );
    }

    const report: Record<string, unknown> = {
      schema_version: 1,
      status: 'passed',
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      elapsed_ms: Date.now() - started,
      fixture: {
        repo_id: FIXTURE.repoId,
        requested_revision: FIXTURE.revision,
        resolved_revision: resolved.resolvedRevision,
        filename: FIXTURE.filename,
        size_bytes: expectedSize,
        sha256: expectedDigest,
      },
      proxy: http.proxyStatus(),
      phases: {
        full_download: phase(fullResponse),
        controlled_interruption: {
          ...phase(interruptedResponse),
          partial_bytes: partialBytes,
        },
        resumed_download: phase(resumedResponse),
      },
      artifacts: {
        work_dir: workDir,
        full_file: fullPath,
        resumed_file: resumedPath,
      },
    };
    writeJsonAtomic(reportPath, report);
    process.stdout.write(`${JSON.stringify({ ...report, report: reportPath })}\n`);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeJsonAtomic(reportPath, {
      schema_version: 1,
      status: 'failed',
      started_at: startedAt,
      completed_at: new Date().toISOString(),
      elapsed_ms: Date.now() - started,
      proxy: http.proxyStatus(),
      error: message,
      artifacts: { work_dir: workDir },
    });
    process.stderr.write(`network smoke failed: ${message}\nreport: ${reportPath}\n`);
    return 1;
  } finally {
    await http.onApplicationShutdown();
    store?.close();
  }
}

if (require.main === module) {
  run(process.argv.slice(2)).then((code) => { process.exitCode = code; });
}
