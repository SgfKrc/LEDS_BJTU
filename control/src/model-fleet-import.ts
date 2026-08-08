/** CLI for M2 batch and catalog migration reports. */
import * as fs from 'fs';
import * as path from 'path';
import { ArtifactStore } from './data/artifact-store';
import {
  CatalogImportCandidate,
  ModelBatchImporter,
} from './data/model-batch-import';
import { ModelImportService } from './data/model-import-service';
import { ModelInspector } from './data/model-inspector';

interface CliOptions {
  source?: string;
  catalog?: string;
  store?: string;
  report?: string;
  namespace?: string;
  tag?: string;
}

function usage(): string {
  return [
    'Usage: node dist/model-fleet-import.js --source <models-dir> [options]',
    '   or: node dist/model-fleet-import.js --catalog <catalog.json> [options]',
    '',
    'Options:',
    '  --store <dir>       Content-addressed store (default: QLH_MODEL_STORE)',
    '  --report <file>     Aggregate JSON report path',
    '  --namespace <name>  Manifest namespace (default: migration)',
    '  --tag <tag>         Manifest tag (default: imported)',
  ].join('\n');
}

function parseArgs(argv: string[]): CliOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === '--help' || key === '-h') throw new Error('help');
    if (!key.startsWith('--') || index + 1 >= argv.length) {
      throw new Error(`invalid argument: ${key}`);
    }
    values.set(key.slice(2), argv[index + 1]);
    index += 1;
  }
  const source = values.get('source');
  const catalog = values.get('catalog');
  if ((!source && !catalog) || (source && catalog)) {
    throw new Error('provide exactly one of --source or --catalog');
  }
  return {
    source,
    catalog,
    store: values.get('store'),
    report: values.get('report'),
    namespace: values.get('namespace'),
    tag: values.get('tag'),
  };
}

export function run(argv: string[]): number {
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

  try {
    const store = new ArtifactStore(options.store && path.resolve(options.store));
    const importer = new ModelBatchImporter(
      new ModelImportService(store, new ModelInspector()),
    );
    if (options.catalog) {
      const catalogPath = path.resolve(options.catalog);
      const candidates = JSON.parse(
        fs.readFileSync(catalogPath, 'utf-8'),
      ) as CatalogImportCandidate[];
      if (!Array.isArray(candidates)) throw new Error('catalog must be an array');
      const report = importer.importCatalog(catalogPath, candidates, {
        namespace: options.namespace,
        tag: options.tag,
      });
      const reportPath = options.report
        ? path.resolve(options.report)
        : path.join(store.root, 'reports', `catalog-migration-${Date.now()}.json`);
      importer.writeCatalogReport(report, reportPath);
      process.stdout.write(`${JSON.stringify({ ...report.totals, report: reportPath })}\n`);
      return report.totals.missing_sources + report.totals.failed_sources > 0 ? 1 : 0;
    }
    const report = importer.importDirectory(path.resolve(options.source as string), {
      namespace: options.namespace,
      tag: options.tag,
    });
    const reportPath = options.report
      ? path.resolve(options.report)
      : path.join(store.root, 'reports', `migration-${Date.now()}.json`);
    importer.writeReport(report, reportPath);
    process.stdout.write(`${JSON.stringify({ ...report.totals, report: reportPath })}\n`);
    return report.totals.failed > 0 ? 1 : 0;
  } catch (error) {
    process.stderr.write(
      `batch import failed: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    return 1;
  }
}

if (require.main === module) {
  process.exitCode = run(process.argv.slice(2));
}
