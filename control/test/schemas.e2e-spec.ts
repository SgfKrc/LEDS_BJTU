/**
 * M0 冻结契约：artifact manifest / pull job / deployment / cluster profile
 * JSON Schema 双语言验证（TypeScript 侧，与 tests/test_model_fleet_schemas.py 对齐）。
 *
 * 验收口径（一键模型部署计划 §16 M0）：同一 fixture 经 Python/TS schema
 * validator 得到一致结果；能力枚举禁止额外键（禁止用 model_type 推导能力）。
 */
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';

const ROOT = join(__dirname, '..', '..');
const SCHEMAS_DIR = join(ROOT, 'schemas');
const FIXTURES_DIR = join(ROOT, 'fixtures', 'model-fleet');
const COMPATIBILITY_FILE = join(SCHEMAS_DIR, 'model-fleet-compatibility.json');

const SCHEMA_FILES: Record<string, string> = {
  'artifact-manifest': 'artifact-manifest.schema.json',
  'pull-job': 'pull-job.schema.json',
  'deployment': 'deployment.schema.json',
  'cluster-profile': 'cluster-profile.schema.json',
  'migration-map': 'migration-map.schema.json',
  'fetcher-progress': 'fetcher-progress.schema.json',
};

const VALID_FIXTURES = [
  'artifact-manifest-valid.json',
  'artifact-manifest-gguf-valid.json',
  'pull-job-valid.json',
  'deployment-valid.json',
  'cluster-profile-valid.json',
  'fetcher-progress-valid.json',
];

const INVALID_FIXTURES = [
  'artifact-manifest-invalid-bad-digest.json',
  'artifact-manifest-invalid-extra-capability.json',
  'pull-job-invalid-state.json',
  'deployment-invalid-status.json',
  'cluster-profile-invalid-endpoint.json',
  'fetcher-progress-invalid-event.json',
];

const VALID_SAMPLES: Record<string, string> = {
  'artifact-manifest': join('fixtures', 'model-fleet', 'artifact-manifest-valid.json'),
  'pull-job': join('fixtures', 'model-fleet', 'pull-job-valid.json'),
  deployment: join('fixtures', 'model-fleet', 'deployment-valid.json'),
  'cluster-profile': join('fixtures', 'model-fleet', 'cluster-profile-valid.json'),
  'migration-map': join('schemas', 'migration-map.json'),
  'fetcher-progress': join('fixtures', 'model-fleet', 'fetcher-progress-valid.json'),
};

const INVALID_SEQUENCES = [
  'fetcher-sequence-invalid-transition.json',
  'fetcher-sequence-invalid-after-terminal.json',
  'fetcher-sequence-invalid-job-id.json',
  'fetcher-sequence-invalid-regression.json',
];

function loadJson(rel: string): unknown {
  return JSON.parse(readFileSync(join(ROOT, rel), 'utf-8'));
}

function fixtureKind(filename: string): string {
  const kind = Object.keys(SCHEMA_FILES).find((k) => filename.startsWith(k));
  if (!kind) throw new Error(`unknown fixture kind: ${filename}`);
  return kind;
}

const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

// schema 只编译一次（ajv 禁止重复注册相同 $id）
const compiled = new Map<string, ValidateFunction>();

function validatorFor(kind: string): ValidateFunction {
  let validate = compiled.get(kind);
  if (!validate) {
    const schema = loadJson(join('schemas', SCHEMA_FILES[kind])) as object;
    validate = ajv.compile(schema);
    compiled.set(kind, validate);
  }
  return validate;
}

function sequenceErrors(data: any): string[] {
  const compatibility = JSON.parse(readFileSync(COMPATIBILITY_FILE, 'utf-8'));
  const policy = compatibility.fetcher_progress_v1;
  const events = data?.events;
  if (!Array.isArray(events) || events.length === 0) {
    return ['events must be a non-empty list'];
  }

  const errors: string[] = [];
  const protocolVersion = compatibility.schemas['fetcher-progress'].schema_version;
  if (data.schema_version !== protocolVersion) {
    errors.push('sequence schema_version does not match the protocol');
  }

  const firstJobId = events[0]?.job_id;
  let previousEvent: string | undefined;
  let previousTime: number | undefined;
  const previousCounters = new Map<string, number>();
  events.forEach((event: any, index: number) => {
    const validate = validatorFor('fetcher-progress');
    if (!validate(event)) {
      errors.push(`events[${index}]: schema validation failed`);
    }

    const eventName = event?.event;
    const allowedPhases = policy.event_phases[eventName] ?? [];
    if (!allowedPhases.includes(event?.phase)) {
      errors.push(`events[${index}]: phase is invalid for ${eventName}`);
    }
    for (const field of policy.required_fields[eventName] ?? []) {
      if (event?.[field] === null || event?.[field] === undefined) {
        errors.push(`events[${index}]: ${field} is required for ${eventName}`);
      }
    }
    if (event?.job_id !== firstJobId) {
      errors.push(`events[${index}]: job_id changed within one sequence`);
    }

    const currentTime = Date.parse(event?.at);
    if (Number.isNaN(currentTime)) {
      errors.push(`events[${index}]: at is not a valid timestamp`);
    } else {
      if (previousTime !== undefined && currentTime < previousTime) {
        errors.push(`events[${index}]: timestamp regressed`);
      }
      previousTime = currentTime;
    }

    if (previousEvent !== undefined) {
      const allowed = policy.transitions[previousEvent] ?? [];
      if (!allowed.includes(eventName)) {
        errors.push(`events[${index}]: transition ${previousEvent} -> ${eventName} is invalid`);
      }
    }
    previousEvent = eventName;

    for (const field of policy.monotonic_fields) {
      const value = event?.[field];
      if (Number.isInteger(value)) {
        const previous = previousCounters.get(field);
        if (previous !== undefined && value < previous) {
          errors.push(`events[${index}]: ${field} regressed`);
        }
        previousCounters.set(field, value);
      }
    }
    if (Number.isInteger(event?.total_bytes) && Number.isInteger(event?.downloaded_bytes)
        && event.downloaded_bytes > event.total_bytes) {
      errors.push(`events[${index}]: downloaded_bytes exceeds total_bytes`);
    }
    if (Number.isInteger(event?.files_total) && Number.isInteger(event?.files_done)
        && event.files_done > event.files_total) {
      errors.push(`events[${index}]: files_done exceeds files_total`);
    }
  });

  if (events[0]?.event !== policy.start_event) {
    errors.push('sequence does not start with started');
  }
  if (!policy.terminal_events.includes(events.at(-1)?.event)) {
    errors.push('sequence does not end in a terminal event');
  }
  return errors;
}

describe('M0 model fleet schemas (TS)', () => {
  it('all schema files compile under Ajv', () => {
    for (const kind of Object.keys(SCHEMA_FILES)) {
      expect(() => validatorFor(kind)).not.toThrow();
    }
  });

  it.each(VALID_FIXTURES)('%s passes validation', (filename) => {
    const data = loadJson(join('fixtures', 'model-fleet', filename));
    const validate = validatorFor(fixtureKind(filename));
    const ok = validate(data);
    if (!ok) {
      throw new Error(`${filename} 应通过校验，但发现: ${JSON.stringify(validate.errors)}`);
    }
  });

  it.each(INVALID_FIXTURES)('%s is rejected', (filename) => {
    const data = loadJson(join('fixtures', 'model-fleet', filename));
    const validate = validatorFor(fixtureKind(filename));
    expect(validate(data)).toBe(false);
  });

  it('capability enum is frozen (no extra keys)', () => {
    const manifestSchema = loadJson(
      join('schemas', 'artifact-manifest.schema.json'),
    ) as any;
    const cap = manifestSchema.properties.capabilities;
    expect(Object.keys(cap.properties).sort()).toEqual([
      'full_worker',
      'llama_cpp',
      'pytorch_layer_pipeline',
      'task_stage',
    ]);
    expect(cap.additionalProperties).toBe(false);
  });

  it('compatibility policy covers all schema versions', () => {
    const policy = JSON.parse(readFileSync(COMPATIBILITY_FILE, 'utf-8'));
    expect(Object.keys(policy.schemas).sort()).toEqual(Object.keys(SCHEMA_FILES).sort());
    for (const [kind, filename] of Object.entries(SCHEMA_FILES)) {
      const schema = loadJson(join('schemas', filename)) as any;
      expect(policy.schemas[kind].file).toBe(filename);
      expect(policy.schemas[kind].schema_version).toBe(
        schema.properties.schema_version.const,
      );
    }
  });

  it('enforces the declared root unknown-field policy', () => {
    const policy = JSON.parse(readFileSync(COMPATIBILITY_FILE, 'utf-8'));
    for (const [kind, rel] of Object.entries(VALID_SAMPLES)) {
      const sample = loadJson(rel) as any;
      sample.__future_field__ = 'probe';
      const accepted = validatorFor(kind)(sample);
      expect(accepted).toBe(policy.schemas[kind].root_unknown_fields === 'allow');
    }
  });

  it('marks contract-shape and state-machine changes as breaking', () => {
    const policy = JSON.parse(readFileSync(COMPATIBILITY_FILE, 'utf-8'));
    const compatible = new Set<string>(policy.evolution.compatible_changes);
    const breaking = new Set<string>(policy.evolution.version_bump_required_for);
    expect(compatible.size).toBeGreaterThan(0);
    expect(breaking.size).toBeGreaterThan(0);
    for (const item of compatible) expect(breaking.has(item)).toBe(false);
    for (const item of [
      'add_required_property',
      'change_enum_members',
      'change_unknown_field_policy',
      'change_state_transition',
    ]) {
      expect(breaking.has(item)).toBe(true);
    }
  });

  it('accepts the valid fetcher event sequence', () => {
    const data = loadJson(join('fixtures', 'model-fleet', 'fetcher-sequence-valid.json'));
    expect(sequenceErrors(data)).toEqual([]);
  });

  it.each(INVALID_SEQUENCES)('%s is rejected by the sequence policy', (filename) => {
    const data = loadJson(join('fixtures', 'model-fleet', filename));
    expect(sequenceErrors(data).length).toBeGreaterThan(0);
  });

  it('migration-map data validates against its schema', () => {
    const data = loadJson(join('schemas', 'migration-map.json'));
    const validate = validatorFor('migration-map');
    const ok = validate(data);
    if (!ok) {
      throw new Error(`migration-map.json 应通过校验: ${JSON.stringify(validate.errors)}`);
    }
  });

  it('migration map is self-consistent', () => {
    const data = loadJson(join('schemas', 'migration-map.json')) as any;
    const tables = new Set<string>(data.target_tables);
    const ids = new Set<string>();
    for (const source of data.sources) {
      expect(tables.has(source.target.table)).toBe(true);
      expect(source.idempotency.trim().length).toBeGreaterThan(0);
      expect(Object.keys(source.target.field_map).length).toBeGreaterThan(0);
      expect(ids.has(source.source_id)).toBe(false);
      ids.add(source.source_id);
    }
    expect(data.rules.length).toBeGreaterThanOrEqual(1);
  });

  it('migration map prescribes no duplicate artifacts', () => {
    const data = loadJson(join('schemas', 'migration-map.json')) as any;
    const joined = data.rules.join(' ');
    expect(joined).toContain('sha256');
  });
});
