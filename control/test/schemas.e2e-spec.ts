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

const SCHEMA_FILES: Record<string, string> = {
  'artifact-manifest': 'artifact-manifest.schema.json',
  'pull-job': 'pull-job.schema.json',
  'deployment': 'deployment.schema.json',
  'cluster-profile': 'cluster-profile.schema.json',
};

const VALID_FIXTURES = [
  'artifact-manifest-valid.json',
  'artifact-manifest-gguf-valid.json',
  'pull-job-valid.json',
  'deployment-valid.json',
  'cluster-profile-valid.json',
];

const INVALID_FIXTURES = [
  'artifact-manifest-invalid-bad-digest.json',
  'artifact-manifest-invalid-extra-capability.json',
  'pull-job-invalid-state.json',
  'deployment-invalid-status.json',
  'cluster-profile-invalid-endpoint.json',
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
});
