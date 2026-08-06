/**
 * M2 内容寻址工件库与本地导入测试：
 *  staging/原子提交/去重/quarantine/引用/GC、GGUF 与 Safetensors 静态
 *  inspector（最小二进制 fixture）、导入流程（源只读/重复去重/坏文件隔离）。
 */
import { mkdirSync, mkdtempSync, writeFileSync, existsSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ArtifactStore, sha256Hex } from '../src/data/artifact-store';
import { ModelInspector } from '../src/data/model-inspector';
import { ModelImportService } from '../src/data/model-import-service';

function tempDir(): string {
  return mkdtempSync(join(tmpdir(), 'qlh-m2-'));
}

// ---------- GGUF fixture 构造 ----------

function ggufString(value: string): Buffer {
  const bytes = Buffer.from(value, 'utf-8');
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(bytes.length));
  return Buffer.concat([len, bytes]);
}

/** 最小合法 GGUF：qwen2 架构 + q4_k（file_type 12）+ context 4096。 */
function buildGguf(arch = 'qwen2', fileType = 12, context = 4096): Buffer {
  const kvs: Buffer[] = [];
  const pushStringKv = (key: string, value: string): void => {
    kvs.push(ggufString(key));
    const type = Buffer.alloc(4);
    type.writeUInt32LE(8); // STRING
    kvs.push(type, ggufString(value));
  };
  const pushUint32Kv = (key: string, value: number): void => {
    kvs.push(ggufString(key));
    const type = Buffer.alloc(4);
    type.writeUInt32LE(4); // UINT32
    const val = Buffer.alloc(4);
    val.writeUInt32LE(value);
    kvs.push(type, val);
  };
  pushStringKv('general.architecture', arch);
  pushUint32Kv('general.file_type', fileType);
  pushUint32Kv('general.context_length', context);
  const header = Buffer.concat([
    Buffer.from('GGUF', 'ascii'),
    (() => { const v = Buffer.alloc(4); v.writeUInt32LE(3); return v; })(),
    (() => { const t = Buffer.alloc(8); t.writeBigUInt64LE(0n); return t; })(),
    (() => { const k = Buffer.alloc(8); k.writeBigUInt64LE(BigInt(kvs.length / 3)); return k; })(),
    ...kvs,
  ]);
  return header;
}

function writeGguf(dir: string, name: string, content: Buffer): string {
  const p = join(dir, name);
  writeFileSync(p, content);
  return p;
}

// ---------- Safetensors 目录 fixture ----------

function buildSafetensorsDir(dir: string, config: Record<string, unknown>): string {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'config.json'), JSON.stringify(config), 'utf-8');
  const header = JSON.stringify({ model: { dtype: 'F16', shape: [1], data_offsets: [0, 8] } });
  const headLen = Buffer.alloc(8);
  headLen.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(join(dir, 'model.safetensors'), Buffer.concat([headLen, Buffer.from(header), Buffer.alloc(8)]));
  writeFileSync(join(dir, 'tokenizer.json'), '{}', 'utf-8');
  return dir;
}

describe('ArtifactStore（M2）', () => {
  it('stage → 原子提交 → 同内容去重', () => {
    const root = tempDir();
    const store = new ArtifactStore(root);
    store.stageWrite('job1', 'w.bin', Buffer.from('hello'));
    store.stageWrite('job2', 'w.bin', Buffer.from('hello'));
    const first = store.commitBlob('job1', 'w.bin');
    const second = store.commitBlob('job2', 'w.bin');
    expect(first.deduped).toBe(false);
    expect(second.deduped).toBe(true); // 同 digest 去重
    expect(first.digest).toBe(second.digest);
    expect(store.blobExists(first.digest)).toBe(true);
    expect(store.readBlob(first.digest)?.toString()).toBe('hello');
    rmSync(root, { recursive: true, force: true });
  });

  it('staging 路径穿越被拒绝', () => {
    const root = tempDir();
    const store = new ArtifactStore(root);
    expect(() => store.stageWrite('j', '../escape.bin', Buffer.from('x')))
      .toThrow();
    rmSync(root, { recursive: true, force: true });
  });

  it('quarantine 保留失败留痕', () => {
    const root = tempDir();
    const store = new ArtifactStore(root);
    store.stageWrite('bad', 'f.bin', Buffer.from('x'));
    const qDir = store.quarantine('bad', 'inspection failed: bad magic');
    expect(existsSync(join(qDir, 'f.bin'))).toBe(true);
    expect(existsSync(join(qDir, 'reason.txt'))).toBe(true);
    expect(store.listStaging('bad')).toEqual([]);
    rmSync(root, { recursive: true, force: true });
  });

  it('manifest 引用与 GC：无引用删除、有引用保留', () => {
    const root = tempDir();
    const store = new ArtifactStore(root);
    store.stageWrite('j1', 'keep.bin', Buffer.from('keep'));
    store.stageWrite('j2', 'drop.bin', Buffer.from('drop'));
    const keep = store.commitBlob('j1', 'keep.bin');
    const drop = store.commitBlob('j2', 'drop.bin');
    store.writeManifest({
      namespace: 'user', name: 'gc-test', tag: 'latest',
      artifact_id: 'sha256:abc', files: [{ sha256: keep.digest }],
    });
    const retained = store.referencedDigests();
    expect(retained.has(keep.digest)).toBe(true);
    expect(retained.has(drop.digest)).toBe(false);
    const gc = store.gc();
    expect(gc.freed).toBe(1); // drop 被回收
    expect(store.blobExists(keep.digest)).toBe(true);
    expect(store.blobExists(drop.digest)).toBe(false);
    rmSync(root, { recursive: true, force: true });
  });
});

describe('ModelInspector（M2）', () => {
  it('解析合法 GGUF：架构/量化/上下文/能力判定', () => {
    const dir = tempDir();
    const gguf = writeGguf(dir, 'model.gguf', buildGguf('qwen2', 12, 4096));
    const result = new ModelInspector().inspectGguf(gguf);
    expect(result.ok).toBe(true);
    expect(result.architecture).toBe('qwen2');
    expect(result.family).toBe('qwen2');
    expect(result.quantization).toBe('q4_k');
    expect(result.context_length).toBe(4096);
    expect(result.capabilities.llama_cpp).toBe(true);
    expect(result.capabilities.pytorch_layer_pipeline).toBe(false);
    rmSync(dir, { recursive: true, force: true });
  });

  it('拒绝坏 magic 与未知架构（fail-closed）', () => {
    const dir = tempDir();
    const bad = writeGguf(dir, 'bad.gguf', Buffer.from('NOTGGUF-JUNK'));
    const badResult = new ModelInspector().inspectGguf(bad);
    expect(badResult.ok).toBe(false);
    expect(badResult.errors.join()).toContain('magic');
    const unknown = writeGguf(dir, 'unk.gguf', buildGguf('weird-arch', 2));
    const unkResult = new ModelInspector().inspectGguf(unknown);
    expect(unkResult.ok).toBe(true); // 结构合法
    expect(unkResult.capabilities.llama_cpp).toBe(false); // 未知架构 fail-closed
    expect(unkResult.warnings.join()).toContain('不在已知列表');
    rmSync(dir, { recursive: true, force: true });
  });

  it('Safetensors 目录：已知 family → full_worker；缺 config → 拒绝', () => {
    const dir = tempDir();
    const good = buildSafetensorsDir(join(dir, 'good'), {
      model_type: 'qwen2', architectures: ['Qwen2ForCausalLM'],
      max_position_embeddings: 32768,
    });
    const goodResult = new ModelInspector().inspectSafetensorsDir(good);
    expect(goodResult.ok).toBe(true);
    expect(goodResult.family).toBe('qwen2');
    expect(goodResult.capabilities.full_worker).toBe(true);
    expect(goodResult.has_tokenizer).toBe(true);

    const bad = buildSafetensorsDir(join(dir, 'bad'), {});
    const badResult = new ModelInspector().inspectSafetensorsDir(bad);
    // 空 config：结构 ok 但能力 fail-closed（inspection_only）
    expect(badResult.ok).toBe(true);
    expect(badResult.capabilities.full_worker).toBe(false);
    expect(badResult.warnings.join()).toContain('不在已知列表');
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('ModelImportService（M2）', () => {
  it('目录导入：manifest 生成、源文件只读、重复导入去重', () => {
    const root = tempDir();
    const src = buildSafetensorsDir(join(root, 'src'), {
      model_type: 'qwen2', architectures: ['Qwen2ForCausalLM'],
    });
    const store = new ArtifactStore(join(root, 'store'));
    const service = new ModelImportService(store, new ModelInspector());
    const srcBefore = existsSync(join(src, 'model.safetensors'));

    const first = service.importLocal(src, { name: 'qwen-test' });
    expect(first.failed).toBe(0);
    expect(first.imported).toBeGreaterThanOrEqual(2);
    expect(first.artifact_id).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(first.manifest_path).toBeTruthy();
    // 源文件未被修改
    expect(existsSync(join(src, 'model.safetensors'))).toBe(srcBefore);
    const manifest = store.readManifest('user', 'qwen-test', 'latest');
    expect(manifest?.family).toBe('qwen2');
    expect((manifest?.files as Array<{ sha256: string }>).length).toBeGreaterThanOrEqual(2);

    // 重复导入：全部 deduped
    const second = service.importLocal(src, { name: 'qwen-test' });
    expect(second.failed).toBe(0);
    expect(second.deduped).toBeGreaterThanOrEqual(2);
    rmSync(root, { recursive: true, force: true });
  });

  it('单 GGUF 导入：llama_cpp 能力与量化', () => {
    const root = tempDir();
    const ggufPath = writeGguf(root, 'qwen.gguf', buildGguf('qwen2', 12));
    const store = new ArtifactStore(join(root, 'store'));
    const service = new ModelImportService(store, new ModelInspector());
    const report = service.importLocal(ggufPath, { name: 'qwen-gguf' });
    expect(report.failed).toBe(0);
    const manifest = store.readManifest('user', 'qwen-gguf', 'latest');
    expect(manifest?.format).toBe('gguf');
    expect(manifest?.quantization).toBe('q4_k');
    expect((manifest?.capabilities as Record<string, boolean>).llama_cpp).toBe(true);
    rmSync(root, { recursive: true, force: true });
  });

  it('坏文件 → quarantine（错误摘要永不进入 registry）', () => {
    const root = tempDir();
    const badGguf = writeGguf(root, 'bad.gguf', Buffer.from('NOTGGUF'));
    const store = new ArtifactStore(join(root, 'store'));
    const service = new ModelImportService(store, new ModelInspector());
    const report = service.importLocal(badGguf, { name: 'bad' });
    expect(report.failed).toBe(1);
    expect(report.quarantined).toBe(1);
    expect(report.artifact_id).toBeNull();
    // quarantine 留痕
    expect(existsSync(join(store.quarantineDir(report.job_id), 'reason.txt'))).toBe(true);
    rmSync(root, { recursive: true, force: true });
  });
});
