/**
 * control-svc models 域契约测试（阶段 3.2 模型注册表域）
 *
 * 语义对齐 api_server.py:5173-5238（registry CRUD）+ 6800-6880（gguf/download）：
 *  - GET  /models/registry → {models:[...]}；POST 注册（quant_types 按
 *    model_type 服务端生成：非 gguf → ['fp16','int8','int4']，gguf →
 *    ['Q4_K_M']；重复注册覆盖）；DELETE 内置模型 400 / 未注册 404
 *  - 校验：model_type 非法 400；缺 model_id/name 422（对齐 FastAPI 必填）
 *  - GET /models/gguf：扫描 .gguf + .sha256 缓存读取/计算写回；
 *    directory/exists/count；目录不存在 → exists:false
 *  - GET /models/download/{name}：附件流 octet-stream；Range 206 断点续传、
 *    416 越界；穿越/非 .gguf 400；不存在 404
 *
 * 存储目录均注入临时路径（QLH_MODELS_DIR + registry 文件），不触碰仓库
 * models/ 真实目录。
 */
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { createApp } from '../src/app';
import type { NestFastifyApplication } from '@nestjs/platform-fastify';
import { ConfigDao } from '../src/data/config-dao';
import { ModelRegistryStore } from '../src/data/model-registry-store';

describe('control-svc models 域（阶段 3.2 模型注册表）', () => {
  let app: NestFastifyApplication | null = null;
  let tmpBase: string;
  let tmpModelsDir: string;
  let registryFile: string;

  const dbDisabledDao = new ConfigDao({
    host: 'localhost',
    port: 5432,
    name: 'x',
    user: 'postgres',
    password: '',
    enabled: false,
    sslmode: 'prefer',
  });

  beforeEach(() => {
    tmpBase = fs.mkdtempSync(path.join(os.tmpdir(), 'control-models-'));
    tmpModelsDir = path.join(tmpBase, 'models');
    fs.mkdirSync(tmpModelsDir, { recursive: true });
    registryFile = path.join(tmpBase, 'registry.json');
    process.env.QLH_MODELS_DIR = tmpModelsDir;
  });

  afterEach(async () => {
    if (app) {
      await app.close();
      app = null;
    }
    delete process.env.QLH_MODELS_DIR;
    fs.rmSync(tmpBase, { recursive: true, force: true });
  });

  async function createTestApp(): Promise<NestFastifyApplication> {
    const { Test } = require('@nestjs/testing');
    const { AppModule } = require('../src/app');
    const moduleRef = await Test.createTestingModule({
      imports: [AppModule],
    })
      .overrideProvider(ModelRegistryStore)
      .useValue(new ModelRegistryStore(registryFile))
      .overrideProvider(ConfigDao)
      .useValue(dbDisabledDao)
      .compile();
    const fastifyAdapter = new (require('@nestjs/platform-fastify').FastifyAdapter)();
    const testApp = moduleRef.createNestApplication(fastifyAdapter);
    const { JsonDetailFilter } = require('../src/common/json-detail.filter');
    const { RequestIdInterceptor } = require('../src/common/request-id');
    testApp.useGlobalFilters(new JsonDetailFilter());
    testApp.useGlobalInterceptors(new RequestIdInterceptor());
    await testApp.init();
    await testApp.getHttpAdapter().getInstance().ready();
    return testApp;
  }

  function writeGguf(name: string, content: string): void {
    fs.writeFileSync(path.join(tmpModelsDir, name), content, 'utf-8');
  }

  function sha256Of(content: string): string {
    return createHash('sha256').update(content, 'utf-8').digest('hex');
  }

  // ---------- registry CRUD ----------

  it('GET /models/registry 空 → {models:[]}', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/models/registry' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ models: [] });
  });

  it('POST /models/registry 注册 → registered + 持久化 + 列表可见', async () => {
    app = await createTestApp();
    const res = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: {
        model_id: 'my-model',
        name: '我的模型',
        model_type: 'safetensors',
        model_path: 'D:/models/my-model',
        recommended_vram_gb: 12.5,
        max_context: 8192,
        description: '实验模型',
      },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'registered', model_id: 'my-model' });
    const list = await app.inject({ method: 'GET', url: '/models/registry' });
    expect(list.json().models).toHaveLength(1);
    const m = list.json().models[0];
    expect(m.model_id).toBe('my-model');
    expect(m.name).toBe('我的模型');
    expect(m.recommended_vram_gb).toBe(12.5);
    expect(m.max_context).toBe(8192);
    // 默认值
    expect(m.model_path).toBe('D:/models/my-model');
    expect(m.huggingface_id).toBe('');
    // quant_types 服务端生成（safetensors → torch 三档）
    expect(m.quant_types).toEqual(['fp16', 'int8', 'int4']);
    // 持久化文件
    expect(fs.existsSync(registryFile)).toBe(true);
  });

  it('POST quant_types 按 model_type 生成（gguf → Q4_K_M，both → torch 三档）', async () => {
    app = await createTestApp();
    const gguf = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'gguf-model', name: 'G', model_type: 'gguf' },
    });
    expect(gguf.statusCode).toBe(200);
    const both = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'both-model', name: 'B', model_type: 'both' },
    });
    expect(both.statusCode).toBe(200);
    const list = await app.inject({ method: 'GET', url: '/models/registry' });
    const byId = (id: string) => list.json().models.find((x: { model_id: string }) => x.model_id === id);
    expect(byId('gguf-model').quant_types).toEqual(['Q4_K_M']);
    expect(byId('both-model').quant_types).toEqual(['fp16', 'int8', 'int4']);
  });

  it('POST 重复注册 → 覆盖（upsert）', async () => {
    app = await createTestApp();
    await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'dup', name: '旧名', model_type: 'safetensors' },
    });
    const again = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'dup', name: '新名', model_type: 'gguf' },
    });
    expect(again.statusCode).toBe(200);
    const list = await app.inject({ method: 'GET', url: '/models/registry' });
    expect(list.json().models).toHaveLength(1);
    expect(list.json().models[0].name).toBe('新名');
    expect(list.json().models[0].quant_types).toEqual(['Q4_K_M']);
  });

  it('POST model_type 非法 → 400；缺 model_id/name → 422', async () => {
    app = await createTestApp();
    const badType = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'x', name: 'y', model_type: 'tensorflow' },
    });
    expect(badType.statusCode).toBe(400);
    expect(badType.json().detail).toBe('model_type 必须是 safetensors | gguf | both');
    const noId = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { name: 'y' },
    });
    expect(noId.statusCode).toBe(422);
    const noName = await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'x' },
    });
    expect(noName.statusCode).toBe(422);
  });

  it('DELETE 未注册 → 404；内置模型 → 400；成功 → deleted', async () => {
    app = await createTestApp();
    const missing = await app.inject({ method: 'DELETE', url: '/models/registry/nope' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json().detail).toBe("模型 'nope' 未注册");
    const builtin = await app.inject({ method: 'DELETE', url: '/models/registry/qwen-1_8b' });
    expect(builtin.statusCode).toBe(400);
    expect(builtin.json().detail).toBe("内置模型 'qwen-1_8b' 不允许删除。");
    await app.inject({
      method: 'POST',
      url: '/models/registry',
      payload: { model_id: 'del-me', name: '待删', model_type: 'gguf' },
    });
    const ok = await app.inject({ method: 'DELETE', url: '/models/registry/del-me' });
    expect(ok.statusCode).toBe(200);
    expect(ok.json()).toEqual({ status: 'deleted', model_id: 'del-me' });
    const list = await app.inject({ method: 'GET', url: '/models/registry' });
    expect(list.json().models).toHaveLength(0);
  });

  // ---------- gguf 扫描 ----------

  it('GET /models/gguf 目录不存在 → exists:false', async () => {
    fs.rmSync(tmpModelsDir, { recursive: true, force: true });
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/models/gguf' });
    expect(res.statusCode).toBe(200);
    expect(res.json().exists).toBe(false);
    expect(res.json().models).toEqual([]);
    expect(res.json().directory).toBe(tmpModelsDir);
  });

  it('GET /models/gguf 扫描 + .sha256 缓存读取', async () => {
    app = await createTestApp();
    writeGguf('a.gguf', 'content-a');
    writeGguf('b.gguf', 'content-b');
    fs.writeFileSync(path.join(tmpModelsDir, 'a.gguf.sha256'), 'abc123def  a.gguf\n', 'utf-8');
    fs.writeFileSync(path.join(tmpModelsDir, 'notes.txt'), 'not-model', 'utf-8');
    const res = await app.inject({ method: 'GET', url: '/models/gguf' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.exists).toBe(true);
    expect(body.count).toBe(2);
    expect(body.directory).toBe(path.resolve(tmpModelsDir));
    const names = body.models.map((m: { filename: string }) => m.filename);
    expect(names).toEqual(['a.gguf', 'b.gguf']); // sorted
    expect(names).not.toContain('notes.txt');
    const a = body.models.find((m: { filename: string }) => m.filename === 'a.gguf');
    expect(a.sha256).toBe('abc123def'); // 从缓存读（split()[0]）
    expect(a.size_bytes).toBe('content-a'.length);
    expect(a.size_mb).toBe(Math.round(('content-a'.length / 1048576) * 10) / 10);
    expect(a.download_url).toBe('/api/models/download/a.gguf');
  });

  it('GET /models/gguf 无缓存 → 现算 sha256 并写回缓存文件', async () => {
    app = await createTestApp();
    const content = '需要计算哈希的内容';
    writeGguf('calc.gguf', content);
    const res = await app.inject({ method: 'GET', url: '/models/gguf' });
    const m = res.json().models[0];
    expect(m.sha256).toBe(sha256Of(content));
    // 缓存写回
    const cache = fs.readFileSync(path.join(tmpModelsDir, 'calc.gguf.sha256'), 'utf-8');
    expect(cache.trim().split(/\s+/)[0]).toBe(sha256Of(content));
  });

  // ---------- download ----------

  it('GET /models/download/{name} → 附件流 + 内容', async () => {
    app = await createTestApp();
    writeGguf('dl.gguf', 'download-me');
    const res = await app.inject({ method: 'GET', url: '/models/download/dl.gguf' });
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toContain('application/octet-stream');
    expect(res.headers['content-disposition']).toContain('attachment; filename="dl.gguf"');
    expect(res.body).toBe('download-me');
  });

  it('GET /models/download Range 断点续传 → 206 + Content-Range', async () => {
    app = await createTestApp();
    writeGguf('range.gguf', '0123456789');
    const res = await app.inject({
      method: 'GET',
      url: '/models/download/range.gguf',
      headers: { range: 'bytes=2-5' },
    });
    expect(res.statusCode).toBe(206);
    expect(res.headers['content-range']).toBe('bytes 2-5/10');
    expect(res.headers['accept-ranges']).toBe('bytes');
    expect(res.body).toBe('2345');
    // suffix-range: bytes=-3 → 最后 3 字节
    const suffix = await app.inject({
      method: 'GET',
      url: '/models/download/range.gguf',
      headers: { range: 'bytes=-3' },
    });
    expect(suffix.statusCode).toBe(206);
    expect(suffix.body).toBe('789');
  });

  it('GET /models/download Range end 越界 → 截断 206（对齐 Starlette）', async () => {
    app = await createTestApp();
    writeGguf('range.gguf', '0123456789');
    const res = await app.inject({
      method: 'GET',
      url: '/models/download/range.gguf',
      headers: { range: 'bytes=8-20' },
    });
    expect(res.statusCode).toBe(206);
    expect(res.headers['content-range']).toBe('bytes 8-9/10');
    expect(res.body).toBe('89');
  });

  it('GET /models/download Range start 越界 → 416 + Content-Range */size', async () => {
    app = await createTestApp();
    writeGguf('range.gguf', '0123456789');
    const res = await app.inject({
      method: 'GET',
      url: '/models/download/range.gguf',
      headers: { range: 'bytes=12-20' },
    });
    expect(res.statusCode).toBe(416);
    expect(res.headers['content-range']).toBe('bytes */10');
  });

  it('GET /models/download 穿越/非 gguf → 400；不存在 → 404', async () => {
    app = await createTestApp();
    const traversal = await app.inject({ method: 'GET', url: '/models/download/..%2Fsecret.gguf' });
    expect(traversal.statusCode).toBe(400);
    expect(traversal.json().detail).toBe('无效的文件名');
    const notGguf = await app.inject({ method: 'GET', url: '/models/download/x.safetensors' });
    expect(notGguf.statusCode).toBe(400);
    expect(notGguf.json().detail).toBe('仅支持 .gguf 模型文件下载');
    const missing = await app.inject({ method: 'GET', url: '/models/download/nope.gguf' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json().detail).toBe('模型文件不存在: nope.gguf');
  });

  // ---------- 预设问题（对齐 api_server.py:1698-1790，运行态降级） ----------

  it('GET /presets → 6 项 + 降级默认值（int4 29 tok/s、512、current_quant null）', async () => {
    app = await createTestApp();
    const res = await app.inject({ method: 'GET', url: '/presets' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.presets).toHaveLength(6);
    expect(body.current_speed_tok_s).toBe(29);
    expect(body.current_quant).toBeNull();
    expect(body.max_new_tokens).toBe(512);
    // 数值对齐 api_server round(x, 1)（int4 速度档）
    const ids = body.presets.map((p: { id: string }) => p.id);
    expect(ids).toEqual([
      'intro',
      'edge_computing',
      'model_quantization',
      'code_assist',
      'creative',
      'reasoning',
    ]);
    const intro = body.presets[0];
    expect(intro.estimated_memory_mb).toBe(13.6); // 145*96/1024
    expect(intro.estimated_seconds).toBe(4.1); // 120/29
    const reasoning = body.presets[5];
    expect(reasoning.estimated_memory_mb).toBe(38.0); // 405*96/1024
    expect(reasoning.estimated_seconds).toBe(12.1); // 350/29
    // 每个 preset 契约字段完整
    for (const p of body.presets) {
      expect(typeof p.id).toBe('string');
      expect(typeof p.icon).toBe('string');
      expect(typeof p.label).toBe('string');
      expect(typeof p.question).toBe('string');
      expect(typeof p.estimated_prompt_tokens).toBe('number');
      expect(typeof p.estimated_response_tokens).toBe('number');
      expect(typeof p.estimated_memory_mb).toBe('number');
      expect(typeof p.estimated_seconds).toBe('number');
    }
  });
});
