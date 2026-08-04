/**
 * 模型注册表控制器 — 阶段 3.2 模型注册表域（语义对齐 api_server.py:5173-5238、
 * 6800-6880）
 *
 * 端点（5 个，全部纯控制面；推理面 /models/load|switch|current|available
 * 已由网关转发 inference-svc :8010）：
 *   GET    /models/registry            → {models: RegisteredModel[]}
 *   POST   /models/registry            → {status:'registered', model_id}；
 *                                       model_type 非法 400；缺字段 422
 *   DELETE /models/registry/:modelId   → {status:'deleted', model_id}；
 *                                       内置模型 400；未注册 404
 *   GET    /models/gguf                → {models:[{filename,size_bytes,size_mb,
 *                                       sha256,download_url}], directory, exists, count}
 *   GET    /models/download/*          → 文件流（application/octet-stream +
 *                                       attachment），Range 断点续传
 *
 * 降级说明（已记录计划文档）：DB 分支（cluster_config experimental_model:）
 * 未迁移——JSON 存储恒可用，不返回 503；CUDA 门控未迁移。
 * 模型目录：QLH_MODELS_DIR 优先，否则 <cwd>/models（对齐 api_server _MODELS_DIR）。
 */
import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Param,
  Post,
  Req,
  Res,
} from '@nestjs/common';
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import type { FastifyReply, FastifyRequest } from 'fastify';
import {
  BUILTIN_MODEL_IDS,
  ModelRegistryStore,
  quantTypesFor,
  RegisteredModel,
  RegisteredModelType,
} from '../../data/model-registry-store';

interface RegisterModelRequest {
  model_id?: string;
  name?: string;
  model_type?: string;
  model_path?: string;
  gguf_path?: string;
  recommended_vram_gb?: number;
  max_context?: number;
  huggingface_id?: string;
  description?: string;
}

export function resolveModelsDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.QLH_MODELS_DIR?.trim() || path.join(process.cwd(), 'models');
}

@Controller()
export class ModelsController {
  constructor(private readonly registry: ModelRegistryStore) {}

  // ---------- 注册表 CRUD（对齐 api_server.py:5173-5238） ----------

  @Get('models/registry')
  listRegistry(): Record<string, unknown> {
    return { models: this.registry.list() };
  }

  @Post('models/registry')
  @HttpCode(200)
  register(@Body() body: RegisterModelRequest): Record<string, unknown> {
    const modelId = body?.model_id;
    const name = body?.name;
    if (typeof modelId !== 'string' || !modelId.trim()) {
      // 对齐 pydantic RegisterModelRequest 必填校验（FastAPI 422）
      throw new HttpException('model_id 必填', 422);
    }
    if (typeof name !== 'string' || !name.trim()) {
      throw new HttpException('name 必填', 422);
    }
    const modelType = (body.model_type ?? 'safetensors') as string;
    if (!['safetensors', 'gguf', 'both'].includes(modelType)) {
      // 对齐 api_server.py:5187 的显式 400
      throw new HttpException('model_type 必须是 safetensors | gguf | both', 400);
    }
    const model: RegisteredModel = {
      model_id: modelId.trim(),
      name: name.trim(),
      model_type: modelType as RegisteredModelType,
      model_path: typeof body.model_path === 'string' ? body.model_path : '',
      gguf_path: typeof body.gguf_path === 'string' ? body.gguf_path : '',
      recommended_vram_gb: Number(body.recommended_vram_gb) || 8.0,
      max_context: Number(body.max_context) || 4096,
      huggingface_id: typeof body.huggingface_id === 'string' ? body.huggingface_id : '',
      description: typeof body.description === 'string' ? body.description : '',
      // 对齐 api_server.py:5201：quant_types 服务端按 model_type 生成
      quant_types: quantTypesFor(modelType as RegisteredModelType),
    };
    this.registry.upsert(model);
    return { status: 'registered', model_id: model.model_id };
  }

  @Delete('models/registry/:modelId')
  @HttpCode(200)
  unregister(@Param('modelId') modelId: string): Record<string, unknown> {
    if (BUILTIN_MODEL_IDS.has(modelId)) {
      throw new HttpException(`内置模型 '${modelId}' 不允许删除。`, 400);
    }
    if (!this.registry.delete(modelId)) {
      throw new HttpException(`模型 '${modelId}' 未注册`, 404);
    }
    return { status: 'deleted', model_id: modelId };
  }

  // ---------- GGUF 扫描（对齐 api_server.py:6800-6854） ----------

  @Get('models/gguf')
  async listGguf(): Promise<Record<string, unknown>> {
    const dir = resolveModelsDir();
    let isDir = false;
    try {
      isDir = fs.statSync(dir).isDirectory();
    } catch {
      return { models: [], directory: dir, exists: false };
    }
    const models = [];
    const files = fs
      .readdirSync(dir)
      .filter((f) => f.toLowerCase().endsWith('.gguf'))
      .sort();
    for (const fname of files) {
      const fpath = path.join(dir, fname);
      let st;
      try {
        st = fs.statSync(fpath);
      } catch {
        continue;
      }
      if (!st.isFile()) continue;
      const size = st.size;
      // 读取或计算 SHA256（优先读 .sha256 缓存；对齐 api_server.py:6819-6839）
      let sha256 = readSha256Cache(fpath);
      if (!sha256) {
        sha256 = await computeSha256(fpath);
        if (sha256) {
          try {
            fs.writeFileSync(`${fpath}.sha256`, `${sha256}  ${fname}\n`, 'utf-8');
          } catch {
            /* 缓存写失败不影响响应 */
          }
        }
      }
      models.push({
        filename: fname,
        size_bytes: size,
        size_mb: Math.round(size / (1024 * 1024) * 10) / 10,
        sha256,
        // 对外契约保持 /api 前缀（Android ModelManager 直接使用）
        download_url: `/api/models/download/${fname}`,
      });
    }
    return { models, directory: path.resolve(dir), exists: true, count: models.length };
  }

  // ---------- GGUF 下载（对齐 api_server.py:6857-6880，补 Range） ----------

  @Get('models/download/*')
  download(
    @Param('*') filename: string,
    @Req() req: FastifyRequest,
    @Res() reply: FastifyReply,
  ): void {
    // 防路径穿越（对齐 :6865-6867）
    if (filename !== path.basename(filename) || filename.includes('..')) {
      throw new HttpException('无效的文件名', 400);
    }
    if (!filename.toLowerCase().endsWith('.gguf')) {
      throw new HttpException('仅支持 .gguf 模型文件下载', 400);
    }
    const filePath = path.join(resolveModelsDir(), filename);
    let size: number;
    try {
      size = fs.statSync(filePath).size;
    } catch {
      throw new HttpException(`模型文件不存在: ${filename}`, 404);
    }

    // Range 断点续传（对齐 Starlette FileResponse 行为）
    const range = req.headers.range;
    let status = 200;
    let start = 0;
    let end = size - 1;
    if (typeof range === 'string' && range.startsWith('bytes=')) {
      const m = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (m) {
        if (m[1] !== '') {
          start = Number(m[1]);
          // end 超界截断到 size-1（对齐 Starlette FileResponse；仅 start 越界才 416）
          end = m[2] !== '' ? Math.min(Number(m[2]), size - 1) : size - 1;
        } else if (m[2] !== '') {
          // suffix-range: bytes=-N 最后 N 字节
          start = Math.max(0, size - Number(m[2]));
          end = size - 1;
        }
        if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
          reply
            .status(416)
            .header('Content-Range', `bytes */${size}`)
            .send();
          return;
        }
        status = 206;
        reply.header('Content-Range', `bytes ${start}-${end}/${size}`);
      }
    }
    reply
      .status(status)
      .header('Content-Type', 'application/octet-stream')
      .header('Content-Disposition', `attachment; filename="${filename}"`)
      .header('Accept-Ranges', 'bytes')
      .send(fs.createReadStream(filePath, { start, end }));
  }
}

function readSha256Cache(fpath: string): string {
  try {
    const text = fs.readFileSync(`${fpath}.sha256`, 'utf-8');
    return text.trim().split(/\s+/)[0] ?? '';
  } catch {
    return '';
  }
}

async function computeSha256(fpath: string): Promise<string> {
  try {
    const hash = createHash('sha256');
    await new Promise<void>((resolve, reject) => {
      const stream = fs.createReadStream(fpath);
      stream.on('data', (chunk) => hash.update(chunk));
      stream.on('end', () => resolve());
      stream.on('error', reject);
    });
    return hash.digest('hex');
  } catch {
    return '';
  }
}
